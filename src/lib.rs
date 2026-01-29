use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rangebar_core::{
    AggTrade, AnomalySummary, Checkpoint, CheckpointError, FixedPoint, PositionVerification,
    RangeBar, RangeBarProcessor,
};

// Arrow export support (feature-gated)
#[cfg(feature = "arrow-export")]
use pyo3_arrow::PyRecordBatch;
#[cfg(feature = "arrow-export")]
use rangebar_core::{aggtrades_to_record_batch, rangebar_vec_to_record_batch};

#[cfg(feature = "data-providers")]
use rangebar_providers::exness::{
    ExnessInstrument, ExnessRangeBar, ExnessRangeBarBuilder, ExnessTick, SpreadStats,
    ValidationStrictness,
};

#[cfg(feature = "data-providers")]
use rangebar_providers::binance::{HistoricalDataLoader, IntraDayChunkIterator};

/// Convert f64 to `FixedPoint` (8 decimal precision)
fn f64_to_fixed_point(value: f64) -> FixedPoint {
    // FixedPoint uses i64 with 8 decimal places (scale = 100_000_000)
    let scaled = (value * 100_000_000.0).round() as i64;
    FixedPoint(scaled)
}

/// Convert Python dict to Rust `AggTrade`
fn dict_to_agg_trade(py: Python, trade_dict: &Bound<PyDict>, index: usize) -> PyResult<AggTrade> {
    // Extract required fields
    let timestamp_value: PyObject = trade_dict
        .get_item("timestamp")?
        .ok_or_else(|| PyKeyError::new_err(format!("Trade {index}: missing 'timestamp'")))?
        .extract()?;

    let price_value: PyObject = trade_dict
        .get_item("price")?
        .ok_or_else(|| PyKeyError::new_err(format!("Trade {index}: missing 'price'")))?
        .extract()?;

    // Extract as i64/f64
    let timestamp_ms: i64 = timestamp_value.extract(py)?;
    let price: f64 = price_value.extract(py)?;

    // Support both "quantity" and "volume" keys
    let volume: f64 = match trade_dict.get_item("quantity")? {
        Some(val) => val.extract()?,
        None => trade_dict
            .get_item("volume")?
            .ok_or_else(|| {
                PyKeyError::new_err(format!("Trade {index}: missing 'quantity' or 'volume'"))
            })?
            .extract()?,
    };

    // Extract optional fields
    let agg_trade_id: i64 = trade_dict
        .get_item("agg_trade_id")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(index as i64);

    let first_trade_id: i64 = trade_dict
        .get_item("first_trade_id")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(agg_trade_id);

    let last_trade_id: i64 = trade_dict
        .get_item("last_trade_id")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(agg_trade_id);

    let is_buyer_maker: bool = trade_dict
        .get_item("is_buyer_maker")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(false);

    // Convert timestamp from milliseconds to microseconds (Binance → rangebar-core)
    let timestamp_us = timestamp_ms * 1000;

    Ok(AggTrade {
        agg_trade_id,
        price: f64_to_fixed_point(price),
        volume: f64_to_fixed_point(volume),
        first_trade_id,
        last_trade_id,
        timestamp: timestamp_us,
        is_buyer_maker,
        is_best_match: None, // Not needed for futures markets
    })
}

/// Convert Rust `RangeBar` to Python dict
fn rangebar_to_dict(py: Python, bar: &RangeBar) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    // Convert timestamp from microseconds to RFC3339 string
    let timestamp_seconds = bar.close_time as f64 / 1_000_000.0;
    let datetime = chrono::DateTime::from_timestamp(
        timestamp_seconds as i64,
        (timestamp_seconds.fract() * 1_000_000_000.0) as u32,
    )
    .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;

    dict.set_item("timestamp", datetime.to_rfc3339())?;

    // Convert OHLCV from FixedPoint to f64
    dict.set_item("open", bar.open.to_f64())?;
    dict.set_item("high", bar.high.to_f64())?;
    dict.set_item("low", bar.low.to_f64())?;
    dict.set_item("close", bar.close.to_f64())?;
    dict.set_item("volume", bar.volume.to_f64())?;

    // Optional: Include market microstructure data
    dict.set_item("vwap", bar.vwap.to_f64())?;
    dict.set_item("buy_volume", bar.buy_volume.to_f64())?;
    dict.set_item("sell_volume", bar.sell_volume.to_f64())?;
    dict.set_item("individual_trade_count", bar.individual_trade_count)?;
    dict.set_item("agg_record_count", bar.agg_record_count)?;

    // Microstructure features (Issue #25)
    dict.set_item("duration_us", bar.duration_us)?;
    dict.set_item("ofi", bar.ofi)?;
    dict.set_item("vwap_close_deviation", bar.vwap_close_deviation)?;
    dict.set_item("price_impact", bar.price_impact)?;
    dict.set_item("kyle_lambda_proxy", bar.kyle_lambda_proxy)?;
    dict.set_item("trade_intensity", bar.trade_intensity)?;
    dict.set_item("volume_per_trade", bar.volume_per_trade)?;
    dict.set_item("aggression_ratio", bar.aggression_ratio)?;
    dict.set_item("aggregation_density", bar.aggregation_density_f64)?;
    dict.set_item("turnover_imbalance", bar.turnover_imbalance)?;

    Ok(dict.into())
}

/// Convert Rust `Checkpoint` to Python dict (JSON-serializable)
fn checkpoint_to_dict(py: Python, checkpoint: &Checkpoint) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    // Identification
    dict.set_item("symbol", &checkpoint.symbol)?;
    dict.set_item("threshold_decimal_bps", checkpoint.threshold_decimal_bps)?;

    // Incomplete bar (convert to dict if present)
    if let Some(ref bar) = checkpoint.incomplete_bar {
        dict.set_item("incomplete_bar", rangebar_to_dict(py, bar)?)?;

        // Also store raw OHLCV for easy access
        let bar_dict = PyDict::new_bound(py);
        bar_dict.set_item("open", bar.open.to_f64())?;
        bar_dict.set_item("high", bar.high.to_f64())?;
        bar_dict.set_item("low", bar.low.to_f64())?;
        bar_dict.set_item("close", bar.close.to_f64())?;
        bar_dict.set_item("volume", bar.volume.to_f64())?;
        bar_dict.set_item("open_time", bar.open_time)?;
        bar_dict.set_item("close_time", bar.close_time)?;
        bar_dict.set_item("agg_record_count", bar.agg_record_count)?;
        dict.set_item("incomplete_bar_raw", bar_dict.into_py(py))?;
    } else {
        dict.set_item("incomplete_bar", py.None())?;
        dict.set_item("incomplete_bar_raw", py.None())?;
    }

    // Thresholds (FixedPoint as f64)
    if let Some((upper, lower)) = checkpoint.thresholds {
        let thresholds_dict = PyDict::new_bound(py);
        thresholds_dict.set_item("upper", upper.to_f64())?;
        thresholds_dict.set_item("lower", lower.to_f64())?;
        dict.set_item("thresholds", thresholds_dict.into_py(py))?;
    } else {
        dict.set_item("thresholds", py.None())?;
    }

    // Position tracking
    dict.set_item("last_timestamp_us", checkpoint.last_timestamp_us)?;
    // Convert to milliseconds for Python consistency
    dict.set_item("last_timestamp_ms", checkpoint.last_timestamp_us / 1000)?;
    dict.set_item("last_trade_id", checkpoint.last_trade_id)?;

    // Integrity
    dict.set_item("price_hash", checkpoint.price_hash)?;

    // Anomaly summary
    let anomaly_dict = PyDict::new_bound(py);
    anomaly_dict.set_item("gaps_detected", checkpoint.anomaly_summary.gaps_detected)?;
    anomaly_dict.set_item(
        "overlaps_detected",
        checkpoint.anomaly_summary.overlaps_detected,
    )?;
    anomaly_dict.set_item(
        "timestamp_anomalies",
        checkpoint.anomaly_summary.timestamp_anomalies,
    )?;
    dict.set_item("anomaly_summary", anomaly_dict.into_py(py))?;

    // Has incomplete bar flag
    dict.set_item("has_incomplete_bar", checkpoint.has_incomplete_bar())?;

    // Behavior flags (Issue #36)
    dict.set_item(
        "prevent_same_timestamp_close",
        checkpoint.prevent_same_timestamp_close,
    )?;

    // Issue #46: Persist defer_open state for cross-session continuity
    dict.set_item("defer_open", checkpoint.defer_open)?;

    Ok(dict.into())
}

/// Convert Python dict to Rust `Checkpoint`
fn dict_to_checkpoint(py: Python, dict: &Bound<PyDict>) -> PyResult<Checkpoint> {
    // Extract required fields
    let symbol: String = dict
        .get_item("symbol")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'symbol'"))?
        .extract()?;

    let threshold_decimal_bps: u32 = dict
        .get_item("threshold_decimal_bps")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'threshold_decimal_bps'"))?
        .extract()?;

    let last_timestamp_us: i64 = dict
        .get_item("last_timestamp_us")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'last_timestamp_us'"))?
        .extract()?;

    let last_trade_id: Option<i64> = dict
        .get_item("last_trade_id")?
        .and_then(|v| v.extract().ok());

    let price_hash: u64 = dict
        .get_item("price_hash")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'price_hash'"))?
        .extract()?;

    // Extract incomplete bar if present
    let incomplete_bar: Option<RangeBar> =
        if let Some(bar_raw) = dict.get_item("incomplete_bar_raw")? {
            if bar_raw.is_none() {
                None
            } else {
                let bar_dict: &Bound<PyDict> = bar_raw.downcast()?;
                Some(dict_to_rangebar(py, bar_dict)?)
            }
        } else {
            None
        };

    // Extract thresholds if present
    let thresholds: Option<(FixedPoint, FixedPoint)> =
        if let Some(thresholds_obj) = dict.get_item("thresholds")? {
            if thresholds_obj.is_none() {
                None
            } else {
                let thresholds_dict: &Bound<PyDict> = thresholds_obj.downcast()?;
                let upper: f64 = thresholds_dict
                    .get_item("upper")?
                    .ok_or_else(|| PyKeyError::new_err("Missing 'thresholds.upper'"))?
                    .extract()?;
                let lower: f64 = thresholds_dict
                    .get_item("lower")?
                    .ok_or_else(|| PyKeyError::new_err("Missing 'thresholds.lower'"))?
                    .extract()?;
                Some((f64_to_fixed_point(upper), f64_to_fixed_point(lower)))
            }
        } else {
            None
        };

    // Extract anomaly summary
    let anomaly_summary = if let Some(anomaly_obj) = dict.get_item("anomaly_summary")? {
        let anomaly_dict: &Bound<PyDict> = anomaly_obj.downcast()?;
        AnomalySummary {
            gaps_detected: anomaly_dict
                .get_item("gaps_detected")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            overlaps_detected: anomaly_dict
                .get_item("overlaps_detected")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            timestamp_anomalies: anomaly_dict
                .get_item("timestamp_anomalies")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
        }
    } else {
        AnomalySummary::default()
    };

    // Extract behavior flags (Issue #36)
    // Default to true (new behavior) if not present in checkpoint
    let prevent_same_timestamp_close: bool = dict
        .get_item("prevent_same_timestamp_close")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(true);

    // Issue #46: Extract defer_open flag from checkpoint dict
    let defer_open: bool = dict
        .get_item("defer_open")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(false);

    Ok(Checkpoint {
        symbol,
        threshold_decimal_bps,
        incomplete_bar,
        thresholds,
        last_timestamp_us,
        last_trade_id,
        price_hash,
        anomaly_summary,
        prevent_same_timestamp_close,
        defer_open,
    })
}

/// Convert Python dict to Rust `RangeBar` (for checkpoint restoration)
fn dict_to_rangebar(_py: Python, dict: &Bound<PyDict>) -> PyResult<RangeBar> {
    let open: f64 = dict
        .get_item("open")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'open'"))?
        .extract()?;
    let high: f64 = dict
        .get_item("high")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'high'"))?
        .extract()?;
    let low: f64 = dict
        .get_item("low")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'low'"))?
        .extract()?;
    let close: f64 = dict
        .get_item("close")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'close'"))?
        .extract()?;
    let volume: f64 = dict
        .get_item("volume")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'volume'"))?
        .extract()?;
    let open_time: i64 = dict
        .get_item("open_time")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'open_time'"))?
        .extract()?;
    let close_time: i64 = dict
        .get_item("close_time")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'close_time'"))?
        .extract()?;
    let agg_record_count: u32 = dict
        .get_item("agg_record_count")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);

    Ok(RangeBar {
        open_time,
        close_time,
        open: f64_to_fixed_point(open),
        high: f64_to_fixed_point(high),
        low: f64_to_fixed_point(low),
        close: f64_to_fixed_point(close),
        volume: f64_to_fixed_point(volume),
        turnover: 0,
        individual_trade_count: 0,
        agg_record_count,
        first_trade_id: 0,
        last_trade_id: 0,
        data_source: rangebar_core::DataSource::BinanceSpot,
        buy_volume: FixedPoint(0),
        sell_volume: FixedPoint(0),
        buy_trade_count: 0,
        sell_trade_count: 0,
        vwap: FixedPoint(0),
        buy_turnover: 0,
        sell_turnover: 0,
        // Microstructure features (Issue #25) - initialized to defaults
        duration_us: 0,
        ofi: 0.0,
        vwap_close_deviation: 0.0,
        price_impact: 0.0,
        kyle_lambda_proxy: 0.0,
        trade_intensity: 0.0,
        volume_per_trade: 0.0,
        aggression_ratio: 0.0,
        aggregation_density_f64: 0.0,
        turnover_imbalance: 0.0,
    })
}

/// Convert `CheckpointError` to Python exception
fn checkpoint_error_to_pyerr(e: CheckpointError) -> PyErr {
    match e {
        CheckpointError::SymbolMismatch {
            checkpoint,
            expected,
        } => PyValueError::new_err(format!(
            "Symbol mismatch: checkpoint has '{checkpoint}', expected '{expected}'"
        )),
        CheckpointError::ThresholdMismatch {
            checkpoint,
            expected,
        } => PyValueError::new_err(format!(
            "Threshold mismatch: checkpoint has {checkpoint}, expected {expected}"
        )),
        CheckpointError::PriceHashMismatch {
            checkpoint,
            computed,
        } => PyValueError::new_err(format!(
            "Price hash mismatch: checkpoint has {checkpoint}, computed {computed}"
        )),
        CheckpointError::MissingThresholds => {
            PyValueError::new_err("Checkpoint has incomplete bar but missing thresholds")
        }
        CheckpointError::SerializationError { message } => {
            PyRuntimeError::new_err(format!("Checkpoint serialization error: {message}"))
        }
    }
}

/// Position verification result for Python
#[pyclass(name = "PositionVerification")]
#[derive(Clone)]
struct PyPositionVerification {
    verification: PositionVerification,
}

#[pymethods]
impl PyPositionVerification {
    /// Check if position is exact match
    #[getter]
    const fn is_exact(&self) -> bool {
        matches!(self.verification, PositionVerification::Exact)
    }

    /// Check if there's a gap
    #[getter]
    fn has_gap(&self) -> bool {
        self.verification.has_gap()
    }

    /// Get gap details if any
    const fn gap_details(&self) -> Option<(i64, i64, i64)> {
        match &self.verification {
            PositionVerification::Gap {
                expected_id,
                actual_id,
                missing_count,
            } => Some((*expected_id, *actual_id, *missing_count)),
            _ => None,
        }
    }

    /// Get timestamp gap in milliseconds (for Exness/timestamp-only sources)
    const fn timestamp_gap_ms(&self) -> Option<i64> {
        match &self.verification {
            PositionVerification::TimestampOnly { gap_ms } => Some(*gap_ms),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.verification {
            PositionVerification::Exact => "PositionVerification::Exact".to_string(),
            PositionVerification::Gap {
                expected_id,
                actual_id,
                missing_count,
            } => format!(
                "PositionVerification::Gap(expected={expected_id}, actual={actual_id}, missing={missing_count})"
            ),
            PositionVerification::TimestampOnly { gap_ms } => {
                format!("PositionVerification::TimestampOnly(gap_ms={gap_ms})")
            }
        }
    }
}

/// Python-exposed `RangeBarProcessor`
#[pyclass(name = "PyRangeBarProcessor")]
struct PyRangeBarProcessor {
    processor: RangeBarProcessor,
    threshold_decimal_bps: u32,
    symbol: Option<String>,
}

#[pymethods]
impl PyRangeBarProcessor {
    /// Create new processor
    ///
    /// Args:
    ///     `threshold_decimal_bps`: Threshold in decimal basis points (250 = 25bps = 0.25%)
    ///     symbol: Optional symbol for checkpoint creation (e.g., "BTCUSDT")
    ///     `prevent_same_timestamp_close`: If True (default), bars cannot close on the
    ///         same timestamp they opened. This prevents flash crash scenarios from
    ///         creating thousands of bars at identical timestamps. Set to False for
    ///         legacy v8 behavior for comparative analysis. (Issue #36)
    ///
    /// Raises:
    ///     `ValueError`: If threshold is out of range [1, `100_000`]
    #[new]
    #[pyo3(signature = (threshold_decimal_bps, symbol = None, prevent_same_timestamp_close = true))]
    fn new(
        threshold_decimal_bps: u32,
        symbol: Option<String>,
        prevent_same_timestamp_close: bool,
    ) -> PyResult<Self> {
        let processor =
            RangeBarProcessor::with_options(threshold_decimal_bps, prevent_same_timestamp_close)
                .map_err(|e| PyValueError::new_err(format!("Failed to create processor: {e}")))?;

        Ok(Self {
            processor,
            threshold_decimal_bps,
            symbol,
        })
    }

    /// Create processor from checkpoint for cross-file continuation
    ///
    /// Args:
    ///     checkpoint: Dict containing checkpoint state from `create_checkpoint()`
    ///
    /// Returns:
    ///     New processor with restored state (incomplete bar continues building)
    ///
    /// Raises:
    ///     `ValueError`: If checkpoint is invalid or corrupted
    #[staticmethod]
    fn from_checkpoint(py: Python, checkpoint: &Bound<PyDict>) -> PyResult<Self> {
        let rust_checkpoint = dict_to_checkpoint(py, checkpoint)?;
        let threshold = rust_checkpoint.threshold_decimal_bps;
        let symbol = Some(rust_checkpoint.symbol.clone());

        let processor = RangeBarProcessor::from_checkpoint(rust_checkpoint)
            .map_err(checkpoint_error_to_pyerr)?;

        Ok(Self {
            processor,
            threshold_decimal_bps: threshold,
            symbol,
        })
    }

    /// Process aggregated trades into range bars (batch mode - resets state)
    ///
    /// WARNING: This method resets processor state on each call. For streaming
    /// across multiple batches (e.g., month-by-month processing), use
    /// `process_trades_streaming()` instead.
    ///
    /// Args:
    ///     trades: List of trade dicts with keys: timestamp (ms), price, quantity
    ///
    /// Returns:
    ///     List of range bar dicts with OHLCV data
    ///
    /// Raises:
    ///     `KeyError`: If required trade fields are missing
    ///     `RuntimeError`: If trade processing fails (e.g., unsorted trades)
    fn process_trades(
        &mut self,
        py: Python,
        trades: Vec<Bound<PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        if trades.is_empty() {
            return Ok(Vec::new());
        }

        // Convert Python dicts to AggTrade
        let agg_trades: Vec<AggTrade> = trades
            .iter()
            .enumerate()
            .map(|(i, trade_dict)| dict_to_agg_trade(py, trade_dict, i))
            .collect::<PyResult<Vec<_>>>()?;

        // Process through rangebar-core
        let bars = self
            .processor
            .process_agg_trade_records(&agg_trades)
            .map_err(|e| PyRuntimeError::new_err(format!("Processing failed: {e}")))?;

        // Convert RangeBars to Python dicts
        bars.iter().map(|bar| rangebar_to_dict(py, bar)).collect()
    }

    /// Process aggregated trades into range bars (streaming mode - preserves state)
    ///
    /// Unlike `process_trades()`, this method maintains processor state across
    /// calls, enabling continuous processing across multiple batches (e.g.,
    /// month-by-month or chunk-by-chunk processing).
    ///
    /// Use this method for:
    /// - Multi-month precomputation (Issue #16)
    /// - Chunked processing of large datasets
    /// - Any scenario requiring bar continuity across batches
    ///
    /// Args:
    ///     trades: List of trade dicts with keys: timestamp (ms), price, quantity
    ///
    /// Returns:
    ///     List of range bar dicts with OHLCV data (only completed bars)
    ///
    /// Raises:
    ///     `KeyError`: If required trade fields are missing
    ///     `RuntimeError`: If trade processing fails
    fn process_trades_streaming(
        &mut self,
        py: Python,
        trades: Vec<Bound<PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        if trades.is_empty() {
            return Ok(Vec::new());
        }

        // Convert Python dicts to AggTrade
        let agg_trades: Vec<AggTrade> = trades
            .iter()
            .enumerate()
            .map(|(i, trade_dict)| dict_to_agg_trade(py, trade_dict, i))
            .collect::<PyResult<Vec<_>>>()?;

        // Process each trade individually to maintain state (Issue #16 fix)
        let mut bars = Vec::new();
        for trade in agg_trades {
            match self.processor.process_single_trade(trade) {
                Ok(Some(bar)) => bars.push(rangebar_to_dict(py, &bar)?),
                Ok(None) => {} // No bar completed yet
                Err(e) => return Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

        Ok(bars)
    }

    /// Process aggregated trades into range bars with Arrow output (streaming mode)
    ///
    /// Same as `process_trades_streaming()` but returns Arrow RecordBatch for
    /// zero-copy transfer to Polars. This is the recommended method for
    /// memory-efficient processing of large datasets.
    ///
    /// Args:
    ///     trades: List of trade dicts with keys: timestamp (ms), price, quantity
    ///
    /// Returns:
    ///     Arrow RecordBatch containing completed range bars (30 columns)
    ///
    /// Raises:
    ///     `KeyError`: If required trade fields are missing
    ///     `RuntimeError`: If trade processing fails
    ///
    /// Example:
    ///     ```python
    ///     from rangebar._core import stream_binance_trades, PyRangeBarProcessor
    ///     import polars as pl
    ///
    ///     processor = PyRangeBarProcessor(250, symbol="BTCUSDT")
    ///
    ///     for trades_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-31"):
    ///         arrow_batch = processor.process_trades_streaming_arrow(trades_batch)
    ///         df = pl.from_arrow(arrow_batch)  # Zero-copy!
    ///         # Process df...
    ///     ```
    #[cfg(feature = "arrow-export")]
    fn process_trades_streaming_arrow(
        &mut self,
        py: Python,
        trades: Vec<Bound<PyDict>>,
    ) -> PyResult<PyRecordBatch> {
        if trades.is_empty() {
            // Return empty RecordBatch with correct schema
            let empty_batch = rangebar_vec_to_record_batch(&[]);
            return Ok(PyRecordBatch::new(empty_batch));
        }

        // Convert Python dicts to AggTrade
        let agg_trades: Vec<AggTrade> = trades
            .iter()
            .enumerate()
            .map(|(i, trade_dict)| dict_to_agg_trade(py, trade_dict, i))
            .collect::<PyResult<Vec<_>>>()?;

        // Process each trade individually to maintain state
        let mut bars = Vec::new();
        for trade in agg_trades {
            match self.processor.process_single_trade(trade) {
                Ok(Some(bar)) => bars.push(bar),
                Ok(None) => {} // No bar completed yet
                Err(e) => return Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

        // Convert to Arrow RecordBatch
        let batch = rangebar_vec_to_record_batch(&bars);
        Ok(PyRecordBatch::new(batch))
    }

    /// Create checkpoint for cross-file continuation
    ///
    /// Captures current processing state including incomplete bar (if any).
    /// The checkpoint can be serialized to JSON and used to resume processing
    /// across file boundaries while maintaining bar continuity.
    ///
    /// Args:
    ///     symbol: Symbol being processed (e.g., "BTCUSDT"). If None, uses the
    ///             symbol provided at construction time.
    ///
    /// Returns:
    ///     Dict containing checkpoint state (JSON-serializable)
    ///
    /// Example:
    ///     ```python
    ///     # Process first file
    ///     processor = RangeBarProcessor(250, symbol="BTCUSDT")
    ///     bars_1 = processor.process_trades(file1_trades)
    ///     checkpoint = processor.create_checkpoint()
    ///
    ///     # Save checkpoint to JSON
    ///     import json
    ///     with open("checkpoint.json", "w") as f:
    ///         json.dump(checkpoint, f)
    ///
    ///     # Resume from checkpoint (later, possibly different process)
    ///     with open("checkpoint.json") as f:
    ///         checkpoint = json.load(f)
    ///     processor = RangeBarProcessor.from_checkpoint(checkpoint)
    ///     bars_2 = processor.process_trades(file2_trades)
    ///     # Incomplete bar from file 1 continues correctly!
    ///     ```
    #[pyo3(signature = (symbol = None))]
    fn create_checkpoint(&self, py: Python, symbol: Option<String>) -> PyResult<PyObject> {
        let sym = symbol
            .or_else(|| self.symbol.clone())
            .ok_or_else(|| PyValueError::new_err("Symbol required for checkpoint creation"))?;

        let checkpoint = self.processor.create_checkpoint(&sym);
        checkpoint_to_dict(py, &checkpoint)
    }

    /// Verify position in data stream at file boundary
    ///
    /// Checks if the first trade of the next file matches the expected position
    /// based on the checkpoint state. Useful for detecting data gaps.
    ///
    /// Args:
    ///     first_trade: Dict with first trade of next file
    ///
    /// Returns:
    ///     `PositionVerification` object with verification result
    fn verify_position(
        &self,
        py: Python,
        first_trade: &Bound<PyDict>,
    ) -> PyResult<PyPositionVerification> {
        let trade = dict_to_agg_trade(py, first_trade, 0)?;
        let verification = self.processor.verify_position(&trade);
        Ok(PyPositionVerification { verification })
    }

    /// Get incomplete bar if any
    ///
    /// Returns the bar currently being built (not yet breached threshold).
    /// Returns None if the last trade completed a bar cleanly.
    fn get_incomplete_bar(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.processor.get_incomplete_bar() {
            Some(bar) => Ok(Some(rangebar_to_dict(py, &bar)?)),
            None => Ok(None),
        }
    }

    /// Check if there's an incomplete bar
    #[getter]
    fn has_incomplete_bar(&self) -> bool {
        self.processor.get_incomplete_bar().is_some()
    }

    /// Get threshold value
    #[getter]
    const fn threshold_decimal_bps(&self) -> u32 {
        self.threshold_decimal_bps
    }

    /// Get symbol if set
    #[getter]
    fn symbol(&self) -> Option<String> {
        self.symbol.clone()
    }

    /// Get prevent_same_timestamp_close setting
    ///
    /// Returns True if bars cannot close on the same timestamp they opened
    /// (timestamp gating enabled, default). Returns False for legacy v8 behavior.
    #[getter]
    fn prevent_same_timestamp_close(&self) -> bool {
        self.processor.prevent_same_timestamp_close()
    }

    /// Reset processor state at an ouroboros boundary.
    ///
    /// Clears the incomplete bar and position tracking while preserving
    /// the threshold configuration. Use this when starting fresh at a
    /// known boundary (year/month/week) for reproducibility.
    ///
    /// Returns:
    ///     The orphaned incomplete bar as a dict (if any), or None.
    ///     Mark returned bars with `is_orphan=True` for ML filtering.
    ///
    /// Example:
    ///     ```python
    ///     # At year boundary (Jan 1 00:00:00 UTC)
    ///     orphaned = processor.reset_at_ouroboros()
    ///     if orphaned:
    ///         orphaned["is_orphan"] = True
    ///         orphaned["ouroboros_boundary"] = "2024-01-01T00:00:00Z"
    ///         orphaned["reason"] = "year_boundary"
    ///     # Continue processing with clean state
    ///     ```
    fn reset_at_ouroboros(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        match self.processor.reset_at_ouroboros() {
            Some(bar) => Ok(Some(rangebar_to_dict(py, &bar)?)),
            None => Ok(None),
        }
    }
}

// ============================================================================
// Exness Bindings (feature-gated)
// ============================================================================

#[cfg(feature = "data-providers")]
mod exness_bindings {
    use super::*;

    /// Python-exposed ExnessInstrument enum
    ///
    /// Supported forex instruments from Exness Raw_Spread data.
    #[pyclass(name = "ExnessInstrument", eq, eq_int)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum PyExnessInstrument {
        EURUSD = 0,
        GBPUSD = 1,
        USDJPY = 2,
        AUDUSD = 3,
        USDCAD = 4,
        NZDUSD = 5,
        EURGBP = 6,
        EURJPY = 7,
        GBPJPY = 8,
        XAUUSD = 9,
    }

    #[pymethods]
    impl PyExnessInstrument {
        /// Get instrument symbol string
        #[getter]
        fn symbol(&self) -> &'static str {
            self.to_rust().symbol()
        }

        /// Get Raw_Spread symbol string (e.g., "EURUSD_Raw_Spread")
        #[getter]
        fn raw_spread_symbol(&self) -> String {
            self.to_rust().raw_spread_symbol()
        }

        /// Check if this is a JPY pair (different pip value)
        #[getter]
        fn is_jpy_pair(&self) -> bool {
            self.to_rust().is_jpy_pair()
        }

        /// Get all supported instruments
        #[staticmethod]
        fn all() -> Vec<PyExnessInstrument> {
            vec![
                Self::EURUSD,
                Self::GBPUSD,
                Self::USDJPY,
                Self::AUDUSD,
                Self::USDCAD,
                Self::NZDUSD,
                Self::EURGBP,
                Self::EURJPY,
                Self::GBPJPY,
                Self::XAUUSD,
            ]
        }
    }

    impl PyExnessInstrument {
        /// Convert to Rust ExnessInstrument
        const fn to_rust(self) -> ExnessInstrument {
            match self {
                Self::EURUSD => ExnessInstrument::EURUSD,
                Self::GBPUSD => ExnessInstrument::GBPUSD,
                Self::USDJPY => ExnessInstrument::USDJPY,
                Self::AUDUSD => ExnessInstrument::AUDUSD,
                Self::USDCAD => ExnessInstrument::USDCAD,
                Self::NZDUSD => ExnessInstrument::NZDUSD,
                Self::EURGBP => ExnessInstrument::EURGBP,
                Self::EURJPY => ExnessInstrument::EURJPY,
                Self::GBPJPY => ExnessInstrument::GBPJPY,
                Self::XAUUSD => ExnessInstrument::XAUUSD,
            }
        }
    }

    /// Python-exposed ValidationStrictness enum
    #[pyclass(name = "ValidationStrictness", eq, eq_int)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum PyValidationStrictness {
        /// Basic checks only (bid > 0, ask > 0, bid < ask)
        Permissive = 0,
        /// + Spread < 10% (catches obvious errors) [DEFAULT]
        Strict = 1,
        /// + Spread < 1% (flags suspicious data)
        Paranoid = 2,
    }

    impl PyValidationStrictness {
        const fn to_rust(self) -> ValidationStrictness {
            match self {
                Self::Permissive => ValidationStrictness::Permissive,
                Self::Strict => ValidationStrictness::Strict,
                Self::Paranoid => ValidationStrictness::Paranoid,
            }
        }
    }

    /// Convert SpreadStats to Python dict
    fn spread_stats_to_dict(py: Python, stats: &SpreadStats) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("min_spread", stats.min_spread.to_f64())?;
        dict.set_item("max_spread", stats.max_spread.to_f64())?;
        dict.set_item("avg_spread", stats.avg_spread().to_f64())?;
        dict.set_item("tick_count", stats.tick_count)?;
        Ok(dict.into())
    }

    /// Convert ExnessRangeBar to Python dict
    fn exness_rangebar_to_dict(py: Python, bar: &ExnessRangeBar) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);

        // Convert timestamp from microseconds to RFC3339 string
        let timestamp_seconds = bar.base.close_time as f64 / 1_000_000.0;
        let datetime = chrono::DateTime::from_timestamp(
            timestamp_seconds as i64,
            (timestamp_seconds.fract() * 1_000_000_000.0) as u32,
        )
        .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;

        dict.set_item("timestamp", datetime.to_rfc3339())?;

        // OHLCV from base bar
        dict.set_item("open", bar.base.open.to_f64())?;
        dict.set_item("high", bar.base.high.to_f64())?;
        dict.set_item("low", bar.base.low.to_f64())?;
        dict.set_item("close", bar.base.close.to_f64())?;
        dict.set_item("volume", bar.base.volume.to_f64())?; // Always 0 for Exness

        // Spread statistics
        dict.set_item("spread_stats", spread_stats_to_dict(py, &bar.spread_stats)?)?;

        // Market microstructure (mostly zeros for Exness)
        dict.set_item("vwap", bar.base.vwap.to_f64())?;
        dict.set_item("tick_count", bar.spread_stats.tick_count)?;

        Ok(dict.into())
    }

    /// Python-exposed ExnessRangeBarBuilder
    #[pyclass(name = "ExnessRangeBarBuilder")]
    pub struct PyExnessRangeBarBuilder {
        builder: ExnessRangeBarBuilder,
        threshold_decimal_bps: u32,
        instrument: PyExnessInstrument,
    }

    #[pymethods]
    impl PyExnessRangeBarBuilder {
        /// Create new builder for instrument
        ///
        /// Args:
        ///     instrument: ExnessInstrument enum value
        ///     threshold_decimal_bps: Threshold in decimal basis points (250 = 25bps = 0.25%)
        ///     strictness: ValidationStrictness enum (default: Strict)
        #[new]
        #[pyo3(signature = (instrument, threshold_decimal_bps, strictness = PyValidationStrictness::Strict))]
        fn new(
            instrument: PyExnessInstrument,
            threshold_decimal_bps: u32,
            strictness: PyValidationStrictness,
        ) -> PyResult<Self> {
            let builder = ExnessRangeBarBuilder::for_instrument(
                instrument.to_rust(),
                threshold_decimal_bps,
                strictness.to_rust(),
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create builder: {e}")))?;

            Ok(Self {
                builder,
                threshold_decimal_bps,
                instrument,
            })
        }

        /// Process a single tick
        ///
        /// Args:
        ///     tick: Dict with keys: bid, ask, `timestamp_ms`
        ///
        /// Returns:
        ///     None if bar still accumulating, Dict if bar completed
        fn process_tick(&mut self, py: Python, tick: &Bound<PyDict>) -> PyResult<Option<PyObject>> {
            // Extract tick fields
            let bid: f64 = tick
                .get_item("bid")?
                .ok_or_else(|| PyKeyError::new_err("Missing 'bid'"))?
                .extract()?;
            let ask: f64 = tick
                .get_item("ask")?
                .ok_or_else(|| PyKeyError::new_err("Missing 'ask'"))?
                .extract()?;
            let timestamp_ms: i64 = tick
                .get_item("timestamp_ms")?
                .ok_or_else(|| PyKeyError::new_err("Missing 'timestamp_ms'"))?
                .extract()?;

            let exness_tick = ExnessTick {
                bid,
                ask,
                timestamp_ms,
            };

            match self.builder.process_tick(&exness_tick) {
                Ok(Some(bar)) => Ok(Some(exness_rangebar_to_dict(py, &bar)?)),
                Ok(None) => Ok(None),
                Err(e) => Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

        /// Process multiple ticks at once
        ///
        /// Args:
        ///     ticks: List of tick dicts with keys: bid, ask, `timestamp_ms`
        ///
        /// Returns:
        ///     List of completed bar dicts
        fn process_ticks(
            &mut self,
            py: Python,
            ticks: Vec<Bound<PyDict>>,
        ) -> PyResult<Vec<PyObject>> {
            let mut bars = Vec::new();

            for tick in ticks {
                if let Some(bar) = self.process_tick(py, &tick)? {
                    bars.push(bar);
                }
            }

            Ok(bars)
        }

        /// Get incomplete bar if exists
        fn get_incomplete_bar(&self, py: Python) -> PyResult<Option<PyObject>> {
            match self.builder.get_incomplete_bar() {
                Some(bar) => Ok(Some(exness_rangebar_to_dict(py, &bar)?)),
                None => Ok(None),
            }
        }

        /// Get threshold value
        #[getter]
        const fn threshold_decimal_bps(&self) -> u32 {
            self.threshold_decimal_bps
        }

        /// Get instrument
        #[getter]
        const fn instrument(&self) -> PyExnessInstrument {
            self.instrument
        }
    }
}

// ============================================================================
// Arrow Export Bindings (feature-gated)
// ============================================================================

#[cfg(feature = "arrow-export")]
mod arrow_bindings {
    use super::*;

    /// Convert a list of range bar dicts to an Arrow RecordBatch.
    ///
    /// This is the primary export function for streaming range bars to Python
    /// with zero-copy semantics. The resulting RecordBatch can be converted
    /// directly to Polars/PyArrow DataFrames.
    ///
    /// Args:
    ///     bars: List of range bar dicts (as returned by `process_trades()`)
    ///
    /// Returns:
    ///     Arrow RecordBatch with 30 columns including all microstructure features
    ///
    /// Example:
    ///     ```python
    ///     import polars as pl
    ///     from rangebar._core import bars_to_arrow
    ///
    ///     # Get bars as dicts (traditional way)
    ///     bars = processor.process_trades(trades)
    ///
    ///     # Convert to Arrow RecordBatch for zero-copy to Polars
    ///     arrow_batch = bars_to_arrow(bars)
    ///     df = pl.from_arrow(arrow_batch)  # Zero-copy!
    ///     ```
    #[pyfunction]
    pub fn bars_to_arrow(py: Python, bars: Vec<Bound<PyDict>>) -> PyResult<PyRecordBatch> {
        if bars.is_empty() {
            // Return empty RecordBatch with correct schema
            let empty_batch = rangebar_vec_to_record_batch(&[]);
            return Ok(PyRecordBatch::new(empty_batch));
        }

        // Convert Python dicts to Rust RangeBars
        let rust_bars: Vec<RangeBar> = bars
            .iter()
            .enumerate()
            .map(|(i, bar_dict)| dict_to_rangebar_full(py, bar_dict, i))
            .collect::<PyResult<Vec<_>>>()?;

        // Convert to Arrow RecordBatch
        let batch = rangebar_vec_to_record_batch(&rust_bars);
        Ok(PyRecordBatch::new(batch))
    }

    /// Convert a list of trade dicts to an Arrow RecordBatch.
    ///
    /// Converts trade data directly to Arrow format for efficient streaming.
    ///
    /// Args:
    ///     trades: List of trade dicts with keys: timestamp, price, quantity, etc.
    ///
    /// Returns:
    ///     Arrow RecordBatch with 8 columns for trade data
    #[pyfunction]
    pub fn trades_to_arrow(py: Python, trades: Vec<Bound<PyDict>>) -> PyResult<PyRecordBatch> {
        if trades.is_empty() {
            let empty_batch = aggtrades_to_record_batch(&[]);
            return Ok(PyRecordBatch::new(empty_batch));
        }

        // Convert Python dicts to Rust AggTrades
        let rust_trades: Vec<AggTrade> = trades
            .iter()
            .enumerate()
            .map(|(i, trade_dict)| dict_to_agg_trade(py, trade_dict, i))
            .collect::<PyResult<Vec<_>>>()?;

        // Convert to Arrow RecordBatch
        let batch = aggtrades_to_record_batch(&rust_trades);
        Ok(PyRecordBatch::new(batch))
    }

    /// Convert Python dict to Rust `RangeBar` (full conversion with all fields)
    fn dict_to_rangebar_full(
        _py: Python,
        dict: &Bound<PyDict>,
        index: usize,
    ) -> PyResult<RangeBar> {
        // Helper to extract f64 with error context
        fn get_f64(dict: &Bound<PyDict>, key: &str, index: usize) -> PyResult<f64> {
            dict.get_item(key)?
                .ok_or_else(|| PyKeyError::new_err(format!("Bar {index}: missing '{key}'")))?
                .extract()
        }

        // Helper to extract i64 with error context
        fn get_i64(dict: &Bound<PyDict>, key: &str, index: usize) -> PyResult<i64> {
            dict.get_item(key)?
                .ok_or_else(|| PyKeyError::new_err(format!("Bar {index}: missing '{key}'")))?
                .extract()
        }

        // Helper to extract u32 with default
        fn get_u32_opt(dict: &Bound<PyDict>, key: &str, default: u32) -> PyResult<u32> {
            Ok(dict
                .get_item(key)?
                .and_then(|v| v.extract().ok())
                .unwrap_or(default))
        }

        // Helper to extract f64 with default
        fn get_f64_opt(dict: &Bound<PyDict>, key: &str, default: f64) -> PyResult<f64> {
            Ok(dict
                .get_item(key)?
                .and_then(|v| v.extract().ok())
                .unwrap_or(default))
        }

        // Core OHLCV
        let open = get_f64(dict, "open", index)?;
        let high = get_f64(dict, "high", index)?;
        let low = get_f64(dict, "low", index)?;
        let close = get_f64(dict, "close", index)?;
        let volume = get_f64(dict, "volume", index)?;

        // Timestamps - try close_time first, then open_time
        let close_time = dict
            .get_item("close_time")?
            .and_then(|v| v.extract::<i64>().ok())
            .unwrap_or(0);
        let open_time = dict
            .get_item("open_time")?
            .and_then(|v| v.extract::<i64>().ok())
            .unwrap_or(close_time);

        // Trade tracking
        let individual_trade_count = get_u32_opt(dict, "individual_trade_count", 0)?;
        let agg_record_count = get_u32_opt(dict, "agg_record_count", 0)?;
        let first_trade_id = get_i64(dict, "first_trade_id", index).unwrap_or(0);
        let last_trade_id = get_i64(dict, "last_trade_id", index).unwrap_or(0);

        // Order flow
        let buy_volume = get_f64_opt(dict, "buy_volume", 0.0)?;
        let sell_volume = get_f64_opt(dict, "sell_volume", 0.0)?;
        let buy_trade_count = get_u32_opt(dict, "buy_trade_count", 0)?;
        let sell_trade_count = get_u32_opt(dict, "sell_trade_count", 0)?;
        let vwap = get_f64_opt(dict, "vwap", 0.0)?;

        // Microstructure features
        let duration_us = dict
            .get_item("duration_us")?
            .and_then(|v| v.extract::<i64>().ok())
            .unwrap_or(0);
        let ofi = get_f64_opt(dict, "ofi", 0.0)?;
        let vwap_close_deviation = get_f64_opt(dict, "vwap_close_deviation", 0.0)?;
        let price_impact = get_f64_opt(dict, "price_impact", 0.0)?;
        let kyle_lambda_proxy = get_f64_opt(dict, "kyle_lambda_proxy", 0.0)?;
        let trade_intensity = get_f64_opt(dict, "trade_intensity", 0.0)?;
        let volume_per_trade = get_f64_opt(dict, "volume_per_trade", 0.0)?;
        let aggression_ratio = get_f64_opt(dict, "aggression_ratio", 0.0)?;
        let aggregation_density_f64 = get_f64_opt(dict, "aggregation_density", 0.0)?;
        let turnover_imbalance = get_f64_opt(dict, "turnover_imbalance", 0.0)?;

        Ok(RangeBar {
            open_time,
            close_time,
            open: f64_to_fixed_point(open),
            high: f64_to_fixed_point(high),
            low: f64_to_fixed_point(low),
            close: f64_to_fixed_point(close),
            volume: f64_to_fixed_point(volume),
            turnover: 0, // Not typically stored in dict
            individual_trade_count,
            agg_record_count,
            first_trade_id,
            last_trade_id,
            data_source: rangebar_core::DataSource::BinanceFuturesUM,
            buy_volume: f64_to_fixed_point(buy_volume),
            sell_volume: f64_to_fixed_point(sell_volume),
            buy_trade_count,
            sell_trade_count,
            vwap: f64_to_fixed_point(vwap),
            buy_turnover: 0, // Not typically stored in dict
            sell_turnover: 0,
            duration_us,
            ofi,
            vwap_close_deviation,
            price_impact,
            kyle_lambda_proxy,
            trade_intensity,
            volume_per_trade,
            aggression_ratio,
            aggregation_density_f64,
            turnover_imbalance,
        })
    }
}

/// Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRangeBarProcessor>()?;
    m.add_class::<PyPositionVerification>()?;

    // Add Arrow export functions if feature enabled
    #[cfg(feature = "arrow-export")]
    {
        m.add_function(wrap_pyfunction!(arrow_bindings::bars_to_arrow, m)?)?;
        m.add_function(wrap_pyfunction!(arrow_bindings::trades_to_arrow, m)?)?;
    }

    // Add Exness classes if feature enabled
    #[cfg(feature = "data-providers")]
    {
        m.add_class::<exness_bindings::PyExnessInstrument>()?;
        m.add_class::<exness_bindings::PyValidationStrictness>()?;
        m.add_class::<exness_bindings::PyExnessRangeBarBuilder>()?;
        m.add_class::<binance_bindings::PyMarketType>()?;
        m.add_class::<binance_bindings::PyBinanceTradeStream>()?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::fetch_binance_aggtrades,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::stream_binance_trades,
            m
        )?)?;
    }

    Ok(())
}

// ============================================================================
// Binance Bindings (feature-gated)
// ============================================================================

#[cfg(feature = "data-providers")]
mod binance_bindings {
    use super::*;
    use chrono::NaiveDate;

    /// Binance market type enum
    #[pyclass(name = "MarketType", eq, eq_int)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum PyMarketType {
        /// Spot market (BTCUSDT on spot.binance.com)
        Spot = 0,
        /// USD-M Futures (perpetual, USDT-margined)
        FuturesUM = 1,
        /// COIN-M Futures (perpetual, coin-margined)
        FuturesCM = 2,
    }

    impl PyMarketType {
        const fn to_market_str(self) -> &'static str {
            match self {
                Self::Spot => "spot",
                Self::FuturesUM => "um",
                Self::FuturesCM => "cm",
            }
        }
    }

    /// Fetch Binance aggTrades data for a symbol and date range.
    ///
    /// Downloads from data.binance.vision and returns as list of trade dicts.
    /// This is an internal function - use `get_range_bars()` in Python for
    /// the high-level API that handles caching and processing automatically.
    ///
    /// Args:
    ///     symbol: Trading pair (e.g., "BTCUSDT")
    ///     start_date: Start date as "YYYY-MM-DD"
    ///     end_date: End date as "YYYY-MM-DD"
    ///     market_type: Market type (Spot, FuturesUM, FuturesCM)
    ///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
    ///         Default: True. Set to False for faster downloads when data integrity
    ///         is verified elsewhere.
    ///
    /// Returns:
    ///     List of trade dicts with keys: timestamp, price, quantity, agg_trade_id,
    ///     first_trade_id, last_trade_id, is_buyer_maker
    ///
    /// Raises:
    ///     RuntimeError: If checksum verification fails (data corruption detected)
    #[pyfunction]
    #[pyo3(signature = (symbol, start_date, end_date, market_type = PyMarketType::Spot, verify_checksum = true))]
    pub fn fetch_binance_aggtrades(
        py: Python,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        market_type: PyMarketType,
        verify_checksum: bool,
    ) -> PyResult<Vec<PyObject>> {
        // Parse dates
        let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("Invalid start_date format: {e}")))?;
        let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("Invalid end_date format: {e}")))?;

        if start > end {
            return Err(PyValueError::new_err("start_date must be <= end_date"));
        }

        // Create loader with market type
        let loader = HistoricalDataLoader::new_with_market(symbol, market_type.to_market_str());

        // Create tokio runtime and fetch data
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        // Load each day in the range (with checksum verification per Issue #43)
        let mut all_trades = Vec::new();
        let mut current_date = start;

        while current_date <= end {
            let day_result = rt.block_on(
                loader.load_single_day_trades_with_checksum(current_date, verify_checksum),
            );
            match day_result {
                Ok(mut day_trades) => all_trades.append(&mut day_trades),
                Err(rangebar_providers::binance::HistoricalError::ChecksumMismatch {
                    date,
                    message,
                }) => {
                    // Checksum mismatch is a hard error - data corruption detected
                    return Err(PyRuntimeError::new_err(format!(
                        "Checksum verification failed for {date}: {message}"
                    )));
                }
                Err(e) => {
                    // Skip days with no data (weekends, holidays, etc.)
                    eprintln!("Warning: No data for {current_date}: {e}");
                }
            }
            current_date += chrono::Duration::days(1);
        }

        if all_trades.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "No data available for {symbol} from {start_date} to {end_date}"
            )));
        }

        // Sort by timestamp
        all_trades.sort_by_key(|trade| trade.timestamp);

        // Convert AggTrades to Python dicts
        let mut results = Vec::with_capacity(all_trades.len());
        for trade in &all_trades {
            let dict = PyDict::new_bound(py);
            // Timestamp is already in microseconds from rangebar-core
            // Convert to milliseconds for Python API consistency
            dict.set_item("timestamp", trade.timestamp / 1000)?;
            dict.set_item("price", trade.price.to_f64())?;
            dict.set_item("quantity", trade.volume.to_f64())?;
            dict.set_item("agg_trade_id", trade.agg_trade_id)?;
            dict.set_item("first_trade_id", trade.first_trade_id)?;
            dict.set_item("last_trade_id", trade.last_trade_id)?;
            dict.set_item("is_buyer_maker", trade.is_buyer_maker)?;
            results.push(dict.into());
        }

        Ok(results)
    }

    // ========================================================================
    // Streaming Trade Iterator (Phase 2: Memory-Efficient Architecture)
    // ========================================================================

    /// Iterator over Binance aggTrades in hour-based chunks.
    ///
    /// Memory-efficient streaming that yields trade batches instead of loading
    /// entire date ranges. Each iteration returns trades for a configurable
    /// hour window (default: 6 hours).
    ///
    /// # Memory Efficiency
    ///
    /// | Chunk Size | Peak Memory | Reduction |
    /// |------------|-------------|-----------|
    /// | 24 hours   | ~213 MB     | 1x        |
    /// | 6 hours    | ~46 MB      | 4.6x      |
    /// | 1 hour     | ~15 MB      | 14x       |
    ///
    /// # Checksum Verification (Issue #43)
    ///
    /// By default, SHA-256 checksums are verified for each downloaded file.
    /// Set `verify_checksum=False` to disable for faster downloads when
    /// data integrity is verified elsewhere.
    ///
    /// # Example
    ///
    /// ```python
    /// from rangebar._core import stream_binance_trades
    ///
    /// for trades_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-07"):
    ///     # Each batch is ~46 MB (6 hours of trades)
    ///     print(f"Processing {len(trades_batch)} trades")
    ///     bars = processor.process_trades_streaming(trades_batch)
    /// ```
    #[pyclass(name = "BinanceTradeStream")]
    pub struct PyBinanceTradeStream {
        inner: IntraDayChunkIterator,
        symbol: String,
    }

    #[pymethods]
    impl PyBinanceTradeStream {
        /// Create a new trade stream.
        ///
        /// Args:
        ///     symbol: Trading pair (e.g., "BTCUSDT")
        ///     start_date: Start date as "YYYY-MM-DD"
        ///     end_date: End date as "YYYY-MM-DD"
        ///     chunk_hours: Hours per chunk (1, 6, 12, or 24). Default: 6.
        ///     market_type: Market type (Spot, FuturesUM, FuturesCM). Default: Spot.
        ///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
        ///         Default: True. Set to False for faster downloads when data integrity
        ///         is verified elsewhere.
        #[new]
        #[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
        fn new(
            symbol: &str,
            start_date: &str,
            end_date: &str,
            chunk_hours: u32,
            market_type: PyMarketType,
            verify_checksum: bool,
        ) -> PyResult<Self> {
            // Parse dates
            let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
                .map_err(|e| PyValueError::new_err(format!("Invalid start_date format: {e}")))?;
            let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
                .map_err(|e| PyValueError::new_err(format!("Invalid end_date format: {e}")))?;

            if start > end {
                return Err(PyValueError::new_err("start_date must be <= end_date"));
            }

            if chunk_hours == 0 || chunk_hours > 24 {
                return Err(PyValueError::new_err("chunk_hours must be 1-24"));
            }

            let loader = HistoricalDataLoader::new_with_market(symbol, market_type.to_market_str());
            let inner = IntraDayChunkIterator::with_checksum(
                loader,
                start,
                end,
                chunk_hours,
                verify_checksum,
            );

            Ok(Self {
                inner,
                symbol: symbol.to_uppercase(),
            })
        }

        /// Get the symbol being streamed.
        #[getter]
        fn symbol(&self) -> &str {
            &self.symbol
        }

        /// Get the current date being processed.
        #[getter]
        fn current_date(&self) -> String {
            self.inner.current_date().format("%Y-%m-%d").to_string()
        }

        /// Get the current hour within the day.
        #[getter]
        fn current_hour(&self) -> u32 {
            self.inner.current_hour()
        }

        #[allow(clippy::missing_const_for_fn)]
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        /// Get next chunk of trades.
        ///
        /// Returns:
        ///     List of trade dicts, or None if exhausted.
        ///
        /// Raises:
        ///     RuntimeError: If data fetching fails.
        fn __next__(&mut self, py: Python) -> PyResult<Option<Vec<PyObject>>> {
            match self.inner.next() {
                Some(Ok(trades)) => {
                    // Convert AggTrades to Python dicts
                    let mut results = Vec::with_capacity(trades.len());
                    for trade in &trades {
                        let dict = PyDict::new_bound(py);
                        // Convert microseconds to milliseconds for Python API
                        dict.set_item("timestamp", trade.timestamp / 1000)?;
                        dict.set_item("price", trade.price.to_f64())?;
                        dict.set_item("quantity", trade.volume.to_f64())?;
                        dict.set_item("agg_trade_id", trade.agg_trade_id)?;
                        dict.set_item("first_trade_id", trade.first_trade_id)?;
                        dict.set_item("last_trade_id", trade.last_trade_id)?;
                        dict.set_item("is_buyer_maker", trade.is_buyer_maker)?;
                        results.push(dict.into());
                    }
                    Ok(Some(results))
                }
                Some(Err(e)) => Err(PyRuntimeError::new_err(format!("Data fetch error: {e}"))),
                None => Ok(None),
            }
        }
    }

    /// Stream Binance aggTrades data in memory-efficient chunks.
    ///
    /// Returns an iterator that yields trade batches instead of loading
    /// the entire date range into memory. This is the recommended way to
    /// process large date ranges.
    ///
    /// Args:
    ///     symbol: Trading pair (e.g., "BTCUSDT")
    ///     start_date: Start date as "YYYY-MM-DD"
    ///     end_date: End date as "YYYY-MM-DD"
    ///     chunk_hours: Hours per chunk (1, 6, 12, or 24). Default: 6.
    ///     market_type: Market type (Spot, FuturesUM, FuturesCM). Default: Spot.
    ///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
    ///         Default: True. Set to False for faster downloads when data integrity
    ///         is verified elsewhere.
    ///
    /// Returns:
    ///     Iterator yielding lists of trade dicts.
    ///
    /// Example:
    ///     ```python
    ///     from rangebar._core import stream_binance_trades, PyRangeBarProcessor
    ///
    ///     processor = PyRangeBarProcessor(250, symbol="BTCUSDT")
    ///
    ///     for trades_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-31"):
    ///         bars = processor.process_trades_streaming(trades_batch)
    ///         # Process bars...
    ///
    ///     # Get final incomplete bar
    ///     final_bar = processor.get_incomplete_bar()
    ///     ```
    #[pyfunction]
    #[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
    pub fn stream_binance_trades(
        symbol: &str,
        start_date: &str,
        end_date: &str,
        chunk_hours: u32,
        market_type: PyMarketType,
        verify_checksum: bool,
    ) -> PyResult<PyBinanceTradeStream> {
        PyBinanceTradeStream::new(
            symbol,
            start_date,
            end_date,
            chunk_hours,
            market_type,
            verify_checksum,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_to_fixed_point() {
        let fp = f64_to_fixed_point(42000.12345678);
        assert_eq!(fp.to_f64(), 42000.12345678);

        let fp2 = f64_to_fixed_point(1.5);
        assert_eq!(fp2.to_f64(), 1.5);
    }

    #[test]
    fn test_processor_creation() {
        let processor = PyRangeBarProcessor::new(250, None);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().threshold_decimal_bps, 250);
    }

    #[test]
    fn test_invalid_threshold() {
        let processor = PyRangeBarProcessor::new(0, None);
        assert!(processor.is_err());

        let processor = PyRangeBarProcessor::new(200_000, None);
        assert!(processor.is_err());
    }

    #[test]
    fn test_f64_to_fixed_point_extremes() {
        // Test zero
        let fp_zero = f64_to_fixed_point(0.0);
        assert_eq!(fp_zero.to_f64(), 0.0);

        // Test negative values
        let fp_neg = f64_to_fixed_point(-42000.12345678);
        assert_eq!(fp_neg.to_f64(), -42000.12345678);

        // Test very small values (precision boundary)
        let fp_small = f64_to_fixed_point(0.00000001);
        assert_eq!(fp_small.to_f64(), 0.00000001);

        // Test very large values
        let fp_large = f64_to_fixed_point(1_000_000.12345678);
        assert_eq!(fp_large.to_f64(), 1_000_000.12345678);

        // Test maximum precision (8 decimal places)
        let fp_precision = f64_to_fixed_point(123.45678901);
        // Should round to 8 decimal places
        let rounded = fp_precision.to_f64();
        assert!((rounded - 123.45678901).abs() < 0.000001);
    }

    #[test]
    fn test_processor_boundary_thresholds() {
        // Test minimum valid threshold (1 = 0.1 basis points)
        let processor_min = PyRangeBarProcessor::new(1, None);
        assert!(processor_min.is_ok());
        assert_eq!(processor_min.unwrap().threshold_decimal_bps, 1);

        // Test maximum valid threshold (100_000 = 10,000 basis points = 100%)
        let processor_max = PyRangeBarProcessor::new(100_000, None);
        assert!(processor_max.is_ok());
        assert_eq!(processor_max.unwrap().threshold_decimal_bps, 100_000);

        // Test common valid thresholds
        for threshold in [10, 100, 250, 500, 1000, 10_000] {
            let processor = PyRangeBarProcessor::new(threshold, None);
            assert!(processor.is_ok(), "Threshold {} should be valid", threshold);
        }
    }
}
