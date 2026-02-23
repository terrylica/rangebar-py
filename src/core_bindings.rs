// FILE-SIZE-OK: Core bindings hub, multi-domain consolidation justified (see Issue #94)
use super::*;

/// Return the raw TOML feature manifest string (Issue #95).
/// Python parses this with tomllib (stdlib since 3.11).
#[pyfunction]
pub(crate) fn get_feature_manifest_raw() -> &'static str {
    rangebar_core::FEATURE_MANIFEST_TOML
}

/// Position verification result for Python
#[pyclass(name = "PositionVerification")]
#[derive(Clone)]
pub(crate) struct PyPositionVerification {
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
pub(crate) struct PyRangeBarProcessor {
    processor: RangeBarProcessor,
    pub(crate) threshold_decimal_bps: u32,
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
    ///     `inter_bar_lookback_count`: If set, enables inter-bar features computed from
    ///         a lookback window of this many trades BEFORE each bar opens. Recommended
    ///         values: 100-500. Set to None (default) to disable inter-bar features.
    ///         (Issue #59)
    ///     `include_intra_bar_features`: If True, enables intra-bar features computed from
    ///         trades WITHIN each bar. Adds 22 features including ITH (Investment Time
    ///         Horizon), statistical, and complexity metrics. Default: False (Issue #59)
    ///     `inter_bar_lookback_bars`: If set, enables inter-bar features with bar-relative
    ///         lookback mode. The lookback window = all trades from the last N completed
    ///         bars, self-adapting to bar size. Takes precedence over
    ///         `inter_bar_lookback_count`. Recommended: 3. (Issue #81)
    ///
    /// Raises:
    ///     `ValueError`: If threshold is out of range [1, `100_000`]
    #[new]
    #[pyo3(signature = (threshold_decimal_bps, symbol = None, prevent_same_timestamp_close = true, inter_bar_lookback_count = None, include_intra_bar_features = false, inter_bar_lookback_bars = None))]
    pub(crate) fn new(
        threshold_decimal_bps: u32,
        symbol: Option<String>,
        prevent_same_timestamp_close: bool,
        inter_bar_lookback_count: Option<usize>,
        include_intra_bar_features: bool,
        inter_bar_lookback_bars: Option<usize>,
    ) -> PyResult<Self> {
        // Issue #59: Build processor with optional inter-bar feature config
        let mut processor =
            RangeBarProcessor::with_options(threshold_decimal_bps, prevent_same_timestamp_close)
                .map_err(|e| PyValueError::new_err(format!("Failed to create processor: {e}")))?;

        // Issue #81: inter_bar_lookback_bars takes precedence over inter_bar_lookback_count
        if let Some(n_bars) = inter_bar_lookback_bars {
            let config = InterBarConfig {
                lookback_mode: LookbackMode::BarRelative(n_bars),
                ..Default::default()
            };
            processor = processor.with_inter_bar_config(config);
        } else if let Some(count) = inter_bar_lookback_count {
            let config = InterBarConfig {
                lookback_mode: LookbackMode::FixedCount(count),
                ..Default::default()
            };
            processor = processor.with_inter_bar_config(config);
        }

        // Issue #59: Enable intra-bar features if requested
        if include_intra_bar_features {
            processor = processor.with_intra_bar_features();
        }

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

    /// Re-enable microstructure features on a restored processor (Issue #97).
    ///
    /// After `from_checkpoint()`, the processor has bar state but loses
    /// microstructure config. Call this before processing trades to re-enable
    /// inter-bar lookback and intra-bar features.
    ///
    /// Args:
    ///     inter_bar_lookback_count: Fixed trade count for lookback window
    ///     inter_bar_lookback_bars: Bar-relative lookback (takes precedence)
    ///     include_intra_bar_features: Enable intra-bar features
    #[pyo3(signature = (inter_bar_lookback_count = None, inter_bar_lookback_bars = None, include_intra_bar_features = false))]
    fn enable_microstructure(
        &mut self,
        inter_bar_lookback_count: Option<usize>,
        inter_bar_lookback_bars: Option<usize>,
        include_intra_bar_features: bool,
    ) {
        if let Some(n_bars) = inter_bar_lookback_bars {
            let config = InterBarConfig {
                lookback_mode: LookbackMode::BarRelative(n_bars),
                ..Default::default()
            };
            self.processor.set_inter_bar_config(config);
        } else if let Some(count) = inter_bar_lookback_count {
            let config = InterBarConfig {
                lookback_mode: LookbackMode::FixedCount(count),
                ..Default::default()
            };
            self.processor.set_inter_bar_config(config);
        }

        if include_intra_bar_features {
            self.processor.set_intra_bar_features(true);
        }
    }

    /// Process aggregated trades into range bars (batch mode - resets state)
    ///
    /// WARNING: This method resets processor state on each call. For streaming
    /// across multiple batches (e.g., month-by-month processing), use
    /// `process_trades_streaming()` instead.
    ///
    /// Args:
    ///     trades: List of trade dicts with keys: timestamp (ms), price, quantity
    ///     return_format: Format for returned bars: "dict" (default) or "arrow" (3-5x faster)
    ///
    /// Returns:
    ///     List of range bar dicts with OHLCV data (dict mode) or PyRecordBatch (arrow mode)
    ///
    /// Raises:
    ///     `KeyError`: If required trade fields are missing
    ///     `RuntimeError`: If trade processing fails (e.g., unsorted trades)
    ///     `ValueError`: If return_format is not "dict" or "arrow"
    #[pyo3(signature = (trades, return_format = "dict"))]
    fn process_trades(
        &mut self,
        py: Python,
        trades: Vec<Bound<PyDict>>,
        return_format: &str,
    ) -> PyResult<PyObject> {
        if trades.is_empty() {
            match return_format {
                "dict" => Ok(Vec::<PyObject>::new().into_py(py)),
                "arrow" => {
                    let empty_batch = rangebar_vec_to_record_batch(&[]);
                    Ok(PyRecordBatch::new(empty_batch).into_py(py))
                }
                _ => Err(PyValueError::new_err(
                    format!("Invalid return_format: '{}'. Must be 'dict' or 'arrow'", return_format)
                ))
            }
        } else {
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

            // Return in requested format
            match return_format {
                "dict" => {
                    // Convert RangeBars to Python dicts
                    let dicts: Vec<PyObject> = bars
                        .iter()
                        .map(|bar| rangebar_to_dict(py, bar))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(dicts.into_py(py))
                }
                "arrow" => {
                    // Convert RangeBars to Arrow RecordBatch (3-5x faster)
                    let batch = rangebar_vec_to_record_batch(&bars);
                    Ok(PyRecordBatch::new(batch).into_py(py))
                }
                _ => Err(PyValueError::new_err(
                    format!("Invalid return_format: '{}'. Must be 'dict' or 'arrow'", return_format)
                ))
            }
        }
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

        // Issue #96 Task #84: Process each trade individually to maintain state (Issue #16 fix)
        // Pre-allocate Vec with capacity estimate: typical bar completion is 2-10 trades per bar
        let mut bars = Vec::with_capacity((agg_trades.len() + 9) / 10);
        for trade in agg_trades {
            match self.processor.process_single_trade(&trade) {
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

        // Issue #96 Task #84: Process each trade individually to maintain state
        // Pre-allocate Vec with capacity estimate: typical bar completion is 2-10 trades per bar
        let mut bars = Vec::with_capacity((agg_trades.len() + 9) / 10);
        for trade in agg_trades {
            match self.processor.process_single_trade(&trade) {
                Ok(Some(bar)) => bars.push(bar),
                Ok(None) => {} // No bar completed yet
                Err(e) => return Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

        // Convert to Arrow RecordBatch
        let batch = rangebar_vec_to_record_batch(&bars);
        Ok(PyRecordBatch::new(batch))
    }

    /// Process trades from Arrow RecordBatch input, return Arrow RecordBatch output
    ///
    /// Issue #88: Arrow-native input path — eliminates .to_dicts() bottleneck.
    /// Accepts an Arrow RecordBatch with trade data and returns completed range bars
    /// as an Arrow RecordBatch. This bypasses Python dict creation entirely,
    /// providing ~3x speedup over the dict-based process_trades_streaming().
    ///
    /// Required columns: timestamp (Int64, ms), price (Float64), volume/quantity (Float64)
    /// Optional columns: agg_trade_id, first_trade_id, last_trade_id, is_buyer_maker, is_best_match
    ///
    /// Args:
    ///     batch: Arrow RecordBatch containing trade data
    ///
    /// Returns:
    ///     Arrow RecordBatch containing completed range bars
    ///
    /// Raises:
    ///     `ValueError`: If required columns are missing or have wrong types
    ///     `RuntimeError`: If trade processing fails
    #[cfg(feature = "arrow-export")]
    fn process_trades_arrow(&mut self, batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
        let record_batch = batch.as_ref();

        if record_batch.num_rows() == 0 {
            let empty_batch = rangebar_vec_to_record_batch(&[]);
            return Ok(PyRecordBatch::new(empty_batch));
        }

        // Convert Arrow RecordBatch to AggTrades (zero Python interaction)
        // timestamp_is_microseconds=false: Python API passes millisecond timestamps
        let agg_trades = record_batch_to_aggtrades(record_batch, false).map_err(|e| {
            PyValueError::new_err(format!("Arrow import failed: {e}"))
        })?;

        // Process each trade individually to maintain streaming state
        let mut bars = Vec::with_capacity(agg_trades.len() / 100);
        for trade in agg_trades {
            match self.processor.process_single_trade(&trade) {
                Ok(Some(bar)) => bars.push(bar),
                Ok(None) => {}
                Err(e) => return Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

        let batch = rangebar_vec_to_record_batch(&bars);
        Ok(PyRecordBatch::new(batch))
    }

    /// Process trades from Arrow RecordBatch with microsecond timestamps (internal path).
    ///
    /// Unlike `process_trades_arrow()` which expects millisecond timestamps from Python,
    /// this method expects microsecond timestamps as produced by `stream_binance_trades_arrow()`.
    /// This is the Phase 3 zero-copy pipeline: data never leaves Rust's internal format.
    ///
    /// Args:
    ///     batch: Arrow RecordBatch with microsecond timestamps (from `stream_binance_trades_arrow()`)
    ///
    /// Returns:
    ///     Arrow RecordBatch containing completed range bars
    ///
    /// Raises:
    ///     `ValueError`: If required columns are missing or have wrong types
    ///     `RuntimeError`: If trade processing fails
    #[cfg(feature = "arrow-export")]
    fn process_trades_arrow_native(&mut self, batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
        let record_batch = batch.as_ref();

        if record_batch.num_rows() == 0 {
            let empty_batch = rangebar_vec_to_record_batch(&[]);
            return Ok(PyRecordBatch::new(empty_batch));
        }

        // timestamp_is_microseconds=true: internal path, timestamps already in μs
        let agg_trades = record_batch_to_aggtrades(record_batch, true).map_err(|e| {
            PyValueError::new_err(format!("Arrow import failed: {e}"))
        })?;

        // Process each trade individually to maintain streaming state
        let mut bars = Vec::with_capacity(agg_trades.len() / 100);
        for trade in agg_trades {
            match self.processor.process_single_trade(&trade) {
                Ok(Some(bar)) => bars.push(bar),
                Ok(None) => {}
                Err(e) => return Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
            }
        }

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
