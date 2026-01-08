use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rangebar_core::{AggTrade, FixedPoint, RangeBar, RangeBarProcessor};

#[cfg(feature = "data-providers")]
use rangebar_providers::exness::{
    ExnessInstrument, ExnessRangeBar, ExnessRangeBarBuilder, ExnessTick, SpreadStats,
    ValidationStrictness,
};

#[cfg(feature = "data-providers")]
use rangebar_providers::binance::HistoricalDataLoader;

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

    Ok(dict.into())
}

/// Python-exposed `RangeBarProcessor`
#[pyclass(name = "PyRangeBarProcessor")]
struct PyRangeBarProcessor {
    processor: RangeBarProcessor,
    threshold_bps: u32,
}

#[pymethods]
impl PyRangeBarProcessor {
    /// Create new processor
    ///
    /// Args:
    ///     `threshold_bps`: Threshold in 0.1 basis point units (250 = 25bps = 0.25%)
    ///
    /// Raises:
    ///     `ValueError`: If threshold is out of range [1, `100_000`]
    #[new]
    fn new(threshold_bps: u32) -> PyResult<Self> {
        let processor = RangeBarProcessor::new(threshold_bps)
            .map_err(|e| PyValueError::new_err(format!("Failed to create processor: {e}")))?;

        Ok(Self {
            processor,
            threshold_bps,
        })
    }

    /// Process aggregated trades into range bars
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

    /// Get threshold value
    #[getter]
    const fn threshold_bps(&self) -> u32 {
        self.threshold_bps
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
        threshold_bps: u32,
        instrument: PyExnessInstrument,
    }

    #[pymethods]
    impl PyExnessRangeBarBuilder {
        /// Create new builder for instrument
        ///
        /// Args:
        ///     instrument: ExnessInstrument enum value
        ///     threshold_bps: Threshold in 0.1 basis point units (250 = 25bps = 0.25%)
        ///     strictness: ValidationStrictness enum (default: Strict)
        #[new]
        #[pyo3(signature = (instrument, threshold_bps, strictness = PyValidationStrictness::Strict))]
        fn new(
            instrument: PyExnessInstrument,
            threshold_bps: u32,
            strictness: PyValidationStrictness,
        ) -> PyResult<Self> {
            let builder = ExnessRangeBarBuilder::for_instrument(
                instrument.to_rust(),
                threshold_bps,
                strictness.to_rust(),
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create builder: {e}")))?;

            Ok(Self {
                builder,
                threshold_bps,
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
        const fn threshold_bps(&self) -> u32 {
            self.threshold_bps
        }

        /// Get instrument
        #[getter]
        const fn instrument(&self) -> PyExnessInstrument {
            self.instrument
        }
    }
}

/// Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRangeBarProcessor>()?;

    // Add Exness classes if feature enabled
    #[cfg(feature = "data-providers")]
    {
        m.add_class::<exness_bindings::PyExnessInstrument>()?;
        m.add_class::<exness_bindings::PyValidationStrictness>()?;
        m.add_class::<exness_bindings::PyExnessRangeBarBuilder>()?;
        m.add_class::<binance_bindings::PyMarketType>()?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::fetch_binance_aggtrades,
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
        fn to_market_str(self) -> &'static str {
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
    ///
    /// Returns:
    ///     List of trade dicts with keys: timestamp, price, quantity, agg_trade_id,
    ///     first_trade_id, last_trade_id, is_buyer_maker
    #[pyfunction]
    #[pyo3(signature = (symbol, start_date, end_date, market_type = PyMarketType::Spot))]
    pub fn fetch_binance_aggtrades(
        py: Python,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        market_type: PyMarketType,
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

        // Load each day in the range
        let mut all_trades = Vec::new();
        let mut current_date = start;

        while current_date <= end {
            let day_result = rt.block_on(loader.load_single_day_trades(current_date));
            match day_result {
                Ok(mut day_trades) => all_trades.append(&mut day_trades),
                Err(e) => {
                    // Skip days with no data (weekends, holidays, etc.)
                    eprintln!("Warning: No data for {}: {}", current_date, e);
                }
            }
            current_date += chrono::Duration::days(1);
        }

        if all_trades.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "No data available for {} from {} to {}",
                symbol, start_date, end_date
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
        let processor = PyRangeBarProcessor::new(250);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().threshold_bps, 250);
    }

    #[test]
    fn test_invalid_threshold() {
        let processor = PyRangeBarProcessor::new(0);
        assert!(processor.is_err());

        let processor = PyRangeBarProcessor::new(200_000);
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
        let processor_min = PyRangeBarProcessor::new(1);
        assert!(processor_min.is_ok());
        assert_eq!(processor_min.unwrap().threshold_bps, 1);

        // Test maximum valid threshold (100_000 = 10,000 basis points = 100%)
        let processor_max = PyRangeBarProcessor::new(100_000);
        assert!(processor_max.is_ok());
        assert_eq!(processor_max.unwrap().threshold_bps, 100_000);

        // Test common valid thresholds
        for threshold in [10, 100, 250, 500, 1000, 10_000] {
            let processor = PyRangeBarProcessor::new(threshold);
            assert!(processor.is_ok(), "Threshold {} should be valid", threshold);
        }
    }
}
