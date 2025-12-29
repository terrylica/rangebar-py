use pyo3::prelude::*;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::types::PyDict;
use rangebar_core::{AggTrade, FixedPoint, RangeBar, RangeBarProcessor};

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
                PyKeyError::new_err(format!(
                    "Trade {index}: missing 'quantity' or 'volume'"
                ))
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
    let datetime = chrono::DateTime::from_timestamp(timestamp_seconds as i64,
        (timestamp_seconds.fract() * 1_000_000_000.0) as u32)
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
        let processor = RangeBarProcessor::new(threshold_bps).map_err(|e| {
            PyValueError::new_err(format!("Failed to create processor: {e}"))
        })?;

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
    fn process_trades(&mut self, py: Python, trades: Vec<Bound<PyDict>>) -> PyResult<Vec<PyObject>> {
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
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Processing failed: {e}"))
            })?;

        // Convert RangeBars to Python dicts
        bars.iter()
            .map(|bar| rangebar_to_dict(py, bar))
            .collect()
    }

    /// Get threshold value
    #[getter]
    const fn threshold_bps(&self) -> u32 {
        self.threshold_bps
    }
}

/// Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRangeBarProcessor>()?;
    Ok(())
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
