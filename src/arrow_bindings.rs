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

/// Issue #96 Task #76: Inline helper functions for batch dict extraction (1.5-2.5x speedup)
/// These are pulled out to avoid redefinition overhead per bar in large batches (10K+ bars)
#[inline]
fn get_f64_required(dict: &Bound<PyDict>, key: &str, index: usize) -> PyResult<f64> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("Bar {index}: missing '{key}'")))?
        .extract()
}

#[inline]
fn get_i64_required(dict: &Bound<PyDict>, key: &str, index: usize) -> PyResult<i64> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("Bar {index}: missing '{key}'")))?
        .extract()
}

#[inline]
fn get_u32_optional(dict: &Bound<PyDict>, key: &str, default: u32) -> PyResult<u32> {
    Ok(dict
        .get_item(key)?
        .and_then(|v| v.extract().ok())
        .unwrap_or(default))
}

#[inline]
fn get_f64_optional(dict: &Bound<PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(dict
        .get_item(key)?
        .and_then(|v| v.extract().ok())
        .unwrap_or(default))
}

#[inline]
fn get_i64_optional(dict: &Bound<PyDict>, key: &str, default: i64) -> PyResult<i64> {
    Ok(dict
        .get_item(key)?
        .and_then(|v| v.extract().ok())
        .unwrap_or(default))
}

/// Convert Python dict to Rust `RangeBar` (full conversion with all fields)
fn dict_to_rangebar_full(
    _py: Python,
    dict: &Bound<PyDict>,
    index: usize,
) -> PyResult<RangeBar> {

    // Core OHLCV
    let open = get_f64_required(dict, "open", index)?;
    let high = get_f64_required(dict, "high", index)?;
    let low = get_f64_required(dict, "low", index)?;
    let close = get_f64_required(dict, "close", index)?;
    let volume = get_f64_required(dict, "volume", index)?;

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
    let individual_trade_count = get_u32_optional(dict, "individual_trade_count", 0)?;
    let agg_record_count = get_u32_optional(dict, "agg_record_count", 0)?;
    let first_trade_id = get_i64_optional(dict, "first_trade_id", 0)?;
    let last_trade_id = get_i64_optional(dict, "last_trade_id", 0)?;
    // Issue #72: Aggregate trade ID range for data integrity verification
    let first_agg_trade_id = get_i64_optional(dict, "first_agg_trade_id", 0)?;
    let last_agg_trade_id = get_i64_optional(dict, "last_agg_trade_id", 0)?;

    // Order flow
    let buy_volume = get_f64_optional(dict, "buy_volume", 0.0)?;
    let sell_volume = get_f64_optional(dict, "sell_volume", 0.0)?;
    let buy_trade_count = get_u32_optional(dict, "buy_trade_count", 0)?;
    let sell_trade_count = get_u32_optional(dict, "sell_trade_count", 0)?;
    let vwap = get_f64_optional(dict, "vwap", 0.0)?;

    // Microstructure features
    let duration_us = dict
        .get_item("duration_us")?
        .and_then(|v| v.extract::<i64>().ok())
        .unwrap_or(0);
    let ofi = get_f64_optional(dict, "ofi", 0.0)?;
    let vwap_close_deviation = get_f64_optional(dict, "vwap_close_deviation", 0.0)?;
    let price_impact = get_f64_optional(dict, "price_impact", 0.0)?;
    let kyle_lambda_proxy = get_f64_optional(dict, "kyle_lambda_proxy", 0.0)?;
    let trade_intensity = get_f64_optional(dict, "trade_intensity", 0.0)?;
    let volume_per_trade = get_f64_optional(dict, "volume_per_trade", 0.0)?;
    let aggression_ratio = get_f64_optional(dict, "aggression_ratio", 0.0)?;
    let aggregation_density_f64 = get_f64_optional(dict, "aggregation_density", 0.0)?;
    let turnover_imbalance = get_f64_optional(dict, "turnover_imbalance", 0.0)?;

    Ok(RangeBar {
        open_time,
        close_time,
        open: f64_to_fixed_point(open),
        high: f64_to_fixed_point(high),
        low: f64_to_fixed_point(low),
        close: f64_to_fixed_point(close),
        // Issue #88: volume fields are i128, not FixedPoint
        volume: (volume * 100_000_000.0).round() as i128,
        turnover: 0, // Not typically stored in dict
        individual_trade_count,
        agg_record_count,
        first_trade_id,
        last_trade_id,
        first_agg_trade_id, // Issue #72
        last_agg_trade_id,  // Issue #72
        data_source: rangebar_core::DataSource::BinanceFuturesUM,
        buy_volume: (buy_volume * 100_000_000.0).round() as i128,
        sell_volume: (sell_volume * 100_000_000.0).round() as i128,
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
        // Inter-bar features (Issue #59) - initialized to None from dict parsing
        lookback_trade_count: None,
        lookback_ofi: None,
        lookback_duration_us: None,
        lookback_intensity: None,
        lookback_vwap_raw: None,
        lookback_vwap_position: None,
        lookback_count_imbalance: None,
        lookback_kyle_lambda: None,
        lookback_burstiness: None,
        lookback_volume_skew: None,
        lookback_volume_kurt: None,
        lookback_price_range: None,
        lookback_kaufman_er: None,
        lookback_garman_klass_vol: None,
        lookback_hurst: None,
        lookback_permutation_entropy: None,
        // Intra-bar features (Issue #59) - initialized to None from dict parsing
        intra_bull_epoch_density: None,
        intra_bear_epoch_density: None,
        intra_bull_excess_gain: None,
        intra_bear_excess_gain: None,
        intra_bull_cv: None,
        intra_bear_cv: None,
        intra_max_drawdown: None,
        intra_max_runup: None,
        intra_trade_count: None,
        intra_ofi: None,
        intra_duration_us: None,
        intra_intensity: None,
        intra_vwap_position: None,
        intra_count_imbalance: None,
        intra_kyle_lambda: None,
        intra_burstiness: None,
        intra_volume_skew: None,
        intra_volume_kurt: None,
        intra_kaufman_er: None,
        intra_garman_klass_vol: None,
        intra_hurst: None,
        intra_permutation_entropy: None,
    })
}
