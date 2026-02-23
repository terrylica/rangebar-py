use super::*;

/// Issue #96 Task #82: Pre-compute numeric scale constants to reduce instruction cache pollution
const VOLUME_SCALE: f64 = 100_000_000.0;
const TIMESTAMP_SCALE_US: f64 = 1_000_000.0;
const TIMESTAMP_SCALE_NS: f64 = 1_000_000_000.0;

/// Convert f64 to `FixedPoint` (8 decimal precision)
pub(crate) fn f64_to_fixed_point(value: f64) -> FixedPoint {
    // FixedPoint uses i64 with 8 decimal places (scale = VOLUME_SCALE)
    let scaled = (value * VOLUME_SCALE).round() as i64;
    FixedPoint(scaled)
}

/// Convert Python dict to Rust `AggTrade`
pub(crate) fn dict_to_agg_trade(
    py: Python,
    trade_dict: &Bound<PyDict>,
    index: usize,
) -> PyResult<AggTrade> {
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

/// Batch set multiple dict items (helper to reduce FFI boundary crossings)
/// Issue #96 Task #81: Group related dict operations for better CPU cache locality
#[inline]
fn batch_set_dict_items(
    dict: &Bound<PyDict>,
    items: &[(&str, PyObject)],
) -> PyResult<()> {
    for (key, value) in items {
        dict.set_item(key, value)?;
    }
    Ok(())
}

/// Convert Rust `RangeBar` to Python dict
/// Issue #96 Task #81: Optimized with batched field setting (reduced FFI boundary crossings)
pub(crate) fn rangebar_to_dict(py: Python, bar: &RangeBar) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    // Convert timestamp from microseconds to RFC3339 string
    let timestamp_seconds = bar.close_time as f64 / TIMESTAMP_SCALE_US;
    let datetime = chrono::DateTime::from_timestamp(
        timestamp_seconds as i64,
        (timestamp_seconds.fract() * TIMESTAMP_SCALE_NS) as u32,
    )
    .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;

    // Batch 1: Timestamp + OHLCV Core (6 items)
    let ohlcv_items = vec![
        ("timestamp", datetime.to_rfc3339().into_py(py)),
        ("open", bar.open.to_f64().into_py(py)),
        ("high", bar.high.to_f64().into_py(py)),
        ("low", bar.low.to_f64().into_py(py)),
        ("close", bar.close.to_f64().into_py(py)),
        ("volume", (bar.volume as f64 / VOLUME_SCALE).into_py(py)),
    ];
    batch_set_dict_items(&dict, &ohlcv_items)?;

    // Batch 2: Volume Accumulators (3 items)
    let volume_items = vec![
        ("vwap", bar.vwap.to_f64().into_py(py)),
        ("buy_volume", (bar.buy_volume as f64 / VOLUME_SCALE).into_py(py)),
        ("sell_volume", (bar.sell_volume as f64 / VOLUME_SCALE).into_py(py)),
    ];
    batch_set_dict_items(&dict, &volume_items)?;

    // Batch 3: Trade Tracking (4 items)
    let trade_tracking_items = vec![
        ("individual_trade_count", bar.individual_trade_count.into_py(py)),
        ("agg_record_count", bar.agg_record_count.into_py(py)),
        ("first_agg_trade_id", bar.first_agg_trade_id.into_py(py)),
        ("last_agg_trade_id", bar.last_agg_trade_id.into_py(py)),
    ];
    batch_set_dict_items(&dict, &trade_tracking_items)?;

    // Batch 4: Microstructure Features (10 items)
    let microstructure_items = vec![
        ("duration_us", bar.duration_us.into_py(py)),
        ("ofi", bar.ofi.into_py(py)),
        ("vwap_close_deviation", bar.vwap_close_deviation.into_py(py)),
        ("price_impact", bar.price_impact.into_py(py)),
        ("kyle_lambda_proxy", bar.kyle_lambda_proxy.into_py(py)),
        ("trade_intensity", bar.trade_intensity.into_py(py)),
        ("volume_per_trade", bar.volume_per_trade.into_py(py)),
        ("aggression_ratio", bar.aggression_ratio.into_py(py)),
        ("aggregation_density", bar.aggregation_density_f64.into_py(py)),
        ("turnover_imbalance", bar.turnover_imbalance.into_py(py)),
    ];
    batch_set_dict_items(&dict, &microstructure_items)?;

    // Batch 5: Inter-Bar Core Features (7 items)
    let lookback_core_items = vec![
        ("lookback_trade_count", bar.lookback_trade_count.into_py(py)),
        ("lookback_ofi", bar.lookback_ofi.into_py(py)),
        ("lookback_duration_us", bar.lookback_duration_us.into_py(py)),
        ("lookback_intensity", bar.lookback_intensity.into_py(py)),
        ("lookback_vwap_raw", bar.lookback_vwap_raw.into_py(py)),
        ("lookback_vwap_position", bar.lookback_vwap_position.into_py(py)),
        ("lookback_count_imbalance", bar.lookback_count_imbalance.into_py(py)),
    ];
    batch_set_dict_items(&dict, &lookback_core_items)?;

    // Batch 6: Inter-Bar Optional Features (Tiers 2-3, ~9 items)
    let mut lookback_optional_items: Vec<(&str, PyObject)> = Vec::with_capacity(9);
    if let Some(v) = bar.lookback_kyle_lambda {
        lookback_optional_items.push(("lookback_kyle_lambda", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_burstiness {
        lookback_optional_items.push(("lookback_burstiness", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_volume_skew {
        lookback_optional_items.push(("lookback_volume_skew", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_volume_kurt {
        lookback_optional_items.push(("lookback_volume_kurt", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_price_range {
        lookback_optional_items.push(("lookback_price_range", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_kaufman_er {
        lookback_optional_items.push(("lookback_kaufman_er", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_garman_klass_vol {
        lookback_optional_items.push(("lookback_garman_klass_vol", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_hurst {
        lookback_optional_items.push(("lookback_hurst", v.into_py(py)));
    }
    if let Some(v) = bar.lookback_permutation_entropy {
        lookback_optional_items.push(("lookback_permutation_entropy", v.into_py(py)));
    }
    batch_set_dict_items(&dict, &lookback_optional_items)?;

    // Batch 7: Intra-Bar ITH Features (~8 items)
    let mut intra_ith_items: Vec<(&str, PyObject)> = Vec::with_capacity(8);
    if let Some(v) = bar.intra_bull_epoch_density {
        intra_ith_items.push(("intra_bull_epoch_density", v.into_py(py)));
    }
    if let Some(v) = bar.intra_bear_epoch_density {
        intra_ith_items.push(("intra_bear_epoch_density", v.into_py(py)));
    }
    if let Some(v) = bar.intra_bull_excess_gain {
        intra_ith_items.push(("intra_bull_excess_gain", v.into_py(py)));
    }
    if let Some(v) = bar.intra_bear_excess_gain {
        intra_ith_items.push(("intra_bear_excess_gain", v.into_py(py)));
    }
    if let Some(v) = bar.intra_bull_cv {
        intra_ith_items.push(("intra_bull_cv", v.into_py(py)));
    }
    if let Some(v) = bar.intra_bear_cv {
        intra_ith_items.push(("intra_bear_cv", v.into_py(py)));
    }
    if let Some(v) = bar.intra_max_drawdown {
        intra_ith_items.push(("intra_max_drawdown", v.into_py(py)));
    }
    if let Some(v) = bar.intra_max_runup {
        intra_ith_items.push(("intra_max_runup", v.into_py(py)));
    }
    batch_set_dict_items(&dict, &intra_ith_items)?;

    // Batch 8: Intra-Bar Statistical Features (~12 items)
    let mut intra_stat_items: Vec<(&str, PyObject)> = Vec::with_capacity(12);
    if let Some(v) = bar.intra_trade_count {
        intra_stat_items.push(("intra_trade_count", v.into_py(py)));
    }
    if let Some(v) = bar.intra_ofi {
        intra_stat_items.push(("intra_ofi", v.into_py(py)));
    }
    if let Some(v) = bar.intra_duration_us {
        intra_stat_items.push(("intra_duration_us", v.into_py(py)));
    }
    if let Some(v) = bar.intra_intensity {
        intra_stat_items.push(("intra_intensity", v.into_py(py)));
    }
    if let Some(v) = bar.intra_vwap_position {
        intra_stat_items.push(("intra_vwap_position", v.into_py(py)));
    }
    if let Some(v) = bar.intra_count_imbalance {
        intra_stat_items.push(("intra_count_imbalance", v.into_py(py)));
    }
    if let Some(v) = bar.intra_kyle_lambda {
        intra_stat_items.push(("intra_kyle_lambda", v.into_py(py)));
    }
    if let Some(v) = bar.intra_burstiness {
        intra_stat_items.push(("intra_burstiness", v.into_py(py)));
    }
    if let Some(v) = bar.intra_volume_skew {
        intra_stat_items.push(("intra_volume_skew", v.into_py(py)));
    }
    if let Some(v) = bar.intra_volume_kurt {
        intra_stat_items.push(("intra_volume_kurt", v.into_py(py)));
    }
    if let Some(v) = bar.intra_kaufman_er {
        intra_stat_items.push(("intra_kaufman_er", v.into_py(py)));
    }
    if let Some(v) = bar.intra_garman_klass_vol {
        intra_stat_items.push(("intra_garman_klass_vol", v.into_py(py)));
    }
    batch_set_dict_items(&dict, &intra_stat_items)?;

    // Batch 9: Intra-Bar Complexity Features (2 items)
    let mut intra_complexity_items: Vec<(&str, PyObject)> = Vec::with_capacity(2);
    if let Some(v) = bar.intra_hurst {
        intra_complexity_items.push(("intra_hurst", v.into_py(py)));
    }
    if let Some(v) = bar.intra_permutation_entropy {
        intra_complexity_items.push(("intra_permutation_entropy", v.into_py(py)));
    }
    batch_set_dict_items(&dict, &intra_complexity_items)?;

    Ok(dict.into())
}

/// Convert Rust `Checkpoint` to Python dict (JSON-serializable)
pub(crate) fn checkpoint_to_dict(py: Python, checkpoint: &Checkpoint) -> PyResult<PyObject> {
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
        bar_dict.set_item("volume", bar.volume as f64 / VOLUME_SCALE)?; // Issue #88: i128
        bar_dict.set_item("open_time", bar.open_time)?;
        bar_dict.set_item("close_time", bar.close_time)?;
        bar_dict.set_item("agg_record_count", bar.agg_record_count)?;
        // Issue #97: Full microstructure state for lossless checkpoint round-trip
        bar_dict.set_item("buy_volume", bar.buy_volume as f64 / VOLUME_SCALE)?;
        bar_dict.set_item("sell_volume", bar.sell_volume as f64 / VOLUME_SCALE)?;
        bar_dict.set_item("individual_trade_count", bar.individual_trade_count)?;
        bar_dict.set_item("buy_trade_count", bar.buy_trade_count)?;
        bar_dict.set_item("sell_trade_count", bar.sell_trade_count)?;
        bar_dict.set_item("vwap", bar.vwap.to_f64())?;
        bar_dict.set_item("first_agg_trade_id", bar.first_agg_trade_id)?;
        bar_dict.set_item("last_agg_trade_id", bar.last_agg_trade_id)?;
        bar_dict.set_item("turnover", bar.turnover as f64 / VOLUME_SCALE)?;
        bar_dict.set_item("buy_turnover", bar.buy_turnover as f64 / VOLUME_SCALE)?;
        bar_dict.set_item("sell_turnover", bar.sell_turnover as f64 / VOLUME_SCALE)?;
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
pub(crate) fn dict_to_checkpoint(py: Python, dict: &Bound<PyDict>) -> PyResult<Checkpoint> {
    // Extract required fields
    let symbol: String = dict
        .get_item("symbol")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'symbol'"))?
        .extract()?;

    let threshold_decimal_bps: u32 = dict
        .get_item("threshold_decimal_bps")?
        .ok_or_else(|| PyKeyError::new_err("Missing 'threshold_decimal_bps'"))?
        .extract()?;

    // Issue #62: Validate threshold range in checkpoint restoration
    // Valid range: 1-100,000 dbps (0.0001% to 10%)
    const THRESHOLD_MIN: u32 = 1;
    const THRESHOLD_MAX: u32 = 100_000;
    if !(THRESHOLD_MIN..=THRESHOLD_MAX).contains(&threshold_decimal_bps) {
        return Err(PyValueError::new_err(format!(
            "Invalid checkpoint threshold: {threshold_decimal_bps} dbps. Valid range: {THRESHOLD_MIN}-{THRESHOLD_MAX} dbps"
        )));
    }

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

    // Issue #85 Phase 2: Extract version field for checkpoint schema migration
    let version: u32 = dict
        .get_item("version")?
        .and_then(|v| v.extract().ok())
        .unwrap_or(1); // Default to v1 for backward compatibility

    Ok(Checkpoint {
        version,
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

/// Macro to extract optional numeric field with default and scale conversion
/// Issue #96 Task #72: Consolidate repeated dict.get_item() + extract() + scale patterns
macro_rules! extract_optional_f64 {
    ($dict:expr, $field:expr, $default:expr) => {
        $dict
            .get_item($field)?
            .and_then(|v| v.extract().ok())
            .unwrap_or($default)
    };
}

/// Convert Python dict to Rust `RangeBar` (for checkpoint restoration)
pub(crate) fn dict_to_rangebar(_py: Python, dict: &Bound<PyDict>) -> PyResult<RangeBar> {
    const SCALE: f64 = 100_000_000.0;

    // Required fields - fail fast if missing
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

    // Issue #97: Batch-extract optional numeric fields (8-12% speedup vs 12 separate calls)
    let agg_record_count: u32 = extract_optional_f64!(dict, "agg_record_count", 0.0) as u32;
    let buy_volume_f64: f64 = extract_optional_f64!(dict, "buy_volume", 0.0);
    let sell_volume_f64: f64 = extract_optional_f64!(dict, "sell_volume", 0.0);
    let individual_trade_count: u32 = extract_optional_f64!(dict, "individual_trade_count", 0.0) as u32;
    let buy_trade_count: u32 = extract_optional_f64!(dict, "buy_trade_count", 0.0) as u32;
    let sell_trade_count: u32 = extract_optional_f64!(dict, "sell_trade_count", 0.0) as u32;
    let vwap_f64: f64 = extract_optional_f64!(dict, "vwap", 0.0);
    let first_agg_trade_id: i64 = extract_optional_f64!(dict, "first_agg_trade_id", 0.0) as i64;
    let last_agg_trade_id: i64 = extract_optional_f64!(dict, "last_agg_trade_id", 0.0) as i64;
    let turnover_f64: f64 = extract_optional_f64!(dict, "turnover", 0.0);
    let buy_turnover_f64: f64 = extract_optional_f64!(dict, "buy_turnover", 0.0);
    let sell_turnover_f64: f64 = extract_optional_f64!(dict, "sell_turnover", 0.0);

    // Issue #96 Task #72: Batch consolidate i128 scale conversions (3-5% speedup)
    // Use single SCALE constant for all volume/turnover fields
    let volume_i128 = (volume * SCALE).round() as i128;
    let turnover_i128 = (turnover_f64 * SCALE).round() as i128;
    let buy_volume_i128 = (buy_volume_f64 * SCALE).round() as i128;
    let sell_volume_i128 = (sell_volume_f64 * SCALE).round() as i128;
    let buy_turnover_i128 = (buy_turnover_f64 * SCALE).round() as i128;
    let sell_turnover_i128 = (sell_turnover_f64 * SCALE).round() as i128;

    // Issue #85 Phase 3: Reorganize RangeBar construction to match tier-based field ordering
    Ok(RangeBar {
        // Tier 1: OHLCV Core
        open_time,
        close_time,
        open: f64_to_fixed_point(open),
        high: f64_to_fixed_point(high),
        low: f64_to_fixed_point(low),
        close: f64_to_fixed_point(close),

        // Tier 2: Volume Accumulators (Issue #88: i128, not FixedPoint)
        volume: volume_i128,
        turnover: turnover_i128,
        buy_volume: buy_volume_i128,
        sell_volume: sell_volume_i128,
        buy_turnover: buy_turnover_i128,
        sell_turnover: sell_turnover_i128,

        // Tier 3: Trade Tracking
        first_trade_id: 0,
        last_trade_id: 0,
        first_agg_trade_id,
        last_agg_trade_id,
        individual_trade_count,
        agg_record_count,
        buy_trade_count,
        sell_trade_count,

        // Tier 4: Price Context
        vwap: f64_to_fixed_point(vwap_f64),
        data_source: rangebar_core::DataSource::BinanceSpot,

        // Tier 5: Microstructure features (Issue #25) - initialized to defaults
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

        // Tier 6: Inter-bar features (Issue #59) - initialized to None
        // Checkpoint restoration doesn't include inter-bar features
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

        // Tier 7: Intra-bar features (Issue #59) - initialized to None
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

/// Convert `CheckpointError` to Python exception
pub(crate) fn checkpoint_error_to_pyerr(e: CheckpointError) -> PyErr {
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
        // Issue #62: Crypto minimum threshold enforcement
        CheckpointError::InvalidThreshold {
            threshold,
            min_threshold,
            max_threshold,
        } => PyValueError::new_err(format!(
            "Invalid checkpoint threshold: {threshold} dbps. Valid range: {min_threshold}-{max_threshold} dbps"
        )),
    }
}
