use super::*;

/// Convert f64 to `FixedPoint` (8 decimal precision)
pub(crate) fn f64_to_fixed_point(value: f64) -> FixedPoint {
    // FixedPoint uses i64 with 8 decimal places (scale = 100_000_000)
    let scaled = (value * 100_000_000.0).round() as i64;
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

/// Convert Rust `RangeBar` to Python dict
pub(crate) fn rangebar_to_dict(py: Python, bar: &RangeBar) -> PyResult<PyObject> {
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
    // Issue #88: i128 volume → f64 (FixedPoint scale)
    dict.set_item("volume", bar.volume as f64 / 100_000_000.0)?;

    // Optional: Include market microstructure data
    dict.set_item("vwap", bar.vwap.to_f64())?;
    dict.set_item("buy_volume", bar.buy_volume as f64 / 100_000_000.0)?;
    dict.set_item("sell_volume", bar.sell_volume as f64 / 100_000_000.0)?;
    dict.set_item("individual_trade_count", bar.individual_trade_count)?;
    dict.set_item("agg_record_count", bar.agg_record_count)?;

    // Trade ID range (Issue #72)
    dict.set_item("first_agg_trade_id", bar.first_agg_trade_id)?;
    dict.set_item("last_agg_trade_id", bar.last_agg_trade_id)?;

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

    // Inter-bar features (Issue #59) - computed from lookback window BEFORE bar opens
    // Tier 1: Core features
    dict.set_item("lookback_trade_count", bar.lookback_trade_count)?;
    dict.set_item("lookback_ofi", bar.lookback_ofi)?;
    dict.set_item("lookback_duration_us", bar.lookback_duration_us)?;
    dict.set_item("lookback_intensity", bar.lookback_intensity)?;
    dict.set_item("lookback_vwap_raw", bar.lookback_vwap_raw)?;
    dict.set_item("lookback_vwap_position", bar.lookback_vwap_position)?;
    dict.set_item("lookback_count_imbalance", bar.lookback_count_imbalance)?;

    // Tier 2: Statistical features
    dict.set_item("lookback_kyle_lambda", bar.lookback_kyle_lambda)?;
    dict.set_item("lookback_burstiness", bar.lookback_burstiness)?;
    dict.set_item("lookback_volume_skew", bar.lookback_volume_skew)?;
    dict.set_item("lookback_volume_kurt", bar.lookback_volume_kurt)?;
    dict.set_item("lookback_price_range", bar.lookback_price_range)?;

    // Tier 3: trading-fitness features
    dict.set_item("lookback_kaufman_er", bar.lookback_kaufman_er)?;
    dict.set_item("lookback_garman_klass_vol", bar.lookback_garman_klass_vol)?;
    dict.set_item("lookback_hurst", bar.lookback_hurst)?;
    dict.set_item("lookback_permutation_entropy", bar.lookback_permutation_entropy)?;

    // Intra-bar features (Issue #59) - computed from trades WITHIN each bar
    // ITH features (Investment Time Horizon)
    dict.set_item("intra_bull_epoch_density", bar.intra_bull_epoch_density)?;
    dict.set_item("intra_bear_epoch_density", bar.intra_bear_epoch_density)?;
    dict.set_item("intra_bull_excess_gain", bar.intra_bull_excess_gain)?;
    dict.set_item("intra_bear_excess_gain", bar.intra_bear_excess_gain)?;
    dict.set_item("intra_bull_cv", bar.intra_bull_cv)?;
    dict.set_item("intra_bear_cv", bar.intra_bear_cv)?;
    dict.set_item("intra_max_drawdown", bar.intra_max_drawdown)?;
    dict.set_item("intra_max_runup", bar.intra_max_runup)?;

    // Statistical features
    dict.set_item("intra_trade_count", bar.intra_trade_count)?;
    dict.set_item("intra_ofi", bar.intra_ofi)?;
    dict.set_item("intra_duration_us", bar.intra_duration_us)?;
    dict.set_item("intra_intensity", bar.intra_intensity)?;
    dict.set_item("intra_vwap_position", bar.intra_vwap_position)?;
    dict.set_item("intra_count_imbalance", bar.intra_count_imbalance)?;
    dict.set_item("intra_kyle_lambda", bar.intra_kyle_lambda)?;
    dict.set_item("intra_burstiness", bar.intra_burstiness)?;
    dict.set_item("intra_volume_skew", bar.intra_volume_skew)?;
    dict.set_item("intra_volume_kurt", bar.intra_volume_kurt)?;
    dict.set_item("intra_kaufman_er", bar.intra_kaufman_er)?;
    dict.set_item("intra_garman_klass_vol", bar.intra_garman_klass_vol)?;

    // Complexity features
    dict.set_item("intra_hurst", bar.intra_hurst)?;
    dict.set_item("intra_permutation_entropy", bar.intra_permutation_entropy)?;

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
        bar_dict.set_item("volume", bar.volume as f64 / 100_000_000.0)?; // Issue #88: i128
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
pub(crate) fn dict_to_rangebar(_py: Python, dict: &Bound<PyDict>) -> PyResult<RangeBar> {
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
        // Issue #88: volume fields are i128, not FixedPoint
        volume: (volume * 100_000_000.0).round() as i128,
        turnover: 0,
        individual_trade_count: 0,
        agg_record_count,
        first_trade_id: 0,
        last_trade_id: 0,
        first_agg_trade_id: 0, // Issue #72 - default 0 for backward compatibility
        last_agg_trade_id: 0,  // Issue #72 - default 0 for backward compatibility
        data_source: rangebar_core::DataSource::BinanceSpot,
        buy_volume: 0i128,
        sell_volume: 0i128,
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
        // Inter-bar features (Issue #59) - initialized to None
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
        // Intra-bar features (Issue #59) - initialized to None
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
