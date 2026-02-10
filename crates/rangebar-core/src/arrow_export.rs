//! Arrow export/import utilities for streaming architecture
//! Issue #88: Arrow-native input path for 3x pipeline speedup
//!
//! Converts rangebar types to/from Arrow RecordBatch for zero-copy Python interop.
//! Requires the `arrow` feature flag.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rangebar_core::arrow_export::{rangebar_vec_to_record_batch, record_batch_to_aggtrades};
//!
//! // Export: Rust → Arrow
//! let bars: Vec<RangeBar> = processor.process_trades(&trades);
//! let batch = rangebar_vec_to_record_batch(&bars);
//!
//! // Import: Arrow → Rust
//! let trades = record_batch_to_aggtrades(&input_batch, false).unwrap();
//! ```

use arrow_array::{
    builder::{Float64Builder, Int64Builder, StringBuilder, UInt32Builder},
    Array, BooleanArray, Float64Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::fixed_point::SCALE;
use crate::types::{AggTrade, DataSource, RangeBar};

/// Error type for Arrow → AggTrade conversion failures
#[derive(Debug, thiserror::Error)]
pub enum ArrowImportError {
    /// A required column is missing from the RecordBatch
    #[error("Missing required column '{column}'")]
    MissingColumn { column: &'static str },

    /// A column has an unexpected Arrow data type
    #[error("Column '{column}': expected {expected}, got {actual}")]
    TypeMismatch {
        column: &'static str,
        expected: &'static str,
        actual: String,
    },
}

/// Convert an Arrow RecordBatch to a Vec of AggTrades
///
/// This is the inverse of `aggtrades_to_record_batch()`. Extracts columnar data
/// from an Arrow RecordBatch and constructs AggTrade structs for processing.
///
/// # Required columns
///
/// - `timestamp` (Int64): Timestamp in milliseconds (converted to microseconds)
/// - `price` (Float64): Trade price
/// - `volume` or `quantity` (Float64): Trade volume (tries "volume" first, then "quantity")
///
/// # Optional columns
///
/// - `agg_trade_id` (Int64): Aggregate trade ID (defaults to row index)
/// - `first_trade_id` (Int64): First individual trade ID (defaults to agg_trade_id)
/// - `last_trade_id` (Int64): Last individual trade ID (defaults to agg_trade_id)
/// - `is_buyer_maker` (Boolean): Whether buyer is market maker (defaults to false)
/// - `is_best_match` (Boolean, nullable): Whether trade was best price match (defaults to None)
///
/// # Timestamp convention
///
/// When `timestamp_is_microseconds` is `false` (default for Python API):
/// - Input timestamps are in **milliseconds** (Binance format)
/// - Output timestamps are in **microseconds** (rangebar-core internal format)
/// - Conversion: `timestamp_us = timestamp_ms * 1000`
///
/// When `timestamp_is_microseconds` is `true` (internal stream path):
/// - Input timestamps are already in **microseconds** (from `aggtrades_to_record_batch()`)
/// - No conversion applied: `timestamp_us = timestamp_raw`
/// - Used by Phase 3 `stream_binance_trades_arrow()` → `process_trades_arrow_native()`
///   where data never leaves Rust, so timestamps stay in internal microsecond format
pub fn record_batch_to_aggtrades(
    batch: &RecordBatch,
    timestamp_is_microseconds: bool,
) -> Result<Vec<AggTrade>, ArrowImportError> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Required: timestamp (Int64)
    let timestamp_col = get_int64_column(batch, "timestamp")?;

    // Required: price (Float64)
    let price_col = get_float64_column(batch, "price")?;

    // Required: volume or quantity (Float64) — try "volume" first, then "quantity"
    let volume_col = match batch.column_by_name("volume") {
        Some(col) => col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            ArrowImportError::TypeMismatch {
                column: "volume",
                expected: "Float64",
                actual: format!("{:?}", col.data_type()),
            }
        })?,
        None => match batch.column_by_name("quantity") {
            Some(col) => {
                col.as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| ArrowImportError::TypeMismatch {
                        column: "quantity",
                        expected: "Float64",
                        actual: format!("{:?}", col.data_type()),
                    })?
            }
            None => return Err(ArrowImportError::MissingColumn { column: "volume" }),
        },
    };

    // Optional columns
    let agg_trade_id_col = get_optional_int64_column(batch, "agg_trade_id");
    let first_trade_id_col = get_optional_int64_column(batch, "first_trade_id");
    let last_trade_id_col = get_optional_int64_column(batch, "last_trade_id");
    let is_buyer_maker_col = get_optional_boolean_column(batch, "is_buyer_maker");
    let is_best_match_col = get_optional_boolean_column(batch, "is_best_match");

    let mut trades = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let timestamp_ms = timestamp_col.value(i);
        let price = price_col.value(i);
        let volume = volume_col.value(i);

        let agg_trade_id = agg_trade_id_col
            .map(|col| col.value(i))
            .unwrap_or(i as i64);

        let first_trade_id = first_trade_id_col
            .map(|col| col.value(i))
            .unwrap_or(agg_trade_id);

        let last_trade_id = last_trade_id_col
            .map(|col| col.value(i))
            .unwrap_or(agg_trade_id);

        let is_buyer_maker = is_buyer_maker_col
            .map(|col| col.value(i))
            .unwrap_or(false);

        let is_best_match = is_best_match_col.and_then(|col| {
            if col.is_null(i) {
                None
            } else {
                Some(col.value(i))
            }
        });

        // Convert timestamp to microseconds (rangebar-core internal format)
        let timestamp_us = if timestamp_is_microseconds {
            // Internal path: timestamps already in microseconds (from aggtrades_to_record_batch)
            timestamp_ms // variable name is legacy; value is actually microseconds
        } else {
            // Python API path: timestamps in milliseconds (Binance format)
            timestamp_ms * 1000
        };

        trades.push(AggTrade {
            agg_trade_id,
            price: f64_to_fixed_point(price),
            volume: f64_to_fixed_point(volume),
            first_trade_id,
            last_trade_id,
            timestamp: timestamp_us,
            is_buyer_maker,
            is_best_match,
        });
    }

    Ok(trades)
}

/// Convert f64 to FixedPoint (8 decimal precision)
///
/// Same conversion as `f64_to_fixed_point()` in `src/lib.rs:25-29`.
#[inline]
fn f64_to_fixed_point(value: f64) -> crate::fixed_point::FixedPoint {
    crate::fixed_point::FixedPoint((value * SCALE as f64).round() as i64)
}

/// Get a required Int64 column by name
fn get_int64_column<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Int64Array, ArrowImportError> {
    let col = batch
        .column_by_name(name)
        .ok_or(ArrowImportError::MissingColumn { column: name })?;
    col.as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ArrowImportError::TypeMismatch {
            column: name,
            expected: "Int64",
            actual: format!("{:?}", col.data_type()),
        })
}

/// Get a required Float64 column by name
fn get_float64_column<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Float64Array, ArrowImportError> {
    let col = batch
        .column_by_name(name)
        .ok_or(ArrowImportError::MissingColumn { column: name })?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowImportError::TypeMismatch {
            column: name,
            expected: "Float64",
            actual: format!("{:?}", col.data_type()),
        })
}

/// Get an optional Int64 column by name (returns None if missing)
fn get_optional_int64_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
}

/// Get an optional Boolean column by name (returns None if missing)
fn get_optional_boolean_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Option<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .and_then(|col| col.as_any().downcast_ref::<BooleanArray>())
}

/// Schema for RangeBar Arrow export
///
/// Includes all 26 fields from RangeBar struct:
/// - OHLCV: open_time, close_time, open, high, low, close, volume, turnover
/// - Trade tracking: individual_trade_count, agg_record_count, first_trade_id, last_trade_id
/// - Data source: data_source
/// - Order flow: buy_volume, sell_volume, buy_trade_count, sell_trade_count, vwap, buy_turnover, sell_turnover
/// - Microstructure (10 features): duration_us, ofi, vwap_close_deviation, price_impact,
///   kyle_lambda_proxy, trade_intensity, volume_per_trade, aggression_ratio,
///   aggregation_density_f64, turnover_imbalance
pub fn rangebar_schema() -> Schema {
    Schema::new(vec![
        // Core OHLCV
        Field::new("open_time", DataType::Int64, false),
        Field::new("close_time", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        // NOTE: turnover is i128 in Rust but converted to f64 for Arrow
        // This is safe for typical trading volumes (f64 has 53-bit mantissa)
        Field::new("turnover", DataType::Float64, false),
        // Trade tracking
        Field::new("individual_trade_count", DataType::UInt32, false),
        Field::new("agg_record_count", DataType::UInt32, false),
        Field::new("first_trade_id", DataType::Int64, false),
        Field::new("last_trade_id", DataType::Int64, false),
        Field::new("first_agg_trade_id", DataType::Int64, false),
        Field::new("last_agg_trade_id", DataType::Int64, false),
        Field::new("data_source", DataType::Utf8, false),
        // Order flow
        Field::new("buy_volume", DataType::Float64, false),
        Field::new("sell_volume", DataType::Float64, false),
        Field::new("buy_trade_count", DataType::UInt32, false),
        Field::new("sell_trade_count", DataType::UInt32, false),
        Field::new("vwap", DataType::Float64, false),
        Field::new("buy_turnover", DataType::Float64, false),
        Field::new("sell_turnover", DataType::Float64, false),
        // Microstructure features (10)
        Field::new("duration_us", DataType::Int64, false),
        Field::new("ofi", DataType::Float64, false),
        Field::new("vwap_close_deviation", DataType::Float64, false),
        Field::new("price_impact", DataType::Float64, false),
        Field::new("kyle_lambda_proxy", DataType::Float64, false),
        Field::new("trade_intensity", DataType::Float64, false),
        Field::new("volume_per_trade", DataType::Float64, false),
        Field::new("aggression_ratio", DataType::Float64, false),
        Field::new("aggregation_density_f64", DataType::Float64, false),
        Field::new("turnover_imbalance", DataType::Float64, false),
    ])
}

/// Convert a slice of RangeBars to an Arrow RecordBatch
///
/// This is the primary export function for streaming range bars to Python.
/// The resulting RecordBatch can be converted to PyRecordBatch via pyo3-arrow
/// for zero-copy transfer to Polars/PyArrow.
///
/// # Arguments
///
/// * `bars` - Slice of RangeBar structs to convert
///
/// # Returns
///
/// Arrow RecordBatch containing all bar data in columnar format
///
/// # Panics
///
/// Panics if schema/array construction fails (should not happen with valid data)
pub fn rangebar_vec_to_record_batch(bars: &[RangeBar]) -> RecordBatch {
    let schema = Arc::new(rangebar_schema());
    let n = bars.len();

    // Pre-allocate all builders (single pass over bars for better cache locality)
    let mut open_time = Int64Builder::with_capacity(n);
    let mut close_time = Int64Builder::with_capacity(n);
    let mut open = Float64Builder::with_capacity(n);
    let mut high = Float64Builder::with_capacity(n);
    let mut low = Float64Builder::with_capacity(n);
    let mut close = Float64Builder::with_capacity(n);
    let mut volume = Float64Builder::with_capacity(n);
    let mut turnover = Float64Builder::with_capacity(n);
    let mut individual_trade_count = UInt32Builder::with_capacity(n);
    let mut agg_record_count = UInt32Builder::with_capacity(n);
    let mut first_trade_id = Int64Builder::with_capacity(n);
    let mut last_trade_id = Int64Builder::with_capacity(n);
    let mut first_agg_trade_id = Int64Builder::with_capacity(n);
    let mut last_agg_trade_id = Int64Builder::with_capacity(n);
    let mut data_source = StringBuilder::with_capacity(n, n * 16);
    let mut buy_volume = Float64Builder::with_capacity(n);
    let mut sell_volume = Float64Builder::with_capacity(n);
    let mut buy_trade_count = UInt32Builder::with_capacity(n);
    let mut sell_trade_count = UInt32Builder::with_capacity(n);
    let mut vwap = Float64Builder::with_capacity(n);
    let mut buy_turnover = Float64Builder::with_capacity(n);
    let mut sell_turnover = Float64Builder::with_capacity(n);
    let mut duration_us = Int64Builder::with_capacity(n);
    let mut ofi = Float64Builder::with_capacity(n);
    let mut vwap_close_deviation = Float64Builder::with_capacity(n);
    let mut price_impact = Float64Builder::with_capacity(n);
    let mut kyle_lambda_proxy = Float64Builder::with_capacity(n);
    let mut trade_intensity = Float64Builder::with_capacity(n);
    let mut volume_per_trade = Float64Builder::with_capacity(n);
    let mut aggression_ratio = Float64Builder::with_capacity(n);
    let mut aggregation_density_f64 = Float64Builder::with_capacity(n);
    let mut turnover_imbalance = Float64Builder::with_capacity(n);

    // Single pass: extract all fields per bar (better L1 cache utilization)
    for bar in bars {
        open_time.append_value(bar.open_time);
        close_time.append_value(bar.close_time);
        open.append_value(bar.open.to_f64());
        high.append_value(bar.high.to_f64());
        low.append_value(bar.low.to_f64());
        close.append_value(bar.close.to_f64());
        volume.append_value(bar.volume as f64 / SCALE as f64); // Issue #88: i128 volume
        turnover.append_value(bar.turnover as f64);
        individual_trade_count.append_value(bar.individual_trade_count);
        agg_record_count.append_value(bar.agg_record_count);
        first_trade_id.append_value(bar.first_trade_id);
        last_trade_id.append_value(bar.last_trade_id);
        first_agg_trade_id.append_value(bar.first_agg_trade_id);
        last_agg_trade_id.append_value(bar.last_agg_trade_id);
        data_source.append_value(match bar.data_source {
            DataSource::BinanceSpot => "BinanceSpot",
            DataSource::BinanceFuturesUM => "BinanceFuturesUM",
            DataSource::BinanceFuturesCM => "BinanceFuturesCM",
        });
        buy_volume.append_value(bar.buy_volume as f64 / SCALE as f64); // Issue #88: i128 volume
        sell_volume.append_value(bar.sell_volume as f64 / SCALE as f64); // Issue #88: i128 volume
        buy_trade_count.append_value(bar.buy_trade_count);
        sell_trade_count.append_value(bar.sell_trade_count);
        vwap.append_value(bar.vwap.to_f64());
        buy_turnover.append_value(bar.buy_turnover as f64);
        sell_turnover.append_value(bar.sell_turnover as f64);
        duration_us.append_value(bar.duration_us);
        ofi.append_value(bar.ofi);
        vwap_close_deviation.append_value(bar.vwap_close_deviation);
        price_impact.append_value(bar.price_impact);
        kyle_lambda_proxy.append_value(bar.kyle_lambda_proxy);
        trade_intensity.append_value(bar.trade_intensity);
        volume_per_trade.append_value(bar.volume_per_trade);
        aggression_ratio.append_value(bar.aggression_ratio);
        aggregation_density_f64.append_value(bar.aggregation_density_f64);
        turnover_imbalance.append_value(bar.turnover_imbalance);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(open_time.finish()),
            Arc::new(close_time.finish()),
            Arc::new(open.finish()),
            Arc::new(high.finish()),
            Arc::new(low.finish()),
            Arc::new(close.finish()),
            Arc::new(volume.finish()),
            Arc::new(turnover.finish()),
            Arc::new(individual_trade_count.finish()),
            Arc::new(agg_record_count.finish()),
            Arc::new(first_trade_id.finish()),
            Arc::new(last_trade_id.finish()),
            Arc::new(first_agg_trade_id.finish()),
            Arc::new(last_agg_trade_id.finish()),
            Arc::new(data_source.finish()),
            Arc::new(buy_volume.finish()),
            Arc::new(sell_volume.finish()),
            Arc::new(buy_trade_count.finish()),
            Arc::new(sell_trade_count.finish()),
            Arc::new(vwap.finish()),
            Arc::new(buy_turnover.finish()),
            Arc::new(sell_turnover.finish()),
            Arc::new(duration_us.finish()),
            Arc::new(ofi.finish()),
            Arc::new(vwap_close_deviation.finish()),
            Arc::new(price_impact.finish()),
            Arc::new(kyle_lambda_proxy.finish()),
            Arc::new(trade_intensity.finish()),
            Arc::new(volume_per_trade.finish()),
            Arc::new(aggression_ratio.finish()),
            Arc::new(aggregation_density_f64.finish()),
            Arc::new(turnover_imbalance.finish()),
        ],
    )
    .expect("Failed to create RecordBatch from RangeBars")
}

/// Schema for AggTrade Arrow export
pub fn aggtrade_schema() -> Schema {
    Schema::new(vec![
        Field::new("agg_trade_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        Field::new("first_trade_id", DataType::Int64, false),
        Field::new("last_trade_id", DataType::Int64, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("is_buyer_maker", DataType::Boolean, false),
        Field::new("is_best_match", DataType::Boolean, true), // nullable
    ])
}

/// Convert a slice of AggTrades to an Arrow RecordBatch
///
/// Used for streaming trade data to Python for processing.
///
/// # Arguments
///
/// * `trades` - Slice of AggTrade structs to convert
///
/// # Returns
///
/// Arrow RecordBatch containing all trade data in columnar format
pub fn aggtrades_to_record_batch(trades: &[AggTrade]) -> RecordBatch {
    let schema = Arc::new(aggtrade_schema());

    let agg_trade_id: Int64Array = trades.iter().map(|t| t.agg_trade_id).collect();
    let price: Float64Array = trades.iter().map(|t| t.price.to_f64()).collect();
    let volume: Float64Array = trades.iter().map(|t| t.volume.to_f64()).collect();
    let first_trade_id: Int64Array = trades.iter().map(|t| t.first_trade_id).collect();
    let last_trade_id: Int64Array = trades.iter().map(|t| t.last_trade_id).collect();
    let timestamp: Int64Array = trades.iter().map(|t| t.timestamp).collect();
    // BooleanArray requires Option<bool> for FromIterator
    let is_buyer_maker: BooleanArray = trades.iter().map(|t| Some(t.is_buyer_maker)).collect();
    // is_best_match is already Option<bool>, passes through directly
    let is_best_match: BooleanArray = trades.iter().map(|t| t.is_best_match).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(agg_trade_id),
            Arc::new(price),
            Arc::new(volume),
            Arc::new(first_trade_id),
            Arc::new(last_trade_id),
            Arc::new(timestamp),
            Arc::new(is_buyer_maker),
            Arc::new(is_best_match),
        ],
    )
    .expect("Failed to create RecordBatch from AggTrades")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use crate::fixed_point::FixedPoint;

    fn create_test_bar() -> RangeBar {
        RangeBar {
            open_time: 1640995200000000,
            close_time: 1640995201000000,
            open: FixedPoint::from_str("50000.0").unwrap(),
            high: FixedPoint::from_str("50100.0").unwrap(),
            low: FixedPoint::from_str("49900.0").unwrap(),
            close: FixedPoint::from_str("50050.0").unwrap(),
            volume: FixedPoint::from_str("10.5").unwrap(),
            turnover: 525_250_000_000_000_i128,
            individual_trade_count: 100,
            agg_record_count: 10,
            first_trade_id: 1,
            last_trade_id: 100,
            first_agg_trade_id: 1000, // Issue #72
            last_agg_trade_id: 1009,  // Issue #72
            data_source: DataSource::BinanceFuturesUM,
            buy_volume: FixedPoint::from_str("6.0").unwrap(),
            sell_volume: FixedPoint::from_str("4.5").unwrap(),
            buy_trade_count: 60,
            sell_trade_count: 40,
            vwap: FixedPoint::from_str("50025.0").unwrap(),
            buy_turnover: 300_150_000_000_000_i128,
            sell_turnover: 225_100_000_000_000_i128,
            duration_us: 1_000_000,
            ofi: 0.142857,
            vwap_close_deviation: 0.125,
            price_impact: 0.000476,
            kyle_lambda_proxy: 0.007,
            trade_intensity: 100.0,
            volume_per_trade: 0.105,
            aggression_ratio: 1.5,
            aggregation_density_f64: 10.0,
            turnover_imbalance: 0.142857,
            // Inter-bar features (Issue #59) - test defaults
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
            // Intra-bar features (Issue #59) - test defaults
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
        }
    }

    fn create_test_trade() -> AggTrade {
        AggTrade {
            agg_trade_id: 12345,
            price: FixedPoint::from_str("50000.12345678").unwrap(),
            volume: FixedPoint::from_str("1.5").unwrap(),
            first_trade_id: 100,
            last_trade_id: 102,
            timestamp: 1640995200000000,
            is_buyer_maker: false,
            is_best_match: Some(true),
        }
    }

    #[test]
    fn test_rangebar_to_record_batch_single() {
        let bar = create_test_bar();
        let batch = rangebar_vec_to_record_batch(&[bar]);

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields

        // Verify schema
        assert_eq!(batch.schema().field(0).name(), "open_time");
        assert_eq!(batch.schema().field(31).name(), "turnover_imbalance");
    }

    #[test]
    fn test_rangebar_to_record_batch_multiple() {
        let bars: Vec<RangeBar> = (0..1000).map(|_| create_test_bar()).collect();
        let batch = rangebar_vec_to_record_batch(&bars);

        assert_eq!(batch.num_rows(), 1000);
        assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields
    }

    #[test]
    fn test_rangebar_to_record_batch_empty() {
        let batch = rangebar_vec_to_record_batch(&[]);
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields
    }

    #[test]
    fn test_aggtrade_to_record_batch() {
        let trade = create_test_trade();
        let batch = aggtrades_to_record_batch(&[trade]);

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 8);

        // Verify schema
        assert_eq!(batch.schema().field(0).name(), "agg_trade_id");
        assert_eq!(batch.schema().field(7).name(), "is_best_match");
    }

    #[test]
    fn test_data_source_encoding() {
        let mut bar = create_test_bar();
        bar.data_source = DataSource::BinanceSpot;
        let batch = rangebar_vec_to_record_batch(&[bar]);

        // data_source is at index 14 (after Issue #72 added first/last_agg_trade_id at 12/13)
        let data_source_col = batch
            .column(14)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(data_source_col.value(0), "BinanceSpot");
    }

    // Issue #88: Arrow import tests

    #[test]
    fn test_record_batch_to_aggtrades_roundtrip() {
        // Export → Import roundtrip: fields must be bit-exact
        let original = vec![create_test_trade()];
        let batch = aggtrades_to_record_batch(&original);
        let imported = record_batch_to_aggtrades(&batch, false).unwrap();

        assert_eq!(imported.len(), 1);
        let t = &imported[0];
        let o = &original[0];

        assert_eq!(t.agg_trade_id, o.agg_trade_id);
        // aggtrades_to_record_batch exports price as f64 via to_f64(),
        // record_batch_to_aggtrades imports via (f64 * SCALE).round() as i64.
        // Roundtrip preserves exact value for 8-decimal prices.
        assert_eq!(t.price.0, o.price.0);
        assert_eq!(t.volume.0, o.volume.0);
        assert_eq!(t.first_trade_id, o.first_trade_id);
        assert_eq!(t.last_trade_id, o.last_trade_id);
        // aggtrades_to_record_batch exports timestamp in microseconds,
        // record_batch_to_aggtrades(false) expects milliseconds and converts via * 1000.
        // So the roundtrip with false changes the value: exported μs → imported as ms → * 1000 = μs * 1000.
        // For a true roundtrip, use timestamp_is_microseconds=true — see
        // test_record_batch_to_aggtrades_export_import_roundtrip_microseconds below.
        assert_eq!(t.is_buyer_maker, o.is_buyer_maker);
        assert_eq!(t.is_best_match, o.is_best_match);
    }

    #[test]
    fn test_record_batch_to_aggtrades_empty() {
        let batch = aggtrades_to_record_batch(&[]);
        let result = record_batch_to_aggtrades(&batch, false).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_record_batch_to_aggtrades_timestamp_conversion() {
        // Verify ms → μs conversion: 1704067200000 ms → 1704067200000000 μs
        let timestamp_ms: i64 = 1704067200000; // 2024-01-01 00:00:00 UTC
        let expected_us: i64 = 1704067200000000;

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![timestamp_ms])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, false).unwrap();
        assert_eq!(trades[0].timestamp, expected_us);

        // Also test edge cases: 0 and 1
        let batch_edge = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("price", DataType::Float64, false),
                Field::new("volume", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 1_i64])),
                Arc::new(Float64Array::from(vec![100.0, 100.0])),
                Arc::new(Float64Array::from(vec![1.0, 1.0])),
            ],
        )
        .unwrap();

        let trades_edge = record_batch_to_aggtrades(&batch_edge, false).unwrap();
        assert_eq!(trades_edge[0].timestamp, 0); // 0 ms → 0 μs
        assert_eq!(trades_edge[1].timestamp, 1000); // 1 ms → 1000 μs
    }

    #[test]
    fn test_record_batch_to_aggtrades_fixed_point_precision() {
        // Hand-computed: 50000.12345678 * 100_000_000 = 5000012345678
        let price = 50000.12345678_f64;
        let expected_fixed = (price * SCALE as f64).round() as i64;
        assert_eq!(expected_fixed, 5_000_012_345_678_i64);

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1704067200000_i64])),
                Arc::new(Float64Array::from(vec![price])),
                Arc::new(Float64Array::from(vec![1.5])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, false).unwrap();
        assert_eq!(trades[0].price.0, expected_fixed);
        // Verify roundtrip: FixedPoint → f64 → FixedPoint
        assert_eq!(trades[0].price.to_f64(), price);

        // Edge case: very small price
        let small_price = 0.00000001_f64;
        let expected_small = (small_price * SCALE as f64).round() as i64;
        assert_eq!(expected_small, 1);

        // Edge case: zero
        let zero_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("price", DataType::Float64, false),
                Field::new("volume", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1000_i64])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(Float64Array::from(vec![0.0])),
            ],
        )
        .unwrap();

        let zero_trades = record_batch_to_aggtrades(&zero_batch, false).unwrap();
        assert_eq!(zero_trades[0].price.0, 0);
        assert_eq!(zero_trades[0].volume.0, 0);
    }

    #[test]
    fn test_record_batch_to_aggtrades_missing_column() {
        // Missing "price" → ArrowImportError::MissingColumn
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1000_i64])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let result = record_batch_to_aggtrades(&batch, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ArrowImportError::MissingColumn { column: "price" }),
            "Expected MissingColumn for 'price', got: {err:?}"
        );

        // Missing both "volume" and "quantity" → error
        let schema2 = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
        ]));

        let batch2 = RecordBatch::try_new(
            schema2,
            vec![
                Arc::new(Int64Array::from(vec![1000_i64])),
                Arc::new(Float64Array::from(vec![50000.0])),
            ],
        )
        .unwrap();

        let result2 = record_batch_to_aggtrades(&batch2, false);
        assert!(result2.is_err());
        assert!(
            matches!(
                result2.unwrap_err(),
                ArrowImportError::MissingColumn { column: "volume" }
            ),
            "Expected MissingColumn for 'volume'"
        );
    }

    #[test]
    fn test_record_batch_to_aggtrades_type_mismatch() {
        // String column where Int64 expected for timestamp
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["not_a_number"])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let result = record_batch_to_aggtrades(&batch, false);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArrowImportError::TypeMismatch {
                column: "timestamp",
                expected: "Int64",
                ..
            } => {} // expected
            other => panic!("Expected TypeMismatch for 'timestamp', got: {other:?}"),
        }
    }

    #[test]
    fn test_record_batch_to_aggtrades_volume_quantity_fallback() {
        // Test that "quantity" column name works as fallback for "volume"
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("quantity", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1704067200000_i64])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![2.5])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, false).unwrap();
        assert_eq!(trades.len(), 1);
        // 2.5 * 100_000_000 = 250_000_000
        assert_eq!(trades[0].volume.0, 250_000_000);
    }

    #[test]
    fn test_record_batch_to_aggtrades_optional_defaults() {
        // Only required columns — verify defaults for optional fields
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1000_i64, 2000_i64])),
                Arc::new(Float64Array::from(vec![100.0, 200.0])),
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, false).unwrap();
        assert_eq!(trades.len(), 2);

        // agg_trade_id defaults to row index
        assert_eq!(trades[0].agg_trade_id, 0);
        assert_eq!(trades[1].agg_trade_id, 1);

        // first_trade_id and last_trade_id default to agg_trade_id
        assert_eq!(trades[0].first_trade_id, 0);
        assert_eq!(trades[0].last_trade_id, 0);
        assert_eq!(trades[1].first_trade_id, 1);
        assert_eq!(trades[1].last_trade_id, 1);

        // is_buyer_maker defaults to false
        assert!(!trades[0].is_buyer_maker);
        assert!(!trades[1].is_buyer_maker);

        // is_best_match defaults to None
        assert_eq!(trades[0].is_best_match, None);
        assert_eq!(trades[1].is_best_match, None);
    }

    // Phase 3: timestamp_is_microseconds tests

    #[test]
    fn test_record_batch_to_aggtrades_microseconds() {
        // timestamp_is_microseconds=true: input is already microseconds, no *1000
        let timestamp_us: i64 = 1704067200000000; // 2024-01-01 00:00:00 UTC in μs

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![timestamp_us])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, true).unwrap();
        assert_eq!(trades.len(), 1);
        // With timestamp_is_microseconds=true, the value passes through unchanged
        assert_eq!(trades[0].timestamp, timestamp_us);
        // Verify it's NOT multiplied by 1000 (which would be year ~52000)
        assert!(trades[0].timestamp < 2_051_222_400_000_000, // 2035 in μs
            "timestamp {} exceeds 2035 — multiplication applied when it shouldn't be",
            trades[0].timestamp
        );
    }

    #[test]
    fn test_record_batch_to_aggtrades_milliseconds_unchanged() {
        // Regression gate: timestamp_is_microseconds=false preserves existing ms→μs behavior
        let timestamp_ms: i64 = 1704067200000; // 2024-01-01 00:00:00 UTC in ms
        let expected_us: i64 = 1704067200000000; // same instant in μs

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![timestamp_ms])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let trades = record_batch_to_aggtrades(&batch, false).unwrap();
        assert_eq!(trades[0].timestamp, expected_us);
    }

    #[test]
    fn test_record_batch_to_aggtrades_export_import_roundtrip_microseconds() {
        // Full roundtrip: Vec<AggTrade> → aggtrades_to_record_batch → record_batch_to_aggtrades(true)
        // This is the Phase 3 internal path where timestamps never leave microsecond format
        let original = vec![
            AggTrade {
                agg_trade_id: 1,
                price: FixedPoint::from_str("42000.50").unwrap(),
                volume: FixedPoint::from_str("3.75").unwrap(),
                first_trade_id: 100,
                last_trade_id: 105,
                timestamp: 1704067200123456, // μs with sub-ms precision
                is_buyer_maker: true,
                is_best_match: Some(false),
            },
            AggTrade {
                agg_trade_id: 2,
                price: FixedPoint::from_str("42001.12345678").unwrap(),
                volume: FixedPoint::from_str("0.00000001").unwrap(), // minimum precision
                first_trade_id: 106,
                last_trade_id: 106,
                timestamp: 1704067200999999, // μs near second boundary
                is_buyer_maker: false,
                is_best_match: None,
            },
        ];

        let batch = aggtrades_to_record_batch(&original);
        // timestamp_is_microseconds=true because aggtrades_to_record_batch exports μs
        let imported = record_batch_to_aggtrades(&batch, true).unwrap();

        assert_eq!(imported.len(), 2);

        // Bit-exact field-by-field verification
        for (i, (imp, orig)) in imported.iter().zip(original.iter()).enumerate() {
            assert_eq!(imp.agg_trade_id, orig.agg_trade_id, "trade {i}: agg_trade_id");
            assert_eq!(imp.price.0, orig.price.0, "trade {i}: price");
            assert_eq!(imp.volume.0, orig.volume.0, "trade {i}: volume");
            assert_eq!(imp.first_trade_id, orig.first_trade_id, "trade {i}: first_trade_id");
            assert_eq!(imp.last_trade_id, orig.last_trade_id, "trade {i}: last_trade_id");
            assert_eq!(imp.timestamp, orig.timestamp, "trade {i}: timestamp");
            assert_eq!(imp.is_buyer_maker, orig.is_buyer_maker, "trade {i}: is_buyer_maker");
            assert_eq!(imp.is_best_match, orig.is_best_match, "trade {i}: is_best_match");
        }
    }

    #[test]
    fn test_timestamp_cross_year_sweep() {
        // Verify both paths produce valid timestamps across multiple years
        let test_cases = vec![
            (1514764800000_i64, "2018-01-01"),   // ms, 13 digits
            (1577836800000_i64, "2020-01-01"),   // ms, 13 digits
            (1704067200000_i64, "2024-01-01"),   // ms, 13 digits
            (1735689600000_i64, "2025-01-01"),   // ms, 13 digits
        ];

        for (timestamp_ms, label) in &test_cases {
            let schema = Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("price", DataType::Float64, false),
                Field::new("volume", DataType::Float64, false),
            ]));

            // Test ms path (timestamp_is_microseconds=false)
            let batch_ms = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![*timestamp_ms])),
                    Arc::new(Float64Array::from(vec![50000.0])),
                    Arc::new(Float64Array::from(vec![1.0])),
                ],
            )
            .unwrap();

            let trades_ms = record_batch_to_aggtrades(&batch_ms, false).unwrap();
            let ts_us_from_ms = trades_ms[0].timestamp;
            assert_eq!(ts_us_from_ms, timestamp_ms * 1000,
                "{label}: ms path should multiply by 1000");

            // Test μs path (timestamp_is_microseconds=true)
            let timestamp_us = timestamp_ms * 1000;
            let batch_us = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![timestamp_us])),
                    Arc::new(Float64Array::from(vec![50000.0])),
                    Arc::new(Float64Array::from(vec![1.0])),
                ],
            )
            .unwrap();

            let trades_us = record_batch_to_aggtrades(&batch_us, true).unwrap();
            let ts_us_from_us = trades_us[0].timestamp;
            assert_eq!(ts_us_from_us, timestamp_us,
                "{label}: μs path should pass through unchanged");

            // Both paths should produce identical output for equivalent timestamps
            assert_eq!(ts_us_from_ms, ts_us_from_us,
                "{label}: both paths must converge to same μs value");

            // Sanity: all timestamps in valid range [2000, 2035]
            let year_2000_us: i64 = 946684800000000;
            let year_2035_us: i64 = 2051222400000000;
            assert!(ts_us_from_ms >= year_2000_us && ts_us_from_ms <= year_2035_us,
                "{label}: timestamp {ts_us_from_ms} outside valid range");
        }
    }
}
