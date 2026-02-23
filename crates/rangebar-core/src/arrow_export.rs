//! Arrow export/import utilities for streaming architecture
//! Issue #88: Arrow-native input path for 3x pipeline speedup
//!
//! Converts rangebar types to/from Arrow RecordBatch for zero-copy Python interop.
//! Requires the `arrow` feature flag.
//!
//! # FILE-SIZE-OK
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
    Array, BooleanArray, Float64Array, Int64Array, RecordBatch,
    builder::{Float64Builder, Int64Builder, StringBuilder, UInt32Builder},
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
            Some(col) => col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                ArrowImportError::TypeMismatch {
                    column: "quantity",
                    expected: "Float64",
                    actual: format!("{:?}", col.data_type()),
                }
            })?,
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

    // Issue #112: Batch columnar extraction - optimize required columns with iterators
    // Use iterator-based access for required columns (timestamp, price, volume) to improve
    // CPU cache locality and eliminate per-row method call overheads. Keep optional columns
    // with per-row .value(i) access since they're sparse and less critical to performance.
    // Expected speedup: 1.5-2x on Arrow→AggTrade conversion for large batches (100K+ trades)

    // Extract iterators for required columns (hot path)
    let timestamp_iter = timestamp_col.iter();
    let price_iter = price_col.iter();
    let volume_iter = volume_col.iter();

    // Process rows via zipped iterators for required columns
    for (i, ((timestamp_ms, price), volume)) in
        timestamp_iter.zip(price_iter).zip(volume_iter).enumerate()
    {
        // Unwrap required fields - Arrow guarantees these are non-null
        let timestamp_ms = timestamp_ms.expect("timestamp column has non-null rows");
        let price = price.expect("price column has non-null rows");
        let volume = volume.expect("volume column has non-null rows");

        // Handle optional columns via per-row access (cold path, sparse columns)
        let agg_trade_id = agg_trade_id_col.map(|col| col.value(i)).unwrap_or(i as i64);

        let first_trade_id = first_trade_id_col
            .map(|col| col.value(i))
            .unwrap_or(agg_trade_id);

        let last_trade_id = last_trade_id_col
            .map(|col| col.value(i))
            .unwrap_or(agg_trade_id);

        let is_buyer_maker = is_buyer_maker_col.map(|col| col.value(i)).unwrap_or(false);

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
fn get_optional_boolean_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a BooleanArray> {
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
// Tests moved to crates/rangebar-core/tests/arrow_export_tests.rs (Phase 1b refactoring)
