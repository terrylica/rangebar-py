//! Arrow export utilities for streaming architecture
//!
//! Converts rangebar types to Arrow RecordBatch for zero-copy Python interop.
//! Requires the `arrow` feature flag.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rangebar_core::arrow_export::rangebar_vec_to_record_batch;
//!
//! let bars: Vec<RangeBar> = processor.process_trades(&trades);
//! let batch = rangebar_vec_to_record_batch(&bars);
//! // batch can now be passed to Python via pyo3-arrow
//! ```

use arrow_array::{BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::types::{AggTrade, DataSource, RangeBar};

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

    // Extract columns (vectorized iteration, no per-element allocation)
    let open_time: Int64Array = bars.iter().map(|b| b.open_time).collect();
    let close_time: Int64Array = bars.iter().map(|b| b.close_time).collect();
    let open: Float64Array = bars.iter().map(|b| b.open.to_f64()).collect();
    let high: Float64Array = bars.iter().map(|b| b.high.to_f64()).collect();
    let low: Float64Array = bars.iter().map(|b| b.low.to_f64()).collect();
    let close: Float64Array = bars.iter().map(|b| b.close.to_f64()).collect();
    let volume: Float64Array = bars.iter().map(|b| b.volume.to_f64()).collect();
    // Convert i128 turnover to f64 (safe for typical trading volumes)
    let turnover: Float64Array = bars.iter().map(|b| b.turnover as f64).collect();

    let individual_trade_count: UInt32Array =
        bars.iter().map(|b| b.individual_trade_count).collect();
    let agg_record_count: UInt32Array = bars.iter().map(|b| b.agg_record_count).collect();
    let first_trade_id: Int64Array = bars.iter().map(|b| b.first_trade_id).collect();
    let last_trade_id: Int64Array = bars.iter().map(|b| b.last_trade_id).collect();
    // StringArray requires Option<&str> for FromIterator
    let data_source: StringArray = bars
        .iter()
        .map(|b| {
            Some(match b.data_source {
                DataSource::BinanceSpot => "BinanceSpot",
                DataSource::BinanceFuturesUM => "BinanceFuturesUM",
                DataSource::BinanceFuturesCM => "BinanceFuturesCM",
            })
        })
        .collect();

    let buy_volume: Float64Array = bars.iter().map(|b| b.buy_volume.to_f64()).collect();
    let sell_volume: Float64Array = bars.iter().map(|b| b.sell_volume.to_f64()).collect();
    let buy_trade_count: UInt32Array = bars.iter().map(|b| b.buy_trade_count).collect();
    let sell_trade_count: UInt32Array = bars.iter().map(|b| b.sell_trade_count).collect();
    let vwap: Float64Array = bars.iter().map(|b| b.vwap.to_f64()).collect();
    // Convert i128 turnover to f64
    let buy_turnover: Float64Array = bars.iter().map(|b| b.buy_turnover as f64).collect();
    let sell_turnover: Float64Array = bars.iter().map(|b| b.sell_turnover as f64).collect();

    // Microstructure features
    let duration_us: Int64Array = bars.iter().map(|b| b.duration_us).collect();
    let ofi: Float64Array = bars.iter().map(|b| b.ofi).collect();
    let vwap_close_deviation: Float64Array = bars.iter().map(|b| b.vwap_close_deviation).collect();
    let price_impact: Float64Array = bars.iter().map(|b| b.price_impact).collect();
    let kyle_lambda_proxy: Float64Array = bars.iter().map(|b| b.kyle_lambda_proxy).collect();
    let trade_intensity: Float64Array = bars.iter().map(|b| b.trade_intensity).collect();
    let volume_per_trade: Float64Array = bars.iter().map(|b| b.volume_per_trade).collect();
    let aggression_ratio: Float64Array = bars.iter().map(|b| b.aggression_ratio).collect();
    let aggregation_density_f64: Float64Array =
        bars.iter().map(|b| b.aggregation_density_f64).collect();
    let turnover_imbalance: Float64Array = bars.iter().map(|b| b.turnover_imbalance).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(open_time),
            Arc::new(close_time),
            Arc::new(open),
            Arc::new(high),
            Arc::new(low),
            Arc::new(close),
            Arc::new(volume),
            Arc::new(turnover),
            Arc::new(individual_trade_count),
            Arc::new(agg_record_count),
            Arc::new(first_trade_id),
            Arc::new(last_trade_id),
            Arc::new(data_source),
            Arc::new(buy_volume),
            Arc::new(sell_volume),
            Arc::new(buy_trade_count),
            Arc::new(sell_trade_count),
            Arc::new(vwap),
            Arc::new(buy_turnover),
            Arc::new(sell_turnover),
            Arc::new(duration_us),
            Arc::new(ofi),
            Arc::new(vwap_close_deviation),
            Arc::new(price_impact),
            Arc::new(kyle_lambda_proxy),
            Arc::new(trade_intensity),
            Arc::new(volume_per_trade),
            Arc::new(aggression_ratio),
            Arc::new(aggregation_density_f64),
            Arc::new(turnover_imbalance),
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
        assert_eq!(batch.num_columns(), 30);

        // Verify schema
        assert_eq!(batch.schema().field(0).name(), "open_time");
        assert_eq!(batch.schema().field(29).name(), "turnover_imbalance");
    }

    #[test]
    fn test_rangebar_to_record_batch_multiple() {
        let bars: Vec<RangeBar> = (0..1000).map(|_| create_test_bar()).collect();
        let batch = rangebar_vec_to_record_batch(&bars);

        assert_eq!(batch.num_rows(), 1000);
        assert_eq!(batch.num_columns(), 30);
    }

    #[test]
    fn test_rangebar_to_record_batch_empty() {
        let batch = rangebar_vec_to_record_batch(&[]);
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 30);
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

        let data_source_col = batch
            .column(12)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(data_source_col.value(0), "BinanceSpot");
    }
}
