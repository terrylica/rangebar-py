//! Test utilities for consistent test data creation across the codebase
//!
//! This module provides centralized functions for creating test data structures,
//! eliminating hardcoded values scattered throughout test files.
//!
//! ## Module Organization
//!
//! - `mod.rs`: Small-scale unit test utilities (builders, scenarios)
//! - `generators.rs`: Large-scale integration test data generators

pub mod generators; // Large-scale data generation for integration tests

use crate::types::{AggTrade, DataSource, RangeBar};
use crate::FixedPoint;

/// Creates a standard test AggTrade with sensible defaults
pub fn create_test_agg_trade(id: i64, price: &str, volume: &str, timestamp: i64) -> AggTrade {
    AggTrade {
        agg_trade_id: id,
        price: FixedPoint::from_str(price).unwrap(),
        volume: FixedPoint::from_str(volume).unwrap(),
        first_trade_id: id * 10,
        last_trade_id: id * 10,
        timestamp,
        is_buyer_maker: id % 2 == 0, // Alternate buy/sell for realistic testing
        is_best_match: None,         // Default for futures data
    }
}

/// Creates a test AggTrade with custom trade ID range (for testing aggregation)
pub fn create_test_agg_trade_with_range(
    agg_id: i64,
    price: &str,
    volume: &str,
    timestamp: i64,
    first_trade_id: i64,
    last_trade_id: i64,
    is_buyer_maker: bool,
) -> AggTrade {
    AggTrade {
        agg_trade_id: agg_id,
        price: FixedPoint::from_str(price).unwrap(),
        volume: FixedPoint::from_str(volume).unwrap(),
        first_trade_id,
        last_trade_id,
        timestamp,
        is_buyer_maker,
        is_best_match: None,
    }
}

/// Creates a test AggTrade for spot market (with is_best_match field)
pub fn create_test_spot_agg_trade(
    id: i64,
    price: &str,
    volume: &str,
    timestamp: i64,
    is_best_match: bool,
) -> AggTrade {
    AggTrade {
        agg_trade_id: id,
        price: FixedPoint::from_str(price).unwrap(),
        volume: FixedPoint::from_str(volume).unwrap(),
        first_trade_id: id * 10,
        last_trade_id: id * 10,
        timestamp,
        is_buyer_maker: id % 2 == 0,
        is_best_match: Some(is_best_match),
    }
}

/// Creates a test RangeBar with sensible defaults
#[allow(clippy::too_many_arguments)]
pub fn create_test_range_bar(
    open_time: i64,
    close_time: i64,
    open: &str,
    high: &str,
    low: &str,
    close: &str,
    volume: &str,
    individual_trade_count: u32,
) -> RangeBar {
    RangeBar {
        open_time,
        close_time,
        open: FixedPoint::from_str(open).unwrap(),
        high: FixedPoint::from_str(high).unwrap(),
        low: FixedPoint::from_str(low).unwrap(),
        close: FixedPoint::from_str(close).unwrap(),
        volume: FixedPoint::from_str(volume).unwrap(),
        turnover: 0,
        individual_trade_count,
        agg_record_count: 1,
        first_trade_id: 1,
        last_trade_id: individual_trade_count as i64,
        data_source: DataSource::BinanceFuturesUM,
        buy_volume: FixedPoint::from_str("0.0").unwrap(),
        buy_turnover: 0,
        sell_volume: FixedPoint::from_str("0.0").unwrap(),
        sell_turnover: 0,
        buy_trade_count: 0,
        sell_trade_count: 0,
        vwap: FixedPoint::from_str(open).unwrap(), // Simple default
        // Microstructure features (Issue #25) - defaults
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
    }
}

/// Standard test constants for consistent testing
pub mod constants {
    pub const BASE_PRICE: &str = "50000.00000000";
    pub const BASE_VOLUME: &str = "1.50000000";
    pub const BASE_TIMESTAMP: i64 = 1640995200000; // 2022-01-01 00:00:00 UTC
    pub const BTCUSDT_PRICE: &str = "50000.00000000";
    pub const ETHUSDT_PRICE: &str = "4000.00000000";
}

/// Creates a sequence of test trades for breach testing
pub fn create_breach_test_sequence() -> Vec<AggTrade> {
    vec![
        create_test_agg_trade(1, "50000.0", "1.0", constants::BASE_TIMESTAMP),
        create_test_agg_trade(2, "50200.0", "1.0", constants::BASE_TIMESTAMP + 1000), // +0.4%
        create_test_agg_trade(3, "50300.0", "1.0", constants::BASE_TIMESTAMP + 2000), // +0.6% - breach
    ]
}

/// Builder pattern for creating custom AggTrade sequences
pub struct AggTradeBuilder {
    base_price: f64,
    base_timestamp: i64,
    base_volume: String,
    trades: Vec<AggTrade>,
}

impl Default for AggTradeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AggTradeBuilder {
    pub fn new() -> Self {
        Self {
            base_price: 50000.0,
            base_timestamp: constants::BASE_TIMESTAMP,
            base_volume: "1.0".to_string(),
            trades: Vec::new(),
        }
    }

    pub fn with_base_price(mut self, price: f64) -> Self {
        self.base_price = price;
        self
    }

    pub fn with_base_timestamp(mut self, timestamp: i64) -> Self {
        self.base_timestamp = timestamp;
        self
    }

    pub fn with_base_volume(mut self, volume: &str) -> Self {
        self.base_volume = volume.to_string();
        self
    }

    pub fn add_trade(mut self, id: i64, price_factor: f64, time_offset_ms: i64) -> Self {
        let price = self.base_price * price_factor;
        let trade = create_test_agg_trade(
            id,
            &format!("{:.8}", price),
            &self.base_volume,
            self.base_timestamp + time_offset_ms,
        );
        self.trades.push(trade);
        self
    }

    pub fn add_trade_with_volume(
        mut self,
        id: i64,
        price_factor: f64,
        volume: &str,
        time_offset_ms: i64,
    ) -> Self {
        let price = self.base_price * price_factor;
        let trade = create_test_agg_trade(
            id,
            &format!("{:.8}", price),
            volume,
            self.base_timestamp + time_offset_ms,
        );
        self.trades.push(trade);
        self
    }

    pub fn build(self) -> Vec<AggTrade> {
        self.trades
    }
}

/// Common test scenarios
pub mod scenarios {
    use super::*;

    /// Creates trades that should produce no range bars (all within threshold)
    /// v3.0.0: threshold now in dbps (divide by 100,000)
    pub fn no_breach_sequence(threshold_decimal_bps: u32) -> Vec<AggTrade> {
        let max_change = (threshold_decimal_bps as f64 / 100_000.0) * 0.8; // Stay within threshold
        AggTradeBuilder::new()
            .add_trade(1, 1.0, 0)
            .add_trade(2, 1.0 + max_change, 1000)
            .add_trade(3, 1.0 - max_change, 2000)
            .build()
    }

    /// Creates trades that should produce exactly one range bar
    /// v3.0.0: threshold now in dbps (divide by 100,000)
    pub fn single_breach_sequence(threshold_decimal_bps: u32) -> Vec<AggTrade> {
        let breach_change = (threshold_decimal_bps as f64 / 100_000.0) * 1.2; // Exceed threshold
        AggTradeBuilder::new()
            .add_trade(1, 1.0, 0)
            .add_trade(2, 1.0 + breach_change, 1000) // Breach upward
            .build()
    }

    /// Creates trades for testing empty trade arrays
    pub fn empty_sequence() -> Vec<AggTrade> {
        Vec::new()
    }

    /// Creates trades for exact threshold breach testing
    /// v3.0.0: threshold now in dbps (divide by 100,000)
    pub fn exact_breach_upward(threshold_decimal_bps: u32) -> Vec<AggTrade> {
        let breach_change = threshold_decimal_bps as f64 / 100_000.0; // Exact threshold
        AggTradeBuilder::new()
            .add_trade(1, 1.0, 0) // Open
            .add_trade(2, 1.0 + breach_change * 0.8, 1000) // Approach threshold
            .add_trade(3, 1.0 + breach_change, 2000) // Exact breach
            .add_trade(4, 1.01, 3000) // New bar start
            .build()
    }

    /// Creates trades for exact threshold breach testing (downward)
    /// v3.0.0: threshold now in dbps (divide by 100,000)
    pub fn exact_breach_downward(threshold_decimal_bps: u32) -> Vec<AggTrade> {
        let breach_change = threshold_decimal_bps as f64 / 100_000.0; // Exact threshold
        AggTradeBuilder::new()
            .add_trade(1, 1.0, 0) // Open
            .add_trade(2, 1.0 - breach_change * 0.8, 1000) // Approach threshold
            .add_trade(3, 1.0 - breach_change, 2000) // Exact breach
            .add_trade(4, 0.99, 3000) // New bar start
            .build()
    }

    /// Creates trades with large price gaps for gap testing
    pub fn large_gap_sequence() -> Vec<AggTrade> {
        AggTradeBuilder::new()
            .add_trade(1, 1.0, 0) // Open at 50000
            .add_trade(2, 1.02, 1000) // +2% gap to 51000
            .build()
    }

    /// Creates unsorted trades for error testing
    pub fn unsorted_sequence() -> Vec<AggTrade> {
        use super::constants;
        vec![
            create_test_agg_trade_with_range(
                1,
                "50000.0",
                "1.0",
                constants::BASE_TIMESTAMP + 2000,
                10,
                10,
                false,
            ), // Later timestamp
            create_test_agg_trade_with_range(
                2,
                "50100.0",
                "1.0",
                constants::BASE_TIMESTAMP + 1000,
                20,
                20,
                false,
            ), // Earlier timestamp
        ]
    }

    /// Creates a large sequence for performance testing
    pub fn large_sequence(count: usize) -> Vec<AggTrade> {
        let mut builder = AggTradeBuilder::new();
        for i in 0..count {
            let price_factor = 1.0 + (i as f64 * 0.001); // Gradual price increase
            builder = builder.add_trade(i as i64 + 1, price_factor, i as i64 * 100);
        }
        builder.build()
    }
}
