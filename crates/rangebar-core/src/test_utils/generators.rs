//! Large-scale test data generators for integration testing
//!
//! ## Service Level Objectives (SLOs)
//!
//! ### Availability SLO: 100% deterministic generation
//! - All functions are pure (same input → same output)
//! - No external dependencies or I/O
//! - No randomness (all patterns are mathematical functions)
//!
//! ### Correctness SLO: 100% data integrity
//! - All generated trades have valid timestamps (monotonically increasing)
//! - All prices are properly formatted (8 decimal places)
//! - No data corruption or invalid values
//!
//! ### Observability SLO: 100% parameter traceability
//! - All functions document their parameters and patterns
//! - Generated data characteristics are predictable
//! - Clear naming indicates the type of data generated
//!
//! ### Maintainability SLO: Single source of truth
//! - One implementation per helper function (no duplicates)
//! - Used by all integration test files
//! - Changes propagate automatically to all tests

use crate::FixedPoint;
use crate::processor::ExportRangeBarProcessor;
use crate::types::{AggTrade, RangeBar};

// =============================================================================
// Test Trade Creation (Centralized - used by all test files)
// =============================================================================

/// Create a test trade with formatted price (8 decimal places)
///
/// SLO: Fail-fast on parse errors (no defaults, no fallbacks)
pub fn create_test_trade(id: u64, price: f64, timestamp: u64) -> AggTrade {
    // Format price to 8 decimal places to avoid TooManyDecimals error
    let price_str = format!("{:.8}", price);
    AggTrade {
        agg_trade_id: id as i64,
        price: FixedPoint::from_str(&price_str).unwrap(),
        volume: FixedPoint::from_str("1.0").unwrap(),
        first_trade_id: id as i64,
        last_trade_id: id as i64,
        timestamp: timestamp as i64,
        is_buyer_maker: false,
        is_best_match: None,
    }
}

// =============================================================================
// Processing Functions (Batch and Streaming)
// =============================================================================

/// Process trades in batch style (single continuous processing)
///
/// Used for baseline comparison in integration tests
pub fn process_batch_style(trades: &[AggTrade], threshold_decimal_bps: u32) -> Vec<RangeBar> {
    let mut processor = ExportRangeBarProcessor::new(threshold_decimal_bps).unwrap();

    // Process all trades continuously (simulating boundary-safe mode)
    processor.process_trades_continuously(trades);

    // Get all completed bars
    let mut bars = processor.get_all_completed_bars();

    // Add incomplete bar if exists
    if let Some(incomplete) = processor.get_incomplete_bar() {
        bars.push(incomplete);
    }

    bars
}

/// Process trades in streaming style (chunked processing)
///
/// Simulates real-world streaming behavior with memory constraints
pub async fn process_streaming_style(
    trades: &[AggTrade],
    threshold_decimal_bps: u32,
) -> Vec<RangeBar> {
    // Use the corrected streaming approach that matches our fix
    let mut range_processor = ExportRangeBarProcessor::new(threshold_decimal_bps).unwrap();

    // Simulate the corrected streaming behavior:
    // Process in chunks and accumulate results (like our csv_streaming.rs fix)
    let chunk_size = 10000; // Larger chunks for performance
    let mut all_bars = Vec::new();

    for chunk in trades.chunks(chunk_size) {
        range_processor.process_trades_continuously(chunk);
        // Get completed bars from this chunk and clear state
        let chunk_bars = range_processor.get_all_completed_bars();
        all_bars.extend(chunk_bars);
    }

    // Add final incomplete bar if exists
    if let Some(incomplete) = range_processor.get_incomplete_bar() {
        all_bars.push(incomplete);
    }

    all_bars
}

// =============================================================================
// Large-Scale Data Generation (Multi-Million Trade Datasets)
// =============================================================================

/// Create massive realistic dataset for boundary testing
///
/// Generates realistic market conditions with:
/// - Long-term trend (sine wave)
/// - Volatility (multi-frequency oscillation)
/// - Market noise
pub fn create_massive_realistic_dataset(count: usize) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let base_price = 23000.0;
    let base_time = 1659312000000i64; // Aug 1, 2022

    // Simulate realistic market conditions
    for i in 0..count {
        let time_progress = i as f64 / count as f64;

        // Multi-layered price movement simulation
        let trend = (time_progress * 2.0 * std::f64::consts::PI).sin() * 500.0; // Long-term trend
        let volatility = ((i as f64 * 0.01).sin() * 50.0) + ((i as f64 * 0.001).cos() * 20.0); // Volatility
        let noise = (i as f64 * 0.1).sin() * 5.0; // Market noise

        let price = base_price + trend + volatility + noise;
        let timestamp = base_time + (i as i64 * 100); // 100ms intervals

        trades.push(create_test_trade(
            1000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

// =============================================================================
// Multi-Day Boundary Data Generation
// =============================================================================

/// Create multi-day boundary dataset
///
/// Each day has different trading patterns:
/// - Day 0, 3, 6: High volatility
/// - Day 1, 4: Low volatility
/// - Day 2, 5: Strong trend
pub fn create_multi_day_boundary_dataset(days: usize) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_time = 1659312000000i64; // Aug 1, 2022
    let day_ms = 24 * 60 * 60 * 1000; // Milliseconds per day

    for day in 0..days {
        let day_start = base_time + (day as i64 * day_ms);

        // Each day has different trading patterns
        let daily_trades = match day % 3 {
            0 => create_volatile_day_data(day_start, 100000), // High volatility
            1 => create_stable_day_data(day_start, 80000),    // Low volatility
            _ => create_trending_day_data(day_start, 120000), // Strong trend
        };

        trades.extend(daily_trades);
    }

    trades
}

/// Create volatile day data (frequent reversals)
pub fn create_volatile_day_data(start_time: i64, count: usize) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;

    for i in 0..count {
        // High volatility with frequent reversals
        let volatility = ((i as f64 * 0.02).sin() * 200.0) + ((i as f64 * 0.005).cos() * 100.0);
        let price = base_price + volatility;
        let timestamp = start_time + (i as i64 * 500); // 500ms intervals

        trades.push(create_test_trade(
            2000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

/// Create stable day data (low volatility, gradual movements)
pub fn create_stable_day_data(start_time: i64, count: usize) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;

    for i in 0..count {
        // Low volatility, gradual movements
        let movement = (i as f64 * 0.001).sin() * 20.0;
        let price = base_price + movement;
        let timestamp = start_time + (i as i64 * 800); // 800ms intervals

        trades.push(create_test_trade(
            3000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

/// Create trending day data (strong upward trend)
pub fn create_trending_day_data(start_time: i64, count: usize) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;

    for i in 0..count {
        // Strong upward trend with some noise
        let trend = (i as f64 / count as f64) * 800.0; // +800 over the day
        let noise = (i as f64 * 0.01).sin() * 30.0;
        let price = base_price + trend + noise;
        let timestamp = start_time + (i as i64 * 600); // 600ms intervals

        trades.push(create_test_trade(
            4000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

// =============================================================================
// Market Session Data Generation
// =============================================================================

/// Create Asian trading session data (lower volatility, steady)
pub fn create_asian_session_data() -> Vec<AggTrade> {
    create_session_data(1659312000000, 50000, 0.5, 0.8)
}

/// Create European trading session data (medium volatility, active)
pub fn create_european_session_data() -> Vec<AggTrade> {
    create_session_data(1659340800000, 80000, 1.0, 1.2)
}

/// Create US trading session data (high volatility, very active)
pub fn create_us_session_data() -> Vec<AggTrade> {
    create_session_data(1659369600000, 120000, 1.5, 2.0)
}

/// Create weekend gap data (very low activity)
pub fn create_weekend_gap_data() -> Vec<AggTrade> {
    create_session_data(1659484800000, 5000, 0.2, 0.3)
}

/// Generic session data generator
fn create_session_data(
    start_time: i64,
    count: usize,
    volatility_factor: f64,
    activity_factor: f64,
) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;

    for i in 0..count {
        let volatility = ((i as f64 * 0.01).sin() * 100.0 * volatility_factor)
            + ((i as f64 * 0.003).cos() * 50.0 * volatility_factor);
        let price = base_price + volatility;
        let interval = (1000.0 / activity_factor) as i64; // Adjust interval based on activity
        let timestamp = start_time + (i as i64 * interval);

        trades.push(create_test_trade(
            5000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

// =============================================================================
// Frequency Variation Data Generation
// =============================================================================

/// Create high-frequency trading data (dense, small movements)
pub fn create_high_frequency_data(interval_ms: i64) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let base_time = 1659312000000i64;

    // Dense, high-frequency trading
    for i in 0..10000 {
        let micro_movement = (i as f64 * 0.1).sin() * 0.5; // Very small movements
        let price = base_price + micro_movement;
        let timestamp = base_time + (i as i64 * interval_ms);

        trades.push(create_test_trade(
            6000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

/// Create medium-frequency trading data
pub fn create_medium_frequency_data(interval_ms: i64) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let base_time = 1659312000000i64;

    for i in 0..5000 {
        let movement = (i as f64 * 0.05).sin() * 10.0;
        let price = base_price + movement;
        let timestamp = base_time + (i as i64 * interval_ms);

        trades.push(create_test_trade(
            7000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

/// Create low-frequency trading data
pub fn create_low_frequency_data(interval_ms: i64) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let base_time = 1659312000000i64;

    for i in 0..1000 {
        let movement = (i as f64 * 0.01).sin() * 50.0;
        let price = base_price + movement;
        let timestamp = base_time + (i as i64 * interval_ms);

        trades.push(create_test_trade(
            8000000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

/// Create mixed-frequency trading data (variable intervals)
pub fn create_mixed_frequency_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let base_time = 1659312000000i64;
    let mut current_time = base_time;

    // Variable intervals: sometimes fast, sometimes slow
    for i in 0..3000 {
        let movement = (i as f64 * 0.02).sin() * 25.0;
        let price = base_price + movement;

        // Variable interval based on market conditions
        let interval = if i % 10 < 3 {
            50 // Fast periods
        } else if i % 10 < 7 {
            200 // Medium periods
        } else {
            1000 // Slow periods
        };

        current_time += interval;
        trades.push(create_test_trade(
            9000000 + i as u64,
            price,
            current_time as u64,
        ));
    }

    trades
}

// =============================================================================
// Stress Test Data Generation
// =============================================================================

/// Create rapid threshold hit data (stress the algorithm)
pub fn create_rapid_threshold_hit_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let threshold = 0.0025; // 0.25%
    let base_time = 1659312000000i64;

    // Create rapid threshold hits to stress the algorithm
    for i in 0..1000 {
        let phase = (i / 10) % 4;
        let price = match phase {
            0 => base_price,                           // Base
            1 => base_price * (1.0 + threshold * 1.1), // Above threshold
            2 => base_price,                           // Back to base
            _ => base_price * (1.0 - threshold * 1.1), // Below threshold
        };

        trades.push(create_test_trade(
            10000000 + i as u64,
            price,
            (base_time + i as i64 * 10) as u64,
        ));
    }

    trades
}

/// Create precision limit data (test FixedPoint edge cases)
pub fn create_precision_limit_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_time = 1659312000000i64;

    // Test precision limits of FixedPoint (8 decimal places)
    let precision_prices = [
        23000.12345678,    // Max precision
        23000.00000001,    // Minimum increment
        99999999.99999999, // Large number with precision
        0.00000001,        // Smallest possible
    ];

    for (i, price) in precision_prices.iter().enumerate() {
        trades.push(create_test_trade(
            11000000 + i as u64,
            *price,
            (base_time + i as i64 * 1000) as u64,
        ));
    }

    trades
}

/// Create volume extreme data
pub fn create_volume_extreme_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;
    let base_time = 1659312000000i64;

    // Test extreme volume conditions
    for i in 0..100 {
        let price = base_price + (i as f64 * 0.1);
        // Note: We use volume=1.0 consistently as per our test pattern
        trades.push(create_test_trade(
            12000000 + i as u64,
            price,
            (base_time + i as i64 * 100) as u64,
        ));
    }

    trades
}

/// Create timestamp edge data
pub fn create_timestamp_edge_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_price = 23000.0;

    // Test timestamp edge cases
    let edge_timestamps: Vec<i64> = vec![
        1,                   // Near epoch start
        1659312000000,       // Normal timestamp
        9223372036854775807, // Near i64 max
    ];

    for (i, timestamp) in edge_timestamps.iter().enumerate() {
        let price = base_price + (i as f64 * 10.0);
        trades.push(create_test_trade(
            13000000 + i as u64,
            price,
            *timestamp as u64,
        ));
    }

    trades
}

/// Create floating point stress data
pub fn create_floating_point_stress_data() -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let base_time = 1659312000000i64;

    // Test floating point edge cases that could cause precision issues
    let stress_prices = [
        23000.1 + 0.1,        // Addition that might cause precision loss
        23000.0 / 3.0,        // Division creating repeating decimals
        23000.0 * 1.1,        // Multiplication
        (23000.0_f64).sqrt(), // Square root
    ];

    for (i, price) in stress_prices.iter().enumerate() {
        trades.push(create_test_trade(
            14000000 + i as u64,
            *price,
            (base_time + i as i64 * 100) as u64,
        ));
    }

    trades
}
