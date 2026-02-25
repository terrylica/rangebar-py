//! Tests for types.rs - RangeBar, AggTrade, microstructure features
//!
//! Extracted from crates/rangebar-core/src/types.rs (Phase 1a refactoring)

use rangebar_core::fixed_point::SCALE;
use rangebar_core::types::RangeBar;

// Re-use the test utility for creating AggTrades
use rangebar_core::test_utils;

#[test]
fn test_agg_trade_creation() {
    let trade = test_utils::create_test_agg_trade_with_range(
        12345,
        "50000.12345678",
        "1.5",
        1640995200000,
        100,
        102,
        false, // Buy pressure (taker buying from maker)
    );

    assert_eq!(trade.individual_trade_count(), 3); // 102 - 100 + 1
    assert!(trade.turnover() > 0);
}

#[test]
fn test_range_bar_creation() {
    let trade = test_utils::create_test_agg_trade_with_range(
        12345,
        "50000.0",
        "1.0",
        1640995200000,
        100,
        100,
        true, // Sell pressure (taker selling to maker)
    );

    let bar = RangeBar::new(&trade);
    assert_eq!(bar.open, trade.price);
    assert_eq!(bar.high, trade.price);
    assert_eq!(bar.low, trade.price);
    assert_eq!(bar.close, trade.price);
}

#[test]
fn test_range_bar_update() {
    let trade1 = test_utils::create_test_agg_trade_with_range(
        12345,
        "50000.0",
        "1.0",
        1640995200000,
        100,
        100,
        false, // Buy pressure
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        12346,
        "50100.0",
        "2.0",
        1640995201000,
        101,
        101,
        true, // Sell pressure
    );

    bar.update_with_trade(&trade2);

    assert_eq!(bar.open.to_string(), "50000.00000000");
    assert_eq!(bar.high.to_string(), "50100.00000000");
    assert_eq!(bar.low.to_string(), "50000.00000000");
    assert_eq!(bar.close.to_string(), "50100.00000000");
    // Issue #88: volume is i128 raw (3.0 * SCALE = 300_000_000)
    assert_eq!(bar.volume, 300_000_000i128);
    assert_eq!(bar.individual_trade_count, 2);
}

#[test]
fn test_microstructure_segregation() {
    // Create buy trade (is_buyer_maker = false)
    let buy_trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.5",
        1640995200000,
        1,
        1,
        false, // Buy pressure (taker buying from maker)
    );

    let mut bar = RangeBar::new(&buy_trade);

    // Create sell trade (is_buyer_maker = true)
    let sell_trade = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "2.5",
        1640995201000,
        2,
        3,    // Multiple trades aggregated
        true, // Sell pressure (taker selling to maker)
    );

    bar.update_with_trade(&sell_trade);

    // Verify order flow segregation
    // Issue #88: volume fields are i128 raw (value * SCALE)
    assert_eq!(bar.buy_volume, 150_000_000i128); // 1.5 * SCALE
    assert_eq!(bar.sell_volume, 250_000_000i128); // 2.5 * SCALE
    assert_eq!(bar.buy_trade_count, 1); // First trade count
    assert_eq!(bar.sell_trade_count, 2); // Second trade count (3 - 2 + 1 = 2)

    // Verify totals
    assert_eq!(bar.volume, 400_000_000i128); // (1.5 + 2.5) * SCALE
    assert_eq!(bar.individual_trade_count, 3); // 1 + 2

    // Verify VWAP calculation
    // VWAP = (50000 * 1.5 + 50050 * 2.5) / 4.0 = (75000 + 125125) / 4.0 = 50031.25
    assert_eq!(bar.vwap.to_string(), "50031.25000000");

    println!("Microstructure segregation test passed:");
    println!(
        "   Buy volume: {}, Sell volume: {}",
        bar.buy_volume, bar.sell_volume
    );
    println!(
        "   Buy trades: {}, Sell trades: {}",
        bar.buy_trade_count, bar.sell_trade_count
    );
    println!("   VWAP: {}", bar.vwap);
}

// =========================================================================
// Microstructure Features Tests (Issue #25)
// =========================================================================

#[test]
fn test_ofi_balanced() {
    // Create a bar with equal buy and sell volumes
    let buy_trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000, // microseconds
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&buy_trade);

    let sell_trade = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995201000000, // 1 second later
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&sell_trade);
    bar.compute_microstructure_features();

    // OFI should be 0 when buy_volume == sell_volume
    assert!(
        bar.ofi.abs() < f64::EPSILON,
        "OFI should be 0 for balanced volumes, got {}",
        bar.ofi
    );
}

#[test]
fn test_ofi_all_buys() {
    // Create a bar with only buy volume
    let buy_trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&buy_trade1);

    let buy_trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995201000000,
        2,
        2,
        false, // Buy
    );

    bar.update_with_trade(&buy_trade2);
    bar.compute_microstructure_features();

    // OFI should be 1.0 when all buys
    assert!(
        (bar.ofi - 1.0).abs() < f64::EPSILON,
        "OFI should be 1.0 for all buys, got {}",
        bar.ofi
    );
}

#[test]
fn test_ofi_all_sells() {
    // Create a bar with only sell volume
    let sell_trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        true, // Sell
    );

    let mut bar = RangeBar::new(&sell_trade1);

    let sell_trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995201000000,
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&sell_trade2);
    bar.compute_microstructure_features();

    // OFI should be -1.0 when all sells
    assert!(
        (bar.ofi - (-1.0)).abs() < f64::EPSILON,
        "OFI should be -1.0 for all sells, got {}",
        bar.ofi
    );
}

#[test]
fn test_turnover_imbalance_bounded() {
    // Create a bar with mixed buy/sell turnover
    let buy_trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&buy_trade);

    let sell_trade = test_utils::create_test_agg_trade_with_range(
        2,
        "50100.0",
        "2.0",
        1640995201000000,
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&sell_trade);
    bar.compute_microstructure_features();

    // Turnover imbalance should be in [-1, 1]
    assert!(
        bar.turnover_imbalance >= -1.0 && bar.turnover_imbalance <= 1.0,
        "Turnover imbalance should be in [-1, 1], got {}",
        bar.turnover_imbalance
    );
}

#[test]
fn test_kyle_lambda_div_zero() {
    // Create a bar with equal buy/sell -> imbalance = 0
    let buy_trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&buy_trade);

    let sell_trade = test_utils::create_test_agg_trade_with_range(
        2,
        "50100.0",
        "1.0",
        1640995201000000,
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&sell_trade);
    bar.compute_microstructure_features();

    // Kyle lambda should be 0 when imbalance is 0
    assert!(
        bar.kyle_lambda_proxy.abs() < f64::EPSILON,
        "Kyle lambda should be 0 when imbalance is 0, got {}",
        bar.kyle_lambda_proxy
    );
}

#[test]
fn test_duration_positive() {
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000, // microseconds
        1,
        1,
        false,
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995205000000, // 5 seconds later
        2,
        2,
        true,
    );

    bar.update_with_trade(&trade2);
    bar.compute_microstructure_features();

    // Duration should be 5 seconds = 5,000,000 microseconds
    assert_eq!(
        bar.duration_us, 5_000_000,
        "Duration should be 5,000,000 microseconds, got {}",
        bar.duration_us
    );
}

#[test]
fn test_trade_intensity() {
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000, // microseconds
        1,
        1,
        false,
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995202000000, // 2 seconds later
        2,
        2,
        true,
    );

    bar.update_with_trade(&trade2);
    bar.compute_microstructure_features();

    // Trade intensity = 2 trades / 2 seconds = 1 trade/sec
    assert!(
        (bar.trade_intensity - 1.0).abs() < 0.01,
        "Trade intensity should be ~1 trade/sec, got {}",
        bar.trade_intensity
    );
}

#[test]
fn test_aggression_ratio_capped() {
    // Create a bar with only buy trades (no sells)
    let buy_trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&buy_trade1);

    let buy_trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "1.0",
        1640995201000000,
        2,
        2,
        false, // Buy
    );

    bar.update_with_trade(&buy_trade2);
    bar.compute_microstructure_features();

    // Aggression ratio should be capped at 100 when no sells
    assert_eq!(
        bar.aggression_ratio, 100.0,
        "Aggression ratio should be 100.0 when no sells, got {}",
        bar.aggression_ratio
    );
}

#[test]
fn test_aggregation_density() {
    // Create a bar with multiple individual trades per agg record
    let trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "5.0",
        1640995200000000,
        1,
        10, // 10 individual trades in this agg record
        false,
    );

    let mut bar = RangeBar::new(&trade);
    bar.compute_microstructure_features();

    // individual_trade_count / agg_record_count = 10 / 1 = 10.0
    assert!(
        (bar.aggregation_density_f64 - 10.0).abs() < 0.01,
        "Aggregation efficiency should be 10.0, got {}",
        bar.aggregation_density_f64
    );
}

#[test]
fn test_vwap_close_deviation_zero_range() {
    // Create a bar with high == low (zero range)
    let trade = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "1.0",
        1640995200000000,
        1,
        1,
        false,
    );

    let mut bar = RangeBar::new(&trade);
    bar.compute_microstructure_features();

    // VWAP close deviation should be 0 when high == low
    assert_eq!(
        bar.vwap_close_deviation, 0.0,
        "VWAP close deviation should be 0 when high == low, got {}",
        bar.vwap_close_deviation
    );
}

#[test]
fn test_volume_per_trade() {
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "50000.0",
        "3.0",
        1640995200000000,
        1,
        1,
        false,
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "50050.0",
        "7.0",
        1640995201000000,
        2,
        2,
        true,
    );

    bar.update_with_trade(&trade2);
    bar.compute_microstructure_features();

    // volume_per_trade = 10 / 2 = 5.0
    assert!(
        (bar.volume_per_trade - 5.0).abs() < 0.01,
        "Volume per trade should be 5.0, got {}",
        bar.volume_per_trade
    );
}

// =========================================================================
// Issue #88: i64->i128 Volume Overflow Tests
//
// SHIB-like tokens with volumes of 50 billion tokens per trade produce
// FixedPoint raw values of 5_000_000_000_000_000_000 (5e18). Two such
// trades summed (10e18) exceed i64::MAX (9.22e18), causing silent
// wraparound to negative values with the old i64 accumulator.
// i128 handles this correctly.
// =========================================================================

#[test]
fn test_volume_no_overflow_with_large_trades() {
    // Issue #88: 50 billion tokens per trade.
    // FixedPoint raw = 50_000_000_000 * SCALE(100_000_000) = 5_000_000_000_000_000_000 (5e18)
    // Two trades summed = 10e18 > i64::MAX (9.22e18) — would overflow i64.
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "0.00003000",    // SHIB-like price
        "50000000000.0", // 50 billion tokens
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "0.00003100",
        "50000000000.0", // Another 50 billion tokens
        1640995201000000,
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&trade2);

    // Verify volume is positive (old i64 would wrap to negative)
    assert!(
        bar.volume > 0,
        "Volume should be positive with i128, got {} (i64 would have overflowed)",
        bar.volume
    );

    // Verify exact value: 2 * 50_000_000_000 * 100_000_000 = 10_000_000_000_000_000_000
    let expected_volume: i128 = 10_000_000_000_000_000_000;
    assert_eq!(
        bar.volume, expected_volume,
        "Volume should be exactly 10e18, got {}",
        bar.volume
    );

    // Prove this would have overflowed i64
    assert!(
        expected_volume > i64::MAX as i128,
        "Expected volume {} should exceed i64::MAX {} to prove overflow prevention",
        expected_volume,
        i64::MAX
    );
}

#[test]
fn test_vwap_correct_with_large_volume() {
    // Issue #88: VWAP must be computed correctly even with i128 volumes.
    // Two trades at different prices with large SHIB-like volumes.
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "0.00003000",
        "50000000000.0", // 50B tokens
        1640995200000000,
        1,
        1,
        false, // Buy
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "0.00004000",
        "50000000000.0", // 50B tokens
        1640995201000000,
        2,
        2,
        true, // Sell
    );

    bar.update_with_trade(&trade2);

    // VWAP = (0.00003 * 50B + 0.00004 * 50B) / 100B = 0.000035
    let vwap_f64 = bar.vwap.to_f64();
    assert!(vwap_f64 > 0.0, "VWAP should be positive, got {}", vwap_f64);
    assert!(!vwap_f64.is_nan(), "VWAP should not be NaN");
    assert!(
        vwap_f64 >= 0.00003 && vwap_f64 <= 0.00004,
        "VWAP should be between the two prices (0.00003..0.00004), got {}",
        vwap_f64
    );
    // Check it's close to the expected midpoint 0.000035
    assert!(
        (vwap_f64 - 0.000035).abs() < 0.000001,
        "VWAP should be ~0.000035 (equal-weight midpoint), got {}",
        vwap_f64
    );
}

#[test]
fn test_microstructure_features_with_i128_volume() {
    // Issue #88: Microstructure features must compute correctly from i128 volumes.
    // SHIB-like token with large volumes.
    let trade1 = test_utils::create_test_agg_trade_with_range(
        1,
        "0.00003000",
        "50000000000.0", // 50B tokens — buy
        1640995200000000,
        1,
        1,
        false, // Buy pressure
    );

    let mut bar = RangeBar::new(&trade1);

    let trade2 = test_utils::create_test_agg_trade_with_range(
        2,
        "0.00003100",
        "30000000000.0", // 30B tokens — sell
        1640995201000000,
        2,
        2,
        true, // Sell pressure
    );

    bar.update_with_trade(&trade2);
    bar.compute_microstructure_features();

    // OFI = (buy_vol - sell_vol) / total = (50B - 30B) / 80B = 0.25
    assert!(
        bar.ofi >= -1.0 && bar.ofi <= 1.0,
        "OFI should be in [-1, 1], got {}",
        bar.ofi
    );
    assert!(
        (bar.ofi - 0.25).abs() < 0.01,
        "OFI should be ~0.25 (50B buy vs 30B sell), got {}",
        bar.ofi
    );

    // Price impact must be non-negative
    assert!(
        bar.price_impact >= 0.0,
        "Price impact should be >= 0, got {}",
        bar.price_impact
    );

    // Volume per trade must be positive
    assert!(
        bar.volume_per_trade > 0.0,
        "Volume per trade should be > 0, got {}",
        bar.volume_per_trade
    );

    // Turnover imbalance should be in [-1, 1]
    assert!(
        bar.turnover_imbalance >= -1.0 && bar.turnover_imbalance <= 1.0,
        "Turnover imbalance should be in [-1, 1], got {}",
        bar.turnover_imbalance
    );
}

#[test]
fn test_volume_conservation_with_i128() {
    // Issue #88: buy_volume + sell_volume must equal total volume (no precision loss).
    // Use 50B + 50B = 100B tokens so total (10e18) exceeds i64::MAX (9.22e18).
    let buy_trade = test_utils::create_test_agg_trade_with_range(
        1,
        "0.00003000",
        "50000000000.0", // 50B tokens — buy
        1640995200000000,
        1,
        1,
        false, // Buy pressure
    );

    let mut bar = RangeBar::new(&buy_trade);

    let sell_trade = test_utils::create_test_agg_trade_with_range(
        2,
        "0.00003100",
        "50000000000.0", // 50B tokens — sell (same size for symmetry)
        1640995201000000,
        2,
        2,
        true, // Sell pressure
    );

    bar.update_with_trade(&sell_trade);

    // Volume conservation: buy_volume + sell_volume == total volume (exact, no rounding)
    assert_eq!(
        bar.buy_volume + bar.sell_volume,
        bar.volume,
        "buy_volume ({}) + sell_volume ({}) should equal volume ({})",
        bar.buy_volume,
        bar.sell_volume,
        bar.volume
    );

    // Verify individual components
    let expected_each: i128 = 50_000_000_000 * SCALE as i128;
    assert_eq!(
        bar.buy_volume, expected_each,
        "Buy volume should be 50B * SCALE = {}, got {}",
        expected_each, bar.buy_volume
    );
    assert_eq!(
        bar.sell_volume, expected_each,
        "Sell volume should be 50B * SCALE = {}, got {}",
        expected_each, bar.sell_volume
    );

    // Prove the total would have overflowed i64
    let total = bar.buy_volume + bar.sell_volume;
    assert!(
        total > i64::MAX as i128,
        "Total volume {} should exceed i64::MAX {} to prove overflow prevention",
        total,
        i64::MAX
    );
}

// === Issue #96 Task #81: update_with_trade() invariant tests ===

#[test]
fn test_high_only_extends_upward() {
    // High must only increase, never decrease — even when later trades are lower
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "1.0", 1000, 10, 10, false);
    let mut bar = RangeBar::new(&t1);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "50200.0", "1.0", 2000, 20, 20, true);
    bar.update_with_trade(&t2);
    assert_eq!(bar.high.to_string(), "50200.00000000");

    // Lower trade should NOT reduce high
    let t3 = test_utils::create_test_agg_trade_with_range(3, "49800.0", "1.0", 3000, 30, 30, false);
    bar.update_with_trade(&t3);
    assert_eq!(bar.high.to_string(), "50200.00000000", "High must not decrease after lower trade");
}

#[test]
fn test_low_only_extends_downward() {
    // Low must only decrease, never increase — even when later trades are higher
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "1.0", 1000, 10, 10, false);
    let mut bar = RangeBar::new(&t1);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "49800.0", "1.0", 2000, 20, 20, true);
    bar.update_with_trade(&t2);
    assert_eq!(bar.low.to_string(), "49800.00000000");

    // Higher trade should NOT raise low
    let t3 = test_utils::create_test_agg_trade_with_range(3, "50200.0", "1.0", 3000, 30, 30, false);
    bar.update_with_trade(&t3);
    assert_eq!(bar.low.to_string(), "49800.00000000", "Low must not increase after higher trade");
}

#[test]
fn test_close_tracks_latest_trade() {
    // Close and close_time must always reflect the most recent trade
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "1.0", 1000, 10, 10, false);
    let mut bar = RangeBar::new(&t1);
    assert_eq!(bar.close.to_string(), "50000.00000000");
    assert_eq!(bar.close_time, 1000);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "50100.0", "1.0", 2000, 20, 20, true);
    bar.update_with_trade(&t2);
    assert_eq!(bar.close.to_string(), "50100.00000000");
    assert_eq!(bar.close_time, 2000);

    let t3 = test_utils::create_test_agg_trade_with_range(3, "49900.0", "1.0", 3000, 30, 30, false);
    bar.update_with_trade(&t3);
    assert_eq!(bar.close.to_string(), "49900.00000000");
    assert_eq!(bar.close_time, 3000);
}

#[test]
fn test_open_never_changes() {
    // Open price and open_time are set at construction and must never change
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "1.0", 1000, 10, 10, false);
    let mut bar = RangeBar::new(&t1);
    let original_open = bar.open;
    let original_open_time = bar.open_time;

    for i in 2..=5 {
        let t = test_utils::create_test_agg_trade_with_range(
            i, &format!("{:.1}", 49000.0 + (i as f64) * 500.0), "1.0",
            1000 + i * 1000, i * 10, i * 10, i % 2 == 0,
        );
        bar.update_with_trade(&t);
        assert_eq!(bar.open, original_open, "Open must never change after trade {i}");
        assert_eq!(bar.open_time, original_open_time, "Open time must never change after trade {i}");
    }
}

#[test]
fn test_volume_conservation_buy_sell() {
    // Invariant: buy_volume + sell_volume == total volume (exact, not approximate)
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "2.5", 1000, 10, 12, false);
    let mut bar = RangeBar::new(&t1);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "50100.0", "3.7", 2000, 20, 22, true);
    bar.update_with_trade(&t2);

    let t3 = test_utils::create_test_agg_trade_with_range(3, "49900.0", "1.3", 3000, 30, 30, false);
    bar.update_with_trade(&t3);

    assert_eq!(
        bar.buy_volume + bar.sell_volume, bar.volume,
        "buy_vol ({}) + sell_vol ({}) != total ({})", bar.buy_volume, bar.sell_volume, bar.volume
    );
}

#[test]
fn test_trade_count_consistency() {
    // Invariant: buy_trade_count + sell_trade_count == individual_trade_count
    let t1 = test_utils::create_test_agg_trade_with_range(1, "50000.0", "1.0", 1000, 10, 14, false); // 5 individual
    let mut bar = RangeBar::new(&t1);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "50100.0", "1.0", 2000, 20, 22, true); // 3 individual
    bar.update_with_trade(&t2);

    assert_eq!(bar.individual_trade_count, 8, "5 + 3 = 8 individual trades");
    assert_eq!(bar.buy_trade_count + bar.sell_trade_count, bar.individual_trade_count,
        "buy_count ({}) + sell_count ({}) != individual ({})",
        bar.buy_trade_count, bar.sell_trade_count, bar.individual_trade_count);
    assert_eq!(bar.agg_record_count, 2);
}

#[test]
fn test_vwap_between_min_max_price() {
    // VWAP must always be between the lowest and highest trade prices
    let t1 = test_utils::create_test_agg_trade_with_range(1, "49000.0", "1.0", 1000, 10, 10, false);
    let mut bar = RangeBar::new(&t1);

    let t2 = test_utils::create_test_agg_trade_with_range(2, "51000.0", "1.0", 2000, 20, 20, true);
    bar.update_with_trade(&t2);

    let t3 = test_utils::create_test_agg_trade_with_range(3, "50000.0", "1.0", 3000, 30, 30, false);
    bar.update_with_trade(&t3);

    assert!(bar.vwap >= bar.low, "VWAP ({}) must be >= low ({})", bar.vwap, bar.low);
    assert!(bar.vwap <= bar.high, "VWAP ({}) must be <= high ({})", bar.vwap, bar.high);
}

#[test]
fn test_agg_trade_id_range_tracking() {
    // first_agg_trade_id stays at opening trade, last_agg_trade_id tracks latest
    let t1 = test_utils::create_test_agg_trade_with_range(100, "50000.0", "1.0", 1000, 1000, 1002, false);
    let mut bar = RangeBar::new(&t1);
    assert_eq!(bar.first_agg_trade_id, 100);
    assert_eq!(bar.last_agg_trade_id, 100);

    let t2 = test_utils::create_test_agg_trade_with_range(105, "50100.0", "1.0", 2000, 1050, 1055, true);
    bar.update_with_trade(&t2);
    assert_eq!(bar.first_agg_trade_id, 100, "first_agg_trade_id must not change");
    assert_eq!(bar.last_agg_trade_id, 105);
    assert_eq!(bar.last_trade_id, 1055, "last_trade_id tracks individual trade ID");
}

// === Memory efficiency tests (R15) ===

#[test]
fn test_fixedpoint_default() {
    use rangebar_core::fixed_point::FixedPoint;
    let fp = FixedPoint::default();
    assert_eq!(fp, FixedPoint(0), "FixedPoint default should be 0");
}

#[test]
fn test_rangebar_default() {
    let bar = RangeBar::default();

    // Numeric fields default to 0
    assert_eq!(bar.open_time, 0);
    assert_eq!(bar.close_time, 0);
    assert_eq!(bar.volume, 0);
    assert_eq!(bar.turnover, 0);
    assert_eq!(bar.individual_trade_count, 0);
    assert_eq!(bar.duration_us, 0);
    assert_eq!(bar.ofi, 0.0);

    // FixedPoint fields default to FixedPoint(0)
    use rangebar_core::fixed_point::FixedPoint;
    assert_eq!(bar.open, FixedPoint(0));
    assert_eq!(bar.high, FixedPoint(0));
    assert_eq!(bar.low, FixedPoint(0));
    assert_eq!(bar.close, FixedPoint(0));
    assert_eq!(bar.vwap, FixedPoint(0));

    // All Option<T> fields default to None
    assert!(bar.lookback_ofi.is_none());
    assert!(bar.lookback_trade_count.is_none());
    assert!(bar.lookback_hurst.is_none());
    assert!(bar.lookback_permutation_entropy.is_none());
    assert!(bar.intra_trade_count.is_none());
    assert!(bar.intra_ofi.is_none());
    assert!(bar.intra_bull_epoch_density.is_none());
    assert!(bar.intra_hurst.is_none());
}
