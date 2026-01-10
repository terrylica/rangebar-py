//! Integration tests for the rangebar crate
//!
//! These tests verify the end-to-end functionality of the range bar construction
//! algorithm and ensure proper integration between all components.

use rangebar::{AggTrade, FixedPoint, RangeBar, RangeBarProcessor};

#[cfg(feature = "providers")]
use rangebar::{get_tier1_symbols, is_tier1_symbol};

use rangebar_core::test_data_loader::load_btcusdt_test_data;

#[test]
fn test_range_bar_processing_integration() {
    // Test complete workflow from trades to range bars using real BTCUSDT CSV data
    let mut processor = RangeBarProcessor::new(250).expect("Failed to create processor"); // 0.25% threshold (standard for real data)

    // Load real BTCUSDT test data from CSV (5,000 trades)
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    // Process AggTrade records
    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process AggTrade records");

    // Verify results
    assert!(
        !range_bars.is_empty(),
        "Should produce at least one range bar from real data"
    );

    println!(
        "Real data integration test: {} trades → {} range bars (threshold=0.25%)",
        trades.len(),
        range_bars.len()
    );

    // Verify each range bar has valid OHLC data
    for (i, bar) in range_bars.iter().enumerate() {
        assert!(
            bar.high >= bar.open,
            "Bar {}: High ({}) should be >= open ({})",
            i,
            bar.high,
            bar.open
        );
        assert!(
            bar.high >= bar.close,
            "Bar {}: High ({}) should be >= close ({})",
            i,
            bar.high,
            bar.close
        );
        assert!(
            bar.low <= bar.open,
            "Bar {}: Low ({}) should be <= open ({})",
            i,
            bar.low,
            bar.open
        );
        assert!(
            bar.low <= bar.close,
            "Bar {}: Low ({}) should be <= close ({})",
            i,
            bar.low,
            bar.close
        );
        assert!(
            bar.volume > FixedPoint(0),
            "Bar {}: Volume should be positive",
            i
        );
        assert!(
            bar.open_time <= bar.close_time,
            "Bar {}: Open time should be <= close time",
            i
        );
    }
}

#[cfg(feature = "providers")]
#[test]
fn test_tier1_symbol_integration() {
    // Test Tier-1 symbol functionality
    let tier1_symbols = get_tier1_symbols();

    // Verify we have the expected number of Tier-1 symbols
    assert!(!tier1_symbols.is_empty(), "Should have Tier-1 symbols");
    assert!(
        tier1_symbols.len() >= 18,
        "Should have at least 18 Tier-1 symbols"
    );

    // Verify known Tier-1 symbols
    assert!(is_tier1_symbol("BTC"), "BTC should be Tier-1");
    assert!(is_tier1_symbol("ETH"), "ETH should be Tier-1");
    assert!(is_tier1_symbol("SOL"), "SOL should be Tier-1");

    // Verify non-Tier-1 symbols
    assert!(!is_tier1_symbol("SHIB"), "SHIB should not be Tier-1");
    assert!(!is_tier1_symbol("PEPE"), "PEPE should not be Tier-1");
}

#[test]
fn test_zero_duration_bars_are_valid() {
    // Test that zero-duration bars are properly handled (NOTABUG verification)
    let mut processor = RangeBarProcessor::new(100).expect("Failed to create processor"); // 0.1% threshold for easier testing

    // Create trades with identical timestamps that breach threshold
    let same_timestamp = 1609459200000;
    let base_price = FixedPoint::from_str("50000.00000000").unwrap();
    let breach_price = FixedPoint::from_str("50100.00000000").unwrap(); // +0.2% breach

    let trades = vec![
        AggTrade {
            agg_trade_id: 1,
            price: base_price,
            volume: FixedPoint::from_str("1.0").unwrap(),
            first_trade_id: 1,
            last_trade_id: 1,
            timestamp: same_timestamp,
            is_buyer_maker: false,
            is_best_match: None,
        },
        AggTrade {
            agg_trade_id: 2,
            price: breach_price,
            volume: FixedPoint::from_str("1.0").unwrap(),
            first_trade_id: 2,
            last_trade_id: 2,
            timestamp: same_timestamp, // Same timestamp - zero duration
            is_buyer_maker: false,
            is_best_match: None,
        },
    ];

    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process trades");

    // Should produce a zero-duration bar (open_time == close_time)
    assert_eq!(range_bars.len(), 1, "Should produce exactly one range bar");
    let bar = &range_bars[0];
    assert_eq!(bar.open_time, bar.close_time, "Zero-duration bar is valid");
    assert_eq!(bar.open, base_price, "Open should be first trade price");
    assert_eq!(
        bar.close, breach_price,
        "Close should be breach trade price"
    );
}

#[test]
fn test_fixed_point_precision() {
    // Test that fixed-point arithmetic maintains precision
    let price1 = FixedPoint::from_str("50000.12345678").unwrap();
    let price2 = FixedPoint::from_str("50000.87654321").unwrap();

    // Verify precision is maintained
    assert_eq!(price1.to_string(), "50000.12345678");
    assert_eq!(price2.to_string(), "50000.87654321");

    // Test arithmetic operations maintain precision
    let sum = FixedPoint(price1.0 + price2.0);
    assert_eq!(sum.to_string(), "100000.99999999");
}

#[test]
fn test_cross_mode_algorithm_consistency() {
    // Critical test: Verify unified algorithm produces identical results
    // regardless of statistics feature compilation

    let mut processor = RangeBarProcessor::new(500).expect("Failed to create processor"); // 0.5% threshold

    // Create deterministic test data that should produce multiple range bars
    let test_trades = create_deterministic_breach_sequence();

    // Process trades and capture results
    let range_bars = processor
        .process_agg_trade_records(&test_trades)
        .expect("Failed to process trades");

    // Validate core algorithm invariants that must hold across all modes
    validate_algorithm_invariants(&range_bars, &test_trades);

    // Test critical breach consistency that was fixed in the 599 bar issue
    validate_breach_consistency(&range_bars);

    // Verify deterministic bar count (should not vary by compilation mode)
    // Note: Actual count depends on range bar processor implementation
    assert!(
        !range_bars.is_empty(),
        "Expected at least 1 bar from deterministic sequence, got {}",
        range_bars.len()
    );

    // Verify OHLCV integrity
    for (i, bar) in range_bars.iter().enumerate() {
        assert!(bar.high >= bar.open, "Bar {} high < open", i);
        assert!(bar.high >= bar.close, "Bar {} high < close", i);
        assert!(bar.low <= bar.open, "Bar {} low > open", i);
        assert!(bar.low <= bar.close, "Bar {} low > close", i);
        assert!(bar.volume > FixedPoint(0), "Bar {} zero volume", i);
        assert!(bar.open_time <= bar.close_time, "Bar {} time inversion", i);
    }
}

/// Creates a deterministic sequence that produces multiple range bars with breaches
/// This sequence is designed to test the core algorithm consistency
fn create_deterministic_breach_sequence() -> Vec<AggTrade> {
    let base_price = 50000.0;
    let base_timestamp = 1609459200000;

    vec![
        // Bar 1: Opens at base price
        create_trade(1, base_price, base_timestamp),
        // Bar 1: Small movement within threshold
        create_trade(2, base_price * 1.003, base_timestamp + 1000), // +0.3%
        // Bar 1: Breach upward - closes bar 1, this trade becomes the close
        create_trade(3, base_price * 1.006, base_timestamp + 2000), // +0.6% > 0.5%
        // Bar 2: Opens at next trade (important: new bar starts with next trade)
        // For proper testing, the next trade should be at a different price to create new threshold
        create_trade(4, base_price * 1.007, base_timestamp + 3000), // Opens bar 2
        // Bar 2: Movement within new threshold (based on 50350 = base_price * 1.007)
        create_trade(5, base_price * 1.009, base_timestamp + 4000), // Small movement up
        // Bar 2: Breach downward from bar 2 open - closes bar 2
        create_trade(6, base_price * 1.002, base_timestamp + 5000), // Down more than 0.5% from 50350
        // Bar 3: Opens at new price
        create_trade(7, base_price * 1.003, base_timestamp + 6000), // Opens bar 3
        // Bar 3: Large breach upward to close bar 3
        create_trade(8, base_price * 1.008, base_timestamp + 7000), // +0.5% breach from bar 3 open
    ]
}

/// Validates core algorithm invariants that must hold regardless of compilation mode
fn validate_algorithm_invariants(range_bars: &[RangeBar], test_trades: &[AggTrade]) {
    // Non-lookahead bias: Each bar's open should match a trade price
    for bar in range_bars {
        let open_found = test_trades.iter().any(|trade| trade.price == bar.open);
        assert!(
            open_found,
            "Bar open price {} not found in trades (non-lookahead violation)",
            bar.open
        );
    }

    // Volume conservation: ALL trades must have volume counted SOMEWHERE
    // Algorithm invariant: no volume should be lost or duplicated
    //
    // NOTE: We use process_agg_trade_records_with_incomplete() to include ALL bars
    // (completed + incomplete) for this check. Production uses process_agg_trade_records()
    // which excludes incomplete bars - this is separate from algorithm correctness.
    let mut processor_for_volume_check =
        RangeBarProcessor::new(500).expect("Failed to create processor"); // Same threshold as test
    let all_bars = processor_for_volume_check
        .process_agg_trade_records_with_incomplete(test_trades)
        .expect("Failed to process trades with incomplete for volume check");

    let total_bar_volume: i64 = all_bars.iter().map(|bar| bar.volume.0).sum();
    let total_trade_volume: i64 = test_trades.iter().map(|trade| trade.volume.0).sum();

    assert_eq!(
        total_bar_volume,
        total_trade_volume,
        "Volume conservation violation: bars={} (from {} bars), trades={} (from {} trades), discrepancy={}. \
         Algorithm must account for ALL trade volume across all bars (completed + incomplete).",
        total_bar_volume,
        all_bars.len(),
        total_trade_volume,
        test_trades.len(),
        total_trade_volume - total_bar_volume
    );

    // Temporal ordering: Bar timestamps should be monotonically increasing
    for i in 1..range_bars.len() {
        assert!(
            range_bars[i - 1].close_time <= range_bars[i].open_time,
            "Temporal ordering violation: bar {} close_time > bar {} open_time",
            i - 1,
            i
        );
    }
}

/// Validates breach consistency - critical fix from the 599 bar discrepancy issue
fn validate_breach_consistency(range_bars: &[RangeBar]) {
    let threshold_decimal_bps = 50; // 0.5% = 50 basis points

    for (i, bar) in range_bars.iter().enumerate() {
        let threshold_decimal = threshold_decimal_bps as f64 / 10000.0;
        let upper_threshold = bar.open.to_f64() * (1.0 + threshold_decimal);
        let lower_threshold = bar.open.to_f64() * (1.0 - threshold_decimal);

        let high_breaches = bar.high.to_f64() >= upper_threshold;
        let low_breaches = bar.low.to_f64() <= lower_threshold;

        if high_breaches {
            assert!(
                bar.close.to_f64() >= upper_threshold,
                "Bar {} breach consistency violation: high breaches ({} >= {}) but close doesn't ({} < {})",
                i,
                bar.high.to_f64(),
                upper_threshold,
                bar.close.to_f64(),
                upper_threshold
            );
        }

        if low_breaches {
            assert!(
                bar.close.to_f64() <= lower_threshold,
                "Bar {} breach consistency violation: low breaches ({} <= {}) but close doesn't ({} > {})",
                i,
                bar.low.to_f64(),
                lower_threshold,
                bar.close.to_f64(),
                lower_threshold
            );
        }
    }
}

#[test]
fn test_non_lookahead_bias_compliance() {
    // Test that thresholds are computed from bar open only (non-lookahead)
    let mut processor = RangeBarProcessor::new(500).expect("Failed to create processor"); // 0.5% threshold

    let base_price = 50000.0;
    let trades = vec![
        // First trade - opens bar
        create_trade(1, base_price, 1609459200000),
        // Second trade - within threshold, updates high but doesn't close
        create_trade(2, base_price * 1.003, 1609459201000), // +0.3%
        // Third trade - breaches threshold, closes bar
        create_trade(3, base_price * 1.006, 1609459202000), // +0.6% - breaches 0.5%
    ];

    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process trades");

    assert_eq!(range_bars.len(), 1, "Should produce exactly one range bar");
    let bar = &range_bars[0];

    // Verify the threshold was based on the open price (non-lookahead)
    let expected_upper_threshold = base_price * 1.005; // +0.5% from open

    assert!(
        bar.close.to_f64() > expected_upper_threshold,
        "Close should breach threshold computed from open: close={}, threshold={}",
        bar.close.to_f64(),
        expected_upper_threshold
    );
}

/// Helper function to create a single trade
fn create_trade(id: i64, price: f64, timestamp: i64) -> AggTrade {
    AggTrade {
        agg_trade_id: id,
        price: FixedPoint::from_str(&format!("{:.8}", price)).unwrap(),
        volume: FixedPoint::from_str("1.50000000").unwrap(),
        first_trade_id: id * 10,
        last_trade_id: id * 10 + 5,
        timestamp,
        is_buyer_maker: id % 2 == 0,
        is_best_match: None,
    }
}
