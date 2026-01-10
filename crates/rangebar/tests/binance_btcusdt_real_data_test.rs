//! Binance BTCUSDT Real Data Integration Tests
//!
//! Validates range bar construction using real BTCUSDT market data from CSV.
//! Tests multiple threshold scenarios and validates data integrity.

use rangebar::{FixedPoint, RangeBar, RangeBarProcessor};
use rangebar_core::test_data_loader::load_btcusdt_test_data;

#[test]
fn test_btcusdt_data_integrity() {
    // Validate real BTCUSDT CSV data integrity
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    assert_eq!(trades.len(), 5000, "Expected 5,000 BTCUSDT trades");

    // Validate temporal ordering
    for i in 1..trades.len() {
        assert!(
            trades[i].timestamp >= trades[i - 1].timestamp,
            "Temporal ordering violation at trade {}: {} < {}",
            i,
            trades[i].timestamp,
            trades[i - 1].timestamp
        );
    }

    // Validate price and volume are positive
    for (i, trade) in trades.iter().enumerate() {
        assert!(
            trade.price > FixedPoint(0),
            "Trade {}: Invalid price {}",
            i,
            trade.price
        );
        assert!(
            trade.volume > FixedPoint(0),
            "Trade {}: Invalid volume {}",
            i,
            trade.volume
        );
    }

    println!(
        "✅ BTCUSDT data integrity validated: {} trades",
        trades.len()
    );
}

#[test]
fn test_btcusdt_standard_threshold() {
    // Test standard 25 bps (0.25%) threshold for real market data
    let mut processor =
        RangeBarProcessor::new(25).expect("Failed to create processor with valid threshold");
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process BTCUSDT trades");

    assert!(
        !range_bars.is_empty(),
        "Should produce at least one range bar"
    );

    println!(
        "BTCUSDT (25 bps): {} trades → {} range bars",
        trades.len(),
        range_bars.len()
    );

    validate_ohlcv_integrity(&range_bars);
    validate_temporal_ordering(&range_bars);
}

#[test]
fn test_btcusdt_medium_threshold() {
    // Test medium 50 bps (0.5%) threshold
    let mut processor =
        RangeBarProcessor::new(50).expect("Failed to create processor with valid threshold");
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process BTCUSDT trades");

    assert!(
        !range_bars.is_empty(),
        "Should produce at least one range bar"
    );

    println!(
        "BTCUSDT (50 bps): {} trades → {} range bars",
        trades.len(),
        range_bars.len()
    );

    validate_ohlcv_integrity(&range_bars);
    validate_temporal_ordering(&range_bars);
}

#[test]
fn test_btcusdt_wide_threshold() {
    // Test wide 100 bps (1.0%) threshold
    let mut processor =
        RangeBarProcessor::new(100).expect("Failed to create processor with valid threshold");
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    let range_bars = processor
        .process_agg_trade_records(&trades)
        .expect("Failed to process BTCUSDT trades");

    assert!(
        !range_bars.is_empty(),
        "Should produce at least one range bar"
    );

    println!(
        "BTCUSDT (100 bps): {} trades → {} range bars",
        trades.len(),
        range_bars.len()
    );

    validate_ohlcv_integrity(&range_bars);
    validate_temporal_ordering(&range_bars);
}

#[test]
fn test_btcusdt_threshold_scaling() {
    // Verify that bar count decreases with wider thresholds
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

    let bars_25 = RangeBarProcessor::new(25)
        .expect("Failed to create processor")
        .process_agg_trade_records(&trades)
        .expect("Failed to process trades");

    let bars_50 = RangeBarProcessor::new(50)
        .expect("Failed to create processor")
        .process_agg_trade_records(&trades)
        .expect("Failed to process trades");

    let bars_100 = RangeBarProcessor::new(100)
        .expect("Failed to create processor")
        .process_agg_trade_records(&trades)
        .expect("Failed to process trades");

    assert!(
        bars_25.len() >= bars_50.len(),
        "25 bps should produce >= bars than 50 bps (got {} vs {})",
        bars_25.len(),
        bars_50.len()
    );

    assert!(
        bars_50.len() >= bars_100.len(),
        "50 bps should produce >= bars than 100 bps (got {} vs {})",
        bars_50.len(),
        bars_100.len()
    );

    println!("BTCUSDT threshold scaling:");
    println!("  25 bps: {} bars", bars_25.len());
    println!("  50 bps: {} bars", bars_50.len());
    println!(" 100 bps: {} bars", bars_100.len());
}

/// Validates OHLCV integrity for all range bars
fn validate_ohlcv_integrity(range_bars: &[RangeBar]) {
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
    }
}

/// Validates temporal ordering of range bars
fn validate_temporal_ordering(range_bars: &[RangeBar]) {
    for (i, bar) in range_bars.iter().enumerate() {
        assert!(
            bar.open_time <= bar.close_time,
            "Bar {}: Open time ({}) should be <= close time ({})",
            i,
            bar.open_time,
            bar.close_time
        );

        if i > 0 {
            assert!(
                range_bars[i - 1].close_time <= bar.open_time,
                "Bar {}: Previous close time ({}) should be <= current open time ({})",
                i,
                range_bars[i - 1].close_time,
                bar.open_time
            );
        }
    }
}
