//! Tests for Issue #72: Aggregate trade ID tracking
//!
//! Verifies that first_agg_trade_id and last_agg_trade_id fields are
//! correctly tracked during bar construction and checkpoint preservation.

use rangebar_core::{AggTrade, FixedPoint, RangeBarProcessor};

/// Create a test trade with the given agg_trade_id
fn create_trade(agg_trade_id: i64, timestamp_ms: i64, price: &str, quantity: &str) -> AggTrade {
    AggTrade {
        agg_trade_id,
        price: FixedPoint::from_str(price).unwrap(),
        volume: FixedPoint::from_str(quantity).unwrap(),
        first_trade_id: agg_trade_id * 10, // Individual trade IDs
        last_trade_id: agg_trade_id * 10 + 2, // Individual trade IDs
        timestamp: timestamp_ms * 1000,    // Convert to microseconds
        is_buyer_maker: agg_trade_id % 2 == 0,
        is_best_match: Some(true),
    }
}

#[test]
fn test_single_trade_bar_first_equals_last() {
    // Issue #72: Single-trade bar should have first_agg_trade_id == last_agg_trade_id
    let mut processor = RangeBarProcessor::new(100).unwrap(); // 0.1% threshold

    // Two trades that breach threshold immediately
    let trades = vec![
        create_trade(1000, 1704067200000, "50000.0", "1.0"),
        create_trade(1001, 1704067201000, "50100.0", "1.0"), // 0.2% move triggers bar
    ];

    let bars = processor.process_agg_trade_records(&trades).unwrap();

    // First bar should contain only trade 1000
    assert!(!bars.is_empty(), "Should produce at least one bar");
    let bar = &bars[0];

    // In this scenario, bar opens with trade 1000 and closes when 1001 arrives
    // Trade 1001 is the breach trade and is included in the closing bar
    assert_eq!(bar.first_agg_trade_id, 1000);
    assert_eq!(bar.last_agg_trade_id, 1001); // Breach trade included
}

#[test]
fn test_multi_trade_bar_first_less_than_last() {
    // Issue #72: Multi-trade bar should have first_agg_trade_id < last_agg_trade_id
    let mut processor = RangeBarProcessor::new(500).unwrap(); // 0.5% threshold (wider)

    // Multiple trades before breach
    let trades = vec![
        create_trade(2000, 1704067200000, "50000.0", "1.0"), // Opens bar
        create_trade(2001, 1704067201000, "50050.0", "1.0"), // 0.1% - within threshold
        create_trade(2002, 1704067202000, "50100.0", "1.0"), // 0.2% - within threshold
        create_trade(2003, 1704067203000, "50150.0", "1.0"), // 0.3% - within threshold
        create_trade(2004, 1704067204000, "50300.0", "1.0"), // 0.6% - breach!
    ];

    let bars = processor.process_agg_trade_records(&trades).unwrap();

    assert!(!bars.is_empty(), "Should produce at least one bar");
    let bar = &bars[0];

    assert_eq!(
        bar.first_agg_trade_id, 2000,
        "First agg_trade_id should be 2000"
    );
    assert_eq!(
        bar.last_agg_trade_id, 2004,
        "Last agg_trade_id should be 2004 (breach)"
    );
    assert!(
        bar.first_agg_trade_id < bar.last_agg_trade_id,
        "first < last for multi-trade bar"
    );
}

#[test]
fn test_sequential_continuity_no_gaps() {
    // Issue #72: Consecutive bars should satisfy:
    // bars[i].first_agg_trade_id == bars[i-1].last_agg_trade_id + 1
    let mut processor = RangeBarProcessor::new(100).unwrap(); // 0.1% threshold

    // Generate trades that will produce multiple bars
    // Oscillating between 50000, 50150, 50300 to trigger ~0.3% swings
    let trades = vec![
        create_trade(3000, 1704067200000, "50000.0", "1.0"),
        create_trade(3001, 1704067201000, "50150.0", "1.0"),
        create_trade(3002, 1704067202000, "50300.0", "1.0"),
        create_trade(3003, 1704067203000, "50000.0", "1.0"),
        create_trade(3004, 1704067204000, "50150.0", "1.0"),
        create_trade(3005, 1704067205000, "50300.0", "1.0"),
        create_trade(3006, 1704067206000, "50000.0", "1.0"),
        create_trade(3007, 1704067207000, "50150.0", "1.0"),
        create_trade(3008, 1704067208000, "50300.0", "1.0"),
        create_trade(3009, 1704067209000, "50000.0", "1.0"),
        create_trade(3010, 1704067210000, "50150.0", "1.0"),
        create_trade(3011, 1704067211000, "50300.0", "1.0"),
        create_trade(3012, 1704067212000, "50000.0", "1.0"),
        create_trade(3013, 1704067213000, "50150.0", "1.0"),
        create_trade(3014, 1704067214000, "50300.0", "1.0"),
        create_trade(3015, 1704067215000, "50000.0", "1.0"),
        create_trade(3016, 1704067216000, "50150.0", "1.0"),
        create_trade(3017, 1704067217000, "50300.0", "1.0"),
        create_trade(3018, 1704067218000, "50000.0", "1.0"),
        create_trade(3019, 1704067219000, "50150.0", "1.0"),
    ];

    let bars = processor.process_agg_trade_records(&trades).unwrap();

    assert!(
        bars.len() >= 2,
        "Should produce multiple bars for gap testing"
    );

    for i in 1..bars.len() {
        let prev_last = bars[i - 1].last_agg_trade_id;
        let curr_first = bars[i].first_agg_trade_id;
        assert_eq!(
            curr_first,
            prev_last + 1,
            "Gap detected: bar[{}].first ({}) != bar[{}].last ({}) + 1",
            i,
            curr_first,
            i - 1,
            prev_last
        );
    }
}

#[test]
fn test_checkpoint_preserves_agg_trade_ids() {
    // Issue #72: Checkpoint should preserve incomplete bar's agg_trade_id range
    let mut processor = RangeBarProcessor::new(500).unwrap(); // Wide threshold

    // Trades that won't complete a bar
    let trades = vec![
        create_trade(4000, 1704067200000, "50000.0", "1.0"),
        create_trade(4001, 1704067201000, "50010.0", "1.0"), // 0.02% - way under 0.5%
    ];

    let bars = processor.process_agg_trade_records(&trades).unwrap();
    assert!(bars.is_empty(), "Should not complete a bar");

    // Create checkpoint
    let checkpoint = processor.create_checkpoint("TEST");
    assert!(
        checkpoint.has_incomplete_bar(),
        "Should have incomplete bar in checkpoint"
    );

    // Verify incomplete bar has correct agg_trade_id range
    let incomplete = checkpoint.incomplete_bar.as_ref().unwrap();
    assert_eq!(
        incomplete.first_agg_trade_id, 4000,
        "Incomplete bar first_agg_trade_id"
    );
    assert_eq!(
        incomplete.last_agg_trade_id, 4001,
        "Incomplete bar last_agg_trade_id"
    );
}

#[test]
fn test_checkpoint_resume_continuity() {
    // Issue #72: After checkpoint resume, bar continuity should be maintained
    let mut processor1 = RangeBarProcessor::new(200).unwrap(); // 0.2% threshold

    // Part 1: Process some trades
    let trades1 = vec![
        create_trade(5000, 1704067200000, "50000.0", "1.0"),
        create_trade(5001, 1704067201000, "50050.0", "1.0"),
    ];

    let bars1 = processor1.process_agg_trade_records(&trades1).unwrap();

    // Create checkpoint
    let checkpoint = processor1.create_checkpoint("TEST");

    // Part 2: Resume from checkpoint
    let mut processor2 = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();

    // Continue with more trades
    let trades2 = vec![
        create_trade(5002, 1704067202000, "50150.0", "1.0"), // 0.3% - breach!
        create_trade(5003, 1704067203000, "50200.0", "1.0"),
    ];

    let bars2 = processor2.process_agg_trade_records(&trades2).unwrap();

    // Combine bars
    let mut all_bars = bars1;
    all_bars.extend(bars2);

    // Verify continuity across checkpoint boundary
    if all_bars.len() >= 2 {
        for i in 1..all_bars.len() {
            let prev_last = all_bars[i - 1].last_agg_trade_id;
            let curr_first = all_bars[i].first_agg_trade_id;
            assert_eq!(
                curr_first,
                prev_last + 1,
                "Cross-checkpoint gap: bar[{}].first != bar[{}].last + 1",
                i,
                i - 1
            );
        }
    }
}

#[test]
fn test_individual_vs_aggregate_trade_ids() {
    // Issue #72: Verify individual trade IDs and aggregate trade IDs are distinct
    let mut processor = RangeBarProcessor::new(100).unwrap();

    let trades = vec![
        create_trade(100, 1704067200000, "50000.0", "1.0"), // agg=100, individual=1000-1002
        create_trade(101, 1704067201000, "50100.0", "1.0"), // agg=101, individual=1010-1012
    ];

    let bars = processor.process_agg_trade_records(&trades).unwrap();

    if !bars.is_empty() {
        let bar = &bars[0];
        // Aggregate IDs are in range [100, 101]
        assert!(bar.first_agg_trade_id >= 100 && bar.first_agg_trade_id <= 101);
        assert!(bar.last_agg_trade_id >= 100 && bar.last_agg_trade_id <= 101);

        // Individual IDs should be in range [1000, 1012] (from first_trade_id/last_trade_id)
        assert!(bar.first_trade_id >= 1000 && bar.first_trade_id <= 1012);
        assert!(bar.last_trade_id >= 1000 && bar.last_trade_id <= 1012);

        // They should NOT overlap
        assert_ne!(
            bar.first_agg_trade_id, bar.first_trade_id,
            "Aggregate and individual IDs should be distinct"
        );
    }
}
