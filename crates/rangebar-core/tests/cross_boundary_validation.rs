//! Cross-boundary validation test with real BTCUSDT data
//!
//! Validates that incomplete bars at file boundaries continue correctly.
//! This is the PRIMARY validation for GitHub Issues #2 and #3.

use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::test_data_loader::load_btcusdt_test_data;

#[test]
fn test_cross_boundary_bar_continuation_with_real_data() {
    println!("\n=== Cross-Boundary Range Bar Continuation Validation ===\n");

    // Load real BTCUSDT data (5,000 trades)
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");
    println!("Loaded {} real BTCUSDT trades", trades.len());

    // Use 250 decimal bps = 25 bps = 0.25% threshold
    let threshold = 250;

    // === FULL PROCESSING (baseline) ===
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let bars_full = processor_full.process_agg_trade_records(&trades).unwrap();
    println!(
        "\nFull processing (baseline): {} completed bars",
        bars_full.len()
    );

    // === SPLIT PROCESSING AT MID-BAR BOUNDARY ===
    let split_point = 2500;

    println!("\n--- Split at trade {} ---", split_point);

    // Process first "file"
    let mut processor_1 = RangeBarProcessor::new(threshold).unwrap();
    let bars_1 = processor_1
        .process_agg_trade_records(&trades[0..split_point])
        .unwrap();
    println!("File 1: {} completed bars", bars_1.len());

    // Check for incomplete bar at boundary
    let incomplete_bar = processor_1.get_incomplete_bar();
    if let Some(ref bar) = incomplete_bar {
        println!("\n*** INCOMPLETE BAR AT BOUNDARY ***");
        println!("  Open: {}", bar.open);
        println!("  High: {}", bar.high);
        println!("  Low: {}", bar.low);
        println!("  Close: {}", bar.close);
        println!("  Trade count: {}", bar.agg_record_count);
    } else {
        println!("No incomplete bar at boundary (clean split)");
    }

    // Create checkpoint
    let checkpoint = processor_1.create_checkpoint("BTCUSDT");
    println!("\nCheckpoint created:");
    println!("  Has incomplete bar: {}", checkpoint.has_incomplete_bar());
    println!("  Last trade ID: {:?}", checkpoint.last_trade_id);

    // Resume from checkpoint
    let mut processor_2 = RangeBarProcessor::from_checkpoint(checkpoint.clone()).unwrap();

    // Verify position continuity
    let verification = processor_2.verify_position(&trades[split_point]);
    println!(
        "\nPosition verification at trade {}: {:?}",
        split_point, verification
    );

    // Process second "file"
    let bars_2 = processor_2
        .process_agg_trade_records(&trades[split_point..])
        .unwrap();
    println!("File 2: {} completed bars", bars_2.len());

    // === CRITICAL VALIDATION ===
    let split_total = bars_1.len() + bars_2.len();
    println!("\n=== VALIDATION RESULTS ===");
    println!("Full processing: {} bars", bars_full.len());
    println!(
        "Split processing: {} + {} = {} bars",
        bars_1.len(),
        bars_2.len(),
        split_total
    );

    assert_eq!(
        split_total,
        bars_full.len(),
        "Split processing should produce same bar count as full processing"
    );

    // Verify bar-by-bar content
    let all_split_bars: Vec<_> = bars_1.iter().chain(bars_2.iter()).collect();

    for (i, (full, split)) in bars_full.iter().zip(all_split_bars.iter()).enumerate() {
        assert_eq!(
            full.open.0, split.open.0,
            "Bar {} open mismatch: {} vs {}",
            i, full.open, split.open
        );
        assert_eq!(
            full.high.0, split.high.0,
            "Bar {} high mismatch: {} vs {}",
            i, full.high, split.high
        );
        assert_eq!(
            full.low.0, split.low.0,
            "Bar {} low mismatch: {} vs {}",
            i, full.low, split.low
        );
        assert_eq!(
            full.close.0, split.close.0,
            "Bar {} close mismatch: {} vs {}",
            i, full.close, split.close
        );
        assert_eq!(
            full.volume.0, split.volume.0,
            "Bar {} volume mismatch: {} vs {}",
            i, full.volume, split.volume
        );
    }

    println!("✓ ALL {} BARS MATCH EXACTLY", bars_full.len());

    // === Show what happens to the cross-boundary bar ===
    if checkpoint.has_incomplete_bar() {
        println!("\n=== CROSS-BOUNDARY BAR ANALYSIS ===");
        let inc_bar = checkpoint.incomplete_bar.as_ref().unwrap();

        if !bars_2.is_empty() {
            let first_bar_2 = &bars_2[0];
            println!("Incomplete bar at boundary:");
            println!("  Started with open: {}", inc_bar.open);
            println!("  Had {} trades before boundary", inc_bar.agg_record_count);
            println!("\nFirst completed bar in File 2:");
            println!("  Open: {}", first_bar_2.open);
            println!("  Total trades: {}", first_bar_2.agg_record_count);

            if first_bar_2.open.0 == inc_bar.open.0 {
                let additional_trades = first_bar_2.agg_record_count - inc_bar.agg_record_count;
                println!("\n✓ CONFIRMED: Incomplete bar CONTINUED across boundary!");
                println!(
                    "  {} trades before boundary + {} after = {} total",
                    inc_bar.agg_record_count, additional_trades, first_bar_2.agg_record_count
                );
            }
        }
    }

    println!("\n=== ALL CROSS-BOUNDARY TESTS PASSED ===");
}

#[test]
fn test_multiple_split_points_with_real_data() {
    // Test that cross-boundary continuation works at multiple split points
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");
    let threshold = 250;

    // Baseline
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let bars_full = processor_full.process_agg_trade_records(&trades).unwrap();

    // Test multiple split points
    for split_point in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500] {
        let mut processor_1 = RangeBarProcessor::new(threshold).unwrap();
        let bars_1 = processor_1
            .process_agg_trade_records(&trades[0..split_point])
            .unwrap();

        let checkpoint = processor_1.create_checkpoint("BTCUSDT");
        let mut processor_2 = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
        let bars_2 = processor_2
            .process_agg_trade_records(&trades[split_point..])
            .unwrap();

        let split_total = bars_1.len() + bars_2.len();

        assert_eq!(
            split_total,
            bars_full.len(),
            "Split at {} failed: {} != {}",
            split_point,
            split_total,
            bars_full.len()
        );
    }

    println!("✓ All 9 split points validated successfully");
}

#[test]
fn test_debug_split_at_500() {
    let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");
    let threshold = 250;

    // Baseline
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let bars_full = processor_full.process_agg_trade_records(&trades).unwrap();
    println!("\nFull processing: {} bars", bars_full.len());

    // Split at 500
    let split_point = 500;

    let mut processor_1 = RangeBarProcessor::new(threshold).unwrap();
    let bars_1 = processor_1
        .process_agg_trade_records(&trades[0..split_point])
        .unwrap();

    // Check incomplete bar BEFORE checkpoint
    let incomplete = processor_1.get_incomplete_bar();
    println!("\nFile 1: {} completed bars", bars_1.len());
    if let Some(ref bar) = incomplete {
        println!("INCOMPLETE BAR at boundary:");
        println!("  Open: {}", bar.open);
        println!("  Trades: {}", bar.agg_record_count);
    } else {
        println!("No incomplete bar at boundary");
    }

    let checkpoint = processor_1.create_checkpoint("BTCUSDT");
    println!(
        "\nCheckpoint has_incomplete_bar: {}",
        checkpoint.has_incomplete_bar()
    );

    let mut processor_2 = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    let bars_2 = processor_2
        .process_agg_trade_records(&trades[split_point..])
        .unwrap();

    println!("\nFile 2: {} completed bars", bars_2.len());

    let split_total = bars_1.len() + bars_2.len();
    println!(
        "\nTotal: {} + {} = {} (expected {})",
        bars_1.len(),
        bars_2.len(),
        split_total,
        bars_full.len()
    );
    println!(
        "Difference: {}",
        bars_full.len() as i32 - split_total as i32
    );

    // This test is for debugging - don't assert yet
}
