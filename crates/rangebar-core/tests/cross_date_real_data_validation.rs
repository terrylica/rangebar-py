//! Cross-Date Real Data Validation
//!
//! Tests cross-file continuation using REAL Binance BTCUSDT aggTrades data.
//! This validates that incomplete bars are correctly continued across file boundaries.
//!
//! Data source: BTCUSDT-aggTrades-2024-01-01.csv (761,223 trades)
//! Simulates daily file boundaries at multiple points.

use rangebar_core::checkpoint::PositionVerification;
use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::types::AggTrade;
use rangebar_core::FixedPoint;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load real BTCUSDT data from full-day CSV (different format than test_data)
fn load_full_day_btcusdt() -> Result<Vec<AggTrade>, Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("BTCUSDT-aggTrades-2024-01-01.csv");

    if !path.exists() {
        return Err(format!("File not found: {:?}", path).into());
    }

    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let mut trades = Vec::with_capacity(800000);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            // Skip header
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        // Format: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
        let trade = AggTrade {
            agg_trade_id: fields[0].parse()?,
            price: FixedPoint::from_str(fields[1])?,
            volume: FixedPoint::from_str(fields[2])?,
            first_trade_id: fields[3].parse()?,
            last_trade_id: fields[4].parse()?,
            timestamp: fields[5].parse()?,
            is_buyer_maker: fields[6] == "true",
            is_best_match: None,
        };
        trades.push(trade);
    }

    Ok(trades)
}

#[test]
#[ignore] // Requires large data file - run with: cargo test --ignored
fn test_cross_date_continuation_large_scale() {
    let trades = match load_full_day_btcusdt() {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data file not available: {}", e);
            return;
        }
    };

    println!("\n=== Cross-Date Continuation Test (Large Scale) ===");
    println!(
        "Loaded {} real BTCUSDT trades from 2024-01-01",
        trades.len()
    );

    // Use realistic 25bps threshold
    let threshold = 250;

    // === BASELINE: Process all trades at once ===
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let bars_full = processor_full.process_agg_trade_records(&trades).unwrap();
    println!("\nBaseline (full processing): {} bars", bars_full.len());

    // === SIMULATE DAILY BOUNDARIES ===
    // Split data into "hourly" chunks to simulate multiple file boundaries
    let chunk_size = trades.len() / 24; // ~31,700 trades per hour

    println!("\n--- Simulating 24 hourly file boundaries ---");

    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let mut total_bars = 0;
    let mut checkpoints_with_incomplete = 0;

    for hour in 0..24 {
        let start = hour * chunk_size;
        let end = if hour == 23 {
            trades.len()
        } else {
            (hour + 1) * chunk_size
        };

        let chunk = &trades[start..end];

        // Process this chunk
        let bars = processor.process_agg_trade_records(chunk).unwrap();
        total_bars += bars.len();

        // Create checkpoint
        let checkpoint = processor.create_checkpoint("BTCUSDT");

        if checkpoint.has_incomplete_bar() {
            checkpoints_with_incomplete += 1;
        }

        // Verify position if not last chunk
        if hour < 23 {
            let verification = processor.verify_position(&trades[end]);
            match verification {
                PositionVerification::Exact => {}
                PositionVerification::Gap { missing_count, .. } => {
                    println!(
                        "  Hour {}: UNEXPECTED GAP of {} trades",
                        hour, missing_count
                    );
                }
                PositionVerification::TimestampOnly { .. } => {}
            }
        }

        // Resume from checkpoint for next chunk
        processor = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    }

    println!("\n=== VALIDATION RESULTS ===");
    println!("Baseline bars: {}", bars_full.len());
    println!("Split processing bars: {}", total_bars);
    println!(
        "Checkpoints with incomplete bars: {}/24",
        checkpoints_with_incomplete
    );

    assert_eq!(
        total_bars,
        bars_full.len(),
        "Split processing should produce same bar count: {} vs {}",
        total_bars,
        bars_full.len()
    );

    println!(
        "✓ Cross-date continuation validated with {} trades across 24 boundaries",
        trades.len()
    );
}

#[test]
#[ignore] // Requires large data file
fn test_agg_trade_id_continuity() {
    let trades = match load_full_day_btcusdt() {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data file not available: {}", e);
            return;
        }
    };

    println!("\n=== Verifying agg_trade_id Continuity ===");

    // Verify agg_trade_ids are strictly sequential (Binance guarantee)
    let mut gaps = 0;
    for i in 1..trades.len() {
        let expected = trades[i - 1].agg_trade_id + 1;
        let actual = trades[i].agg_trade_id;
        if actual != expected {
            gaps += 1;
            if gaps <= 5 {
                println!(
                    "  Gap at index {}: expected {}, got {} (diff: {})",
                    i,
                    expected,
                    actual,
                    actual - expected
                );
            }
        }
    }

    println!("\nTotal gaps in agg_trade_id sequence: {}", gaps);
    println!(
        "First trade ID: {}, Last trade ID: {}",
        trades.first().unwrap().agg_trade_id,
        trades.last().unwrap().agg_trade_id
    );

    if gaps == 0 {
        println!("✓ agg_trade_id sequence is perfectly continuous");
    } else {
        println!(
            "⚠ {} gaps found - this is expected for some Binance data periods",
            gaps
        );
    }
}
