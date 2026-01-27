//! Cross-Year Boundary Test with Real Binance aggTrades Data
//!
//! Tests checkpoint continuation across REAL year boundaries:
//! - 2023-12-31 → 2024-01-01 (Year 2023→2024 boundary)
//! - 2024-12-31 → 2025-01-01 (Year 2024→2025 boundary)
//!
//! Data source: Binance aggTrades via Vision API (spot/daily/aggTrades)
//! Validates that incomplete bars are correctly continued across year boundaries.

use rangebar_core::checkpoint::PositionVerification;
use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::types::AggTrade;
use rangebar_core::FixedPoint;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Load aggTrades from Binance CSV file (no header, 8 columns)
/// Format: agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,is_buyer_maker,is_best_match
fn load_aggtrades_csv(path: &PathBuf) -> Result<Vec<AggTrade>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("File not found: {:?}", path).into());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut trades = Vec::with_capacity(1_000_000);

    for line in reader.lines() {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        // Skip header if present
        if fields[0] == "agg_trade_id" || fields[0].starts_with("a") {
            continue;
        }

        let trade = AggTrade {
            agg_trade_id: fields[0].parse()?,
            price: FixedPoint::from_str(fields[1])?,
            volume: FixedPoint::from_str(fields[2])?,
            first_trade_id: fields[3].parse()?,
            last_trade_id: fields[4].parse()?,
            timestamp: fields[5].parse()?,
            is_buyer_maker: fields[6] == "True" || fields[6] == "true",
            is_best_match: if fields.len() > 7 {
                Some(fields[7] == "True" || fields[7] == "true")
            } else {
                None
            },
        };
        trades.push(trade);
    }

    Ok(trades)
}

fn cross_year_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/cross_year/spot/daily/aggTrades/BTCUSDT")
        .join(filename)
}

#[test]
#[ignore] // Requires cross-year data files - run with: cargo test --ignored
fn test_cross_year_2023_2024_boundary() {
    println!("\n=== Cross-Year Boundary Test: 2023→2024 ===\n");

    // Load real Binance aggTrades data
    let dec31_path = cross_year_data_path("BTCUSDT-aggTrades-2023-12-31.csv");
    let jan01_path = cross_year_data_path("BTCUSDT-aggTrades-2024-01-01.csv");

    let dec31_trades = match load_aggtrades_csv(&dec31_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data not available: {}", e);
            return;
        }
    };

    let jan01_trades = match load_aggtrades_csv(&jan01_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data not available: {}", e);
            return;
        }
    };

    println!("Dec 31, 2023: {} trades", dec31_trades.len());
    println!("Jan 1, 2024: {} trades", jan01_trades.len());

    // Verify agg_trade_id continuity across year boundary
    let last_dec31_id = dec31_trades.last().unwrap().agg_trade_id;
    let first_jan01_id = jan01_trades.first().unwrap().agg_trade_id;

    println!(
        "\nBoundary check: Dec 31 last ID={} → Jan 1 first ID={}",
        last_dec31_id, first_jan01_id
    );
    assert_eq!(
        first_jan01_id,
        last_dec31_id + 1,
        "agg_trade_id should be continuous across year boundary"
    );
    println!("✓ agg_trade_id is continuous across year boundary");

    // === CROSS-YEAR CONTINUATION TEST ===
    let threshold = 250; // 250 dbps = 0.25%

    // Process Dec 31 data
    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let dec31_bars = processor.process_agg_trade_records(&dec31_trades).unwrap();
    println!("\nDec 31 completed bars: {}", dec31_bars.len());

    // Create checkpoint at year boundary
    let checkpoint = processor.create_checkpoint("BTCUSDT");
    println!(
        "Checkpoint at year boundary: has_incomplete_bar={}",
        checkpoint.has_incomplete_bar()
    );

    if let Some(ref inc_bar) = checkpoint.incomplete_bar {
        println!(
            "  Incomplete bar: open={}, trades={}",
            inc_bar.open, inc_bar.agg_record_count
        );
    }

    // Resume from checkpoint and process Jan 1 data
    let mut processor_resumed = RangeBarProcessor::from_checkpoint(checkpoint.clone()).unwrap();

    // Verify position
    let verification = processor_resumed.verify_position(&jan01_trades[0]);
    println!(
        "\nPosition verification at year boundary: {:?}",
        verification
    );
    match verification {
        PositionVerification::Exact => println!("✓ Exact position match at year boundary"),
        PositionVerification::Gap { missing_count, .. } => {
            panic!(
                "Unexpected gap of {} trades at year boundary",
                missing_count
            );
        }
        PositionVerification::TimestampOnly { .. } => {}
    }

    let jan01_bars = processor_resumed
        .process_agg_trade_records(&jan01_trades)
        .unwrap();
    println!("Jan 1 completed bars: {}", jan01_bars.len());

    // === BASELINE COMPARISON ===
    // Process all data together (combine Dec 31 + Jan 1)
    let mut all_trades = dec31_trades.clone();
    all_trades.extend(jan01_trades.clone());

    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let full_bars = processor_full
        .process_agg_trade_records(&all_trades)
        .unwrap();

    let split_total = dec31_bars.len() + jan01_bars.len();
    println!("\n=== VALIDATION RESULTS ===");
    println!("Full processing (combined): {} bars", full_bars.len());
    println!(
        "Split processing: {} + {} = {} bars",
        dec31_bars.len(),
        jan01_bars.len(),
        split_total
    );

    assert_eq!(
        split_total,
        full_bars.len(),
        "Cross-year split should produce same bar count: {} vs {}",
        split_total,
        full_bars.len()
    );

    println!("\n✓ CROSS-YEAR 2023→2024 BOUNDARY VALIDATED");
    println!(
        "  {} trades processed across year boundary",
        dec31_trades.len() + jan01_trades.len()
    );
}

#[test]
#[ignore]
fn test_cross_year_2024_2025_boundary() {
    println!("\n=== Cross-Year Boundary Test: 2024→2025 ===\n");

    let dec31_path = cross_year_data_path("BTCUSDT-aggTrades-2024-12-31.csv");
    let jan01_path = cross_year_data_path("BTCUSDT-aggTrades-2025-01-01.csv");

    let dec31_trades = match load_aggtrades_csv(&dec31_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data not available: {}", e);
            return;
        }
    };

    let jan01_trades = match load_aggtrades_csv(&jan01_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data not available: {}", e);
            return;
        }
    };

    println!("Dec 31, 2024: {} trades", dec31_trades.len());
    println!("Jan 1, 2025: {} trades", jan01_trades.len());

    // Verify continuity
    let last_dec31_id = dec31_trades.last().unwrap().agg_trade_id;
    let first_jan01_id = jan01_trades.first().unwrap().agg_trade_id;
    println!(
        "Boundary: Dec 31 last ID={} → Jan 1 first ID={}",
        last_dec31_id, first_jan01_id
    );

    let threshold = 250;

    // Process with checkpoint at year boundary
    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let dec31_bars = processor.process_agg_trade_records(&dec31_trades).unwrap();

    let checkpoint = processor.create_checkpoint("BTCUSDT");
    let mut processor_resumed = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    let jan01_bars = processor_resumed
        .process_agg_trade_records(&jan01_trades)
        .unwrap();

    // Baseline
    let mut all_trades = dec31_trades.clone();
    all_trades.extend(jan01_trades.clone());
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let full_bars = processor_full
        .process_agg_trade_records(&all_trades)
        .unwrap();

    let split_total = dec31_bars.len() + jan01_bars.len();

    println!("\n=== VALIDATION ===");
    println!(
        "Full: {} bars, Split: {} + {} = {}",
        full_bars.len(),
        dec31_bars.len(),
        jan01_bars.len(),
        split_total
    );

    assert_eq!(
        split_total,
        full_bars.len(),
        "Cross-year 2024→2025 should match: {} vs {}",
        split_total,
        full_bars.len()
    );

    println!("✓ CROSS-YEAR 2024→2025 BOUNDARY VALIDATED");
}

#[test]
#[ignore]
fn test_multi_day_sequential_processing() {
    println!("\n=== Multi-Day Sequential Processing Test ===\n");

    // Test processing 7 consecutive days with checkpoints between each
    let days = [
        "BTCUSDT-aggTrades-2023-12-26.csv",
        "BTCUSDT-aggTrades-2023-12-27.csv",
        "BTCUSDT-aggTrades-2023-12-28.csv",
        "BTCUSDT-aggTrades-2023-12-29.csv",
        "BTCUSDT-aggTrades-2023-12-30.csv",
        "BTCUSDT-aggTrades-2023-12-31.csv",
        "BTCUSDT-aggTrades-2024-01-01.csv",
    ];

    let threshold = 250;
    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let mut total_bars = 0;
    let mut total_trades = 0;
    let mut all_trades = Vec::new();

    for (i, day) in days.iter().enumerate() {
        let path = cross_year_data_path(day);
        let trades = match load_aggtrades_csv(&path) {
            Ok(t) => t,
            Err(e) => {
                println!("Skipping {} - not available: {}", day, e);
                continue;
            }
        };

        total_trades += trades.len();
        all_trades.extend(trades.clone());

        let bars = processor.process_agg_trade_records(&trades).unwrap();
        total_bars += bars.len();

        let checkpoint = processor.create_checkpoint("BTCUSDT");
        println!(
            "Day {}: {} → {} bars (incomplete: {})",
            i + 1,
            day,
            bars.len(),
            checkpoint.has_incomplete_bar()
        );

        // Resume from checkpoint for next day
        processor = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    }

    // Baseline: process all days together
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let full_bars = processor_full
        .process_agg_trade_records(&all_trades)
        .unwrap();

    println!("\n=== VALIDATION ===");
    println!("Total trades: {}", total_trades);
    println!("Sequential processing: {} bars", total_bars);
    println!("Full processing: {} bars", full_bars.len());

    assert_eq!(
        total_bars,
        full_bars.len(),
        "Sequential multi-day should match full: {} vs {}",
        total_bars,
        full_bars.len()
    );

    println!("✓ MULTI-DAY SEQUENTIAL PROCESSING VALIDATED (including year boundary)");
}
