//! Incomplete Bar Continuation Proof
//!
//! This test EXPLICITLY proves that:
//! 1. An incomplete bar exists at file boundary (hasn't breached threshold)
//! 2. That SAME bar continues building with trades from the NEXT file
//! 3. The bar eventually completes when threshold is breached in the next file
//!
//! This is the TRUE validation of cross-file continuation.

use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::types::AggTrade;
use rangebar_core::FixedPoint;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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
        if fields.len() < 7 || fields[0] == "agg_trade_id" {
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
#[ignore]
fn test_incomplete_bar_continues_across_year_boundary() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  INCOMPLETE BAR CONTINUATION PROOF - Cross-Year Boundary         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  This test PROVES that an incomplete bar at Dec 31 end           ║");
    println!("║  CONTINUES building with Jan 1 trades until threshold breach.    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let dec31_path = cross_year_data_path("BTCUSDT-aggTrades-2023-12-31.csv");
    let jan01_path = cross_year_data_path("BTCUSDT-aggTrades-2024-01-01.csv");

    let dec31_trades = match load_aggtrades_csv(&dec31_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping - data not available: {}", e);
            return;
        }
    };

    let jan01_trades = match load_aggtrades_csv(&jan01_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping - data not available: {}", e);
            return;
        }
    };

    let threshold = 250; // 250 dbps = 0.25%

    // ═══════════════════════════════════════════════════════════════════
    // STEP 1: Process Dec 31 and capture the incomplete bar at boundary
    // ═══════════════════════════════════════════════════════════════════
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 1: Process Dec 31, 2023 - Capture incomplete bar          │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let dec31_bars = processor.process_agg_trade_records(&dec31_trades).unwrap();

    println!("  Dec 31 trades: {}", dec31_trades.len());
    println!("  Dec 31 completed bars: {}", dec31_bars.len());

    // Get the incomplete bar at boundary
    let incomplete_bar = processor.get_incomplete_bar();
    assert!(
        incomplete_bar.is_some(),
        "CRITICAL: There should be an incomplete bar at Dec 31 end!"
    );

    let inc_bar = incomplete_bar.unwrap();
    println!("\n  *** INCOMPLETE BAR AT DEC 31 END ***");
    println!("  Open price:     {}", inc_bar.open);
    println!("  Current high:   {}", inc_bar.high);
    println!("  Current low:    {}", inc_bar.low);
    println!("  Current close:  {}", inc_bar.close);
    println!("  Trade count:    {}", inc_bar.agg_record_count);
    println!("  Open time:      {}", inc_bar.open_time);

    let incomplete_open = inc_bar.open.clone();
    let incomplete_trade_count = inc_bar.agg_record_count;

    // ═══════════════════════════════════════════════════════════════════
    // STEP 2: Create checkpoint with the incomplete bar
    // ═══════════════════════════════════════════════════════════════════
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 2: Create checkpoint at year boundary                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let checkpoint = processor.create_checkpoint("BTCUSDT");

    assert!(
        checkpoint.has_incomplete_bar(),
        "Checkpoint should contain incomplete bar"
    );

    let cp_bar = checkpoint.incomplete_bar.as_ref().unwrap();
    println!("  Checkpoint incomplete bar open: {}", cp_bar.open);
    println!(
        "  Checkpoint incomplete bar trades: {}",
        cp_bar.agg_record_count
    );

    assert_eq!(
        cp_bar.open.0, incomplete_open.0,
        "Checkpoint bar should have same open as processor's incomplete bar"
    );

    // ═══════════════════════════════════════════════════════════════════
    // STEP 3: Resume from checkpoint and process Jan 1 trades
    // ═══════════════════════════════════════════════════════════════════
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 3: Resume from checkpoint - Process Jan 1, 2024            │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let mut processor_resumed = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    let jan01_bars = processor_resumed
        .process_agg_trade_records(&jan01_trades)
        .unwrap();

    println!("  Jan 1 trades: {}", jan01_trades.len());
    println!("  Jan 1 completed bars: {}", jan01_bars.len());

    // ═══════════════════════════════════════════════════════════════════
    // STEP 4: PROVE the incomplete bar was continued
    // ═══════════════════════════════════════════════════════════════════
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 4: PROVE incomplete bar continued across boundary          │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    assert!(
        !jan01_bars.is_empty(),
        "Jan 1 should have at least one completed bar"
    );

    let first_jan01_bar = &jan01_bars[0];
    println!("\n  First completed bar in Jan 1:");
    println!("  Open price:     {}", first_jan01_bar.open);
    println!("  High:           {}", first_jan01_bar.high);
    println!("  Low:            {}", first_jan01_bar.low);
    println!("  Close:          {}", first_jan01_bar.close);
    println!("  Trade count:    {}", first_jan01_bar.agg_record_count);

    // THE CRITICAL ASSERTION: First bar in Jan 1 should have the SAME open
    // as the incomplete bar from Dec 31 - proving it CONTINUED
    assert_eq!(
        first_jan01_bar.open.0, incomplete_open.0,
        "\n\n*** CRITICAL FAILURE ***\n\
         The first completed bar in Jan 1 should have the SAME open price\n\
         as the incomplete bar from Dec 31!\n\
         Dec 31 incomplete open: {}\n\
         Jan 1 first bar open:   {}\n\
         This proves the bar was NOT continued across the boundary!\n",
        incomplete_open, first_jan01_bar.open
    );

    // The first Jan 1 bar should have MORE trades than the Dec 31 incomplete bar
    // (because it accumulated trades from both files)
    assert!(
        first_jan01_bar.agg_record_count > incomplete_trade_count,
        "\n\n*** CRITICAL FAILURE ***\n\
         The continued bar should have MORE trades than Dec 31 incomplete bar!\n\
         Dec 31 incomplete trades: {}\n\
         Jan 1 first bar trades:   {}\n",
        incomplete_trade_count,
        first_jan01_bar.agg_record_count
    );

    let additional_trades = first_jan01_bar.agg_record_count - incomplete_trade_count;

    println!("\n  ╔════════════════════════════════════════════════════════════╗");
    println!("  ║  ✓ PROOF: INCOMPLETE BAR CONTINUED ACROSS YEAR BOUNDARY    ║");
    println!("  ╠════════════════════════════════════════════════════════════╣");
    println!(
        "  ║  Dec 31 incomplete bar: {} trades, open={}",
        incomplete_trade_count, incomplete_open
    );
    println!(
        "  ║  Jan 1 first bar:       {} trades, open={}",
        first_jan01_bar.agg_record_count, first_jan01_bar.open
    );
    println!("  ║  Additional trades from Jan 1: {}", additional_trades);
    println!("  ║                                                            ║");
    println!("  ║  The bar CONTINUED building with Jan 1 trades until       ║");
    println!("  ║  the 25bps threshold was breached!                         ║");
    println!("  ╚════════════════════════════════════════════════════════════╝");

    // ═══════════════════════════════════════════════════════════════════
    // STEP 5: Verify total bar count matches full processing
    // ═══════════════════════════════════════════════════════════════════
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STEP 5: Verify bar count matches full processing                │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let mut all_trades = dec31_trades.clone();
    all_trades.extend(jan01_trades.clone());

    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let full_bars = processor_full
        .process_agg_trade_records(&all_trades)
        .unwrap();

    let split_total = dec31_bars.len() + jan01_bars.len();

    println!("  Full processing:  {} bars", full_bars.len());
    println!(
        "  Split processing: {} + {} = {} bars",
        dec31_bars.len(),
        jan01_bars.len(),
        split_total
    );

    assert_eq!(
        split_total,
        full_bars.len(),
        "Split should match full processing"
    );

    println!("\n  ✓ Bar count matches - cross-file continuation is CORRECT");

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ALL ASSERTIONS PASSED - CROSS-YEAR CONTINUATION PROVEN         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

#[test]
#[ignore]
fn test_bar_content_identical_across_boundary() {
    println!("\n=== Bar Content Identity Test ===\n");

    let dec31_path = cross_year_data_path("BTCUSDT-aggTrades-2023-12-31.csv");
    let jan01_path = cross_year_data_path("BTCUSDT-aggTrades-2024-01-01.csv");

    let dec31_trades = match load_aggtrades_csv(&dec31_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping: {}", e);
            return;
        }
    };

    let jan01_trades = match load_aggtrades_csv(&jan01_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping: {}", e);
            return;
        }
    };

    let threshold = 250;

    // Split processing
    let mut processor = RangeBarProcessor::new(threshold).unwrap();
    let dec31_bars = processor.process_agg_trade_records(&dec31_trades).unwrap();
    let checkpoint = processor.create_checkpoint("BTCUSDT");
    let mut processor_resumed = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
    let jan01_bars = processor_resumed
        .process_agg_trade_records(&jan01_trades)
        .unwrap();

    // Full processing
    let mut all_trades = dec31_trades;
    all_trades.extend(jan01_trades);
    let mut processor_full = RangeBarProcessor::new(threshold).unwrap();
    let full_bars = processor_full
        .process_agg_trade_records(&all_trades)
        .unwrap();

    // Combine split bars
    let split_bars: Vec<_> = dec31_bars.iter().chain(jan01_bars.iter()).collect();

    // Verify EVERY bar is identical
    let mut mismatches = 0;
    for (i, (full, split)) in full_bars.iter().zip(split_bars.iter()).enumerate() {
        if full.open.0 != split.open.0
            || full.high.0 != split.high.0
            || full.low.0 != split.low.0
            || full.close.0 != split.close.0
            || full.volume.0 != split.volume.0
            || full.agg_record_count != split.agg_record_count
        {
            println!("Bar {} MISMATCH:", i);
            println!(
                "  Full:  O={} H={} L={} C={} trades={}",
                full.open, full.high, full.low, full.close, full.agg_record_count
            );
            println!(
                "  Split: O={} H={} L={} C={} trades={}",
                split.open, split.high, split.low, split.close, split.agg_record_count
            );
            mismatches += 1;
        }
    }

    assert_eq!(
        mismatches, 0,
        "{} bars have mismatched content - cross-file continuation is broken!",
        mismatches
    );

    println!(
        "✓ All {} bars are IDENTICAL between split and full processing",
        full_bars.len()
    );
    println!("✓ This proves the checkpoint correctly continues incomplete bars");
}
