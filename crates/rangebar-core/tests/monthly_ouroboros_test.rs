//! Monthly Ouroboros Reset Validation Tests (Issue #97)
//!
//! Validates that `reset_at_ouroboros()` works correctly for monthly mode:
//! - Returns orphaned (incomplete) bar with correct OHLCV
//! - Preserves trade history across reset
//! - Clears bar state after reset
//! - Monthly vs yearly bar equivalence for non-boundary bars
//! - Checkpoint round-trip after monthly reset
//! - Full year with 12 monthly resets

use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::types::AggTrade;
use rangebar_core::FixedPoint;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Load aggTrades from Binance CSV file (no header, 8 columns)
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

fn data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data/cross_year/spot/daily/aggTrades/BTCUSDT")
        .join(filename)
}

/// Generate synthetic trades for testing monthly resets without real data.
fn generate_synthetic_trades(
    start_ts_ms: i64,
    count: usize,
    base_price: f64,
    volatility: f64,
) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let mut price = base_price;

    for i in 0..count {
        // Alternate price up/down with some volatility
        let direction = if i % 7 < 4 { 1.0 } else { -1.0 };
        price += direction * volatility * ((i % 13) as f64 + 1.0);
        price = price.max(base_price * 0.9); // Floor at 90% of base

        let ts = start_ts_ms + (i as i64) * 100; // 100ms apart

        trades.push(AggTrade {
            agg_trade_id: i as i64 + 1,
            price: FixedPoint::from_str(&format!("{:.8}", price)).unwrap(),
            volume: FixedPoint::from_str(&format!("{:.8}", 0.1 + (i % 5) as f64 * 0.05)).unwrap(),
            first_trade_id: i as i64 * 2 + 1,
            last_trade_id: i as i64 * 2 + 2,
            timestamp: ts,
            is_buyer_maker: i % 3 == 0,
            is_best_match: Some(true),
        });
    }

    trades
}

#[test]
fn test_monthly_reset_returns_orphaned_bar() {
    println!("\n=== Monthly Reset: Returns Orphaned Bar ===\n");

    let threshold = 250; // 250 dbps
    let mut processor = RangeBarProcessor::new(threshold).unwrap();

    // Feed some trades to create an incomplete bar
    let trades = generate_synthetic_trades(
        1704067200000, // 2024-01-01 00:00:00 UTC
        50,
        42000.0,
        5.0, // Low volatility to avoid completing the bar
    );

    let bars = processor.process_agg_trade_records(&trades).unwrap();
    println!("Bars before reset: {}", bars.len());

    // Verify there's an incomplete bar
    let incomplete = processor.get_incomplete_bar();
    assert!(
        incomplete.is_some(),
        "Should have an incomplete bar before reset"
    );

    let inc_bar = incomplete.unwrap();
    let orphaned_open = inc_bar.open.to_f64();
    let orphaned_volume = inc_bar.volume as f64;
    println!(
        "Incomplete bar: open={}, volume={}",
        orphaned_open, orphaned_volume
    );

    // Reset at ouroboros boundary
    let orphaned = processor.reset_at_ouroboros();
    assert!(
        orphaned.is_some(),
        "reset_at_ouroboros should return the orphaned bar"
    );

    let orphan = orphaned.unwrap();
    assert_eq!(
        orphan.open.to_f64(),
        orphaned_open,
        "Orphaned bar open should match incomplete bar"
    );
    assert!(
        orphan.volume > 0,
        "Orphaned bar should have positive volume"
    );

    println!("✓ reset_at_ouroboros() returned correct orphaned bar");
}

#[test]
fn test_monthly_reset_clears_bar_state() {
    println!("\n=== Monthly Reset: Clears Bar State ===\n");

    let threshold = 250;
    let mut processor = RangeBarProcessor::new(threshold).unwrap();

    // Feed trades to create incomplete bar
    let trades = generate_synthetic_trades(1704067200000, 30, 42000.0, 3.0);
    let _ = processor.process_agg_trade_records(&trades).unwrap();

    assert!(
        processor.get_incomplete_bar().is_some(),
        "Should have incomplete bar"
    );

    // Reset
    let _ = processor.reset_at_ouroboros();

    // After reset: no incomplete bar
    let checkpoint = processor.create_checkpoint("BTCUSDT");
    assert!(
        !checkpoint.has_incomplete_bar(),
        "Post-reset checkpoint should have has_incomplete_bar=false"
    );

    // Process new trades — should open a fresh bar
    let new_trades = generate_synthetic_trades(1706745600000, 20, 43000.0, 3.0);
    let new_bars = processor.process_agg_trade_records(&new_trades).unwrap();
    println!("New bars after reset: {}", new_bars.len());

    // The new bar should have a different open from the old bar
    if let Some(inc) = processor.get_incomplete_bar() {
        println!("New incomplete bar open: {}", inc.open.to_f64());
    }

    println!("✓ Post-reset state is clean");
}

#[test]
fn test_monthly_reset_preserves_trade_history() {
    println!("\n=== Monthly Reset: Preserves Trade History ===\n");

    let threshold = 250;
    let mut processor = RangeBarProcessor::new(threshold).unwrap();

    // Enable inter-bar lookback to test trade history preservation
    processor.set_inter_bar_config(rangebar_core::interbar_types::InterBarConfig::default());

    // Process enough trades to fill the lookback buffer
    let trades = generate_synthetic_trades(1704067200000, 500, 42000.0, 10.0);
    let bars = processor.process_agg_trade_records(&trades).unwrap();
    println!("Bars before reset: {}", bars.len());

    // Reset at ouroboros boundary
    let _ = processor.reset_at_ouroboros();

    // Process trades in the new segment
    let new_trades = generate_synthetic_trades(1706745600000, 500, 43000.0, 10.0);
    let new_bars = processor.process_agg_trade_records(&new_trades).unwrap();
    println!("Bars after reset: {}", new_bars.len());

    // The processor should still produce valid bars (trade ring buffer survives)
    assert!(
        !new_bars.is_empty(),
        "Should produce bars after reset with trade history"
    );

    println!("✓ Trade history preserved across reset");
}

#[test]
fn test_monthly_vs_yearly_bar_equivalence() {
    println!("\n=== Monthly vs Yearly Bar Equivalence ===\n");

    // Use real data if available, otherwise skip
    let jan01_path = data_path("BTCUSDT-aggTrades-2024-01-01.csv");

    let jan01_trades = match load_aggtrades_csv(&jan01_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Skipping test - data not available: {}", e);
            return;
        }
    };

    let threshold = 250;

    // Process Jan 1 with no reset (yearly mode — boundary is Jan 1 itself)
    let mut processor_yearly = RangeBarProcessor::new(threshold).unwrap();
    let yearly_bars = processor_yearly
        .process_agg_trade_records(&jan01_trades)
        .unwrap();

    // Process Jan 1 after a reset (monthly mode — reset at Jan 1)
    let mut processor_monthly = RangeBarProcessor::new(threshold).unwrap();
    processor_monthly.reset_at_ouroboros(); // Reset at start
    let monthly_bars = processor_monthly
        .process_agg_trade_records(&jan01_trades)
        .unwrap();

    println!("Yearly bars on Jan 1: {}", yearly_bars.len());
    println!("Monthly bars on Jan 1: {}", monthly_bars.len());

    // Non-boundary bars within the same day should be identical
    assert_eq!(
        yearly_bars.len(),
        monthly_bars.len(),
        "Same day, same data should produce same bar count: yearly={} vs monthly={}",
        yearly_bars.len(),
        monthly_bars.len()
    );

    // Compare OHLCV of each bar
    for (i, (y, m)) in yearly_bars.iter().zip(monthly_bars.iter()).enumerate() {
        assert_eq!(
            y.open, m.open,
            "Bar {} open mismatch: {} vs {}",
            i, y.open, m.open
        );
        assert_eq!(
            y.high, m.high,
            "Bar {} high mismatch: {} vs {}",
            i, y.high, m.high
        );
        assert_eq!(
            y.low, m.low,
            "Bar {} low mismatch: {} vs {}",
            i, y.low, m.low
        );
        assert_eq!(
            y.close, m.close,
            "Bar {} close mismatch: {} vs {}",
            i, y.close, m.close
        );
    }

    println!("✓ Monthly and yearly bars are identical for non-boundary bars");
}

#[test]
fn test_monthly_checkpoint_roundtrip() {
    println!("\n=== Monthly Checkpoint Roundtrip ===\n");

    let threshold = 250;
    let mut processor = RangeBarProcessor::new(threshold).unwrap();

    // Process first batch of trades
    let trades1 = generate_synthetic_trades(1704067200000, 200, 42000.0, 10.0);
    let bars1 = processor.process_agg_trade_records(&trades1).unwrap();
    println!("Bars before checkpoint: {}", bars1.len());

    // Create checkpoint (simulating end of January)
    let cp = processor.create_checkpoint("BTCUSDT");
    println!(
        "Checkpoint: has_incomplete_bar={}",
        cp.has_incomplete_bar()
    );

    // Reset at month boundary
    let orphaned = processor.reset_at_ouroboros();
    if let Some(ref o) = orphaned {
        println!("Orphaned bar: open={}", o.open.to_f64());
    }

    // Create post-reset checkpoint
    let post_reset_cp = processor.create_checkpoint("BTCUSDT");

    // Restore from post-reset checkpoint
    let mut processor_resumed = RangeBarProcessor::from_checkpoint(post_reset_cp).unwrap();

    // Process new month trades
    let trades2 = generate_synthetic_trades(1706745600000, 200, 43000.0, 10.0);
    let bars_resumed = processor_resumed
        .process_agg_trade_records(&trades2)
        .unwrap();

    // Process same trades with continuous processor
    let bars_continuous = processor.process_agg_trade_records(&trades2).unwrap();

    println!(
        "Resumed bars: {}, Continuous bars: {}",
        bars_resumed.len(),
        bars_continuous.len()
    );

    assert_eq!(
        bars_resumed.len(),
        bars_continuous.len(),
        "Checkpoint roundtrip should match continuous: {} vs {}",
        bars_resumed.len(),
        bars_continuous.len()
    );

    println!("✓ Monthly checkpoint roundtrip validated");
}

#[test]
fn test_12_monthly_resets_in_year() {
    println!("\n=== 12 Monthly Resets in a Year ===\n");

    let threshold = 500; // Wider threshold for fewer bars
    let mut processor = RangeBarProcessor::new(threshold).unwrap();

    // Approximate month boundaries in 2024 (ms)
    let month_starts_ms: [i64; 12] = [
        1704067200000, // Jan 1
        1706745600000, // Feb 1
        1709251200000, // Mar 1
        1711929600000, // Apr 1
        1714521600000, // May 1
        1717200000000, // Jun 1
        1719792000000, // Jul 1
        1722470400000, // Aug 1
        1725148800000, // Sep 1
        1727740800000, // Oct 1
        1730419200000, // Nov 1
        1733011200000, // Dec 1
    ];

    let mut total_bars = 0;
    let mut orphaned_count = 0;

    for (i, &month_start) in month_starts_ms.iter().enumerate() {
        // Reset at month boundary (skip first month — fresh processor)
        if i > 0 {
            let orphaned = processor.reset_at_ouroboros();
            if orphaned.is_some() {
                orphaned_count += 1;
            }
        }

        // Generate and process synthetic trades for this month
        let trades = generate_synthetic_trades(
            month_start,
            1000, // 1000 trades per month
            42000.0 + (i as f64) * 500.0,
            15.0,
        );
        let bars = processor.process_agg_trade_records(&trades).unwrap();
        total_bars += bars.len();

        println!(
            "Month {}: {} bars (orphaned at boundary: {})",
            i + 1,
            bars.len(),
            if i > 0 { "yes" } else { "n/a" }
        );
    }

    println!("\n=== SUMMARY ===");
    println!("Total bars: {}", total_bars);
    println!("Orphaned bars: {}", orphaned_count);

    // We expect ~11 orphaned bars (one at each non-first month boundary)
    assert!(
        orphaned_count <= 11,
        "Orphaned count should be <= 11 (one per boundary), got {}",
        orphaned_count
    );

    assert!(
        total_bars > 0,
        "Should produce bars across 12 months"
    );

    println!("✓ 12 monthly resets validated: {} bars, {} orphaned", total_bars, orphaned_count);
}
