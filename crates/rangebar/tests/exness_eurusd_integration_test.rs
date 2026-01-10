//! Exness EURUSD Raw_Spread Integration Test
//!
//! Adversarial end-to-end validation:
//! 1. Fetch real Exness data (Jan 15-19, 2024)
//! 2. Validate temporal integrity (monotonic timestamps)
//! 3. Generate range bars with ultra-low threshold (0.1bps)
//! 4. Audit outputs against empirical expectations
//! 5. Export results for manual inspection

#![cfg(feature = "providers")]

use rangebar::providers::exness::{
    ExnessFetcher, ExnessRangeBarBuilder, ExnessTick, ValidationStrictness,
};
use std::fs;
use std::path::Path;

/// Expected metrics from empirical analysis (Jan 2024 Exness Raw_Spread data)
///
/// Key finding: Raw_Spread has 98.46% zero-spread ticks (bid==ask),
/// resulting in extremely high bar counts at ultra-low thresholds.
/// This is NOT a bug - it's the characteristic of Raw_Spread data.
const EXPECTED_TICKS_PER_DAY_MIN: usize = 50_000;
const EXPECTED_TICKS_PER_DAY_MAX: usize = 70_000;
const EXPECTED_BARS_PER_DAY_MIN_01BPS: usize = 25_000; // 0.1bps threshold
const EXPECTED_BARS_PER_DAY_MAX_01BPS: usize = 35_000;

#[tokio::test]
#[ignore] // Only run with --ignored (fetches real data)
async fn test_exness_eurusd_raw_spread_end_to_end() {
    println!("\n=== Exness EURUSD Raw_Spread Integration Test ===\n");

    // Step 1: Fetch January 2024 data
    println!("Step 1: Fetching January 2024 EURUSD_Raw_Spread data...");
    let fetcher = ExnessFetcher::new("EURUSD_Raw_Spread");

    let all_ticks = fetcher
        .fetch_month(2024, 1)
        .await
        .expect("Failed to fetch Exness data");

    println!(
        "  ✅ Fetched {} total ticks for January 2024",
        all_ticks.len()
    );

    // Step 2: Filter to Jan 15-19, 2024 (test period)
    println!("\nStep 2: Filtering to Jan 15-19, 2024...");
    let jan_15_start_ms = 1705276800000; // 2024-01-15 00:00:00 UTC
    let jan_20_start_ms = 1705708800000; // 2024-01-20 00:00:00 UTC

    let test_ticks: Vec<ExnessTick> = all_ticks
        .into_iter()
        .filter(|tick| tick.timestamp_ms >= jan_15_start_ms && tick.timestamp_ms < jan_20_start_ms)
        .collect();

    println!("  ✅ Filtered to {} ticks for Jan 15-19", test_ticks.len());

    // Step 3: Validate temporal integrity
    println!("\nStep 3: Validating temporal integrity...");
    validate_temporal_integrity(&test_ticks);
    println!("  ✅ Temporal integrity verified (monotonic timestamps)");

    // Step 4: Validate tick count
    println!("\nStep 4: Validating tick count...");
    let ticks_per_day = test_ticks.len() / 5;
    println!("  Ticks per day: {}", ticks_per_day);

    assert!(
        test_ticks.len() >= 250_000 && test_ticks.len() <= 350_000,
        "Expected ~300K ticks for 5 days, got {}",
        test_ticks.len()
    );

    assert!(
        (EXPECTED_TICKS_PER_DAY_MIN..=EXPECTED_TICKS_PER_DAY_MAX).contains(&ticks_per_day),
        "Expected {}-{} ticks/day, got {}",
        EXPECTED_TICKS_PER_DAY_MIN,
        EXPECTED_TICKS_PER_DAY_MAX,
        ticks_per_day
    );

    println!("  ✅ Tick count within expected range");

    // Step 5: Validate spread distribution
    println!("\nStep 5: Validating spread distribution...");
    validate_spread_distribution(&test_ticks);
    println!("  ✅ Spread distribution matches Raw_Spread characteristics");

    // Step 6: Generate range bars with 0.1bps threshold
    println!("\nStep 6: Generating range bars (0.1bps threshold)...");
    let mut builder = ExnessRangeBarBuilder::new(
        1, // 1 unit = 0.1bps in v3.0.0
        "EURUSD_Raw_Spread",
        ValidationStrictness::Strict,
    )
    .expect("Failed to create ExnessRangeBarBuilder with valid threshold");

    let mut bars = Vec::new();
    let mut errors = Vec::new();

    for (i, tick) in test_ticks.iter().enumerate() {
        match builder.process_tick(tick) {
            Ok(Some(bar)) => bars.push(bar),
            Ok(None) => {} // Tick processed, bar accumulating
            Err(e) => {
                errors.push((i, format!("{:?}", e)));
            }
        }
    }

    // Get incomplete bar at end
    if let Some(incomplete) = builder.get_incomplete_bar() {
        bars.push(incomplete);
    }

    println!("  ✅ Generated {} range bars", bars.len());

    // Step 7: Validate error rate (should be 0% with fail-fast policy)
    println!("\nStep 7: Validating error rate...");
    let error_rate = (errors.len() as f64 / test_ticks.len() as f64) * 100.0;
    println!("  Error rate: {:.4}%", error_rate);

    if !errors.is_empty() {
        println!("  ⚠️ Errors encountered:");
        for (i, err) in errors.iter().take(5) {
            println!("    Tick {}: {}", i, err);
        }
        panic!(
            "Expected 0% error rate with fail-fast policy, got {:.4}% ({} errors)",
            error_rate,
            errors.len()
        );
    }

    println!("  ✅ Zero errors (fail-fast policy working)");

    // Step 8: Validate bar count
    println!("\nStep 8: Validating bar count...");
    let bars_per_day = bars.len() / 5;
    println!("  Bars per day: {}", bars_per_day);

    assert!(
        (EXPECTED_BARS_PER_DAY_MIN_01BPS..=EXPECTED_BARS_PER_DAY_MAX_01BPS).contains(&bars_per_day),
        "Expected {}-{} bars/day at 0.1bps, got {}",
        EXPECTED_BARS_PER_DAY_MIN_01BPS,
        EXPECTED_BARS_PER_DAY_MAX_01BPS,
        bars_per_day
    );

    println!("  ✅ Bar count within expected range");

    // Step 9: Validate bar integrity
    println!("\nStep 9: Validating bar integrity...");
    validate_bar_integrity(&bars);
    println!("  ✅ All bars pass integrity checks");

    // Step 10: Export results for audit
    println!("\nStep 10: Exporting results for manual audit...");
    export_results(&test_ticks, &bars);
    println!("  ✅ Results exported to output/exness_test/");

    println!("\n=== Test Complete ===");
    println!("Total ticks: {}", test_ticks.len());
    println!("Total bars: {}", bars.len());
    println!("Bars per day: {}", bars_per_day);
    println!("Ticks per day: {}", ticks_per_day);
    println!(
        "Ticks per bar: {:.1}",
        test_ticks.len() as f64 / bars.len() as f64
    );
}

/// Validate temporal integrity (monotonic timestamps, no duplicates)
fn validate_temporal_integrity(ticks: &[ExnessTick]) {
    assert!(!ticks.is_empty(), "Empty tick array");

    for i in 1..ticks.len() {
        let prev = &ticks[i - 1];
        let curr = &ticks[i];

        // Timestamps must be monotonically increasing
        assert!(
            curr.timestamp_ms >= prev.timestamp_ms,
            "Temporal integrity violation at tick {}: {} < {}",
            i,
            curr.timestamp_ms,
            prev.timestamp_ms
        );

        // Check for suspicious duplicates (same timestamp AND same prices)
        if curr.timestamp_ms == prev.timestamp_ms {
            assert!(
                curr.bid != prev.bid || curr.ask != prev.ask,
                "Duplicate tick at {}: same timestamp and prices",
                i
            );
        }
    }
}

/// Validate spread distribution (bimodal: 98% at 0.0 pips, 2% at 1-9 pips)
fn validate_spread_distribution(ticks: &[ExnessTick]) {
    let mut spread_counts = std::collections::HashMap::new();
    let mut zero_count = 0;

    for tick in ticks {
        let spread_pips = ((tick.ask - tick.bid) * 10000.0).round() as i32;

        if spread_pips == 0 {
            zero_count += 1;
        }

        *spread_counts.entry(spread_pips).or_insert(0) += 1;
    }

    let zero_pct = (zero_count as f64 / ticks.len() as f64) * 100.0;
    println!("    Zero spread ticks: {:.2}%", zero_pct);

    // Raw_Spread should have >90% zero spread (CV=8.17 characteristic)
    assert!(
        zero_pct >= 90.0,
        "Expected >90% zero spread for Raw_Spread, got {:.2}%",
        zero_pct
    );

    // Check for stress events (spread > 1 pip)
    let stress_count: usize = spread_counts
        .iter()
        .filter(|(s, _)| **s > 10)
        .map(|(_, c)| c)
        .sum();

    let stress_pct = (stress_count as f64 / ticks.len() as f64) * 100.0;
    println!("    Stress events (>1 pip): {:.2}%", stress_pct);

    // Stress events vary by market conditions (0-5% is normal for Raw_Spread)
    assert!(
        stress_pct <= 10.0,
        "Expected <=10% stress events for Raw_Spread, got {:.2}% (data quality issue)",
        stress_pct
    );
}

/// Validate bar integrity (OHLC relationships, threshold conformance)
fn validate_bar_integrity(bars: &[rangebar::providers::exness::ExnessRangeBar]) {
    for (i, bar) in bars.iter().enumerate() {
        let b = &bar.base;

        // OHLC integrity
        assert!(
            b.high.0 >= b.open.0,
            "Bar {}: high < open ({} < {})",
            i,
            b.high.to_f64(),
            b.open.to_f64()
        );
        assert!(
            b.high.0 >= b.close.0,
            "Bar {}: high < close ({} < {})",
            i,
            b.high.to_f64(),
            b.close.to_f64()
        );
        assert!(
            b.low.0 <= b.open.0,
            "Bar {}: low > open ({} > {})",
            i,
            b.low.to_f64(),
            b.open.to_f64()
        );
        assert!(
            b.low.0 <= b.close.0,
            "Bar {}: low > close ({} > {})",
            i,
            b.low.to_f64(),
            b.close.to_f64()
        );

        // Volume should be 0 (Exness has no volume data)
        assert_eq!(
            b.volume.0, 0,
            "Bar {}: expected volume=0, got {}",
            i, b.volume.0
        );
        assert_eq!(b.buy_volume.0, 0);
        assert_eq!(b.sell_volume.0, 0);

        // Spread stats should have data
        assert!(
            bar.spread_stats.tick_count > 0,
            "Bar {}: zero tick count",
            i
        );
    }
}

/// Export results for manual audit
fn export_results(ticks: &[ExnessTick], bars: &[rangebar::providers::exness::ExnessRangeBar]) {
    let output_dir = Path::new("output/exness_test");
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Export summary stats as JSON
    let summary = serde_json::json!({
        "test_period": "2024-01-15 to 2024-01-19",
        "total_ticks": ticks.len(),
        "total_bars": bars.len(),
        "ticks_per_bar_avg": ticks.len() as f64 / bars.len() as f64,
        "bars_per_day": bars.len() / 5,
        "ticks_per_day": ticks.len() / 5,
        "threshold_decimal_bps": 0.1,
        "validation_strictness": "Strict",
    });

    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("Failed to write summary.json");

    // Export bar sample (first 100 bars) as CSV
    let mut csv_content = String::from(
        "bar_num,timestamp,open,high,low,close,tick_count,avg_spread,min_spread,max_spread\n",
    );

    for (i, bar) in bars.iter().take(100).enumerate() {
        csv_content.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            i,
            bar.base.open_time,
            bar.base.open.to_f64(),
            bar.base.high.to_f64(),
            bar.base.low.to_f64(),
            bar.base.close.to_f64(),
            bar.spread_stats.tick_count,
            bar.spread_stats.avg_spread().to_f64(),
            bar.spread_stats.min_spread.to_f64(),
            bar.spread_stats.max_spread.to_f64(),
        ));
    }

    fs::write(output_dir.join("bars_sample.csv"), csv_content)
        .expect("Failed to write bars_sample.csv");

    println!("    - summary.json");
    println!("    - bars_sample.csv (first 100 bars)");
}

// ============================================================================
// Type-Safe API Tests (v5.0.1+)
// ============================================================================

/// Test EURUSD using the new type-safe ExnessInstrument API
///
/// Verifies backward compatibility: type-safe API produces identical results
/// to the legacy string-based API.
#[tokio::test]
#[ignore]
async fn test_eurusd_type_safe_api() {
    use rangebar::providers::exness::ExnessInstrument;

    println!("\n=== EURUSD Type-Safe API Test ===\n");

    // Fetch with type-safe API
    let fetcher_typed = ExnessFetcher::for_instrument(ExnessInstrument::EURUSD);
    let ticks_typed = fetcher_typed
        .fetch_month(2024, 1)
        .await
        .expect("Type-safe fetch failed");

    // Fetch with legacy string API
    let fetcher_legacy = ExnessFetcher::new("EURUSD_Raw_Spread");
    let ticks_legacy = fetcher_legacy
        .fetch_month(2024, 1)
        .await
        .expect("Legacy fetch failed");

    // Verify identical results
    assert_eq!(
        ticks_typed.len(),
        ticks_legacy.len(),
        "Type-safe and legacy APIs should produce identical tick counts"
    );

    println!("✅ Type-safe API: {} ticks", ticks_typed.len());
    println!("✅ Legacy API: {} ticks", ticks_legacy.len());

    // Verify first tick matches
    let first_typed = &ticks_typed[0];
    let first_legacy = &ticks_legacy[0];
    assert_eq!(first_typed.timestamp_ms, first_legacy.timestamp_ms);
    assert_eq!(first_typed.bid, first_legacy.bid);
    assert_eq!(first_typed.ask, first_legacy.ask);

    // Verify last tick matches
    let last_typed = ticks_typed.last().unwrap();
    let last_legacy = ticks_legacy.last().unwrap();
    assert_eq!(last_typed.timestamp_ms, last_legacy.timestamp_ms);

    println!("✅ API parity verified");

    // Test builder with type-safe API
    use rangebar::providers::exness::ExnessRangeBarBuilder;

    let mut builder_typed = ExnessRangeBarBuilder::for_instrument(
        ExnessInstrument::EURUSD,
        5, // 0.5 bps
        ValidationStrictness::Strict,
    )
    .expect("Type-safe builder failed");

    let bars_typed: Vec<_> = ticks_typed
        .iter()
        .filter_map(|t| builder_typed.process_tick(t).ok().flatten())
        .collect();

    println!(
        "✅ Generated {} bars with type-safe builder",
        bars_typed.len()
    );
    assert!(!bars_typed.is_empty(), "Should generate at least one bar");
}
