//! XAUUSD Range Bar Integration Test with Artifact Generation
//!
//! Produces validation artifacts for sensibility analysis:
//! 1. NDJSON bar data (output/validation/xauusd_*.jsonl)
//! 2. Markdown status gate report (output/validation/xauusd_report.md)
//! 3. JSON validation report (output/validation/xauusd_validation.json)
//! 4. Insta snapshots for regression detection
//!
//! XAUUSD characteristics:
//! - Price range: $1500-$3000 USD
//! - Higher volatility than forex pairs
//! - 166.8M validated ticks in ~/eon/exness-data-preprocess

#![cfg(feature = "providers")]

mod common;

use common::{
    validate_bar_generation, validate_bar_integrity, validate_price_range,
    validate_spread_distribution, validate_temporal_integrity, BarRecord, ValidationReport,
    ValidationSummary,
};
use rangebar::providers::exness::{
    ExnessFetcher, ExnessInstrument, ExnessRangeBarBuilder, ValidationStrictness,
};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// XAUUSD price range validation (Gold in USD)
const XAUUSD_PRICE_MIN: f64 = 1500.0;
const XAUUSD_PRICE_MAX: f64 = 3000.0;

/// Minimum tight spread percentage for Raw_Spread validation
/// XAUUSD has consistent ~$0.06 spreads (NOT zero like forex pairs)
/// 99.6% of spreads are < $0.10, so 80% threshold is conservative
const XAUUSD_MIN_TIGHT_SPREAD_PCT: f64 = 80.0;

/// XAUUSD spread tolerance for "tight spread" validation
/// XAUUSD Raw_Spread has consistent ~$0.06 spreads (not zero like forex).
/// Use $0.10 tolerance to capture 99.6% of ticks as "tight spread".
const XAUUSD_SPREAD_TOLERANCE: f64 = 0.10;

#[tokio::test]
#[ignore] // Only run with --ignored (fetches real data)
async fn test_xauusd_end_to_end_with_artifacts() {
    println!("\n=== XAUUSD Integration Test with Artifacts ===\n");

    let output_dir = Path::new("output/validation");
    fs::create_dir_all(output_dir).expect("Create output dir");

    let mut report = ValidationReport::new("XAUUSD");

    // === Gate 1: Data Fetch ===
    println!("Gate 1: Fetching XAUUSD data...");
    let fetcher = ExnessFetcher::for_instrument(ExnessInstrument::XAUUSD);
    let all_ticks = fetcher
        .fetch_month(2024, 6)
        .await
        .expect("XAUUSD fetch failed");

    report.add_gate(common::ValidationGate::pass(
        "Data Fetch",
        format!("{} ticks fetched", all_ticks.len()),
    ));
    println!("  ✅ Fetched {} ticks", all_ticks.len());

    // === Gate 2: Temporal Integrity ===
    println!("\nGate 2: Validating temporal integrity...");
    validate_temporal_integrity(&all_ticks, &mut report);

    // === Gate 3: Price Range ===
    println!("\nGate 3: Validating price range...");
    validate_price_range(
        &all_ticks,
        XAUUSD_PRICE_MIN,
        XAUUSD_PRICE_MAX,
        "XAUUSD",
        &mut report,
    );

    // === Gate 4: Spread Distribution ===
    println!("\nGate 4: Validating spread distribution...");
    let tight_spread_pct = validate_spread_distribution(
        &all_ticks,
        XAUUSD_MIN_TIGHT_SPREAD_PCT,
        XAUUSD_SPREAD_TOLERANCE,
        &mut report,
    );
    println!("  Tight spread: {:.1}%", tight_spread_pct);

    // === Gate 5: Bar Generation (5 dbps threshold) ===
    println!("\nGate 5: Generating range bars (5 dbps)...");
    let threshold = 5; // 5 dbps = 0.005%
    let mut builder = ExnessRangeBarBuilder::for_instrument(
        ExnessInstrument::XAUUSD,
        threshold,
        ValidationStrictness::Strict,
    )
    .expect("Builder creation failed");

    let mut bars = Vec::new();
    for tick in &all_ticks {
        if let Ok(Some(bar)) = builder.process_tick(tick) {
            bars.push(bar);
        }
    }

    // Get incomplete bar at end
    if let Some(incomplete) = builder.get_incomplete_bar() {
        bars.push(incomplete);
    }

    validate_bar_generation(&bars, &mut report);
    println!("  ✅ Generated {} bars", bars.len());

    // === Gate 6: OHLCV Integrity ===
    println!("\nGate 6: Validating OHLCV integrity...");
    validate_bar_integrity(&bars, &mut report);

    // === ARTIFACT GENERATION ===
    println!("\n=== Generating Artifacts ===");

    // Artifact 1: NDJSON bar data
    let jsonl_path = output_dir.join("xauusd_05bps_202406.jsonl");
    let file = File::create(&jsonl_path).expect("Create NDJSON file");
    let mut writer = BufWriter::new(file);
    for (i, bar) in bars.iter().enumerate() {
        let record = BarRecord::from_bar(i, bar);
        serde_json::to_writer(&mut writer, &record).unwrap();
        writeln!(writer).unwrap();
    }
    writer.flush().unwrap();
    println!("  ✅ NDJSON: {}", jsonl_path.display());

    // Artifact 2: Insta snapshot for validation summary
    let summary = ValidationSummary {
        instrument: "XAUUSD".to_string(),
        ticks: all_ticks.len(),
        bars: bars.len(),
        zero_spread_pct: tight_spread_pct, // Named for API compat, actually tight spread for XAUUSD
        all_gates_passed: report.all_passed,
    };
    insta::assert_json_snapshot!("xauusd_validation_summary", summary);

    // Artifact 3: Finalize report (writes JSON + Markdown, fails if any gate failed)
    report.finalize(output_dir);

    println!("\n=== Test Complete ===");
    println!("Total ticks: {}", all_ticks.len());
    println!("Total bars: {}", bars.len());
    println!(
        "Ticks per bar: {:.1}",
        all_ticks.len() as f64 / bars.len() as f64
    );
}

#[tokio::test]
#[ignore]
async fn test_xauusd_multi_threshold_scaling() {
    println!("\n=== XAUUSD Multi-Threshold Scaling Test ===\n");

    // Fetch data
    let fetcher = ExnessFetcher::for_instrument(ExnessInstrument::XAUUSD);
    let ticks = fetcher
        .fetch_month(2024, 6)
        .await
        .expect("XAUUSD fetch failed");

    println!("Fetched {} ticks", ticks.len());

    // Test multiple thresholds
    let thresholds = [2, 5, 10, 50, 100]; // 2, 5, 10, 50, 100 dbps
    let mut bar_counts = Vec::new();

    for &threshold in &thresholds {
        let mut builder = ExnessRangeBarBuilder::for_instrument(
            ExnessInstrument::XAUUSD,
            threshold,
            ValidationStrictness::Strict,
        )
        .expect("Builder failed");

        let count = ticks
            .iter()
            .filter_map(|t| builder.process_tick(t).ok().flatten())
            .count();
        bar_counts.push(count);

        println!("  {} dbps: {} bars", threshold, count);
    }

    // Verify threshold scaling: narrower threshold → more bars
    for i in 1..bar_counts.len() {
        assert!(
            bar_counts[i - 1] >= bar_counts[i],
            "Threshold scaling violated: {} dbps ({} bars) should produce >= bars than {} dbps ({} bars)",
            thresholds[i - 1],
            bar_counts[i - 1],
            thresholds[i],
            bar_counts[i]
        );
    }

    println!("\n✅ Threshold scaling invariant verified");
}

#[tokio::test]
#[ignore]
async fn test_xauusd_type_safe_api_parity() {
    println!("\n=== XAUUSD Type-Safe API Parity Test ===\n");

    // Fetch with type-safe API
    let fetcher_typed = ExnessFetcher::for_instrument(ExnessInstrument::XAUUSD);
    let ticks_typed = fetcher_typed
        .fetch_month(2024, 6)
        .await
        .expect("Type-safe fetch failed");

    // Fetch with legacy string API
    let fetcher_legacy = ExnessFetcher::new("XAUUSD_Raw_Spread");
    let ticks_legacy = fetcher_legacy
        .fetch_month(2024, 6)
        .await
        .expect("Legacy fetch failed");

    // Verify parity
    assert_eq!(
        ticks_typed.len(),
        ticks_legacy.len(),
        "Type-safe and legacy APIs should produce identical results"
    );

    // Verify first and last tick match
    assert_eq!(
        ticks_typed.first().unwrap().timestamp_ms,
        ticks_legacy.first().unwrap().timestamp_ms,
        "First tick timestamps should match"
    );
    assert_eq!(
        ticks_typed.last().unwrap().timestamp_ms,
        ticks_legacy.last().unwrap().timestamp_ms,
        "Last tick timestamps should match"
    );

    println!("✅ API parity verified: {} ticks", ticks_typed.len());
}
