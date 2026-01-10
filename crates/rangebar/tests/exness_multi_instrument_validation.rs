//! Multi-instrument validation test
//!
//! Verifies all 10 Exness instruments follow consistent patterns:
//! 1. Data fetchable from Exness API
//! 2. Temporal integrity maintained
//! 3. Spread distribution matches Raw_Spread characteristics:
//!    - Forex pairs: >80% zero spread (bid == ask, tolerance 0.000001)
//!    - XAUUSD: >80% tight spread (~$0.06 avg, tolerance 0.10)
//! 4. Range bar generation produces valid OHLCV
//! 5. Threshold scaling behaves correctly (more bars at lower thresholds)

#![cfg(feature = "providers")]

use rangebar::providers::exness::{
    ExnessFetcher, ExnessInstrument, ExnessRangeBarBuilder, ExnessTick, ValidationStrictness,
};
use std::collections::HashMap;

/// Validate all 10 instruments produce consistent results
#[tokio::test]
#[ignore] // Full validation - run with cargo test --ignored
async fn test_all_10_instruments_consistency() {
    println!("\n=== Multi-Instrument Consistency Test ===\n");

    let mut results: HashMap<&str, InstrumentResult> = HashMap::new();

    for instrument in ExnessInstrument::all() {
        println!("\n--- Testing {} ---", instrument);

        let fetcher = ExnessFetcher::for_instrument(*instrument);
        let ticks = match fetcher.fetch_month(2024, 6).await {
            Ok(t) => t,
            Err(e) => {
                panic!(
                    "\n=== FETCH FAILED ===\n\
                    Instrument: {}\n\
                    Error: {:?}\n",
                    instrument, e
                );
            }
        };

        println!("  Fetched {} ticks", ticks.len());

        // Validate temporal ordering
        validate_temporal_integrity(&ticks, instrument.symbol());

        // Validate price range
        let (price_min, price_max) = instrument.price_range();
        validate_price_range(&ticks, price_min, price_max, instrument.symbol());

        // Validate spread distribution (Raw_Spread characteristic)
        let zero_pct = calculate_zero_spread_pct(&ticks, instrument.spread_tolerance());
        println!("  Zero spread: {:.1}%", zero_pct);
        assert!(
            zero_pct >= 80.0,
            "{}: Only {:.1}% zero spread (expected >80%)",
            instrument,
            zero_pct
        );

        // Multi-threshold bar generation
        let mut bar_counts = Vec::new();
        for threshold in [2, 5, 10] {
            let mut builder = ExnessRangeBarBuilder::for_instrument(
                *instrument,
                threshold,
                ValidationStrictness::Strict,
            )
            .expect("Builder failed");

            let count = ticks
                .iter()
                .filter_map(|t| builder.process_tick(t).ok().flatten())
                .count();
            bar_counts.push(count);
        }

        // Threshold scaling: narrower threshold → more bars
        assert!(
            bar_counts[0] >= bar_counts[1],
            "{}: 0.2bps ({} bars) should produce >= bars than 0.5bps ({} bars)",
            instrument,
            bar_counts[0],
            bar_counts[1]
        );
        assert!(
            bar_counts[1] >= bar_counts[2],
            "{}: 0.5bps ({} bars) should produce >= bars than 1.0bps ({} bars)",
            instrument,
            bar_counts[1],
            bar_counts[2]
        );

        results.insert(
            instrument.symbol(),
            InstrumentResult {
                ticks: ticks.len(),
                zero_spread_pct: zero_pct,
                bar_counts,
            },
        );

        println!(
            "  Bars: {} / {} / {} (0.2/0.5/1.0 bps)",
            results[instrument.symbol()].bar_counts[0],
            results[instrument.symbol()].bar_counts[1],
            results[instrument.symbol()].bar_counts[2]
        );
        println!("  ✅ {} validated", instrument);
    }

    // Summary report
    println!("\n=== Multi-Instrument Summary ===\n");
    println!(
        "{:<10} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "Symbol", "Ticks", "Zero%", "0.2bps", "0.5bps", "1.0bps"
    );
    println!("{}", "-".repeat(65));

    for instrument in ExnessInstrument::all() {
        let r = &results[instrument.symbol()];
        println!(
            "{:<10} {:>12} {:>9.1}% {:>10} {:>10} {:>10}",
            instrument.symbol(),
            r.ticks,
            r.zero_spread_pct,
            r.bar_counts[0],
            r.bar_counts[1],
            r.bar_counts[2]
        );
    }

    println!("\n✅ All 10 instruments validated successfully");
}

/// Test JPY pairs have correct pip value handling
#[tokio::test]
#[ignore]
async fn test_jpy_pairs_price_scale() {
    println!("\n=== JPY Pairs Price Scale Test ===\n");

    let jpy_pairs = [
        ExnessInstrument::USDJPY,
        ExnessInstrument::EURJPY,
        ExnessInstrument::GBPJPY,
    ];

    for instrument in jpy_pairs {
        assert!(
            instrument.is_jpy_pair(),
            "{} should be identified as JPY pair",
            instrument
        );

        let fetcher = ExnessFetcher::for_instrument(instrument);
        let ticks = fetcher
            .fetch_month(2024, 6)
            .await
            .expect(&format!("{} fetch failed", instrument));

        // Verify prices are in expected JPY range (100-230)
        let (min, max) = instrument.price_range();
        for tick in &ticks {
            assert!(
                tick.bid >= min && tick.bid <= max,
                "{}: price {} outside expected range [{}, {}]",
                instrument,
                tick.bid,
                min,
                max
            );
        }

        println!("  ✅ {} prices in range [{}, {}]", instrument, min, max);
    }
}

/// Test cross pairs behave correctly
#[tokio::test]
#[ignore]
async fn test_cross_pairs() {
    println!("\n=== Cross Pairs Test ===\n");

    let cross_pairs = [ExnessInstrument::EURGBP];

    for instrument in cross_pairs {
        let fetcher = ExnessFetcher::for_instrument(instrument);
        let ticks = fetcher
            .fetch_month(2024, 6)
            .await
            .expect(&format!("{} fetch failed", instrument));

        // Verify temporal integrity
        validate_temporal_integrity(&ticks, instrument.symbol());

        // Verify price range (EURGBP typically 0.75-0.95)
        let (min, max) = instrument.price_range();
        validate_price_range(&ticks, min, max, instrument.symbol());

        println!("  ✅ {} validated ({} ticks)", instrument, ticks.len());
    }
}

// ============================================================================
// Helper Structures and Functions
// ============================================================================

struct InstrumentResult {
    ticks: usize,
    zero_spread_pct: f64,
    bar_counts: Vec<usize>,
}

fn validate_temporal_integrity(ticks: &[ExnessTick], symbol: &str) {
    for i in 1..ticks.len() {
        assert!(
            ticks[i].timestamp_ms >= ticks[i - 1].timestamp_ms,
            "{}: Temporal violation at tick {}: {} < {}",
            symbol,
            i,
            ticks[i].timestamp_ms,
            ticks[i - 1].timestamp_ms
        );
    }
}

fn validate_price_range(ticks: &[ExnessTick], min: f64, max: f64, symbol: &str) {
    for tick in ticks {
        assert!(
            tick.bid >= min && tick.bid <= max,
            "{}: price {} outside valid range [{}, {}]",
            symbol,
            tick.bid,
            min,
            max
        );
    }
}

fn calculate_zero_spread_pct(ticks: &[ExnessTick], spread_tolerance: f64) -> f64 {
    let zero_spreads = ticks
        .iter()
        .filter(|t| (t.ask - t.bid).abs() < spread_tolerance)
        .count();
    (zero_spreads as f64 / ticks.len() as f64) * 100.0
}
