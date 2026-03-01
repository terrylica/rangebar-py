//! Tier-by-Tier Performance Breakdown on Real Binance Data
//!
//! Task #140 (Phase 7b): Identify which feature contributes to performance degradation
//! on large lookback windows (500+ trades) using real data patterns.
//!
//! Measures performance with different Tier configurations to isolate bottlenecks.

#[cfg(feature = "test-utils")]
fn main() {
    use rangebar_core::test_data_loader::load_btcusdt_test_data;
    use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
    use std::time::Instant;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Tier-by-Tier Performance Breakdown (Real Data)            ║");
    println!("║  Task #140: Identify feature bottlenecks                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let trades = load_btcusdt_test_data().expect("Failed to load test data");
    println!("Loaded {} real Binance BTCUSDT trades\n", trades.len());

    // Test at large lookback where degradation occurs
    let lookback_size = 500;

    println!("Performance Breakdown @ {} trades lookback:", lookback_size);
    println!("{}", "─".repeat(70));

    // Measure Tier 1 only
    let t1_only = measure_tier_config(&trades, lookback_size, true, false, false);
    println!("Tier 1 only (core features):        {:.2} µs", t1_only);

    // Measure Tier 1 + 2
    let t1_t2 = measure_tier_config(&trades, lookback_size, true, true, false);
    let t2_cost = t1_t2 - t1_only;
    println!("Tier 1 + 2 (+ statistical):          {:.2} µs (Tier 2 cost: {:.2} µs)", t1_t2, t2_cost);

    // Measure Tier 1 + 2 + 3
    let t1_t2_t3 = measure_tier_config(&trades, lookback_size, true, true, true);
    let t3_cost = t1_t2_t3 - t1_t2;
    println!("Tier 1 + 2 + 3 (full):               {:.2} µs (Tier 3 cost: {:.2} µs)", t1_t2_t3, t3_cost);

    println!("{}", "─".repeat(70));
    println!("\nPercentage Breakdown:");
    println!("  Tier 1: {:.1}%", (t1_only / t1_t2_t3) * 100.0);
    println!("  Tier 2: {:.1}%", (t2_cost / t1_t2_t3) * 100.0);
    println!("  Tier 3: {:.1}%", (t3_cost / t1_t2_t3) * 100.0);

    println!("\n{}", "═".repeat(70));
    println!("\nAnalysis:");

    if t3_cost > (t1_t2_t3 * 0.8) {
        println!("⚠️  Tier 3 dominates: {:.1}% of total time", (t3_cost / t1_t2_t3) * 100.0);
        println!("   This explains the large lookback slowdown on real data.");
        println!("   Recommendation: Profile entropy and Hurst separately.");
    } else if t2_cost > (t1_t2_t3 * 0.3) {
        println!("⚠️  Tier 2 is significant: {:.1}% of total time", (t2_cost / t1_t2_t3) * 100.0);
        println!("   Statistical features (burstiness, Kyle lambda) may need optimization.");
    } else {
        println!("✅ Tiers 2-3 are proportional. Investigate real data characteristics.");
    }

    println!("\n{}", "═".repeat(70));
    println!("\nProduction vs Synthetic Comparison:");
    println!("  Synthetic (cumulative benchmark): 671.16 µs @ 500-trade");
    println!("  Real data (production validation): 1172.62 µs @ 500-trade");
    println!("  Difference: {:.1}% SLOWER on real data", ((1172.62 - 671.16) / 671.16 * 100.0));
    println!("\nLikely Causes:");
    println!("  1. Real data has autocorrelated prices (entropy is higher/more complex)");
    println!("  2. Volatility clustering triggers more Hurst iterations");
    println!("  3. Cache misses on realistic trade patterns");
    println!("  4. SIMD less effective on non-aligned real data distribution");
}

#[cfg(feature = "test-utils")]
fn measure_tier_config(
    trades: &[rangebar_core::types::AggTrade],
    lookback_size: usize,
    tier1: bool,
    tier2: bool,
    tier3: bool,
) -> f64 {
    use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
    use std::time::Instant;

    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: tier2,
        compute_tier3: tier3,
        ..Default::default()
    };
    let mut history = TradeHistory::new(config);

    // Populate with real trades
    for trade in trades {
        history.push(trade);
    }

    // Warm up
    for _ in 0..5 {
        let _ = history.compute_features(trades.last().map(|t| t.timestamp).unwrap_or(0));
    }

    // Measure
    let num_iterations = 20;
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = history.compute_features(trades.last().map(|t| t.timestamp).unwrap_or(0));
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / num_iterations as f64
}

#[cfg(not(feature = "test-utils"))]
fn main() {
    eprintln!("This benchmark requires the 'test-utils' feature");
    eprintln!("Run: cargo bench --bench tier_breakdown_real_data --features test-utils");
}
