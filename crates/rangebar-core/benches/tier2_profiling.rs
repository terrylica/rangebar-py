// Benchmark for Tier 2 CPU contribution analysis
// Phase 5A: Profiling work to determine if SIMD burstiness optimization (Phase 5B) is justified
//
// Issue: #96 Task #126 (Extend profiling to measure Tier 2 CPU contribution)
// Goal: Measure CPU time spent in Tier 2 features (kyle_lambda, burstiness, volume moments)
// Success Criteria: If Tier 2 > 10% of total inter-bar time, Phase 5B SIMD is justified

use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

fn main() {
    println!("=== Phase 5A: Tier 2 CPU Contribution Profiling ===\n");
    println!("Goal: Determine if Tier 2 features exceed 10% of inter-bar computation time");
    println!("Success Criteria: Tier 2 CPU contribution >= 10% justifies Phase 5B (SIMD burstiness)\n");

    // Test different lookback sizes
    let lookback_sizes = vec![100, 250, 500];

    for lookback_size in lookback_sizes {
        profile_tier2_contribution(lookback_size);
    }

    println!("\n=== Profiling Complete ===");
    println!("\nInterpretation Guide:");
    println!("- If Tier 2 % > 10%: Phase 5B (SIMD burstiness) should be implemented");
    println!("- If Tier 2 % < 10%: Phase 5B deferred; focus on other bottlenecks");
    println!("- Tier 3 (Hurst/Entropy) should be significantly higher due to O(n²) complexity");
}

fn profile_tier2_contribution(lookback_size: usize) {
    let separator = "=".repeat(70);
    println!("\n{}", separator);
    println!("Profiling: {} trades lookback", lookback_size);
    println!("{}", separator);

    // === Setup: Generate synthetic trades ===
    let num_trades = lookback_size * 2; // 2x to ensure we have enough lookback after pruning
    let mut trades = Vec::with_capacity(num_trades);
    for i in 0..num_trades as i64 {
        let trade = AggTrade {
            agg_trade_id: i,
            price: FixedPoint(((100.0 + (i % 20) as f64) * 1e8) as i64),
            volume: FixedPoint(((10.0 + (i % 10) as f64) * 1e8) as i64),
            first_trade_id: i,
            last_trade_id: i,
            timestamp: i * 1000,
            is_buyer_maker: i % 2 == 0,
            is_best_match: Some(false),
        };
        trades.push(trade);
    }

    // === Measurement 1: Tier 1 only ===
    let config_tier1 = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: false,
        compute_tier3: false,
    };
    let mut history_tier1 = TradeHistory::new(config_tier1);
    for trade in &trades {
        history_tier1.push(trade);
    }

    // Warm up
    for _ in 0..10 {
        let _ = history_tier1.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark Tier 1
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = history_tier1.compute_features((num_trades as i64 - 1) * 1000);
    }
    let tier1_elapsed = start.elapsed();
    let tier1_micros = tier1_elapsed.as_micros() as f64 / iterations as f64;

    // === Measurement 2: Tier 1 + Tier 2 ===
    let config_tier12 = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: true,
        compute_tier3: false,
    };
    let mut history_tier12 = TradeHistory::new(config_tier12);
    for trade in &trades {
        history_tier12.push(trade);
    }

    // Warm up
    for _ in 0..10 {
        let _ = history_tier12.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark Tier 1+2
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = history_tier12.compute_features((num_trades as i64 - 1) * 1000);
    }
    let tier12_elapsed = start.elapsed();
    let tier12_micros = tier12_elapsed.as_micros() as f64 / iterations as f64;

    // === Measurement 3: Tier 1 + Tier 2 + Tier 3 ===
    let config_all = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: true,
        compute_tier3: true,
    };
    let mut history_all = TradeHistory::new(config_all);
    for trade in &trades {
        history_all.push(trade);
    }

    // Warm up
    for _ in 0..10 {
        let _ = history_all.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark all tiers
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = history_all.compute_features((num_trades as i64 - 1) * 1000);
    }
    let all_elapsed = start.elapsed();
    let all_micros = all_elapsed.as_micros() as f64 / iterations as f64;

    // === Calculate contributions ===
    let tier2_micros = tier12_micros - tier1_micros;
    let tier3_micros = all_micros - tier12_micros;

    let tier2_percent = if tier12_micros > 0.0 {
        (tier2_micros / tier12_micros) * 100.0
    } else {
        0.0
    };

    let tier3_percent = if all_micros > 0.0 {
        (tier3_micros / all_micros) * 100.0
    } else {
        0.0
    };

    // === Report results ===
    println!("\nTiming Results:");
    println!("  Tier 1 only:              {:>8.2}µs", tier1_micros);
    println!("  Tier 1 + 2:               {:>8.2}µs", tier12_micros);
    println!("  Tier 1 + 2 + 3 (all):     {:>8.2}µs", all_micros);

    println!("\nCPU Time Breakdown:");
    println!("  Tier 1 contribution:      {:>8.2}µs ({:>6.1}% of Tier 1+2)", tier1_micros, 100.0 - tier2_percent);
    println!("  Tier 2 contribution:      {:>8.2}µs ({:>6.1}% of Tier 1+2)", tier2_micros, tier2_percent);
    println!("  Tier 3 contribution:      {:>8.2}µs ({:>6.1}% of all tiers)", tier3_micros, tier3_percent);

    // === Phase 5B Justification ===
    println!("\nPhase 5B (SIMD Burstiness) Decision:");
    if tier2_percent >= 10.0 {
        println!("  ✓ TIER 2 > 10% THRESHOLD - Phase 5B IS JUSTIFIED");
        println!("    Tier 2 CPU: {:.1}% (threshold: 10%)", tier2_percent);
        println!("    Action: Implement SIMD burstiness optimization (expected 2-3x speedup)");
    } else {
        println!("  ✗ Tier 2 < 10% threshold - Phase 5B deferred");
        println!("    Tier 2 CPU: {:.1}% (threshold: 10%)", tier2_percent);
        println!("    Recommendation: Focus on other optimizations (e.g., Tier 3 SIMD)");
    }

    // === Additional Analysis ===
    println!("\nBreakdown by lookback:");
    match lookback_size {
        100 => println!("  Note: Small lookback - Tier 1 dominant, minimal Tier 2/3 overhead"),
        250 => println!("  Note: Typical lookback - balanced feature distribution"),
        500 => println!("  Note: Large lookback - Tier 3 (Hurst/Entropy) expensive"),
        _ => {}
    }

    // === Performance sanity checks ===
    println!("\nSanity Checks:");
    if tier2_micros < 0.0 {
        println!("  ⚠ WARNING: Tier 2 time is negative (measurement noise or overhead)");
    } else {
        println!("  ✓ Tier 2 time is positive");
    }

    if tier3_micros < tier2_micros {
        println!("  ⚠ WARNING: Tier 3 < Tier 2 (unexpected, Tier 3 should be expensive)");
    } else {
        println!("  ✓ Tier 3 >= Tier 2 (expected behavior)");
    }
}
