// Detailed Permutation Entropy SIMD microbenchmark
// Phase 5B+.1: Measure entropy performance for Task #129
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
// Task #129: Implement Permutation Entropy SIMD with wide crate

use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

fn main() {
    println!("=== Entropy SIMD Microbenchmark ===\n");
    println!("Task #129: Baseline measurement for Permutation Entropy\n");

    for lookback_size in vec![100, 250, 500] {
        bench_entropy_detail(lookback_size);
    }

    println!("\nAnalysis:");
    println!("- Current implementation uses 8x loop unrolling (ILP)");
    println!("- Task #129 target: True SIMD with wide::u8x16 for ordinal pattern batch compute");
    println!("- Expected improvement: 3-5x speedup (from current 400µs to 80-130µs @ 500-trade)");
}

fn bench_entropy_detail(lookback_size: usize) {
    let separator = "=".repeat(60);
    println!("\n{}", separator);
    println!("Lookback: {} trades", lookback_size);
    println!("{}", separator);

    // Generate synthetic trades
    let num_trades = lookback_size * 2;
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

    // Measure Tier 3 only (entropy dominant)
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: false,
        compute_tier3: true,
        ..Default::default()
    };
    let mut history = TradeHistory::new(config);
    for trade in &trades {
        history.push(trade);
    }

    // Warm up
    for _ in 0..10 {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark (higher iterations for accurate entropy timing)
    let iterations = 200;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as f64 / iterations as f64;

    println!("\nTier 3 Total: {:.3}µs", avg_micros);
    println!("Composition (estimated from Phase 5A profiling):");

    match lookback_size {
        100 => {
            // From profiling: Entropy ~60%, Hurst ~40%
            let entropy_estimate = avg_micros * 0.6;
            let hurst_estimate = avg_micros * 0.4;
            println!("  - Entropy (60%): ~{:.2}µs", entropy_estimate);
            println!("  - Hurst (40%):   ~{:.2}µs", hurst_estimate);
            println!("\n  Task #129 Target: {:.2}µs → {:.2}µs (3-5x speedup)", entropy_estimate, entropy_estimate / 5.0);
        }
        250 => {
            let entropy_estimate = avg_micros * 0.6;
            let hurst_estimate = avg_micros * 0.4;
            println!("  - Entropy (60%): ~{:.2}µs", entropy_estimate);
            println!("  - Hurst (40%):   ~{:.2}µs", hurst_estimate);
            println!("\n  Task #129 Target: {:.2}µs → {:.2}µs (3-5x speedup)", entropy_estimate, entropy_estimate / 5.0);
        }
        500 => {
            let entropy_estimate = avg_micros * 0.6;
            let hurst_estimate = avg_micros * 0.4;
            println!("  - Entropy (60%): ~{:.2}µs", entropy_estimate);
            println!("  - Hurst (40%):   ~{:.2}µs", hurst_estimate);
            println!("\n  Task #129 Target: {:.2}µs → {:.2}µs (3-5x speedup)", entropy_estimate, entropy_estimate / 5.0);
            println!("\n  Overall Impact:");
            println!("    - Current Tier 3: {:.2}µs", avg_micros);
            println!("    - With entropy SIMD (3-5x): {:.2}-{:.2}µs",
                avg_micros - entropy_estimate + (entropy_estimate/5.0),
                avg_micros - entropy_estimate + (entropy_estimate/3.0));
            println!("    - Expected reduction: {:.0}-{:.0}µs ({:.0}-{:.0}% improvement)",
                entropy_estimate * 0.8, entropy_estimate * 0.67,
                (entropy_estimate * 0.8 / avg_micros) * 100.0,
                (entropy_estimate * 0.67 / avg_micros) * 100.0);
        }
        _ => {}
    }

    println!("\nImplementation Notes:");
    println!("- Current: 8x loop unroll (ILP optimization)");
    println!("- Task #129: Add true SIMD with wide::u8x16");
    println!("- Strategy: Batch ordinal pattern computation in parallel");
}
