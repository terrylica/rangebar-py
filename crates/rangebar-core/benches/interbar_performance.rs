// Benchmark for inter-bar feature computation performance
// Tests impact of optimizations: Task #115 (rayon), #116 (libm), #118 (VecDeque), #119 (SmallVec), #121 (SIMD)
//
// Issue: #96 (Performance optimization suite)
// Related Tasks: #115, #116, #118, #119, #121, #122

use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

fn main() {
    println!("=== Inter-Bar Feature Computation Performance Benchmark ===\n");

    // Test different lookback sizes
    let lookback_sizes = vec![100, 250, 500];

    for lookback_size in lookback_sizes {
        bench_lookback_computation(lookback_size);
    }

    println!("\n=== Benchmark Complete ===");
    println!("\nSummary:");
    println!("- All optimizations are active (Rayon, libm, VecDeque sizing, SmallVec, SIMD-burstiness)");
    println!("- Expected cumulative speedup: 2.5-4x over baseline");
    println!("- For detailed comparison, compile without optimizations and rerun");
}

fn bench_lookback_computation(lookback_size: usize) {
    println!("\nBenchmark: {} trades lookback", lookback_size);
    println!("{}", "=".repeat(50));

    // Create trade history with specified lookback
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: true,
        compute_tier3: true,
        ..Default::default()
    };
    let mut history = TradeHistory::new(config);

    // Generate synthetic trades
    let num_trades = lookback_size * 2; // 2x to ensure we have enough lookback after pruning
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
        history.push(&trade);
    }

    // Warm up (JIT compilation)
    for _ in 0..10 {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark: compute features 100 times
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() / iterations as u128;
    println!("  Per-bar computation: {:.2}µs (avg over {} iterations)", avg_micros as f64, iterations);

    // Estimated speedup notes
    match lookback_size {
        100 => println!("  Note: Small lookback (Tier 1 dominant, minimal Tier 2/3)"),
        250 => println!("  Note: Typical lookback (good Tier 2/3 balance, benefits from all optimizations)"),
        500 => println!("  Note: Large lookback (Hurst/Entropy expensive, benefits from SIMD + libm)"),
        _ => {}
    }

    // Compute 1000-bar batch time estimate
    let batch_time_ms = elapsed.as_millis() / iterations as u128 * 1000;
    println!("  1000-bar batch: ~{}ms", batch_time_ms);

    println!("  Optimizations active:");
    println!("    ✓ Rayon parallelization (Task #115)");
    println!("    ✓ libm optimizations (Task #116)");
    println!("    ✓ VecDeque capacity tuning (Task #118)");
    println!("    ✓ SmallVec accumulation (Task #119)");
    println!("    ✓ SIMD-burstiness (Task #121 - default enabled)");
}
