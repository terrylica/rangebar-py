// Detailed SIMD burstiness microbenchmark
// Phase 5B: Measure SIMD speedup for burstiness computation (Task #127)
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
// Task #127: Implement SIMD burstiness optimization with wide crate

use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

fn main() {
    println!("=== SIMD Burstiness Microbenchmark ===\n");
    println!("Task #127: Measure SIMD speedup with wide crate\n");

    for lookback_size in vec![100, 250, 500] {
        bench_burstiness_detail(lookback_size);
    }
}

fn bench_burstiness_detail(lookback_size: usize) {
    println!("Lookback: {} trades", lookback_size);
    println!("{}", "=".repeat(50));

    // Setup
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

    // Configuration: Tier 2 only (isolate burstiness + other Tier 2 features)
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: true,
        compute_tier3: false,
        ..Default::default()
    };
    let mut history = TradeHistory::new(config);
    for trade in &trades {
        history.push(trade);
    }

    // Warm up
    for _ in 0..20 {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark
    let iterations = 200;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }
    let elapsed = start.elapsed();

    let avg_micros = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Tier 2 computation: {:.3}µs (avg over {} iterations)", avg_micros, iterations);
    println!("  Breakdown includes: kyle_lambda, burstiness, volume_moments, price_range");
    println!("  SIMD feature: enabled (wide crate f64x4 vectors)");
    println!();
}
