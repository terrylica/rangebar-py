//! Benchmark: Tier 1 Inter-Bar Feature Optimization
//!
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59
//! Task #153: SIMD optimization of core inter-bar features (OFI, VWAP, imbalance)
//! Measures current scalar performance and validates improvements.
//!
//! Run with:
//! ```
//! cargo bench --bench tier1_optimization
//! ```

use rangebar_core::interbar::{TradeHistory, InterBarConfig, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;

fn create_test_trades(count: usize) -> Vec<AggTrade> {
    (0..count)
        .map(|i| {
            let price = 50000.0 + (i as f64 * 0.5).sin() * 100.0;
            let volume = 1000.0 + (i as f64 * 0.3).cos() * 500.0;
            let is_buyer_maker = i % 3 == 0; // Alternating buy/sell pressure

            // Convert f64 to FixedPoint by multiplying by 1e8 and truncating
            let price_fixed = FixedPoint((price * 1e8) as i64);
            let volume_fixed = FixedPoint((volume * 1e8) as i64);

            AggTrade {
                agg_trade_id: i as i64,
                timestamp: 1000000 + (i as i64 * 100),
                price: price_fixed,
                volume: volume_fixed,
                first_trade_id: (i * 10) as i64,
                last_trade_id: (i * 10 + 9) as i64,
                is_buyer_maker,
                is_best_match: Some(true),
            }
        })
        .collect()
}

fn benchmark_tier1_small_window() {
    println!("\n=== BENCHMARK: Tier 1 Features (50 trades) ===");

    let mut history = TradeHistory::new(InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(50),
        compute_tier2: false,
        compute_tier3: false,
    });

    let trades = create_test_trades(50);
    for trade in &trades {
        history.push(trade);
    }

    // Warmup
    for _ in 0..100 {
        let _ = history.compute_features(2000000);
    }

    // Benchmark: 10,000 iterations
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let features = history.compute_features(2000000);
        // Consume features to prevent optimization
        let _ = (
            features.lookback_ofi,
            features.lookback_vwap,
            features.lookback_count_imbalance,
        );
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() / 10000;
    println!("50-trade window:");
    println!("  Time per call: {} ns", per_call_ns);
    println!("  Throughput: {:.1} M calls/sec", 1_000_000_000.0 / per_call_ns as f64);
    println!("  Time per trade: {:.2} ns", per_call_ns as f64 / 50.0);
}

fn benchmark_tier1_medium_window() {
    println!("\n=== BENCHMARK: Tier 1 Features (500 trades) ===");

    let mut history = TradeHistory::new(InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(500),
        compute_tier2: false,
        compute_tier3: false,
    });

    let trades = create_test_trades(500);
    for trade in &trades {
        history.push(trade);
    }

    // Warmup
    for _ in 0..100 {
        let _ = history.compute_features(2000000);
    }

    // Benchmark: 1,000 iterations (slower due to larger window)
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let features = history.compute_features(2000000);
        let _ = (
            features.lookback_ofi,
            features.lookback_vwap,
            features.lookback_count_imbalance,
        );
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() / 1000;
    println!("500-trade window:");
    println!("  Time per call: {} ns", per_call_ns);
    println!("  Throughput: {:.1} K calls/sec", 1_000_000_000.0 / per_call_ns as f64 / 1000.0);
    println!("  Time per trade: {:.2} ns", per_call_ns as f64 / 500.0);
}

fn benchmark_tier1_large_window() {
    println!("\n=== BENCHMARK: Tier 1 Features (1000 trades) ===");

    let mut history = TradeHistory::new(InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(1000),
        compute_tier2: false,
        compute_tier3: false,
    });

    let trades = create_test_trades(1000);
    for trade in &trades {
        history.push(trade);
    }

    // Warmup
    for _ in 0..50 {
        let _ = history.compute_features(2000000);
    }

    // Benchmark: 500 iterations
    let start = std::time::Instant::now();
    for _ in 0..500 {
        let features = history.compute_features(2000000);
        let _ = (
            features.lookback_ofi,
            features.lookback_vwap,
            features.lookback_count_imbalance,
        );
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() / 500;
    println!("1000-trade window:");
    println!("  Time per call: {} ns", per_call_ns);
    println!("  Throughput: {:.1} K calls/sec", 1_000_000_000.0 / per_call_ns as f64 / 1000.0);
    println!("  Time per trade: {:.2} ns", per_call_ns as f64 / 1000.0);
}

fn benchmark_feature_mix() {
    println!("\n=== BENCHMARK: Tier 1 Feature Accuracy ===");

    let mut history = TradeHistory::new(InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(100),
        compute_tier2: false,
        compute_tier3: false,
    });

    let trades = create_test_trades(100);
    for trade in &trades {
        history.push(trade);
    }

    let features = history.compute_features(2000000);

    println!("Feature values (100-trade sample):");
    if let Some(ofi) = features.lookback_ofi {
        println!("  OFI: {:.6}", ofi);
    }
    if let Some(vwap) = features.lookback_vwap {
        println!("  VWAP: {:.8}", vwap.to_f64());
    }
    if let Some(imbalance) = features.lookback_count_imbalance {
        println!("  Count Imbalance: {:.6}", imbalance);
    }
    if let Some(intensity) = features.lookback_intensity {
        println!("  Intensity: {:.2} trades/sec", intensity);
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Task #153 Phase 1: Tier 1 SIMD Optimization Baseline         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    benchmark_tier1_small_window();
    benchmark_tier1_medium_window();
    benchmark_tier1_large_window();
    benchmark_feature_mix();

    println!("\n=== BASELINE SUMMARY ===");
    println!("✓ 50 trades: <1µs per call expected");
    println!("✓ 500 trades: ~5-10µs per call expected");
    println!("✓ 1000 trades: ~10-20µs per call expected");
    println!("\nOptimization target: 15-20% speedup via vectorized fold");
    println!("\n");
}
