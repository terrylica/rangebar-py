//! Hurst DFA memoization benchmark (Issue #96 Task #192)
//!
//! Measures the performance improvement from memoizing x_mean computation
//! outside the segment loop in Hurst DFA detrending.
//!
//! **Hypothesis**: Eliminates ~200-500 redundant divide operations per bar,
//! delivering 3-7% speedup on Hurst computation for large lookback windows.

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::interbar::{InterBarConfig, LookbackMode};
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create synthetic trades with varying prices
fn create_trades_with_trend(count: usize, trend: f64) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let mut price = 100.0;

    for idx in 0..count {
        price += trend * 0.001; // Trending movement
        trades.push(AggTrade {
            agg_trade_id: idx as i64,
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((10_000_000) as i64),
            first_trade_id: idx as i64,
            last_trade_id: idx as i64,
            timestamp: 1000000 + (idx as i64),
            is_buyer_maker: idx % 2 == 0,
            is_best_match: None,
        });
    }

    trades
}

/// Benchmark Hurst computation on different window sizes
fn bench_hurst_computation(window_size: usize, iterations: usize) {
    println!("\nHurst DFA computation on {} trades ({} iterations)", window_size, iterations);

    let trades = create_trades_with_trend(window_size, 1.0);
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(window_size),
        compute_tier2: false,
        compute_tier3: true, // Enable Hurst (Tier 3)
        ..Default::default()
    };

    let mut history = rangebar_core::interbar::TradeHistory::new(config);

    // Add trades to history
    for trade in &trades {
        history.push(trade);
    }

    // Benchmark: compute features (which includes Hurst)
    let test_timestamp = 1000000 + (window_size as i64);

    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _iter in 0..iterations {
        let features = history.compute_features(test_timestamp);
        dummy += features.lookback_hurst.unwrap_or(0.0);
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_hurst = duration.as_nanos() as f64 / iterations as f64;
    println!("  Avg latency: {:.2} ns/computation", ns_per_hurst);
}

fn main() {
    println!("=== Hurst DFA Memoization Benchmark (Task #192) ===");
    println!("Measures x_mean memoization impact on Hurst computation\n");

    // Test different window sizes to show impact scaling
    // Small windows (n < 64): Hurst not computed, minimal impact
    // Large windows (n >= 64): Hurst computed, significant savings from memoization

    println!("--- Small Windows (Hurst not computed) ---");
    bench_hurst_computation(50, 500);

    println!("\n--- Medium Windows (Hurst computed) ---");
    bench_hurst_computation(100, 200);
    bench_hurst_computation(250, 100);

    println!("\n--- Large Windows (Hurst computed, max savings) ---");
    bench_hurst_computation(500, 50);

    println!("\n=== Summary ===");
    println!("Memoization eliminates redundant x_mean computation by:");
    println!("1. Moving division outside segment loop");
    println!("2. Computing x_mean once per scale (not per segment)");
    println!("3. Reusing value across 50-100+ segments");
    println!("\nExpected improvement: 3-7% on Hurst computation");
    println!("Most impact on large windows (500+ trades)");
}
