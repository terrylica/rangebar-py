//! Intra-bar early-exit optimization benchmark (Issue #96 Task #193)
//!
//! Measures the performance improvement from early-exiting complexity feature
//! computation (Hurst, Permutation Entropy) for small intra-bar windows.
//!
//! **Hypothesis**: Skipping O(n log n) complexity computations for n < 60-64
//! delivers 2-5% speedup on consolidation bars (typically 5-50 trades).

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create synthetic trades with realistic price distribution
fn create_trades_with_pattern(count: usize, pattern: &str) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let mut price = 100.0;
    let mut state = 12345u64;

    for idx in 0..count {
        // Pseudo-random walk
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        match pattern {
            "uptrend" => price += 0.01,
            "downtrend" => price -= 0.01,
            "volatile" => price += (state as i32 as f64 / 1e9) * 0.1,
            _ => {
                let drift = if (state & 1) == 0 { 0.001 } else { -0.001 };
                price += drift;
            }
        }

        trades.push(AggTrade {
            agg_trade_id: idx as i64,
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((0.1 + (state as f64 % 0.9) * 1e8) as i64),
            first_trade_id: idx as i64,
            last_trade_id: idx as i64,
            timestamp: 1000000 + (idx as i64),
            is_buyer_maker: (state & 1) == 0,
            is_best_match: None,
        });
    }

    trades
}

/// Benchmark intra-bar feature computation on different window sizes
fn bench_intra_bar_computation(window_size: usize, iterations: usize) {
    println!(
        "\n{} trades ({} iterations)",
        window_size, iterations
    );

    let trades = create_trades_with_pattern(window_size, "volatile");

    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _iter in 0..iterations {
        let features = rangebar_core::intrabar::features::compute_intra_bar_features(
            std::hint::black_box(&trades),
        );
        dummy += features.intra_ofi.unwrap_or(0.0);
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_feature = duration.as_nanos() as f64 / iterations as f64;
    println!("  Avg latency: {:.2} ns/computation", ns_per_feature);
}

fn main() {
    println!("=== Intra-Bar Early-Exit Optimization Benchmark (Task #193) ===");
    println!("Measures speedup from skipping complexity features for small windows\n");

    println!("--- Small Windows (Hurst/PE not computed) ---");
    println!("Below Hurst (n < 64) and PE (n < 60) thresholds:");
    bench_intra_bar_computation(5, 2000);
    bench_intra_bar_computation(10, 2000);
    bench_intra_bar_computation(25, 1000);
    bench_intra_bar_computation(50, 500);

    println!("\n--- Medium Windows (PE computed, Hurst skipped) ---");
    println!("Above PE threshold (60) but below Hurst (64):");
    bench_intra_bar_computation(60, 300);
    bench_intra_bar_computation(63, 300);

    println!("\n--- Large Windows (All features computed) ---");
    println!("Above both Hurst (64) and PE (60) thresholds:");
    bench_intra_bar_computation(64, 100);
    bench_intra_bar_computation(100, 100);
    bench_intra_bar_computation(250, 50);

    println!("\n=== Summary ===");
    println!("Early-exit optimization skips expensive features by:");
    println!("1. Checking n < 64 before Hurst DFA computation");
    println!("2. Checking n < 60 before Permutation Entropy computation");
    println!("3. Always computing ITH features (linear time, always valuable)");
    println!("\nExpected improvement: 2-5% on consolidation bars (n < 60)");
    println!("Most impact on small bars (5-50 trades) - common pattern");
}
