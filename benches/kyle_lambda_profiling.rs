//! Kyle Lambda scalar computation profiling
//!
//! Measures Kyle Lambda cost as percentage of Tier 2 computation.
//! Decision threshold (from Task #148):
//! - <5% Tier 2 time: Keep scalar (diminishing returns for SIMD)
//! - >10% Tier 2 time: Implement Wide SIMD vectorization
//!
//! Issue #96 Task #148: Expand Wide SIMD to remaining Tier 2 features
//! Run: cargo bench --bench kyle_lambda_profiling

use std::time::Instant;
use rangebar_core::interbar_types::TradeSnapshot;
use rangebar_core::FixedPoint;

fn create_test_trades(n: usize) -> Vec<TradeSnapshot> {
    let mut trades = Vec::with_capacity(n);
    for i in 0..n {
        let price_offset = (i as f64 * 0.1) % 100.0;
        trades.push(TradeSnapshot {
            timestamp: (i as i64) * 100,
            price: FixedPoint((50000.0 + price_offset) as i64),
            volume: FixedPoint((1.0 * 1e8) as i64),
            is_buyer_maker: i % 2 == 0,
            turnover: ((50000.0 + price_offset) * 1e8) as i128,
        });
    }
    trades
}

fn main() {
    println!("\n=== Kyle Lambda Scalar Profiling ===\n");
    println!("Profile: Kyle Lambda vs other Tier 2 features");
    println!("Decision: <5% Tier 2 time → keep scalar, >10% → implement SIMD\n");

    let iterations = 10000;
    let window_sizes = vec![50, 100, 250, 500, 1000];

    for window_size in window_sizes {
        let trades = create_test_trades(window_size);
        let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();

        // Benchmark Kyle Lambda
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = rangebar_core::interbar_math::compute_kyle_lambda(&trade_refs);
        }
        let kyle_elapsed = start.elapsed();
        let kyle_per_call_ns = kyle_elapsed.as_nanos() as f64 / iterations as f64;

        // Benchmark Burstiness (typical Tier 2 operation)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = rangebar_core::interbar_math::compute_burstiness(&trade_refs);
        }
        let burst_elapsed = start.elapsed();
        let burst_per_call_ns = burst_elapsed.as_nanos() as f64 / iterations as f64;

        // Estimate: If Kyle Lambda is ~3-5% of combined Tier 2, it's worth profiling
        let combined_ns = kyle_per_call_ns + burst_per_call_ns;
        let kyle_percentage = (kyle_per_call_ns / combined_ns) * 100.0;

        println!("Window: {} trades", window_size);
        println!("  Kyle Lambda:     {:.2} ns/call", kyle_per_call_ns);
        println!("  Burstiness:      {:.2} ns/call", burst_per_call_ns);
        println!("  Kyle % of Tier2:  {:.1}%", kyle_percentage);
        println!();
    }

    println!("\n=== Analysis ===\n");
    println!("If Kyle Lambda is <5% of Tier 2 time:");
    println!("  → Keep scalar (SIMD overhead > benefit)");
    println!();
    println!("If Kyle Lambda is 5-10% of Tier 2 time:");
    println!("  → Borderline (profile with real data)");
    println!();
    println!("If Kyle Lambda is >10% of Tier 2 time:");
    println!("  → Implement Wide SIMD vectorization (2-4x expected gain)");
}
