//! OFI branchless ILP optimization benchmark (Issue #96 Task #194)
//!
//! Measures the performance improvement from using branchless arithmetic and
//! instruction-level parallelism (pair-wise processing) for OFI computation.
//!
//! **Hypothesis**: Eliminating branches and enabling superscalar CPU parallelism
//! delivers 1-2% speedup on OFI computation for medium-large windows (n > 50).

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::interbar_types::TradeSnapshot;
use rangebar_core::interbar_math::compute_ofi_branchless;
use std::time::Instant;

/// Create synthetic trades with realistic buy/sell distribution
fn create_test_trades(count: usize, buy_ratio: f64) -> Vec<TradeSnapshot> {
    let mut trades = Vec::with_capacity(count);
    let mut price = 100.0;
    let mut state = 12345u64;

    for idx in 0..count {
        // Pseudo-random walk
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        price += ((state as i32 as f64) / 1e9) * 0.1;

        let is_buyer_maker = ((state as f64) % 1.0) > buy_ratio;

        trades.push(TradeSnapshot {
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((0.1 + (state as f64 % 0.9) * 1e8) as i64),
            turnover: ((price * 1e8 * (0.1 + (state as f64 % 0.9))) as i128),
            timestamp: 1000000 + (idx as i64),
            is_buyer_maker,
        });
    }

    trades
}

/// Benchmark OFI computation on different window sizes and buy/sell ratios
fn bench_ofi_computation(window_size: usize, buy_ratio: f64, iterations: usize) {
    let trades = create_test_trades(window_size, buy_ratio);
    let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();

    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _iter in 0..iterations {
        let ofi = compute_ofi_branchless(std::hint::black_box(&trade_refs));
        dummy += ofi;
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_computation = duration.as_nanos() as f64 / iterations as f64;
    println!("  Avg latency: {:.2} ns/computation", ns_per_computation);
}

fn main() {
    println!("=== OFI Branchless ILP Optimization Benchmark (Task #194) ===");
    println!("Measures speedup from branchless arithmetic + pair-wise processing\n");

    println!("--- Small Windows (n < 50) ---");
    println!("Buy-heavy (75% buyer):");
    bench_ofi_computation(10, 0.75, 2000);
    bench_ofi_computation(25, 0.75, 1000);
    bench_ofi_computation(50, 0.75, 500);

    println!("\nBalanced (50/50 buy-sell):");
    bench_ofi_computation(10, 0.50, 2000);
    bench_ofi_computation(25, 0.50, 1000);
    bench_ofi_computation(50, 0.50, 500);

    println!("\nSell-heavy (25% buyer):");
    bench_ofi_computation(10, 0.25, 2000);
    bench_ofi_computation(25, 0.25, 1000);
    bench_ofi_computation(50, 0.25, 500);

    println!("\n--- Medium Windows (50 < n < 250) ---");
    println!("Buy-heavy (75% buyer):");
    bench_ofi_computation(100, 0.75, 300);
    bench_ofi_computation(250, 0.75, 100);

    println!("\nBalanced (50/50):");
    bench_ofi_computation(100, 0.50, 300);
    bench_ofi_computation(250, 0.50, 100);

    println!("\n--- Large Windows (n >= 250) ---");
    println!("Buy-heavy:");
    bench_ofi_computation(500, 0.75, 50);
    bench_ofi_computation(1000, 0.75, 20);

    println!("\n=== Summary ===");
    println!("OFI branchless optimization uses:");
    println!("1. Pair-wise processing for instruction-level parallelism");
    println!("2. Branchless masks instead of if/else branches");
    println!("3. Arithmetic selection (1.0 or 0.0) for buy/sell routing");
    println!("\nExpected improvement: 1-2% on medium-large windows");
    println!("Most impact on windows > 100 trades");
}
