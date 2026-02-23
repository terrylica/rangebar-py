//! Benchmark: ApEn SIMD Optimization Analysis (Task #161)
//! Issue #96 Task #161: Profile and optimize Approximate Entropy via SIMD
//!
//! Context: Task #157 found ApEn is 1000x SLOWER than PE
//! Goal: SIMD optimization to close performance gap (5-10x speedup target)
//!
//! Analysis strategy:
//! 1. Profile current scalar ApEn implementation
//! 2. Identify bottleneck: O(n²) pattern comparison loop
//! 3. Design SIMD-accelerated pattern distance computation
//! 4. Implement batch vectorization

use rangebar_core::interbar_math::compute_approximate_entropy;
use std::time::Instant;

fn create_price_series(count: usize, volatility: f64) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 50000.0;
            let trend = (i as f64 / count as f64) * 100.0;
            let noise = (i as f64 * 0.1).sin() * volatility;
            base + trend + noise
        })
        .collect()
}

fn benchmark_apen_cost_by_window_size() {
    println!("\n=== BENCHMARK: ApEn Cost vs Window Size (Scalar Implementation) ===");
    println!("Current bottleneck: O(n²) pattern comparison in compute_phi\n");

    let trade_counts = vec![100, 200, 300, 400, 500];
    let m = 2; // ApEn embedding dimension
    let r = 0.5; // tolerance

    for count in trade_counts {
        let prices = create_price_series(count, 0.5);

        let start = Instant::now();
        let num_iters = 100;
        for _ in 0..num_iters {
            let _ = compute_approximate_entropy(&prices, m, r);
        }
        let elapsed = start.elapsed();

        let per_call_us = elapsed.as_nanos() as f64 / (num_iters as f64 * 1000.0);

        // Estimate complexity: O(n²) pattern pairs × O(m) distance checks
        let estimated_ops = (count as f64).powi(2) * m as f64;
        let ops_per_ns = estimated_ops / (per_call_us * 1000.0);

        println!(
            "ApEn @ {:3} trades: {:.2} µs (est. {:.0} ops/ns, O(n²) ~{:.1}M pairs)",
            count, per_call_us, ops_per_ns,
            (count as f64).powi(2) / 1_000_000.0
        );
    }
}

fn analyze_apen_bottleneck() {
    println!("\n=== ANALYSIS: ApEn Bottleneck Decomposition ===\n");

    let count = 500;
    let prices = create_price_series(count, 0.5);

    // Estimate breakdown
    let pattern_pairs = (count as f64).powi(2) as usize / 2;
    let distance_checks = pattern_pairs * 2; // m=2, check 2 elements per pair
    let comparisons_per_check = 1; // abs difference <= r

    println!("Input: {} trades (m=2, r=0.5)", count);
    println!();
    println!("Operation Breakdown:");
    println!("  1. Pattern pair enumeration: O(n²) = ~{} pairs", pattern_pairs);
    println!("  2. Distance checks per pair: O(m) = 2 elements");
    println!("  3. Comparison per element: (a-b).abs() <= r");
    println!("  4. Total operations: {} distance checks", distance_checks);
    println!();
    println!("Current Implementation (Scalar):");
    println!("  - Nested loops: for i in 0..n, for j in i+1..n");
    println!("  - Per pair: zip + all() + abs() + comparison");
    println!("  - Latency: ~208 µs for 500 trades");
    println!("  - Throughput: ~{:.0} ops/µs", distance_checks as f64 / 208.0);
    println!();
    println!("SIMD Optimization Opportunities:");
    println!("  ✓ Vectorize abs(a-b) <= r checks (f64x2 or f64x4)");
    println!("  ✓ Batch process pattern pairs (compare 2-4 pairs per iteration)");
    println!("  ✓ Reduce branching (all() logic → bitmask operations)");
    println!("  ✓ Pre-compute tolerance bounds (avoid repeated r comparisons)");
    println!();
    println!("Target Optimization:");
    println!("  Current: ~208 µs @ 500 trades");
    println!("  Target: <30 µs (7x speedup, matching PE performance)");
    println!("  SIMD estimate: 3-5x via vectorization");
    println!("  Additional: 1-2x via algorithm improvements");
}

fn compare_apen_pe_performance() {
    println!("\n=== COMPARISON: ApEn vs PE Performance (Why Gap Exists) ===\n");

    let count = 500;
    let prices = create_price_series(count, 0.5);

    // Benchmark both
    let start = Instant::now();
    for _ in 0..100 {
        let _ = compute_approximate_entropy(&prices, 2, 0.5);
    }
    let apen_elapsed = start.elapsed();
    let apen_us = apen_elapsed.as_nanos() as f64 / (100.0 * 1000.0);

    // PE would be ~0.2 µs (from Task #157 benchmark)
    let pe_us = 0.22;

    println!("ApEn implementation:");
    println!("  Complexity: O(n²) pattern comparisons");
    println!("  Characteristics: Scalar loops, branching, repeated calculations");
    println!("  Performance: {:.2} µs @ 500 trades", apen_us);
    println!();
    println!("PE implementation (from Task #157):");
    println!("  Complexity: O(n log n) with SIMD optimization");
    println!("  Characteristics: Vectorized ordinal patterns, cache-optimized");
    println!("  Performance: {:.2} µs @ 500 trades", pe_us);
    println!();
    println!("Performance Gap:");
    println!("  Ratio: {:.0}x (ApEn slower than PE)", apen_us / pe_us);
    println!("  Root cause: Unoptimized ApEn (no SIMD), not algorithm");
    println!("  Fix: Apply SIMD optimization (Task #129 pattern) to ApEn");
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Task #161: ApEn SIMD Optimization Analysis                   ║");
    println!("║ Goal: 5-10x speedup to close PE performance gap              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    benchmark_apen_cost_by_window_size();
    analyze_apen_bottleneck();
    compare_apen_pe_performance();

    println!("\n=== OPTIMIZATION STRATEGY ===");
    println!();
    println!("Phase 1: Scalar Optimizations (1-2x speedup)");
    println!("  - Precompute r bounds to avoid repeated comparisons");
    println!("  - Use max(abs()) instead of zip+all() for Chebyshev distance");
    println!("  - Cache pattern offsets to improve memory locality");
    println!();
    println!("Phase 2: SIMD Vectorization (2-4x speedup)");
    println!("  - Batch pattern pair comparisons using f64x2/f64x4");
    println!("  - Vectorize distance computation: SIMD max(abs(a-b)) <= r");
    println!("  - Use SIMD reductions for histogram counting");
    println!();
    println!("Phase 3: Algorithm Improvements (1-2x speedup)");
    println!("  - Sample patterns for approximate matching (like PE)");
    println!("  - Early termination on high pattern diversity");
    println!("  - Exploit symmetry in pattern matching");
    println!();
    println!("Combined Target: 7-10x speedup achievable");
    println!("  Current: ~208 µs @ 500 trades");
    println!("  Target: <30 µs (performance parity with PE)");
}
