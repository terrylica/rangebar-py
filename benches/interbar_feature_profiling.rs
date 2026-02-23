// Comprehensive profiling for Tier 2/3 inter-bar features
// Measures individual feature computation times to identify bottlenecks
//
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96 Task #7
// Run: cargo bench --bench interbar_feature_profiling
// PROCESS-STORM-OK

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rangebar_core::interbar_types::TradeSnapshot;
use rangebar_core::FixedPoint;
use std::time::Instant;

#[allow(dead_code)]
mod profiler {
    use std::time::Instant;

    pub struct FeatureProfiler;

    impl FeatureProfiler {
        pub fn run() {
            println!("\n=== Manual Feature Timing Profile ===\n");
            println!("(Profiling data will be displayed by criterion benchmarks below)\n");
        }
    }
}

fn create_test_trades(n: usize) -> Vec<TradeSnapshot> {
    let mut trades = Vec::with_capacity(n);
    for i in 0..n {
        let price_offset = (i as f64 * 0.1) % 100.0;
        trades.push(TradeSnapshot {
            timestamp: (i as i64) * 100,
            price: FixedPoint((50000.0 + price_offset) as i64 * 100_000_000 as i64 / 100_000_000),
            volume: FixedPoint((1.0 * 1e8) as i64),
            is_buyer_maker: i % 2 == 0,
            turnover: (50000i128 + price_offset as i128) * 100_000_000i128 / 100_000_000i128,
        });
    }
    trades
}

fn bench_tier2_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier2_features");
    group.sample_size(100);

    for window_size in &[10, 50, 100, 500, 1000] {
        let trades = create_test_trades(*window_size);
        let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();

        // Kyle Lambda
        group.bench_with_input(
            BenchmarkId::new("kyle_lambda", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_kyle_lambda(
                        black_box(&trade_refs),
                    );
                    black_box(result);
                });
            },
        );

        // Burstiness (scalar fallback on stable, SIMD on nightly with feature)
        group.bench_with_input(
            BenchmarkId::new("burstiness", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_burstiness(
                        black_box(&trade_refs),
                    );
                    black_box(result);
                });
            },
        );

        // Volume Moments (skewness + kurtosis in single pass)
        group.bench_with_input(
            BenchmarkId::new("volume_moments", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_volume_moments(
                        black_box(&trade_refs),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_tier3_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier3_features");
    group.sample_size(50);

    for window_size in &[64, 100, 500, 1000] {
        let trades = create_test_trades(*window_size);
        let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();

        // Kaufman Efficiency Ratio
        group.bench_with_input(
            BenchmarkId::new("kaufman_er", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_kaufman_er(
                        black_box(&prices),
                    );
                    black_box(result);
                });
            },
        );

        // Garman-Klass volatility
        group.bench_with_input(
            BenchmarkId::new("garman_klass", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_garman_klass(
                        black_box(&trade_refs),
                    );
                    black_box(result);
                });
            },
        );

        // Hurst exponent (already using evrom/hurst R/S, O(n log n))
        group.bench_with_input(
            BenchmarkId::new("hurst_dfa", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_hurst_dfa(
                        black_box(&prices),
                    );
                    black_box(result);
                });
            },
        );

        // Permutation Entropy (adaptive M=2 for small, M=3 for large)
        group.bench_with_input(
            BenchmarkId::new("permutation_entropy", window_size),
            window_size,
            |b, _| {
                b.iter(|| {
                    let result = rangebar_core::interbar_math::compute_permutation_entropy(
                        black_box(&prices),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn manual_profiling() {
    println!("\n=== Manual Feature Timing Profile ===\n");

    let window_sizes = vec![10, 50, 100, 500, 1000];

    for &size in &window_sizes {
        println!("Window size: {}", size);
        let trades = create_test_trades(size);
        let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();

        const ITERATIONS: usize = 1000;

        // Kyle Lambda
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_kyle_lambda(&trade_refs);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Kyle Lambda:        {:?}/call", avg);

        // Burstiness
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_burstiness(&trade_refs);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Burstiness:         {:?}/call", avg);

        // Volume Moments
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_volume_moments(&trade_refs);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Volume Moments:     {:?}/call", avg);

        // Kaufman ER
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_kaufman_er(&prices);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Kaufman ER:         {:?}/call", avg);

        // Garman-Klass
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_garman_klass(&trade_refs);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Garman-Klass:       {:?}/call", avg);

        // Hurst
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_hurst_dfa(&prices);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Hurst Exponent:     {:?}/call", avg);

        // Permutation Entropy
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rangebar_core::interbar_math::compute_permutation_entropy(&prices);
        }
        let avg = start.elapsed() / ITERATIONS as u32;
        println!("  Permutation Entropy: {:?}/call", avg);

        println!();
    }
}

criterion_group!(benches, bench_tier2_features, bench_tier3_features);
criterion_main!(benches);
