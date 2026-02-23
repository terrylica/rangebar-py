//! Benchmark for normalization LUT optimization (Issue #96 Task #197)
//!
//! Compares performance of:
//! - Original: logistic_sigmoid + exp, tanh() transcendental functions
//! - Optimized: sigmoid_lut, tanh_lut lookup tables
//!
//! Expected speedup: 3-8% on normalize_epochs and normalize_excess calls

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Import the normalization functions
use rangebar_core::intrabar::normalize::{normalize_epochs, normalize_excess};

fn bench_normalize_epochs(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_epochs_lut");

    // Test across density ranges: 0, 0.25, 0.5, 0.75, 1.0
    let test_cases = vec![
        (0, 100),       // 0% density
        (25, 100),      // 25% density
        (50, 100),      // 50% density
        (75, 100),      // 75% density
        (100, 100),     // 100% density
        (1, 100),       // 1% density
        (10, 200),      // 5% density
    ];

    for (epochs, lookback) in test_cases {
        let label = format!("{}%", (epochs as f64 / lookback as f64 * 100.0) as i32);
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(epochs, lookback),
            |b, &(epochs, lookback)| {
                b.iter(|| {
                    black_box(normalize_epochs(black_box(epochs), black_box(lookback)))
                });
            },
        );
    }

    group.finish();
}

fn bench_normalize_excess(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_excess_lut");

    // Test across typical ITH excess values: 0.01, 0.05, 0.1, 0.2, 1.0
    let test_values = vec![
        0.01,   // 1% excess
        0.05,   // 5% excess
        0.10,   // 10% excess
        0.20,   // 20% excess
        0.50,   // 50% excess
        1.00,   // 100% excess (saturates at tanh(5))
    ];

    for value in test_values {
        let label = format!("{:.0}%", value * 100.0);
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &value,
            |b, &value| {
                b.iter(|| {
                    black_box(normalize_excess(black_box(value)))
                });
            },
        );
    }

    group.finish();
}

fn bench_normalize_burst(c: &mut Criterion) {
    // Burst benchmark: repeatedly call both functions as would happen in feature computation
    c.bench_function("normalize_burst_1000_calls", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let epochs = (i * 17) % 100; // Pseudo-random epochs
                let lookback = 100 + (i % 50);
                let excess = (i as f64) * 0.001;

                black_box(normalize_epochs(epochs, lookback));
                black_box(normalize_excess(excess));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_normalize_epochs,
    bench_normalize_excess,
    bench_normalize_burst
);
criterion_main!(benches);
