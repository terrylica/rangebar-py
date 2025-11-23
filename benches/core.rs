use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Import directly from the upstream rangebar-core crate
// This is the dependency declared in Cargo.toml, not our Python bindings
use rangebar_core::{AggTrade, FixedPoint, RangeBarProcessor};

/// Helper: Convert f64 to FixedPoint (8 decimal precision)
fn f64_to_fixed(value: f64) -> FixedPoint {
    let scaled = (value * 100_000_000.0).round() as i64;
    FixedPoint(scaled)
}

/// Generate synthetic trade data for benchmarking
fn generate_trades(count: usize, base_price: f64, volatility: f64) -> Vec<AggTrade> {
    (0..count)
        .map(|i| {
            // Simulate price movement (simple sine wave + noise)
            let price_offset = (i as f64 / 100.0).sin() * volatility;
            let price = base_price + price_offset + (i % 10) as f64 * 0.5;

            AggTrade {
                agg_trade_id: i as i64,
                price: f64_to_fixed(price),
                volume: f64_to_fixed(0.01 + (i % 5) as f64 * 0.001),
                first_trade_id: i as i64,
                last_trade_id: i as i64,
                timestamp: (i as i64 * 100_000), // 100ms intervals (in microseconds)
                is_buyer_maker: i % 2 == 0,
                is_best_match: Some(true),
            }
        })
        .collect()
}

/// Benchmark: Process 1K trades
fn bench_throughput_1k(c: &mut Criterion) {
    let trades = generate_trades(1_000, 42000.0, 100.0);

    c.bench_function("process_1k_trades", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 bps = 0.25%
            processor.process_agg_trade_records(black_box(&trades)).unwrap();
        });
    });
}

/// Benchmark: Process 100K trades
fn bench_throughput_100k(c: &mut Criterion) {
    let trades = generate_trades(100_000, 42000.0, 100.0);

    c.bench_function("process_100k_trades", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(250).unwrap();
            processor.process_agg_trade_records(black_box(&trades)).unwrap();
        });
    });
}

/// Benchmark: Process 1M trades (primary performance target)
fn bench_throughput_1m(c: &mut Criterion) {
    let trades = generate_trades(1_000_000, 42000.0, 100.0);

    c.bench_function("process_1m_trades", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(250).unwrap();
            processor.process_agg_trade_records(black_box(&trades)).unwrap();
        });
    });
}

/// Benchmark: Throughput scaling across different trade counts
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let trades = generate_trades(*size, 42000.0, 100.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut processor = RangeBarProcessor::new(250).unwrap();
                processor.process_agg_trade_records(black_box(&trades)).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark: Impact of threshold (compression ratio)
fn bench_threshold_impact(c: &mut Criterion) {
    let trades = generate_trades(100_000, 42000.0, 100.0);
    let mut group = c.benchmark_group("threshold_impact");

    for threshold_bps in [100, 250, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(threshold_bps),
            threshold_bps,
            |b, &threshold| {
                b.iter(|| {
                    let mut processor = RangeBarProcessor::new(threshold).unwrap();
                    processor.process_agg_trade_records(black_box(&trades)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory allocation patterns
fn bench_memory_allocations(c: &mut Criterion) {
    let trades = generate_trades(1_000_000, 42000.0, 100.0);

    c.bench_function("memory_allocations_1m", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(250).unwrap();
            let result = processor.process_agg_trade_records(black_box(&trades)).unwrap();
            black_box(result); // Ensure result isn't optimized away
        });
    });
}

criterion_group!(
    benches,
    bench_throughput_1k,
    bench_throughput_100k,
    bench_throughput_1m,
    bench_throughput_scaling,
    bench_threshold_impact,
    bench_memory_allocations,
);

criterion_main!(benches);
