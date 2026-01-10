// Performance benchmarks for range bar processing
//
// Target: 1M ticks < 100ms, 1B ticks < 30 seconds

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rangebar::{AggTrade, FixedPoint, RangeBar, RangeBarProcessor};

fn create_test_trades(count: usize, base_price: f64, volatility: f64) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let mut price = base_price;
    let mut rng = 0x12345678u64; // Simple deterministic RNG

    for i in 0..count {
        // Simple LCG for deterministic "random" price movements
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let random = (rng >> 16) as f64 / 65536.0; // [0, 1)
        let price_change = (random - 0.5) * volatility * 2.0;
        price += price_change;

        // Alternate buy/sell pressure for realistic microstructure testing
        let is_buyer_maker = (i % 2) == 0; // 50/50 split for balanced microstructure

        trades.push(AggTrade {
            agg_trade_id: i as i64,
            price: FixedPoint::from_str(&format!("{:.8}", price)).unwrap(),
            volume: FixedPoint::from_str("1.0").unwrap(),
            first_trade_id: i as i64,
            last_trade_id: i as i64,
            timestamp: 1640995200000 + (i as i64 * 100), // 100ms intervals
            is_buyer_maker,                              // PHASE 0: Market microstructure field
            is_best_match: None,
        });
    }

    trades
}

fn bench_range_bar_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rangebar_processing");

    // Test different scales
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let trades = create_test_trades(*size, 50000.0, 100.0); // BTC-like volatility

        group.bench_with_input(BenchmarkId::new("process_trades", size), size, |b, _| {
            b.iter(|| {
                let mut proc = RangeBarProcessor::new(8000);
                let bars = proc.process_agg_trade_records(black_box(&trades)).unwrap();
                black_box(bars);
            });
        });
    }

    group.finish();
}

fn bench_threshold_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_calculation");

    let price = FixedPoint::from_str("50000.12345678").unwrap();

    group.bench_function("compute_thresholds", |b| {
        b.iter(|| {
            let (upper, lower) = price.compute_range_thresholds(black_box(8000));
            black_box((upper, lower));
        });
    });

    group.finish();
}

fn bench_breach_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("breach_detection");

    let trade = AggTrade {
        agg_trade_id: 1,
        price: FixedPoint::from_str("50000.0").unwrap(),
        volume: FixedPoint::from_str("1.0").unwrap(),
        first_trade_id: 1,
        last_trade_id: 1,
        timestamp: 1640995200000,
        is_buyer_maker: false, // Buy pressure for testing
        is_best_match: None,
    };

    let bar = RangeBar::new(&trade);
    let (upper_threshold, lower_threshold) = trade.price.compute_range_thresholds(8000);
    let test_price = FixedPoint::from_str("50400.0").unwrap(); // Breach price

    group.bench_function("is_breach", |b| {
        b.iter(|| {
            let result = bar.is_breach(
                black_box(test_price),
                black_box(upper_threshold),
                black_box(lower_threshold),
            );
            black_box(result);
        });
    });

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory allocation patterns for different batch sizes
    for batch_size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_processing", batch_size),
            batch_size,
            |b, &size| {
                let trades = create_test_trades(size, 50000.0, 100.0);

                b.iter(|| {
                    let mut processor = RangeBarProcessor::new(8000);
                    // Process in batches to simulate real-world streaming
                    let batch_size = 1000;
                    let mut total_bars = 0;

                    for chunk in trades.chunks(batch_size) {
                        let bars = processor
                            .process_agg_trade_records(black_box(chunk))
                            .unwrap();
                        total_bars += bars.len();
                    }

                    black_box(total_bars);
                });
            },
        );
    }

    group.finish();
}

fn bench_extreme_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("extreme_cases");

    // High volatility scenario (many range bar completions)
    let high_volatility_trades = create_test_trades(10000, 50000.0, 500.0); // Very volatile

    group.bench_function("high_volatility", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(8000);
            let bars = processor
                .process_agg_trade_records(black_box(&high_volatility_trades))
                .unwrap();
            black_box(bars);
        });
    });

    // Low volatility scenario (few range bar completions)
    let low_volatility_trades = create_test_trades(10000, 50000.0, 10.0); // Very stable

    group.bench_function("low_volatility", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(8000);
            let bars = processor
                .process_agg_trade_records(black_box(&low_volatility_trades))
                .unwrap();
            black_box(bars);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_range_bar_processing,
    bench_threshold_calculation,
    bench_breach_detection,
    bench_memory_efficiency,
    bench_extreme_cases
);
criterion_main!(benches);
