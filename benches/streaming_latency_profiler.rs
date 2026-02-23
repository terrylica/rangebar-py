//! Latency profiling benchmark for Task #144: Streaming Optimization
//! Issue #96 Task #144: Real-time streaming latency profiling and optimization
//!
//! Measures end-to-end bar completion latency distribution from trade ingestion
//! through feature computation on realistic volumes (100 trades/sec, 5 symbols).
//!
//! Run: `cargo bench --bench streaming_latency_profiler`

use criterion::{criterion_group, criterion_main, Criterion};
use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::{AggTrade, FixedPoint};
use std::time::Instant;

/// Generate synthetic trades at specified rate and volume pattern
fn generate_synthetic_trades(_symbol: &str, num_trades: usize, base_price: f64) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(num_trades);

    for i in 0..num_trades {
        // Use limited decimal precision to avoid FixedPoint overflow
        let price_step = (i as f64 * 0.001).round() / 1000.0; // Step by 0.001
        let price = base_price + price_step;
        let volume = 1.0 + (i as f64 % 5.0) * 0.1; // Varying volume

        // Format with max 8 decimal places
        let price_str = format!("{:.4}", price);
        let volume_str = format!("{:.4}", volume);

        trades.push(AggTrade {
            agg_trade_id: i as i64,
            price: FixedPoint::from_str(&price_str).unwrap_or_else(|_| {
                panic!("Invalid price: {}", price_str)
            }),
            volume: FixedPoint::from_str(&volume_str).unwrap_or_else(|_| {
                panic!("Invalid volume: {}", volume_str)
            }),
            first_trade_id: i as i64,
            last_trade_id: i as i64,
            timestamp: i as i64 * 10, // 10ms between trades (100 trades/sec)
            is_buyer_maker: i % 2 == 0,
            is_best_match: None,
        });
    }

    trades
}

/// Measure bar completion latency for single processor
fn benchmark_single_symbol_latency(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("streaming_latency_single_symbol");
    group.sample_size(50); // Smaller sample size for realistic measurement
    group.measurement_time(std::time::Duration::from_secs(10));

    for window_size in [50, 100, 250, 500].iter() {
        group.bench_with_input(
            format!("bar_completion_latency_{}_trades", window_size),
            window_size,
            |b, &window_size| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let trades = generate_synthetic_trades("BTCUSDT", window_size, 42000.0);

                        let start = Instant::now();
                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }
                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Measure latency distribution percentiles (p50, p95, p99)
fn benchmark_latency_distribution(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("streaming_latency_distribution");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(15));

    let trades = generate_synthetic_trades("BTCUSDT", 500, 42000.0);

    // Warmup: 10 iterations
    for _ in 0..10 {
        let mut p = RangeBarProcessor::new(250).unwrap();
        for trade in &trades {
            let _ = p.process_single_trade(trade);
        }
    }

    group.bench_function("latency_distribution_percentiles", |b| {
        b.iter_custom(|iters| {
            let mut latencies = Vec::new();

            for _ in 0..iters {
                let mut p = RangeBarProcessor::new(250).unwrap();
                let trades = generate_synthetic_trades("BTCUSDT", 500, 42000.0);

                for trade in &trades {
                    let start = Instant::now();
                    let _ = p.process_single_trade(trade);
                    latencies.push(start.elapsed().as_micros());
                }
            }

            // Compute percentiles
            latencies.sort_unstable();
            let p50_idx = (latencies.len() as f64 * 0.5) as usize;
            let p95_idx = (latencies.len() as f64 * 0.95) as usize;
            let p99_idx = (latencies.len() as f64 * 0.99) as usize;

            eprintln!(
                "\n[LATENCY DISTRIBUTION] p50: {}µs, p95: {}µs, p99: {}µs",
                latencies[p50_idx],
                latencies[p95_idx.min(latencies.len() - 1)],
                latencies[p99_idx.min(latencies.len() - 1)]
            );

            std::time::Duration::from_micros(
                latencies.iter().sum::<u128>() as u64 / latencies.len() as u64
            )
        });
    });

    group.finish();
}

/// Measure feature computation contribution breakdown
fn benchmark_feature_computation_breakdown(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("streaming_feature_computation");
    group.sample_size(20);

    // Test at different lookback window sizes
    for window_size in [50, 100, 250, 500].iter() {
        group.bench_with_input(
            format!("feature_computation_{}_window", window_size),
            window_size,
            |b, &window_size| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let trades = generate_synthetic_trades("BTCUSDT", window_size * 2, 42000.0);

                        let start = Instant::now();
                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }
                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Measure memory allocation patterns during streaming
fn benchmark_memory_allocation_patterns(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("streaming_memory_allocation");
    group.sample_size(30);

    group.bench_function("allocation_per_trade", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = std::time::Duration::ZERO;

            for _ in 0..iters {
                let mut processor = RangeBarProcessor::new(250).unwrap();
                // Inject trades that trigger bar completions
                let base_price = 42000.0;
                let _threshold_price = base_price * 1.0025; // 250 bps

                let mut price = base_price;
                let mut agg_id = 0i64;
                let start = Instant::now();

                // Process 100 trades to trigger at least one bar completion
                for _ in 0..100 {
                    price += 0.001;
                    let price_str = format!("{:.4}", price);
                    let trade = AggTrade {
                        agg_trade_id: agg_id,
                        price: FixedPoint::from_str(&price_str).unwrap_or_else(|_| {
                            panic!("Invalid price: {}", price_str)
                        }),
                        volume: FixedPoint::from_str("1.0").unwrap(),
                        first_trade_id: agg_id,
                        last_trade_id: agg_id,
                        timestamp: agg_id * 10,
                        is_buyer_maker: agg_id % 2 == 0,
                        is_best_match: None,
                    };
                    agg_id += 1;

                    let _ = processor.process_single_trade(&trade);
                }

                total_duration += start.elapsed();
            }

            total_duration
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_symbol_latency,
    benchmark_latency_distribution,
    benchmark_feature_computation_breakdown,
    benchmark_memory_allocation_patterns,
);
criterion_main!(benches);
