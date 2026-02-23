// Multi-threaded streaming benchmarks for range bar processing
// # PROCESS-STORM-OK: Bounded thread spawning (max 8 threads) for benchmarking
//
// Issue #96 Task #24: Identify real-world performance bottlenecks
// Tests concurrent bar updates, thread coordination, and stress scenarios

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rangebar::{AggTrade, FixedPoint, RangeBarProcessor};
use std::sync::{Arc, Mutex};
use std::thread;

/// Generate synthetic trades for benchmarking
fn create_test_trades(count: usize, base_price: f64, volatility: f64) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(count);
    let mut price = base_price;
    let mut rng = 0x12345678u64;

    for i in 0..count {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let random = (rng >> 16) as f64 / 65536.0;
        let price_change = (random - 0.5) * volatility * 2.0;
        price += price_change;

        let is_buyer_maker = (i % 2) == 0;

        trades.push(AggTrade {
            agg_trade_id: i as i64,
            price: FixedPoint::from_str(&format!("{:.8}", price)).unwrap(),
            volume: FixedPoint::from_str("1.0").unwrap(),
            first_trade_id: i as i64,
            last_trade_id: i as i64,
            timestamp: 1640995200000 + (i as i64 * 100),
            is_buyer_maker,
            is_best_match: None,
        });
    }

    trades
}

/// Single-threaded baseline for comparison
fn bench_single_threaded_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_threaded_baseline");

    for trade_count in [100_000, 500_000, 1_000_000].iter() {
        let trades = create_test_trades(*trade_count, 50000.0, 100.0);

        group.bench_with_input(
            BenchmarkId::new("baseline_process", trade_count),
            trade_count,
            |b, _| {
                b.iter(|| {
                    let mut processor = RangeBarProcessor::new(250).unwrap();
                    let bars = processor
                        .process_agg_trade_records(black_box(&trades))
                        .unwrap();
                    black_box(bars);
                });
            },
        );
    }

    group.finish();
}

/// Multi-symbol concurrent processing (bounded to 4 symbols)
fn bench_concurrent_multi_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_multi_symbol");
    group.sample_size(20);

    for trade_count in [100_000, 250_000].iter() {
        let symbol_trades = vec![
            create_test_trades(*trade_count, 50000.0, 100.0),
            create_test_trades(*trade_count, 3000.0, 50.0),
            create_test_trades(*trade_count, 1.0, 0.1),
            create_test_trades(*trade_count, 0.5, 0.05),
        ];

        group.bench_with_input(
            BenchmarkId::new("concurrent_4_symbols", trade_count),
            trade_count,
            |b, _| {
                b.iter(|| {
                    let mut handles = vec![];
                    for trades in symbol_trades.iter().take(4) {
                        let trades = trades.clone();
                        handles.push(thread::spawn(move || {
                            let mut processor = RangeBarProcessor::new(250).unwrap();
                            processor
                                .process_agg_trade_records(black_box(&trades))
                                .unwrap()
                        }));
                    }

                    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// 8-thread parallel batch processing (bounded to 8 threads)
fn bench_parallel_batch_8threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_8threads");
    group.sample_size(20);

    for trade_count in [100_000, 500_000].iter() {
        let trades = create_test_trades(*trade_count, 50000.0, 100.0);

        group.bench_with_input(
            BenchmarkId::new("batch_split_8threads", trade_count),
            trade_count,
            |b, _| {
                b.iter(|| {
                    let chunk_size = (*trade_count / 8).max(1);
                    let mut handles = vec![];
                    let mut chunk_count = 0;
                    
                    for chunk in trades.chunks(chunk_size) {
                        if chunk_count >= 8 { break; }
                        chunk_count += 1;
                        
                        let chunk = chunk.to_vec();
                        handles.push(thread::spawn(move || {
                            let mut processor = RangeBarProcessor::new(250).unwrap();
                            processor
                                .process_agg_trade_records(black_box(&chunk))
                                .unwrap()
                        }));
                    }

                    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    let all_bars: Vec<_> = results.into_iter().flatten().collect();
                    black_box(all_bars);
                });
            },
        );
    }

    group.finish();
}

/// Stress test: High-frequency stream at 100K trades/sec
fn bench_high_frequency_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_frequency_stress");
    group.sample_size(10);

    let stress_trades = create_test_trades(500_000, 50000.0, 200.0);

    group.bench_function("stress_500k_high_volatility", |b| {
        b.iter(|| {
            let mut processor = RangeBarProcessor::new(250).unwrap();
            let bars = processor
                .process_agg_trade_records(black_box(&stress_trades))
                .unwrap();
            black_box(bars);
        });
    });

    let parallel_stress_trades = vec![
        create_test_trades(125_000, 50000.0, 200.0),
        create_test_trades(125_000, 3000.0, 100.0),
        create_test_trades(125_000, 1.0, 0.2),
        create_test_trades(125_000, 0.5, 0.1),
    ];

    group.bench_function("stress_4parallel_25k_each", |b| {
        b.iter(|| {
            let mut handles = vec![];
            for trades in parallel_stress_trades.iter().take(4) {
                let trades = trades.clone();
                handles.push(thread::spawn(move || {
                    let mut processor = RangeBarProcessor::new(250).unwrap();
                    processor
                        .process_agg_trade_records(black_box(&trades))
                        .unwrap()
                }));
            }

            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Shared processor with lock contention (bounded to 8 threads)
fn bench_shared_processor_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_processor_contention");
    group.sample_size(15);

    for thread_count in [2, 4, 8].iter() {
        let trades = create_test_trades(50_000, 50000.0, 100.0);
        let trades_per_thread = trades.len() / *thread_count;

        group.bench_with_input(
            BenchmarkId::new("shared_processor", thread_count),
            thread_count,
            |b, _| {
                b.iter(|| {
                    let shared_processor = Arc::new(Mutex::new(
                        RangeBarProcessor::new(250).unwrap(),
                    ));

                    let mut handles = vec![];
                    for i in 0..*thread_count.min(&8) {
                        let processor = Arc::clone(&shared_processor);
                        let start = i * trades_per_thread;
                        let end = (start + trades_per_thread).min(trades.len());
                        let trades_slice = trades[start..end].to_vec();

                        handles.push(thread::spawn(move || {
                            let mut proc = processor.lock().unwrap();
                            let _ = proc.process_agg_trade_records(black_box(&trades_slice));
                        }));
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Sequential trade processing from a channel
/// Tests the overhead of channel-based trade submission vs direct batch processing
fn bench_producer_consumer_pattern(c: &mut Criterion) {
    use std::sync::mpsc;

    let mut group = c.benchmark_group("producer_consumer_pattern");
    group.sample_size(15);

    for trade_count in [100_000, 250_000].iter() {
        let trades = create_test_trades(*trade_count, 50000.0, 100.0);

        group.bench_with_input(
            BenchmarkId::new("channel_submission", trade_count),
            trade_count,
            |b, _| {
                b.iter(|| {
                    let (tx, rx) = mpsc::channel();

                    // Single producer thread sends trades through channel
                    let trades_clone = trades.clone();
                    let producer = thread::spawn(move || {
                        for trade in trades_clone {
                            tx.send(trade).ok();
                        }
                    });

                    // Collect from channel
                    let all_trades: Vec<_> = rx.into_iter().collect();

                    // Process collected trades
                    let mut processor = RangeBarProcessor::new(250).unwrap();
                    let bars = processor
                        .process_agg_trade_records(black_box(&all_trades))
                        .unwrap();
                    black_box(bars);

                    producer.join().ok();
                });
            },
        );
    }

    group.finish();
}

/// Memory efficiency under concurrent load (bounded to 8 threads)
fn bench_concurrent_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_memory_patterns");
    group.sample_size(10);

    let batch_configs = vec![
        (4, 50_000),
        (8, 50_000),
        (4, 250_000),
    ];

    for (thread_count, trades_per_thread) in batch_configs {
        let trades_vec: Vec<_> = (0..thread_count.min(8))
            .map(|_| create_test_trades(trades_per_thread, 50000.0, 100.0))
            .collect();

        let label = format!("{}t_{}k", thread_count, trades_per_thread / 1000);

        group.bench_with_input(
            BenchmarkId::new("memory_concurrent", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let mut handles = vec![];
                    for trades in trades_vec.iter() {
                        let trades = trades.clone();
                        handles.push(thread::spawn(move || {
                            let mut processor = RangeBarProcessor::new(250).unwrap();
                            processor
                                .process_agg_trade_records(black_box(&trades))
                                .unwrap()
                        }));
                    }

                    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_threaded_baseline,
    bench_concurrent_multi_symbol,
    bench_parallel_batch_8threads,
    bench_high_frequency_stress,
    bench_shared_processor_contention,
    bench_producer_consumer_pattern,
    bench_concurrent_memory_patterns,
);
criterion_main!(benches);
