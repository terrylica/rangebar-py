// Benchmarks for burstiness SIMD acceleration (Issue #96 Task #4)
// Run: cargo bench --bench burstiness_simd_bench
// PROCESS-STORM-OK

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rangebar_core::interbar_types::TradeSnapshot;
use rangebar_core::types::FixedPoint;

fn create_test_trades(n: usize) -> Vec<TradeSnapshot> {
    let mut trades = Vec::with_capacity(n);
    for i in 0..n {
        trades.push(TradeSnapshot {
            timestamp: (i as i64) * 100,
            price: FixedPoint((100.0 * 1e8) as i64),
            volume: FixedPoint((1.0 * 1e8) as i64),
            is_buyer_maker: i % 2 == 0,
            turnover: (100 * 1) as i128 * 100000000i128,
        });
    }
    trades
}

fn bench_burstiness_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("burstiness_simd");

    for window_size in &[10, 50, 100, 500, 1000] {
        let trades = create_test_trades(*window_size);
        let trade_refs: Vec<&TradeSnapshot> = trades.iter().collect();

        group.bench_with_input(
            BenchmarkId::new("compute", window_size),
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
    }

    group.finish();
}

criterion_group!(benches, bench_burstiness_performance);
criterion_main!(benches);
