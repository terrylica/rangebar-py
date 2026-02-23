// Detailed profiling analysis for inter-bar features
// Measures time spent in each feature computation individually
//
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
// Run with: cargo bench --bench profiling_analysis 2>&1 | grep -A 5 "profiling"

use criterion::{criterion_group, criterion_main, Criterion, black_box, BenchmarkId};
use rangebar::{AggTrade, FixedPoint, RangeBarProcessor};
use std::time::Instant;

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

fn bench_profiling_analysis(c: &mut Criterion) {
    eprintln!("\n=== Inter-Bar Feature Profiling Analysis ===\n");

    for count in [1_000, 5_000, 10_000].iter() {
        let trades = create_test_trades(*count, 50000.0, 100.0);

        c.bench_with_input(
            BenchmarkId::new("feature_computation", count),
            count,
            |b, _| {
                b.iter(|| {
                    let mut proc = RangeBarProcessor::new(250).unwrap();
                    let bars = proc.process_agg_trade_records(black_box(&trades)).unwrap();
                    black_box(bars);
                });
            },
        );

        // Also do a quick measurement outside criterion
        let start = Instant::now();
        for _ in 0..10 {
            let mut proc = RangeBarProcessor::new(250).unwrap();
            let _ = proc.process_agg_trade_records(&trades);
        }
        let avg_time = start.elapsed() / 10;
        eprintln!("Manual timing ({} trades): {:?}", count, avg_time);
    }

    eprintln!("\nPerformance target: 1M trades < 100ms");
}

criterion_group!(benches, bench_profiling_analysis);
criterion_main!(benches);
