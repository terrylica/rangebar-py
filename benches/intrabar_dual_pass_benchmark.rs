//! Benchmark: Intra-bar dual-pass elimination optimization
//!
//! Issue #96 Task #166: Measure performance impact of fusing OHLC tracking
//! into moment computation to eliminate redundant loop iteration.
//!
//! Before: Pass 1 (OHLC + volumes) + Pass 2 (moments) = 2 iterations
//! After: Pass 1 (volumes only) + Pass 2 (moments + OHLC) = 2 iterations, but OHLC moved to data dependency
//!
//! Expected improvement: 2-4% by reducing memory pressure and CPU cache misses

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::intrabar::features::compute_intra_bar_features;

fn create_test_trade(price: f64, volume: f64, timestamp: i64, is_buyer_maker: bool) -> AggTrade {
    AggTrade {
        agg_trade_id: timestamp,
        price: FixedPoint((price * 1e8) as i64),
        volume: FixedPoint((volume * 1e8) as i64),
        first_trade_id: timestamp,
        last_trade_id: timestamp,
        timestamp,
        is_buyer_maker,
        is_best_match: None,
    }
}

fn benchmark_intrabar_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("intrabar_features");

    // Test with various trade counts (50, 100, 200, 500)
    for count in [50, 100, 200, 500].iter() {
        let trades: Vec<AggTrade> = (0..*count)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.05 % 2.0 - 1.0); // Oscillating prices
                let volume = 1.0 + (i as f64 * 0.1 % 0.5);
                let is_buyer = i % 2 == 0;
                create_test_trade(price, volume, i as i64 * 1000, is_buyer)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_trades", count)),
            count,
            |b, _| {
                b.iter(|| {
                    compute_intra_bar_features(black_box(&trades))
                });
            },
        );
    }

    // Additional stress test: 1000 trades
    let large_trades: Vec<AggTrade> = (0..1000)
        .map(|i| {
            let price = 100.0 + ((i as f64 * 0.01).sin() * 2.0);
            let volume = 1.0 + (i as f64 * 0.05 % 1.0);
            let is_buyer = i % 3 == 0;
            create_test_trade(price, volume, i as i64 * 500, is_buyer)
        })
        .collect();

    group.bench_with_input(
        BenchmarkId::from_parameter("1000_trades"),
        &1000,
        |b, _| {
            b.iter(|| {
                compute_intra_bar_features(black_box(&large_trades))
            });
        },
    );

    group.finish();
}

criterion_group!(benches, benchmark_intrabar_features);
criterion_main!(benches);
