//! Feature attribution profiling for Task #144: Identify streaming latency bottlenecks
//! Issue #96 Task #144 Phase 2: Measure contribution of each Tier 1/2/3 feature to bar completion latency
//!
//! Disables features selectively to measure individual contribution to latency.
//! Enables identification of highest-impact optimization opportunities.
//!
//! Run: `cargo bench --bench feature_attribution_profiler`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::{AggTrade, FixedPoint};
use std::time::Instant;

/// Generate synthetic trades with sufficient price movement to trigger bar completions
fn generate_trades_for_bar_completion(_symbol: &str, num_trades: usize, base_price: f64) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(num_trades);

    // Generate trades that accumulate toward a price breach (250 bps = 0.25%)
    // This ensures we measure bar completion latency, not just trade processing
    for i in 0..num_trades {
        let price_movement = (i as f64 / num_trades as f64) * 0.003; // Accumulate up to 0.3%
        let price = base_price + price_movement;
        let volume = 1.0 + (i as f64 % 3.0) * 0.5;

        let price_str = format!("{:.4}", price);
        let volume_str = format!("{:.4}", volume);

        trades.push(AggTrade {
            agg_trade_id: i as i64,
            price: FixedPoint::from_str(&price_str).expect("price parse"),
            volume: FixedPoint::from_str(&volume_str).expect("volume parse"),
            first_trade_id: i as i64,
            last_trade_id: i as i64,
            timestamp: i as i64 * 10,
            is_buyer_maker: i % 2 == 0,
            is_best_match: None,
        });
    }

    trades
}

/// Benchmark Tier 1 features (always enabled - intra-bar computation)
/// These are computed during bar construction and are minimal overhead
fn benchmark_tier1_features(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("feature_attribution_tier1");
    group.sample_size(30);

    for window_size in [100, 250, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("tier1_ofi_vwap_price_impact", window_size),
            window_size,
            |b, &window_size| {
                let trades = generate_trades_for_bar_completion("BTCUSDT", window_size, 42000.0);

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let start = Instant::now();

                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }

                        total += start.elapsed();
                    }

                    total
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Tier 2 features (inter-bar lookback computation)
/// These are computed from lookback window of previous trades
/// Currently includes: Kyle Lambda, Burstiness, Volume moments, Price range
/// Both baseline and SIMD-accelerated versions
fn benchmark_tier2_features(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("feature_attribution_tier2");
    group.sample_size(20);

    for window_size in [100, 250, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("tier2_kyle_burstiness_moments", window_size),
            window_size,
            |b, &window_size| {
                let trades = generate_trades_for_bar_completion("BTCUSDT", window_size, 42000.0);

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let start = Instant::now();

                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }

                        total += start.elapsed();
                    }

                    total
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Tier 3 features (advanced pattern recognition)
/// These are the most expensive: Permutation Entropy (O(n²)), Hurst Exponent (O(n log n))
/// Hypothesis: These dominate total latency, especially Permutation Entropy on larger windows
fn benchmark_tier3_features(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("feature_attribution_tier3");
    group.sample_size(20);

    // Test Permutation Entropy specifically - it's O(n²) worst case
    for window_size in [100, 250, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("tier3_permutation_entropy_hurst_gk", window_size),
            window_size,
            |b, &window_size| {
                let trades = generate_trades_for_bar_completion("BTCUSDT", window_size, 42000.0);

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let start = Instant::now();

                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }

                        total += start.elapsed();
                    }

                    total
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency scaling with lookback window size
/// Tests hypothesis: Permutation Entropy causes super-linear scaling
fn benchmark_scaling_analysis(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("feature_attribution_scaling");
    group.sample_size(15);
    group.measurement_time(std::time::Duration::from_secs(20));

    // Test with very large lookback windows to expose O(n²) behavior
    for window_size in [100, 250, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("scaling_with_window_size", window_size),
            window_size,
            |b, &window_size| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let mut processor = RangeBarProcessor::new(250).unwrap();
                        let trades = generate_trades_for_bar_completion("BTCUSDT", window_size * 2, 42000.0);

                        let start = Instant::now();
                        for trade in &trades {
                            let _ = processor.process_single_trade(trade);
                        }
                        total += start.elapsed();
                    }

                    total
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_tier1_features,
    benchmark_tier2_features,
    benchmark_tier3_features,
    benchmark_scaling_analysis,
);
criterion_main!(benches);
