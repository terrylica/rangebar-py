//! Conversion caching benchmark (Issue #96 Task #188)
//!
//! Measures the performance improvement from caching FixedPoint-to-f64 conversions
//! in statistical features computation.
//!
//! **Hypothesis**: Caching volume conversions in SmallVec reduces redundant FixedPoint.to_f64()
//! calls, improving throughput by 3-5% on typical bar feature computation.

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create a test trade with realistic values
fn create_test_trade(idx: usize, price: f64, volume: f64, is_buyer_maker: bool) -> AggTrade {
    AggTrade {
        agg_trade_id: idx as i64,
        price: FixedPoint((price * 1e8) as i64),
        volume: FixedPoint((volume * 1e8) as i64),
        first_trade_id: idx as i64,
        last_trade_id: idx as i64,
        timestamp: 1000000 + (idx as i64),
        is_buyer_maker,
        is_best_match: None,
    }
}

/// Generate test data with realistic price/volume movements
fn generate_test_trades(count: usize) -> Vec<AggTrade> {
    let mut trades = vec![];
    let mut price = 100.0;
    let mut state = 12345u64;

    for idx in 0..count {
        // Pseudo-random walk for price
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let price_change = ((state as i32 as f64) / 1e9) * 0.1;
        price = (price + price_change).max(50.0).min(150.0);

        let volume = 0.5 + (state as f64 % 2.0);
        trades.push(create_test_trade(
            idx,
            price,
            volume,
            (state & 1) == 0,
        ));
    }

    trades
}

/// Benchmark function
fn bench_conversion_caching(count: usize, iterations: usize) {
    println!("\nBar size: {} trades ({} iterations)", count, iterations);

    let trades = generate_test_trades(count);
    
    // Warm-up
    for _ in 0..10 {
        let _ = rangebar_core::intrabar::features::compute_intra_bar_features(&trades);
    }

    // Benchmark
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let features = rangebar_core::intrabar::features::compute_intra_bar_features(
            std::hint::black_box(&trades)
        );
        dummy += features.intra_ofi.unwrap_or(0.0) + features.intra_intensity.unwrap_or(0.0);
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_bar = duration.as_nanos() as f64 / iterations as f64;
    println!("  Time: {:.2} ns/bar", ns_per_bar);
}

fn main() {
    println!("=== Conversion Caching Benchmark (Task #188) ===");
    println!("Measures performance improvement from caching volume conversions\n");

    // Test typical bar sizes
    for count in [50, 100, 250, 500].iter() {
        let iterations = match count {
            50 => 10000,
            100 => 5000,
            250 => 2000,
            500 => 1000,
            _ => 100,
        };

        bench_conversion_caching(*count, iterations);
    }

    println!("\n=== Summary ===");
    println!("Conversion caching should show 3-5% improvement by:");
    println!("1. Caching volume.to_f64() in SmallVec during Pass 1");
    println!("2. Reusing cached values in Pass 2 (moment computation)");
    println!("3. Eliminating redundant FixedPoint conversions across two loops");
}
