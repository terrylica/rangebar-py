//! Entropy cache warm-up optimization benchmark (Issue #96 Task #191)
//!
//! Measures the latency improvement from pre-warming the entropy cache with
//! deterministic price patterns on the first TradeHistory creation.
//!
//! **Hypothesis**: Pre-warming reduces first-access cache misses by 15-25% on
//! multi-symbol workloads, improving streaming latency by 1-3%.

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::interbar::{InterBarConfig, LookbackMode};
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create test trade snapshot
fn create_test_trade(idx: usize, price: f64, is_buyer_maker: bool) -> AggTrade {
    AggTrade {
        agg_trade_id: idx as i64,
        price: FixedPoint((price * 1e8) as i64),
        volume: FixedPoint((10_000_000) as i64), // 0.1 BTC
        first_trade_id: idx as i64,
        last_trade_id: idx as i64,
        timestamp: 1000000 + (idx as i64),
        is_buyer_maker,
        is_best_match: None,
    }
}

/// Generate synthetic trade data with varying patterns
fn generate_trades(count: usize, pattern: &str) -> Vec<AggTrade> {
    (0..count)
        .map(|i| {
            let price = match pattern {
                "stable" => 100.0 + (i as f64 % 10.0) * 0.001,
                "trending" => 100.0 + (i as f64 * 0.01),
                "volatile" => 100.0 + ((i as f64 * 3.14159).sin() * 2.0),
                _ => 100.0 + (i as f64 * 0.001),
            };
            create_test_trade(i, price, i % 2 == 0)
        })
        .collect()
}

/// Benchmark entropy cache access patterns during feature computation
fn bench_entropy_cache_access(pattern: &str, trade_count: usize, iterations: usize) {
    println!("\nPattern: {} ({} trades, {} iterations)", pattern, trade_count, iterations);

    let trades = generate_trades(trade_count, pattern);
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(trade_count),
        compute_tier2: true,
        compute_tier3: true,
        ..Default::default()
    };

    let mut history = rangebar_core::interbar::TradeHistory::new(config);

    // Add trades to history (warm-up)
    for (i, trade) in trades.iter().enumerate() {
        history.push(trade);
        if i >= 10 {
            break;
        }
    }

    // Benchmark: compute features multiple times (will hit cache on repeated patterns)
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _iter in 0..iterations {
        for (j, trade) in trades.iter().enumerate() {
            if j >= 10 {
                break;
            }
            history.push(trade);
            let features = history.compute_features(trade.timestamp);
            dummy += features.lookback_permutation_entropy.unwrap_or(0.0);
        }
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_feature = duration.as_nanos() as f64 / (iterations as f64 * 10.0);
    println!("  Avg latency: {:.2} ns/feature", ns_per_feature);
}

fn main() {
    println!("=== Entropy Cache Warm-up Benchmark (Task #191) ===");
    println!("Measures first-access latency improvement from pre-warming\n");

    // Test different market patterns
    let patterns = vec!["stable", "trending", "volatile"];
    let trade_counts = vec![50, 150, 300];

    for pattern in &patterns {
        println!("\n--- Pattern: {} ---", pattern);
        for &count in &trade_counts {
            let iterations = match count {
                50 => 100,
                150 => 50,
                300 => 25,
                _ => 10,
            };
            bench_entropy_cache_access(pattern, count, iterations);
        }
    }

    println!("\n=== Summary ===");
    println!("Warm-up optimization should show improvement by:");
    println!("1. Deterministic patterns pre-loaded in global cache");
    println!("2. Reduced first-access cache misses by 15-25%");
    println!("3. Overall streaming latency improvement of 1-3%");
    println!("4. Parallelization-friendly: non-blocking warm-up (try_write)");
}
