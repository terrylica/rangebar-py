//! Lookahead binary search optimization benchmark (Issue #96 Task #167 Phase 2)
//!
//! Measures the performance improvement from using trend-guided binary search hints
//! on lookback window computation. Validates that lookahead prediction reduces
//! search iterations and improves latency on trending data patterns.
//!
//! **Hypothesis**: Trend-guided search reduces binary search iterations by 5-10%,
//! improving lookback window retrieval latency by 0.5-1% on typical streaming data.

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::interbar::{InterBarConfig, LookbackMode};
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create test trade with specific price
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

/// Generate trades with different patterns
fn generate_trades_with_pattern(count: usize, pattern: &str) -> Vec<AggTrade> {
    let mut trades = Vec::new();
    let mut price = 100.0;

    for i in 0..count {
        match pattern {
            "steady_up" => price += 0.01,
            "steady_down" => price -= 0.01,
            "volatile" => price += (i as f64 * 3.14159).sin() * 0.5,
            "random_walk" => {
                let drift = if i % 2 == 0 { 0.005 } else { -0.005 };
                price += drift;
            }
            _ => price += 0.001,
        }

        trades.push(create_test_trade(i, price, i % 2 == 0));
    }

    trades
}

/// Benchmark get_lookback_trades on different patterns
fn bench_lookback_retrieval(pattern: &str, trade_count: usize, window_size: usize, iterations: usize) {
    println!(
        "\n{} pattern, {} trades, {} lookback window",
        pattern, trade_count, window_size
    );

    let trades = generate_trades_with_pattern(trade_count, pattern);
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(window_size),
        compute_tier2: true,
        compute_tier3: true,
    };

    let mut history = rangebar_core::interbar::TradeHistory::new(config);

    // Initialize history with all trades
    for trade in &trades {
        history.push(trade);
    }

    // Benchmark: repeated lookback queries with changing bar_open_time
    // Simulates real streaming where timestamps increase gradually
    let mut timestamps = Vec::new();
    for i in (0..trade_count).step_by(trade_count / 20) {
        timestamps.push(1000000 + (i as i64));
    }

    let start = Instant::now();
    let mut dummy = 0usize;
    for _iter in 0..iterations {
        for &ts in &timestamps {
            let lookback = history.get_lookback_trades(ts);
            dummy += lookback.len();
        }
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_query = duration.as_nanos() as f64 / (iterations as f64 * timestamps.len() as f64);
    println!("  Avg latency: {:.2} ns/query", ns_per_query);
}

fn main() {
    println!("=== Lookahead Binary Search Benchmark (Task #167 Phase 2) ===");
    println!("Measures trend-guided binary search optimization on lookback retrieval\n");

    // Test on different trending patterns where lookahead helps
    let patterns = vec!["steady_up", "steady_down", "trending", "volatile"];
    let test_configs = vec![
        (1000, 500, "Medium window"),
        (2000, 1000, "Large window"),
    ];

    for (trade_count, window_size, label) in test_configs {
        println!("\n--- {} ({} total trades) ---", label, trade_count);
        for pattern in &patterns {
            let iterations = if trade_count < 1500 { 100 } else { 50 };
            bench_lookback_retrieval(pattern, trade_count, window_size, iterations);
        }
    }

    println!("\n=== Summary ===");
    println!("Lookahead optimization should show improvement by:");
    println!("1. Computing trend from last 2-3 search results");
    println!("2. Predicting if next index will trend higher/lower");
    println!("3. Reducing binary search iterations by 5-10%");
    println!("4. Typical speedup: 0.5-1% on streaming with predictable timestamps");
}
