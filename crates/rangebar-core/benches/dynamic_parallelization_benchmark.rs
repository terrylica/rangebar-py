//! Dynamic parallelization benchmark (Issue #96 Task #189)
//!
//! Measures performance improvement from CPU-aware adaptive dispatch of Tier 2/3
//! inter-bar feature computation based on window size and available CPUs.

use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::interbar::InterBarConfig;
use rangebar_core::types::AggTrade;
use std::time::Instant;

/// Create test trade snapshot
fn create_test_trade(idx: usize, price: f64, is_buyer_maker: bool) -> AggTrade {
    AggTrade {
        agg_trade_id: idx as i64,
        price: FixedPoint((price * 1e8) as i64),
        volume: FixedPoint((10_000_000) as i64),  // 0.1 BTC
        first_trade_id: idx as i64,
        last_trade_id: idx as i64,
        timestamp: 1000000 + (idx as i64),
        is_buyer_maker,
        is_best_match: None,
    }
}

/// Generate synthetic trade data with varying sizes
fn generate_trades(count: usize) -> Vec<AggTrade> {
    (0..count)
        .map(|i| {
            let price = 100.0 + (i as f64 * 0.001);
            create_test_trade(i, price, i % 2 == 0)
        })
        .collect()
}

/// Benchmark feature computation at different window sizes
fn bench_dynamic_dispatch(window_size: usize, iterations: usize) {
    println!("\n=== Window size: {} trades ({} iterations) ===", window_size, iterations);

    let trades = generate_trades(window_size);

    // Build history with inter-bar config (Tier 2 + 3 enabled)
    let config = InterBarConfig {
        mode: rangebar_core::interbar::LookbackMode::FixedTradeCount(window_size),
        compute_tier1: true,
        compute_tier2: true,
        compute_tier3: true,
        enable_entropy_cache: false,
    };

    let mut history = rangebar_core::interbar::TradeHistory::new(config);

    // Add trades to history
    for (i, trade) in trades.iter().enumerate() {
        history.add_trade(trade.clone());
        if i < window_size - 1 {
            continue;  // Skip compute until enough trades
        }
        
        // Warm up
        if i < window_size + 10 {
            let _ = history.compute_features(trade.timestamp);
        }
    }

    // Benchmark
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let last_trade = trades.last().unwrap();
        let features = history.compute_features(last_trade.timestamp);
        dummy += features.lookback_kyle_lambda.unwrap_or(0.0)
            + features.lookback_hurst.unwrap_or(0.0);
    }
    let duration = start.elapsed();
    std::hint::black_box(dummy);

    let ns_per_bar = duration.as_nanos() as f64 / iterations as f64;
    println!("  Time: {:.2} ns/computation", ns_per_bar);
}

fn main() {
    println!("=== Dynamic Parallelization Dispatch Benchmark (Task #189) ===");
    println!("Measures adaptive Tier 2/3 parallelization based on window size");
    println!("CPU count: {}\n", num_cpus::get());

    // Test different window sizes to demonstrate dispatch behavior
    // Small windows (<80 trades): Serial
    // Medium windows (80-150 trades): Parallel Tier 2, Serial Tier 3
    // Large windows (>150 trades): Parallel both

    let test_sizes = [
        (50, "Small (serial both)"),
        (100, "Medium (parallel Tier 2)"),
        (250, "Large (parallel both)"),
        (500, "XL (parallel both)"),
    ];

    for (size, label) in &test_sizes {
        let iterations = match size {
            50 => 1000,
            100 => 500,
            250 => 200,
            500 => 100,
            _ => 50,
        };
        
        println!("\n{}", label);
        bench_dynamic_dispatch(*size, iterations);
    }

    println!("\n=== Summary ===");
    println!("Dynamic dispatch should show improvement by:");
    println!("1. Using separate thresholds for Tier 2 (80) and Tier 3 (150)");
    println!("2. Avoiding parallelization overhead for small windows");
    println!("3. CPU-aware scaling to avoid oversubscription");
}
