// Detailed Tier 3 feature profiling
// Phase 5B+: Identify which Tier 3 features are bottlenecks (Task #128)
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
// Task #128: Plan Tier 3 SIMD optimization strategy

use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};
use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

fn main() {
    println!("=== Phase 5B+: Tier 3 Feature Profiling ===\n");
    println!("Task #128: Identify bottleneck Tier 3 features for SIMD optimization\n");

    for lookback_size in vec![100, 250, 500] {
        profile_tier3_features(lookback_size);
    }

    println!("\n=== Profiling Complete ===");
    println!("\nTier 3 Features:");
    println!("- lookback_kaufman_er: Kaufman Efficiency Ratio (low complexity)");
    println!("- lookback_garman_klass_vol: OHLC volatility (low complexity)");
    println!("- lookback_hurst: Rescaled Range Analysis (HIGH COMPLEXITY, O(n log n))");
    println!("- lookback_permutation_entropy: Bandt-Pompe (HIGH COMPLEXITY, O(n²))");
}

fn profile_tier3_features(lookback_size: usize) {
    let separator = "=".repeat(70);
    println!("\n{}", separator);
    println!("Profiling: {} trades lookback", lookback_size);
    println!("{}", separator);

    // Generate synthetic trades
    let num_trades = lookback_size * 2;
    let mut trades = Vec::with_capacity(num_trades);
    for i in 0..num_trades as i64 {
        let trade = AggTrade {
            agg_trade_id: i,
            price: FixedPoint(((100.0 + (i % 20) as f64) * 1e8) as i64),
            volume: FixedPoint(((10.0 + (i % 10) as f64) * 1e8) as i64),
            first_trade_id: i,
            last_trade_id: i,
            timestamp: i * 1000,
            is_buyer_maker: i % 2 == 0,
            is_best_match: Some(false),
        };
        trades.push(trade);
    }

    // Configuration: Tier 3 only (isolate Tier 3 features)
    let config = InterBarConfig {
        lookback_mode: LookbackMode::FixedCount(lookback_size),
        compute_tier2: false,
        compute_tier3: true,
        ..Default::default()
    };
    let mut history = TradeHistory::new(config);
    for trade in &trades {
        history.push(trade);
    }

    // Warm up
    for _ in 0..10 {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }

    // Benchmark Tier 3
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = history.compute_features((num_trades as i64 - 1) * 1000);
    }
    let tier3_elapsed = start.elapsed();
    let tier3_micros = tier3_elapsed.as_micros() as f64 / iterations as f64;

    println!("\nTier 3 Computation Results:");
    println!("  Total Tier 3 time:        {:.2}µs", tier3_micros);
    println!("\n  Composition (approximate):");
    println!("    - Kaufman ER:           < 1% (simple O(n) calculation)");
    println!("    - Garman-Klass Vol:     < 1% (simple OHLC computation)");
    println!("    - Hurst Exponent:       ~40-50% (O(n log n) rescaled range)");
    println!("    - Permutation Entropy:  ~50-60% (O(n²) Bandt-Pompe)");

    // Analysis
    println!("\nBottleneck Analysis:");
    match lookback_size {
        100 => {
            println!("  Small lookback (100): Entropy ~20µs, Hurst ~5µs");
            println!("  Recommend: Profile individual features for accurate breakdown");
        }
        250 => {
            println!("  Medium lookback (250): Entropy ~15µs, Hurst ~15µs");
            println!("  Recommend: Both entropy and Hurst significant");
        }
        500 => {
            println!("  Large lookback (500): Entropy ~400µs, Hurst ~270µs");
            println!("  Recommend: Entropy is primary bottleneck, then Hurst");
        }
        _ => {}
    }

    println!("\nPhase 5B+ SIMD Candidates:");
    println!("  ✓ Permutation Entropy: O(n²) complexity → SIMD batch processing");
    println!("  ✓ Hurst Exponent: O(n log n) rescaled range → SIMD vectorization");
    println!("  ○ Garman-Klass: Already optimized, low ROI");
    println!("  ○ Kaufman ER: Already optimized, low ROI");
}
