//! Kyle Lambda branchless optimization benchmark (Issue #96 Task #184)
//!
//! Measures the performance improvement from branchless conditional accumulation
//! vs traditional if/else branches in buy/sell volume accumulation within Kyle Lambda.
//!
//! **Hypothesis**: Branchless accumulation reduces branch mispredictions, improving
//! throughput by 1.2-2% on real market data with alternating buy/sell patterns.
//!
//! **Test Patterns**:
//! - Random: Realistic market data with pseudo-random buy/sell distribution
//! - Alternating: Stress test for branch prediction (worst case)
//! - Consistent: Best case for branch predictor (easy pattern)

use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

/// Simple TradeSnapshot mock for benchmarking
#[derive(Clone, Copy)]
struct TradeSnapshot {
    price: FixedPoint,
    volume: FixedPoint,
    is_buyer_maker: bool,
}

/// Branchful version: traditional if/else (baseline)
/// This is what Kyle Lambda used to do (before branchless optimization)
#[inline(never)]
fn accumulate_volumes_branchful(trades: &[TradeSnapshot]) -> (f64, f64) {
    let mut buy_vol = 0.0;
    let mut sell_vol = 0.0;

    for trade in trades {
        let vol = trade.volume.to_f64();
        if trade.is_buyer_maker {
            sell_vol += vol;
        } else {
            buy_vol += vol;
        }
    }

    (buy_vol, sell_vol)
}

/// Branchless version: arithmetic selection (optimized)
/// Uses boolean-to-mask conversion and arithmetic operations
#[inline(never)]
fn accumulate_volumes_branchless(trades: &[TradeSnapshot]) -> (f64, f64) {
    let mut buy_vol = 0.0;
    let mut sell_vol = 0.0;

    // Process in pairs for instruction-level parallelism
    let pairs = trades.len() / 2;
    for i in 0..pairs {
        let t0 = &trades[i * 2];
        let t1 = &trades[i * 2 + 1];

        let vol0 = t0.volume.to_f64();
        let vol1 = t1.volume.to_f64();

        // Branchless: Convert bool to mask (0.0 or 1.0)
        let is_buyer_mask0 = t0.is_buyer_maker as u32 as f64;
        let is_buyer_mask1 = t1.is_buyer_maker as u32 as f64;

        // Arithmetic selection (no branches)
        // when is_buyer_maker==true, add to sell_vol; else add to buy_vol
        buy_vol += vol0 * (1.0 - is_buyer_mask0);
        sell_vol += vol0 * is_buyer_mask0;

        buy_vol += vol1 * (1.0 - is_buyer_mask1);
        sell_vol += vol1 * is_buyer_mask1;
    }

    // Scalar remainder for odd-length arrays
    if trades.len() % 2 == 1 {
        let t = &trades[trades.len() - 1];
        let vol = t.volume.to_f64();
        let is_buyer_mask = t.is_buyer_maker as u32 as f64;

        buy_vol += vol * (1.0 - is_buyer_mask);
        sell_vol += vol * is_buyer_mask;
    }

    (buy_vol, sell_vol)
}

/// Generate test data with random buy/sell pattern (realistic market)
fn generate_trades_random_pattern(count: usize) -> Vec<TradeSnapshot> {
    let mut pattern = vec![];
    let mut state = 12345u64;

    for _ in 0..count {
        // XorShift31 pattern for pseudo-random distribution
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let is_buyer_maker = (state & 1) == 1;
        let volume = FixedPoint(10000000000 + (state as i64 % 5000000000)); // ~100-150

        pattern.push(TradeSnapshot {
            price: FixedPoint(10000000000), // 100.0
            volume,
            is_buyer_maker,
        });
    }

    pattern
}

/// Generate test data with alternating buy/sell (worst case for branch prediction)
fn generate_trades_alternating_pattern(count: usize) -> Vec<TradeSnapshot> {
    let mut trades = vec![];
    for i in 0..count {
        trades.push(TradeSnapshot {
            price: FixedPoint(10000000000),
            volume: FixedPoint(10000000000),
            is_buyer_maker: i % 2 == 0,
        });
    }
    trades
}

/// Generate test data with consistent pattern (best case for branch prediction)
fn generate_trades_consistent_pattern(count: usize) -> Vec<TradeSnapshot> {
    let mut trades = vec![];
    let block_size = 10;
    for i in 0..count {
        // 10 sells, then 10 buys, then 10 sells, etc.
        let is_buyer = (i / block_size) % 2 == 0;
        trades.push(TradeSnapshot {
            price: FixedPoint(10000000000),
            volume: FixedPoint(10000000000),
            is_buyer_maker: is_buyer,
        });
    }
    trades
}

/// Benchmark function with timing
fn bench_pattern(name: &str, trades: &[TradeSnapshot], iterations: usize) {
    println!("\n{}:", name);

    // Warm up
    for _ in 0..10 {
        let _ = accumulate_volumes_branchful(trades);
        let _ = accumulate_volumes_branchless(trades);
    }

    // Benchmark branchful (baseline)
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let result = accumulate_volumes_branchful(std::hint::black_box(trades));
        dummy += result.0 + result.1;
    }
    let branchful_duration = start.elapsed();
    let branchful_ns = branchful_duration.as_nanos() as f64 / iterations as f64;
    std::hint::black_box(dummy);

    // Benchmark branchless (optimized)
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let result = accumulate_volumes_branchless(std::hint::black_box(trades));
        dummy += result.0 + result.1;
    }
    let branchless_duration = start.elapsed();
    let branchless_ns = branchless_duration.as_nanos() as f64 / iterations as f64;
    std::hint::black_box(dummy);

    let improvement = (branchful_ns - branchless_ns) / branchful_ns * 100.0;

    println!("  Branchful:  {:.2} ns/iter", branchful_ns);
    println!("  Branchless: {:.2} ns/iter", branchless_ns);
    if improvement > 0.0 {
        println!("  Improvement: {:.2}% faster", improvement);
    } else {
        println!("  Regression: {:.2}% slower", -improvement);
    }
}

fn main() {
    println!("=== Kyle Lambda Branchless Optimization Benchmark ===");
    println!("Issue #96 Task #184: Branchless volume accumulation\n");

    // Test window sizes: 50, 100, 250, 500 trades (typical lookback range)
    for count in [50, 100, 250, 500].iter() {
        let iterations = match count {
            50 => 10000,
            100 => 5000,
            250 => 2000,
            500 => 1000,
            _ => 100,
        };

        println!("\n{}", "=".repeat(60));
        println!("Window size: {} trades ({} iterations)", count, iterations);
        println!("{}", "=".repeat(60));

        // Random pattern (realistic market data)
        let trades_random = generate_trades_random_pattern(*count);
        bench_pattern("Random pattern (realistic)", &trades_random, iterations);

        // Alternating pattern (worst case for branch prediction)
        let trades_alt = generate_trades_alternating_pattern(*count);
        bench_pattern("Alternating pattern (worst case)", &trades_alt, iterations);

        // Consistent pattern (best case for branch prediction)
        let trades_consistent = generate_trades_consistent_pattern(*count);
        bench_pattern(
            "Consistent pattern (best case)",
            &trades_consistent,
            iterations,
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("\n=== Summary ===");
    println!("Branchless accumulation should show 1.2-2% improvement on average");
    println!("Especially on random/alternating patterns (branch misprediction stress)");
    println!("\nNote: Results depend on CPU architecture and cache state");
    println!("      Run multiple times for consistency");
}
