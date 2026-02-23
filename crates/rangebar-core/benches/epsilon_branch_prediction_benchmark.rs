//! Epsilon branch prediction optimization benchmark (Issue #96 Task #177)
//!
//! Measures the performance improvement from branchless conditional accumulation
//! vs traditional if/else branches in buy/sell volume accumulation.
//!
//! **Hypothesis**: Branchless accumulation reduces branch mispredictions in tight
//! loops, improving throughput by 0.8-1.8% on real market data with alternating
//! buy/sell patterns.

use rangebar_core::fixed_point::FixedPoint;
use std::time::Instant;

/// Simple TradeSnapshot mock for benchmarking
#[derive(Clone, Copy)]
struct TradeSnapshot {
    price: FixedPoint,
    volume: FixedPoint,
    is_buyer_maker: bool,
}

/// Traditional branchful accumulation (baseline)
#[inline(never)]
fn accumulate_buy_sell_branchful(trades: &[TradeSnapshot]) -> (f64, f64) {
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

/// Branchless accumulation using arithmetic selection (optimized)
#[inline(never)]
fn accumulate_buy_sell_branchless(trades: &[TradeSnapshot]) -> (f64, f64) {
    let mut buy_vol = 0.0;
    let mut sell_vol = 0.0;

    // Process pairs for ILP + branchless accumulation
    let pairs = trades.len() / 2;
    for i in 0..pairs {
        let t1 = &trades[i * 2];
        let t2 = &trades[i * 2 + 1];

        let vol1 = t1.volume.to_f64();
        let vol2 = t2.volume.to_f64();

        // Branchless selection: Convert bool to f64 (1.0 or 0.0)
        let is_buyer_mask1 = t1.is_buyer_maker as u32 as f64;
        let is_buyer_mask2 = t2.is_buyer_maker as u32 as f64;

        // Arithmetic selection (no branches)
        sell_vol += vol1 * is_buyer_mask1;
        buy_vol += vol1 * (1.0 - is_buyer_mask1);

        sell_vol += vol2 * is_buyer_mask2;
        buy_vol += vol2 * (1.0 - is_buyer_mask2);
    }

    // Scalar remainder for odd-length arrays
    if trades.len() % 2 == 1 {
        let t = &trades[trades.len() - 1];
        let vol = t.volume.to_f64();
        let is_buyer_mask = t.is_buyer_maker as u32 as f64;

        sell_vol += vol * is_buyer_mask;
        buy_vol += vol * (1.0 - is_buyer_mask);
    }

    (buy_vol, sell_vol)
}

/// Generate test data with various branch prediction patterns
fn generate_trades_random_pattern(count: usize) -> Vec<TradeSnapshot> {
    // Pseudo-random pattern using XorShift-like behavior
    // This simulates real market data where buy/sell alternates unpredictably
    let mut pattern = vec![];
    let mut state = 12345u64;

    for _ in 0..count {
        // XorShift31 pattern for pseudo-random branch prediction challenge
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let is_buyer_maker = (state & 1) == 1;
        let volume = FixedPoint(10000000000 + (state as i64 % 5000000000)); // ~100-150 with 8 decimal scale

        pattern.push(TradeSnapshot {
            price: FixedPoint(10000000000), // 100.0 with 8 decimal scale
            volume,
            is_buyer_maker,
        });
    }

    pattern
}

/// Generate test data with alternating pattern (worst case for branch prediction)
fn generate_trades_alternating_pattern(count: usize) -> Vec<TradeSnapshot> {
    // Strict alternation: buy, sell, buy, sell...
    // This maximizes branch misprediction for traditional if/else
    let mut trades = vec![];
    for i in 0..count {
        trades.push(TradeSnapshot {
            price: FixedPoint(10000000000), // 100.0 with 8 decimal scale
            volume: FixedPoint(10000000000), // 100.0 with 8 decimal scale
            is_buyer_maker: i % 2 == 0,
        });
    }
    trades
}

/// Benchmark function with timing
fn bench_pattern(name: &str, trades: &[TradeSnapshot], iterations: usize) {
    println!("\n{}:", name);

    // Warm up
    for _ in 0..10 {
        let _ = accumulate_buy_sell_branchful(trades);
        let _ = accumulate_buy_sell_branchless(trades);
    }

    // Benchmark branchful
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let result = accumulate_buy_sell_branchful(std::hint::black_box(trades));
        dummy += result.0 + result.1; // Ensure result is used
    }
    let branchful_duration = start.elapsed();
    let branchful_ns = branchful_duration.as_nanos() as f64 / iterations as f64;
    std::hint::black_box(dummy); // Prevent optimization of dummy variable

    // Benchmark branchless
    let start = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..iterations {
        let result = accumulate_buy_sell_branchless(std::hint::black_box(trades));
        dummy += result.0 + result.1; // Ensure result is used
    }
    let branchless_duration = start.elapsed();
    let branchless_ns = branchless_duration.as_nanos() as f64 / iterations as f64;
    std::hint::black_box(dummy); // Prevent optimization of dummy variable

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
    println!("=== Epsilon Branch Prediction Optimization Benchmark ===");
    println!("Issue #96 Task #177: Branchless conditional accumulation\n");

    // Test sizes: 50, 100, 250, 500 trades (typical lookback window range)
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
    }

    println!("\n{}", "=".repeat(60));
    println!("\n=== Summary ===");
    println!("Branchless accumulation should show 0.8-1.8% improvement on average");
    println!("Especially on alternating patterns (branch misprediction stress)");
    println!("\nNote: Results depend on CPU architecture and cache state");
    println!("      Run multiple times for consistency");
}
