//! SmallVec buffer sizing profiling
//!
//! Measures trade distribution across bars to determine optimal SmallVec inline capacity.
//! Task #136: Profile and optimize SmallVec<[AggTrade; 512]> sizing.

use std::collections::HashMap;

#[cfg(feature = "test-utils")]
fn main() {
    use rangebar_core::test_data_loader::load_btcusdt_test_data;
    use rangebar_core::processor::RangeBarProcessor;

    println!("=== SmallVec Buffer Sizing Profile ===\n");

    // Load real trade data
    let trades = load_btcusdt_test_data().expect("Failed to load test data");

    if trades.is_empty() {
        eprintln!("No trades loaded");
        return;
    }

    println!("Processing {} trades...\n", trades.len());

    // Process trades and track buffer utilization
    let mut processor = RangeBarProcessor::new(250).expect("Failed to create processor");
    let mut max_trades_per_bar = 0usize;
    let mut trades_per_bar_distribution: HashMap<usize, usize> = HashMap::new();
    let mut heap_allocations = 0usize;
    let mut total_bars = 0usize;

    // Batch process and monitor
    let batch_size = 10000;
    for chunk in trades.chunks(batch_size) {
        let bars = processor
            .process_agg_trade_records(chunk)
            .expect("Processing failed");

        total_bars += bars.len();
    }

    // Manual reprocessing to count trades per bar precisely
    let mut processor = RangeBarProcessor::new(250).expect("Failed to create processor");
    let mut current_bar_trades = 0usize;

    for trade in &trades {
        // Track trades being accumulated
        current_bar_trades += 1;

        let bar_closed = processor
            .process_single_trade(trade)
            .expect("Processing failed");

        // When a bar closes (returns Some), record the trade count
        if bar_closed.is_some() {
            max_trades_per_bar = max_trades_per_bar.max(current_bar_trades);
            *trades_per_bar_distribution
                .entry(current_bar_trades)
                .or_insert(0) += 1;

            // Check if heap allocation would have occurred
            if current_bar_trades > 512 {
                heap_allocations += 1;
            }

            current_bar_trades = 0;
        }
    }

    // Calculate statistics
    let mut sorted_counts: Vec<_> = trades_per_bar_distribution.iter().collect();
    sorted_counts.sort_by_key(|&(count, _)| count);

    let total_count: usize = trades_per_bar_distribution.values().sum();
    let mut cumulative = 0usize;
    let mut p50_count = 0usize;
    let mut p90_count = 0usize;
    let mut p99_count = 0usize;

    for &(count, &freq) in &sorted_counts {
        cumulative += freq;
        let percentile = (cumulative as f64 / total_count as f64) * 100.0;

        if percentile >= 50.0 && p50_count == 0 {
            p50_count = *count;
        }
        if percentile >= 90.0 && p90_count == 0 {
            p90_count = *count;
        }
        if percentile >= 99.0 && p99_count == 0 {
            p99_count = *count;
        }
    }

    // Print results
    println!("Trade Distribution Summary:");
    println!("  Total bars: {}", total_bars);
    println!("  Max trades per bar: {}", max_trades_per_bar);
    println!("  Heap allocations (> 512): {}", heap_allocations);
    println!("  Heap allocation frequency: {:.2}%\n", (heap_allocations as f64 / total_bars as f64) * 100.0);

    println!("Percentile Statistics:");
    println!("  P50 (median): {} trades", p50_count);
    println!("  P90: {} trades", p90_count);
    println!("  P99: {} trades", p99_count);
    println!("  P100 (max): {} trades\n", max_trades_per_bar);

    // Memory analysis
    const AGGTRADE_SIZE: usize = 64; // Approximate with alignment
    let current_buffer_bytes = 512 * AGGTRADE_SIZE;
    println!("Memory Analysis:");
    println!("  Current buffer (512 slots): {} KB", current_buffer_bytes / 1024);
    println!("  Per-slot cost: {} bytes\n", AGGTRADE_SIZE);

    // Recommendations
    println!("Recommendations:");
    if heap_allocations == 0 {
        println!("  ✓ 512 slots avoids ALL heap allocations (optimal)");
    } else if (heap_allocations as f64 / total_bars as f64) < 0.01 {
        println!("  ✓ 512 slots avoids heap allocations for 99%+ of bars (very good)");
    } else {
        println!("  ⚠ Consider increasing buffer size (heap allocations: {:.2}%)",
                 (heap_allocations as f64 / total_bars as f64) * 100.0);
    }

    if p99_count < 256 {
        println!("  → Consider: SmallVec<[AggTrade; 256]> saves ~16KB per bar");
        println!("    Pros: Less stack memory");
        println!("    Cons: {} bars would heap-allocate", heap_allocations);
    } else if p99_count < 512 {
        println!("  → Current 512 is good; 99% of bars fit within 512 slots");
    } else {
        println!("  → Current 512 may need increase; P99 exceeds 512");
    }

    println!("\nDetailed Distribution (sample):");
    for (count, freq) in sorted_counts.iter().skip(sorted_counts.len().saturating_sub(10)) {
        let pct = (**freq as f64 / total_count as f64) * 100.0;
        println!("  {} trades: {} bars ({:.2}%)", count, freq, pct);
    }
}

#[cfg(not(feature = "test-utils"))]
fn main() {
    eprintln!("This benchmark requires the 'test-utils' feature");
    eprintln!("Run: cargo bench --bench smallvec_profiling --features test-utils");
}
