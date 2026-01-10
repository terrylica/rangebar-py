//! Exness EURUSD Raw_Spread Statistical Deep Dive
//!
//! Multi-threshold analysis with comprehensive statistics:
//! - 0.2bps, 0.5bps, 1bps thresholds
//! - Bar count distributions
//! - Tick-per-bar distributions
//! - Price movement analysis
//! - Spread dynamics
//! - Bar duration statistics

#![cfg(feature = "providers")]

use rangebar::providers::exness::{
    ExnessFetcher, ExnessRangeBarBuilder, ExnessTick, ValidationStrictness,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[tokio::test]
#[ignore] // Only run with --ignored (fetches real data)
async fn test_exness_statistical_deep_dive() {
    println!("\n=== Exness EURUSD Raw_Spread Statistical Deep Dive ===\n");

    // Fetch January 2024 data
    println!("Fetching January 2024 EURUSD_Raw_Spread data...");
    let fetcher = ExnessFetcher::new("EURUSD_Raw_Spread");
    let all_ticks = fetcher.fetch_month(2024, 1).await.expect("Failed to fetch");

    // Filter to Jan 15-19, 2024
    let jan_15_start_ms = 1705276800000;
    let jan_20_start_ms = 1705708800000;

    let test_ticks: Vec<ExnessTick> = all_ticks
        .into_iter()
        .filter(|tick| tick.timestamp_ms >= jan_15_start_ms && tick.timestamp_ms < jan_20_start_ms)
        .collect();

    println!(
        "✅ Fetched {} ticks for Jan 15-19, 2024\n",
        test_ticks.len()
    );

    // Test thresholds (v3.0.0 units: 1 unit = 0.1bps)
    let thresholds = vec![
        (2, "0.2bps"),  // 0.2 basis points
        (5, "0.5bps"),  // 0.5 basis points
        (10, "1.0bps"), // 1.0 basis point
    ];

    let mut all_results = Vec::new();

    for (threshold_units, threshold_label) in thresholds {
        println!(
            "=== Testing Threshold: {} (units={}) ===\n",
            threshold_label, threshold_units
        );

        let mut builder = ExnessRangeBarBuilder::new(
            threshold_units,
            "EURUSD_Raw_Spread",
            ValidationStrictness::Strict,
        )
        .expect("Failed to create ExnessRangeBarBuilder");

        let mut bars = Vec::new();
        for tick in &test_ticks {
            if let Ok(Some(bar)) = builder.process_tick(tick) {
                bars.push(bar);
            }
        }

        // Get incomplete bar
        if let Some(incomplete) = builder.get_incomplete_bar() {
            bars.push(incomplete);
        }

        println!("Generated {} bars", bars.len());

        // Compute statistics
        let stats = compute_statistics(&test_ticks, &bars, threshold_units, threshold_label);
        print_statistics(&stats);
        all_results.push(stats);

        println!("\n");
    }

    // Export results
    export_statistical_analysis(&all_results, &test_ticks);
    println!("\n✅ Statistical analysis exported to output/exness_stats/");
}

#[derive(Debug, Clone)]
struct ThresholdStatistics {
    #[allow(dead_code)]
    threshold_units: u32,
    threshold_label: String,
    threshold_decimal_bps: f64,
    #[allow(dead_code)]
    total_ticks: usize,
    total_bars: usize,
    bars_per_day: f64,
    ticks_per_bar_mean: f64,
    ticks_per_bar_median: f64,
    ticks_per_bar_p95: usize,
    ticks_per_bar_max: usize,
    bar_duration_mean_ms: f64,
    bar_duration_median_ms: f64,
    bar_duration_p95_ms: i64,
    price_movement_mean_pips: f64,
    price_movement_median_pips: f64,
    spread_mean_pips: f64,
    spread_median_pips: f64,
    spread_max_pips: f64,
    zero_spread_bars_pct: f64,
    tick_distribution: HashMap<usize, usize>, // ticks_per_bar -> count
}

fn compute_statistics(
    ticks: &[ExnessTick],
    bars: &[rangebar::providers::exness::ExnessRangeBar],
    threshold_units: u32,
    threshold_label: &str,
) -> ThresholdStatistics {
    let threshold_decimal_bps = threshold_units as f64 * 0.1;

    // Ticks per bar
    let mut ticks_per_bar: Vec<usize> = bars
        .iter()
        .map(|b| b.spread_stats.tick_count as usize)
        .collect();
    ticks_per_bar.sort_unstable();

    let ticks_per_bar_mean = ticks_per_bar.iter().sum::<usize>() as f64 / bars.len() as f64;
    let ticks_per_bar_median = ticks_per_bar[ticks_per_bar.len() / 2];
    let ticks_per_bar_p95 = ticks_per_bar[(ticks_per_bar.len() as f64 * 0.95) as usize];
    let ticks_per_bar_max = *ticks_per_bar.last().unwrap_or(&0);

    // Bar durations (time between open and close)
    let mut bar_durations_ms: Vec<i64> = bars
        .iter()
        .map(|b| b.base.close_time - b.base.open_time)
        .collect();
    bar_durations_ms.sort_unstable();

    let bar_duration_mean_ms = bar_durations_ms.iter().sum::<i64>() as f64 / bars.len() as f64;
    let bar_duration_median_ms = bar_durations_ms[bar_durations_ms.len() / 2];
    let bar_duration_p95_ms = bar_durations_ms[(bar_durations_ms.len() as f64 * 0.95) as usize];

    // Price movements (high - low in pips)
    let mut price_movements_pips: Vec<f64> = bars
        .iter()
        .map(|b| (b.base.high.to_f64() - b.base.low.to_f64()) * 10000.0)
        .collect();
    price_movements_pips.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let price_movement_mean_pips = price_movements_pips.iter().sum::<f64>() / bars.len() as f64;
    let price_movement_median_pips = price_movements_pips[price_movements_pips.len() / 2];

    // Spread statistics
    let spreads_pips: Vec<f64> = bars
        .iter()
        .map(|b| b.spread_stats.avg_spread().to_f64() * 10000.0)
        .collect();

    let spread_mean_pips = spreads_pips.iter().sum::<f64>() / bars.len() as f64;
    let mut spreads_sorted = spreads_pips.clone();
    spreads_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spread_median_pips = spreads_sorted[spreads_sorted.len() / 2];
    let spread_max_pips = *spreads_sorted.last().unwrap_or(&0.0);

    let zero_spread_bars = bars
        .iter()
        .filter(|b| b.spread_stats.avg_spread().0 == 0)
        .count();
    let zero_spread_bars_pct = (zero_spread_bars as f64 / bars.len() as f64) * 100.0;

    // Tick distribution
    let mut tick_distribution = HashMap::new();
    for bar in bars {
        *tick_distribution
            .entry(bar.spread_stats.tick_count as usize)
            .or_insert(0) += 1;
    }

    ThresholdStatistics {
        threshold_units,
        threshold_label: threshold_label.to_string(),
        threshold_decimal_bps,
        total_ticks: ticks.len(),
        total_bars: bars.len(),
        bars_per_day: bars.len() as f64 / 5.0,
        ticks_per_bar_mean,
        ticks_per_bar_median: ticks_per_bar_median as f64,
        ticks_per_bar_p95,
        ticks_per_bar_max,
        bar_duration_mean_ms,
        bar_duration_median_ms: bar_duration_median_ms as f64,
        bar_duration_p95_ms,
        price_movement_mean_pips,
        price_movement_median_pips,
        spread_mean_pips,
        spread_median_pips,
        spread_max_pips,
        zero_spread_bars_pct,
        tick_distribution,
    }
}

fn print_statistics(stats: &ThresholdStatistics) {
    println!(
        "Threshold: {} ({} bps)",
        stats.threshold_label, stats.threshold_decimal_bps
    );
    println!("────────────────────────────────────────");
    println!("Total bars:           {}", stats.total_bars);
    println!("Bars per day:         {:.0}", stats.bars_per_day);
    println!();
    println!("Ticks per bar:");
    println!("  Mean:               {:.2}", stats.ticks_per_bar_mean);
    println!("  Median:             {}", stats.ticks_per_bar_median);
    println!("  P95:                {}", stats.ticks_per_bar_p95);
    println!("  Max:                {}", stats.ticks_per_bar_max);
    println!();
    println!("Bar duration (ms):");
    println!(
        "  Mean:               {:.0} ms ({:.1}s)",
        stats.bar_duration_mean_ms,
        stats.bar_duration_mean_ms / 1000.0
    );
    println!(
        "  Median:             {} ms ({:.1}s)",
        stats.bar_duration_median_ms,
        stats.bar_duration_median_ms / 1000.0
    );
    println!(
        "  P95:                {} ms ({:.1}s)",
        stats.bar_duration_p95_ms,
        stats.bar_duration_p95_ms as f64 / 1000.0
    );
    println!();
    println!("Price movement (pips):");
    println!(
        "  Mean:               {:.4}",
        stats.price_movement_mean_pips
    );
    println!(
        "  Median:             {:.4}",
        stats.price_movement_median_pips
    );
    println!();
    println!("Spread dynamics:");
    println!("  Mean:               {:.4} pips", stats.spread_mean_pips);
    println!("  Median:             {:.4} pips", stats.spread_median_pips);
    println!("  Max:                {:.4} pips", stats.spread_max_pips);
    println!("  Zero spread bars:   {:.2}%", stats.zero_spread_bars_pct);
    println!();
    println!("Tick distribution (top 10):");
    let mut dist_vec: Vec<_> = stats.tick_distribution.iter().collect();
    dist_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (ticks, count) in dist_vec.iter().take(10) {
        let pct = (**count as f64 / stats.total_bars as f64) * 100.0;
        println!("  {} ticks: {} bars ({:.2}%)", ticks, count, pct);
    }
}

fn export_statistical_analysis(results: &[ThresholdStatistics], ticks: &[ExnessTick]) {
    let output_dir = Path::new("output/exness_stats");
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Export summary comparison table
    let mut summary_csv = String::from(
        "threshold_decimal_bps,threshold_label,total_bars,bars_per_day,ticks_per_bar_mean,ticks_per_bar_median,bar_duration_mean_ms,price_movement_mean_pips,spread_mean_pips,zero_spread_bars_pct\n",
    );

    for stats in results {
        summary_csv.push_str(&format!(
            "{},{},{},{:.0},{:.2},{},{:.0},{:.4},{:.4},{:.2}\n",
            stats.threshold_decimal_bps,
            stats.threshold_label,
            stats.total_bars,
            stats.bars_per_day,
            stats.ticks_per_bar_mean,
            stats.ticks_per_bar_median,
            stats.bar_duration_mean_ms,
            stats.price_movement_mean_pips,
            stats.spread_mean_pips,
            stats.zero_spread_bars_pct,
        ));
    }

    fs::write(output_dir.join("threshold_comparison.csv"), summary_csv)
        .expect("Failed to write threshold_comparison.csv");

    // Export detailed statistics as JSON
    let detailed_json = serde_json::json!({
        "test_period": "2024-01-15 to 2024-01-19",
        "total_ticks": ticks.len(),
        "thresholds": results.iter().map(|s| {
            serde_json::json!({
                "threshold_decimal_bps": s.threshold_decimal_bps,
                "threshold_label": s.threshold_label,
                "total_bars": s.total_bars,
                "bars_per_day": s.bars_per_day,
                "ticks_per_bar": {
                    "mean": s.ticks_per_bar_mean,
                    "median": s.ticks_per_bar_median,
                    "p95": s.ticks_per_bar_p95,
                    "max": s.ticks_per_bar_max,
                },
                "bar_duration_ms": {
                    "mean": s.bar_duration_mean_ms,
                    "median": s.bar_duration_median_ms,
                    "p95": s.bar_duration_p95_ms,
                },
                "price_movement_pips": {
                    "mean": s.price_movement_mean_pips,
                    "median": s.price_movement_median_pips,
                },
                "spread_pips": {
                    "mean": s.spread_mean_pips,
                    "median": s.spread_median_pips,
                    "max": s.spread_max_pips,
                },
                "zero_spread_bars_pct": s.zero_spread_bars_pct,
            })
        }).collect::<Vec<_>>(),
    });

    fs::write(
        output_dir.join("detailed_statistics.json"),
        serde_json::to_string_pretty(&detailed_json).unwrap(),
    )
    .expect("Failed to write detailed_statistics.json");

    println!("\nExported files:");
    println!("  - threshold_comparison.csv");
    println!("  - detailed_statistics.json");
}
