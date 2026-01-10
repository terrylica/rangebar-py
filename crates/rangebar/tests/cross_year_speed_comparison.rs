//! Cross-year speed comparison test: Oct 2024 - Feb 2025
//!
//! Comprehensive performance benchmarking of batch vs streaming rangebar construction
//! across year boundary with detailed memory usage tracking and throughput analysis.
//!
//! Now includes Production Streaming V2 with bounded memory architecture.

#![cfg(feature = "streaming")]

use rangebar::range_bars::ExportRangeBarProcessor;
use rangebar::types::AggTrade;
use rangebar::{StreamingProcessor, StreamingProcessorConfig};
use rangebar_core::test_utils::generators::create_test_trade;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Performance metrics for cross-year comparison
#[derive(Debug, Clone)]
struct CrossYearPerformanceMetrics {
    month: String,
    trade_count: usize,
    bar_count: usize,

    // Batch processing metrics
    batch_duration_ms: u64,
    batch_memory_peak_kb: u64,
    batch_throughput_aggtrades_per_sec: f64,

    // Production Streaming V2 metrics (bounded memory)
    streaming_v2_duration_ms: u64,
    streaming_v2_memory_peak_kb: u64,
    streaming_v2_throughput_aggtrades_per_sec: f64,

    // V2 comparison metrics
    v2_speed_ratio: f64,       // batch_throughput / streaming_v2_throughput
    v2_memory_efficiency: f64, // (streaming_v2_memory - batch_memory) / batch_memory * 100
    v2_duration_ratio: f64,    // streaming_v2_duration / batch_duration
}

/// Cross-year test covering 2024 Oct to 2025 Feb with realistic trade volumes
#[test]
fn test_cross_year_speed_comparison_oct2024_feb2025() {
    println!("🚀 Cross-Year Speed Comparison: Oct 2024 - Feb 2025");
    println!("📊 Testing batch vs streaming rangebar construction with memory tracking\n");

    let threshold_decimal_bps = 25; // 0.25% standard threshold

    // Realistic monthly trade volumes (crypto market activity patterns)
    let monthly_tests = vec![
        ("2024-10", 1727740800000, 2_500_000), // Oct 2024: 2.5M trades
        ("2024-11", 1730419200000, 2_800_000), // Nov 2024: 2.8M trades
        ("2024-12", 1733011200000, 3_200_000), // Dec 2024: 3.2M trades (holiday volatility)
        ("2025-01", 1735689600000, 3_000_000), // Jan 2025: 3.0M trades (new year)
        ("2025-02", 1738368000000, 2_600_000), // Feb 2025: 2.6M trades
    ];

    let mut all_metrics = Vec::new();
    let mut total_trades = 0;
    let mut total_batch_duration = 0;

    for (month, base_timestamp, trade_count) in monthly_tests {
        println!("📅 Testing {}: {} trades", month, trade_count);

        // Generate realistic monthly trade data
        let trades = generate_monthly_trade_data(trade_count, base_timestamp, month);
        total_trades += trade_count;

        // Batch processing benchmark
        let batch_metrics = benchmark_batch_processing(&trades, threshold_decimal_bps);
        total_batch_duration += batch_metrics.duration_ms;

        // Production Streaming V2 benchmark (bounded memory)
        let streaming_v2_metrics =
            benchmark_streaming_v2_processing(&trades, threshold_decimal_bps);

        println!(
            "  📊 {}: V2 = {:.0} t/s, {:.1}MB, {}ms",
            month,
            streaming_v2_metrics.throughput_aggtrades_per_sec,
            streaming_v2_metrics.memory_peak_kb as f64 / 1024.0,
            streaming_v2_metrics.duration_ms
        );

        // Ensure consistency between batch and V2
        assert_eq!(
            batch_metrics.bar_count, streaming_v2_metrics.bar_count,
            "Bar counts must match for {}: batch={}, v2={}",
            month, batch_metrics.bar_count, streaming_v2_metrics.bar_count
        );

        println!(
            "  📊 Bar counts: batch={}, v2={}",
            batch_metrics.bar_count, streaming_v2_metrics.bar_count
        );

        // V2 comparison metrics
        let v2_speed_ratio = batch_metrics.throughput_aggtrades_per_sec
            / streaming_v2_metrics.throughput_aggtrades_per_sec;
        let v2_memory_efficiency = if batch_metrics.memory_peak_kb > 0 {
            (streaming_v2_metrics.memory_peak_kb as f64 - batch_metrics.memory_peak_kb as f64)
                / batch_metrics.memory_peak_kb as f64
                * 100.0
        } else {
            0.0
        };
        let v2_duration_ratio =
            streaming_v2_metrics.duration_ms as f64 / batch_metrics.duration_ms as f64;

        let month_metrics = CrossYearPerformanceMetrics {
            month: month.to_string(),
            trade_count,
            bar_count: batch_metrics.bar_count,
            batch_duration_ms: batch_metrics.duration_ms,
            batch_memory_peak_kb: batch_metrics.memory_peak_kb,
            batch_throughput_aggtrades_per_sec: batch_metrics.throughput_aggtrades_per_sec,
            streaming_v2_duration_ms: streaming_v2_metrics.duration_ms,
            streaming_v2_memory_peak_kb: streaming_v2_metrics.memory_peak_kb,
            streaming_v2_throughput_aggtrades_per_sec: streaming_v2_metrics
                .throughput_aggtrades_per_sec,
            v2_speed_ratio,
            v2_memory_efficiency,
            v2_duration_ratio,
        };

        print_monthly_results(&month_metrics);
        all_metrics.push(month_metrics);
        println!();
    }

    // Print comprehensive summary
    print_cross_year_summary(&all_metrics, total_trades, total_batch_duration);

    // Validate performance criteria
    validate_performance_criteria(&all_metrics);
}

/// Test year boundary edge cases with speed comparison
#[test]
fn test_year_boundary_speed_edge_cases() {
    println!("🎯 Year Boundary Speed Edge Cases (2024→2025)");

    let threshold_decimal_bps = 25;
    let edge_cases = vec![
        ("new_years_eve", 1735603200000, 500_000),   // Dec 31, 2024
        ("new_years_day", 1735689600000, 500_000),   // Jan 1, 2025
        ("year_transition", 1735646400000, 200_000), // 4-hour window crossing midnight
    ];

    for (case_name, base_timestamp, trade_count) in edge_cases {
        println!("  🔍 Testing {}: {} trades", case_name, trade_count);

        let trades = generate_year_boundary_data(trade_count, base_timestamp, case_name);

        let batch_metrics = benchmark_batch_processing(&trades, threshold_decimal_bps);
        let v2_metrics = benchmark_streaming_v2_processing(&trades, threshold_decimal_bps);

        assert_eq!(
            batch_metrics.bar_count, v2_metrics.bar_count,
            "Year boundary consistency failed for {}",
            case_name
        );

        let speed_ratio =
            batch_metrics.throughput_aggtrades_per_sec / v2_metrics.throughput_aggtrades_per_sec;
        let memory_efficiency = (v2_metrics.memory_peak_kb as f64
            - batch_metrics.memory_peak_kb as f64)
            / batch_metrics.memory_peak_kb as f64
            * 100.0;

        println!(
            "    Batch: {:.0} t/s, {:.1}MB | V2: {:.0} t/s, {:.1}MB",
            batch_metrics.throughput_aggtrades_per_sec,
            batch_metrics.memory_peak_kb as f64 / 1024.0,
            v2_metrics.throughput_aggtrades_per_sec,
            v2_metrics.memory_peak_kb as f64 / 1024.0
        );

        println!(
            "    Speed ratio: {:.2}x, Memory efficiency: {:.1}%",
            speed_ratio, memory_efficiency
        );
    }

    println!("  ✅ Year boundary edge cases completed");
}

// Benchmark helper functions

#[derive(Debug)]
struct ProcessingMetrics {
    duration_ms: u64,
    memory_peak_kb: u64,
    bar_count: usize,
    throughput_aggtrades_per_sec: f64,
}

fn benchmark_batch_processing(
    trades: &[AggTrade],
    threshold_decimal_bps: u32,
) -> ProcessingMetrics {
    let initial_memory = get_memory_usage_kb();

    let start_time = Instant::now();
    let mut processor = ExportRangeBarProcessor::new(threshold_decimal_bps)
        .expect("Failed to create processor with valid threshold");
    processor.process_trades_continuously(trades);
    let mut bars = processor.get_all_completed_bars();

    if let Some(incomplete) = processor.get_incomplete_bar() {
        bars.push(incomplete);
    }

    let duration = start_time.elapsed();
    let final_memory = get_memory_usage_kb();
    let memory_used = final_memory.saturating_sub(initial_memory);

    let throughput = trades.len() as f64 / duration.as_secs_f64();

    ProcessingMetrics {
        duration_ms: duration.as_millis() as u64,
        memory_peak_kb: memory_used,
        bar_count: bars.len(),
        throughput_aggtrades_per_sec: throughput,
    }
}

/// Benchmark Production Streaming V2 (bounded memory architecture)
fn benchmark_streaming_v2_processing(
    trades: &[AggTrade],
    threshold_decimal_bps: u32,
) -> ProcessingMetrics {
    let rt = Runtime::new().unwrap();
    let initial_memory = get_memory_usage_kb();

    let start_time = Instant::now();

    let bar_count = rt.block_on(async {
        let config = StreamingProcessorConfig {
            trade_channel_capacity: 10000,
            bar_channel_capacity: 1000,
            memory_threshold_bytes: 100_000_000, // 100MB
            ..Default::default()
        };

        let mut processor = StreamingProcessor::with_config(threshold_decimal_bps, config)
            .expect("Failed to create processor");

        // Get channels for streaming
        let trade_sender = processor.trade_sender().expect("Should have trade sender");
        let mut bar_receiver = processor.bar_receiver().expect("Should have bar receiver");

        // Start processing task
        let process_task = tokio::spawn(async move { processor.start_processing().await });

        // Send trades task
        let trades_to_send = trades.to_vec();
        let send_task = tokio::spawn(async move {
            for trade in trades_to_send {
                if trade_sender.send(trade).await.is_err() {
                    break;
                }
            }
            drop(trade_sender); // Close sender
        });

        // Receive bars task (count only, maintaining bounded memory)
        let mut bar_count = 0;
        while let Some(_bar) = bar_receiver.recv().await {
            bar_count += 1;
            // Note: bars are immediately discarded to maintain bounded memory
        }

        // Wait for tasks to complete
        let _process_result = tokio::try_join!(send_task, process_task);

        // Final incomplete bar is now automatically sent through bar channel

        bar_count
    });

    let duration = start_time.elapsed();
    let final_memory = get_memory_usage_kb();
    let memory_used = final_memory.saturating_sub(initial_memory);

    let throughput = trades.len() as f64 / duration.as_secs_f64();

    ProcessingMetrics {
        duration_ms: duration.as_millis() as u64,
        memory_peak_kb: memory_used,
        bar_count,
        throughput_aggtrades_per_sec: throughput,
    }
}

// Data generation functions

fn generate_monthly_trade_data(
    trade_count: usize,
    base_timestamp: u64,
    month: &str,
) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(trade_count);
    let base_price = 23000.0;

    for i in 0..trade_count {
        let progress = i as f64 / trade_count as f64;

        // Month-specific patterns
        let monthly_trend = match month {
            "2024-10" => (progress * std::f64::consts::PI).sin() * 800.0, // October volatility
            "2024-11" => progress * 1200.0 - 600.0,                       // November uptrend
            "2024-12" => (progress * 2.0 * std::f64::consts::PI).sin() * 1500.0, // December holiday volatility
            "2025-01" => (progress * 3.0 * std::f64::consts::PI).cos() * 1000.0, // January new year patterns
            "2025-02" => (1.0 - progress) * 500.0, // February stabilization
            _ => 0.0,
        };

        // Daily and weekly cycles
        let daily_cycle = ((i as f64 / 1440.0) * 2.0 * std::f64::consts::PI).sin() * 300.0; // Daily pattern
        let weekly_cycle = ((i as f64 / 10080.0) * 2.0 * std::f64::consts::PI).sin() * 600.0; // Weekly pattern

        // Market microstructure noise
        let volatility = (i as f64 * 0.01).sin() * 150.0;
        let noise = (i as f64 * 0.1).sin() * 30.0;

        let price = base_price + monthly_trend + daily_cycle + weekly_cycle + volatility + noise;
        let timestamp = base_timestamp + (i as u64 * 60_000_000); // 1-minute intervals in microseconds

        trades.push(create_test_trade(i as u64 + 1_000_000, price, timestamp));
    }

    trades
}

fn generate_year_boundary_data(
    trade_count: usize,
    base_timestamp: u64,
    case_name: &str,
) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(trade_count);
    let base_price = 23000.0;

    for i in 0..trade_count {
        let time_progress = i as f64 / trade_count as f64;

        // Year boundary specific patterns
        let boundary_effect = match case_name {
            "new_years_eve" => (time_progress * 8.0 * std::f64::consts::PI).sin() * 500.0, // High frequency
            "new_years_day" => (time_progress * std::f64::consts::PI).sin() * 200.0,       // Calmer
            "year_transition" => {
                if time_progress < 0.5 {
                    -200.0
                } else {
                    200.0
                }
            } // Step function
            _ => 0.0,
        };

        let noise = (i as f64 * 0.05).sin() * 50.0;
        let price = base_price + boundary_effect + noise;
        let timestamp = base_timestamp + (i as u64 * 1_000_000); // 1-second intervals for boundary cases in microseconds

        trades.push(create_test_trade(i as u64 + 2_000_000, price, timestamp));
    }

    trades
}

// Memory tracking function

fn get_memory_usage_kb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            && let Ok(rss_str) = String::from_utf8(output.stdout)
            && let Ok(rss_kb) = rss_str.trim().parse::<u64>()
        {
            return rss_kb;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(rss_kb) = kb_str.parse::<u64>() {
                            return rss_kb;
                        }
                    }
                    break;
                }
            }
        }
    }

    0 // Fallback for unsupported platforms
}

// Output formatting functions

fn print_monthly_results(metrics: &CrossYearPerformanceMetrics) {
    println!("  📊 {} Results:", metrics.month);
    println!("    Trade Count: {}", metrics.trade_count);
    println!("    Bar Count: {}", metrics.bar_count);
    println!(
        "    🏃 Batch: {:.0} t/s, {}ms, {:.1}MB peak",
        metrics.batch_throughput_aggtrades_per_sec,
        metrics.batch_duration_ms,
        metrics.batch_memory_peak_kb as f64 / 1024.0
    );
    println!(
        "    ⚡ Streaming: {:.0} t/s, {}ms, {:.1}MB peak",
        metrics.streaming_v2_throughput_aggtrades_per_sec,
        metrics.streaming_v2_duration_ms,
        metrics.streaming_v2_memory_peak_kb as f64 / 1024.0
    );
    println!(
        "    🔒 Streaming V2 (Bounded): {:.0} t/s, {}ms, {:.1}MB peak",
        metrics.streaming_v2_throughput_aggtrades_per_sec,
        metrics.streaming_v2_duration_ms,
        metrics.streaming_v2_memory_peak_kb as f64 / 1024.0
    );

    println!("    📈 Speed Ratios:");
    println!("      Legacy vs Streaming: {:.2}x", metrics.v2_speed_ratio);
    println!("      Legacy vs V2 Bounded: {:.2}x", metrics.v2_speed_ratio);

    println!("    💾 Memory Efficiency:");
    println!(
        "      Streaming vs Batch: {:.1}% {} batch",
        metrics.v2_memory_efficiency.abs(),
        if metrics.v2_memory_efficiency < 0.0 {
            "better than"
        } else {
            "worse than"
        }
    );
    println!(
        "      V2 Bounded vs Batch: {:.1}% {} batch",
        metrics.v2_memory_efficiency.abs(),
        if metrics.v2_memory_efficiency < 0.0 {
            "better than"
        } else {
            "worse than"
        }
    );
}

fn print_cross_year_summary(
    all_metrics: &[CrossYearPerformanceMetrics],
    total_trades: usize,
    total_batch_duration: u64,
) {
    println!("🎯 Cross-Year Performance Summary (Oct 2024 - Feb 2025)");
    println!("{}", "=".repeat(60));

    let avg_v2_speed_ratio: f64 =
        all_metrics.iter().map(|m| m.v2_speed_ratio).sum::<f64>() / all_metrics.len() as f64;
    let avg_v2_memory_efficiency: f64 = all_metrics
        .iter()
        .map(|m| m.v2_memory_efficiency)
        .sum::<f64>()
        / all_metrics.len() as f64;
    let avg_v2_duration_ratio: f64 =
        all_metrics.iter().map(|m| m.v2_duration_ratio).sum::<f64>() / all_metrics.len() as f64;

    let total_batch_throughput = total_trades as f64 / (total_batch_duration as f64 / 1000.0);

    println!("📊 Overall Statistics:");
    println!("  Total Trades Processed: {}", total_trades);
    println!(
        "  Total Batch Duration: {:.2}s",
        total_batch_duration as f64 / 1000.0
    );
    println!();

    println!("🏃 Throughput Comparison:");
    println!("  Batch: {:.0} trades/sec", total_batch_throughput);
    println!();

    println!("📈 Average Performance Metrics (Batch vs V2):");
    println!("  Speed Ratio: {:.2}x", avg_v2_speed_ratio);
    println!("  Memory Efficiency: {:.1}%", avg_v2_memory_efficiency);
    println!("  Duration Ratio: {:.2}x", avg_v2_duration_ratio);
    println!();

    // Month-by-month breakdown table
    println!("📅 Month-by-Month Performance:");
    println!("  Month     | Trades    | Bars     | Batch t/s | V2 t/s     | Ratio | Mem Eff");
    println!("  ----------|-----------|----------|-----------|------------|-------|--------");
    for metrics in all_metrics {
        println!(
            "  {:8} | {:8} | {:7} | {:8.0} | {:9.0} | {:4.2}x | {:6.1}%",
            metrics.month,
            metrics.trade_count,
            metrics.bar_count,
            metrics.batch_throughput_aggtrades_per_sec,
            metrics.streaming_v2_throughput_aggtrades_per_sec,
            metrics.v2_speed_ratio,
            metrics.v2_memory_efficiency
        );
    }
}

fn validate_performance_criteria(all_metrics: &[CrossYearPerformanceMetrics]) {
    println!("🔍 Performance Validation:");

    let mut validation_passed = true;

    for metrics in all_metrics {
        // Validate that both implementations produce results
        if metrics.bar_count == 0 {
            println!("  ❌ {} failed: No bars generated", metrics.month);
            validation_passed = false;
        }

        // Validate reasonable throughput (>10k trades/sec)
        if metrics.batch_throughput_aggtrades_per_sec < 10_000.0 {
            println!(
                "  ⚠️  {} batch throughput low: {:.0} t/s",
                metrics.month, metrics.batch_throughput_aggtrades_per_sec
            );
        }

        if metrics.streaming_v2_throughput_aggtrades_per_sec < 10_000.0 {
            println!(
                "  ⚠️  {} streaming throughput low: {:.0} t/s",
                metrics.month, metrics.streaming_v2_throughput_aggtrades_per_sec
            );
        }

        // Memory usage validation (should not exceed 1GB)
        if metrics.batch_memory_peak_kb > 1_000_000 {
            println!(
                "  ⚠️  {} batch memory high: {:.1}MB",
                metrics.month,
                metrics.batch_memory_peak_kb as f64 / 1024.0
            );
        }

        if metrics.streaming_v2_memory_peak_kb > 1_000_000 {
            println!(
                "  ⚠️  {} streaming memory high: {:.1}MB",
                metrics.month,
                metrics.streaming_v2_memory_peak_kb as f64 / 1024.0
            );
        }
    }

    if validation_passed {
        println!("  ✅ All cross-year performance validations passed");
    } else {
        println!("  ❌ Some cross-year performance validations failed");
        panic!("Performance validation failed");
    }
}
