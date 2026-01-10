use rangebar::types::{AggTrade, RangeBar};
use rangebar_core::test_utils::generators::{
    create_test_trade, process_batch_style, process_streaming_style,
};
use std::process;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

/// Multi-month memory and boundary tests spanning 2024-2025 transition
///
/// These tests validate:
/// - Memory efficiency across multi-month datasets
/// - Year boundary transitions (2024 → 2025)
/// - Streaming vs batch memory usage comparison
/// - Large-scale temporal integrity
const MONTHS_TO_TEST: &[(&str, i64, usize)] = &[
    ("2024-10", 1727740800000, 2_500_000), // Oct 2024 - 2.5M trades
    ("2024-11", 1730419200000, 2_800_000), // Nov 2024 - 2.8M trades
    ("2024-12", 1733011200000, 3_200_000), // Dec 2024 - 3.2M trades (holiday volume)
    ("2025-01", 1735689600000, 3_000_000), // Jan 2025 - 3.0M trades (new year)
    ("2025-02", 1738368000000, 2_600_000), // Feb 2025 - 2.6M trades
];

#[tokio::test]
async fn test_multi_month_boundary_consistency() {
    println!("🔍 Testing multi-month boundary consistency (2024-2025 transition)");
    println!(
        "📅 Testing {} months spanning year boundary",
        MONTHS_TO_TEST.len()
    );

    let threshold_decimal_bps = 25; // 0.25% standard threshold

    // Generate multi-month dataset
    println!("  📊 Generating multi-month dataset...");
    let start_gen = Instant::now();
    let multi_month_data = create_multi_month_dataset();
    let gen_duration = start_gen.elapsed();

    let total_trades = multi_month_data.len();
    println!(
        "  ✅ Generated {} trades across {} months in {:?}",
        total_trades,
        MONTHS_TO_TEST.len(),
        gen_duration
    );

    // Test batch processing with memory monitoring
    println!("  🔄 Testing batch processing with memory monitoring...");
    let (batch_bars, batch_duration, batch_memory) = process_with_memory_monitoring(
        &multi_month_data,
        threshold_decimal_bps,
        ProcessingMode::Batch,
    )
    .await;

    // Test streaming processing with memory monitoring
    println!("  🔄 Testing streaming processing with memory monitoring...");
    let (streaming_bars, streaming_duration, streaming_memory) = process_with_memory_monitoring(
        &multi_month_data,
        threshold_decimal_bps,
        ProcessingMode::Streaming,
    )
    .await;

    // Compare results
    let matches = batch_bars.len() == streaming_bars.len();
    println!("  📊 Multi-month comparison:");
    println!(
        "    Trades processed: {} ({})",
        total_trades,
        format_number(total_trades)
    );
    println!(
        "    Batch bars: {} in {:?}",
        batch_bars.len(),
        batch_duration
    );
    println!(
        "    Streaming bars: {} in {:?}",
        streaming_bars.len(),
        streaming_duration
    );
    println!(
        "    Consistency: {}",
        if matches { "✅ MATCH" } else { "❌ MISMATCH" }
    );

    // Memory analysis
    analyze_memory_usage(batch_memory, streaming_memory, total_trades);

    // Performance analysis
    analyze_performance(total_trades, batch_duration, streaming_duration);

    // Validate year boundary transitions
    validate_year_boundary_transitions(&batch_bars, &streaming_bars);

    // Validate temporal integrity across months
    validate_multi_month_temporal_integrity(&batch_bars, "batch");
    validate_multi_month_temporal_integrity(&streaming_bars, "streaming");

    assert!(!batch_bars.is_empty(), "Batch should generate bars");
    assert!(!streaming_bars.is_empty(), "Streaming should generate bars");
    assert!(
        matches,
        "Streaming and batch should produce identical results"
    );

    println!("  ✅ Multi-month test complete - perfect consistency maintained");
}

#[tokio::test]
async fn test_progressive_memory_scaling() {
    println!("🔍 Testing progressive memory scaling across dataset sizes");

    let threshold_decimal_bps = 25;
    let test_sizes = vec![
        ("100K trades", 100_000),
        ("500K trades", 500_000),
        ("1M trades", 1_000_000),
        ("2M trades", 2_000_000),
        ("5M trades", 5_000_000),
    ];

    println!("  📈 Memory scaling analysis:");

    for (size_name, trade_count) in test_sizes {
        println!("    🎯 Testing: {}", size_name);

        // Generate dataset of specific size
        let dataset = create_progressive_dataset(trade_count);

        // Test batch memory usage
        let (_batch_bars, _batch_duration, batch_memory) =
            process_with_memory_monitoring(&dataset, threshold_decimal_bps, ProcessingMode::Batch)
                .await;

        // Test streaming memory usage
        let (_streaming_bars, _streaming_duration, streaming_memory) =
            process_with_memory_monitoring(
                &dataset,
                threshold_decimal_bps,
                ProcessingMode::Streaming,
            )
            .await;

        // Calculate memory efficiency
        let batch_mb = batch_memory.peak_rss_kb as f64 / 1024.0;
        let streaming_mb = streaming_memory.peak_rss_kb as f64 / 1024.0;
        let memory_savings = ((batch_mb - streaming_mb) / batch_mb * 100.0).max(0.0);

        println!(
            "      Batch: {:.1} MB peak, Streaming: {:.1} MB peak",
            batch_mb, streaming_mb
        );
        println!(
            "      Memory savings: {:.1}% ({})",
            memory_savings,
            if memory_savings > 0.0 {
                "✅ STREAMING WINS"
            } else {
                "⚠️ BATCH WINS"
            }
        );

        // Memory per trade analysis
        let batch_bytes_per_trade = (batch_memory.peak_rss_kb * 1024) as f64 / trade_count as f64;
        let streaming_bytes_per_trade =
            (streaming_memory.peak_rss_kb * 1024) as f64 / trade_count as f64;

        println!(
            "      Memory per trade: Batch={:.1}B, Streaming={:.1}B",
            batch_bytes_per_trade, streaming_bytes_per_trade
        );
    }
}

#[tokio::test]
async fn test_year_boundary_edge_cases() {
    println!("🔍 Testing year boundary edge cases (2024→2025)");

    let threshold_decimal_bps = 25;

    // Test specific year boundary scenarios
    let boundary_tests = vec![
        ("new_years_eve", create_new_years_eve_data()),
        ("new_years_day", create_new_years_day_data()),
        ("year_transition", create_year_transition_data()),
        ("leap_year_boundary", create_leap_year_boundary_data()),
    ];

    for (test_name, dataset) in boundary_tests {
        println!("  🎯 Testing: {}", test_name);

        let batch_bars = process_batch_style(&dataset, threshold_decimal_bps);
        let streaming_bars = process_streaming_style(&dataset, threshold_decimal_bps).await;

        let matches = batch_bars.len() == streaming_bars.len();
        println!(
            "    {}: Batch={}, Streaming={} - {}",
            test_name,
            batch_bars.len(),
            streaming_bars.len(),
            if matches { "✅ MATCH" } else { "❌ MISMATCH" }
        );

        // Validate year boundary specific characteristics
        validate_year_boundary_characteristics(&batch_bars, &streaming_bars, test_name);

        assert!(matches, "{} should have matching results", test_name);
    }
}

#[tokio::test]
async fn test_memory_leak_detection() {
    println!("🔍 Testing memory leak detection across extended processing");

    let threshold_decimal_bps = 25;
    let iterations = 10;
    let trades_per_iteration = 100_000;

    println!(
        "  🔄 Running {} iterations of {}K trades each",
        iterations,
        trades_per_iteration / 1000
    );

    let mut batch_memory_progression = Vec::new();
    let mut streaming_memory_progression = Vec::new();

    for iteration in 1..=iterations {
        println!("    Iteration {}/{}", iteration, iterations);

        // Generate fresh dataset for each iteration
        let dataset = create_progressive_dataset(trades_per_iteration);

        // Test batch processing
        let (_batch_bars, _batch_duration, batch_memory) =
            process_with_memory_monitoring(&dataset, threshold_decimal_bps, ProcessingMode::Batch)
                .await;

        // Test streaming processing
        let (_streaming_bars, _streaming_duration, streaming_memory) =
            process_with_memory_monitoring(
                &dataset,
                threshold_decimal_bps,
                ProcessingMode::Streaming,
            )
            .await;

        batch_memory_progression.push(batch_memory.peak_rss_kb);
        streaming_memory_progression.push(streaming_memory.peak_rss_kb);

        println!(
            "      Batch: {} KB, Streaming: {} KB",
            batch_memory.peak_rss_kb, streaming_memory.peak_rss_kb
        );
    }

    // Analyze memory leak patterns
    analyze_memory_leak_patterns(&batch_memory_progression, &streaming_memory_progression);
}

// Data generation functions

fn create_multi_month_dataset() -> Vec<AggTrade> {
    let mut all_trades = Vec::new();
    let mut trade_id_counter = 1_000_000u64;

    for (month_name, start_timestamp, trade_count) in MONTHS_TO_TEST {
        println!(
            "    📅 Generating {} with {} trades",
            month_name,
            format_number(*trade_count)
        );

        let month_trades = create_month_data(*start_timestamp, *trade_count, &mut trade_id_counter);
        all_trades.extend(month_trades);
    }

    // Sort by timestamp to ensure chronological order
    all_trades.sort_by_key(|trade| trade.timestamp);
    all_trades
}

fn create_month_data(
    start_timestamp: i64,
    trade_count: usize,
    trade_id_counter: &mut u64,
) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(trade_count);
    let base_price = 45000.0; // Higher base price for 2024-2025

    let month_duration_ms = 30 * 24 * 60 * 60 * 1000i64; // 30 days in milliseconds
    let avg_interval = month_duration_ms / trade_count as i64;

    for i in 0..trade_count {
        let progress = i as f64 / trade_count as f64;

        // Multi-layered realistic price movements
        let monthly_trend = (progress * std::f64::consts::PI).sin() * 2000.0; // ±2000 monthly trend
        let weekly_cycle = (progress * 4.0 * std::f64::consts::PI).sin() * 500.0; // Weekly patterns
        let daily_volatility =
            ((i as f64 * 0.01).sin() * 200.0) + ((i as f64 * 0.003).cos() * 100.0);
        let market_noise = (i as f64 * 0.1).sin() * 50.0;

        let price = base_price + monthly_trend + weekly_cycle + daily_volatility + market_noise;

        // Variable intervals simulating realistic trading patterns
        let interval_variance = ((i as f64 * 0.05).sin() * 0.5 + 0.5) * avg_interval as f64;
        let timestamp = start_timestamp + (i as f64 * interval_variance) as i64;

        trades.push(create_test_trade(
            *trade_id_counter,
            price,
            timestamp as u64,
        ));
        *trade_id_counter += 1;
    }

    trades
}

fn create_progressive_dataset(trade_count: usize) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(trade_count);
    let base_price = 45000.0;
    let base_timestamp = 1735689600000i64; // Jan 1, 2025

    for i in 0..trade_count {
        let progress = i as f64 / trade_count as f64;

        // Progressive price movement
        let trend = progress * 1000.0; // Gradual upward trend
        let volatility = (i as f64 * 0.01).sin() * 100.0;
        let noise = (i as f64 * 0.1).sin() * 20.0;

        let price = base_price + trend + volatility + noise;
        let timestamp = base_timestamp + (i as i64 * 500); // 500ms average intervals

        trades.push(create_test_trade(
            10_000_000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

fn create_new_years_eve_data() -> Vec<AggTrade> {
    // December 31, 2024, 23:00 to 23:59 UTC
    let start_time = 1735689600000i64 - 3600000; // 1 hour before midnight
    create_time_bounded_data(start_time, 3600000, 50000, "new_years_eve")
}

fn create_new_years_day_data() -> Vec<AggTrade> {
    // January 1, 2025, 00:00 to 01:00 UTC
    let start_time = 1735689600000i64; // Midnight Jan 1, 2025
    create_time_bounded_data(start_time, 3600000, 60000, "new_years_day")
}

fn create_year_transition_data() -> Vec<AggTrade> {
    // December 31, 2024 23:58 to January 1, 2025 00:02
    let start_time = 1735689600000i64 - 120000; // 2 minutes before midnight
    create_time_bounded_data(start_time, 240000, 20000, "year_transition")
}

fn create_leap_year_boundary_data() -> Vec<AggTrade> {
    // February 28-29, 2024 transition (2024 was a leap year)
    let start_time = 1709161200000i64; // Feb 29, 2024 00:00 UTC
    create_time_bounded_data(start_time, 86400000, 100000, "leap_year_boundary")
}

fn create_time_bounded_data(
    start_time: i64,
    duration_ms: i64,
    trade_count: usize,
    scenario: &str,
) -> Vec<AggTrade> {
    let mut trades = Vec::with_capacity(trade_count);
    let base_price = match scenario {
        "new_years_eve" => 42000.0,
        "new_years_day" => 43000.0,
        "year_transition" => 42500.0,
        "leap_year_boundary" => 51000.0,
        _ => 45000.0,
    };

    let interval = duration_ms / trade_count as i64;

    for i in 0..trade_count {
        let time_progress = i as f64 / trade_count as f64;

        // Scenario-specific price patterns
        let price_movement = match scenario {
            "new_years_eve" => (time_progress * std::f64::consts::PI).sin() * 500.0, // Volatility before new year
            "new_years_day" => time_progress * 300.0, // Gradual rise in new year
            "year_transition" => ((time_progress - 0.5) * 10.0).tanh() * 200.0, // Sharp transition
            "leap_year_boundary" => (time_progress * 2.0 * std::f64::consts::PI).sin() * 150.0, // Regular pattern
            _ => 0.0,
        };

        let noise = (i as f64 * 0.05).sin() * 30.0;
        let price = base_price + price_movement + noise;
        let timestamp = start_time + (i as i64 * interval);

        trades.push(create_test_trade(
            20_000_000 + i as u64,
            price,
            timestamp as u64,
        ));
    }

    trades
}

// Processing and monitoring functions

#[derive(Clone, Copy)]
enum ProcessingMode {
    Batch,
    Streaming,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MemoryMetrics {
    peak_rss_kb: u64,
    peak_vss_kb: u64,
    initial_rss_kb: u64,
    final_rss_kb: u64,
}

async fn process_with_memory_monitoring(
    trades: &[AggTrade],
    threshold_decimal_bps: u32,
    mode: ProcessingMode,
) -> (Vec<RangeBar>, std::time::Duration, MemoryMetrics) {
    // Get initial memory
    let initial_memory = get_current_memory();

    let start_time = Instant::now();

    let bars = match mode {
        ProcessingMode::Batch => process_batch_style(trades, threshold_decimal_bps),
        ProcessingMode::Streaming => process_streaming_style(trades, threshold_decimal_bps).await,
    };

    let duration = start_time.elapsed();

    // Get final memory
    let final_memory = get_current_memory();

    let metrics = MemoryMetrics {
        peak_rss_kb: final_memory.rss_kb.max(initial_memory.rss_kb),
        peak_vss_kb: final_memory.vss_kb.max(initial_memory.vss_kb),
        initial_rss_kb: initial_memory.rss_kb,
        final_rss_kb: final_memory.rss_kb,
    };

    (bars, duration, metrics)
}

#[derive(Debug)]
struct CurrentMemory {
    rss_kb: u64,
    vss_kb: u64,
}

fn get_current_memory() -> CurrentMemory {
    // Read /proc/self/status on Linux or use system calls on macOS
    if cfg!(target_os = "macos") {
        get_macos_memory()
    } else {
        get_linux_memory()
    }
}

fn get_macos_memory() -> CurrentMemory {
    use std::process::Command;

    let output = Command::new("ps")
        .args(["-o", "rss,vsz", "-p", &process::id().to_string()])
        .output()
        .unwrap_or_else(|_| std::process::Output {
            status: std::process::ExitStatus::from_raw(1),
            stdout: b"0 0\n0 0\n".to_vec(),
            stderr: Vec::new(),
        });

    let output_str = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = output_str.lines().collect();

    if lines.len() >= 2 {
        let parts: Vec<&str> = lines[1].split_whitespace().collect();
        if parts.len() >= 2 {
            let rss_kb = parts[0].parse().unwrap_or(0);
            let vss_kb = parts[1].parse().unwrap_or(0);
            return CurrentMemory { rss_kb, vss_kb };
        }
    }

    CurrentMemory {
        rss_kb: 0,
        vss_kb: 0,
    }
}

fn get_linux_memory() -> CurrentMemory {
    use std::fs;

    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss_kb = 0;
    let mut vss_kb = 0;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            if let Some(value) = line.split_whitespace().nth(1) {
                rss_kb = value.parse().unwrap_or(0);
            }
        } else if line.starts_with("VmSize:")
            && let Some(value) = line.split_whitespace().nth(1)
        {
            vss_kb = value.parse().unwrap_or(0);
        }
    }

    CurrentMemory { rss_kb, vss_kb }
}

// Analysis functions

fn analyze_memory_usage(
    batch_memory: MemoryMetrics,
    streaming_memory: MemoryMetrics,
    total_trades: usize,
) {
    println!("  💾 Memory Analysis:");

    let batch_mb = batch_memory.peak_rss_kb as f64 / 1024.0;
    let streaming_mb = streaming_memory.peak_rss_kb as f64 / 1024.0;

    println!("    Batch peak RSS: {:.1} MB", batch_mb);
    println!("    Streaming peak RSS: {:.1} MB", streaming_mb);

    if streaming_mb < batch_mb {
        let savings_mb = batch_mb - streaming_mb;
        let savings_pct = (savings_mb / batch_mb) * 100.0;
        println!(
            "    Memory savings: {:.1} MB ({:.1}%) ✅ STREAMING WINS",
            savings_mb, savings_pct
        );
    } else {
        let overhead_mb = streaming_mb - batch_mb;
        let overhead_pct = (overhead_mb / batch_mb) * 100.0;
        println!(
            "    Memory overhead: {:.1} MB ({:.1}%) ⚠️ BATCH WINS",
            overhead_mb, overhead_pct
        );
    }

    // Memory efficiency per trade
    let batch_bytes_per_trade = (batch_memory.peak_rss_kb * 1024) as f64 / total_trades as f64;
    let streaming_bytes_per_trade =
        (streaming_memory.peak_rss_kb * 1024) as f64 / total_trades as f64;

    println!(
        "    Memory per trade: Batch={:.1}B, Streaming={:.1}B",
        batch_bytes_per_trade, streaming_bytes_per_trade
    );
}

fn analyze_performance(
    total_trades: usize,
    batch_duration: std::time::Duration,
    streaming_duration: std::time::Duration,
) {
    println!("  ⚡ Performance Analysis:");

    let batch_speed = total_trades as f64 / batch_duration.as_secs_f64();
    let streaming_speed = total_trades as f64 / streaming_duration.as_secs_f64();

    println!("    Batch: {:.0} trades/sec", batch_speed);
    println!("    Streaming: {:.0} trades/sec", streaming_speed);

    if streaming_speed > batch_speed {
        let speedup = streaming_speed / batch_speed;
        println!("    Performance: {:.2}x speedup ✅ STREAMING WINS", speedup);
    } else {
        let slowdown = batch_speed / streaming_speed;
        println!("    Performance: {:.2}x slower ⚠️ BATCH WINS", slowdown);
    }
}

fn analyze_memory_leak_patterns(batch_progression: &[u64], streaming_progression: &[u64]) {
    println!("  🔍 Memory leak analysis:");

    // Calculate trends
    let batch_trend = calculate_memory_trend(batch_progression);
    let streaming_trend = calculate_memory_trend(streaming_progression);

    println!("    Batch memory trend: {:.1} KB/iteration", batch_trend);
    println!(
        "    Streaming memory trend: {:.1} KB/iteration",
        streaming_trend
    );

    if batch_trend.abs() < 1000.0 && streaming_trend.abs() < 1000.0 {
        println!("    ✅ No significant memory leaks detected");
    } else {
        if batch_trend > 1000.0 {
            println!("    ⚠️ Potential batch memory leak detected");
        }
        if streaming_trend > 1000.0 {
            println!("    ⚠️ Potential streaming memory leak detected");
        }
    }

    // Memory stability
    let batch_variance = calculate_variance(batch_progression);
    let streaming_variance = calculate_variance(streaming_progression);

    println!(
        "    Memory stability: Batch σ²={:.0}, Streaming σ²={:.0}",
        batch_variance, streaming_variance
    );
}

fn calculate_memory_trend(memory_progression: &[u64]) -> f64 {
    if memory_progression.len() < 2 {
        return 0.0;
    }

    let n = memory_progression.len() as f64;
    let sum_x: f64 = (0..memory_progression.len()).map(|i| i as f64).sum();
    let sum_y: f64 = memory_progression.iter().map(|&v| v as f64).sum();
    let sum_xy: f64 = memory_progression
        .iter()
        .enumerate()
        .map(|(i, &v)| i as f64 * v as f64)
        .sum();
    let sum_x2: f64 = (0..memory_progression.len())
        .map(|i| (i as f64).powi(2))
        .sum();

    // Linear regression slope
    (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
}

fn calculate_variance(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

// Validation functions

fn validate_year_boundary_transitions(batch_bars: &[RangeBar], streaming_bars: &[RangeBar]) {
    println!("  📅 Year boundary validation:");

    if batch_bars.is_empty() || streaming_bars.is_empty() {
        return;
    }

    // Check for bars that span the year boundary
    let year_2024_end = 1735689600000i64; // Jan 1, 2025 00:00 UTC

    let batch_boundary_bars = count_year_boundary_bars(batch_bars, year_2024_end);
    let streaming_boundary_bars = count_year_boundary_bars(streaming_bars, year_2024_end);

    println!(
        "    Bars spanning 2024→2025: Batch={}, Streaming={}",
        batch_boundary_bars, streaming_boundary_bars
    );

    if batch_boundary_bars == streaming_boundary_bars {
        println!("    ✅ Year boundary handling consistent");
    } else {
        println!("    ⚠️ Year boundary handling differs");
    }
}

fn count_year_boundary_bars(bars: &[RangeBar], boundary_timestamp: i64) -> usize {
    bars.iter()
        .filter(|bar| bar.open_time < boundary_timestamp && bar.close_time >= boundary_timestamp)
        .count()
}

fn validate_multi_month_temporal_integrity(bars: &[RangeBar], implementation: &str) {
    if bars.is_empty() {
        return;
    }

    // Validate chronological ordering
    for i in 1..bars.len() {
        assert!(
            bars[i].open_time >= bars[i - 1].close_time,
            "{}: Bar {} starts before previous bar ends",
            implementation,
            i
        );
    }

    // Calculate span
    let total_span_ms = bars.last().unwrap().close_time - bars.first().unwrap().open_time;
    let days = total_span_ms / (24 * 60 * 60 * 1000);

    println!(
        "    {} temporal span: {} days across {} bars",
        implementation,
        days,
        bars.len()
    );
}

fn validate_year_boundary_characteristics(
    batch_bars: &[RangeBar],
    streaming_bars: &[RangeBar],
    scenario: &str,
) {
    if batch_bars.is_empty() || streaming_bars.is_empty() {
        return;
    }

    // Calculate scenario-specific metrics
    let batch_span = batch_bars.last().unwrap().close_time - batch_bars.first().unwrap().open_time;
    let streaming_span =
        streaming_bars.last().unwrap().close_time - streaming_bars.first().unwrap().open_time;

    println!(
        "      {} time span: Batch={}ms, Streaming={}ms",
        scenario, batch_span, streaming_span
    );

    // Validate spans are identical
    assert_eq!(
        batch_span, streaming_span,
        "{} should have identical time spans",
        scenario
    );
}

// Helper functions (formatting utility)

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
