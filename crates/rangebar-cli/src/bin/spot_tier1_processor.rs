#!/usr/bin/env cargo run --release --bin spot-tier1-processor --
//! Spot Tier-1 Symbol Range Bar Batch Processor
//! Native Rust implementation for continuous spot market range bar generation across all 18 Tier-1 symbols

use chrono::Utc;
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::time::Instant;

use rangebar_providers::binance::get_tier1_symbols;

#[derive(Debug, Serialize, Deserialize)]
struct SpotBatchConfig {
    start_date: String,
    end_date: String,
    threshold_decimal_bps: u32,
    market: String,
    period_description: String,
    data_source: String,
    analysis_type: String,
}

#[derive(Debug, Serialize)]
struct ExecutionResult {
    symbol: String,
    success: bool,
    processing_time_seconds: f64,
    total_trades: Option<u64>,
    total_bars: Option<u64>,
    throughput_trades_per_sec: Option<f64>,
    error_message: Option<String>,
    output_files: Vec<String>,
    execution_timestamp: String,
}

#[derive(Debug, Serialize)]
struct SpotBatchMetadata {
    execution_id: String,
    execution_timestamp: String,
    total_symbols: usize,
    successful_executions: usize,
    failed_executions: usize,
    total_execution_time_seconds: f64,
    symbols: Vec<String>,
    parameters: SpotBatchConfig,
    results: HashMap<String, ExecutionResult>,
    consolidated_statistics: ConsolidatedStatistics,
    output_summary: OutputSummary,
}

#[derive(Debug, Serialize)]
struct ConsolidatedStatistics {
    total_trades_processed: u64,
    total_bars_generated: u64,
    average_processing_time: f64,
    fastest_execution_time: f64,
    slowest_execution_time: f64,
    aggregate_throughput_trades_per_sec: f64,
    symbol_performance_ranking: Vec<SymbolPerformance>,
}

#[derive(Debug, Serialize)]
struct SymbolPerformance {
    symbol: String,
    bars_per_second: f64,
    total_bars: u64,
    processing_time_seconds: f64,
    throughput_trades_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct OutputSummary {
    total_csv_files: usize,
    total_json_files: usize,
    total_output_files: usize,
    naming_convention: String,
    continuous_date_range: String,
    basis_points_format: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "spot-tier1-processor",
    about = "Batch processor for Tier-1 symbol range bars across Binance spot markets",
    long_about = "
Parallel batch processor for generating range bars across all 18 Binance Tier-1 symbols.
Executes rangebar-export for each symbol in parallel using Rayon thread pool.

Features:
- Processes all Tier-1 symbols simultaneously (BTC, ETH, SOL, etc.)
- Configurable parallelism (default: 8 workers)
- Comprehensive execution statistics and performance metrics
- Automatic JSON metadata generation with symbol rankings
- Continuous date range support

Output:
- CSV range bar files: spot_{SYMBOL}_rangebar_{START}_{END}_{BBPS}bps.csv
- JSON metadata: spot_batch_summary.json with execution statistics

Examples:
  spot-tier1-processor --start-date 2024-07-01 --end-date 2024-10-31 --threshold-decimal-bps 250
  spot-tier1-processor --threshold-decimal-bps 500 --workers 16
  spot-tier1-processor --output-dir ./output/custom_batch
",
    version
)]
struct Args {
    /// Start date (YYYY-MM-DD)
    #[arg(long, default_value = "2024-07-01")]
    start_date: String,

    /// End date (YYYY-MM-DD)
    #[arg(long, default_value = "2024-10-31")]
    end_date: String,

    /// Threshold in decimal basis points (e.g., 250 for 25bps = 0.25%)
    #[arg(long, default_value = "250")]
    threshold_decimal_bps: u32,

    /// Output directory
    #[arg(long, default_value = "./output/spot_tier1_batch")]
    output_dir: String,

    /// Number of parallel workers
    #[arg(long, default_value_t = 8)]
    workers: usize,
}

fn execute_single_symbol(
    symbol: &str,
    config: &SpotBatchConfig,
    output_dir: &str,
) -> ExecutionResult {
    let start_time = Instant::now();
    let execution_timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

    println!("🔧 Processing spot market: {}", symbol);

    // Create output directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(output_dir) {
        return ExecutionResult {
            symbol: symbol.to_string(),
            success: false,
            processing_time_seconds: start_time.elapsed().as_secs_f64(),
            total_trades: None,
            total_bars: None,
            throughput_trades_per_sec: None,
            error_message: Some(format!("Failed to create output directory: {}", e)),
            output_files: vec![],
            execution_timestamp,
        };
    }

    // Execute rangebar-export with spot market parameter
    let symbol_pair = format!("{}USDT", symbol);
    let output = Command::new("./target/release/rangebar-export")
        .arg(&symbol_pair)
        .arg(&config.start_date)
        .arg(&config.end_date)
        .arg(config.threshold_decimal_bps.to_string())
        .arg(output_dir)
        .arg("spot") // Explicit spot market
        .output();

    let processing_time = start_time.elapsed().as_secs_f64();

    match output {
        Ok(cmd_output) if cmd_output.status.success() => {
            let stdout = String::from_utf8_lossy(&cmd_output.stdout);

            // Parse output for statistics
            let (total_trades, total_bars) = parse_output_statistics(&stdout);
            let throughput = total_trades.map(|t| t as f64 / processing_time);

            // Find generated output files
            let output_files =
                find_output_files(output_dir, &symbol_pair, config.threshold_decimal_bps);

            println!("✅ {}: {:.2}s", symbol_pair, processing_time);

            ExecutionResult {
                symbol: symbol_pair,
                success: true,
                processing_time_seconds: processing_time,
                total_trades,
                total_bars,
                throughput_trades_per_sec: throughput,
                error_message: None,
                output_files,
                execution_timestamp,
            }
        }
        Ok(cmd_output) => {
            let stderr = String::from_utf8_lossy(&cmd_output.stderr);
            println!("❌ {}: Failed - {}", symbol_pair, stderr);

            ExecutionResult {
                symbol: symbol_pair,
                success: false,
                processing_time_seconds: processing_time,
                total_trades: None,
                total_bars: None,
                throughput_trades_per_sec: None,
                error_message: Some(stderr.to_string()),
                output_files: vec![],
                execution_timestamp,
            }
        }
        Err(e) => {
            println!("💥 {}: Exception - {}", symbol_pair, e);

            ExecutionResult {
                symbol: symbol_pair,
                success: false,
                processing_time_seconds: processing_time,
                total_trades: None,
                total_bars: None,
                throughput_trades_per_sec: None,
                error_message: Some(format!("Command execution failed: {}", e)),
                output_files: vec![],
                execution_timestamp,
            }
        }
    }
}

fn parse_output_statistics(output: &str) -> (Option<u64>, Option<u64>) {
    let mut total_trades = None;
    let mut total_bars = None;

    for line in output.lines() {
        if line.contains("Total Trades:")
            && let Some(trades_str) = line.split(':').nth(1)
        {
            total_trades = trades_str.trim().replace(',', "").parse().ok();
        }
        if line.contains("Total Bars:")
            && let Some(bars_str) = line.split(':').nth(1)
        {
            total_bars = bars_str.trim().replace(',', "").parse().ok();
        }
    }

    (total_trades, total_bars)
}

fn find_output_files(output_dir: &str, symbol: &str, threshold_decimal_bps: u32) -> Vec<String> {
    let mut files = Vec::new();

    // Expected filename patterns based on new BPS naming convention
    let patterns = [
        format!(
            "spot_{}_rangebar_*_{:04}bps.csv",
            symbol, threshold_decimal_bps
        ),
        format!(
            "spot_{}_rangebar_*_{:04}bps.json",
            symbol, threshold_decimal_bps
        ),
    ];

    for _pattern in &patterns {
        if let Ok(entries) = fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let filename = entry.file_name().to_string_lossy().to_string();
                if filename.contains("spot_")
                    && filename.contains(symbol)
                    && filename.contains(&format!("{:04}bps", threshold_decimal_bps))
                {
                    files.push(filename);
                }
            }
        }
    }

    files
}

fn calculate_consolidated_statistics(
    results: &HashMap<String, ExecutionResult>,
) -> ConsolidatedStatistics {
    let successful_results: Vec<_> = results.values().filter(|r| r.success).collect();

    let total_trades_processed = successful_results
        .iter()
        .filter_map(|r| r.total_trades)
        .sum();

    let total_bars_generated = successful_results.iter().filter_map(|r| r.total_bars).sum();

    let processing_times: Vec<f64> = successful_results
        .iter()
        .map(|r| r.processing_time_seconds)
        .collect();

    let average_processing_time = if !processing_times.is_empty() {
        processing_times.iter().sum::<f64>() / processing_times.len() as f64
    } else {
        0.0
    };

    let fastest_execution_time = processing_times
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let slowest_execution_time = processing_times.iter().cloned().fold(0.0, f64::max);

    let aggregate_throughput = if average_processing_time > 0.0 {
        total_trades_processed as f64 / average_processing_time
    } else {
        0.0
    };

    let mut symbol_performance: Vec<SymbolPerformance> = successful_results
        .iter()
        .map(|r| SymbolPerformance {
            symbol: r.symbol.clone(),
            bars_per_second: r.total_bars.unwrap_or(0) as f64 / r.processing_time_seconds,
            total_bars: r.total_bars.unwrap_or(0),
            processing_time_seconds: r.processing_time_seconds,
            throughput_trades_per_sec: r.throughput_trades_per_sec.unwrap_or(0.0),
        })
        .collect();

    symbol_performance.sort_by(|a, b| {
        b.throughput_trades_per_sec
            .partial_cmp(&a.throughput_trades_per_sec)
            .unwrap()
    });

    ConsolidatedStatistics {
        total_trades_processed,
        total_bars_generated,
        average_processing_time,
        fastest_execution_time,
        slowest_execution_time,
        aggregate_throughput_trades_per_sec: aggregate_throughput,
        symbol_performance_ranking: symbol_performance,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let start_time = Instant::now();
    let execution_timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    let execution_id = format!("spot_batch_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    println!("🚀 Spot Tier-1 Range Bar Batch Processor");
    println!("📅 Date Range: {} to {}", args.start_date, args.end_date);
    println!(
        "🎯 Threshold: {} bps ({:.2}%)",
        args.threshold_decimal_bps,
        args.threshold_decimal_bps as f64 / 100.0
    );
    println!("📂 Output: {}", args.output_dir);
    println!("🔧 Workers: {}", args.workers);

    // Set up rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build_global()
        .unwrap();

    // Load Tier-1 symbols
    let symbols = get_tier1_symbols();
    println!("📊 Processing {} Tier-1 symbols", symbols.len());

    // Create configuration
    let config = SpotBatchConfig {
        start_date: args.start_date.clone(),
        end_date: args.end_date.clone(),
        threshold_decimal_bps: args.threshold_decimal_bps,
        market: "spot".to_string(),
        period_description: format!("{} to {} (continuous)", args.start_date, args.end_date),
        data_source: "Binance spot market aggTrades".to_string(),
        analysis_type: "continuous_range_bar_generation".to_string(),
    };

    // Create output directory
    fs::create_dir_all(&args.output_dir)?;

    // Execute parallel processing
    println!("\n🔄 Starting parallel execution...");
    let results: HashMap<String, ExecutionResult> = symbols
        .par_iter()
        .map(|symbol| {
            let result = execute_single_symbol(symbol, &config, &args.output_dir);
            (result.symbol.clone(), result)
        })
        .collect();

    let total_execution_time = start_time.elapsed().as_secs_f64();

    // Calculate statistics
    let successful_executions = results.values().filter(|r| r.success).count();
    let failed_executions = results.len() - successful_executions;
    let consolidated_statistics = calculate_consolidated_statistics(&results);

    // Count output files
    let csv_count = results
        .values()
        .map(|r| {
            r.output_files
                .iter()
                .filter(|f| f.ends_with(".csv"))
                .count()
        })
        .sum();
    let json_count = results
        .values()
        .map(|r| {
            r.output_files
                .iter()
                .filter(|f| f.ends_with(".json"))
                .count()
        })
        .sum();

    let output_summary = OutputSummary {
        total_csv_files: csv_count,
        total_json_files: json_count,
        total_output_files: csv_count + json_count,
        naming_convention: "spot_{SYMBOL}_rangebar_{STARTDATE}_{ENDDATE}_{BBPS}bps.{ext}"
            .to_string(),
        continuous_date_range: format!("{} to {}", args.start_date, args.end_date),
        basis_points_format: format!("{:04}bps", config.threshold_decimal_bps),
    };

    // Create metadata
    let metadata = SpotBatchMetadata {
        execution_id: execution_id.clone(),
        execution_timestamp: execution_timestamp.clone(),
        total_symbols: symbols.len(),
        successful_executions,
        failed_executions,
        total_execution_time_seconds: total_execution_time,
        symbols,
        parameters: config,
        results,
        consolidated_statistics,
        output_summary,
    };

    // Save metadata
    let metadata_file = format!("{}/spot_batch_summary.json", args.output_dir);
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_file, metadata_json)?;

    // Print summary
    println!("\n✅ Spot Batch Processing Complete!");
    println!("📊 Execution ID: {}", execution_id);
    println!("⏱️  Total Time: {:.1}s", total_execution_time);
    println!("✅ Successful: {}", successful_executions);
    println!("❌ Failed: {}", failed_executions);
    println!("📁 CSV Files: {}", csv_count);
    println!("📁 JSON Files: {}", json_count);
    println!("📄 Summary: {}", metadata_file);

    if failed_executions > 0 {
        println!("\n⚠️  Failed symbols:");
        for (symbol, result) in &metadata.results {
            if !result.success {
                println!(
                    "   ❌ {}: {}",
                    symbol,
                    result.error_message.as_deref().unwrap_or("Unknown error")
                );
            }
        }
    }

    Ok(())
}
