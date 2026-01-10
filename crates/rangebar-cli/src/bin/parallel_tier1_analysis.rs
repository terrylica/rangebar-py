#!/usr/bin/env cargo run --release --bin parallel_tier1_analysis --
//! Parallel Tier-1 Symbol Range Bar Analysis
//! Native Rust implementation for 6-month analysis on 18 Tier-1 USDT pairs

use chrono::Utc;
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::time::Instant;

use rangebar_config::{CliConfigMerge, Settings};

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisConfig {
    start_date: String,
    end_date: String,
    threshold: f64,
    threshold_pct: String,
    period_days: u32,
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
struct ParallelExecutionMetadata {
    execution_id: String,
    execution_timestamp: String,
    total_symbols: usize,
    successful_executions: usize,
    failed_executions: usize,
    total_execution_time_seconds: f64,
    symbols: Vec<String>,
    parameters: AnalysisConfig,
    results: HashMap<String, ExecutionResult>,
    consolidated_statistics: ConsolidatedStatistics,
    claude_code_discovery: DiscoveryMetadata,
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
struct DiscoveryMetadata {
    workspace_path: String,
    analysis_ready: bool,
    machine_readable: bool,
    tags: Vec<String>,
    discovery_file: String,
}

fn load_configuration() -> Result<(AnalysisConfig, Vec<String>), Box<dyn std::error::Error>> {
    // Load configuration
    let config_content = fs::read_to_string("/tmp/range_bar_analysis_config.json")?;
    let config: AnalysisConfig = serde_json::from_str(&config_content)?;

    // Load Tier-1 symbols from Rust-generated file (primary source)
    let symbols_content = fs::read_to_string("/tmp/tier1_usdt_pairs.txt")?;
    let symbols: Vec<String> = symbols_content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    println!(
        "📊 Loaded {} Tier-1 USDT pairs from Rust-generated symbol discovery",
        symbols.len()
    );

    Ok((config, symbols))
}

fn execute_single_symbol(
    symbol: &str,
    config: &AnalysisConfig,
    output_dir: &str,
) -> ExecutionResult {
    let start_time = Instant::now();
    let execution_timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

    println!("🔧 Starting {}", symbol);

    // Create symbol-specific output directory
    let symbol_output = format!("{}/individual/{}", output_dir, symbol);
    if let Err(e) = fs::create_dir_all(&symbol_output) {
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

    // Execute the range bar binary with SPOT data as default (no need to specify "spot")
    let output = Command::new("./target/release/rangebar")
        .arg(symbol)
        .arg(&config.start_date)
        .arg(&config.end_date)
        .arg(config.threshold.to_string())
        .arg(&symbol_output)
        .output();

    let processing_time = start_time.elapsed().as_secs_f64();

    match output {
        Ok(cmd_output) if cmd_output.status.success() => {
            let stdout = String::from_utf8_lossy(&cmd_output.stdout);

            // Parse output for statistics
            let (total_trades, total_bars) = parse_output_statistics(&stdout);
            let throughput = total_trades.map(|t| t as f64 / processing_time);

            // Find generated output files
            let output_files = find_output_files(&symbol_output);

            println!("✅ {}: {:.2}s", symbol, processing_time);

            ExecutionResult {
                symbol: symbol.to_string(),
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
            println!("❌ {}: Failed - {}", symbol, stderr);

            ExecutionResult {
                symbol: symbol.to_string(),
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
            println!("💥 {}: Exception - {}", symbol, e);

            ExecutionResult {
                symbol: symbol.to_string(),
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

fn parse_output_statistics(stdout: &str) -> (Option<u64>, Option<u64>) {
    let mut total_trades = None;
    let mut total_bars = None;

    for line in stdout.lines() {
        // Extract total trades
        if line.contains("trades loaded")
            && let Some(trades_str) = line.split("trades loaded").next()
            && let Ok(trades) = trades_str
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u64>()
        {
            total_trades = Some(trades);
        }

        // Extract total bars
        if line.contains("Total Bars:")
            && let Some(bars_str) = line.split("Total Bars:").nth(1)
            && let Ok(bars) = bars_str.trim().parse::<u64>()
        {
            total_bars = Some(bars);
        }
    }

    (total_trades, total_bars)
}

fn find_output_files(output_dir: &str) -> Vec<String> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension()
                && (ext == "csv" || ext == "json")
                && let Some(path_str) = path.to_str()
            {
                files.push(path_str.to_string());
            }
        }
    }

    files.sort();
    files
}

fn generate_consolidated_statistics(
    results: &HashMap<String, ExecutionResult>,
) -> ConsolidatedStatistics {
    let successful_results: Vec<&ExecutionResult> =
        results.values().filter(|r| r.success).collect();

    let total_trades = successful_results
        .iter()
        .filter_map(|r| r.total_trades)
        .sum::<u64>();

    let total_bars = successful_results
        .iter()
        .filter_map(|r| r.total_bars)
        .sum::<u64>();

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

    let total_processing_time: f64 = processing_times.iter().sum();
    let aggregate_throughput = if total_processing_time > 0.0 {
        total_trades as f64 / total_processing_time
    } else {
        0.0
    };

    // Generate performance ranking
    let mut symbol_performance: Vec<SymbolPerformance> = successful_results
        .iter()
        .filter_map(|r| {
            r.total_bars.map(|bars| {
                let bars_per_second = bars as f64 / r.processing_time_seconds;
                SymbolPerformance {
                    symbol: r.symbol.clone(),
                    bars_per_second,
                    total_bars: bars,
                    processing_time_seconds: r.processing_time_seconds,
                    throughput_trades_per_sec: r.throughput_trades_per_sec.unwrap_or(0.0),
                }
            })
        })
        .collect();

    symbol_performance.sort_by(|a, b| b.bars_per_second.partial_cmp(&a.bars_per_second).unwrap());

    ConsolidatedStatistics {
        total_trades_processed: total_trades,
        total_bars_generated: total_bars,
        average_processing_time,
        fastest_execution_time,
        slowest_execution_time,
        aggregate_throughput_trades_per_sec: aggregate_throughput,
        symbol_performance_ranking: symbol_performance,
    }
}

fn save_results(
    metadata: &ParallelExecutionMetadata,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create consolidated output directory
    let consolidated_dir = format!("{}/consolidated", output_dir);
    fs::create_dir_all(&consolidated_dir)?;

    let discovery_dir = format!("{}/discovery", output_dir);
    fs::create_dir_all(&discovery_dir)?;

    // Save metadata
    let metadata_file = format!(
        "{}/parallel_execution_metadata_{}.json",
        consolidated_dir, metadata.execution_id
    );
    let metadata_json = serde_json::to_string_pretty(metadata)?;
    fs::write(&metadata_file, metadata_json)?;

    // Save discovery file
    let discovery_file = format!(
        "{}/{}",
        discovery_dir, metadata.claude_code_discovery.discovery_file
    );
    let discovery_data = serde_json::json!({
        "type": "parallel_tier1_analysis_results_rust_native",
        "execution_id": metadata.execution_id,
        "execution_timestamp": metadata.execution_timestamp,
        "summary": {
            "total_symbols": metadata.total_symbols,
            "successful_executions": metadata.successful_executions,
            "total_execution_time": metadata.total_execution_time_seconds,
            "aggregate_throughput": metadata.consolidated_statistics.aggregate_throughput_trades_per_sec
        },
        "files": {
            "metadata": metadata_file,
            "individual_results_dir": format!("{}/individual", output_dir)
        },
        "claude_code_analysis_ready": true,
        "tradability_analysis_ready": true,
        "native_rust_execution": true
    });

    fs::write(
        &discovery_file,
        serde_json::to_string_pretty(&discovery_data)?,
    )?;

    println!("💾 Execution metadata: {}", metadata_file);
    println!("🔍 Discovery file: {}", discovery_file);
    println!("📁 Individual results: {}/individual/", output_dir);

    Ok(())
}

/// Command-line arguments for the parallel Tier-1 symbol analysis tool
#[derive(Parser)]
#[command(
    name = "rangebar-analyze",
    version = "0.4.4",
    about = "Parallel Tier-1 Symbol Range Bar Analysis",
    long_about = "Native Rust implementation for parallel analysis of 18 Tier-1 USDT cryptocurrency pairs using range bar construction. Processes massive datasets (1B+ ticks) with optimal performance and memory efficiency."
)]
struct AnalysisArgs {
    /// Run the parallel analysis (default behavior)
    #[arg(
        long,
        help = "Execute parallel range bar analysis on all Tier-1 symbols"
    )]
    run: bool,

    /// Show detailed configuration information
    #[arg(
        long,
        help = "Display current analysis configuration without executing"
    )]
    config: bool,

    /// List available Tier-1 symbols
    #[arg(long, help = "Show the 18 Tier-1 USDT pairs that will be analyzed")]
    list_symbols: bool,

    /// Show system resource information
    #[arg(
        long,
        help = "Display available CPU cores and memory for parallel processing"
    )]
    system_info: bool,
}

impl CliConfigMerge for AnalysisArgs {
    fn merge_into_config(&self, config: &mut Settings) {
        // Enable debug mode if config is requested for detailed output
        if self.config {
            config.app.debug_mode = true;
        }

        // This binary is parallel by nature, so ensure performance settings are optimal
        if self.run {
            config.app.enable_metrics = true;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = AnalysisArgs::parse();

    // Load configuration with CLI argument overrides
    let _config = Settings::load()
        .unwrap_or_else(|_| Settings::default())
        .merge_cli_args(&args);

    // Handle informational flags first (user safety)
    if args.config {
        show_configuration()?;
        return Ok(());
    }

    if args.list_symbols {
        show_tier1_symbols()?;
        return Ok(());
    }

    if args.system_info {
        show_system_info();
        return Ok(());
    }

    // Default behavior: run analysis
    run_parallel_analysis()?;

    Ok(())
}

fn run_parallel_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let execution_id = format!("rust_parallel_tier1_{}", Utc::now().format("%Y%m%d_%H%M%S"));

    println!("🚀 Rust Native Parallel Tier-1 Symbol Range Bar Analysis");
    println!("📊 Execution ID: {}", execution_id);
    println!("🔧 Pure Rust Pipeline: tier1-symbol-discovery → parallel analysis");
    println!("{}", "=".repeat(80));

    // Load configuration and symbols
    let (config, symbols) = load_configuration()?;

    println!("📋 Configuration:");
    println!(
        "   Period: {} to {} ({} days)",
        config.start_date, config.end_date, config.period_days
    );
    println!("   Threshold: {}%", config.threshold * 100.0);
    println!("   Symbols: {} Tier-1 USDT pairs", symbols.len());
    println!(
        "   Parallel Workers: {} (Rayon)",
        rayon::current_num_threads()
    );
    println!();

    // Setup output directories
    let output_dir = "./output/tier1_analysis";
    fs::create_dir_all(format!("{}/individual", output_dir))?;
    fs::create_dir_all(format!("{}/consolidated", output_dir))?;
    fs::create_dir_all(format!("{}/discovery", output_dir))?;

    // Start parallel execution
    let execution_start = Instant::now();
    let execution_timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

    println!(
        "⏱️  Starting parallel execution at {}",
        Utc::now().format("%Y-%m-%d %H:%M:%S")
    );
    println!(
        "🔧 Using {} parallel workers (Rayon)",
        rayon::current_num_threads()
    );
    println!();

    // Execute symbols in parallel using Rayon
    let results: HashMap<String, ExecutionResult> = symbols
        .par_iter()
        .map(|symbol| {
            let result = execute_single_symbol(symbol, &config, output_dir);
            (symbol.clone(), result)
        })
        .collect();

    let total_execution_time = execution_start.elapsed().as_secs_f64();

    println!(
        "\n✅ Parallel execution completed in {:.2} seconds",
        total_execution_time
    );

    // Generate consolidated statistics
    let consolidated_stats = generate_consolidated_statistics(&results);

    let successful_count = results.values().filter(|r| r.success).count();
    let failed_count = results.len() - successful_count;

    println!("📊 Consolidated Statistics:");
    println!("   Successful: {}/{}", successful_count, results.len());
    println!("   Total Bars: {}", consolidated_stats.total_bars_generated);
    println!(
        "   Total Trades: {}",
        consolidated_stats.total_trades_processed
    );
    println!(
        "   Aggregate Throughput: {:.0} trades/sec",
        consolidated_stats.aggregate_throughput_trades_per_sec
    );

    // Create final metadata
    let metadata = ParallelExecutionMetadata {
        execution_id: execution_id.clone(),
        execution_timestamp,
        total_symbols: symbols.len(),
        successful_executions: successful_count,
        failed_executions: failed_count,
        total_execution_time_seconds: total_execution_time,
        symbols,
        parameters: config,
        results,
        consolidated_statistics: consolidated_stats,
        claude_code_discovery: DiscoveryMetadata {
            workspace_path: "/Users/terryli/eon/rangebar".to_string(),
            analysis_ready: true,
            machine_readable: true,
            tags: vec![
                "rust_native_execution".to_string(),
                "parallel_execution".to_string(),
                "tier1_analysis".to_string(),
                "18_symbols".to_string(),
                "6_months".to_string(),
                "tradability".to_string(),
                "rayon_parallel".to_string(),
            ],
            discovery_file: format!(
                "rust_parallel_tier1_analysis_{}_discovery.json",
                execution_id
            ),
        },
    };

    // Save results
    save_results(&metadata, output_dir)?;

    println!("\n🎉 Rust native parallel analysis completed successfully!");
    println!("📊 Execution ID: {}", execution_id);
    println!("🔍 Results are machine-discoverable and analysis-ready");

    Ok(())
}

/// Show current analysis configuration (user safety)
fn show_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Rangebar Parallel Analysis Configuration");
    println!("{}", "=".repeat(50));

    // Load the same configuration that would be used for analysis
    if let Ok((config, _)) = load_configuration() {
        println!(
            "📅 Analysis Period: {} to {}",
            config.start_date, config.end_date
        );
        println!("📊 Period Length: {} days", config.period_days);
        println!(
            "🎯 Range Bar Threshold: {}% ({})",
            config.threshold * 100.0,
            config.threshold_pct
        );
        println!("💾 Data Source: {}", config.data_source);
        println!("🔬 Analysis Type: {}", config.analysis_type);
    } else {
        println!("⚠️  Configuration file not found at /tmp/range_bar_analysis_config.json");
        println!("💡 Default configuration will be used if analysis runs");
    }

    println!("\n🔧 System Configuration:");
    println!("🧵 Available CPU cores: {}", rayon::current_num_threads());
    // Rust version info removed for compilation simplicity
    println!("📦 Rangebar version: {}", env!("CARGO_PKG_VERSION"));

    Ok(())
}

/// Show available Tier-1 symbols (user safety)
fn show_tier1_symbols() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 Tier-1 Cryptocurrency Symbols");
    println!("{}", "=".repeat(40));

    // Load symbols from the same source used by analysis
    if let Ok((_, symbols)) = load_configuration() {
        println!("📊 Total Tier-1 USDT pairs: {}", symbols.len());
        println!("\n🎯 Symbols to be analyzed:");

        for (i, symbol) in symbols.iter().enumerate() {
            println!("  {:2}. {}", i + 1, symbol);
        }

        println!("\n💡 These symbols are discovered by tier1-symbol-discovery");
        println!("   Available across all Binance futures markets (USDT, USDC, Coin-margined)");
    } else {
        println!("⚠️  Symbol list not found at /tmp/tier1_usdt_pairs.txt");
        println!("💡 Run 'cargo run --bin tier1-symbol-discovery' first");
    }

    Ok(())
}

/// Show system resource information (user safety)
fn show_system_info() {
    println!("💻 System Resource Information");
    println!("{}", "=".repeat(40));

    println!("🧵 CPU Cores:");
    println!(
        "   Available for parallel processing: {}",
        rayon::current_num_threads()
    );
    // Logical core count requires additional dependency
    // Physical core count requires additional dependency

    println!("\n⚡ Performance Expectations:");
    println!("   • Range bar construction: ~1B ticks processed");
    println!("   • Expected analysis time: 5-30 minutes per symbol");
    println!("   • Memory usage: ~1-2GB peak during processing");
    println!("   • Output: Comprehensive JSON + CSV results");

    println!("\n🎯 Optimization:");
    println!("   • Pure Rust implementation for maximum performance");
    println!(
        "   • Parallel execution across all {} cores",
        rayon::current_num_threads()
    );
    println!("   • Memory-efficient streaming processing");
    println!("   • Non-blocking I/O for optimal throughput");
}
