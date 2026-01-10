#!/usr/bin/env cargo run --release --bin tier1_symbol_discovery --
//! Tier-1 Symbol Discovery - Pure Rust Implementation
//!
//! Binance multi-market symbol analyzer focusing on Tier-1 instruments
//! available across all three futures markets (USDT, USDC, Coin-margined).
//!
//! Replaces the Python binance_multi_market_symbol_analyzer.py with a
//! high-performance, zero-dependency Rust implementation.

use chrono::Utc;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use rangebar_config::{CliConfigMerge, Settings};

/// Output format for the symbol analysis
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// Comprehensive database with all market data
    Comprehensive,
    /// Minimal format focused on Tier-1 symbols only
    Minimal,
    /// Spot market equivalent mapping only
    SpotOnly,
}

/// Command-line arguments for the Tier-1 symbol discovery tool
#[derive(Parser)]
#[command(
    name = "tier1-symbol-discovery",
    about = "Binance Multi-Market Symbol Analyzer - Pure Rust Implementation",
    long_about = "
Discovers Tier-1 cryptocurrency symbols available across all three Binance futures markets:
- UM Futures USDT-margined (BTCUSDT, ETHUSDT, etc.)
- UM Futures USDC-margined (BTCUSDC, ETHUSDC, etc.)
- CM Futures Coin-margined (BTCUSD_PERP, ETHUSD_PERP, etc.)

Default behavior focuses on cross-market analysis of Tier-1 multi-market symbols.
Generates machine-discoverable JSON databases for further processing.

Examples:
  tier1-symbol-discovery
  tier1-symbol-discovery --format minimal
  tier1-symbol-discovery --include-single-market
  tier1-symbol-discovery --custom-suffix range_bar_ready
",
    version
)]
struct Args {
    /// Output format
    #[arg(short, long, value_enum, default_value = "comprehensive")]
    format: OutputFormat,

    /// Include single-market symbols in comprehensive output
    #[arg(long)]
    include_single_market: bool,

    /// Add custom suffix to output filenames
    #[arg(short = 's', long)]
    custom_suffix: Option<String>,

    /// Skip creating Claude Code discovery index
    #[arg(long)]
    no_discovery_index: bool,

    /// Validate Tier-1 symbols with live data
    #[arg(short = 'v', long)]
    validate_tier1: bool,
}

impl CliConfigMerge for Args {
    fn merge_into_config(&self, config: &mut Settings) {
        // Currently no CLI args directly override configuration values
        // This binary mainly uses its own CLI args for specific functionality
        // In the future, could add args for API timeouts, retry counts, etc.

        // Enable debug mode if validate_tier1 is set (more verbose output)
        if self.validate_tier1 {
            config.app.debug_mode = true;
        }
    }
}

/// Market contract details for a specific market
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MarketContract {
    symbol: String,
    market_type: String,   // 'USDT', 'USDC', 'COIN'
    contract_type: String, // 'PERPETUAL', 'QUARTERLY', etc.
    settlement_asset: String,
}

/// A crypto base symbol with multi-market availability
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MultiMarketSymbol {
    base_symbol: String,
    spot_equivalent: String, // BASE/USDT format
    market_availability: Vec<String>,
    contracts: HashMap<String, String>, // market_type -> contract_symbol
    contract_details: Vec<MarketContract>,
    priority: String, // 'tier1', 'dual', 'single'
    market_count: usize,
}

/// Comprehensive database metadata
#[derive(Debug, Serialize)]
struct DatabaseMetadata {
    #[serde(rename = "type")]
    db_type: String,
    version: String,
    generated: String,
    data_source: DataSource,
    statistics: DatabaseStatistics,
    claude_code_discovery: DiscoveryMetadata,
}

/// Data source information
#[derive(Debug, Serialize)]
struct DataSource {
    um_futures_api: String,
    cm_futures_api: String,
    generation_method: String,
}

/// Database statistics
#[derive(Debug, Serialize)]
struct DatabaseStatistics {
    total_base_symbols: usize,
    multi_market_symbols: usize,
    tier1_symbols: usize,
    dual_market_symbols: usize,
    single_market_symbols: usize,
}

/// Claude Code discovery metadata
#[derive(Debug, Serialize)]
struct DiscoveryMetadata {
    file_type: String,
    use_cases: Vec<String>,
    machine_readable: bool,
    versioned: bool,
    default_focus: String,
}

/// Spot market mapping entry
#[derive(Debug, Serialize)]
struct SpotMapping {
    base: String,
    spot_symbol: String,
    futures_contracts: Vec<String>,
    market_count: usize,
    priority: String,
}

/// Comprehensive database structure
#[derive(Debug, Serialize)]
struct ComprehensiveDatabase {
    metadata: DatabaseMetadata,
    symbol_database: SymbolDatabase,
    spot_market_mapping: Vec<SpotMapping>,
}

/// Symbol database by category
#[derive(Debug, Serialize)]
struct SymbolDatabase {
    tier1_multi_market: Vec<MultiMarketSymbol>,
    dual_market: Vec<MultiMarketSymbol>,
    #[serde(skip_serializing_if = "Option::is_none")]
    single_market: Option<Vec<MultiMarketSymbol>>,
}

/// Binance API response structure for exchange info
#[derive(Debug, Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

/// Individual symbol information from Binance API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SymbolInfo {
    symbol: String,
    contract_type: Option<String>,
    base_asset: Option<String>,
}

/// Main analyzer for Binance multi-market symbols
struct BinanceMultiMarketAnalyzer {
    um_api_base: String,
    cm_api_base: String,
    client: reqwest::Client,
}

impl BinanceMultiMarketAnalyzer {
    /// Create new analyzer from configuration settings
    fn from_config(config: &Settings) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                config.data.request_timeout_secs,
            ))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            um_api_base: "https://fapi.binance.com/fapi/v1".to_string(),
            cm_api_base: "https://dapi.binance.com/dapi/v1".to_string(),
            client,
        }
    }

    /// Fetch exchange information from both UM and CM futures APIs
    async fn fetch_market_data(
        &self,
    ) -> Result<(ExchangeInfo, ExchangeInfo), Box<dyn std::error::Error>> {
        println!("🔍 Fetching comprehensive Binance futures data...");

        // Fetch UM Futures data
        print!("  📡 UM Futures (USDT/USDC perpetuals)...");
        let um_url = format!("{}/exchangeInfo", self.um_api_base);
        let um_response = self.client.get(&um_url).send().await?;

        // Check response status
        if !um_response.status().is_success() {
            let status = um_response.status();
            let error_text = um_response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response".to_string());
            return Err(format!("UM Futures API error {}: {}", status, error_text).into());
        }

        let response_text = um_response.text().await?;
        let um_data: ExchangeInfo = serde_json::from_str(&response_text).map_err(|e| {
            format!(
                "Failed to parse UM response: {}. Response: {}",
                e,
                &response_text[..response_text.len().min(500)]
            )
        })?;
        println!(" ✅ {} symbols", um_data.symbols.len());

        // Fetch CM Futures data
        print!("  📡 CM Futures (Coin-margined)...");
        let cm_url = format!("{}/exchangeInfo", self.cm_api_base);
        let cm_response = self.client.get(&cm_url).send().await?;

        // Check response status
        if !cm_response.status().is_success() {
            let status = cm_response.status();
            let error_text = cm_response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response".to_string());
            return Err(format!("CM Futures API error {}: {}", status, error_text).into());
        }

        let response_text = cm_response.text().await?;
        let cm_data: ExchangeInfo = serde_json::from_str(&response_text).map_err(|e| {
            format!(
                "Failed to parse CM response: {}. Response: {}",
                e,
                &response_text[..response_text.len().min(500)]
            )
        })?;
        println!(" ✅ {} symbols", cm_data.symbols.len());

        Ok((um_data, cm_data))
    }

    /// Analyze symbols for multi-market availability
    fn analyze_multi_market_symbols(
        &self,
        um_data: &ExchangeInfo,
        cm_data: &ExchangeInfo,
    ) -> BTreeMap<String, MultiMarketSymbol> {
        println!("🔄 Analyzing multi-market symbol availability...");

        let mut symbols_map = BTreeMap::new();

        // Process UM futures (USDT + USDC)
        for symbol_info in &um_data.symbols {
            if symbol_info.contract_type.as_deref() != Some("PERPETUAL") {
                continue;
            }

            let full_symbol = &symbol_info.symbol;
            let (base, market_type, settlement) = if full_symbol.ends_with("USDT") {
                (&full_symbol[..full_symbol.len() - 4], "USDT", "USDT")
            } else if full_symbol.ends_with("USDC") {
                (&full_symbol[..full_symbol.len() - 4], "USDC", "USDC")
            } else {
                continue;
            };

            let entry = symbols_map
                .entry(base.to_string())
                .or_insert_with(|| MultiMarketSymbol {
                    base_symbol: base.to_string(),
                    spot_equivalent: format!("{}/USDT", base),
                    market_availability: Vec::new(),
                    contracts: HashMap::new(),
                    contract_details: Vec::new(),
                    priority: "single".to_string(),
                    market_count: 0,
                });

            entry.market_availability.push(market_type.to_string());
            entry.contracts.insert(
                format!("{}_perpetual", market_type.to_lowercase()),
                full_symbol.clone(),
            );
            entry.contract_details.push(MarketContract {
                symbol: full_symbol.clone(),
                market_type: market_type.to_string(),
                contract_type: "PERPETUAL".to_string(),
                settlement_asset: settlement.to_string(),
            });
        }

        // Process CM futures (Coin-margined)
        for symbol_info in &cm_data.symbols {
            let contract_type = symbol_info.contract_type.as_deref();
            if !matches!(
                contract_type,
                Some("PERPETUAL") | Some("PERPETUAL DELIVERING")
            ) {
                continue;
            }

            let full_symbol = &symbol_info.symbol;
            if let Some(base) = full_symbol.strip_suffix("USD_PERP") {
                let entry =
                    symbols_map
                        .entry(base.to_string())
                        .or_insert_with(|| MultiMarketSymbol {
                            base_symbol: base.to_string(),
                            spot_equivalent: format!("{}/USDT", base),
                            market_availability: Vec::new(),
                            contracts: HashMap::new(),
                            contract_details: Vec::new(),
                            priority: "single".to_string(),
                            market_count: 0,
                        });

                entry.market_availability.push("COIN".to_string());
                entry
                    .contracts
                    .insert("coin_margined".to_string(), full_symbol.clone());
                entry.contract_details.push(MarketContract {
                    symbol: full_symbol.clone(),
                    market_type: "COIN".to_string(),
                    contract_type: contract_type.unwrap_or("PERPETUAL").to_string(),
                    settlement_asset: symbol_info
                        .base_asset
                        .clone()
                        .unwrap_or_else(|| base.to_string()),
                });
            }
        }

        // Calculate market counts and priorities
        for symbol in symbols_map.values_mut() {
            symbol.market_count = symbol.market_availability.len();
            symbol.priority = match symbol.market_count {
                3.. => "tier1".to_string(),
                2 => "dual".to_string(),
                _ => "single".to_string(),
            };
        }

        symbols_map
    }

    /// Generate comprehensive database with metadata
    fn generate_comprehensive_database(
        &self,
        symbols_map: &BTreeMap<String, MultiMarketSymbol>,
        include_single_market: bool,
    ) -> ComprehensiveDatabase {
        let now = Utc::now();

        // Categorize symbols
        let mut tier1_symbols: Vec<_> = symbols_map
            .values()
            .filter(|s| s.priority == "tier1")
            .cloned()
            .collect();
        tier1_symbols.sort_by(|a, b| a.base_symbol.cmp(&b.base_symbol));

        let mut dual_symbols: Vec<_> = symbols_map
            .values()
            .filter(|s| s.priority == "dual")
            .cloned()
            .collect();
        dual_symbols.sort_by(|a, b| a.base_symbol.cmp(&b.base_symbol));

        let mut single_symbols: Vec<_> = symbols_map
            .values()
            .filter(|s| s.priority == "single")
            .cloned()
            .collect();
        single_symbols.sort_by(|a, b| a.base_symbol.cmp(&b.base_symbol));

        // Generate spot market mapping
        let mut spot_mapping: Vec<SpotMapping> = symbols_map
            .values()
            .filter(|s| s.market_count >= 2)
            .map(|symbol| SpotMapping {
                base: symbol.base_symbol.clone(),
                spot_symbol: symbol.spot_equivalent.clone(),
                futures_contracts: symbol.contracts.values().cloned().collect(),
                market_count: symbol.market_count,
                priority: symbol.priority.clone(),
            })
            .collect();
        spot_mapping.sort_by(|a, b| {
            b.market_count
                .cmp(&a.market_count)
                .then(a.base.cmp(&b.base))
        });

        ComprehensiveDatabase {
            metadata: DatabaseMetadata {
                db_type: "binance_multi_market_symbol_database".to_string(),
                version: "1.0".to_string(),
                generated: now.to_rfc3339(),
                data_source: DataSource {
                    um_futures_api: format!("{}/exchangeInfo", self.um_api_base),
                    cm_futures_api: format!("{}/exchangeInfo", self.cm_api_base),
                    generation_method: "comprehensive_cross_market_analysis".to_string(),
                },
                statistics: DatabaseStatistics {
                    total_base_symbols: symbols_map.len(),
                    multi_market_symbols: tier1_symbols.len() + dual_symbols.len(),
                    tier1_symbols: tier1_symbols.len(),
                    dual_market_symbols: dual_symbols.len(),
                    single_market_symbols: single_symbols.len(),
                },
                claude_code_discovery: DiscoveryMetadata {
                    file_type: "binance_multi_market_database".to_string(),
                    use_cases: vec![
                        "range_bar_construction".to_string(),
                        "cross_market_analysis".to_string(),
                        "spot_market_mapping".to_string(),
                        "arbitrage_detection".to_string(),
                        "multi_market_backtesting".to_string(),
                    ],
                    machine_readable: true,
                    versioned: true,
                    default_focus: "tier1_multi_market_symbols".to_string(),
                },
            },
            symbol_database: SymbolDatabase {
                tier1_multi_market: tier1_symbols,
                dual_market: dual_symbols,
                single_market: if include_single_market {
                    Some(single_symbols)
                } else {
                    None
                },
            },
            spot_market_mapping: spot_mapping,
        }
    }

    /// Save database to organized directory structure
    fn save_database(
        &self,
        database: &ComprehensiveDatabase,
        format: &OutputFormat,
        custom_suffix: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S");

        // Create output directory
        let output_dir = Path::new("output/symbol_analysis/current");
        fs::create_dir_all(output_dir)?;

        // Generate filename components
        let base_name = match format {
            OutputFormat::Comprehensive => {
                "binance_multi_market_futures_symbol_database_comprehensive"
            }
            OutputFormat::Minimal => "binance_tier1_multi_market_symbols_minimal",
            OutputFormat::SpotOnly => "binance_spot_equivalent_multi_market_mapping",
        };

        let tier1_count = database.metadata.statistics.tier1_symbols;
        let total_multi = database.metadata.statistics.multi_market_symbols;
        let descriptor = format!("{}tier1_{}total", tier1_count, total_multi);

        let filename = if let Some(suffix) = custom_suffix {
            format!(
                "{}_{}_{}_{}_v1.json",
                base_name, descriptor, suffix, timestamp
            )
        } else {
            format!("{}_{}_{}_v1.json", base_name, descriptor, timestamp)
        };

        // Save file
        let file_path = output_dir.join(&filename);
        let json_content = serde_json::to_string_pretty(database)?;
        fs::write(&file_path, json_content)?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// Generate Tier-1 symbols file for existing Rust pipeline
    fn generate_tier1_file(
        &self,
        symbols_map: &BTreeMap<String, MultiMarketSymbol>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tier1_symbols: Vec<String> = symbols_map
            .values()
            .filter(|s| s.priority == "tier1")
            .map(|s| format!("{}USDT", s.base_symbol))
            .collect();

        // Save to /tmp/ for existing pipeline compatibility
        let tier1_content = tier1_symbols.join("\n");
        fs::write("/tmp/tier1_usdt_pairs.txt", tier1_content)?;

        println!(
            "📄 Generated: /tmp/tier1_usdt_pairs.txt ({} symbols)",
            tier1_symbols.len()
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load configuration with CLI argument overrides
    let config = Settings::load()
        .unwrap_or_else(|_| Settings::default())
        .merge_cli_args(&args);

    println!("🚀 BINANCE MULTI-MARKET SYMBOL ANALYZER (Rust)");
    println!("{}", "=".repeat(55));
    println!("🎯 Default Focus: Cross-market Tier-1 symbols");
    println!("📋 Output Format: {:?}", args.format);
    if config.app.is_debug() {
        println!(
            "🔧 Debug mode enabled (validation: {})",
            args.validate_tier1
        );
    }

    // Initialize analyzer with configuration
    let analyzer = BinanceMultiMarketAnalyzer::from_config(&config);

    // Fetch and analyze data
    let (um_data, cm_data) = analyzer.fetch_market_data().await?;
    let symbols_map = analyzer.analyze_multi_market_symbols(&um_data, &cm_data);

    // Count and display results
    let tier1_count = symbols_map
        .values()
        .filter(|s| s.priority == "tier1")
        .count();
    let dual_count = symbols_map
        .values()
        .filter(|s| s.priority == "dual")
        .count();
    let total_multi = tier1_count + dual_count;

    println!("\n📊 MULTI-MARKET ANALYSIS RESULTS");
    println!("{}", "=".repeat(45));
    println!("🏆 Tier-1 symbols (3 markets): {}", tier1_count);
    println!("🔗 Dual-market symbols: {}", dual_count);
    println!("📈 Total multi-market symbols: {}", total_multi);

    // Show sample Tier-1 symbols
    let mut tier1_symbols: Vec<_> = symbols_map
        .values()
        .filter(|s| s.priority == "tier1")
        .collect();
    tier1_symbols.sort_by(|a, b| a.base_symbol.cmp(&b.base_symbol));

    if !tier1_symbols.is_empty() {
        println!("\n🎯 TIER-1 MULTI-MARKET SYMBOLS:");
        println!("{}", "-".repeat(30));
        for symbol in tier1_symbols.iter().take(10) {
            let markets = symbol.market_availability.join("+");
            println!(
                "  {:<8} → {:<12} ({})",
                symbol.base_symbol, symbol.spot_equivalent, markets
            );
        }

        if tier1_symbols.len() > 10 {
            println!("  ... and {} more Tier-1 symbols", tier1_symbols.len() - 10);
        }
    }

    // Generate and save comprehensive database
    let database =
        analyzer.generate_comprehensive_database(&symbols_map, args.include_single_market);
    let saved_path =
        analyzer.save_database(&database, &args.format, args.custom_suffix.as_deref())?;

    // Generate tier1 symbols file for existing Rust pipeline integration
    analyzer.generate_tier1_file(&symbols_map)?;

    println!("\n💾 MACHINE-DISCOVERABLE RESULTS SAVED");
    println!("{}", "=".repeat(45));
    println!("✅ Primary database: {}", saved_path);
    println!("📄 Pipeline integration: /tmp/tier1_usdt_pairs.txt");

    println!("\n🎯 READY FOR FURTHER PROCESSING");
    println!("{}", "=".repeat(35));
    println!("📋 Use cases:");
    println!("  • Range bar construction across multiple markets");
    println!("  • Cross-market arbitrage analysis");
    println!("  • Spot market equivalent processing");
    println!("  • Multi-market backtesting");

    if tier1_count > 0 {
        println!(
            "\n🏆 {} Tier-1 symbols ready for multi-market range bar construction",
            tier1_count
        );
    }

    println!(
        "\n✅ Analysis completed at {}",
        chrono::Utc::now().format("%H:%M:%S")
    );
    Ok(())
}
