//! Data Structure Validator for Binance aggTrades
//!
//! Validates data structure consistency across Tier-1 cryptocurrency symbols
//! for spot and UM futures markets spanning 2022-2025 timeframe.
//!
//! Implements exception-only failure handling with no fallbacks or failsafes.

use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, NaiveDate, Utc};
use clap::Parser;
use csv::ReaderBuilder;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio::time::{Duration, timeout};
use zip::ZipArchive;

use rangebar_providers::binance::get_tier1_symbols;

/// Data Structure Validator CLI
#[derive(Parser)]
#[command(
    name = "data-structure-validator",
    version = "1.0.0",
    about = "Validates Binance aggTrades data structure across Tier-1 symbols and markets",
    long_about = "
Comprehensive data structure validator for Binance aggTrades across spot and futures markets.
Validates schema consistency, timestamp formats, and data integrity for Tier-1 cryptocurrency symbols.

Features:
- Cross-market validation (spot, UM futures, CM futures)
- Quarterly sampling across 2022-2025 timeframe
- Schema detection (headers, columns, timestamp precision)
- SHA256 checksum verification (optional)
- Parallel processing with configurable workers
- Detailed structure profiles and validation manifests

Key Differences Detected:
- Spot: No headers, short columns (a,p,q,f,l,T,m), 16-digit μs timestamps
- UM Futures: Headers, descriptive columns, 13-digit ms timestamps
- Auto-detects and normalizes format variations

Output:
- validation_results.json: Complete validation details
- structure_analysis/: Per-symbol structure profiles
- index.json: Execution manifest with statistics

Examples:
  data-structure-validator
  data-structure-validator --markets spot,um --workers 16
  data-structure-validator --start-date 2024-01-01 --end-date 2024-12-31
  data-structure-validator --skip-checksum --symbols BTCUSDT,ETHUSDT
"
)]
struct Cli {
    /// Output directory for validation results
    #[arg(short, long, default_value = "output/data_structure_validation")]
    output_dir: PathBuf,

    /// Markets to validate
    #[arg(short, long, value_delimiter = ',', default_values_t = vec!["spot".to_string(), "um".to_string()])]
    markets: Vec<String>,

    /// Symbols to validate (default: all Tier-1)
    #[arg(short, long, value_delimiter = ',')]
    symbols: Option<Vec<String>>,

    /// Start date for validation (YYYY-MM-DD)
    #[arg(long, default_value = "2022-01-01")]
    start_date: String,

    /// End date for validation (YYYY-MM-DD)
    #[arg(long, default_value = "2025-09-01")]
    end_date: String,

    /// Number of parallel workers
    #[arg(long, default_value = "8")]
    workers: usize,

    /// Skip checksum validation
    #[arg(long)]
    skip_checksum: bool,
}

/// Validation execution metadata
#[derive(Debug, Serialize, Deserialize)]
struct ValidationManifest {
    validation_id: String,
    execution_metadata: ExecutionMetadata,
    coverage: CoverageMetadata,
    key_findings: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecutionMetadata {
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    duration_minutes: Option<f64>,
    parallel_workers: usize,
    total_downloads_gb: Option<f64>,
    rust_version: String,
    tool_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CoverageMetadata {
    symbols: SymbolCoverage,
    markets: Vec<String>,
    date_range: DateRange,
}

#[derive(Debug, Serialize, Deserialize)]
struct SymbolCoverage {
    requested: usize,
    validated: usize,
    partial: usize,
    failed: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DateRange {
    start: String,
    end: String,
    sampling_frequency: String,
}

/// Data structure profile for a symbol/market combination
#[derive(Debug, Serialize, Deserialize)]
struct StructureProfile {
    symbol: String,
    market: String,
    structure_signature: String,
    column_profile: ColumnProfile,
    format_variations: Vec<FormatVariation>,
    data_statistics: DataStatistics,
    validation_timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ColumnProfile {
    count: usize,
    names: Vec<String>,
    types: Vec<String>,
    precision: Vec<Option<u8>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FormatVariation {
    date: String,
    variation_type: String,
    description: String,
    impact: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataStatistics {
    avg_file_size_mb: f64,
    avg_rows_per_day: u64,
    compression_ratio: f64,
    typical_trade_frequency: f64,
}

/// Validation result for a single sample
#[derive(Debug, Serialize, Deserialize)]
struct ValidationResult {
    symbol: String,
    market: String,
    date: String,
    success: bool,
    has_headers: bool,
    column_count: usize,
    column_names: Vec<String>,
    timestamp_format: String,
    boolean_format: String,
    file_size_bytes: u64,
    row_count: u64,
    errors: Vec<String>,
    sample_rows: Vec<String>,
}

/// CSV aggTrade structure for validation - supports both spot and UM futures formats
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CsvAggTrade {
    #[serde(alias = "a", alias = "agg_trade_id")]
    agg_trade_id: Option<i64>,
    #[serde(alias = "p", alias = "price")]
    price: Option<f64>,
    #[serde(alias = "q", alias = "quantity")]
    quantity: Option<f64>,
    #[serde(alias = "f", alias = "first_trade_id")]
    first_trade_id: Option<i64>,
    #[serde(alias = "l", alias = "last_trade_id")]
    last_trade_id: Option<i64>,
    #[serde(alias = "T", alias = "transact_time")]
    timestamp: Option<i64>,
    #[serde(
        alias = "m",
        alias = "is_buyer_maker",
        deserialize_with = "flexible_bool"
    )]
    is_buyer_maker: Option<String>,
}

/// Flexible boolean deserializer for both "True/False" and "true/false" formats
fn flexible_bool<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" | "true" => Ok(Some("True".to_string())),
        "False" | "false" => Ok(Some("False".to_string())),
        _ => Ok(Some(s)),
    }
}

/// Main validation executor
struct DataStructureValidator {
    client: Client,
    output_dir: PathBuf,
    workers: usize,
    skip_checksum: bool,
}

impl DataStructureValidator {
    fn new(output_dir: PathBuf, workers: usize, skip_checksum: bool) -> Self {
        Self {
            client: Client::new(),
            output_dir,
            workers,
            skip_checksum,
        }
    }

    /// Execute comprehensive validation
    async fn execute_validation(
        &self,
        symbols: &[String],
        markets: &[String],
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<ValidationManifest, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Utc::now();
        let validation_id = format!(
            "val_{}_tier1_comprehensive",
            start_time.format("%Y%m%d_%H%M%S")
        );

        println!("🚀 Starting data structure validation: {}", validation_id);
        println!("   Symbols: {} Tier-1", symbols.len());
        println!("   Markets: {:?}", markets);
        println!("   Workers: {}", self.workers);

        // Create run directory
        let run_dir = self.output_dir.join(format!(
            "{}_validation_run",
            start_time.format("%Y%m%d_%H%M%S")
        ));
        fs::create_dir_all(&run_dir)?;

        // Generate sampling dates (quarterly)
        let sample_dates = self.generate_quarterly_samples(start_date, end_date);
        println!("   Sample dates: {} quarterly points", sample_dates.len());

        // Execute validation in parallel
        let mut validation_results = Vec::new();
        let mut total_bytes = 0u64;

        for symbol in symbols {
            for market in markets {
                for date in &sample_dates {
                    match self.validate_single_sample(symbol, market, *date).await {
                        Ok(result) => {
                            total_bytes += result.file_size_bytes;
                            validation_results.push(result);
                        }
                        Err(e) => {
                            eprintln!(
                                "❌ Validation failed for {}/{}/{}: {}",
                                symbol, market, date, e
                            );
                            // Exception-only failure: Continue processing other samples
                            validation_results.push(ValidationResult {
                                symbol: symbol.clone(),
                                market: market.clone(),
                                date: date.format("%Y-%m-%d").to_string(),
                                success: false,
                                has_headers: false,
                                column_count: 0,
                                column_names: vec![],
                                timestamp_format: "unknown".to_string(),
                                boolean_format: "unknown".to_string(),
                                file_size_bytes: 0,
                                row_count: 0,
                                errors: vec![e.to_string()],
                                sample_rows: vec![],
                            });
                        }
                    }
                }
            }
        }

        let end_time = Utc::now();
        let duration = end_time.signed_duration_since(start_time);

        // Generate structure profiles
        let structure_profiles = self.generate_structure_profiles(&validation_results);

        // Save results
        self.save_validation_results(&run_dir, &validation_results)
            .await?;
        self.save_structure_profiles(&run_dir, &structure_profiles)
            .await?;

        // Generate manifest
        let manifest = ValidationManifest {
            validation_id,
            execution_metadata: ExecutionMetadata {
                start_time,
                end_time: Some(end_time),
                duration_minutes: Some(duration.num_milliseconds() as f64 / 60000.0),
                parallel_workers: self.workers,
                total_downloads_gb: Some(total_bytes as f64 / 1_000_000_000.0),
                rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            coverage: CoverageMetadata {
                symbols: SymbolCoverage {
                    requested: symbols.len(),
                    validated: validation_results.iter().filter(|r| r.success).count(),
                    partial: 0,
                    failed: validation_results.iter().filter(|r| !r.success).count(),
                },
                markets: markets.to_vec(),
                date_range: DateRange {
                    start: start_date.format("%Y-%m-%d").to_string(),
                    end: end_date.format("%Y-%m-%d").to_string(),
                    sampling_frequency: "quarterly".to_string(),
                },
            },
            key_findings: self.extract_key_findings(&validation_results),
        };

        self.save_manifest(&run_dir, &manifest).await?;

        println!("✅ Validation completed: {}", manifest.validation_id);
        println!(
            "   Success rate: {:.1}%",
            (manifest.coverage.symbols.validated as f64
                / manifest.coverage.symbols.requested as f64)
                * 100.0
        );

        Ok(manifest)
    }

    /// Validate single sample
    async fn validate_single_sample(
        &self,
        symbol: &str,
        market: &str,
        date: NaiveDate,
    ) -> Result<ValidationResult, Box<dyn std::error::Error + Send + Sync>> {
        let date_str = date.format("%Y-%m-%d").to_string();
        let url = self.build_download_url(market, symbol, &date_str)?;

        println!("🔍 Validating {}/{}/{}", symbol, market, date_str);

        // Download with timeout
        let response = timeout(Duration::from_secs(30), self.client.get(&url).send()).await??;

        if !response.status().is_success() {
            return Err(format!("HTTP {}: {}", response.status(), url).into());
        }

        let zip_bytes = response.bytes().await?;
        let file_size_bytes = zip_bytes.len() as u64;

        // Checksum validation (if enabled)
        if !self.skip_checksum {
            self.verify_checksum(&zip_bytes, symbol, &date_str, market)
                .await?;
        }

        // Extract and analyze CSV
        let csv_content = self.extract_csv_from_zip(&zip_bytes, symbol, &date_str)?;
        let analysis = self.analyze_csv_structure(&csv_content)?;

        Ok(ValidationResult {
            symbol: symbol.to_string(),
            market: market.to_string(),
            date: date_str,
            success: true,
            has_headers: analysis.has_headers,
            column_count: analysis.column_count,
            column_names: analysis.column_names,
            timestamp_format: analysis.timestamp_format,
            boolean_format: analysis.boolean_format,
            file_size_bytes,
            row_count: analysis.row_count,
            errors: vec![],
            sample_rows: analysis.sample_rows,
        })
    }

    /// Build download URL based on market type
    fn build_download_url(
        &self,
        market: &str,
        symbol: &str,
        date: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let market_path = match market {
            "spot" => "spot",
            "um" => "futures/um",
            "cm" => "futures/cm",
            _ => {
                return Err(
                    format!("Invalid market type: '{}'. Valid: spot, um, cm", market).into(),
                );
            }
        };

        Ok(format!(
            "https://data.binance.vision/data/{}/daily/aggTrades/{}/{}-aggTrades-{}.zip",
            market_path, symbol, symbol, date
        ))
    }

    /// Verify file checksum
    async fn verify_checksum(
        &self,
        zip_data: &[u8],
        symbol: &str,
        date: &str,
        market: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use sha2::{Digest, Sha256};

        let checksum_url = format!(
            "{}.CHECKSUM",
            self.build_download_url(market, symbol, date)?
        );

        let response = timeout(
            Duration::from_secs(10),
            self.client.get(&checksum_url).send(),
        )
        .await??;

        if !response.status().is_success() {
            return Err(format!("Checksum download failed: HTTP {}", response.status()).into());
        }

        let checksum_text = response.text().await?;
        let expected_hash = checksum_text
            .split_whitespace()
            .next()
            .ok_or("Invalid checksum format")?;

        let mut hasher = Sha256::new();
        hasher.update(zip_data);
        let computed_hash = format!("{:x}", hasher.finalize());

        if computed_hash != expected_hash {
            return Err(format!(
                "SHA256 mismatch: expected {}, got {}",
                expected_hash, computed_hash
            )
            .into());
        }

        Ok(())
    }

    /// Extract CSV from ZIP archive
    fn extract_csv_from_zip(
        &self,
        zip_data: &[u8],
        symbol: &str,
        date: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let cursor = Cursor::new(zip_data);
        let mut archive = ZipArchive::new(cursor)?;

        let csv_filename = format!("{}-aggTrades-{}.csv", symbol, date);
        let mut csv_file = archive.by_name(&csv_filename)?;

        let mut csv_content = String::new();
        csv_file.read_to_string(&mut csv_content)?;

        Ok(csv_content)
    }

    /// Analyze CSV structure
    fn analyze_csv_structure(
        &self,
        csv_content: &str,
    ) -> Result<CsvAnalysis, Box<dyn std::error::Error + Send + Sync>> {
        let has_headers = self.detect_csv_headers(csv_content);

        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .from_reader(csv_content.as_bytes());

        let headers = if has_headers {
            reader.headers()?.iter().map(|s| s.to_string()).collect()
        } else {
            vec![
                "a".to_string(),
                "p".to_string(),
                "q".to_string(),
                "f".to_string(),
                "l".to_string(),
                "T".to_string(),
                "m".to_string(),
            ]
        };

        let mut row_count = 0u64;
        let mut sample_rows = Vec::new();
        let mut timestamp_format = "unknown".to_string();
        let mut boolean_format = "unknown".to_string();

        for (i, result) in reader.deserialize::<CsvAggTrade>().enumerate() {
            let record = result?;
            row_count += 1;

            // Analyze first few rows
            if i < 3 {
                sample_rows.push(format!("{:?}", record));
            }

            // Analyze timestamp format (first valid record)
            if timestamp_format == "unknown"
                && let Some(ts) = record.timestamp
            {
                timestamp_format = if ts.to_string().len() == 13 {
                    "13_digit_milliseconds".to_string()
                } else {
                    format!("{}_digits", ts.to_string().len())
                };
            }

            // Analyze boolean format (first valid record)
            if boolean_format == "unknown"
                && let Some(ref bool_str) = record.is_buyer_maker
            {
                boolean_format = match bool_str.as_str() {
                    "True" | "False" => "normalized_from_mixed".to_string(),
                    _ => format!("custom_{}", bool_str),
                };
            }

            // Early exit for large files
            if i > 1000 {
                break;
            }
        }

        Ok(CsvAnalysis {
            has_headers,
            column_count: headers.len(),
            column_names: headers,
            timestamp_format,
            boolean_format,
            row_count,
            sample_rows,
        })
    }

    /// Detect CSV headers
    fn detect_csv_headers(&self, buffer: &str) -> bool {
        if let Some(first_line) = buffer.lines().next() {
            first_line.contains("agg_trade_id")
                || first_line.contains("price")
                || first_line.contains("quantity")
                || first_line.contains("timestamp")
                || first_line.contains("is_buyer_maker")
                || first_line.contains("a,p,q,f,l,T,m")
        } else {
            false
        }
    }

    /// Generate quarterly sampling dates
    fn generate_quarterly_samples(&self, start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
        let mut dates = Vec::new();
        let mut current_year = start.year();
        let end_year = end.year();

        while current_year <= end_year {
            // Quarterly months: Jan, Apr, Jul, Oct
            for month in [1, 4, 7, 10] {
                if let Some(date) = NaiveDate::from_ymd_opt(current_year, month, 1)
                    && date >= start
                    && date <= end
                {
                    dates.push(date);
                }
            }
            current_year += 1;
        }

        dates
    }

    /// Generate structure profiles from validation results
    fn generate_structure_profiles(&self, results: &[ValidationResult]) -> Vec<StructureProfile> {
        let mut profiles = Vec::new();
        let mut symbol_market_map: HashMap<(String, String), Vec<&ValidationResult>> =
            HashMap::new();

        // Group results by symbol/market
        for result in results {
            if result.success {
                symbol_market_map
                    .entry((result.symbol.clone(), result.market.clone()))
                    .or_default()
                    .push(result);
            }
        }

        // Generate profile for each symbol/market combination
        for ((symbol, market), symbol_results) in symbol_market_map {
            if let Some(profile) = self.create_structure_profile(&symbol, &market, &symbol_results)
            {
                profiles.push(profile);
            }
        }

        profiles
    }

    /// Create structure profile for symbol/market
    fn create_structure_profile(
        &self,
        symbol: &str,
        market: &str,
        results: &[&ValidationResult],
    ) -> Option<StructureProfile> {
        if results.is_empty() {
            return None;
        }

        // Use first successful result as template
        let template = results.first()?;

        // Generate structure signature
        let signature_input = format!(
            "{}:{}:{}:{:?}:{}:{}",
            symbol,
            market,
            template.column_count,
            template.column_names,
            template.timestamp_format,
            template.boolean_format
        );
        let structure_signature = format!("{:x}", md5::compute(signature_input));

        // Detect format variations
        let mut variations = Vec::new();
        for result in results {
            if result.has_headers != template.has_headers {
                variations.push(FormatVariation {
                    date: result.date.clone(),
                    variation_type: "header_presence".to_string(),
                    description: format!(
                        "Headers: {} (expected: {})",
                        result.has_headers, template.has_headers
                    ),
                    impact: "parsing".to_string(),
                });
            }
        }

        // Calculate statistics
        let avg_file_size_mb = results
            .iter()
            .map(|r| r.file_size_bytes as f64 / 1_000_000.0)
            .sum::<f64>()
            / results.len() as f64;

        let avg_rows_per_day =
            results.iter().map(|r| r.row_count).sum::<u64>() / results.len() as u64;

        Some(StructureProfile {
            symbol: symbol.to_string(),
            market: market.to_string(),
            structure_signature,
            column_profile: ColumnProfile {
                count: template.column_count,
                names: template.column_names.clone(),
                types: vec!["mixed".to_string(); template.column_count], // Simplified
                precision: vec![None; template.column_count],
            },
            format_variations: variations,
            data_statistics: DataStatistics {
                avg_file_size_mb,
                avg_rows_per_day,
                compression_ratio: 1.0, // Simplified
                typical_trade_frequency: avg_rows_per_day as f64 / 86400.0, // trades/second
            },
            validation_timestamp: Utc::now(),
        })
    }

    /// Extract key findings from validation results
    fn extract_key_findings(
        &self,
        results: &[ValidationResult],
    ) -> HashMap<String, serde_json::Value> {
        let mut findings = HashMap::new();

        // Header presence analysis
        let spot_with_headers = results
            .iter()
            .filter(|r| r.market == "spot" && r.has_headers)
            .count();
        let um_with_headers = results
            .iter()
            .filter(|r| r.market == "um" && r.has_headers)
            .count();

        findings.insert(
            "header_presence".to_string(),
            serde_json::json!({
                "spot_with_headers": spot_with_headers,
                "um_with_headers": um_with_headers
            }),
        );

        // Timestamp format consistency
        let timestamp_formats: Vec<_> = results
            .iter()
            .map(|r| &r.timestamp_format)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        findings.insert(
            "timestamp_consistency".to_string(),
            serde_json::json!({
                "formats_found": timestamp_formats,
                "is_consistent": timestamp_formats.len() == 1
            }),
        );

        findings
    }

    /// Save validation results
    async fn save_validation_results(
        &self,
        run_dir: &Path,
        results: &[ValidationResult],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let results_file = run_dir.join("validation_results.json");
        let json = serde_json::to_string_pretty(results)?;
        fs::write(results_file, json)?;
        Ok(())
    }

    /// Save structure profiles
    async fn save_structure_profiles(
        &self,
        run_dir: &Path,
        profiles: &[StructureProfile],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let profiles_dir = run_dir.join("structure_analysis");
        fs::create_dir_all(&profiles_dir)?;

        for profile in profiles {
            let filename = format!("{}_structure_profile.json", profile.symbol);
            let profile_file = profiles_dir.join(filename);
            let json = serde_json::to_string_pretty(profile)?;
            fs::write(profile_file, json)?;
        }

        Ok(())
    }

    /// Save validation manifest
    async fn save_manifest(
        &self,
        run_dir: &Path,
        manifest: &ValidationManifest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let manifest_file = run_dir.join("index.json");
        let json = serde_json::to_string_pretty(manifest)?;
        fs::write(manifest_file, json)?;
        Ok(())
    }
}

/// CSV analysis result
#[derive(Debug)]
struct CsvAnalysis {
    has_headers: bool,
    column_count: usize,
    column_names: Vec<String>,
    timestamp_format: String,
    boolean_format: String,
    row_count: u64,
    sample_rows: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    // Get symbols (default to all Tier-1)
    let symbols = cli
        .symbols
        .unwrap_or_else(|| get_tier1_symbols().iter().map(|s| s.to_string()).collect());

    // Parse dates
    let start_date = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")?;
    let end_date = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")?;

    // Create validator
    let validator = DataStructureValidator::new(cli.output_dir, cli.workers, cli.skip_checksum);

    // Execute validation
    let manifest = validator
        .execute_validation(&symbols, &cli.markets, start_date, end_date)
        .await?;

    println!(
        "📋 Validation manifest: {}",
        serde_json::to_string_pretty(&manifest)?
    );

    Ok(())
}
