//! Export and output configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Export and output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Default output directory for all exports
    pub default_output_dir: PathBuf,

    /// Default output format for range bars
    pub default_format: OutputFormat,

    /// Enable compression for output files
    pub enable_compression: bool,

    /// Include metadata in output files
    pub include_metadata: bool,

    /// Include statistical summaries in output
    pub include_statistics: bool,

    /// Enable progress reporting during export
    pub show_progress: bool,

    /// Maximum number of bars per output file (0 = unlimited)
    pub max_bars_per_file: usize,

    /// File naming pattern for output files
    pub file_naming_pattern: FileNamingPattern,

    /// Timestamp format for file names
    pub timestamp_format: TimestampFormat,

    /// Include symbol in output file names
    pub include_symbol_in_filename: bool,

    /// Include threshold in output file names
    pub include_threshold_in_filename: bool,

    /// Enable parallel export processing
    pub enable_parallel_export: bool,

    /// Number of export worker threads
    pub export_worker_threads: Option<usize>,

    /// Buffer size for streaming exports (in bars)
    pub streaming_buffer_size: usize,

    /// Enable data validation during export
    pub validate_export_data: bool,

    /// Create timestamped subdirectories
    pub create_timestamped_dirs: bool,
}

/// Supported output formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Comma-separated values
    Csv,
    /// Apache Parquet format
    Parquet,
    /// JavaScript Object Notation
    Json,
    /// Tab-separated values
    Tsv,
    /// Binary format (custom)
    Binary,
}

/// File naming patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FileNamingPattern {
    /// Simple: symbol_threshold.ext
    Simple,
    /// Detailed: symbol_threshold_startdate_enddate.ext
    Detailed,
    /// Timestamped: symbol_threshold_YYYYMMDD_HHMMSS.ext
    Timestamped,
    /// ISO format: symbol_threshold_YYYY-MM-DDTHH:MM:SS.ext
    Iso,
    /// Custom pattern
    Custom(String),
}

/// Timestamp formats for file naming
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampFormat {
    /// YYYYMMDD
    Date,
    /// YYYYMMDD_HHMMSS
    DateTime,
    /// YYYY-MM-DD
    IsoDate,
    /// YYYY-MM-DDTHH:MM:SS
    IsoDateTime,
    /// Unix timestamp
    Unix,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            default_output_dir: PathBuf::from("./output"),
            default_format: OutputFormat::Csv,
            enable_compression: false,
            include_metadata: true,
            include_statistics: true,
            show_progress: true,
            max_bars_per_file: 0, // Unlimited
            file_naming_pattern: FileNamingPattern::Detailed,
            timestamp_format: TimestampFormat::Date,
            include_symbol_in_filename: true,
            include_threshold_in_filename: true,
            enable_parallel_export: true,
            export_worker_threads: None, // Auto-detect
            streaming_buffer_size: 10_000,
            validate_export_data: true,
            create_timestamped_dirs: false,
        }
    }
}

impl ExportConfig {
    /// Get the number of export worker threads
    pub fn export_worker_threads(&self) -> usize {
        self.export_worker_threads
            .unwrap_or_else(|| (num_cpus::get() / 2).max(1))
    }

    /// Generate filename based on configuration
    pub fn generate_filename(
        &self,
        symbol: &str,
        threshold_decimal_bps: u32,
        start_date: Option<&str>,
        end_date: Option<&str>,
        extension: Option<&str>,
    ) -> String {
        let ext = extension.unwrap_or_else(|| self.default_format.file_extension());
        let threshold_str = if self.include_threshold_in_filename {
            format!("_{:04}bps", threshold_decimal_bps)
        } else {
            String::new()
        };

        let symbol_str = if self.include_symbol_in_filename {
            symbol.to_string()
        } else {
            "rangebar".to_string()
        };

        match &self.file_naming_pattern {
            FileNamingPattern::Simple => {
                format!("{}{}.{}", symbol_str, threshold_str, ext)
            }
            FileNamingPattern::Detailed => {
                if let (Some(start), Some(end)) = (start_date, end_date) {
                    format!(
                        "{}{}_{}_to_{}.{}",
                        symbol_str, threshold_str, start, end, ext
                    )
                } else {
                    format!("{}{}.{}", symbol_str, threshold_str, ext)
                }
            }
            FileNamingPattern::Timestamped => {
                let timestamp = self.format_current_timestamp();
                format!("{}{}_{}.{}", symbol_str, threshold_str, timestamp, ext)
            }
            FileNamingPattern::Iso => {
                let timestamp = self.format_current_timestamp_iso();
                format!("{}{}_{}.{}", symbol_str, threshold_str, timestamp, ext)
            }
            FileNamingPattern::Custom(pattern) => pattern
                .replace("{symbol}", symbol)
                .replace("{threshold}", &format!("{:04}bps", threshold_decimal_bps))
                .replace("{extension}", ext)
                .replace("{timestamp}", &self.format_current_timestamp()),
        }
    }

    /// Get output directory, creating timestamped subdirectory if enabled
    pub fn get_output_dir(&self) -> PathBuf {
        if self.create_timestamped_dirs {
            let timestamp = self.format_current_timestamp();
            self.default_output_dir.join(timestamp)
        } else {
            self.default_output_dir.clone()
        }
    }

    /// Format current timestamp based on configuration
    fn format_current_timestamp(&self) -> String {
        let now = chrono::Utc::now();
        match self.timestamp_format {
            TimestampFormat::Date => now.format("%Y%m%d").to_string(),
            TimestampFormat::DateTime => now.format("%Y%m%d_%H%M%S").to_string(),
            TimestampFormat::IsoDate => now.format("%Y-%m-%d").to_string(),
            TimestampFormat::IsoDateTime => now.format("%Y-%m-%dT%H:%M:%S").to_string(),
            TimestampFormat::Unix => now.timestamp().to_string(),
        }
    }

    /// Format current timestamp in ISO format
    fn format_current_timestamp_iso(&self) -> String {
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
    }
}

impl OutputFormat {
    /// Get file extension for the format
    pub fn file_extension(&self) -> &'static str {
        match self {
            OutputFormat::Csv => "csv",
            OutputFormat::Parquet => "parquet",
            OutputFormat::Json => "json",
            OutputFormat::Tsv => "tsv",
            OutputFormat::Binary => "bin",
        }
    }

    /// Check if format supports compression
    pub fn supports_compression(&self) -> bool {
        matches!(self, OutputFormat::Parquet | OutputFormat::Binary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_extensions() {
        assert_eq!(OutputFormat::Csv.file_extension(), "csv");
        assert_eq!(OutputFormat::Parquet.file_extension(), "parquet");
        assert_eq!(OutputFormat::Json.file_extension(), "json");
        assert_eq!(OutputFormat::Tsv.file_extension(), "tsv");
        assert_eq!(OutputFormat::Binary.file_extension(), "bin");
    }

    #[test]
    fn test_output_format_compression_support() {
        assert!(!OutputFormat::Csv.supports_compression());
        assert!(OutputFormat::Parquet.supports_compression());
        assert!(!OutputFormat::Json.supports_compression());
        assert!(!OutputFormat::Tsv.supports_compression());
        assert!(OutputFormat::Binary.supports_compression());
    }

    #[test]
    fn test_filename_generation() {
        let config = ExportConfig::default();

        // Test simple pattern
        let mut config_simple = config.clone();
        config_simple.file_naming_pattern = FileNamingPattern::Simple;
        let filename = config_simple.generate_filename("BTCUSDT", 250, None, None, None);
        assert_eq!(filename, "BTCUSDT_0250bps.csv");

        // Test detailed pattern
        let filename_detailed = config.generate_filename(
            "BTCUSDT",
            800,
            Some("2024-01-01"),
            Some("2024-01-02"),
            Some("parquet"),
        );
        assert_eq!(
            filename_detailed,
            "BTCUSDT_0800bps_2024-01-01_to_2024-01-02.parquet"
        );
    }

    #[test]
    fn test_export_worker_threads() {
        let config = ExportConfig::default();
        let threads = config.export_worker_threads();

        // Should be at least 1, at most half of available CPUs
        assert!(threads >= 1);
        assert!(threads <= num_cpus::get());
    }

    #[test]
    fn test_output_dir_with_timestamp() {
        let config = ExportConfig {
            create_timestamped_dirs: true,
            ..Default::default()
        };

        let dir = config.get_output_dir();
        assert!(dir.to_string_lossy().contains("output"));
        // Should contain some timestamp component
        assert!(dir.file_name().is_some());
    }
}
