//! Data source and processing configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Data source and processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Base URL for Binance historical data
    pub base_url: String,

    /// Asset class for data fetching
    pub default_asset_class: AssetClass,

    /// Data type for processing
    pub default_data_type: DataType,

    /// Directory for storing downloaded data
    pub download_dir: PathBuf,

    /// Directory for processed/cached data
    pub cache_dir: PathBuf,

    /// Maximum number of concurrent downloads
    pub max_concurrent_downloads: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Number of retry attempts for failed downloads
    pub retry_attempts: usize,

    /// Delay between retry attempts in milliseconds
    pub retry_delay_ms: u64,

    /// Chunk size for processing large files (in trades/bars)
    pub processing_chunk_size: usize,

    /// Enable data compression for cached files
    pub enable_compression: bool,

    /// Enable data integrity checks (checksums)
    pub enable_integrity_checks: bool,
}

/// Binance asset class specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AssetClass {
    /// USD-M Futures (USDT-margined perpetuals)
    Um,
    /// Coin-M Futures (coin-margined perpetuals)
    Cm,
    /// Spot markets
    Spot,
}

/// Data type specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum DataType {
    /// Aggregate trades data
    AggTrades,
    /// OHLCV klines data
    Klines,
    /// Order book depth data
    Depth,
    /// Trade data
    Trades,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            base_url: "https://data.binance.vision/data/".to_string(),
            default_asset_class: AssetClass::Um,
            default_data_type: DataType::AggTrades,
            download_dir: PathBuf::from("./data/downloads"),
            cache_dir: PathBuf::from("./data/cache"),
            max_concurrent_downloads: 5,
            request_timeout_secs: 30,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            processing_chunk_size: 1_000_000,
            enable_compression: true,
            enable_integrity_checks: true,
        }
    }
}

impl DataConfig {
    /// Get the full download URL for a specific symbol and date
    pub fn get_download_url(
        &self,
        asset_class: &AssetClass,
        data_type: &DataType,
        symbol: &str,
        date: &str,
    ) -> String {
        format!(
            "{}{}/{}/{}/{}_{}-aggTrades-{}.zip",
            self.base_url,
            asset_class.to_string().to_lowercase(),
            data_type,
            "daily",
            symbol,
            symbol,
            date
        )
    }

    /// Get cache file path for processed data
    pub fn get_cache_path(&self, symbol: &str, date: &str, suffix: &str) -> PathBuf {
        self.cache_dir
            .join(symbol)
            .join(format!("{}_{}.{}", symbol, date, suffix))
    }

    /// Get download directory for a specific symbol
    pub fn get_symbol_download_dir(&self, symbol: &str) -> PathBuf {
        self.download_dir.join(symbol)
    }
}

impl std::fmt::Display for AssetClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetClass::Um => write!(f, "um"),
            AssetClass::Cm => write!(f, "cm"),
            AssetClass::Spot => write!(f, "spot"),
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::AggTrades => write!(f, "aggTrades"),
            DataType::Klines => write!(f, "klines"),
            DataType::Depth => write!(f, "depth"),
            DataType::Trades => write!(f, "trades"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_url_generation() {
        let config = DataConfig::default();
        let url = config.get_download_url(
            &AssetClass::Um,
            &DataType::AggTrades,
            "BTCUSDT",
            "2024-01-01",
        );

        assert!(url.contains("https://data.binance.vision/data/"));
        assert!(url.contains("um/aggTrades/daily"));
        assert!(url.contains("BTCUSDT"));
        assert!(url.contains("2024-01-01"));
    }

    #[test]
    fn test_cache_path_generation() {
        let config = DataConfig::default();
        let path = config.get_cache_path("BTCUSDT", "2024-01-01", "parquet");

        assert!(path.to_string_lossy().contains("BTCUSDT"));
        assert!(path.to_string_lossy().contains("2024-01-01"));
        assert!(path.to_string_lossy().ends_with(".parquet"));
    }

    #[test]
    fn test_asset_class_display() {
        assert_eq!(AssetClass::Um.to_string(), "um");
        assert_eq!(AssetClass::Cm.to_string(), "cm");
        assert_eq!(AssetClass::Spot.to_string(), "spot");
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(DataType::AggTrades.to_string(), "aggTrades");
        assert_eq!(DataType::Klines.to_string(), "klines");
        assert_eq!(DataType::Depth.to_string(), "depth");
        assert_eq!(DataType::Trades.to_string(), "trades");
    }
}
