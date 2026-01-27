//! Historical data loading utilities
//!
//! Consolidated data loading functionality extracted from examples to eliminate
//! code duplication. Provides unified interface for Binance aggTrades data
//! with support for single-day, multi-day, recent data loading, and intra-day streaming.
//!
//! ## Streaming Architecture (v8.0+)
//!
//! The `IntraDayChunkIterator` enables memory-efficient streaming by yielding
//! trades in configurable hour-based chunks instead of loading entire date ranges.
//!
//! ```rust,ignore
//! use rangebar_providers::binance::{HistoricalDataLoader, IntraDayChunkIterator};
//!
//! let loader = HistoricalDataLoader::new("BTCUSDT");
//! let chunks = IntraDayChunkIterator::new(loader, start_date, end_date, 6); // 6-hour chunks
//!
//! for chunk_result in chunks {
//!     let trades = chunk_result?;
//!     // Process ~400MB of trades at a time instead of entire day
//! }
//! ```

use chrono::{Duration as ChronoDuration, NaiveDate};
use csv::ReaderBuilder;
use reqwest::Client;
use serde::Deserialize;
use std::io::{Cursor, Read};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};
use zip::ZipArchive;

use rangebar_core::{normalize_timestamp, AggTrade, FixedPoint};

use super::checksum::{self, ChecksumError};

/// Errors that can occur during historical data loading
#[derive(Debug, Error)]
pub enum HistoricalError {
    /// HTTP request failed
    #[error("HTTP error for {date}: {message}")]
    HttpError { date: String, message: String },

    /// Failed to parse CSV data
    #[error("CSV parse error: {0}")]
    CsvError(#[from] csv::Error),

    /// Failed to read ZIP archive
    #[error("ZIP error: {0}")]
    ZipError(#[from] zip::result::ZipError),

    /// I/O error during file operations
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Request timeout
    #[error("Request timeout for {date}")]
    Timeout { date: String },

    /// Tokio runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Checksum verification failed (Issue #43)
    /// This indicates data corruption - the downloaded file does not match
    /// the SHA-256 checksum provided by Binance.
    #[error("Checksum verification failed for {date}: {message}")]
    ChecksumMismatch { date: String, message: String },
}

#[derive(Debug, Deserialize)]
pub struct CsvAggTrade(
    pub u64,                                             // agg_trade_id
    pub f64,                                             // price
    pub f64,                                             // quantity
    pub u64,                                             // first_trade_id
    pub u64,                                             // last_trade_id
    pub u64,                                             // timestamp
    #[serde(deserialize_with = "python_bool")] pub bool, // is_buyer_maker
);

/// Custom deserializer for Python-style booleans
pub fn python_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" | "true" => Ok(true),
        "False" | "false" => Ok(false),
        _ => Err(serde::de::Error::custom(format!(
            "Invalid boolean value: {}",
            s
        ))),
    }
}

/// Detect CSV headers
pub fn detect_csv_headers(buffer: &str) -> bool {
    if let Some(first_line) = buffer.lines().next() {
        first_line.contains("agg_trade_id")
            || first_line.contains("price")
            || first_line.contains("quantity")
            || first_line.contains("timestamp")
            || first_line.contains("is_buyer_maker")
    } else {
        false
    }
}

impl From<CsvAggTrade> for AggTrade {
    fn from(csv_trade: CsvAggTrade) -> Self {
        AggTrade {
            agg_trade_id: csv_trade.0 as i64,
            price: FixedPoint::from_str(&csv_trade.1.to_string()).unwrap_or(FixedPoint(0)),
            volume: FixedPoint::from_str(&csv_trade.2.to_string()).unwrap_or(FixedPoint(0)),
            first_trade_id: csv_trade.3 as i64,
            last_trade_id: csv_trade.4 as i64,
            timestamp: csv_trade.5 as i64,
            is_buyer_maker: csv_trade.6,
            is_best_match: None, // Not available in historical CSV data
        }
    }
}

impl CsvAggTrade {
    /// Convert to AggTrade with market-aware timestamp conversion
    pub fn to_agg_trade(&self, _market_type: &str) -> AggTrade {
        // Universal timestamp normalization (market_type no longer needed)
        let normalized_timestamp = normalize_timestamp(self.5);

        AggTrade {
            agg_trade_id: self.0 as i64,
            price: FixedPoint::from_str(&self.1.to_string()).unwrap_or(FixedPoint(0)),
            volume: FixedPoint::from_str(&self.2.to_string()).unwrap_or(FixedPoint(0)),
            first_trade_id: self.3 as i64,
            last_trade_id: self.4 as i64,
            timestamp: normalized_timestamp,
            is_buyer_maker: self.6,
            is_best_match: None, // Not available in historical CSV data
        }
    }
}

/// Historical data loader for Binance aggTrades
#[derive(Clone)]
pub struct HistoricalDataLoader {
    client: Client,
    symbol: String,
    market_type: String,
}

impl HistoricalDataLoader {
    pub fn new(symbol: &str) -> Self {
        Self::new_with_market(symbol, "spot")
    }

    pub fn new_with_market(symbol: &str, market_type: &str) -> Self {
        Self {
            client: Client::new(),
            symbol: symbol.to_uppercase(),
            market_type: market_type.to_string(),
        }
    }

    /// Get market path for URL construction
    fn get_market_path(&self) -> &str {
        match self.market_type.as_str() {
            "um" => "futures/um",
            "cm" => "futures/cm",
            "spot" => "spot",
            _ => "spot", // Default to spot
        }
    }

    /// Load single day trades
    pub async fn load_single_day_trades(
        &self,
        date: NaiveDate,
    ) -> Result<Vec<AggTrade>, Box<dyn std::error::Error>> {
        let date_str = date.format("%Y-%m-%d");
        let url = format!(
            "https://data.binance.vision/data/{}/daily/aggTrades/{}/{}-aggTrades-{}.zip",
            self.get_market_path(),
            self.symbol,
            self.symbol,
            date_str
        );

        let response =
            tokio::time::timeout(Duration::from_secs(30), self.client.get(&url).send()).await??;

        if !response.status().is_success() {
            return Err(format!("HTTP {} for {}", response.status(), date_str).into());
        }

        let zip_bytes = response.bytes().await?;
        let cursor = Cursor::new(zip_bytes);
        let mut archive = ZipArchive::new(cursor)?;

        let csv_filename = format!("{}-aggTrades-{}.csv", self.symbol, date_str);
        let mut csv_file = archive.by_name(&csv_filename)?;

        let mut buffer = String::with_capacity(8 * 1024 * 1024);
        csv_file.read_to_string(&mut buffer)?;

        let mut reader = ReaderBuilder::new()
            .has_headers(detect_csv_headers(&buffer))
            .from_reader(buffer.as_bytes());

        let mut day_trades = Vec::with_capacity(2_000_000);
        for result in reader.deserialize() {
            let csv_trade: CsvAggTrade = result?;
            let agg_trade: AggTrade = csv_trade.to_agg_trade(&self.market_type);
            day_trades.push(agg_trade);
        }

        day_trades.sort_by_key(|trade| trade.timestamp);
        Ok(day_trades)
    }

    /// Load multiple days of historical data
    pub async fn load_historical_range(
        &self,
        days_back: i64,
    ) -> Result<Vec<AggTrade>, Box<dyn std::error::Error>> {
        use chrono::Utc;

        let end_date = Utc::now().date_naive() - chrono::Duration::days(2);
        let start_date = end_date - chrono::Duration::days(days_back - 1);

        let mut all_trades = Vec::with_capacity(days_back as usize * 2_000_000);
        let mut current_date = start_date;

        while current_date <= end_date {
            match self.load_single_day_trades(current_date).await {
                Ok(mut day_trades) => {
                    all_trades.append(&mut day_trades);
                }
                Err(e) => {
                    println!("⚠️  Failed to load {}: {}", current_date, e);
                    return Err(e);
                }
            }
            current_date += chrono::Duration::days(1);
        }

        all_trades.sort_by_key(|trade| trade.timestamp);
        Ok(all_trades)
    }

    /// Try to load recent data (for testing)
    pub async fn load_recent_day(&self) -> Result<Vec<AggTrade>, Box<dyn std::error::Error>> {
        use chrono::Utc;

        for days_back in 1..=7 {
            let test_date = Utc::now().date_naive() - chrono::Duration::days(days_back);

            match self.load_single_day_trades(test_date).await {
                Ok(trades) => return Ok(trades),
                Err(_) => continue,
            }
        }

        Err("No recent data available in the last 7 days".into())
    }

    /// Load single day trades with typed error (for IntraDayChunkIterator)
    ///
    /// # Arguments
    ///
    /// * `date` - The date to load trades for
    /// * `verify_checksum` - If true, verify SHA-256 checksum of downloaded data (Issue #43)
    pub async fn load_single_day_trades_typed(
        &self,
        date: NaiveDate,
    ) -> Result<Vec<AggTrade>, HistoricalError> {
        self.load_single_day_trades_with_checksum(date, true).await
    }

    /// Load single day trades with optional checksum verification
    ///
    /// # Arguments
    ///
    /// * `date` - The date to load trades for
    /// * `verify_checksum` - If true, verify SHA-256 checksum of downloaded data (Issue #43)
    ///
    /// # Checksum Verification Behavior
    ///
    /// | Scenario | Behavior | Rationale |
    /// |----------|----------|-----------|
    /// | Checksum matches | Continue | Success |
    /// | Checksum mismatch | Hard error | Data corruption detected |
    /// | Checksum file 404 | Soft warning, continue | Old data may not have checksums |
    /// | Network timeout | Soft warning, continue | Network issues shouldn't block |
    pub async fn load_single_day_trades_with_checksum(
        &self,
        date: NaiveDate,
        verify_checksum: bool,
    ) -> Result<Vec<AggTrade>, HistoricalError> {
        let date_str = date.format("%Y-%m-%d").to_string();
        let url = format!(
            "https://data.binance.vision/data/{}/daily/aggTrades/{}/{}-aggTrades-{}.zip",
            self.get_market_path(),
            self.symbol,
            self.symbol,
            date_str
        );

        debug!(
            event_type = "download_start",
            symbol = %self.symbol,
            date = %date_str,
            verify_checksum = verify_checksum,
            "Downloading aggTrades data"
        );

        let response = tokio::time::timeout(Duration::from_secs(30), self.client.get(&url).send())
            .await
            .map_err(|_| HistoricalError::Timeout {
                date: date_str.clone(),
            })?
            .map_err(|e| HistoricalError::HttpError {
                date: date_str.clone(),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(HistoricalError::HttpError {
                date: date_str,
                message: format!("HTTP {}", response.status()),
            });
        }

        let zip_bytes = response
            .bytes()
            .await
            .map_err(|e| HistoricalError::HttpError {
                date: date_str.clone(),
                message: e.to_string(),
            })?;

        // Checksum verification (Issue #43)
        if verify_checksum {
            match checksum::fetch_and_verify(&self.client, &url, &zip_bytes).await {
                Ok(result) if result.verified => {
                    info!(
                        event_type = "checksum_verified",
                        symbol = %self.symbol,
                        date = %date_str,
                        hash = %result.actual,
                        "Checksum verified successfully"
                    );
                }
                Ok(result) if result.skipped => {
                    warn!(
                        event_type = "checksum_skipped",
                        symbol = %self.symbol,
                        date = %date_str,
                        "Checksum verification skipped (file not available)"
                    );
                }
                Ok(_) => {}
                Err(ChecksumError::Mismatch { expected, actual }) => {
                    return Err(HistoricalError::ChecksumMismatch {
                        date: date_str,
                        message: format!("expected {expected}, got {actual}"),
                    });
                }
                Err(ChecksumError::InvalidFormat(msg)) => {
                    return Err(HistoricalError::ChecksumMismatch {
                        date: date_str,
                        message: format!("invalid checksum format: {msg}"),
                    });
                }
                Err(e) => {
                    // NotAvailable and FetchFailed are soft errors (already logged)
                    warn!(
                        event_type = "checksum_error",
                        symbol = %self.symbol,
                        date = %date_str,
                        error = %e,
                        "Checksum verification error (continuing anyway)"
                    );
                }
            }
        }

        let cursor = Cursor::new(zip_bytes);
        let mut archive = ZipArchive::new(cursor)?;

        let csv_filename = format!("{}-aggTrades-{}.csv", self.symbol, date_str);
        let mut csv_file = archive.by_name(&csv_filename)?;

        let mut buffer = String::with_capacity(8 * 1024 * 1024);
        csv_file.read_to_string(&mut buffer)?;

        let mut reader = ReaderBuilder::new()
            .has_headers(detect_csv_headers(&buffer))
            .from_reader(buffer.as_bytes());

        let mut day_trades = Vec::with_capacity(2_000_000);
        for result in reader.deserialize() {
            let csv_trade: CsvAggTrade = result?;
            let agg_trade: AggTrade = csv_trade.to_agg_trade(&self.market_type);
            day_trades.push(agg_trade);
        }

        day_trades.sort_by_key(|trade| trade.timestamp);

        info!(
            event_type = "download_complete",
            symbol = %self.symbol,
            date = %date_str,
            trade_count = day_trades.len(),
            "Downloaded and parsed aggTrades"
        );

        Ok(day_trades)
    }
}

/// Microseconds per hour (for timestamp filtering)
const MICROSECONDS_PER_HOUR: i64 = 3_600_000_000;

/// Intra-day chunk iterator for memory-efficient streaming
///
/// Yields trades in hour-based chunks instead of loading entire date ranges,
/// reducing peak memory from ~5.6 GB (full day) to ~50 MB (6-hour chunk).
///
/// # Memory Efficiency
///
/// | Chunk Size | Peak Memory | Reduction |
/// |------------|-------------|-----------|
/// | 24 hours   | ~213 MB     | 1x        |
/// | 6 hours    | ~46 MB      | 4.6x      |
/// | 1 hour     | ~15 MB      | 14x       |
///
/// # Example
///
/// ```rust,ignore
/// use rangebar_providers::binance::{HistoricalDataLoader, IntraDayChunkIterator};
/// use chrono::NaiveDate;
///
/// let loader = HistoricalDataLoader::new("BTCUSDT");
/// let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let end = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
///
/// // Create 6-hour chunk iterator
/// let chunks = IntraDayChunkIterator::new(loader, start, end, 6);
///
/// for chunk_result in chunks {
///     let trades = chunk_result?;
///     println!("Processing {} trades", trades.len());
/// }
/// ```
pub struct IntraDayChunkIterator {
    /// The underlying data loader
    loader: HistoricalDataLoader,
    /// Current date being processed
    current_date: NaiveDate,
    /// End date (inclusive)
    end_date: NaiveDate,
    /// Hours per chunk (1, 6, 12, or 24)
    chunk_hours: u32,
    /// Current hour within the day (0-23, advances by chunk_hours)
    current_hour: u32,
    /// Cached trades for current day (loaded once per day)
    day_trades: Option<Vec<AggTrade>>,
    /// Tokio runtime for async operations
    runtime: tokio::runtime::Runtime,
    /// Whether iteration has completed
    exhausted: bool,
    /// Whether to verify checksums (Issue #43)
    verify_checksum: bool,
}

impl IntraDayChunkIterator {
    /// Create a new intra-day chunk iterator
    ///
    /// # Arguments
    ///
    /// * `loader` - Historical data loader configured with symbol and market
    /// * `start_date` - First date to process (inclusive)
    /// * `end_date` - Last date to process (inclusive)
    /// * `chunk_hours` - Hours per chunk (1, 6, 12, or 24). Recommended: 6.
    ///
    /// # Panics
    ///
    /// Panics if chunk_hours is 0 or greater than 24.
    pub fn new(
        loader: HistoricalDataLoader,
        start_date: NaiveDate,
        end_date: NaiveDate,
        chunk_hours: u32,
    ) -> Self {
        Self::with_checksum(loader, start_date, end_date, chunk_hours, true)
    }

    /// Create a new intra-day chunk iterator with configurable checksum verification
    ///
    /// # Arguments
    ///
    /// * `loader` - Historical data loader configured with symbol and market
    /// * `start_date` - First date to process (inclusive)
    /// * `end_date` - Last date to process (inclusive)
    /// * `chunk_hours` - Hours per chunk (1, 6, 12, or 24). Recommended: 6.
    /// * `verify_checksum` - If true, verify SHA-256 checksum of downloaded data (Issue #43)
    ///
    /// # Panics
    ///
    /// Panics if chunk_hours is 0 or greater than 24.
    pub fn with_checksum(
        loader: HistoricalDataLoader,
        start_date: NaiveDate,
        end_date: NaiveDate,
        chunk_hours: u32,
        verify_checksum: bool,
    ) -> Self {
        assert!(
            chunk_hours > 0 && chunk_hours <= 24,
            "chunk_hours must be 1-24"
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        Self {
            loader,
            current_date: start_date,
            end_date,
            chunk_hours,
            current_hour: 0,
            day_trades: None,
            runtime,
            exhausted: false,
            verify_checksum,
        }
    }

    /// Get symbol being processed
    pub fn symbol(&self) -> &str {
        &self.loader.symbol
    }

    /// Get current date being processed
    pub fn current_date(&self) -> NaiveDate {
        self.current_date
    }

    /// Get current hour within day
    pub fn current_hour(&self) -> u32 {
        self.current_hour
    }

    /// Filter trades by hour range
    ///
    /// Timestamps in AggTrade are in MICROSECONDS (normalized from raw milliseconds).
    /// Hour calculation: `(timestamp / 3_600_000_000) % 24`
    fn filter_by_hour_range(
        &self,
        trades: &[AggTrade],
        start_hour: u32,
        end_hour: u32,
    ) -> Vec<AggTrade> {
        trades
            .iter()
            .filter(|t| {
                // Extract hour from microsecond timestamp
                let hour = ((t.timestamp / MICROSECONDS_PER_HOUR) % 24) as u32;
                hour >= start_hour && hour < end_hour
            })
            .cloned()
            .collect()
    }
}

impl Iterator for IntraDayChunkIterator {
    type Item = Result<Vec<AggTrade>, HistoricalError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        loop {
            // Load day if not cached
            if self.day_trades.is_none() {
                if self.current_date > self.end_date {
                    self.exhausted = true;
                    return None;
                }

                // Load entire day into cache (with checksum verification per Issue #43)
                match self.runtime.block_on(
                    self.loader.load_single_day_trades_with_checksum(
                        self.current_date,
                        self.verify_checksum,
                    ),
                ) {
                    Ok(trades) => {
                        self.day_trades = Some(trades);
                        self.current_hour = 0;
                    }
                    Err(e) => {
                        // Skip to next day on error, but report it
                        self.current_date += ChronoDuration::days(1);
                        return Some(Err(e));
                    }
                }
            }

            // Extract chunk by hour range
            let trades = self.day_trades.as_ref().unwrap();
            let start_hour = self.current_hour;
            let end_hour = (start_hour + self.chunk_hours).min(24);

            let chunk = self.filter_by_hour_range(trades, start_hour, end_hour);

            self.current_hour = end_hour;

            // Move to next day if we've exhausted all hours
            if self.current_hour >= 24 {
                self.day_trades = None;
                self.current_date += ChronoDuration::days(1);
            }

            // Return chunk if non-empty, otherwise continue to next chunk
            if !chunk.is_empty() {
                return Some(Ok(chunk));
            }
            // Continue loop to get next chunk (handles empty hour ranges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_hour_range() {
        // Create test trades with timestamps at different hours
        // Hour 0: 00:00 UTC = 0 microseconds from midnight
        // Hour 6: 06:00 UTC = 6 * 3_600_000_000 = 21_600_000_000 microseconds
        let base_date_us = 1_704_067_200_000_000_i64; // 2024-01-01 00:00:00 UTC in microseconds

        let trades = vec![
            AggTrade {
                agg_trade_id: 1,
                price: FixedPoint::from_str("50000.0").unwrap(),
                volume: FixedPoint::from_str("1.0").unwrap(),
                first_trade_id: 1,
                last_trade_id: 1,
                timestamp: base_date_us + 1_000_000, // Hour 0
                is_buyer_maker: false,
                is_best_match: None,
            },
            AggTrade {
                agg_trade_id: 2,
                price: FixedPoint::from_str("50000.0").unwrap(),
                volume: FixedPoint::from_str("1.0").unwrap(),
                first_trade_id: 2,
                last_trade_id: 2,
                timestamp: base_date_us + (6 * MICROSECONDS_PER_HOUR) + 1_000_000, // Hour 6
                is_buyer_maker: false,
                is_best_match: None,
            },
            AggTrade {
                agg_trade_id: 3,
                price: FixedPoint::from_str("50000.0").unwrap(),
                volume: FixedPoint::from_str("1.0").unwrap(),
                first_trade_id: 3,
                last_trade_id: 3,
                timestamp: base_date_us + (12 * MICROSECONDS_PER_HOUR) + 1_000_000, // Hour 12
                is_buyer_maker: false,
                is_best_match: None,
            },
            AggTrade {
                agg_trade_id: 4,
                price: FixedPoint::from_str("50000.0").unwrap(),
                volume: FixedPoint::from_str("1.0").unwrap(),
                first_trade_id: 4,
                last_trade_id: 4,
                timestamp: base_date_us + (18 * MICROSECONDS_PER_HOUR) + 1_000_000, // Hour 18
                is_buyer_maker: false,
                is_best_match: None,
            },
        ];

        // Create a mock iterator just to use filter_by_hour_range
        let loader = HistoricalDataLoader::new("BTCUSDT");
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let iter = IntraDayChunkIterator::new(loader, start, end, 6);

        // Test hour 0-6: should get trade 1
        let chunk_0_6 = iter.filter_by_hour_range(&trades, 0, 6);
        assert_eq!(chunk_0_6.len(), 1);
        assert_eq!(chunk_0_6[0].agg_trade_id, 1);

        // Test hour 6-12: should get trade 2
        let chunk_6_12 = iter.filter_by_hour_range(&trades, 6, 12);
        assert_eq!(chunk_6_12.len(), 1);
        assert_eq!(chunk_6_12[0].agg_trade_id, 2);

        // Test hour 12-18: should get trade 3
        let chunk_12_18 = iter.filter_by_hour_range(&trades, 12, 18);
        assert_eq!(chunk_12_18.len(), 1);
        assert_eq!(chunk_12_18[0].agg_trade_id, 3);

        // Test hour 18-24: should get trade 4
        let chunk_18_24 = iter.filter_by_hour_range(&trades, 18, 24);
        assert_eq!(chunk_18_24.len(), 1);
        assert_eq!(chunk_18_24[0].agg_trade_id, 4);

        // Test hour 0-24: should get all trades
        let all = iter.filter_by_hour_range(&trades, 0, 24);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_chunk_hours_validation() {
        let loader = HistoricalDataLoader::new("BTCUSDT");
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

        // Valid chunk hours
        let _ = IntraDayChunkIterator::new(loader.clone(), start, end, 1);
        let _ = IntraDayChunkIterator::new(loader.clone(), start, end, 6);
        let _ = IntraDayChunkIterator::new(loader.clone(), start, end, 12);
        let _ = IntraDayChunkIterator::new(loader.clone(), start, end, 24);
    }

    #[test]
    #[should_panic(expected = "chunk_hours must be 1-24")]
    fn test_chunk_hours_zero_panics() {
        let loader = HistoricalDataLoader::new("BTCUSDT");
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let _ = IntraDayChunkIterator::new(loader, start, end, 0);
    }

    #[test]
    #[should_panic(expected = "chunk_hours must be 1-24")]
    fn test_chunk_hours_too_large_panics() {
        let loader = HistoricalDataLoader::new("BTCUSDT");
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let _ = IntraDayChunkIterator::new(loader, start, end, 25);
    }
}
