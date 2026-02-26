// FILE-SIZE-OK: Canonical location for all Binance historical data loading (Vision + REST + fromId)
// Issue #92: REST recency backfill + fromId pagination (source-validation skill)
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

    /// REST API error (Issue #92: recency backfill)
    #[error("REST API error: {0}")]
    RestApiError(String),

    /// Rate limited by Binance API (HTTP 429)
    #[error("Rate limited for {symbol}: exceeded {max_retries} retries")]
    RateLimited { symbol: String, max_retries: u32 },
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

/// Binance REST API `/api/v3/aggTrades` response format (Issue #92)
///
/// JSON keys are short single-letter names from the Binance API:
/// `a` = agg_trade_id, `p` = price, `q` = quantity, `f` = first_trade_id,
/// `l` = last_trade_id, `T` = trade_time, `m` = is_buyer_maker
#[derive(Debug, Deserialize)]
pub struct RestAggTrade {
    #[serde(rename = "a")]
    pub agg_trade_id: i64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "f")]
    pub first_trade_id: i64,
    #[serde(rename = "l")]
    pub last_trade_id: i64,
    #[serde(rename = "T")]
    pub trade_time: i64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
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

impl From<RestAggTrade> for AggTrade {
    fn from(rest: RestAggTrade) -> Self {
        AggTrade {
            agg_trade_id: rest.agg_trade_id,
            price: FixedPoint::from_str(&rest.price).unwrap_or(FixedPoint(0)),
            volume: FixedPoint::from_str(&rest.quantity).unwrap_or(FixedPoint(0)),
            first_trade_id: rest.first_trade_id,
            last_trade_id: rest.last_trade_id,
            timestamp: normalize_timestamp(rest.trade_time as u64),
            is_buyer_maker: rest.is_buyer_maker,
            is_best_match: None,
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

/// Maximum retries for HTTP 429 rate limiting
const RATE_LIMIT_MAX_RETRIES: u32 = 5;
/// Initial backoff in milliseconds (doubles each retry: 1s → 2s → 4s → 8s → 16s)
const RATE_LIMIT_INITIAL_BACKOFF_MS: u64 = 1_000;

/// Send an HTTP request with exponential backoff on 429 rate limiting.
///
/// Retries up to `RATE_LIMIT_MAX_RETRIES` times with doubling backoff.
/// Non-429 errors are returned immediately.
async fn send_with_rate_limit_backoff(
    request_builder: impl Fn() -> reqwest::RequestBuilder,
    symbol: &str,
) -> Result<reqwest::Response, HistoricalError> {
    let mut backoff_ms = RATE_LIMIT_INITIAL_BACKOFF_MS;

    for attempt in 0..=RATE_LIMIT_MAX_RETRIES {
        let response = tokio::time::timeout(
            Duration::from_secs(30),
            request_builder().send(),
        )
        .await
        .map_err(|_| HistoricalError::RestApiError(format!(
            "Timeout fetching {} (attempt {})",
            symbol, attempt
        )))?
        .map_err(|e| HistoricalError::RestApiError(format!(
            "HTTP error for {}: {e}",
            symbol
        )))?;

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            if attempt == RATE_LIMIT_MAX_RETRIES {
                return Err(HistoricalError::RateLimited {
                    symbol: symbol.to_string(),
                    max_retries: RATE_LIMIT_MAX_RETRIES,
                });
            }
            warn!(
                symbol = %symbol,
                attempt = attempt,
                backoff_ms = backoff_ms,
                "Rate limited (429), backing off"
            );
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(16_000); // Cap at 16s
            continue;
        }

        return Ok(response);
    }

    // Unreachable due to loop structure, but satisfies compiler
    Err(HistoricalError::RateLimited {
        symbol: symbol.to_string(),
        max_retries: RATE_LIMIT_MAX_RETRIES,
    })
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

    /// Fetch aggregated trades from Binance REST API (Issue #92: recency backfill).
    ///
    /// Paginates through `/api/v3/aggTrades` (max 1000 per request).
    /// Trades are returned sorted by timestamp in ascending order.
    ///
    /// # Arguments
    ///
    /// * `start_ms` - Start time in milliseconds (inclusive)
    /// * `end_ms` - End time in milliseconds (inclusive)
    ///
    /// # Returns
    ///
    /// Vec of `AggTrade` with timestamps in microseconds (rangebar-core convention).
    pub async fn fetch_aggtrades_rest(
        &self,
        start_ms: i64,
        end_ms: i64,
    ) -> Result<Vec<AggTrade>, HistoricalError> {
        let url = "https://api.binance.com/api/v3/aggTrades";
        let mut all_trades: Vec<AggTrade> = Vec::new();
        let mut current_start = start_ms;

        debug!(
            event_type = "rest_fetch_start",
            symbol = %self.symbol,
            start_ms = start_ms,
            end_ms = end_ms,
            "Fetching aggTrades from REST API"
        );

        while current_start < end_ms {
            let client = &self.client;
            let symbol = &self.symbol;
            let current = current_start;
            let end = end_ms;

            let response = send_with_rate_limit_backoff(
                || {
                    client
                        .get(url)
                        .query(&[
                            ("symbol", symbol.as_str()),
                            ("startTime", &current.to_string()),
                            ("endTime", &end.to_string()),
                            ("limit", "1000"),
                        ])
                },
                symbol,
            )
            .await?;

            if !response.status().is_success() {
                return Err(HistoricalError::RestApiError(format!(
                    "HTTP {} for {} from {current_start}",
                    response.status(),
                    self.symbol,
                )));
            }

            let batch: Vec<RestAggTrade> = response
                .json()
                .await
                .map_err(|e| HistoricalError::RestApiError(format!(
                    "JSON parse error for {}: {e}",
                    self.symbol
                )))?;

            if batch.is_empty() {
                break;
            }

            let last_ts = batch.last().unwrap().trade_time;
            let batch_len = batch.len();

            all_trades.extend(batch.into_iter().map(AggTrade::from));

            // Advance past last trade timestamp to avoid duplicates
            if last_ts <= current_start {
                // Safety: prevent infinite loop if API returns same timestamp
                break;
            }
            current_start = last_ts + 1;

            // If we got fewer than limit, we've exhausted the range
            if batch_len < 1000 {
                break;
            }
        }

        all_trades.sort_by_key(|t| t.timestamp);

        info!(
            event_type = "rest_fetch_complete",
            symbol = %self.symbol,
            trade_count = all_trades.len(),
            range_ms = end_ms - start_ms,
            "Fetched aggTrades from REST API"
        );

        Ok(all_trades)
    }

    /// Fetch aggregated trades by `fromId` (zero-gap pagination).
    ///
    /// Single REST call using `fromId` parameter — caller controls cursor.
    /// This is the **correct** pagination method; `startTime` drops trades
    /// at millisecond boundaries.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Starting agg_trade_id (inclusive)
    /// * `limit` - Maximum trades to return (1-1000)
    ///
    /// # Returns
    ///
    /// Vec of `AggTrade` with timestamps in microseconds.
    pub async fn fetch_aggtrades_by_id(
        &self,
        from_id: i64,
        limit: u16,
    ) -> Result<Vec<AggTrade>, HistoricalError> {
        let url = "https://api.binance.com/api/v3/aggTrades";
        let limit_str = limit.min(1000).to_string();
        let from_id_str = from_id.to_string();
        let client = &self.client;
        let symbol = &self.symbol;

        let response = send_with_rate_limit_backoff(
            || {
                client
                    .get(url)
                    .query(&[
                        ("symbol", symbol.as_str()),
                        ("fromId", from_id_str.as_str()),
                        ("limit", limit_str.as_str()),
                    ])
            },
            symbol,
        )
        .await?;

        if !response.status().is_success() {
            return Err(HistoricalError::RestApiError(format!(
                "HTTP {} for {} fromId={from_id}",
                response.status(),
                self.symbol,
            )));
        }

        let batch: Vec<RestAggTrade> = response
            .json()
            .await
            .map_err(|e| HistoricalError::RestApiError(format!(
                "JSON parse error for {}: {e}",
                self.symbol
            )))?;

        Ok(batch.into_iter().map(AggTrade::from).collect())
    }

    /// Fetch the latest aggregated trade for the symbol.
    ///
    /// Single REST call with `limit=1`, no `fromId` or `startTime`.
    /// Returns the most recent trade as an anchor point for cursor-based pagination.
    pub async fn fetch_latest_aggtrade(&self) -> Result<AggTrade, HistoricalError> {
        let url = "https://api.binance.com/api/v3/aggTrades";
        let client = &self.client;
        let symbol = &self.symbol;

        let response = send_with_rate_limit_backoff(
            || {
                client
                    .get(url)
                    .query(&[
                        ("symbol", symbol.as_str()),
                        ("limit", "1"),
                    ])
            },
            symbol,
        )
        .await?;

        if !response.status().is_success() {
            return Err(HistoricalError::RestApiError(format!(
                "HTTP {} for {} (latest)",
                response.status(),
                self.symbol,
            )));
        }

        let batch: Vec<RestAggTrade> = response
            .json()
            .await
            .map_err(|e| HistoricalError::RestApiError(format!(
                "JSON parse error for {}: {e}",
                self.symbol
            )))?;

        batch
            .into_iter()
            .next()
            .map(AggTrade::from)
            .ok_or_else(|| HistoricalError::RestApiError(format!(
                "No trades returned for {} (latest)",
                self.symbol,
            )))
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

    // === Issue #96: detect_csv_headers + CsvAggTrade conversion tests ===

    #[test]
    fn test_detect_csv_headers_with_headers() {
        assert!(detect_csv_headers("agg_trade_id,price,quantity,first_trade_id\n1,50000,1.0,1"));
        assert!(detect_csv_headers("timestamp,is_buyer_maker\n123,true"));
    }

    #[test]
    fn test_detect_csv_headers_without_headers() {
        assert!(!detect_csv_headers("12345,50000.0,1.5,100,102,1640995200000,true,true"));
        assert!(!detect_csv_headers(""));
    }

    #[test]
    fn test_csv_agg_trade_to_agg_trade() {
        let csv = CsvAggTrade(12345, 50000.12345678, 1.5, 100, 102, 1640995200000, true);
        let trade: AggTrade = csv.into();
        assert_eq!(trade.agg_trade_id, 12345);
        assert_eq!(trade.first_trade_id, 100);
        assert_eq!(trade.last_trade_id, 102);
        assert_eq!(trade.timestamp, 1640995200000);
        assert!(trade.is_buyer_maker);
        assert!(trade.price.to_f64() > 49999.0);
        assert!(trade.volume.to_f64() > 1.0);
    }

    #[test]
    fn test_rest_agg_trade_inline_conversion() {
        let rest = RestAggTrade {
            agg_trade_id: 999, price: "50000.0".to_string(), quantity: "2.5".to_string(),
            first_trade_id: 10, last_trade_id: 12, trade_time: 1700000000000, is_buyer_maker: false,
        };
        // Mirrors the inline conversion pattern used in fetch_recent_trades()
        let trade = AggTrade {
            agg_trade_id: rest.agg_trade_id,
            price: FixedPoint::from_str(&rest.price).unwrap_or(FixedPoint(0)),
            volume: FixedPoint::from_str(&rest.quantity).unwrap_or(FixedPoint(0)),
            first_trade_id: rest.first_trade_id,
            last_trade_id: rest.last_trade_id,
            timestamp: rest.trade_time * 1000,
            is_buyer_maker: rest.is_buyer_maker,
            is_best_match: None,
        };
        assert_eq!(trade.agg_trade_id, 999);
        assert!(!trade.is_buyer_maker);
        assert_eq!(trade.timestamp, 1700000000000 * 1000);
        assert!(trade.price.to_f64() > 49999.0);
        assert!(trade.volume.to_f64() > 2.0);
    }

    // === Issue #96: CsvAggTrade.to_agg_trade() + market path tests ===

    #[test]
    fn test_csv_to_agg_trade_normalizes_ms_to_us() {
        let csv = CsvAggTrade(1, 50000.0, 1.0, 1, 1, 1640995200000, false);
        let trade = csv.to_agg_trade("spot");
        // normalize_timestamp converts 13-digit ms → µs
        assert_eq!(trade.timestamp, 1640995200000 * 1000);
        assert_eq!(trade.agg_trade_id, 1);
    }

    #[test]
    fn test_csv_from_vs_to_agg_trade_timestamp_differs() {
        let csv1 = CsvAggTrade(1, 50000.0, 1.0, 1, 1, 1640995200000, false);
        let csv2 = CsvAggTrade(1, 50000.0, 1.0, 1, 1, 1640995200000, false);
        let from_trade: AggTrade = csv1.into();
        let to_trade = csv2.to_agg_trade("spot");
        // From uses raw timestamp; to_agg_trade normalizes ms → µs
        assert_eq!(from_trade.timestamp, 1640995200000);
        assert_eq!(to_trade.timestamp, 1640995200000 * 1000);
        // Other fields identical
        assert_eq!(from_trade.agg_trade_id, to_trade.agg_trade_id);
        assert_eq!(from_trade.price, to_trade.price);
    }

    #[test]
    fn test_get_market_path_variants() {
        assert_eq!(HistoricalDataLoader::new("BTCUSDT").get_market_path(), "spot");
        assert_eq!(HistoricalDataLoader::new_with_market("BTCUSDT", "um").get_market_path(), "futures/um");
        assert_eq!(HistoricalDataLoader::new_with_market("BTCUSDT", "cm").get_market_path(), "futures/cm");
        assert_eq!(HistoricalDataLoader::new_with_market("BTCUSDT", "unknown").get_market_path(), "spot");
    }

    // === fromId pagination + 429 backoff tests ===

    #[test]
    fn test_rest_agg_trade_from_impl() {
        let rest = RestAggTrade {
            agg_trade_id: 42,
            price: "50000.12345678".to_string(),
            quantity: "1.5".to_string(),
            first_trade_id: 100,
            last_trade_id: 102,
            trade_time: 1700000000000,
            is_buyer_maker: true,
        };
        let trade: AggTrade = rest.into();
        assert_eq!(trade.agg_trade_id, 42);
        assert_eq!(trade.first_trade_id, 100);
        assert_eq!(trade.last_trade_id, 102);
        assert!(trade.is_buyer_maker);
        // From impl uses normalize_timestamp: ms → us
        assert_eq!(trade.timestamp, 1700000000000 * 1000);
        assert!(trade.price.to_f64() > 49999.0);
        assert!(trade.volume.to_f64() > 1.0);
    }

    #[test]
    fn test_rest_agg_trade_from_consistency_with_csv() {
        // REST and CSV should produce the same normalized timestamp
        let rest = RestAggTrade {
            agg_trade_id: 1,
            price: "50000.0".to_string(),
            quantity: "1.0".to_string(),
            first_trade_id: 1,
            last_trade_id: 1,
            trade_time: 1640995200000,
            is_buyer_maker: false,
        };
        let csv = CsvAggTrade(1, 50000.0, 1.0, 1, 1, 1640995200000, false);

        let rest_trade: AggTrade = rest.into();
        let csv_trade = csv.to_agg_trade("spot");

        // Both should normalize ms → us identically
        assert_eq!(rest_trade.timestamp, csv_trade.timestamp);
        assert_eq!(rest_trade.agg_trade_id, csv_trade.agg_trade_id);
        assert_eq!(rest_trade.is_buyer_maker, csv_trade.is_buyer_maker);
    }

    #[test]
    fn test_rate_limit_constants() {
        assert_eq!(RATE_LIMIT_MAX_RETRIES, 5);
        assert_eq!(RATE_LIMIT_INITIAL_BACKOFF_MS, 1_000);
        // Verify backoff schedule: 1s → 2s → 4s → 8s → 16s (capped)
        let mut backoff = RATE_LIMIT_INITIAL_BACKOFF_MS;
        let expected = [1_000, 2_000, 4_000, 8_000, 16_000];
        for &exp in &expected {
            assert_eq!(backoff, exp);
            backoff = (backoff * 2).min(16_000);
        }
    }
}
