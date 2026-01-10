//! Exness HTTP client and CSV fetcher
//!
//! Fetches monthly ZIP archives containing CSV tick data from Exness Raw_Spread variant.
//!
//! # Data Format
//!
//! Archive: ZIP compression (~10:1 ratio, ~9MB/month compressed)
//! Content: CSV with schema: "Exness","Symbol","Timestamp","Bid","Ask"
//! Timestamp: ISO 8601 UTC with millisecond precision
//!
//! # URL Pattern
//!
//! ```text
//! https://ticks.ex2archive.com/ticks/{SYMBOL}_Raw_Spread/{year}/{month}/Exness_{SYMBOL}_Raw_Spread_{year}_{month}.zip
//! ```
//!
//! Example:
//! ```text
//! https://ticks.ex2archive.com/ticks/EURUSD_Raw_Spread/2024/01/Exness_EURUSD_Raw_Spread_2024_01.zip
//! ```

use super::types::{ExnessError, ExnessInstrument, ExnessTick};
use chrono::{DateTime, NaiveDateTime};
use reqwest::Client;
use std::io::Read;
use std::time::Duration;
use zip::ZipArchive;

/// Exness HTTP data fetcher
///
/// Fetches monthly tick data for specified Raw_Spread symbol variant.
/// Zero rate limiting (100% reliability).
pub struct ExnessFetcher {
    client: Client,
    symbol: String,
}

impl ExnessFetcher {
    /// Create new fetcher for symbol
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol with Raw_Spread suffix (e.g., "EURUSD_Raw_Spread")
    ///
    /// # HTTP Configuration
    ///
    /// Timeout: 30s (Exness responds in 2-3 seconds typically)
    /// No retry logic needed (zero rate limiting observed)
    pub fn new(symbol: &str) -> Self {
        const EXNESS_REQUEST_TIMEOUT_SECS: u64 = 30;

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(EXNESS_REQUEST_TIMEOUT_SECS))
                .build()
                .expect("Failed to build Exness HTTP client"),
            symbol: symbol.to_string(),
        }
    }

    /// Create fetcher for a specific instrument (type-safe API)
    ///
    /// Preferred over `new()` for type safety and IDE autocomplete.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rangebar_providers::exness::{ExnessFetcher, ExnessInstrument};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let fetcher = ExnessFetcher::for_instrument(ExnessInstrument::XAUUSD);
    /// let ticks = fetcher.fetch_month(2024, 1).await?;
    /// println!("Fetched {} XAUUSD ticks", ticks.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn for_instrument(instrument: ExnessInstrument) -> Self {
        Self::new(&instrument.raw_spread_symbol())
    }

    /// Fetch monthly tick data
    ///
    /// # Arguments
    ///
    /// * `year` - Year (e.g., 2024)
    /// * `month` - Month 1-12
    ///
    /// # Returns
    ///
    /// Vector of ExnessTick sorted by timestamp (ascending)
    ///
    /// # Errors
    ///
    /// Raises immediately on any failure:
    /// - HTTP error (404, 503, timeout)
    /// - ZIP extraction error
    /// - CSV parsing error
    /// - Timestamp parsing error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rangebar_providers::exness::ExnessFetcher;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let fetcher = ExnessFetcher::new("EURUSD_Raw_Spread");
    /// let ticks = fetcher.fetch_month(2024, 1).await?;
    /// println!("Fetched {} ticks", ticks.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn fetch_month(&self, year: u16, month: u8) -> Result<Vec<ExnessTick>, ExnessError> {
        // Construct URL
        let url = format!(
            "https://ticks.ex2archive.com/ticks/{}/{:04}/{:02}/Exness_{}_{:04}_{:02}.zip",
            self.symbol, year, month, self.symbol, year, month
        );

        // 1. HTTP GET (fail-fast on any error)
        let response = self.client.get(&url).send().await?;
        let bytes = response.bytes().await?;

        // 2. Extract ZIP (fail-fast on any error)
        let reader = std::io::Cursor::new(bytes);
        let mut archive = ZipArchive::new(reader)?;

        // Exness ZIPs contain single CSV file
        if archive.len() != 1 {
            return Err(ExnessError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected 1 file in ZIP, found {}", archive.len()),
            )));
        }

        let mut csv_file = archive.by_index(0)?;

        // 3. Read CSV content
        let mut csv_content = String::new();
        csv_file.read_to_string(&mut csv_content)?;

        // 4. Parse CSV (fail-fast on any error)
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_content.as_bytes());

        let mut ticks = Vec::new();
        for result in reader.deserialize() {
            let record: ExnessCsvRecord = result?;
            ticks.push(ExnessTick::from_csv(record)?);
        }

        // Ticks should already be sorted by Exness, but verify
        // (fail-fast if temporal integrity violated)
        for i in 1..ticks.len() {
            if ticks[i].timestamp_ms < ticks[i - 1].timestamp_ms {
                return Err(ExnessError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Temporal integrity violation: tick {} timestamp {} < previous {}",
                        i,
                        ticks[i].timestamp_ms,
                        ticks[i - 1].timestamp_ms
                    ),
                )));
            }
        }

        Ok(ticks)
    }
}

/// CSV record structure (matches Exness format)
///
/// Exness CSV schema:
/// ```csv
/// "Exness","Symbol","Timestamp","Bid","Ask"
/// "exness","EURUSD_Raw_Spread","2024-01-15 00:00:00.032Z",1.0945,1.09456
/// ```
#[derive(serde::Deserialize, Debug)]
struct ExnessCsvRecord {
    #[serde(rename = "Exness")]
    _provider: String, // Ignore (always "exness")

    #[serde(rename = "Symbol")]
    _symbol: String, // Ignore (validated at fetcher level)

    #[serde(rename = "Timestamp")]
    timestamp: String, // ISO 8601 UTC

    #[serde(rename = "Bid")]
    bid: f64,

    #[serde(rename = "Ask")]
    ask: f64,
}

impl ExnessTick {
    /// Convert CSV record to ExnessTick
    ///
    /// # Timestamp Parsing
    ///
    /// Supports two ISO 8601 formats observed in Exness data:
    /// - `2024-01-15 00:00:00.032Z` (with trailing Z)
    /// - `2024-01-15 00:00:00.032` (without trailing Z)
    ///
    /// Both are treated as UTC.
    ///
    /// # Errors
    ///
    /// Raises immediately on timestamp parse failure (fail-fast).
    fn from_csv(record: ExnessCsvRecord) -> Result<Self, ExnessError> {
        // Parse timestamp: try with Z, fallback to without Z
        let timestamp_ms = if record.timestamp.ends_with('Z') {
            // Format: "2024-01-15 00:00:00.032Z"
            let dt = DateTime::parse_from_rfc3339(&record.timestamp)?;
            dt.timestamp_millis()
        } else {
            // Format: "2024-01-15 00:00:00.032" (assume UTC)
            let naive = NaiveDateTime::parse_from_str(&record.timestamp, "%Y-%m-%d %H:%M:%S%.f")?;
            DateTime::<chrono::Utc>::from_naive_utc_and_offset(naive, chrono::Utc)
                .timestamp_millis()
        };

        Ok(Self {
            bid: record.bid,
            ask: record.ask,
            timestamp_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exness_tick_from_csv_with_z() {
        let record = ExnessCsvRecord {
            _provider: "exness".to_string(),
            _symbol: "EURUSD_Raw_Spread".to_string(),
            timestamp: "2024-01-15T00:00:00.032Z".to_string(),
            bid: 1.0945,
            ask: 1.09456,
        };

        let tick = ExnessTick::from_csv(record).unwrap();
        assert_eq!(tick.bid, 1.0945);
        assert_eq!(tick.ask, 1.09456);
        assert!(tick.timestamp_ms > 0);
    }

    #[test]
    fn test_exness_tick_from_csv_without_z() {
        let record = ExnessCsvRecord {
            _provider: "exness".to_string(),
            _symbol: "EURUSD_Raw_Spread".to_string(),
            timestamp: "2024-01-15 00:00:00.032".to_string(),
            bid: 1.0945,
            ask: 1.09456,
        };

        let tick = ExnessTick::from_csv(record).unwrap();
        assert_eq!(tick.bid, 1.0945);
        assert_eq!(tick.ask, 1.09456);
        assert!(tick.timestamp_ms > 0);
    }

    #[test]
    fn test_exness_tick_invalid_timestamp() {
        let record = ExnessCsvRecord {
            _provider: "exness".to_string(),
            _symbol: "EURUSD_Raw_Spread".to_string(),
            timestamp: "invalid".to_string(),
            bid: 1.0945,
            ask: 1.09456,
        };

        // Should fail immediately (no fallback)
        assert!(ExnessTick::from_csv(record).is_err());
    }
}
