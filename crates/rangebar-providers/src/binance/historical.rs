//! Historical data loading utilities
//!
//! Consolidated data loading functionality extracted from examples to eliminate
//! code duplication. Provides unified interface for Binance aggTrades data
//! with support for single-day, multi-day, and recent data loading.

use chrono::NaiveDate;
use csv::ReaderBuilder;
use reqwest::Client;
use serde::Deserialize;
use std::io::{Cursor, Read};
use std::time::Duration;
use zip::ZipArchive;

use rangebar_core::{normalize_timestamp, AggTrade, FixedPoint};

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
}
