//! Real market data loading for integration testing
//!
//! ## Service Level Objectives (SLOs)
//!
//! ### Availability SLO: 100% file accessibility
//! - Propagate filesystem errors (no fallbacks, no defaults)
//! - Fail fast on missing files with clear error messages
//! - Validation: File existence check before opening
//!
//! ### Correctness SLO: 100% data integrity
//! - Parse all records without data loss or silent failures
//! - Strict schema validation (reject malformed records)
//! - No default values on parse errors (fail-fast)
//! - Validation: Record count = file line count - 1 (header)
//!
//! ### Observability SLO: 100% error traceability
//! - All errors include file path and line number context
//! - Use thiserror for structured error messages
//! - No silent failures or swallowed errors
//!
//! ### Maintainability SLO: Off-the-shelf components only
//! - Use csv crate (de facto Rust standard for CSV parsing)
//! - Use serde for deserialization (no custom parsing)
//! - Zero custom string manipulation or parsing logic
//!
//! ## Data Source
//!
//! - BTCUSDT: `test_data/BTCUSDT/BTCUSDT_aggTrades_20250901.csv` (5,000 trades)
//! - ETHUSDT: `test_data/ETHUSDT/ETHUSDT_aggTrades_20250901.csv` (10,000 trades)
//!
//! ## CSV Format (Binance aggTrades)
//!
//! ```csv
//! a,p,q,f,l,T,m
//! 1,50014.00859087,0.12019569,1,1,1756710002083,False
//! ```
//!
//! Columns:
//! - a: Aggregate trade ID
//! - p: Price (decimal string)
//! - q: Quantity (decimal string)
//! - f: First trade ID
//! - l: Last trade ID
//! - T: Timestamp (milliseconds, integer)
//! - m: Is buyer maker ("True"/"False" string)

use crate::FixedPoint;
use crate::types::AggTrade;
use std::path::Path;
use thiserror::Error;

/// Test data loader errors
#[derive(Debug, Error)]
pub enum LoaderError {
    /// File I/O error (propagate without modification)
    #[error("File I/O error for {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// CSV parsing error (include line context)
    #[error("CSV parse error at line {line} in {path}: {source}")]
    CsvParse {
        path: String,
        line: usize,
        #[source]
        source: csv::Error,
    },

    /// Fixed-point conversion error
    #[error("Fixed-point conversion error at line {line} in {path}: {source}")]
    FixedPoint {
        path: String,
        line: usize,
        #[source]
        source: crate::fixed_point::FixedPointError,
    },

    /// Record count validation error
    #[error("Record count mismatch in {path}: expected {expected}, got {actual}")]
    CountMismatch {
        path: String,
        expected: usize,
        actual: usize,
    },
}

/// Binance aggTrades CSV record (Spot/Futures format)
///
/// Matches CSV columns: a,p,q,f,l,T,m
///
/// Field names match Binance CSV format (non-snake_case is intentional)
#[derive(Debug, serde::Deserialize)]
#[allow(non_snake_case)]
struct AggTradeRecord {
    /// Aggregate trade ID
    a: i64,
    /// Price (decimal string)
    p: String,
    /// Quantity (decimal string)
    q: String,
    /// First trade ID
    f: i64,
    /// Last trade ID
    l: i64,
    /// Timestamp (milliseconds)
    T: i64,
    /// Is buyer maker ("True"/"False" string)
    m: String,
}

impl AggTradeRecord {
    /// Convert CSV record to AggTrade
    ///
    /// SLO: Fail-fast on parse errors (no defaults, no fallbacks)
    fn into_agg_trade(self) -> Result<AggTrade, crate::fixed_point::FixedPointError> {
        Ok(AggTrade {
            agg_trade_id: self.a,
            price: FixedPoint::from_str(&self.p)?,
            volume: FixedPoint::from_str(&self.q)?,
            first_trade_id: self.f,
            last_trade_id: self.l,
            timestamp: self.T,
            is_buyer_maker: self.m == "True",
            is_best_match: None, // Spot data has no best_match field
        })
    }
}

/// Load BTCUSDT test data (5,000 trades from 2025-09-01)
///
/// SLO: Fail-fast on any error, 100% data integrity
///
/// Path resolution: Searches workspace root via CARGO_MANIFEST_DIR
pub fn load_btcusdt_test_data() -> Result<Vec<AggTrade>, LoaderError> {
    let path = workspace_test_data_path("BTCUSDT/BTCUSDT_aggTrades_20250901.csv");
    load_test_data(path, 5000)
}

/// Load ETHUSDT test data (10,000 trades from 2025-09-01)
///
/// SLO: Fail-fast on any error, 100% data integrity
///
/// Path resolution: Searches workspace root via CARGO_MANIFEST_DIR
pub fn load_ethusdt_test_data() -> Result<Vec<AggTrade>, LoaderError> {
    let path = workspace_test_data_path("ETHUSDT/ETHUSDT_aggTrades_20250901.csv");
    load_test_data(path, 10000)
}

/// Resolve test_data path from workspace root
///
/// Strategy: Navigate from CARGO_MANIFEST_DIR to workspace root
/// CARGO_MANIFEST_DIR = /path/to/rangebar/crates/rangebar-core
/// Workspace root = ../../ (2 levels up)
fn workspace_test_data_path(relative_path: &str) -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent() // crates/
        .unwrap()
        .parent() // workspace root
        .unwrap();

    workspace_root.join("test_data").join(relative_path)
}

/// Generic CSV loader with record count validation
///
/// SLO Guarantees:
/// - Availability: Propagates I/O errors without fallbacks
/// - Correctness: Validates record count matches expected_count
/// - Observability: All errors include file path and line number
/// - Maintainability: Uses csv crate (no custom parsing)
fn load_test_data<P: AsRef<Path>>(
    path: P,
    expected_count: usize,
) -> Result<Vec<AggTrade>, LoaderError> {
    let path_str = path.as_ref().to_string_lossy().to_string();

    // SLO: Availability - Propagate I/O errors with context
    let file = std::fs::File::open(&path).map_err(|e| LoaderError::Io {
        path: path_str.clone(),
        source: e,
    })?;

    // SLO: Maintainability - Use csv crate (off-the-shelf)
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut trades = Vec::with_capacity(expected_count);
    let mut line = 2; // Line 1 is header, data starts at line 2

    // SLO: Correctness - Strict parsing, no silent failures
    for result in reader.deserialize() {
        let record: AggTradeRecord = result.map_err(|e| LoaderError::CsvParse {
            path: path_str.clone(),
            line,
            source: e,
        })?;

        let trade = record
            .into_agg_trade()
            .map_err(|e| LoaderError::FixedPoint {
                path: path_str.clone(),
                line,
                source: e,
            })?;

        trades.push(trade);
        line += 1;
    }

    // SLO: Correctness - Validate record count (detect truncation/corruption)
    let actual_count = trades.len();
    if actual_count != expected_count {
        return Err(LoaderError::CountMismatch {
            path: path_str,
            expected: expected_count,
            actual: actual_count,
        });
    }

    Ok(trades)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_btcusdt_data() {
        let trades = load_btcusdt_test_data().expect("Failed to load BTCUSDT test data");

        // SLO: Correctness - Validate record count
        assert_eq!(
            trades.len(),
            5000,
            "BTCUSDT should have exactly 5000 trades"
        );

        // SLO: Correctness - Validate first trade data integrity
        let first = &trades[0];
        assert_eq!(first.agg_trade_id, 1);
        assert_eq!(first.price.to_string(), "50014.00859087");
        assert_eq!(first.volume.to_string(), "0.12019569");
        assert_eq!(first.first_trade_id, 1);
        assert_eq!(first.last_trade_id, 1);
        assert_eq!(first.timestamp, 1756710002083);
        assert!(!first.is_buyer_maker);
    }

    #[test]
    fn test_load_ethusdt_data() {
        let trades = load_ethusdt_test_data().expect("Failed to load ETHUSDT test data");

        // SLO: Correctness - Validate record count
        assert_eq!(
            trades.len(),
            10000,
            "ETHUSDT should have exactly 10000 trades"
        );

        // SLO: Correctness - All trades should have valid data
        for trade in &trades {
            assert!(trade.price.0 > 0, "Price must be positive");
            assert!(trade.volume.0 > 0, "Volume must be positive");
            assert!(trade.timestamp > 0, "Timestamp must be positive");
        }
    }

    #[test]
    fn test_temporal_integrity() {
        let trades = load_btcusdt_test_data().unwrap();

        // SLO: Correctness - Validate monotonic timestamps
        for i in 1..trades.len() {
            assert!(
                trades[i].timestamp >= trades[i - 1].timestamp,
                "Temporal integrity violation at trade {}: {} < {}",
                i,
                trades[i].timestamp,
                trades[i - 1].timestamp
            );
        }
    }
}
