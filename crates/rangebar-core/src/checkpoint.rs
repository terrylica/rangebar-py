//! Checkpoint system for cross-file range bar continuation
//!
//! Enables seamless processing across file boundaries by serializing
//! incomplete bar state with IMMUTABLE thresholds.
//!
//! ## Primary Use Case
//!
//! ```text
//! File 1 ends with incomplete bar → save Checkpoint
//! File 2 starts → load Checkpoint → continue building bar
//! ```
//!
//! ## Key Invariants
//!
//! - Thresholds are computed from bar.open and are IMMUTABLE for bar's lifetime
//! - Incomplete bar state preserved across file boundaries
//! - Note: `bar[i+1].open` may differ from `bar[i].close` (next bar opens at first
//!   tick after previous bar closes, not at the close price itself)
//! - Works with both Binance (has agg_trade_id) and Exness (timestamp-only)

use crate::fixed_point::FixedPoint;
use crate::types::RangeBar;
use ahash::AHasher;
use serde::{Deserialize, Serialize};
use std::hash::Hasher;
use thiserror::Error;

/// Price window size for hash calculation (last N prices)
const PRICE_WINDOW_SIZE: usize = 8;

/// Checkpoint for cross-file range bar continuation
///
/// Enables seamless processing across any file boundaries (Binance daily, Exness monthly).
/// Captures minimal state needed to continue building an incomplete bar.
///
/// # Example
///
/// ```ignore
/// // Process first file
/// let bars_1 = processor.process_agg_trade_records(&file1_trades)?;
/// let checkpoint = processor.create_checkpoint("BTCUSDT");
///
/// // Serialize and save checkpoint
/// let json = serde_json::to_string(&checkpoint)?;
/// std::fs::write("checkpoint.json", json)?;
///
/// // ... later, load checkpoint and continue processing ...
/// let json = std::fs::read_to_string("checkpoint.json")?;
/// let checkpoint: Checkpoint = serde_json::from_str(&json)?;
/// let mut processor = RangeBarProcessor::from_checkpoint(checkpoint)?;
/// let bars_2 = processor.process_agg_trade_records(&file2_trades)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    // === IDENTIFICATION (2 fields) ===
    /// Symbol being processed (e.g., "BTCUSDT", "EURUSD")
    pub symbol: String,

    /// Threshold in decimal basis points (v3.0.0+: 0.1bps units)
    /// Example: 250 = 25bps = 0.25%
    pub threshold_decimal_bps: u32,

    // === BAR STATE (2 fields) ===
    /// Incomplete bar at file boundary (None = last bar completed cleanly)
    /// REUSES existing RangeBar type - no separate BarState needed!
    pub incomplete_bar: Option<RangeBar>,

    /// Fixed thresholds for incomplete bar (computed from bar.open, IMMUTABLE)
    /// Stored as (upper_threshold, lower_threshold)
    pub thresholds: Option<(FixedPoint, FixedPoint)>,

    // === POSITION TRACKING (2 fields) ===
    /// Last processed timestamp in microseconds (universal, works for all sources)
    pub last_timestamp_us: i64,

    /// Last trade ID (Some for Binance, None for Exness)
    /// Binance: agg_trade_id is strictly sequential, never resets
    pub last_trade_id: Option<i64>,

    // === INTEGRITY (1 field) ===
    /// Price window hash (ahash of last 8 prices for position verification)
    /// Used to verify we're resuming at the correct position in data stream
    pub price_hash: u64,

    // === MONITORING (1 field) ===
    /// Anomaly summary counts for debugging
    pub anomaly_summary: AnomalySummary,

    // === BEHAVIOR FLAGS (2 fields) ===
    /// Prevent bars from closing on same timestamp as they opened (Issue #36)
    ///
    /// When true (default): A bar cannot close until a trade arrives with a
    /// different timestamp than the bar's open_time. This prevents flash crash
    /// scenarios from creating thousands of bars at identical timestamps.
    ///
    /// When false: Legacy v8 behavior - bars can close immediately on breach.
    #[serde(default = "default_prevent_same_timestamp_close")]
    pub prevent_same_timestamp_close: bool,

    /// Deferred bar open flag (Issue #46)
    ///
    /// When true: The last trade before checkpoint triggered a threshold breach.
    /// On resume, the next trade should open a new bar instead of continuing.
    /// This matches the batch path's `defer_open` semantics.
    #[serde(default)]
    pub defer_open: bool,
}

/// Default value for prevent_same_timestamp_close (true = timestamp gating enabled)
fn default_prevent_same_timestamp_close() -> bool {
    true
}

impl Checkpoint {
    /// Create a new checkpoint with the given parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: String,
        threshold_decimal_bps: u32,
        incomplete_bar: Option<RangeBar>,
        thresholds: Option<(FixedPoint, FixedPoint)>,
        last_timestamp_us: i64,
        last_trade_id: Option<i64>,
        price_hash: u64,
        prevent_same_timestamp_close: bool,
    ) -> Self {
        Self {
            symbol,
            threshold_decimal_bps,
            incomplete_bar,
            thresholds,
            last_timestamp_us,
            last_trade_id,
            price_hash,
            anomaly_summary: AnomalySummary::default(),
            prevent_same_timestamp_close,
            defer_open: false,
        }
    }

    /// Check if there's an incomplete bar that needs to continue
    pub fn has_incomplete_bar(&self) -> bool {
        self.incomplete_bar.is_some()
    }

    /// Get the library version that created this checkpoint
    pub fn library_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

/// Anomaly summary for quick inspection (counts only)
///
/// Tracks anomalies detected during processing for debugging purposes.
/// Does NOT affect processing - purely for monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AnomalySummary {
    /// Number of gaps detected (missing trade IDs or timestamp jumps)
    pub gaps_detected: u32,

    /// Number of overlaps detected (duplicate or out-of-order data)
    pub overlaps_detected: u32,

    /// Number of timestamp anomalies (negative intervals, etc.)
    pub timestamp_anomalies: u32,
}

impl AnomalySummary {
    /// Increment gap counter
    pub fn record_gap(&mut self) {
        self.gaps_detected += 1;
    }

    /// Increment overlap counter
    pub fn record_overlap(&mut self) {
        self.overlaps_detected += 1;
    }

    /// Increment timestamp anomaly counter
    pub fn record_timestamp_anomaly(&mut self) {
        self.timestamp_anomalies += 1;
    }

    /// Check if any anomalies were detected
    pub fn has_anomalies(&self) -> bool {
        self.gaps_detected > 0 || self.overlaps_detected > 0 || self.timestamp_anomalies > 0
    }

    /// Get total anomaly count
    pub fn total(&self) -> u32 {
        self.gaps_detected + self.overlaps_detected + self.timestamp_anomalies
    }
}

/// Position verification result when resuming from checkpoint
#[derive(Debug, Clone, PartialEq)]
pub enum PositionVerification {
    /// Trade ID matches expected (Binance: last_id + 1)
    Exact,

    /// Trade ID gap detected (Binance only)
    /// Contains expected_id, actual_id, and count of missing trades
    Gap {
        expected_id: i64,
        actual_id: i64,
        missing_count: i64,
    },

    /// No trade ID available, timestamp check only (Exness)
    /// Contains gap in milliseconds since last checkpoint
    TimestampOnly { gap_ms: i64 },
}

impl PositionVerification {
    /// Check if position verification indicates a data gap
    pub fn has_gap(&self) -> bool {
        matches!(self, PositionVerification::Gap { .. })
    }
}

/// Checkpoint-related errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CheckpointError {
    /// Symbol mismatch between checkpoint and processor
    #[error("Symbol mismatch: checkpoint has '{checkpoint}', expected '{expected}'")]
    SymbolMismatch {
        checkpoint: String,
        expected: String,
    },

    /// Threshold mismatch between checkpoint and processor
    #[error("Threshold mismatch: checkpoint has {checkpoint} dbps, expected {expected} dbps")]
    ThresholdMismatch { checkpoint: u32, expected: u32 },

    /// Price hash mismatch indicates wrong position in data stream
    #[error("Price hash mismatch: checkpoint has {checkpoint}, computed {computed}")]
    PriceHashMismatch { checkpoint: u64, computed: u64 },

    /// Checkpoint has incomplete bar but no thresholds
    #[error("Checkpoint has incomplete bar but missing thresholds - corrupted checkpoint")]
    MissingThresholds,

    /// Checkpoint serialization/deserialization error
    #[error("Checkpoint serialization error: {message}")]
    SerializationError { message: String },
}

/// Price window for computing position verification hash
///
/// Maintains a circular buffer of the last N prices for hash computation.
#[derive(Debug, Clone)]
pub struct PriceWindow {
    prices: [i64; PRICE_WINDOW_SIZE],
    index: usize,
    count: usize,
}

impl Default for PriceWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl PriceWindow {
    /// Create a new empty price window
    pub fn new() -> Self {
        Self {
            prices: [0; PRICE_WINDOW_SIZE],
            index: 0,
            count: 0,
        }
    }

    /// Add a price to the window (circular buffer)
    pub fn push(&mut self, price: FixedPoint) {
        self.prices[self.index] = price.0;
        self.index = (self.index + 1) % PRICE_WINDOW_SIZE;
        if self.count < PRICE_WINDOW_SIZE {
            self.count += 1;
        }
    }

    /// Compute hash of the price window using ahash
    ///
    /// Returns a 64-bit hash that can be used to verify position in data stream.
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = AHasher::default();

        // Hash prices in order they were added (oldest to newest)
        if self.count < PRICE_WINDOW_SIZE {
            // Buffer not full yet - hash from start
            for i in 0..self.count {
                hasher.write_i64(self.prices[i]);
            }
        } else {
            // Buffer full - hash from current index (oldest) around
            for i in 0..PRICE_WINDOW_SIZE {
                let idx = (self.index + i) % PRICE_WINDOW_SIZE;
                hasher.write_i64(self.prices[idx]);
            }
        }

        hasher.finish()
    }

    /// Get the number of prices in the window
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the window is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new(
            "BTCUSDT".to_string(),
            250, // 25bps
            None,
            None,
            1640995200000000, // timestamp in microseconds
            Some(12345),
            0,
            true, // prevent_same_timestamp_close
        );

        assert_eq!(checkpoint.symbol, "BTCUSDT");
        assert_eq!(checkpoint.threshold_decimal_bps, 250);
        assert!(!checkpoint.has_incomplete_bar());
        assert_eq!(checkpoint.last_trade_id, Some(12345));
        assert!(checkpoint.prevent_same_timestamp_close);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = Checkpoint::new(
            "EURUSD".to_string(),
            10, // 1bps
            None,
            None,
            1640995200000000,
            None, // Exness has no trade IDs
            12345678,
            true, // prevent_same_timestamp_close
        );

        // Serialize to JSON
        let json = serde_json::to_string(&checkpoint).unwrap();
        assert!(json.contains("EURUSD"));
        assert!(json.contains("\"threshold_decimal_bps\":10"));
        assert!(json.contains("\"prevent_same_timestamp_close\":true"));

        // Deserialize back
        let restored: Checkpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.symbol, "EURUSD");
        assert_eq!(restored.threshold_decimal_bps, 10);
        assert_eq!(restored.price_hash, 12345678);
        assert!(restored.prevent_same_timestamp_close);
    }

    #[test]
    fn test_checkpoint_serialization_toggle_false() {
        let checkpoint = Checkpoint::new(
            "BTCUSDT".to_string(),
            100, // 10bps
            None,
            None,
            1640995200000000,
            Some(999),
            12345678,
            false, // Legacy behavior
        );

        // Serialize to JSON
        let json = serde_json::to_string(&checkpoint).unwrap();
        assert!(json.contains("\"prevent_same_timestamp_close\":false"));

        // Deserialize back
        let restored: Checkpoint = serde_json::from_str(&json).unwrap();
        assert!(!restored.prevent_same_timestamp_close);
    }

    #[test]
    fn test_checkpoint_deserialization_default() {
        // Test that old checkpoints without the field default to true
        let json = r#"{
            "symbol": "BTCUSDT",
            "threshold_decimal_bps": 100,
            "incomplete_bar": null,
            "thresholds": null,
            "last_timestamp_us": 1640995200000000,
            "last_trade_id": 12345,
            "price_hash": 0,
            "anomaly_summary": {"gaps_detected": 0, "overlaps_detected": 0, "timestamp_anomalies": 0}
        }"#;

        let checkpoint: Checkpoint = serde_json::from_str(json).unwrap();
        // Missing field should default to true (new behavior)
        assert!(checkpoint.prevent_same_timestamp_close);
    }

    #[test]
    fn test_anomaly_summary() {
        let mut summary = AnomalySummary::default();
        assert!(!summary.has_anomalies());
        assert_eq!(summary.total(), 0);

        summary.record_gap();
        summary.record_gap();
        summary.record_timestamp_anomaly();

        assert!(summary.has_anomalies());
        assert_eq!(summary.gaps_detected, 2);
        assert_eq!(summary.timestamp_anomalies, 1);
        assert_eq!(summary.total(), 3);
    }

    #[test]
    fn test_price_window() {
        let mut window = PriceWindow::new();
        assert!(window.is_empty());

        // Add some prices
        window.push(FixedPoint(5000000000000)); // 50000.0
        window.push(FixedPoint(5001000000000)); // 50010.0
        window.push(FixedPoint(5002000000000)); // 50020.0

        assert_eq!(window.len(), 3);
        assert!(!window.is_empty());

        let hash1 = window.compute_hash();

        // Same prices should produce same hash
        let mut window2 = PriceWindow::new();
        window2.push(FixedPoint(5000000000000));
        window2.push(FixedPoint(5001000000000));
        window2.push(FixedPoint(5002000000000));

        let hash2 = window2.compute_hash();
        assert_eq!(hash1, hash2);

        // Different prices should produce different hash
        let mut window3 = PriceWindow::new();
        window3.push(FixedPoint(5000000000000));
        window3.push(FixedPoint(5001000000000));
        window3.push(FixedPoint(5003000000000)); // Different!

        let hash3 = window3.compute_hash();
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_price_window_circular() {
        let mut window = PriceWindow::new();

        // Fill the window beyond capacity
        for i in 0..12 {
            window.push(FixedPoint(i * 100000000));
        }

        // Should only contain last 8 prices
        assert_eq!(window.len(), PRICE_WINDOW_SIZE);

        // Hash should be consistent
        let hash1 = window.compute_hash();
        let hash2 = window.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_position_verification() {
        let exact = PositionVerification::Exact;
        assert!(!exact.has_gap());

        let gap = PositionVerification::Gap {
            expected_id: 100,
            actual_id: 105,
            missing_count: 5,
        };
        assert!(gap.has_gap());

        let timestamp_only = PositionVerification::TimestampOnly { gap_ms: 1000 };
        assert!(!timestamp_only.has_gap());
    }

    #[test]
    fn test_library_version() {
        let version = Checkpoint::library_version();
        // Should be a valid semver string
        assert!(version.contains('.'));
        println!("Library version: {}", version);
    }
}
