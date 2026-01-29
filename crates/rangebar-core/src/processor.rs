//! Core range bar processing algorithm
//!
//! Implements non-lookahead bias range bar construction where bars close when
//! price moves ±threshold dbps from the bar's OPEN price.

use crate::checkpoint::{
    AnomalySummary, Checkpoint, CheckpointError, PositionVerification, PriceWindow,
};
use crate::fixed_point::FixedPoint;
use crate::types::{AggTrade, RangeBar};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

/// Range bar processor with non-lookahead bias guarantee
pub struct RangeBarProcessor {
    /// Threshold in decimal basis points (250 = 25bps, v3.0.0+)
    threshold_decimal_bps: u32,

    /// Current bar state for streaming processing (Q19)
    /// Enables get_incomplete_bar() and stateful process_single_trade()
    current_bar_state: Option<RangeBarState>,

    /// Price window for checkpoint hash verification
    price_window: PriceWindow,

    /// Last processed trade ID (for gap detection on resume)
    last_trade_id: Option<i64>,

    /// Last processed timestamp (for position verification)
    last_timestamp_us: i64,

    /// Anomaly tracking for debugging
    anomaly_summary: AnomalySummary,

    /// Flag indicating this processor was created from a checkpoint
    /// When true, process_agg_trade_records will continue from existing bar state
    resumed_from_checkpoint: bool,

    /// Prevent bars from closing on same timestamp as they opened (Issue #36)
    ///
    /// When true (default): A bar cannot close until a trade arrives with a
    /// different timestamp than the bar's open_time. This prevents "instant bars"
    /// during flash crashes where multiple trades occur at the same millisecond.
    ///
    /// When false: Legacy behavior - bars can close on any breach regardless
    /// of timestamp, which may produce bars with identical timestamps.
    prevent_same_timestamp_close: bool,

    /// Deferred bar open flag (Issue #46)
    ///
    /// When true: The previous trade triggered a threshold breach and closed a bar.
    /// The next trade arriving via `process_single_trade()` should open a new bar
    /// instead of being treated as a continuation.
    ///
    /// This matches the batch path's `defer_open` semantics in
    /// `process_agg_trade_records()` where the breaching trade closes the current
    /// bar and the NEXT trade opens the new bar.
    defer_open: bool,
}

impl RangeBarProcessor {
    /// Create new processor with given threshold
    ///
    /// Uses default behavior: `prevent_same_timestamp_close = true` (Issue #36)
    ///
    /// # Arguments
    ///
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    ///   - Example: `250` → 25bps = 0.25%
    ///   - Example: `10` → 1bps = 0.01%
    ///   - Minimum: `1` → 0.1bps = 0.001%
    ///
    /// # Breaking Change (v3.0.0)
    ///
    /// Prior to v3.0.0, `threshold_decimal_bps` was in 1bps units.
    /// **Migration**: Multiply all threshold values by 10.
    pub fn new(threshold_decimal_bps: u32) -> Result<Self, ProcessingError> {
        Self::with_options(threshold_decimal_bps, true)
    }

    /// Create new processor with explicit timestamp gating control
    ///
    /// # Arguments
    ///
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    /// * `prevent_same_timestamp_close` - If true, bars cannot close until
    ///   timestamp advances from open_time. This prevents "instant bars" during
    ///   flash crashes. Set to false for legacy behavior (pre-v9).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Default behavior (v9+): timestamp gating enabled
    /// let processor = RangeBarProcessor::new(250)?;
    ///
    /// // Legacy behavior: allow instant bars
    /// let processor = RangeBarProcessor::with_options(250, false)?;
    /// ```
    pub fn with_options(
        threshold_decimal_bps: u32,
        prevent_same_timestamp_close: bool,
    ) -> Result<Self, ProcessingError> {
        // Validation bounds (v3.0.0: dbps units)
        // Min: 1 dbps = 0.001%
        // Max: 100,000 dbps = 100%
        if threshold_decimal_bps < 1 {
            return Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            });
        }
        if threshold_decimal_bps > 100_000 {
            return Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            });
        }

        Ok(Self {
            threshold_decimal_bps,
            current_bar_state: None,
            price_window: PriceWindow::new(),
            last_trade_id: None,
            last_timestamp_us: 0,
            anomaly_summary: AnomalySummary::default(),
            resumed_from_checkpoint: false,
            prevent_same_timestamp_close,
            defer_open: false,
        })
    }

    /// Get the prevent_same_timestamp_close setting
    pub fn prevent_same_timestamp_close(&self) -> bool {
        self.prevent_same_timestamp_close
    }

    /// Process a single trade and return completed bar if any
    ///
    /// Maintains internal state for streaming use case. State persists across calls
    /// until a bar completes (threshold breach), enabling get_incomplete_bar().
    ///
    /// # Arguments
    ///
    /// * `trade` - Single aggregated trade to process
    ///
    /// # Returns
    ///
    /// `Some(RangeBar)` if a bar was completed, `None` otherwise
    ///
    /// # State Management
    ///
    /// - First trade: Initializes new bar state
    /// - Subsequent trades: Updates existing bar or closes on breach
    /// - Breach: Returns completed bar, starts new bar with breaching trade
    pub fn process_single_trade(
        &mut self,
        trade: AggTrade,
    ) -> Result<Option<RangeBar>, ProcessingError> {
        // Track price and position for checkpoint
        self.price_window.push(trade.price);
        self.last_trade_id = Some(trade.agg_trade_id);
        self.last_timestamp_us = trade.timestamp;

        // Issue #46: If previous call triggered a breach, this trade opens the new bar.
        // This matches the batch path's defer_open semantics - the breaching trade
        // closes the current bar, and the NEXT trade opens the new bar.
        if self.defer_open {
            self.current_bar_state =
                Some(RangeBarState::new(&trade, self.threshold_decimal_bps));
            self.defer_open = false;
            return Ok(None);
        }

        match &mut self.current_bar_state {
            None => {
                // First trade - initialize new bar
                self.current_bar_state =
                    Some(RangeBarState::new(&trade, self.threshold_decimal_bps));
                Ok(None)
            }
            Some(bar_state) => {
                // Check for threshold breach
                let price_breaches = bar_state.bar.is_breach(
                    trade.price,
                    bar_state.upper_threshold,
                    bar_state.lower_threshold,
                );

                // Timestamp gate (Issue #36): prevent bars from closing on same timestamp
                // This eliminates "instant bars" during flash crashes where multiple trades
                // occur at the same millisecond.
                let timestamp_allows_close = !self.prevent_same_timestamp_close
                    || trade.timestamp != bar_state.bar.open_time;

                if price_breaches && timestamp_allows_close {
                    // Breach detected AND timestamp changed - close current bar
                    bar_state.bar.update_with_trade(&trade);

                    // Validation: Ensure high/low include open/close extremes
                    debug_assert!(
                        bar_state.bar.high >= bar_state.bar.open.max(bar_state.bar.close)
                    );
                    debug_assert!(bar_state.bar.low <= bar_state.bar.open.min(bar_state.bar.close));

                    // Compute microstructure features at bar finalization (Issue #25)
                    bar_state.bar.compute_microstructure_features();
                    let completed_bar = bar_state.bar.clone();

                    // Issue #46: Don't start new bar with breaching trade.
                    // Next trade will open the new bar via defer_open.
                    self.current_bar_state = None;
                    self.defer_open = true;

                    Ok(Some(completed_bar))
                } else {
                    // Either no breach OR same timestamp (gate active) - update existing bar
                    bar_state.bar.update_with_trade(&trade);
                    Ok(None)
                }
            }
        }
    }

    /// Get any incomplete bar currently being processed
    ///
    /// Returns clone of current bar state for inspection without consuming it.
    /// Useful for final bar at stream end or progress monitoring.
    ///
    /// # Returns
    ///
    /// `Some(RangeBar)` if bar is in progress, `None` if no active bar
    pub fn get_incomplete_bar(&self) -> Option<RangeBar> {
        self.current_bar_state
            .as_ref()
            .map(|state| state.bar.clone())
    }

    /// Process AggTrade records into range bars including incomplete bars for analysis
    ///
    /// # Arguments
    ///
    /// * `agg_trade_records` - Slice of AggTrade records sorted by (timestamp, agg_trade_id)
    ///
    /// # Returns
    ///
    /// Vector of range bars including incomplete bars at end of data
    ///
    /// # Warning
    ///
    /// This method is for analysis purposes only. Incomplete bars violate the
    /// fundamental range bar algorithm and should not be used for production trading.
    pub fn process_agg_trade_records_with_incomplete(
        &mut self,
        agg_trade_records: &[AggTrade],
    ) -> Result<Vec<RangeBar>, ProcessingError> {
        self.process_agg_trade_records_with_options(agg_trade_records, true)
    }

    /// Process Binance aggregated trade records into range bars
    ///
    /// This is the primary method for converting AggTrade records (which aggregate
    /// multiple individual trades) into range bars based on price movement thresholds.
    ///
    /// # Parameters
    ///
    /// * `agg_trade_records` - Slice of AggTrade records sorted by (timestamp, agg_trade_id)
    ///   Each record represents multiple individual trades aggregated at same price
    ///
    /// # Returns
    ///
    /// Vector of completed range bars (ONLY bars that breached thresholds).
    /// Each bar tracks both individual trade count and AggTrade record count.
    pub fn process_agg_trade_records(
        &mut self,
        agg_trade_records: &[AggTrade],
    ) -> Result<Vec<RangeBar>, ProcessingError> {
        self.process_agg_trade_records_with_options(agg_trade_records, false)
    }

    /// Process AggTrade records with options for including incomplete bars
    ///
    /// Batch processing mode: Clears any existing state before processing.
    /// Use process_single_trade() for stateful streaming instead.
    ///
    /// # Parameters
    ///
    /// * `agg_trade_records` - Slice of AggTrade records sorted by (timestamp, agg_trade_id)
    /// * `include_incomplete` - Whether to include incomplete bars at end of processing
    ///
    /// # Returns
    ///
    /// Vector of range bars (completed + incomplete if requested)
    pub fn process_agg_trade_records_with_options(
        &mut self,
        agg_trade_records: &[AggTrade],
        include_incomplete: bool,
    ) -> Result<Vec<RangeBar>, ProcessingError> {
        if agg_trade_records.is_empty() {
            return Ok(Vec::new());
        }

        // Validate records are sorted
        self.validate_trade_ordering(agg_trade_records)?;

        // Use existing bar state if resuming from checkpoint, otherwise start fresh
        // This is CRITICAL for cross-file continuation (Issues #2, #3)
        let mut current_bar: Option<RangeBarState> = if self.resumed_from_checkpoint {
            // Continue from checkpoint's incomplete bar
            self.resumed_from_checkpoint = false; // Consume the flag
            self.current_bar_state.take()
        } else {
            // Start fresh for normal batch processing
            self.current_bar_state = None;
            None
        };

        let mut bars = Vec::with_capacity(agg_trade_records.len() / 100); // Heuristic capacity
        let mut defer_open = false;

        for agg_record in agg_trade_records {
            // Track price and position for checkpoint
            self.price_window.push(agg_record.price);
            self.last_trade_id = Some(agg_record.agg_trade_id);
            self.last_timestamp_us = agg_record.timestamp;

            if defer_open {
                // Previous bar closed, this agg_record opens new bar
                current_bar = Some(RangeBarState::new(agg_record, self.threshold_decimal_bps));
                defer_open = false;
                continue;
            }

            match current_bar {
                None => {
                    // First bar initialization
                    current_bar = Some(RangeBarState::new(agg_record, self.threshold_decimal_bps));
                }
                Some(ref mut bar_state) => {
                    // Check if this AggTrade record breaches the threshold
                    let price_breaches = bar_state.bar.is_breach(
                        agg_record.price,
                        bar_state.upper_threshold,
                        bar_state.lower_threshold,
                    );

                    // Timestamp gate (Issue #36): prevent bars from closing on same timestamp
                    // This eliminates "instant bars" during flash crashes where multiple trades
                    // occur at the same millisecond.
                    let timestamp_allows_close = !self.prevent_same_timestamp_close
                        || agg_record.timestamp != bar_state.bar.open_time;

                    if price_breaches && timestamp_allows_close {
                        // Breach detected AND timestamp changed - update bar with breaching record
                        bar_state.bar.update_with_trade(agg_record);

                        // Validation: Ensure high/low include open/close extremes
                        debug_assert!(
                            bar_state.bar.high >= bar_state.bar.open.max(bar_state.bar.close)
                        );
                        debug_assert!(
                            bar_state.bar.low <= bar_state.bar.open.min(bar_state.bar.close)
                        );

                        // Compute microstructure features at bar finalization (Issue #34)
                        bar_state.bar.compute_microstructure_features();

                        bars.push(bar_state.bar.clone());
                        current_bar = None;
                        defer_open = true; // Next record will open new bar
                    } else {
                        // Either no breach OR same timestamp (gate active) - normal update
                        bar_state.bar.update_with_trade(agg_record);
                    }
                }
            }
        }

        // Save current bar state for checkpoint (preserves incomplete bar)
        self.current_bar_state = current_bar.clone();

        // Add final partial bar only if explicitly requested
        // This preserves algorithm integrity: bars should only close on threshold breach
        // Note: let-chains require Rust 2024, so we use nested if instead
        #[allow(clippy::collapsible_if)]
        if include_incomplete {
            if let Some(mut bar_state) = current_bar {
                // Compute microstructure features for incomplete bar (Issue #34)
                bar_state.bar.compute_microstructure_features();
                bars.push(bar_state.bar);
            }
        }

        Ok(bars)
    }

    // === CHECKPOINT METHODS ===

    /// Create checkpoint for cross-file continuation
    ///
    /// Captures current processing state for seamless continuation:
    /// - Incomplete bar (if any) with FIXED thresholds
    /// - Position tracking (timestamp, trade_id if available)
    /// - Price hash for verification
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol being processed (e.g., "BTCUSDT", "EURUSD")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bars = processor.process_agg_trade_records(&trades)?;
    /// let checkpoint = processor.create_checkpoint("BTCUSDT");
    /// let json = serde_json::to_string(&checkpoint)?;
    /// std::fs::write("checkpoint.json", json)?;
    /// ```
    pub fn create_checkpoint(&self, symbol: &str) -> Checkpoint {
        let (incomplete_bar, thresholds) = match &self.current_bar_state {
            Some(state) => (
                Some(state.bar.clone()),
                Some((state.upper_threshold, state.lower_threshold)),
            ),
            None => (None, None),
        };

        let mut checkpoint = Checkpoint::new(
            symbol.to_string(),
            self.threshold_decimal_bps,
            incomplete_bar,
            thresholds,
            self.last_timestamp_us,
            self.last_trade_id,
            self.price_window.compute_hash(),
            self.prevent_same_timestamp_close,
        );
        // Issue #46: Persist defer_open state for cross-session continuity
        checkpoint.defer_open = self.defer_open;
        checkpoint
    }

    /// Resume processing from checkpoint
    ///
    /// Restores incomplete bar state with IMMUTABLE thresholds.
    /// Next trade continues building the bar until threshold breach.
    ///
    /// # Errors
    ///
    /// - `CheckpointError::MissingThresholds` - Checkpoint has bar but no thresholds
    ///
    /// # Example
    ///
    /// ```ignore
    /// let json = std::fs::read_to_string("checkpoint.json")?;
    /// let checkpoint: Checkpoint = serde_json::from_str(&json)?;
    /// let mut processor = RangeBarProcessor::from_checkpoint(checkpoint)?;
    /// let bars = processor.process_agg_trade_records(&next_file_trades)?;
    /// ```
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Result<Self, CheckpointError> {
        // Validate checkpoint consistency
        if checkpoint.incomplete_bar.is_some() && checkpoint.thresholds.is_none() {
            return Err(CheckpointError::MissingThresholds);
        }

        // Restore bar state if there's an incomplete bar
        let current_bar_state = match (checkpoint.incomplete_bar, checkpoint.thresholds) {
            (Some(bar), Some((upper, lower))) => Some(RangeBarState {
                bar,
                upper_threshold: upper,
                lower_threshold: lower,
            }),
            _ => None,
        };

        Ok(Self {
            threshold_decimal_bps: checkpoint.threshold_decimal_bps,
            current_bar_state,
            price_window: PriceWindow::new(), // Reset - will be rebuilt from new trades
            last_trade_id: checkpoint.last_trade_id,
            last_timestamp_us: checkpoint.last_timestamp_us,
            anomaly_summary: checkpoint.anomaly_summary,
            resumed_from_checkpoint: true, // Signal to continue from existing bar state
            prevent_same_timestamp_close: checkpoint.prevent_same_timestamp_close,
            defer_open: checkpoint.defer_open, // Issue #46: Restore deferred open state
        })
    }

    /// Verify we're at the right position in the data stream
    ///
    /// Call with first trade of new file to verify continuity.
    /// Returns verification result indicating if there's a gap or exact match.
    ///
    /// # Arguments
    ///
    /// * `first_trade` - First trade of the new file/chunk
    ///
    /// # Example
    ///
    /// ```ignore
    /// let processor = RangeBarProcessor::from_checkpoint(checkpoint)?;
    /// let verification = processor.verify_position(&next_file_trades[0]);
    /// match verification {
    ///     PositionVerification::Exact => println!("Perfect continuation!"),
    ///     PositionVerification::Gap { missing_count, .. } => {
    ///         println!("Warning: {} trades missing", missing_count);
    ///     }
    ///     PositionVerification::TimestampOnly { gap_ms } => {
    ///         println!("Exness data: {}ms gap", gap_ms);
    ///     }
    /// }
    /// ```
    pub fn verify_position(&self, first_trade: &AggTrade) -> PositionVerification {
        match self.last_trade_id {
            Some(last_id) => {
                // Binance: has trade IDs - check for gaps
                let expected_id = last_id + 1;
                if first_trade.agg_trade_id == expected_id {
                    PositionVerification::Exact
                } else {
                    let missing_count = first_trade.agg_trade_id - expected_id;
                    PositionVerification::Gap {
                        expected_id,
                        actual_id: first_trade.agg_trade_id,
                        missing_count,
                    }
                }
            }
            None => {
                // Exness: no trade IDs - use timestamp only
                let gap_us = first_trade.timestamp - self.last_timestamp_us;
                let gap_ms = gap_us / 1000;
                PositionVerification::TimestampOnly { gap_ms }
            }
        }
    }

    /// Get the current anomaly summary
    pub fn anomaly_summary(&self) -> &AnomalySummary {
        &self.anomaly_summary
    }

    /// Get the threshold in decimal basis points
    pub fn threshold_decimal_bps(&self) -> u32 {
        self.threshold_decimal_bps
    }

    /// Validate that trades are properly sorted for deterministic processing
    fn validate_trade_ordering(&self, trades: &[AggTrade]) -> Result<(), ProcessingError> {
        for i in 1..trades.len() {
            let prev = &trades[i - 1];
            let curr = &trades[i];

            // Check ordering: (timestamp, agg_trade_id) ascending
            if curr.timestamp < prev.timestamp
                || (curr.timestamp == prev.timestamp && curr.agg_trade_id <= prev.agg_trade_id)
            {
                return Err(ProcessingError::UnsortedTrades {
                    index: i,
                    prev_time: prev.timestamp,
                    prev_id: prev.agg_trade_id,
                    curr_time: curr.timestamp,
                    curr_id: curr.agg_trade_id,
                });
            }
        }

        Ok(())
    }

    /// Reset processor state at an ouroboros boundary (year/month/week).
    ///
    /// Clears the incomplete bar and position tracking while preserving
    /// the threshold configuration. Use this when starting fresh at a
    /// known boundary for reproducibility.
    ///
    /// # Returns
    ///
    /// The orphaned incomplete bar (if any) so caller can decide
    /// whether to include it in results with `is_orphan=True` flag.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // At year boundary (Jan 1 00:00:00 UTC)
    /// let orphaned = processor.reset_at_ouroboros();
    /// if let Some(bar) = orphaned {
    ///     // Handle incomplete bar from previous year
    /// }
    /// // Continue processing new year's data with clean state
    /// ```
    pub fn reset_at_ouroboros(&mut self) -> Option<RangeBar> {
        let orphaned = self.current_bar_state.take().map(|state| state.bar);
        self.price_window = PriceWindow::new();
        self.last_trade_id = None;
        self.last_timestamp_us = 0;
        self.resumed_from_checkpoint = false;
        self.defer_open = false;
        orphaned
    }
}

/// Internal state for a range bar being built
#[derive(Clone)]
struct RangeBarState {
    /// The range bar being constructed
    pub bar: RangeBar,

    /// Upper breach threshold (FIXED from bar open)
    pub upper_threshold: FixedPoint,

    /// Lower breach threshold (FIXED from bar open)
    pub lower_threshold: FixedPoint,
}

impl RangeBarState {
    /// Create new range bar state from opening trade
    fn new(trade: &AggTrade, threshold_decimal_bps: u32) -> Self {
        let bar = RangeBar::new(trade);

        // Compute FIXED thresholds from opening price
        let (upper_threshold, lower_threshold) =
            bar.open.compute_range_thresholds(threshold_decimal_bps);

        Self {
            bar,
            upper_threshold,
            lower_threshold,
        }
    }
}

/// Processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error(
        "Trades not sorted at index {index}: prev=({prev_time}, {prev_id}), curr=({curr_time}, {curr_id})"
    )]
    UnsortedTrades {
        index: usize,
        prev_time: i64,
        prev_id: i64,
        curr_time: i64,
        curr_id: i64,
    },

    #[error("Empty trade data")]
    EmptyData,

    #[error(
        "Invalid threshold: {threshold_decimal_bps} dbps. Valid range: 1-100,000 dbps (0.001%-100%)"
    )]
    InvalidThreshold { threshold_decimal_bps: u32 },
}

#[cfg(feature = "python")]
impl From<ProcessingError> for PyErr {
    fn from(err: ProcessingError) -> PyErr {
        match err {
            ProcessingError::UnsortedTrades {
                index,
                prev_time,
                prev_id,
                curr_time,
                curr_id,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "Trades not sorted at index {}: prev=({}, {}), curr=({}, {})",
                index, prev_time, prev_id, curr_time, curr_id
            )),
            ProcessingError::EmptyData => {
                pyo3::exceptions::PyValueError::new_err("Empty trade data")
            }
            ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid threshold: {} dbps. Valid range: 1-100,000 dbps (0.001%-100%)",
                threshold_decimal_bps
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{self, scenarios};

    #[test]
    fn test_single_bar_no_breach() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 dbps = 0.25%

        // Create trades that stay within 250 dbps threshold
        let trades = scenarios::no_breach_sequence(250);

        // Test strict algorithm compliance: no bars should be created without breach
        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert_eq!(
            bars.len(),
            0,
            "Strict algorithm should not create bars without breach"
        );

        // Test analysis mode: incomplete bar should be available for analysis
        let bars_with_incomplete = processor
            .process_agg_trade_records_with_incomplete(&trades)
            .unwrap();
        assert_eq!(
            bars_with_incomplete.len(),
            1,
            "Analysis mode should include incomplete bar"
        );

        let bar = &bars_with_incomplete[0];
        assert_eq!(bar.open.to_string(), "50000.00000000");
        assert_eq!(bar.high.to_string(), "50100.00000000");
        assert_eq!(bar.low.to_string(), "49900.00000000");
        assert_eq!(bar.close.to_string(), "49900.00000000");
    }

    #[test]
    fn test_exact_breach_upward() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 dbps = 0.25%

        let trades = scenarios::exact_breach_upward(250);

        // Test strict algorithm: only completed bars (with breach)
        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert_eq!(
            bars.len(),
            1,
            "Strict algorithm should only return completed bars"
        );

        // First bar should close at breach
        let bar1 = &bars[0];
        assert_eq!(bar1.open.to_string(), "50000.00000000");
        // Breach at 250 dbps = 0.25% = 50000 * 1.0025 = 50125
        assert_eq!(bar1.close.to_string(), "50125.00000000"); // Breach tick included
        assert_eq!(bar1.high.to_string(), "50125.00000000");
        assert_eq!(bar1.low.to_string(), "50000.00000000");

        // Test analysis mode: includes incomplete second bar
        let bars_with_incomplete = processor
            .process_agg_trade_records_with_incomplete(&trades)
            .unwrap();
        assert_eq!(
            bars_with_incomplete.len(),
            2,
            "Analysis mode should include incomplete bars"
        );

        // Second bar should start at next tick price (not breach price)
        let bar2 = &bars_with_incomplete[1];
        assert_eq!(bar2.open.to_string(), "50500.00000000"); // Next tick after breach
        assert_eq!(bar2.close.to_string(), "50500.00000000");
    }

    #[test]
    fn test_exact_breach_downward() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 × 0.1bps = 25bps = 0.25%

        let trades = scenarios::exact_breach_downward(250);

        let bars = processor.process_agg_trade_records(&trades).unwrap();

        assert_eq!(bars.len(), 1);

        let bar = &bars[0];
        assert_eq!(bar.open.to_string(), "50000.00000000");
        assert_eq!(bar.close.to_string(), "49875.00000000"); // Breach tick included
        assert_eq!(bar.high.to_string(), "50000.00000000");
        assert_eq!(bar.low.to_string(), "49875.00000000");
    }

    #[test]
    fn test_large_gap_single_bar() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 × 0.1bps = 25bps = 0.25%

        let trades = scenarios::large_gap_sequence();

        let bars = processor.process_agg_trade_records(&trades).unwrap();

        // Should create exactly ONE bar, not multiple bars to "fill the gap"
        assert_eq!(bars.len(), 1);

        let bar = &bars[0];
        assert_eq!(bar.open.to_string(), "50000.00000000");
        assert_eq!(bar.close.to_string(), "51000.00000000");
        assert_eq!(bar.high.to_string(), "51000.00000000");
        assert_eq!(bar.low.to_string(), "50000.00000000");
    }

    #[test]
    fn test_unsorted_trades_error() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 × 0.1bps = 25bps

        let trades = scenarios::unsorted_sequence();

        let result = processor.process_agg_trade_records(&trades);
        assert!(result.is_err());

        match result {
            Err(ProcessingError::UnsortedTrades { index, .. }) => {
                assert_eq!(index, 1);
            }
            _ => panic!("Expected UnsortedTrades error"),
        }
    }

    #[test]
    fn test_threshold_calculation() {
        let processor = RangeBarProcessor::new(250).unwrap(); // 250 × 0.1bps = 25bps = 0.25%

        let trade = test_utils::create_test_agg_trade(1, "50000.0", "1.0", 1000);
        let bar_state = RangeBarState::new(&trade, processor.threshold_decimal_bps);

        // 50000 * 0.0025 = 125 (25bps = 0.25%)
        assert_eq!(bar_state.upper_threshold.to_string(), "50125.00000000");
        assert_eq!(bar_state.lower_threshold.to_string(), "49875.00000000");
    }

    #[test]
    fn test_empty_trades() {
        let mut processor = RangeBarProcessor::new(250).unwrap(); // 250 × 0.1bps = 25bps
        let trades = scenarios::empty_sequence();
        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert_eq!(bars.len(), 0);
    }

    #[test]
    fn test_debug_streaming_data() {
        let mut processor = RangeBarProcessor::new(100).unwrap(); // 100 × 0.1bps = 10bps = 0.1%

        // Create trades similar to our test data
        let trades = vec![
            test_utils::create_test_agg_trade(1, "50014.00859087", "0.12019569", 1756710002083),
            test_utils::create_test_agg_trade(2, "50163.87750994", "1.01283708", 1756710005113), // ~0.3% increase
            test_utils::create_test_agg_trade(3, "50032.44128269", "0.69397094", 1756710008770),
        ];

        println!("Test data prices: 50014 -> 50163 -> 50032");
        println!("Expected price movements: +0.3% then -0.26%");

        let bars = processor.process_agg_trade_records(&trades).unwrap();
        println!("Generated {} range bars", bars.len());

        for (i, bar) in bars.iter().enumerate() {
            println!(
                "  Bar {}: O={} H={} L={} C={}",
                i + 1,
                bar.open,
                bar.high,
                bar.low,
                bar.close
            );
        }

        // With a 0.1% threshold and 0.3% price movement, we should get at least 1 bar
        assert!(
            !bars.is_empty(),
            "Expected at least 1 range bar with 0.3% price movement and 0.1% threshold"
        );
    }

    #[test]
    fn test_threshold_validation() {
        // Valid threshold
        assert!(RangeBarProcessor::new(250).is_ok());

        // Invalid: too low (0 × 0.1bps = 0%)
        assert!(matches!(
            RangeBarProcessor::new(0),
            Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps: 0
            })
        ));

        // Invalid: too high (150,000 × 0.1bps = 15,000bps = 150%)
        assert!(matches!(
            RangeBarProcessor::new(150_000),
            Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps: 150_000
            })
        ));

        // Valid boundary: minimum (1 × 0.1bps = 0.1bps = 0.001%)
        assert!(RangeBarProcessor::new(1).is_ok());

        // Valid boundary: maximum (100,000 × 0.1bps = 10,000bps = 100%)
        assert!(RangeBarProcessor::new(100_000).is_ok());
    }

    #[test]
    fn test_export_processor_with_manual_trades() {
        println!("Testing ExportRangeBarProcessor with same trade data...");

        let mut export_processor = ExportRangeBarProcessor::new(100).unwrap(); // 100 × 0.1bps = 10bps = 0.1%

        // Use same trades as the working basic test
        let trades = vec![
            test_utils::create_test_agg_trade(1, "50014.00859087", "0.12019569", 1756710002083),
            test_utils::create_test_agg_trade(2, "50163.87750994", "1.01283708", 1756710005113), // ~0.3% increase
            test_utils::create_test_agg_trade(3, "50032.44128269", "0.69397094", 1756710008770),
        ];

        println!(
            "Processing {} trades with ExportRangeBarProcessor...",
            trades.len()
        );

        export_processor.process_trades_continuously(&trades);
        let bars = export_processor.get_all_completed_bars();

        println!(
            "ExportRangeBarProcessor generated {} range bars",
            bars.len()
        );
        for (i, bar) in bars.iter().enumerate() {
            println!(
                "  Bar {}: O={} H={} L={} C={}",
                i + 1,
                bar.open,
                bar.high,
                bar.low,
                bar.close
            );
        }

        // Should match the basic processor results (1 bar)
        assert!(
            !bars.is_empty(),
            "ExportRangeBarProcessor should generate same results as basic processor"
        );
    }

    // === CHECKPOINT TESTS (Issues #2 and #3) ===

    #[test]
    fn test_checkpoint_creation() {
        let mut processor = RangeBarProcessor::new(250).unwrap();

        // Process some trades that don't complete a bar
        let trades = scenarios::no_breach_sequence(250);
        let _bars = processor.process_agg_trade_records(&trades).unwrap();

        // Create checkpoint
        let checkpoint = processor.create_checkpoint("BTCUSDT");

        assert_eq!(checkpoint.symbol, "BTCUSDT");
        assert_eq!(checkpoint.threshold_decimal_bps, 250);
        assert!(checkpoint.has_incomplete_bar()); // Should have incomplete bar
        assert!(checkpoint.thresholds.is_some()); // Thresholds should be saved
        assert!(checkpoint.last_trade_id.is_some()); // Should track last trade
    }

    #[test]
    fn test_checkpoint_serialization_roundtrip() {
        let mut processor = RangeBarProcessor::new(250).unwrap();

        // Process trades
        let trades = scenarios::no_breach_sequence(250);
        let _bars = processor.process_agg_trade_records(&trades).unwrap();

        // Create checkpoint
        let checkpoint = processor.create_checkpoint("BTCUSDT");

        // Serialize to JSON
        let json = serde_json::to_string(&checkpoint).expect("Serialization should succeed");

        // Deserialize back
        let restored: Checkpoint =
            serde_json::from_str(&json).expect("Deserialization should succeed");

        assert_eq!(restored.symbol, checkpoint.symbol);
        assert_eq!(
            restored.threshold_decimal_bps,
            checkpoint.threshold_decimal_bps
        );
        assert_eq!(
            restored.incomplete_bar.is_some(),
            checkpoint.incomplete_bar.is_some()
        );
    }

    #[test]
    fn test_cross_file_bar_continuation() {
        // This is the PRIMARY test for Issues #2 and #3
        // Verifies that incomplete bars continue correctly across file boundaries

        // Create trades that span multiple bars
        let mut all_trades = Vec::new();

        // Generate enough trades to produce multiple bars
        // Using 100bps threshold (1%) for clearer price movements
        let base_timestamp = 1640995200000000i64; // Microseconds

        // Create a sequence where we'll have ~3-4 completed bars with remainder
        for i in 0..20 {
            let price = 50000.0 + (i as f64 * 100.0) * if i % 4 < 2 { 1.0 } else { -1.0 };
            let trade = test_utils::create_test_agg_trade(
                i + 1,
                &format!("{:.8}", price),
                "1.0",
                base_timestamp + (i as i64 * 1000000),
            );
            all_trades.push(trade);
        }

        // === FULL PROCESSING (baseline) ===
        let mut processor_full = RangeBarProcessor::new(100).unwrap(); // 100 × 0.1bps = 10bps = 0.1%
        let bars_full = processor_full
            .process_agg_trade_records(&all_trades)
            .unwrap();

        // === SPLIT PROCESSING WITH CHECKPOINT ===
        let split_point = 10; // Split in the middle

        // Part 1: Process first half
        let mut processor_1 = RangeBarProcessor::new(100).unwrap();
        let part1_trades = &all_trades[0..split_point];
        let bars_1 = processor_1.process_agg_trade_records(part1_trades).unwrap();

        // Create checkpoint
        let checkpoint = processor_1.create_checkpoint("TEST");

        // Part 2: Resume from checkpoint and process second half
        let mut processor_2 = RangeBarProcessor::from_checkpoint(checkpoint).unwrap();
        let part2_trades = &all_trades[split_point..];
        let bars_2 = processor_2.process_agg_trade_records(part2_trades).unwrap();

        // === VERIFY CONTINUATION ===
        // Total completed bars should match full processing
        let split_total = bars_1.len() + bars_2.len();

        println!("Full processing: {} bars", bars_full.len());
        println!(
            "Split processing: {} + {} = {} bars",
            bars_1.len(),
            bars_2.len(),
            split_total
        );

        assert_eq!(
            split_total,
            bars_full.len(),
            "Split processing should produce same bar count as full processing"
        );

        // Verify the bars themselves match
        let all_split_bars: Vec<_> = bars_1.iter().chain(bars_2.iter()).collect();
        for (i, (full, split)) in bars_full.iter().zip(all_split_bars.iter()).enumerate() {
            assert_eq!(full.open.0, split.open.0, "Bar {} open price mismatch", i);
            assert_eq!(
                full.close.0, split.close.0,
                "Bar {} close price mismatch",
                i
            );
        }
    }

    #[test]
    fn test_verify_position_exact() {
        let mut processor = RangeBarProcessor::new(250).unwrap();

        // Process some trades
        let trade1 = test_utils::create_test_agg_trade(100, "50000.0", "1.0", 1640995200000000);
        let trade2 = test_utils::create_test_agg_trade(101, "50010.0", "1.0", 1640995201000000);

        let _ = processor.process_single_trade(trade1);
        let _ = processor.process_single_trade(trade2);

        // Create next trade in sequence
        let next_trade = test_utils::create_test_agg_trade(102, "50020.0", "1.0", 1640995202000000);

        // Verify position
        let verification = processor.verify_position(&next_trade);

        assert_eq!(verification, PositionVerification::Exact);
    }

    #[test]
    fn test_verify_position_gap() {
        let mut processor = RangeBarProcessor::new(250).unwrap();

        // Process some trades
        let trade1 = test_utils::create_test_agg_trade(100, "50000.0", "1.0", 1640995200000000);
        let trade2 = test_utils::create_test_agg_trade(101, "50010.0", "1.0", 1640995201000000);

        let _ = processor.process_single_trade(trade1);
        let _ = processor.process_single_trade(trade2);

        // Create next trade with gap (skip IDs 102-104)
        let next_trade = test_utils::create_test_agg_trade(105, "50020.0", "1.0", 1640995202000000);

        // Verify position
        let verification = processor.verify_position(&next_trade);

        match verification {
            PositionVerification::Gap {
                expected_id,
                actual_id,
                missing_count,
            } => {
                assert_eq!(expected_id, 102);
                assert_eq!(actual_id, 105);
                assert_eq!(missing_count, 3);
            }
            _ => panic!("Expected Gap verification, got {:?}", verification),
        }
    }

    #[test]
    fn test_checkpoint_clean_completion() {
        // Test when last trade completes a bar with no remainder
        // In range bar algorithm: breach trade closes bar, NEXT trade opens new bar
        // If there's no next trade, there's no incomplete bar
        let mut processor = RangeBarProcessor::new(100).unwrap(); // 10bps

        // Create trades that complete exactly one bar
        let trades = vec![
            test_utils::create_test_agg_trade(1, "50000.0", "1.0", 1640995200000000),
            test_utils::create_test_agg_trade(2, "50100.0", "1.0", 1640995201000000), // ~0.2% move, breaches 0.1%
        ];

        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert_eq!(bars.len(), 1, "Should have exactly one completed bar");

        // Create checkpoint - should NOT have incomplete bar
        // (breach trade closes bar, no next trade to open new bar)
        let checkpoint = processor.create_checkpoint("TEST");

        // With defer_open logic, the next bar isn't started until the next trade
        assert!(
            !checkpoint.has_incomplete_bar(),
            "No incomplete bar when last trade was a breach with no following trade"
        );
    }

    #[test]
    fn test_checkpoint_with_remainder() {
        // Test when we have trades remaining after a completed bar
        let mut processor = RangeBarProcessor::new(100).unwrap(); // 10bps

        // Create trades: bar completes at trade 2, trade 3 starts new bar
        let trades = vec![
            test_utils::create_test_agg_trade(1, "50000.0", "1.0", 1640995200000000),
            test_utils::create_test_agg_trade(2, "50100.0", "1.0", 1640995201000000), // Breach
            test_utils::create_test_agg_trade(3, "50110.0", "1.0", 1640995202000000), // Opens new bar
        ];

        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert_eq!(bars.len(), 1, "Should have exactly one completed bar");

        // Create checkpoint - should have incomplete bar from trade 3
        let checkpoint = processor.create_checkpoint("TEST");

        assert!(
            checkpoint.has_incomplete_bar(),
            "Should have incomplete bar from trade 3"
        );

        // Verify the incomplete bar has correct data
        let incomplete = checkpoint.incomplete_bar.unwrap();
        assert_eq!(
            incomplete.open.to_string(),
            "50110.00000000",
            "Incomplete bar should open at trade 3 price"
        );
    }

    /// Issue #46: Verify streaming and batch paths produce identical bars
    ///
    /// The batch path (`process_agg_trade_records`) and streaming path
    /// (`process_single_trade`) must produce identical OHLCV output for
    /// the same input trades. This test catches regressions where the
    /// breaching trade is double-counted or bar boundaries differ.
    #[test]
    fn test_streaming_batch_parity() {
        let threshold = 250; // 250 dbps = 0.25%

        // Build a sequence with multiple breaches
        let trades = test_utils::AggTradeBuilder::new()
            .add_trade(1, 1.0, 0)          // Open first bar at 50000
            .add_trade(2, 1.001, 1000)     // +0.1% - accumulate
            .add_trade(3, 1.003, 2000)     // +0.3% - breach (>0.25%)
            .add_trade(4, 1.004, 3000)     // Opens second bar
            .add_trade(5, 1.005, 4000)     // Accumulate
            .add_trade(6, 1.008, 5000)     // +0.4% from bar 2 open - breach
            .add_trade(7, 1.009, 6000)     // Opens third bar
            .build();

        // === BATCH PATH ===
        let mut batch_processor = RangeBarProcessor::new(threshold).unwrap();
        let batch_bars = batch_processor.process_agg_trade_records(&trades).unwrap();
        let batch_incomplete = batch_processor.get_incomplete_bar();

        // === STREAMING PATH ===
        let mut stream_processor = RangeBarProcessor::new(threshold).unwrap();
        let mut stream_bars: Vec<RangeBar> = Vec::new();
        for trade in &trades {
            if let Some(bar) = stream_processor.process_single_trade(trade.clone()).unwrap() {
                stream_bars.push(bar);
            }
        }
        let stream_incomplete = stream_processor.get_incomplete_bar();

        // === VERIFY PARITY ===
        assert_eq!(
            batch_bars.len(),
            stream_bars.len(),
            "Batch and streaming should produce same number of completed bars"
        );

        for (i, (batch_bar, stream_bar)) in
            batch_bars.iter().zip(stream_bars.iter()).enumerate()
        {
            assert_eq!(
                batch_bar.open, stream_bar.open,
                "Bar {i}: open price mismatch"
            );
            assert_eq!(
                batch_bar.close, stream_bar.close,
                "Bar {i}: close price mismatch"
            );
            assert_eq!(
                batch_bar.high, stream_bar.high,
                "Bar {i}: high price mismatch"
            );
            assert_eq!(
                batch_bar.low, stream_bar.low,
                "Bar {i}: low price mismatch"
            );
            assert_eq!(
                batch_bar.volume, stream_bar.volume,
                "Bar {i}: volume mismatch (double-counting?)"
            );
            assert_eq!(
                batch_bar.open_time, stream_bar.open_time,
                "Bar {i}: open_time mismatch"
            );
            assert_eq!(
                batch_bar.close_time, stream_bar.close_time,
                "Bar {i}: close_time mismatch"
            );
            assert_eq!(
                batch_bar.individual_trade_count, stream_bar.individual_trade_count,
                "Bar {i}: trade count mismatch"
            );
        }

        // Verify incomplete bars match
        match (batch_incomplete, stream_incomplete) {
            (Some(b), Some(s)) => {
                assert_eq!(b.open, s.open, "Incomplete bar: open mismatch");
                assert_eq!(b.close, s.close, "Incomplete bar: close mismatch");
                assert_eq!(b.volume, s.volume, "Incomplete bar: volume mismatch");
            }
            (None, None) => {} // Both finished cleanly
            _ => panic!("Incomplete bar presence mismatch between batch and streaming"),
        }
    }

    /// Issue #46: After breach, next trade opens new bar (not breaching trade)
    #[test]
    fn test_defer_open_new_bar_opens_with_next_trade() {
        let mut processor = RangeBarProcessor::new(250).unwrap();

        // Trade 1: Opens bar at 50000
        let t1 = test_utils::create_test_agg_trade(1, "50000.0", "1.0", 1000);
        assert!(processor.process_single_trade(t1).unwrap().is_none());

        // Trade 2: Breaches threshold (+0.3%)
        let t2 = test_utils::create_test_agg_trade(2, "50150.0", "2.0", 2000);
        let bar = processor.process_single_trade(t2).unwrap();
        assert!(bar.is_some(), "Should close bar on breach");

        let closed_bar = bar.unwrap();
        assert_eq!(closed_bar.open.to_string(), "50000.00000000");
        assert_eq!(closed_bar.close.to_string(), "50150.00000000");

        // After breach, no incomplete bar should exist
        assert!(
            processor.get_incomplete_bar().is_none(),
            "No incomplete bar after breach - defer_open is true"
        );

        // Trade 3: Should open NEW bar (not the breaching trade)
        let t3 = test_utils::create_test_agg_trade(3, "50100.0", "3.0", 3000);
        assert!(processor.process_single_trade(t3).unwrap().is_none());

        let incomplete = processor.get_incomplete_bar().unwrap();
        assert_eq!(
            incomplete.open.to_string(),
            "50100.00000000",
            "New bar should open at trade 3's price, not trade 2's"
        );
    }
}

/// Internal state for range bar construction with fixed-point precision
#[derive(Debug, Clone)]
struct InternalRangeBar {
    open_time: i64,
    close_time: i64,
    open: FixedPoint,
    high: FixedPoint,
    low: FixedPoint,
    close: FixedPoint,
    volume: FixedPoint,
    turnover: i128,
    individual_trade_count: i64,
    agg_record_count: u32,
    first_trade_id: i64,
    last_trade_id: i64,
    /// Volume from buy-side trades (is_buyer_maker = false)
    buy_volume: FixedPoint,
    /// Volume from sell-side trades (is_buyer_maker = true)
    sell_volume: FixedPoint,
    /// Number of buy-side trades
    buy_trade_count: i64,
    /// Number of sell-side trades
    sell_trade_count: i64,
    /// Volume Weighted Average Price
    vwap: FixedPoint,
    /// Turnover from buy-side trades
    buy_turnover: i128,
    /// Turnover from sell-side trades
    sell_turnover: i128,
}

/// Export-oriented range bar processor for streaming use cases
///
/// This implementation uses the proven fixed-point arithmetic algorithm
/// that achieves 100% breach consistency compliance in multi-year processing.
pub struct ExportRangeBarProcessor {
    threshold_decimal_bps: u32,
    current_bar: Option<InternalRangeBar>,
    completed_bars: Vec<RangeBar>,
    /// Prevent bars from closing on same timestamp as they opened (Issue #36)
    prevent_same_timestamp_close: bool,
    /// Deferred bar open flag (Issue #46) - next trade opens new bar after breach
    defer_open: bool,
}

impl ExportRangeBarProcessor {
    /// Create new export processor with given threshold
    ///
    /// Uses default behavior: `prevent_same_timestamp_close = true` (Issue #36)
    ///
    /// # Arguments
    ///
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    ///   - Example: `250` → 25bps = 0.25%
    ///   - Example: `10` → 1bps = 0.01%
    ///   - Minimum: `1` → 0.1bps = 0.001%
    ///
    /// # Breaking Change (v3.0.0)
    ///
    /// Prior to v3.0.0, `threshold_decimal_bps` was in 1bps units.
    /// **Migration**: Multiply all threshold values by 10.
    pub fn new(threshold_decimal_bps: u32) -> Result<Self, ProcessingError> {
        Self::with_options(threshold_decimal_bps, true)
    }

    /// Create new export processor with explicit timestamp gating control
    pub fn with_options(
        threshold_decimal_bps: u32,
        prevent_same_timestamp_close: bool,
    ) -> Result<Self, ProcessingError> {
        // Validation bounds (v3.0.0: dbps units)
        // Min: 1 dbps = 0.001%
        // Max: 100,000 dbps = 100%
        if threshold_decimal_bps < 1 {
            return Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            });
        }
        if threshold_decimal_bps > 100_000 {
            return Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            });
        }

        Ok(Self {
            threshold_decimal_bps,
            current_bar: None,
            completed_bars: Vec::new(),
            prevent_same_timestamp_close,
            defer_open: false,
        })
    }

    /// Process trades continuously using proven fixed-point algorithm
    /// This method maintains 100% breach consistency by using precise integer arithmetic
    pub fn process_trades_continuously(&mut self, trades: &[AggTrade]) {
        for trade in trades {
            self.process_single_trade_fixed_point(trade);
        }
    }

    /// Process single trade using proven fixed-point algorithm (100% breach consistency)
    fn process_single_trade_fixed_point(&mut self, trade: &AggTrade) {
        // Issue #46: If previous trade triggered a breach, this trade opens the new bar.
        // This matches the batch path's defer_open semantics.
        if self.defer_open {
            self.defer_open = false;
            self.current_bar = None; // Clear any stale state
            // Fall through to the is_none() branch below to start new bar
        }

        if self.current_bar.is_none() {
            // Start new bar
            let trade_turnover = (trade.price.to_f64() * trade.volume.to_f64()) as i128;

            self.current_bar = Some(InternalRangeBar {
                open_time: trade.timestamp,
                close_time: trade.timestamp,
                open: trade.price,
                high: trade.price,
                low: trade.price,
                close: trade.price,
                volume: trade.volume,
                turnover: trade_turnover,
                individual_trade_count: 1,
                agg_record_count: 1,
                first_trade_id: trade.agg_trade_id,
                last_trade_id: trade.agg_trade_id,
                // Market microstructure fields
                buy_volume: if trade.is_buyer_maker {
                    FixedPoint(0)
                } else {
                    trade.volume
                },
                sell_volume: if trade.is_buyer_maker {
                    trade.volume
                } else {
                    FixedPoint(0)
                },
                buy_trade_count: if trade.is_buyer_maker { 0 } else { 1 },
                sell_trade_count: if trade.is_buyer_maker { 1 } else { 0 },
                vwap: trade.price,
                buy_turnover: if trade.is_buyer_maker {
                    0
                } else {
                    trade_turnover
                },
                sell_turnover: if trade.is_buyer_maker {
                    trade_turnover
                } else {
                    0
                },
            });
            return;
        }

        // Process existing bar - work with reference
        // SAFETY: current_bar guaranteed Some - early return above if None
        let bar = self.current_bar.as_mut().unwrap();
        let trade_turnover = (trade.price.to_f64() * trade.volume.to_f64()) as i128;

        // CRITICAL FIX: Use fixed-point integer arithmetic for precise threshold calculation
        // v3.0.0: threshold now in dbps, using BASIS_POINTS_SCALE = 100_000
        let price_val = trade.price.0;
        let bar_open_val = bar.open.0;
        let threshold_decimal_bps = self.threshold_decimal_bps as i64;
        let upper_threshold = bar_open_val + (bar_open_val * threshold_decimal_bps) / 100_000;
        let lower_threshold = bar_open_val - (bar_open_val * threshold_decimal_bps) / 100_000;

        // Update bar with new trade
        bar.close_time = trade.timestamp;
        bar.close = trade.price;
        bar.volume.0 += trade.volume.0;
        bar.turnover += trade_turnover;
        bar.individual_trade_count += 1;
        bar.agg_record_count += 1;
        bar.last_trade_id = trade.agg_trade_id;

        // Update high/low
        if price_val > bar.high.0 {
            bar.high = trade.price;
        }
        if price_val < bar.low.0 {
            bar.low = trade.price;
        }

        // Update market microstructure
        if trade.is_buyer_maker {
            bar.sell_volume.0 += trade.volume.0;
            bar.sell_turnover += trade_turnover;
            bar.sell_trade_count += 1;
        } else {
            bar.buy_volume.0 += trade.volume.0;
            bar.buy_turnover += trade_turnover;
            bar.buy_trade_count += 1;
        }

        // CRITICAL: Fixed-point threshold breach detection (matches proven 100% compliance algorithm)
        let price_breaches = price_val >= upper_threshold || price_val <= lower_threshold;

        // Timestamp gate (Issue #36): prevent bars from closing on same timestamp
        let timestamp_allows_close =
            !self.prevent_same_timestamp_close || trade.timestamp != bar.open_time;

        if price_breaches && timestamp_allows_close {
            // Close current bar and move to completed
            // SAFETY: current_bar guaranteed Some - checked at line 688/734
            let completed_bar = self.current_bar.take().unwrap();

            // Convert to export format (this is from an old internal structure)
            let mut export_bar = RangeBar {
                open_time: completed_bar.open_time,
                close_time: completed_bar.close_time,
                open: completed_bar.open,
                high: completed_bar.high,
                low: completed_bar.low,
                close: completed_bar.close,
                volume: completed_bar.volume,
                turnover: completed_bar.turnover,

                // Enhanced fields
                individual_trade_count: completed_bar.individual_trade_count as u32,
                agg_record_count: completed_bar.agg_record_count,
                first_trade_id: completed_bar.first_trade_id,
                last_trade_id: completed_bar.last_trade_id,
                data_source: crate::types::DataSource::default(),

                // Market microstructure fields
                buy_volume: completed_bar.buy_volume,
                sell_volume: completed_bar.sell_volume,
                buy_trade_count: completed_bar.buy_trade_count as u32,
                sell_trade_count: completed_bar.sell_trade_count as u32,
                vwap: completed_bar.vwap,
                buy_turnover: completed_bar.buy_turnover,
                sell_turnover: completed_bar.sell_turnover,

                // Microstructure features (Issue #25) - computed below
                duration_us: 0,
                ofi: 0.0,
                vwap_close_deviation: 0.0,
                price_impact: 0.0,
                kyle_lambda_proxy: 0.0,
                trade_intensity: 0.0,
                volume_per_trade: 0.0,
                aggression_ratio: 0.0,
                aggregation_density_f64: 0.0,
                turnover_imbalance: 0.0,
            };

            // Compute microstructure features at bar finalization (Issue #25)
            export_bar.compute_microstructure_features();

            self.completed_bars.push(export_bar);

            // Issue #46: Don't start new bar with breaching trade.
            // Next trade will open the new bar via defer_open.
            self.current_bar = None;
            self.defer_open = true;
        }
    }

    /// Get all completed bars accumulated so far
    /// This drains the internal buffer to avoid memory leaks
    pub fn get_all_completed_bars(&mut self) -> Vec<RangeBar> {
        std::mem::take(&mut self.completed_bars)
    }

    /// Get incomplete bar if exists (for final bar processing)
    pub fn get_incomplete_bar(&mut self) -> Option<RangeBar> {
        self.current_bar.as_ref().map(|incomplete| {
            let mut bar = RangeBar {
                open_time: incomplete.open_time,
                close_time: incomplete.close_time,
                open: incomplete.open,
                high: incomplete.high,
                low: incomplete.low,
                close: incomplete.close,
                volume: incomplete.volume,
                turnover: incomplete.turnover,

                // Enhanced fields
                individual_trade_count: incomplete.individual_trade_count as u32,
                agg_record_count: incomplete.agg_record_count,
                first_trade_id: incomplete.first_trade_id,
                last_trade_id: incomplete.last_trade_id,
                data_source: crate::types::DataSource::default(),

                // Market microstructure fields
                buy_volume: incomplete.buy_volume,
                sell_volume: incomplete.sell_volume,
                buy_trade_count: incomplete.buy_trade_count as u32,
                sell_trade_count: incomplete.sell_trade_count as u32,
                vwap: incomplete.vwap,
                buy_turnover: incomplete.buy_turnover,
                sell_turnover: incomplete.sell_turnover,

                // Microstructure features (Issue #25) - computed below
                duration_us: 0,
                ofi: 0.0,
                vwap_close_deviation: 0.0,
                price_impact: 0.0,
                kyle_lambda_proxy: 0.0,
                trade_intensity: 0.0,
                volume_per_trade: 0.0,
                aggression_ratio: 0.0,
                aggregation_density_f64: 0.0,
                turnover_imbalance: 0.0,
            };
            // Compute microstructure features for incomplete bar (Issue #25)
            bar.compute_microstructure_features();
            bar
        })
    }
}
