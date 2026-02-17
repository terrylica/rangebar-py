// FILE-SIZE-OK: 1496 lines — tests are inline (access private RangeBarState)
//! Core range bar processing algorithm
//!
//! Implements non-lookahead bias range bar construction where bars close when
//! price moves ±threshold dbps from the bar's OPEN price.

use crate::checkpoint::{
    AnomalySummary, Checkpoint, CheckpointError, PositionVerification, PriceWindow,
};
use crate::fixed_point::FixedPoint;
use crate::interbar::{InterBarConfig, TradeHistory}; // Issue #59
use crate::intrabar::compute_intra_bar_features; // Issue #59: Intra-bar features
use crate::types::{AggTrade, RangeBar};
#[cfg(feature = "python")]
use pyo3::prelude::*;
// Re-export ProcessingError from errors.rs (Phase 2a extraction)
pub use crate::errors::ProcessingError;
// Re-export ExportRangeBarProcessor from export_processor.rs (Phase 2d extraction)
pub use crate::export_processor::ExportRangeBarProcessor;

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

    /// Trade history for inter-bar feature computation (Issue #59)
    ///
    /// Ring buffer of recent trades for computing lookback-based features.
    /// When Some, features are computed from trades BEFORE each bar's open_time.
    /// When None, inter-bar features are disabled (all lookback_* fields = None).
    trade_history: Option<TradeHistory>,

    /// Configuration for inter-bar features (Issue #59)
    ///
    /// Controls lookback mode (fixed count or time window) and which feature
    /// tiers to compute. When None, inter-bar features are disabled.
    inter_bar_config: Option<InterBarConfig>,

    /// Enable intra-bar feature computation (Issue #59)
    ///
    /// When true, the processor accumulates trades during bar construction
    /// and computes 22 features from trades WITHIN each bar at bar close.
    /// Features include ITH (Investment Time Horizon), statistical, and
    /// complexity metrics. When false, all intra_* fields are None.
    include_intra_bar_features: bool,
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
            trade_history: None,               // Issue #59: disabled by default
            inter_bar_config: None,            // Issue #59: disabled by default
            include_intra_bar_features: false, // Issue #59: disabled by default
        })
    }

    /// Get the prevent_same_timestamp_close setting
    pub fn prevent_same_timestamp_close(&self) -> bool {
        self.prevent_same_timestamp_close
    }

    /// Enable inter-bar feature computation with the given configuration (Issue #59)
    ///
    /// When enabled, the processor maintains a trade history buffer and computes
    /// lookback-based microstructure features on each bar close. Features are
    /// computed from trades that occurred BEFORE each bar's open_time, ensuring
    /// no lookahead bias.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling lookback mode and feature tiers
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rangebar_core::processor::RangeBarProcessor;
    /// use rangebar_core::interbar::{InterBarConfig, LookbackMode};
    ///
    /// let processor = RangeBarProcessor::new(1000)?
    ///     .with_inter_bar_config(InterBarConfig {
    ///         lookback_mode: LookbackMode::FixedCount(500),
    ///         compute_tier2: true,
    ///         compute_tier3: true,
    ///     });
    /// ```
    pub fn with_inter_bar_config(mut self, config: InterBarConfig) -> Self {
        self.trade_history = Some(TradeHistory::new(config.clone()));
        self.inter_bar_config = Some(config);
        self
    }

    /// Check if inter-bar features are enabled
    pub fn inter_bar_enabled(&self) -> bool {
        self.inter_bar_config.is_some()
    }

    /// Enable intra-bar feature computation (Issue #59)
    ///
    /// When enabled, the processor accumulates trades during bar construction
    /// and computes 22 features from trades WITHIN each bar at bar close:
    /// - 8 ITH features (Investment Time Horizon)
    /// - 12 statistical features (OFI, intensity, Kyle lambda, etc.)
    /// - 2 complexity features (Hurst exponent, permutation entropy)
    ///
    /// # Memory Note
    ///
    /// Trades are accumulated per-bar and freed when the bar closes.
    /// Typical 1000 dbps bar: ~50-500 trades, ~2-24 KB overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let processor = RangeBarProcessor::new(1000)?
    ///     .with_intra_bar_features();
    /// ```
    pub fn with_intra_bar_features(mut self) -> Self {
        self.include_intra_bar_features = true;
        self
    }

    /// Check if intra-bar features are enabled
    pub fn intra_bar_enabled(&self) -> bool {
        self.include_intra_bar_features
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

        // Issue #59: Push trade to history buffer for inter-bar feature computation
        // This must happen BEFORE bar processing so lookback window includes recent trades
        if let Some(ref mut history) = self.trade_history {
            history.push(&trade);
        }

        // Issue #46: If previous call triggered a breach, this trade opens the new bar.
        // This matches the batch path's defer_open semantics - the breaching trade
        // closes the current bar, and the NEXT trade opens the new bar.
        if self.defer_open {
            // Issue #68: Notify history that new bar is opening (preserves pre-bar trades)
            if let Some(ref mut history) = self.trade_history {
                history.on_bar_open(trade.timestamp);
            }
            self.current_bar_state = Some(if self.include_intra_bar_features {
                RangeBarState::new_with_trade_accumulation(&trade, self.threshold_decimal_bps)
            } else {
                RangeBarState::new(&trade, self.threshold_decimal_bps)
            });
            self.defer_open = false;
            return Ok(None);
        }

        match &mut self.current_bar_state {
            None => {
                // First trade - initialize new bar
                // Issue #68: Notify history that new bar is opening (preserves pre-bar trades)
                if let Some(ref mut history) = self.trade_history {
                    history.on_bar_open(trade.timestamp);
                }
                self.current_bar_state = Some(if self.include_intra_bar_features {
                    RangeBarState::new_with_trade_accumulation(&trade, self.threshold_decimal_bps)
                } else {
                    RangeBarState::new(&trade, self.threshold_decimal_bps)
                });
                Ok(None)
            }
            Some(bar_state) => {
                // Issue #59: Accumulate trade for intra-bar features (before breach check)
                if self.include_intra_bar_features {
                    bar_state.accumulate_trade(&trade);
                }

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

                    // Issue #59: Compute inter-bar features from lookback window
                    // Features are computed from trades BEFORE bar.open_time (no lookahead)
                    if let Some(ref mut history) = self.trade_history {
                        let inter_bar_features = history.compute_features(bar_state.bar.open_time);
                        bar_state.bar.set_inter_bar_features(&inter_bar_features);
                        // Issue #68: Notify history that bar is closing (resumes normal pruning)
                        history.on_bar_close();
                    }

                    // Issue #59: Compute intra-bar features from accumulated trades
                    if self.include_intra_bar_features {
                        let intra_bar_features =
                            compute_intra_bar_features(&bar_state.accumulated_trades);
                        bar_state.bar.set_intra_bar_features(&intra_bar_features);
                    }

                    // Move bar out instead of cloning — bar_state borrow ends after
                    // last use above (NLL), so take() is safe here.
                    let completed_bar = self.current_bar_state.take().unwrap().bar;

                    // Issue #46: Don't start new bar with breaching trade.
                    // Next trade will open the new bar via defer_open.
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

            // Issue #59: Push trade to history buffer for inter-bar feature computation
            if let Some(ref mut history) = self.trade_history {
                history.push(agg_record);
            }

            if defer_open {
                // Previous bar closed, this agg_record opens new bar
                // Issue #68: Notify history that new bar is opening (preserves pre-bar trades)
                if let Some(ref mut history) = self.trade_history {
                    history.on_bar_open(agg_record.timestamp);
                }
                current_bar = Some(if self.include_intra_bar_features {
                    RangeBarState::new_with_trade_accumulation(
                        agg_record,
                        self.threshold_decimal_bps,
                    )
                } else {
                    RangeBarState::new(agg_record, self.threshold_decimal_bps)
                });
                defer_open = false;
                continue;
            }

            match current_bar {
                None => {
                    // First bar initialization
                    // Issue #68: Notify history that new bar is opening (preserves pre-bar trades)
                    if let Some(ref mut history) = self.trade_history {
                        history.on_bar_open(agg_record.timestamp);
                    }
                    current_bar = Some(if self.include_intra_bar_features {
                        RangeBarState::new_with_trade_accumulation(
                            agg_record,
                            self.threshold_decimal_bps,
                        )
                    } else {
                        RangeBarState::new(agg_record, self.threshold_decimal_bps)
                    });
                }
                Some(ref mut bar_state) => {
                    // Issue #59: Accumulate trade for intra-bar features (before breach check)
                    if self.include_intra_bar_features {
                        bar_state.accumulate_trade(agg_record);
                    }

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

                        // Issue #59: Compute inter-bar features from lookback window
                        if let Some(ref mut history) = self.trade_history {
                            let inter_bar_features =
                                history.compute_features(bar_state.bar.open_time);
                            bar_state.bar.set_inter_bar_features(&inter_bar_features);
                            // Issue #68: Notify history that bar is closing (resumes normal pruning)
                            history.on_bar_close();
                        }

                        // Issue #59: Compute intra-bar features from accumulated trades
                        if self.include_intra_bar_features {
                            let intra_bar_features =
                                compute_intra_bar_features(&bar_state.accumulated_trades);
                            bar_state.bar.set_intra_bar_features(&intra_bar_features);
                        }

                        // Move bar out instead of cloning — bar_state borrow ends
                        // after last use above (NLL), so take() is safe here.
                        bars.push(current_bar.take().unwrap().bar);
                        defer_open = true; // Next record will open new bar
                    } else {
                        // Either no breach OR same timestamp (gate active) - normal update
                        bar_state.bar.update_with_trade(agg_record);
                    }
                }
            }
        }

        // Save current bar state for checkpoint and optionally append incomplete bar.
        // When include_incomplete=true, clone for checkpoint then consume for output.
        // When include_incomplete=false, move directly (no clone needed).
        if include_incomplete {
            self.current_bar_state = current_bar.clone();

            // Add final partial bar only if explicitly requested
            // This preserves algorithm integrity: bars should only close on threshold breach
            if let Some(mut bar_state) = current_bar {
                // Compute microstructure features for incomplete bar (Issue #34)
                bar_state.bar.compute_microstructure_features();

                // Issue #59: Compute inter-bar features from lookback window
                if let Some(ref history) = self.trade_history {
                    let inter_bar_features = history.compute_features(bar_state.bar.open_time);
                    bar_state.bar.set_inter_bar_features(&inter_bar_features);
                }

                // Issue #59: Compute intra-bar features from accumulated trades
                if self.include_intra_bar_features {
                    let intra_bar_features =
                        compute_intra_bar_features(&bar_state.accumulated_trades);
                    bar_state.bar.set_intra_bar_features(&intra_bar_features);
                }

                bars.push(bar_state.bar);
            }
        } else {
            // No incomplete bar appended — move ownership directly, no clone needed
            self.current_bar_state = current_bar;
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
        // Issue #62: Validate threshold range before restoring from checkpoint
        // Valid range: 1-100,000 dbps (0.0001% to 10%)
        const THRESHOLD_MIN: u32 = 1;
        const THRESHOLD_MAX: u32 = 100_000;
        if checkpoint.threshold_decimal_bps < THRESHOLD_MIN
            || checkpoint.threshold_decimal_bps > THRESHOLD_MAX
        {
            return Err(CheckpointError::InvalidThreshold {
                threshold: checkpoint.threshold_decimal_bps,
                min_threshold: THRESHOLD_MIN,
                max_threshold: THRESHOLD_MAX,
            });
        }

        // Validate checkpoint consistency
        if checkpoint.incomplete_bar.is_some() && checkpoint.thresholds.is_none() {
            return Err(CheckpointError::MissingThresholds);
        }

        // Restore bar state if there's an incomplete bar
        // Note: accumulated_trades is reset to empty - intra-bar features won't be
        // accurate for bars resumed from checkpoint (partial trade history lost)
        let current_bar_state = match (checkpoint.incomplete_bar, checkpoint.thresholds) {
            (Some(bar), Some((upper, lower))) => Some(RangeBarState {
                bar,
                upper_threshold: upper,
                lower_threshold: lower,
                accumulated_trades: Vec::new(), // Lost on checkpoint - features may be partial
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
            trade_history: None,               // Issue #59: Must be re-enabled after restore
            inter_bar_config: None,            // Issue #59: Must be re-enabled after restore
            include_intra_bar_features: false, // Issue #59: Must be re-enabled after restore
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
        // Issue #81: Clear bar boundary tracking at ouroboros reset.
        // Trades are preserved — still valid lookback for first bar of new segment.
        if let Some(ref mut history) = self.trade_history {
            history.reset_bar_boundaries();
        }
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

    /// Accumulated trades for intra-bar feature computation (Issue #59)
    ///
    /// When intra-bar features are enabled, trades are accumulated here
    /// during bar construction and used to compute features at bar close.
    /// Cleared when bar closes to free memory.
    pub accumulated_trades: Vec<AggTrade>,
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
            accumulated_trades: Vec::new(),
        }
    }

    /// Create new range bar state with intra-bar feature accumulation
    fn new_with_trade_accumulation(trade: &AggTrade, threshold_decimal_bps: u32) -> Self {
        let bar = RangeBar::new(trade);

        // Compute FIXED thresholds from opening price
        let (upper_threshold, lower_threshold) =
            bar.open.compute_range_thresholds(threshold_decimal_bps);

        Self {
            bar,
            upper_threshold,
            lower_threshold,
            accumulated_trades: vec![trade.clone()],
        }
    }

    /// Accumulate a trade for intra-bar feature computation
    fn accumulate_trade(&mut self, trade: &AggTrade) {
        self.accumulated_trades.push(trade.clone());
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
                base_timestamp + (i * 1000000),
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
            .add_trade(1, 1.0, 0) // Open first bar at 50000
            .add_trade(2, 1.001, 1000) // +0.1% - accumulate
            .add_trade(3, 1.003, 2000) // +0.3% - breach (>0.25%)
            .add_trade(4, 1.004, 3000) // Opens second bar
            .add_trade(5, 1.005, 4000) // Accumulate
            .add_trade(6, 1.008, 5000) // +0.4% from bar 2 open - breach
            .add_trade(7, 1.009, 6000) // Opens third bar
            .build();

        // === BATCH PATH ===
        let mut batch_processor = RangeBarProcessor::new(threshold).unwrap();
        let batch_bars = batch_processor.process_agg_trade_records(&trades).unwrap();
        let batch_incomplete = batch_processor.get_incomplete_bar();

        // === STREAMING PATH ===
        let mut stream_processor = RangeBarProcessor::new(threshold).unwrap();
        let mut stream_bars: Vec<RangeBar> = Vec::new();
        for trade in &trades {
            if let Some(bar) = stream_processor
                .process_single_trade(trade.clone())
                .unwrap()
            {
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

        for (i, (batch_bar, stream_bar)) in batch_bars.iter().zip(stream_bars.iter()).enumerate() {
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
            assert_eq!(batch_bar.low, stream_bar.low, "Bar {i}: low price mismatch");
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

    // === Memory efficiency tests (R1/R2/R3) ===

    #[test]
    fn test_bar_close_take_single_trade() {
        // R1: Verify bar close via single-trade path produces correct OHLCV after
        // clone→take optimization. Uses single_breach_sequence that triggers breach.
        let mut processor = RangeBarProcessor::new(250).unwrap();
        let trades = scenarios::single_breach_sequence(250);

        for trade in &trades[..trades.len() - 1] {
            let result = processor.process_single_trade(trade.clone()).unwrap();
            assert!(result.is_none());
        }

        // Last trade triggers breach
        let bar = processor
            .process_single_trade(trades.last().unwrap().clone())
            .unwrap()
            .expect("Should produce completed bar");

        // Verify OHLCV integrity after take() optimization
        assert_eq!(bar.open.to_string(), "50000.00000000");
        assert!(bar.high >= bar.open.max(bar.close));
        assert!(bar.low <= bar.open.min(bar.close));
        assert!(bar.volume > 0);

        // Verify processor state is clean after bar close
        assert!(processor.get_incomplete_bar().is_none());
    }

    #[test]
    fn test_bar_close_take_batch() {
        // R2: Verify batch path produces correct bars after clone→take optimization.
        // large_sequence generates enough trades to trigger multiple breaches.
        let mut processor = RangeBarProcessor::new(250).unwrap();
        let trades = scenarios::large_sequence(500);

        let bars = processor.process_agg_trade_records(&trades).unwrap();
        assert!(
            !bars.is_empty(),
            "Should produce at least one completed bar"
        );

        // Verify every bar has valid OHLCV invariants
        for bar in &bars {
            assert!(bar.high >= bar.open.max(bar.close));
            assert!(bar.low <= bar.open.min(bar.close));
            assert!(bar.volume > 0);
            assert!(bar.close_time >= bar.open_time);
        }
    }

    #[test]
    fn test_checkpoint_conditional_clone() {
        // R3: Verify checkpoint state is preserved correctly with both
        // include_incomplete=true and include_incomplete=false.
        let trades = scenarios::no_breach_sequence(250);

        // Test with include_incomplete=false (move, no clone)
        let mut processor1 = RangeBarProcessor::new(250).unwrap();
        let bars_without = processor1.process_agg_trade_records(&trades).unwrap();
        assert_eq!(bars_without.len(), 0);
        // Checkpoint should be preserved
        assert!(processor1.get_incomplete_bar().is_some());

        // Test with include_incomplete=true (clone + consume)
        let mut processor2 = RangeBarProcessor::new(250).unwrap();
        let bars_with = processor2
            .process_agg_trade_records_with_incomplete(&trades)
            .unwrap();
        assert_eq!(bars_with.len(), 1);
        // Checkpoint should ALSO be preserved (cloned before consume)
        assert!(processor2.get_incomplete_bar().is_some());

        // Both checkpoints should have identical bar content
        let cp1 = processor1.get_incomplete_bar().unwrap();
        let cp2 = processor2.get_incomplete_bar().unwrap();
        assert_eq!(cp1.open, cp2.open);
        assert_eq!(cp1.close, cp2.close);
        assert_eq!(cp1.high, cp2.high);
        assert_eq!(cp1.low, cp2.low);
    }
}
