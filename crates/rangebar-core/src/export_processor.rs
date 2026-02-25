//! Export-oriented range bar processor
//! Extracted from processor.rs (Phase 2d refactoring)

use crate::errors::ProcessingError;
use crate::fixed_point::FixedPoint;
use crate::types::{AggTrade, RangeBar};

/// Internal state for range bar construction with fixed-point precision
#[derive(Debug, Clone)]
pub(crate) struct InternalRangeBar {
    open_time: i64,
    close_time: i64,
    open: FixedPoint,
    high: FixedPoint,
    low: FixedPoint,
    close: FixedPoint,
    // Issue #88: i128 volume accumulators to prevent FixedPoint(i64) overflow
    // on high-token-count symbols like SHIBUSDT
    volume: i128,
    turnover: i128,
    individual_trade_count: i64,
    agg_record_count: u32,
    first_trade_id: i64,
    last_trade_id: i64,
    /// First aggregate trade ID in this range bar (Issue #72)
    first_agg_trade_id: i64,
    /// Last aggregate trade ID in this range bar (Issue #72)
    last_agg_trade_id: i64,
    /// Volume from buy-side trades (is_buyer_maker = false)
    // Issue #88: i128 to prevent overflow
    buy_volume: i128,
    /// Volume from sell-side trades (is_buyer_maker = true)
    // Issue #88: i128 to prevent overflow
    sell_volume: i128,
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
    /// Issue #96 Task #71: Reuse pool for completed_bars vec (streaming hot path)
    completed_bars_pool: Option<Vec<RangeBar>>,
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
    ///   - Example: `250` -> 25bps = 0.25%
    ///   - Example: `10` -> 1bps = 0.01%
    ///   - Minimum: `1` -> 0.1bps = 0.001%
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
            completed_bars_pool: None,
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
            // Issue #96: Use integer turnover (matches main processor) — eliminates 2 f64 conversions
            let trade_turnover = trade.turnover();
            let vol = trade.volume.0 as i128;

            // Single branch for buy/sell classification
            let (buy_vol, sell_vol, buy_count, sell_count, buy_turn, sell_turn) =
                if trade.is_buyer_maker {
                    (0i128, vol, 0i64, 1i64, 0i128, trade_turnover)
                } else {
                    (vol, 0i128, 1i64, 0i64, trade_turnover, 0i128)
                };

            self.current_bar = Some(InternalRangeBar {
                open_time: trade.timestamp,
                close_time: trade.timestamp,
                open: trade.price,
                high: trade.price,
                low: trade.price,
                close: trade.price,
                // Issue #88: i128 volume accumulators
                volume: vol,
                turnover: trade_turnover,
                individual_trade_count: 1,
                agg_record_count: 1,
                first_trade_id: trade.first_trade_id,
                last_trade_id: trade.last_trade_id,
                // Issue #72: Track aggregate trade IDs
                first_agg_trade_id: trade.agg_trade_id,
                last_agg_trade_id: trade.agg_trade_id,
                // Market microstructure fields (Issue #88: i128)
                buy_volume: buy_vol,
                sell_volume: sell_vol,
                buy_trade_count: buy_count,
                sell_trade_count: sell_count,
                vwap: trade.price,
                buy_turnover: buy_turn,
                sell_turnover: sell_turn,
            });
            return;
        }

        // Process existing bar - work with reference
        // SAFETY: current_bar guaranteed Some - early return above if None
        let bar = self.current_bar.as_mut().unwrap();
        // Issue #96: Use integer turnover (matches main processor) — eliminates 2 f64 conversions
        let trade_turnover = trade.turnover();

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
        bar.volume += trade.volume.0 as i128; // Issue #88: i128 accumulator
        bar.turnover += trade_turnover;
        bar.individual_trade_count += 1;
        bar.agg_record_count += 1;
        bar.last_trade_id = trade.last_trade_id;
        bar.last_agg_trade_id = trade.agg_trade_id; // Issue #72

        // Update high/low
        if price_val > bar.high.0 {
            bar.high = trade.price;
        }
        if price_val < bar.low.0 {
            bar.low = trade.price;
        }

        // Update market microstructure
        if trade.is_buyer_maker {
            bar.sell_volume += trade.volume.0 as i128; // Issue #88: i128 accumulator
            bar.sell_turnover += trade_turnover;
            bar.sell_trade_count += 1;
        } else {
            bar.buy_volume += trade.volume.0 as i128; // Issue #88: i128 accumulator
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

            // Convert to export format — uses ..Default::default() for all
            // microstructure/inter-bar/intra-bar fields (0/0.0/None)
            let mut export_bar = RangeBar {
                open_time: completed_bar.open_time,
                close_time: completed_bar.close_time,
                open: completed_bar.open,
                high: completed_bar.high,
                low: completed_bar.low,
                close: completed_bar.close,
                volume: completed_bar.volume,
                turnover: completed_bar.turnover,
                individual_trade_count: completed_bar.individual_trade_count as u32,
                agg_record_count: completed_bar.agg_record_count,
                first_trade_id: completed_bar.first_trade_id,
                last_trade_id: completed_bar.last_trade_id,
                first_agg_trade_id: completed_bar.first_agg_trade_id, // Issue #72
                last_agg_trade_id: completed_bar.last_agg_trade_id,
                buy_volume: completed_bar.buy_volume,
                sell_volume: completed_bar.sell_volume,
                buy_trade_count: completed_bar.buy_trade_count as u32,
                sell_trade_count: completed_bar.sell_trade_count as u32,
                vwap: completed_bar.vwap,
                buy_turnover: completed_bar.buy_turnover,
                sell_turnover: completed_bar.sell_turnover,
                ..Default::default() // Issue #25/#59: microstructure computed below; inter/intra-bar not used
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
        // Issue #96 Task #71: Vec reuse pool to reduce allocation overhead on hot path
        let mut result = if let Some(mut pool_vec) = self.completed_bars_pool.take() {
            // Reuse pool vec for next batch
            pool_vec.clear();
            pool_vec
        } else {
            // First call or pool was None
            Vec::new()
        };

        // Swap current completed bars with pool vec
        std::mem::swap(&mut result, &mut self.completed_bars);

        // Store the now-empty completed_bars in pool for next cycle
        self.completed_bars_pool = Some(std::mem::take(&mut self.completed_bars));

        result
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
                first_agg_trade_id: incomplete.first_agg_trade_id,
                last_agg_trade_id: incomplete.last_agg_trade_id,
                data_source: crate::types::DataSource::default(),

                // Market microstructure fields
                buy_volume: incomplete.buy_volume,
                sell_volume: incomplete.sell_volume,
                buy_trade_count: incomplete.buy_trade_count as u32,
                sell_trade_count: incomplete.sell_trade_count as u32,
                vwap: incomplete.vwap,
                buy_turnover: incomplete.buy_turnover,
                sell_turnover: incomplete.sell_turnover,

                // All microstructure, inter-bar, and intra-bar features default to 0/None
                ..Default::default()
            };
            // Compute microstructure features for incomplete bar (Issue #25)
            bar.compute_microstructure_features();
            bar
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_test_agg_trade_with_range;

    /// Helper: create a buy trade at given price/time
    fn buy_trade(id: i64, price: &str, vol: &str, ts: i64) -> AggTrade {
        create_test_agg_trade_with_range(id, price, vol, ts, id * 10, id * 10, false)
    }

    /// Helper: create a sell trade at given price/time
    fn sell_trade(id: i64, price: &str, vol: &str, ts: i64) -> AggTrade {
        create_test_agg_trade_with_range(id, price, vol, ts, id * 10, id * 10, true)
    }

    #[test]
    fn test_new_valid_threshold() {
        let proc = ExportRangeBarProcessor::new(250);
        assert!(proc.is_ok());
    }

    #[test]
    fn test_new_invalid_threshold_zero() {
        match ExportRangeBarProcessor::new(0) {
            Err(ProcessingError::InvalidThreshold {
                threshold_decimal_bps: 0,
            }) => {}
            Err(e) => panic!("Expected InvalidThreshold(0), got error: {e}"),
            Ok(_) => panic!("Expected error for threshold 0"),
        }
    }

    #[test]
    fn test_new_invalid_threshold_too_high() {
        let proc = ExportRangeBarProcessor::new(100_001);
        assert!(proc.is_err());
    }

    #[test]
    fn test_new_boundary_thresholds() {
        // Minimum valid
        assert!(ExportRangeBarProcessor::new(1).is_ok());
        // Maximum valid
        assert!(ExportRangeBarProcessor::new(100_000).is_ok());
    }

    #[test]
    fn test_with_options_timestamp_gating() {
        let proc = ExportRangeBarProcessor::with_options(250, false);
        assert!(proc.is_ok());
    }

    #[test]
    fn test_single_trade_no_bar_completion() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![buy_trade(1, "100.0", "1.0", 1000)];
        proc.process_trades_continuously(&trades);

        let completed = proc.get_all_completed_bars();
        assert_eq!(completed.len(), 0, "Single trade should not complete a bar");

        let incomplete = proc.get_incomplete_bar();
        assert!(incomplete.is_some(), "Should have an incomplete bar");
        let bar = incomplete.unwrap();
        assert_eq!(bar.open, FixedPoint::from_str("100.0").unwrap());
        assert_eq!(bar.close, FixedPoint::from_str("100.0").unwrap());
    }

    #[test]
    fn test_breach_completes_bar() {
        // 250 dbps = 0.25%. At open=100.0, upper=100.25, lower=99.75
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.10", "1.0", 2000),
            buy_trade(3, "100.25", "1.0", 3000), // Breach: >= upper threshold
        ];
        proc.process_trades_continuously(&trades);

        let completed = proc.get_all_completed_bars();
        assert_eq!(completed.len(), 1, "Breach should complete one bar");

        let bar = &completed[0];
        assert_eq!(bar.open, FixedPoint::from_str("100.0").unwrap());
        assert_eq!(bar.close, FixedPoint::from_str("100.25").unwrap());
        assert_eq!(bar.high, FixedPoint::from_str("100.25").unwrap());
        assert_eq!(bar.low, FixedPoint::from_str("100.0").unwrap());
    }

    #[test]
    fn test_defer_open_semantics() {
        // Issue #46: Breaching trade should NOT open next bar
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.25", "1.0", 2000), // Breach → completes bar 1
            buy_trade(3, "100.50", "1.0", 3000),  // Opens bar 2 (defer_open)
        ];
        proc.process_trades_continuously(&trades);

        let completed = proc.get_all_completed_bars();
        assert_eq!(completed.len(), 1);
        // Bar 1 was opened by trade 1, closed by trade 2
        assert_eq!(completed[0].open, FixedPoint::from_str("100.0").unwrap());
        assert_eq!(completed[0].close, FixedPoint::from_str("100.25").unwrap());

        // Incomplete bar should be opened by trade 3 (not trade 2)
        let incomplete = proc.get_incomplete_bar();
        assert!(incomplete.is_some());
        let bar2 = incomplete.unwrap();
        assert_eq!(
            bar2.open,
            FixedPoint::from_str("100.50").unwrap(),
            "Bar 2 should open at trade 3's price, not the breaching trade"
        );
    }

    #[test]
    fn test_timestamp_gate_prevents_same_ts_close() {
        // Issue #36: Bar cannot close on same timestamp as it opened
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.30", "1.0", 1000), // Same ts as open, breach but gated
        ];
        proc.process_trades_continuously(&trades);

        let completed = proc.get_all_completed_bars();
        assert_eq!(
            completed.len(),
            0,
            "Timestamp gate should prevent close on same ms"
        );
    }

    #[test]
    fn test_timestamp_gate_disabled() {
        // With timestamp gating off, same-ts breach closes the bar
        let mut proc = ExportRangeBarProcessor::with_options(250, false).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.30", "1.0", 1000), // Same ts, breach allowed
        ];
        proc.process_trades_continuously(&trades);

        let completed = proc.get_all_completed_bars();
        assert_eq!(
            completed.len(),
            1,
            "With gating disabled, same-ts breach should close"
        );
    }

    #[test]
    fn test_get_all_completed_bars_drains() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.25", "1.0", 2000), // Breach
        ];
        proc.process_trades_continuously(&trades);

        let bars1 = proc.get_all_completed_bars();
        assert_eq!(bars1.len(), 1);

        // Second call should return empty (drained)
        let bars2 = proc.get_all_completed_bars();
        assert_eq!(bars2.len(), 0, "get_all_completed_bars should drain buffer");
    }

    #[test]
    fn test_vec_reuse_pool() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();

        // First batch: produce a bar
        proc.process_trades_continuously(&[
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.25", "1.0", 2000),
        ]);
        let _bars1 = proc.get_all_completed_bars();

        // Second batch: produce another bar — pool should be reused
        proc.process_trades_continuously(&[
            sell_trade(3, "100.50", "1.0", 3000),
            sell_trade(4, "100.75", "1.0", 4000),
            sell_trade(5, "100.24", "1.0", 5000), // Breach lower
        ]);
        let bars2 = proc.get_all_completed_bars();
        assert_eq!(bars2.len(), 1);
    }

    #[test]
    fn test_buy_sell_volume_segregation() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "2.0", 1000),   // Buy: 2.0
            sell_trade(2, "100.05", "3.0", 2000),  // Sell: 3.0
            buy_trade(3, "100.25", "1.0", 3000),   // Buy: 1.0, breach
        ];
        proc.process_trades_continuously(&trades);

        let bars = proc.get_all_completed_bars();
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];

        let buy_vol = bar.buy_volume;
        let sell_vol = bar.sell_volume;
        // Buy trades: 2.0 + 1.0 = 3.0, Sell trades: 3.0
        assert_eq!(buy_vol, 300_000_000, "Buy volume should be 3.0 in FixedPoint i128");
        assert_eq!(sell_vol, 300_000_000, "Sell volume should be 3.0 in FixedPoint i128");
    }

    #[test]
    fn test_trade_id_tracking() {
        // Issue #72: Verify first/last agg trade ID tracking
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            create_test_agg_trade_with_range(100, "100.0", "1.0", 1000, 1000, 1005, false),
            create_test_agg_trade_with_range(101, "100.10", "1.0", 2000, 1006, 1010, true),
            create_test_agg_trade_with_range(102, "100.25", "1.0", 3000, 1011, 1015, false), // Breach
        ];
        proc.process_trades_continuously(&trades);

        let bars = proc.get_all_completed_bars();
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.first_agg_trade_id, 100);
        assert_eq!(bar.last_agg_trade_id, 102);
        assert_eq!(bar.first_trade_id, 1000);
        assert_eq!(bar.last_trade_id, 1015);
    }

    #[test]
    fn test_microstructure_features_computed() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            buy_trade(1, "100.0", "5.0", 1000),
            sell_trade(2, "100.10", "3.0", 2000),
            buy_trade(3, "100.25", "2.0", 3000), // Breach
        ];
        proc.process_trades_continuously(&trades);

        let bars = proc.get_all_completed_bars();
        let bar = &bars[0];

        // OFI should be computed (buy_vol > sell_vol → positive)
        // buy = 5.0 + 2.0 = 7.0, sell = 3.0, ofi = (7-3)/10 = 0.4
        assert!(bar.ofi != 0.0, "OFI should be computed");
        assert!(bar.trade_intensity > 0.0, "Trade intensity should be > 0");
        assert!(bar.volume_per_trade > 0.0, "Volume per trade should be > 0");
    }

    #[test]
    fn test_incomplete_bar_has_microstructure() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        proc.process_trades_continuously(&[
            buy_trade(1, "100.0", "5.0", 1000),
            sell_trade(2, "100.10", "3.0", 2000),
        ]);

        let incomplete = proc.get_incomplete_bar().unwrap();
        // Microstructure should be computed on incomplete bars too
        assert!(
            incomplete.volume_per_trade > 0.0,
            "Incomplete bar should have microstructure features"
        );
    }

    #[test]
    fn test_multiple_bars_sequence() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        // Generate enough trades for 2 complete bars
        let trades = vec![
            buy_trade(1, "100.0", "1.0", 1000),
            buy_trade(2, "100.25", "1.0", 2000),   // Breach → bar 1
            buy_trade(3, "100.50", "1.0", 3000),    // Opens bar 2
            buy_trade(4, "100.76", "1.0", 4000),    // Breach → bar 2 (100.50 * 1.0025 = 100.75125)
        ];
        proc.process_trades_continuously(&trades);

        let bars = proc.get_all_completed_bars();
        assert_eq!(bars.len(), 2, "Should produce 2 complete bars");
        assert_eq!(bars[0].open, FixedPoint::from_str("100.0").unwrap());
        assert_eq!(bars[1].open, FixedPoint::from_str("100.50").unwrap());
    }

    #[test]
    fn test_downward_breach() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        let trades = vec![
            sell_trade(1, "100.0", "1.0", 1000),
            sell_trade(2, "99.75", "1.0", 2000), // Breach lower: <= 99.75
        ];
        proc.process_trades_continuously(&trades);

        let bars = proc.get_all_completed_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close, FixedPoint::from_str("99.75").unwrap());
    }

    #[test]
    fn test_empty_trades_no_op() {
        let mut proc = ExportRangeBarProcessor::new(250).unwrap();
        proc.process_trades_continuously(&[]);
        assert_eq!(proc.get_all_completed_bars().len(), 0);
        assert!(proc.get_incomplete_bar().is_none());
    }
}
