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
                // Issue #88: i128 volume accumulators
                volume: trade.volume.0 as i128,
                turnover: trade_turnover,
                individual_trade_count: 1,
                agg_record_count: 1,
                first_trade_id: trade.first_trade_id,
                last_trade_id: trade.last_trade_id,
                // Issue #72: Track aggregate trade IDs
                first_agg_trade_id: trade.agg_trade_id,
                last_agg_trade_id: trade.agg_trade_id,
                // Market microstructure fields (Issue #88: i128)
                buy_volume: if trade.is_buyer_maker {
                    0i128
                } else {
                    trade.volume.0 as i128
                },
                sell_volume: if trade.is_buyer_maker {
                    trade.volume.0 as i128
                } else {
                    0i128
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
                // Issue #72: Aggregate trade ID tracking
                first_agg_trade_id: completed_bar.first_agg_trade_id,
                last_agg_trade_id: completed_bar.last_agg_trade_id,
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

                // Inter-bar features (Issue #59) - not computed in ExportRangeBarProcessor
                lookback_trade_count: None,
                lookback_ofi: None,
                lookback_duration_us: None,
                lookback_intensity: None,
                lookback_vwap_raw: None,
                lookback_vwap_position: None,
                lookback_count_imbalance: None,
                lookback_kyle_lambda: None,
                lookback_burstiness: None,
                lookback_volume_skew: None,
                lookback_volume_kurt: None,
                lookback_price_range: None,
                lookback_kaufman_er: None,
                lookback_garman_klass_vol: None,
                lookback_hurst: None,
                lookback_permutation_entropy: None,

                // Intra-bar features (Issue #59) - not computed in ExportRangeBarProcessor
                intra_bull_epoch_density: None,
                intra_bear_epoch_density: None,
                intra_bull_excess_gain: None,
                intra_bear_excess_gain: None,
                intra_bull_cv: None,
                intra_bear_cv: None,
                intra_max_drawdown: None,
                intra_max_runup: None,
                intra_trade_count: None,
                intra_ofi: None,
                intra_duration_us: None,
                intra_intensity: None,
                intra_vwap_position: None,
                intra_count_imbalance: None,
                intra_kyle_lambda: None,
                intra_burstiness: None,
                intra_volume_skew: None,
                intra_volume_kurt: None,
                intra_kaufman_er: None,
                intra_garman_klass_vol: None,
                intra_hurst: None,
                intra_permutation_entropy: None,
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

                // Inter-bar features (Issue #59) - not computed in ExportRangeBarProcessor
                lookback_trade_count: None,
                lookback_ofi: None,
                lookback_duration_us: None,
                lookback_intensity: None,
                lookback_vwap_raw: None,
                lookback_vwap_position: None,
                lookback_count_imbalance: None,
                lookback_kyle_lambda: None,
                lookback_burstiness: None,
                lookback_volume_skew: None,
                lookback_volume_kurt: None,
                lookback_price_range: None,
                lookback_kaufman_er: None,
                lookback_garman_klass_vol: None,
                lookback_hurst: None,
                lookback_permutation_entropy: None,

                // Intra-bar features (Issue #59) - not computed in ExportRangeBarProcessor
                intra_bull_epoch_density: None,
                intra_bear_epoch_density: None,
                intra_bull_excess_gain: None,
                intra_bear_excess_gain: None,
                intra_bull_cv: None,
                intra_bear_cv: None,
                intra_max_drawdown: None,
                intra_max_runup: None,
                intra_trade_count: None,
                intra_ofi: None,
                intra_duration_us: None,
                intra_intensity: None,
                intra_vwap_position: None,
                intra_count_imbalance: None,
                intra_kyle_lambda: None,
                intra_burstiness: None,
                intra_volume_skew: None,
                intra_volume_kurt: None,
                intra_kaufman_er: None,
                intra_garman_klass_vol: None,
                intra_hurst: None,
                intra_permutation_entropy: None,
            };
            // Compute microstructure features for incomplete bar (Issue #25)
            bar.compute_microstructure_features();
            bar
        })
    }
}
