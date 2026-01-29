//! Exness range bar builder with streaming state management
//!
//! Adapter pattern wraps RangeBarProcessor with zero core changes.
//! Maintains SpreadStats across ticks, resets on bar close.

use crate::exness::conversion::tick_to_synthetic_trade;
use crate::exness::types::{
    ExnessError, ExnessInstrument, ExnessRangeBar, ExnessTick, SpreadStats, ValidationStrictness,
};
use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::processor::RangeBarProcessor;

/// Streaming range bar builder for Exness tick data
///
/// Adapter pattern:
/// - Wraps RangeBarProcessor (zero algorithm changes)
/// - Converts ticks → synthetic trades → range bars
/// - Maintains SpreadStats for current bar (reset on close)
///
/// Volume semantics:
/// - bar.volume = 0 (Exness Raw_Spread has no volume data)
/// - buy_volume = 0, sell_volume = 0 (direction unknown for quote data)
/// - SpreadStats tracks spread dynamics as market stress signal
pub struct ExnessRangeBarBuilder {
    /// Core range bar processor (stateful)
    processor: RangeBarProcessor,

    /// Synthetic trade ID counter
    tick_counter: i64,

    /// Instrument symbol
    instrument: String,

    /// Validation strictness level
    validation_strictness: ValidationStrictness,

    /// Current bar spread statistics (reset on close)
    current_spread_stats: SpreadStats,
}

impl ExnessRangeBarBuilder {
    /// Create new builder for instrument with threshold and validation level
    ///
    /// # Arguments
    ///
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    ///   - Example: `250` → 25bps = 0.25%
    ///   - Example: `10` → 1bps = 0.01%
    ///   - Minimum: `1` → 0.1bps = 0.001%
    /// * `instrument` - Instrument symbol (e.g., "EURUSD_Raw_Spread")
    /// * `validation_strictness` - Validation level (Permissive/Strict/Paranoid)
    ///
    /// # Returns
    ///
    /// New builder instance with zero state
    ///
    /// # Breaking Change (v3.0.0)
    ///
    /// Prior to v3.0.0, `threshold_decimal_bps` was in 1bps units.
    /// **Migration**: Multiply all threshold values by 10.
    ///
    /// # Examples
    ///
    /// ```
    /// use rangebar_providers::exness::ExnessRangeBarBuilder;
    /// use rangebar_providers::exness::ValidationStrictness;
    ///
    /// let builder = ExnessRangeBarBuilder::new(
    ///     250,                          // 250 dbps = 0.25% (v3.0.0)
    ///     "EURUSD_Raw_Spread",          // Exness Raw_Spread variant
    ///     ValidationStrictness::Strict  // Default level
    /// );
    /// ```
    pub fn new(
        threshold_decimal_bps: u32,
        instrument: impl Into<String>,
        validation_strictness: ValidationStrictness,
    ) -> Result<Self, rangebar_core::processor::ProcessingError> {
        Ok(Self {
            processor: RangeBarProcessor::new(threshold_decimal_bps)?,
            tick_counter: 0,
            instrument: instrument.into(),
            validation_strictness,
            current_spread_stats: SpreadStats::new(),
        })
    }

    /// Create builder for a specific instrument (type-safe API)
    ///
    /// Preferred over `new()` for type safety and IDE autocomplete.
    ///
    /// # Arguments
    ///
    /// * `instrument` - Exness instrument enum
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    /// * `validation_strictness` - Validation level (Permissive/Strict/Paranoid)
    ///
    /// # Example
    ///
    /// ```
    /// use rangebar_providers::exness::{ExnessRangeBarBuilder, ExnessInstrument, ValidationStrictness};
    ///
    /// let builder = ExnessRangeBarBuilder::for_instrument(
    ///     ExnessInstrument::XAUUSD,
    ///     50,                           // 50 dbps = 0.05%
    ///     ValidationStrictness::Strict,
    /// ).unwrap();
    /// ```
    pub fn for_instrument(
        instrument: ExnessInstrument,
        threshold_decimal_bps: u32,
        validation_strictness: ValidationStrictness,
    ) -> Result<Self, rangebar_core::processor::ProcessingError> {
        Self::new(
            threshold_decimal_bps,
            instrument.raw_spread_symbol(),
            validation_strictness,
        )
    }

    /// Process single tick, returning completed bar if threshold breached
    ///
    /// State management:
    /// 1. Update current bar's spread stats (accumulate)
    /// 2. Convert tick → synthetic trade
    /// 3. Process through core processor
    /// 4. If bar closes: wrap with spread stats, reset stats for next bar
    ///
    /// # Arguments
    ///
    /// * `tick` - Exness tick (Bid/Ask quote data)
    ///
    /// # Returns
    ///
    /// - `Ok(Some(bar))` - Bar completed (threshold breached)
    /// - `Ok(None)` - Tick processed, bar accumulating
    /// - `Err(...)` - Validation or processing error (raise immediately)
    ///
    /// # Error Policy
    ///
    /// Fail-fast: All errors propagated immediately to caller.
    /// No fallbacks, no skipping, no error rate thresholds.
    pub fn process_tick(
        &mut self,
        tick: &ExnessTick,
    ) -> Result<Option<ExnessRangeBar>, ExnessError> {
        // 1. Update spread stats (before conversion, accumulate for current bar)
        self.current_spread_stats.update(tick);

        // 2. Convert tick to synthetic trade (raises on error)
        let synthetic_trade = tick_to_synthetic_trade(
            tick,
            &self.instrument,
            self.tick_counter,
            self.validation_strictness,
        )?;
        self.tick_counter += 1;

        // 3. Process through core processor (raises on error)
        let maybe_bar = self.processor.process_single_trade(synthetic_trade)?;

        // 4. If bar closed, wrap with spread stats and reset
        if let Some(mut base) = maybe_bar {
            // Zero out buy/sell volume (Exness has no volume data)
            // Synthetic trades use mid-price, but direction is unknown
            base.buy_volume = FixedPoint(0);
            base.sell_volume = FixedPoint(0);
            base.buy_trade_count = 0;
            base.sell_trade_count = 0;
            base.buy_turnover = 0;
            base.sell_turnover = 0;

            let completed_bar = ExnessRangeBar {
                base,
                spread_stats: self.current_spread_stats.clone(),
            };

            // Reset spread stats for next bar (per-bar semantics)
            // Issue #46: Breaching tick belongs to closing bar only.
            // Next tick will open the new bar and start accumulating spread stats.
            self.current_spread_stats = SpreadStats::new();

            Ok(Some(completed_bar))
        } else {
            Ok(None)
        }
    }

    /// Get incomplete bar if exists (for final bar at stream end)
    ///
    /// Returns current bar state with accumulated spread stats.
    /// Useful for retrieving partial bar when stream ends.
    ///
    /// # Returns
    ///
    /// `Some(ExnessRangeBar)` if bar in progress, `None` if no active bar
    pub fn get_incomplete_bar(&self) -> Option<ExnessRangeBar> {
        self.processor.get_incomplete_bar().map(|mut base| {
            // Zero out buy/sell volume (Exness has no volume data)
            base.buy_volume = FixedPoint(0);
            base.sell_volume = FixedPoint(0);
            base.buy_trade_count = 0;
            base.sell_trade_count = 0;
            base.buy_turnover = 0;
            base.sell_turnover = 0;

            ExnessRangeBar {
                base,
                spread_stats: self.current_spread_stats.clone(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_streaming_state() {
        let mut builder = ExnessRangeBarBuilder::new(
            250, // 250 dbps = 0.25%
            "EURUSD_Raw_Spread",
            ValidationStrictness::Strict,
        )
        .unwrap();

        // First tick - initializes bar
        let tick1 = ExnessTick {
            bid: 1.0800,
            ask: 1.0815,
            timestamp_ms: 1_600_000_000_000,
        };

        let result = builder.process_tick(&tick1).unwrap();
        assert!(result.is_none()); // No bar closed yet

        // Check incomplete bar exists
        let incomplete = builder.get_incomplete_bar();
        assert!(incomplete.is_some());
        let incomplete_bar = incomplete.unwrap();
        assert_eq!(incomplete_bar.spread_stats.tick_count, 1);
    }

    #[test]
    fn test_spread_stats_reset_on_bar_close() {
        let mut builder =
            ExnessRangeBarBuilder::new(250, "EURUSD_Raw_Spread", ValidationStrictness::Strict)
                .unwrap();

        // First tick at 1.0800 mid
        let tick1 = ExnessTick {
            bid: 1.0792,
            ask: 1.0808,
            timestamp_ms: 1_600_000_000_000,
        };
        builder.process_tick(&tick1).unwrap();

        // Second tick at 1.0800 mid (no breach)
        let tick2 = ExnessTick {
            bid: 1.0793,
            ask: 1.0807,
            timestamp_ms: 1_600_001_000_000,
        };
        builder.process_tick(&tick2).unwrap();

        // Third tick breaches +0.25% threshold (forces bar close)
        // Mid-price needs to be > 1.0800 * 1.0025 = 1.0827
        let tick3 = ExnessTick {
            bid: 1.0825,
            ask: 1.0835, // mid = 1.0830 (breach!)
            timestamp_ms: 1_600_002_000_000,
        };

        let maybe_bar = builder.process_tick(&tick3).unwrap();
        assert!(maybe_bar.is_some());

        let completed_bar = maybe_bar.unwrap();

        // Verify spread stats were captured (all 3 ticks including breach)
        assert_eq!(completed_bar.spread_stats.tick_count, 3);

        // Issue #46: After breach, no incomplete bar until next tick arrives.
        // The breaching tick closes the current bar; the NEXT tick opens the new bar.
        assert!(
            builder.get_incomplete_bar().is_none(),
            "No incomplete bar after breach - next tick opens new bar (Issue #46)"
        );
    }

    #[test]
    fn test_validation_error_propagation() {
        let mut builder =
            ExnessRangeBarBuilder::new(250, "EURUSD_Raw_Spread", ValidationStrictness::Strict)
                .unwrap();

        // Crossed market tick
        let bad_tick = ExnessTick {
            bid: 1.0815,
            ask: 1.0800, // bid > ask
            timestamp_ms: 1_600_000_000_000,
        };

        let result = builder.process_tick(&bad_tick);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_volume_semantics() {
        let mut builder =
            ExnessRangeBarBuilder::new(250, "EURUSD_Raw_Spread", ValidationStrictness::Strict)
                .unwrap();

        let tick1 = ExnessTick {
            bid: 1.0800,
            ask: 1.0810,
            timestamp_ms: 1_600_000_000_000,
        };

        let tick2 = ExnessTick {
            bid: 1.0828,
            ask: 1.0838, // Breach
            timestamp_ms: 1_600_001_000_000,
        };

        builder.process_tick(&tick1).unwrap();
        let bar = builder.process_tick(&tick2).unwrap().unwrap();

        // Exness has no volume data
        assert_eq!(bar.base.volume.0, 0);
        assert_eq!(bar.base.buy_volume.0, 0);
        assert_eq!(bar.base.sell_volume.0, 0);
    }
}
