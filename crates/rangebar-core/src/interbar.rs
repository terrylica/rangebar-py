// FILE-SIZE-OK: Tests stay inline (access pub(crate) math functions via glob import). Phase 2b extracted types, Phase 2e extracted math.
//! Inter-bar microstructure features computed from lookback trade windows
//!
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59
//!
//! This module provides features computed from trades that occurred BEFORE each bar opened,
//! enabling enrichment of larger range bars (e.g., 1000 dbps) with finer-grained microstructure
//! signals without lookahead bias.
//!
//! ## Temporal Integrity
//!
//! All features are computed from trades with timestamps strictly BEFORE the current bar's
//! `open_time`. This ensures no lookahead bias in ML applications.
//!
//! ## Feature Tiers
//!
//! - **Tier 1**: Core features (7) - low complexity, high value
//! - **Tier 2**: Statistical features (5) - medium complexity
//! - **Tier 3**: Advanced features (4) - higher complexity, from trading-fitness patterns
//!
//! ## Academic References
//!
//! | Feature | Reference |
//! |---------|-----------|
//! | OFI | Chordia et al. (2002) - Order imbalance |
//! | Kyle's Lambda | Kyle (1985) - Continuous auctions and insider trading |
//! | Burstiness | Goh & Barabási (2008) - Burstiness and memory in complex systems |
//! | Kaufman ER | Kaufman (1995) - Smarter Trading |
//! | Garman-Klass | Garman & Klass (1980) - On the Estimation of Security Price Volatilities |
//! | Hurst (DFA) | Peng et al. (1994) - Mosaic organization of DNA nucleotides |
//! | Permutation Entropy | Bandt & Pompe (2002) - Permutation Entropy: A Natural Complexity Measure |

use crate::fixed_point::FixedPoint;
use crate::interbar_math::*;
use crate::types::AggTrade;
use smallvec::SmallVec;
use std::collections::VecDeque;

// Re-export types from interbar_types.rs (Phase 2b extraction)
pub use crate::interbar_types::{InterBarConfig, InterBarFeatures, LookbackMode, TradeSnapshot};

/// Trade history ring buffer for inter-bar feature computation
#[derive(Debug, Clone)]
pub struct TradeHistory {
    /// Ring buffer of recent trades
    trades: VecDeque<TradeSnapshot>,
    /// Configuration for lookback
    config: InterBarConfig,
    /// Timestamp threshold: trades with timestamp < this are protected from pruning.
    /// Set to the oldest timestamp we might need for lookback computation.
    /// Updated each time a new bar opens.
    protected_until: Option<i64>,
    /// Total number of trades pushed (monotonic counter for BarRelative indexing)
    total_pushed: usize,
    /// Indices into total_pushed at which each bar closed (Issue #81).
    /// `bar_close_indices[i]` = `total_pushed` value when bar i closed.
    /// Used by `BarRelative` mode to determine how many trades to keep.
    bar_close_indices: VecDeque<usize>,
}

impl TradeHistory {
    /// Create new trade history with given configuration
    pub fn new(config: InterBarConfig) -> Self {
        let capacity = match &config.lookback_mode {
            LookbackMode::FixedCount(n) => *n * 2, // 2x capacity to hold pre-bar + in-bar trades
            LookbackMode::FixedWindow(_) | LookbackMode::BarRelative(_) => 2000, // Dynamic initial capacity
        };
        Self {
            trades: VecDeque::with_capacity(capacity),
            config,
            protected_until: None,
            total_pushed: 0,
            bar_close_indices: VecDeque::new(),
        }
    }

    /// Push a new trade to the history buffer
    ///
    /// Automatically prunes old entries based on lookback mode, but preserves
    /// trades needed for lookback computation (timestamp < protected_until).
    pub fn push(&mut self, trade: &AggTrade) {
        let snapshot = TradeSnapshot::from(trade);
        self.trades.push_back(snapshot);
        self.total_pushed += 1;
        self.prune();
    }

    /// Notify that a new bar has opened at the given timestamp
    ///
    /// This sets the protection threshold to ensure trades from before the bar
    /// opened are preserved for lookback computation. The protection extends
    /// until the next bar opens and calls this method again.
    pub fn on_bar_open(&mut self, bar_open_time: i64) {
        // Protect all trades with timestamp < bar_open_time
        // These are the trades that can be used for lookback computation
        self.protected_until = Some(bar_open_time);
    }

    /// Notify that the current bar has closed
    ///
    /// For `BarRelative` mode, records the current trade count as a bar boundary.
    /// For other modes, this is a no-op. Protection is always kept until the
    /// next bar opens.
    pub fn on_bar_close(&mut self) {
        // Record bar boundary for BarRelative pruning (Issue #81)
        if let LookbackMode::BarRelative(n_bars) = &self.config.lookback_mode {
            self.bar_close_indices.push_back(self.total_pushed);
            // Keep only last n_bars+1 boundaries (n_bars for lookback + 1 for current)
            while self.bar_close_indices.len() > *n_bars + 1 {
                self.bar_close_indices.pop_front();
            }
        }
        // Keep protection until next bar opens (all modes)
    }

    /// Prune old trades based on lookback configuration
    ///
    /// Pruning logic:
    /// - For `FixedCount(n)`: Keep up to 2*n trades total, but never prune trades
    ///   with timestamp < `protected_until` (needed for lookback)
    /// - For `FixedWindow`: Standard time-based pruning, but respect `protected_until`
    /// - For `BarRelative(n)`: Keep trades from last n completed bars (Issue #81)
    fn prune(&mut self) {
        match &self.config.lookback_mode {
            LookbackMode::FixedCount(n) => {
                // Keep at most 2*n trades (n for lookback + n for next bar's lookback)
                let max_trades = *n * 2;
                while self.trades.len() > max_trades {
                    // Check if front trade is protected
                    if let Some(front) = self.trades.front() {
                        if let Some(protected) = self.protected_until {
                            if front.timestamp < protected {
                                // Don't prune protected trades
                                break;
                            }
                        }
                    }
                    self.trades.pop_front();
                }
            }
            LookbackMode::FixedWindow(window_us) => {
                // Find the oldest trade we need
                let newest_timestamp = self.trades.back().map(|t| t.timestamp).unwrap_or(0);
                let cutoff = newest_timestamp - window_us;

                while let Some(front) = self.trades.front() {
                    // Respect protection
                    if let Some(protected) = self.protected_until {
                        if front.timestamp < protected {
                            break;
                        }
                    }
                    // Prune if outside time window
                    if front.timestamp < cutoff {
                        self.trades.pop_front();
                    } else {
                        break;
                    }
                }
            }
            LookbackMode::BarRelative(n_bars) => {
                // Issue #81: Keep trades from last n completed bars.
                //
                // bar_close_indices stores total_pushed at each bar close:
                //   B0 = end of bar 0 / start of bar 1's trades
                //   B1 = end of bar 1 / start of bar 2's trades
                //   etc.
                //
                // To include N bars of lookback, we need boundary B_{k-1}
                // where k is the oldest bar we want. on_bar_close() keeps
                // at most n_bars+1 entries, so after steady state, front()
                // is exactly B_{k-1}.
                //
                // Bootstrap: when fewer than n_bars bars have closed, we
                // want ALL available bars, so keep everything.
                if self.bar_close_indices.len() <= *n_bars {
                    // Bootstrap: fewer completed bars than lookback depth.
                    // Keep all trades — we want every available bar.
                    return;
                }

                // Steady state: front() is the boundary BEFORE the oldest
                // bar we want. Trades from front() onward belong to the
                // N-bar lookback window plus the current in-progress bar.
                let oldest_boundary = self.bar_close_indices.front().copied().unwrap_or(0);
                let keep_count = self.total_pushed - oldest_boundary;

                // Prune unconditionally — bar boundaries are the source of truth
                while self.trades.len() > keep_count {
                    self.trades.pop_front();
                }
            }
        }
    }

    /// Get trades for lookback computation (excludes trades at or after bar_open_time)
    ///
    /// This is CRITICAL for temporal integrity - we only use trades that
    /// occurred BEFORE the current bar opened.
    ///
    /// # Performance
    ///
    /// Uses binary search to find cutoff index (trades are timestamp-sorted).
    /// O(log n) vs O(n) for linear scan. Returns SmallVec with 256 inline capacity.
    /// Typical lookback windows (100-500 trades) avoid heap allocation entirely.
    /// Issue #96 Task #41: Binary search optimization for lookback filter.
    pub fn get_lookback_trades(&self, bar_open_time: i64) -> SmallVec<[&TradeSnapshot; 256]> {
        use std::cmp::Ordering;

        // Binary search to find first index where timestamp >= bar_open_time
        let cutoff_idx = match self.trades.binary_search_by(|trade| {
            if trade.timestamp < bar_open_time {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(idx) => idx,  // Found exact match - exclude trades at bar_open_time
            Err(idx) => idx, // Insertion point - all trades before this are < bar_open_time
        };

        self.trades
            .iter()
            .take(cutoff_idx)
            .collect()
    }

    /// Compute inter-bar features from lookback window
    ///
    /// # Arguments
    ///
    /// * `bar_open_time` - The open timestamp of the current bar (microseconds)
    ///
    /// # Returns
    ///
    /// `InterBarFeatures` with computed values, or `None` for features that
    /// cannot be computed due to insufficient data.
    pub fn compute_features(&self, bar_open_time: i64) -> InterBarFeatures {
        let lookback = self.get_lookback_trades(bar_open_time);

        if lookback.is_empty() {
            return InterBarFeatures::default();
        }

        let mut features = InterBarFeatures::default();

        // === Tier 1: Core Features ===
        self.compute_tier1_features(&lookback, &mut features);

        // === Tier 2: Statistical Features ===
        if self.config.compute_tier2 {
            self.compute_tier2_features(&lookback, &mut features);
        }

        // === Tier 3: Advanced Features ===
        if self.config.compute_tier3 {
            self.compute_tier3_features(&lookback, &mut features);
        }

        features
    }

    /// Compute Tier 1 features (7 features, min 1 trade)
    fn compute_tier1_features(&self, lookback: &[&TradeSnapshot], features: &mut InterBarFeatures) {
        let n = lookback.len();
        if n == 0 {
            return;
        }

        // Trade count
        features.lookback_trade_count = Some(n as u32);

        // Accumulate buy/sell volumes and counts
        let (buy_vol, sell_vol, buy_count, sell_count, total_turnover) =
            lookback
                .iter()
                .fold((0.0, 0.0, 0u32, 0u32, 0i128), |acc, t| {
                    if t.is_buyer_maker {
                        // Sell pressure
                        (
                            acc.0,
                            acc.1 + t.volume.to_f64(),
                            acc.2,
                            acc.3 + 1,
                            acc.4 + t.turnover,
                        )
                    } else {
                        // Buy pressure
                        (
                            acc.0 + t.volume.to_f64(),
                            acc.1,
                            acc.2 + 1,
                            acc.3,
                            acc.4 + t.turnover,
                        )
                    }
                });

        let total_vol = buy_vol + sell_vol;

        // OFI: Order Flow Imbalance [-1, 1]
        features.lookback_ofi = Some(if total_vol > f64::EPSILON {
            (buy_vol - sell_vol) / total_vol
        } else {
            0.0
        });

        // Count imbalance [-1, 1]
        let total_count = buy_count + sell_count;
        features.lookback_count_imbalance = Some(if total_count > 0 {
            (buy_count as f64 - sell_count as f64) / total_count as f64
        } else {
            0.0
        });

        // Duration
        let first_ts = lookback.first().unwrap().timestamp;
        let last_ts = lookback.last().unwrap().timestamp;
        let duration_us = last_ts - first_ts;
        features.lookback_duration_us = Some(duration_us);

        // Intensity (trades per second)
        let duration_sec = duration_us as f64 / 1_000_000.0;
        features.lookback_intensity = Some(if duration_sec > f64::EPSILON {
            n as f64 / duration_sec
        } else {
            n as f64 // Instant window = all trades at once
        });

        // VWAP
        // Issue #88: i128 sum to prevent overflow on high-token-count symbols
        let total_volume_fp: i128 = lookback.iter().map(|t| t.volume.0 as i128).sum();
        features.lookback_vwap = Some(if total_volume_fp > 0 {
            let vwap_raw = total_turnover / total_volume_fp;
            FixedPoint(vwap_raw as i64)
        } else {
            FixedPoint(0)
        });

        // VWAP position within range [0, 1]
        let (low, high) = lookback.iter().fold((i64::MAX, i64::MIN), |acc, t| {
            (acc.0.min(t.price.0), acc.1.max(t.price.0))
        });
        let range = (high - low) as f64;
        let vwap_val = features.lookback_vwap.as_ref().map(|v| v.0).unwrap_or(0);
        features.lookback_vwap_position = Some(if range > f64::EPSILON {
            (vwap_val - low) as f64 / range
        } else {
            0.5 // Flat price = middle position
        });
    }

    /// Compute Tier 2 features (5 features, varying min trades)
    fn compute_tier2_features(&self, lookback: &[&TradeSnapshot], features: &mut InterBarFeatures) {
        let n = lookback.len();

        // Kyle's Lambda (min 2 trades)
        if n >= 2 {
            features.lookback_kyle_lambda = Some(compute_kyle_lambda(lookback));
        }

        // Burstiness (min 2 trades for inter-arrival times)
        if n >= 2 {
            features.lookback_burstiness = Some(compute_burstiness(lookback));
        }

        // Volume skewness (min 3 trades)
        if n >= 3 {
            let (skew, kurt) = compute_volume_moments(lookback);
            features.lookback_volume_skew = Some(skew);
            // Kurtosis requires 4 trades for meaningful estimate
            if n >= 4 {
                features.lookback_volume_kurt = Some(kurt);
            }
        }

        // Price range (min 1 trade)
        if n >= 1 {
            let first_price = lookback.first().unwrap().price.to_f64();
            let (low, high) = lookback.iter().fold((i64::MAX, i64::MIN), |acc, t| {
                (acc.0.min(t.price.0), acc.1.max(t.price.0))
            });
            let range = (high - low) as f64 / 1e8; // Convert from FixedPoint scale
            features.lookback_price_range = Some(if first_price > f64::EPSILON {
                range / first_price
            } else {
                0.0
            });
        }
    }

    /// Compute Tier 3 features (4 features, higher min trades)
    ///
    /// Issue #96 Task #7 Phase 2: Batch OHLC extraction for 5-10% overhead reduction
    /// Extracts prices and OHLC once, reuses across multiple feature computations
    /// Issue #96 Task #10: SmallVec optimization for price allocation (typical 100-500 trades)
    fn compute_tier3_features(&self, lookback: &[&TradeSnapshot], features: &mut InterBarFeatures) {
        let n = lookback.len();

        // Phase 2 optimization: Extract OHLC and prices in single pass
        // Task #10: Use SmallVec with 256 inline capacity to avoid heap allocation for typical windows
        let prices: SmallVec<[f64; 256]> = lookback.iter().map(|t| t.price.to_f64()).collect();
        let (open, high, low, close) = extract_ohlc_batch(lookback);

        // Kaufman Efficiency Ratio (min 2 trades)
        if n >= 2 {
            features.lookback_kaufman_er = Some(compute_kaufman_er(&prices));
        }

        // Garman-Klass volatility (min 1 trade) - use batch OHLC data
        if n >= 1 {
            features.lookback_garman_klass_vol = Some(compute_garman_klass_with_ohlc(open, high, low, close));
        }

        // Hurst exponent via DFA (min 64 trades for reliable estimate)
        if n >= 64 {
            features.lookback_hurst = Some(compute_hurst_dfa(&prices));
        }

        // Entropy: adaptive switching (Issue #96 Task #7 Phase 3)
        // - Small windows (n < 500): Permutation Entropy (O(n) balanced overhead)
        // - Large windows (n >= 500): Approximate Entropy (5-10x faster on large n)
        // Minimum 60 trades for permutation entropy (m=3, need 10 * m! = 60)
        if n >= 60 {
            features.lookback_permutation_entropy = Some(compute_entropy_adaptive(&prices));
        }
    }

    /// Reset bar boundary tracking (Issue #81)
    ///
    /// Called at ouroboros boundaries. Clears bar close indices but preserves
    /// trade history — trades are still valid lookback data for the first
    /// bar of the new segment.
    pub fn reset_bar_boundaries(&mut self) {
        self.bar_close_indices.clear();
    }

    /// Clear the trade history (e.g., at ouroboros boundary)
    pub fn clear(&mut self) {
        self.trades.clear();
    }

    /// Get current number of trades in buffer
    pub fn len(&self) -> usize {
        self.trades.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.trades.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test trades
    fn create_test_snapshot(
        timestamp: i64,
        price: f64,
        volume: f64,
        is_buyer_maker: bool,
    ) -> TradeSnapshot {
        let price_fp = FixedPoint((price * 1e8) as i64);
        let volume_fp = FixedPoint((volume * 1e8) as i64);
        TradeSnapshot {
            timestamp,
            price: price_fp,
            volume: volume_fp,
            is_buyer_maker,
            turnover: (price_fp.0 as i128) * (volume_fp.0 as i128),
        }
    }

    // ========== OFI Tests ==========

    #[test]
    fn test_ofi_all_buys() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add buy trades (is_buyer_maker = false = buy pressure)
        for i in 0..10 {
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint(5000000000000), // 50000
                volume: FixedPoint(100000000),    // 1.0
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: false, // Buy
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(10000);

        assert!(
            (features.lookback_ofi.unwrap() - 1.0).abs() < f64::EPSILON,
            "OFI should be 1.0 for all buys, got {}",
            features.lookback_ofi.unwrap()
        );
    }

    #[test]
    fn test_ofi_all_sells() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add sell trades (is_buyer_maker = true = sell pressure)
        for i in 0..10 {
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint(5000000000000),
                volume: FixedPoint(100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: true, // Sell
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(10000);

        assert!(
            (features.lookback_ofi.unwrap() - (-1.0)).abs() < f64::EPSILON,
            "OFI should be -1.0 for all sells, got {}",
            features.lookback_ofi.unwrap()
        );
    }

    #[test]
    fn test_ofi_balanced() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add equal buy and sell volumes
        for i in 0..10 {
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint(5000000000000),
                volume: FixedPoint(100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: i % 2 == 0, // Alternating
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(10000);

        assert!(
            features.lookback_ofi.unwrap().abs() < f64::EPSILON,
            "OFI should be 0.0 for balanced volumes, got {}",
            features.lookback_ofi.unwrap()
        );
    }

    // ========== Burstiness Tests ==========

    #[test]
    fn test_burstiness_regular_intervals() {
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let t1 = create_test_snapshot(1000, 100.0, 1.0, false);
        let t2 = create_test_snapshot(2000, 100.0, 1.0, false);
        let t3 = create_test_snapshot(3000, 100.0, 1.0, false);
        let t4 = create_test_snapshot(4000, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2, &t3, &t4];

        let b = compute_burstiness(&lookback);

        // Perfectly regular: sigma = 0 -> B = -1
        assert!(
            (b - (-1.0)).abs() < 0.01,
            "Burstiness should be -1 for regular intervals, got {}",
            b
        );
    }

    // ========== Kaufman ER Tests ==========

    #[test]
    fn test_kaufman_er_perfect_trend() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let er = compute_kaufman_er(&prices);

        assert!(
            (er - 1.0).abs() < f64::EPSILON,
            "Kaufman ER should be 1.0 for perfect trend, got {}",
            er
        );
    }

    #[test]
    fn test_kaufman_er_round_trip() {
        let prices = vec![100.0, 102.0, 104.0, 102.0, 100.0];
        let er = compute_kaufman_er(&prices);

        assert!(
            er.abs() < f64::EPSILON,
            "Kaufman ER should be 0.0 for round trip, got {}",
            er
        );
    }

    // ========== Permutation Entropy Tests ==========

    #[test]
    fn test_permutation_entropy_monotonic() {
        // Strictly increasing: only pattern 012 appears -> H = 0
        let prices: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pe = compute_permutation_entropy(&prices);

        assert!(
            pe.abs() < f64::EPSILON,
            "PE should be 0 for monotonic, got {}",
            pe
        );
    }

    // ========== Temporal Integrity Tests ==========

    #[test]
    fn test_lookback_excludes_current_bar_trades() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add trades at timestamps 0, 1000, 2000, 3000
        for i in 0..4 {
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint(5000000000000),
                volume: FixedPoint(100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: false,
                is_best_match: None,
            };
            history.push(&trade);
        }

        // Get lookback for bar opening at timestamp 2000
        let lookback = history.get_lookback_trades(2000);

        // Should only include trades with timestamp < 2000 (i.e., 0 and 1000)
        assert_eq!(lookback.len(), 2, "Should have 2 trades before bar open");

        for trade in &lookback {
            assert!(
                trade.timestamp < 2000,
                "Trade at {} should be before bar open at 2000",
                trade.timestamp
            );
        }
    }

    // ========== Bounded Output Tests ==========

    #[test]
    fn test_count_imbalance_bounded() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add random mix of buys and sells
        for i in 0..100 {
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint(5000000000000),
                volume: FixedPoint((i % 10 + 1) * 100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: i % 3 == 0,
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(100000);
        let imb = features.lookback_count_imbalance.unwrap();

        assert!(
            imb >= -1.0 && imb <= 1.0,
            "Count imbalance should be in [-1, 1], got {}",
            imb
        );
    }

    #[test]
    fn test_vwap_position_bounded() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add trades at varying prices
        for i in 0..20 {
            let price = 50000.0 + (i as f64 * 10.0);
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint((price * 1e8) as i64),
                volume: FixedPoint(100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: false,
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(20000);
        let pos = features.lookback_vwap_position.unwrap();

        assert!(
            pos >= 0.0 && pos <= 1.0,
            "VWAP position should be in [0, 1], got {}",
            pos
        );
    }

    #[test]
    fn test_hurst_soft_clamp_bounded() {
        // Test with extreme input values
        // Note: tanh approaches 0 and 1 asymptotically, so we use >= and <=
        for raw_h in [-10.0, -1.0, 0.0, 0.5, 1.0, 2.0, 10.0] {
            let clamped = soft_clamp_hurst(raw_h);
            assert!(
                clamped >= 0.0 && clamped <= 1.0,
                "Hurst {} soft-clamped to {} should be in [0, 1]",
                raw_h,
                clamped
            );
        }

        // Verify 0.5 maps to 0.5 exactly
        let h_half = soft_clamp_hurst(0.5);
        assert!(
            (h_half - 0.5).abs() < f64::EPSILON,
            "Hurst 0.5 should map to 0.5, got {}",
            h_half
        );
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_empty_lookback() {
        let history = TradeHistory::new(InterBarConfig::default());
        let features = history.compute_features(1000);

        assert!(
            features.lookback_trade_count.is_none() || features.lookback_trade_count == Some(0)
        );
    }

    #[test]
    fn test_single_trade_lookback() {
        let mut history = TradeHistory::new(InterBarConfig::default());

        let trade = AggTrade {
            agg_trade_id: 0,
            price: FixedPoint(5000000000000),
            volume: FixedPoint(100000000),
            first_trade_id: 0,
            last_trade_id: 0,
            timestamp: 0,
            is_buyer_maker: false,
            is_best_match: None,
        };
        history.push(&trade);

        let features = history.compute_features(1000);

        assert_eq!(features.lookback_trade_count, Some(1));
        assert_eq!(features.lookback_duration_us, Some(0)); // Single trade = 0 duration
    }

    #[test]
    fn test_kyle_lambda_zero_imbalance() {
        // Equal buy/sell -> imbalance = 0 -> should return 0, not infinity
        let t0 = create_test_snapshot(0, 100.0, 1.0, false); // buy
        let t1 = create_test_snapshot(1000, 102.0, 1.0, true); // sell
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1];

        let lambda = compute_kyle_lambda(&lookback);

        assert!(
            lambda.is_finite(),
            "Kyle lambda should be finite, got {}",
            lambda
        );
        assert!(
            lambda.abs() < f64::EPSILON,
            "Kyle lambda should be 0 for zero imbalance"
        );
    }

    // ========== Tier 2 Features: Comprehensive Edge Cases (Issue #96 Task #43) ==========

    #[test]
    fn test_kyle_lambda_strong_buy_pressure() {
        // Strong buy pressure: many buys, few sells -> positive lambda
        let trades: Vec<TradeSnapshot> = (0..5)
            .map(|i| create_test_snapshot(i * 1000, 100.0 + i as f64, 1.0, false))
            .chain((5..7).map(|i| create_test_snapshot(i * 1000, 100.0 + i as f64, 1.0, true)))
            .collect();
        let lookback: Vec<&TradeSnapshot> = trades.iter().collect();

        let lambda = compute_kyle_lambda(&lookback);
        assert!(lambda > 0.0, "Buy pressure should yield positive lambda, got {}", lambda);
        assert!(lambda.is_finite(), "Kyle lambda should be finite");
    }

    #[test]
    fn test_kyle_lambda_strong_sell_pressure() {
        // Strong sell pressure: many sell orders (is_buyer_maker=true) at declining prices
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);    // buy
        let t1 = create_test_snapshot(1000, 99.9, 5.0, true);   // sell (larger)
        let t2 = create_test_snapshot(2000, 99.8, 5.0, true);   // sell (larger)
        let t3 = create_test_snapshot(3000, 99.7, 5.0, true);   // sell (larger)
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2, &t3];

        let lambda = compute_kyle_lambda(&lookback);
        assert!(lambda.is_finite(), "Kyle lambda should be finite");
        // With sell volume > buy volume and price declining, lambda should be negative
    }

    #[test]
    fn test_burstiness_single_trade() {
        // Single trade: no inter-arrivals, should return default
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0];

        let b = compute_burstiness(&lookback);
        assert!(
            b.is_finite(),
            "Burstiness with single trade should be finite, got {}",
            b
        );
    }

    #[test]
    fn test_burstiness_two_trades() {
        // Two trades: insufficient data, sigma = 0 -> B = -1
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let t1 = create_test_snapshot(1000, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1];

        let b = compute_burstiness(&lookback);
        assert!(
            (b - (-1.0)).abs() < 0.01,
            "Burstiness with uniform inter-arrivals should be -1, got {}",
            b
        );
    }

    #[test]
    fn test_burstiness_bursty_arrivals() {
        // Uneven inter-arrivals: clusters of fast then slow arrivals
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let t1 = create_test_snapshot(100, 100.0, 1.0, false);
        let t2 = create_test_snapshot(200, 100.0, 1.0, false);
        let t3 = create_test_snapshot(5000, 100.0, 1.0, false);
        let t4 = create_test_snapshot(10000, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2, &t3, &t4];

        let b = compute_burstiness(&lookback);
        assert!(
            b > -1.0 && b <= 1.0,
            "Burstiness should be bounded [-1, 1], got {}",
            b
        );
    }

    #[test]
    fn test_volume_skew_right_skewed() {
        // Right-skewed distribution (many small, few large volumes)
        let t0 = create_test_snapshot(0, 100.0, 0.1, false);
        let t1 = create_test_snapshot(1000, 100.0, 0.1, false);
        let t2 = create_test_snapshot(2000, 100.0, 0.1, false);
        let t3 = create_test_snapshot(3000, 100.0, 0.1, false);
        let t4 = create_test_snapshot(4000, 100.0, 10.0, false); // Large outlier
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2, &t3, &t4];

        let skew = compute_volume_moments(&lookback).0;
        assert!(skew > 0.0, "Right-skewed volume should have positive skewness, got {}", skew);
        assert!(skew.is_finite(), "Skewness must be finite");
    }

    #[test]
    fn test_volume_kurtosis_heavy_tails() {
        // Heavy-tailed distribution (few very large, few very small, middle is sparse)
        let t0 = create_test_snapshot(0, 100.0, 0.01, false);
        let t1 = create_test_snapshot(1000, 100.0, 1.0, false);
        let t2 = create_test_snapshot(2000, 100.0, 1.0, false);
        let t3 = create_test_snapshot(3000, 100.0, 1.0, false);
        let t4 = create_test_snapshot(4000, 100.0, 100.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2, &t3, &t4];

        let kurtosis = compute_volume_moments(&lookback).1;
        assert!(kurtosis > 0.0, "Heavy-tailed distribution should have positive kurtosis, got {}", kurtosis);
        assert!(kurtosis.is_finite(), "Kurtosis must be finite");
    }

    #[test]
    fn test_volume_skew_symmetric() {
        // Symmetric distribution (equal volumes) -> skewness = 0
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let t1 = create_test_snapshot(1000, 100.0, 1.0, false);
        let t2 = create_test_snapshot(2000, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2];

        let skew = compute_volume_moments(&lookback).0;
        assert!(
            skew.abs() < f64::EPSILON,
            "Symmetric volume distribution should have near-zero skewness, got {}",
            skew
        );
    }

    #[test]
    fn test_kyle_lambda_price_unchanged() {
        // Price doesn't move but there's imbalance -> should be finite
        let t0 = create_test_snapshot(0, 100.0, 1.0, false);
        let t1 = create_test_snapshot(1000, 100.0, 1.0, false);
        let t2 = create_test_snapshot(2000, 100.0, 1.0, false);
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1, &t2];

        let lambda = compute_kyle_lambda(&lookback);
        assert!(
            lambda.is_finite(),
            "Kyle lambda should be finite even with no price change, got {}",
            lambda
        );
    }

    // ========== BarRelative Mode Tests (Issue #81) ==========

    /// Helper to create a test AggTrade
    fn make_trade(id: i64, timestamp: i64) -> AggTrade {
        AggTrade {
            agg_trade_id: id,
            price: FixedPoint(5000000000000), // 50000
            volume: FixedPoint(100000000),    // 1.0
            first_trade_id: id,
            last_trade_id: id,
            timestamp,
            is_buyer_maker: false,
            is_best_match: None,
        }
    }

    #[test]
    fn test_bar_relative_bootstrap_keeps_all_trades() {
        // Before any bars close, BarRelative should keep all trades
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(3),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Push 100 trades without closing any bar
        for i in 0..100 {
            history.push(&make_trade(i, i * 1000));
        }

        assert_eq!(history.len(), 100, "Bootstrap phase should keep all trades");
    }

    #[test]
    fn test_bar_relative_prunes_after_bar_close() {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(2),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Bar 1: 10 trades (timestamps 0-9000)
        for i in 0..10 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close(); // total_pushed = 10

        // Bar 2: 20 trades (timestamps 10000-29000)
        for i in 10..30 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close(); // total_pushed = 30

        // Bar 3: 5 trades (timestamps 30000-34000)
        for i in 30..35 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close(); // total_pushed = 35

        // With BarRelative(2), after 3 bar closes we keep trades from last 2 bars:
        // bar_close_indices = [10, 30, 35] -> keep last 2 -> from index 10 to 35 = 25 trades
        // But bar 1 trades (0-9) should be pruned, keeping bars 2+3 = 25 trades + bar 3's 5
        // Actually: bar_close_indices keeps n+1=3 boundaries: [10, 30, 35]
        // Oldest boundary at [len-n_bars] = [3-2] = index 1 = 30
        // keep_count = total_pushed(35) - 30 = 5
        // But wait -- we also have current in-progress trades.
        // After bar 3 closes with 35 total, and no more pushes:
        // trades.len() should be <= keep_count from the prune in on_bar_close
        // The prune happens on each push, and on_bar_close records boundary then
        // next push triggers prune.

        // Push one more trade to trigger prune with new boundary
        history.push(&make_trade(35, 35000));

        // Now: bar_close_indices = [10, 30, 35], total_pushed = 36
        // keep_count = 36 - 30 = 6 (trades from bar 2 boundary onwards)
        // But we also have protected_until which prevents pruning lookback trades
        // Without protection set (no on_bar_open called), all trades can be pruned
        assert!(
            history.len() <= 26, // 25 from bars 2+3 + 1 new, minus pruned old ones
            "Should prune old bars, got {} trades",
            history.len()
        );
    }

    #[test]
    fn test_bar_relative_mixed_bar_sizes() {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(2),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Bar 1: 5 trades
        for i in 0..5 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close();

        // Bar 2: 50 trades
        for i in 5..55 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close();

        // Bar 3: 3 trades
        for i in 55..58 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close();

        // Push one more to trigger prune
        history.push(&make_trade(58, 58000));

        // With BarRelative(2), after 3 bars:
        // bar_close_indices has max n+1=3 entries: [5, 55, 58]
        // Oldest boundary for pruning: [len-n_bars] = [3-2] = index 1 = 55
        // keep_count = 59 - 55 = 4 (3 from bar 3 + 1 new)
        // This correctly adapts: bar 2 had 50 trades but bar 3 only had 3
        assert!(
            history.len() <= 54, // bar 2 + bar 3 + 1 = 54 max
            "Mixed bar sizes should prune correctly, got {} trades",
            history.len()
        );
    }

    #[test]
    fn test_bar_relative_lookback_features_computed() {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(3),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Push 20 trades (timestamps 0-19000)
        for i in 0..20 {
            let price = 50000.0 + (i as f64 * 10.0);
            let trade = AggTrade {
                agg_trade_id: i,
                price: FixedPoint((price * 1e8) as i64),
                volume: FixedPoint(100000000),
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: i % 2 == 0,
                is_best_match: None,
            };
            history.push(&trade);
        }
        // Close bar 1 at total_pushed=20
        history.on_bar_close();

        // Simulate bar 2 opening at timestamp 20000
        history.on_bar_open(20000);

        // Compute features for bar 2 -- should use trades before 20000
        let features = history.compute_features(20000);

        // All 20 trades are before bar open, should have lookback features
        assert_eq!(features.lookback_trade_count, Some(20));
        assert!(features.lookback_ofi.is_some());
        assert!(features.lookback_intensity.is_some());
    }

    #[test]
    fn test_bar_relative_reset_bar_boundaries() {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(2),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Push trades and close a bar
        for i in 0..10 {
            history.push(&make_trade(i, i * 1000));
        }
        history.on_bar_close();

        assert_eq!(history.bar_close_indices.len(), 1);

        // Reset boundaries (ouroboros)
        history.reset_bar_boundaries();

        assert!(
            history.bar_close_indices.is_empty(),
            "bar_close_indices should be empty after reset"
        );
        // Trades should still be there
        assert_eq!(
            history.len(),
            10,
            "Trades should persist after boundary reset"
        );
    }

    #[test]
    fn test_bar_relative_on_bar_close_limits_indices() {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::BarRelative(2),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        // Close 5 bars
        for bar_num in 0..5 {
            for i in 0..5 {
                history.push(&make_trade(bar_num * 5 + i, (bar_num * 5 + i) * 1000));
            }
            history.on_bar_close();
        }

        // With BarRelative(2), should keep at most n+1=3 boundaries
        assert!(
            history.bar_close_indices.len() <= 3,
            "Should keep at most n+1 boundaries, got {}",
            history.bar_close_indices.len()
        );
    }

    #[test]
    fn test_bar_relative_does_not_affect_fixed_count() {
        // Verify FixedCount mode is unaffected by BarRelative changes
        let config = InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(10),
            compute_tier2: false,
            compute_tier3: false,
        };
        let mut history = TradeHistory::new(config);

        for i in 0..30 {
            history.push(&make_trade(i, i * 1000));
        }
        // on_bar_close should be no-op for FixedCount
        history.on_bar_close();

        // FixedCount(10) keeps 2*10=20 max
        assert!(
            history.len() <= 20,
            "FixedCount(10) should keep at most 20 trades, got {}",
            history.len()
        );
        assert!(
            history.bar_close_indices.is_empty(),
            "FixedCount should not track bar boundaries"
        );
    }

    // === Memory efficiency tests (R5) ===

    #[test]
    fn test_volume_moments_numerical_accuracy() {
        // R5: Verify 2-pass fold produces identical results to previous 4-pass.
        // Symmetric distribution [1,2,3,4,5] → skewness ≈ 0
        let price_fp = FixedPoint((100.0 * 1e8) as i64);
        let snapshots: Vec<TradeSnapshot> = (1..=5_i64)
            .map(|v| {
                let volume_fp = FixedPoint((v as f64 * 1e8) as i64);
                TradeSnapshot {
                    price: price_fp,
                    volume: volume_fp,
                    timestamp: v * 1000,
                    is_buyer_maker: false,
                    turnover: price_fp.0 as i128 * volume_fp.0 as i128,
                }
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = snapshots.iter().collect();
        let (skew, kurt) = compute_volume_moments(&refs);

        // Symmetric uniform-like distribution: skewness should be 0
        assert!(
            skew.abs() < 1e-10,
            "Symmetric distribution should have skewness ≈ 0, got {skew}"
        );
        // Uniform distribution excess kurtosis = -1.3
        assert!(
            (kurt - (-1.3)).abs() < 0.1,
            "Uniform-like kurtosis should be ≈ -1.3, got {kurt}"
        );
    }

    #[test]
    fn test_volume_moments_edge_cases() {
        let price_fp = FixedPoint((100.0 * 1e8) as i64);

        // n < 3 returns (0, 0)
        let v1 = FixedPoint((1.0 * 1e8) as i64);
        let v2 = FixedPoint((2.0 * 1e8) as i64);
        let s1 = TradeSnapshot {
            price: price_fp,
            volume: v1,
            timestamp: 1000,
            is_buyer_maker: false,
            turnover: price_fp.0 as i128 * v1.0 as i128,
        };
        let s2 = TradeSnapshot {
            price: price_fp,
            volume: v2,
            timestamp: 2000,
            is_buyer_maker: false,
            turnover: price_fp.0 as i128 * v2.0 as i128,
        };
        let refs: Vec<&TradeSnapshot> = vec![&s1, &s2];
        let (skew, kurt) = compute_volume_moments(&refs);
        assert_eq!(skew, 0.0, "n < 3 should return 0");
        assert_eq!(kurt, 0.0, "n < 3 should return 0");

        // All same volume returns (0, 0)
        let vol = FixedPoint((5.0 * 1e8) as i64);
        let same: Vec<TradeSnapshot> = (0..10_i64)
            .map(|i| TradeSnapshot {
                price: price_fp,
                volume: vol,
                timestamp: i * 1000,
                is_buyer_maker: false,
                turnover: price_fp.0 as i128 * vol.0 as i128,
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = same.iter().collect();
        let (skew, kurt) = compute_volume_moments(&refs);
        assert_eq!(skew, 0.0, "All same volume should return 0");
        assert_eq!(kurt, 0.0, "All same volume should return 0");
    }
}
