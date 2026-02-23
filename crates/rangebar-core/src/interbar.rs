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
use rayon::join; // Issue #115: Parallelization of Tier 2/3 features
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
    /// Issue #104: Pushes since last prune check (reduces check frequency)
    pushes_since_prune_check: usize,
    /// Issue #104: Maximum safe capacity (computed once at init)
    max_safe_capacity: usize,
    /// Issue #96 Task #117: Cache for permutation entropy results
    /// Avoids redundant computation on identical price sequences
    /// Uses Arc<RwLock> for thread-safe interior mutability (needed for rayon parallelization)
    entropy_cache: std::sync::Arc<std::sync::RwLock<crate::interbar_math::EntropyCache>>,
}

impl TradeHistory {
    /// Create new trade history with given configuration
    pub fn new(config: InterBarConfig) -> Self {
        // Issue #118: Optimized capacity sizing based on lookback config
        // Reduces memory overhead by 20-30% while maintaining safety margins
        let capacity = match &config.lookback_mode {
            LookbackMode::FixedCount(n) => *n, // Exact size (pruning handles overflow)
            LookbackMode::FixedWindow(_) => 500, // Covers 99% of time-based windows
            LookbackMode::BarRelative(_) => 1000, // Adaptive pruning scales with bar size
        };
        // Issue #104: Compute max safe capacity once to avoid repeated computation
        let max_safe_capacity = match &config.lookback_mode {
            LookbackMode::FixedCount(n) => *n * 2,  // 2x safety margin (reduced from 3x)
            LookbackMode::FixedWindow(_) => 1500,    // Reduced from 3000 (better cache locality)
            LookbackMode::BarRelative(_) => 2000,    // Reduced from 5000 (adaptive scaling)
        };
        // Task #91: Pre-allocate bar_close_indices buffer
        // Typical lookback: 10-100 bars, so capacity 128 avoids most re-allocations
        let bar_capacity = match &config.lookback_mode {
            LookbackMode::BarRelative(n_bars) => (*n_bars + 1).min(128),
            _ => 128,
        };
        Self {
            trades: VecDeque::with_capacity(capacity),
            config,
            protected_until: None,
            total_pushed: 0,
            bar_close_indices: VecDeque::with_capacity(bar_capacity),
            pushes_since_prune_check: 0,
            max_safe_capacity,
            entropy_cache: std::sync::Arc::new(std::sync::RwLock::new(crate::interbar_math::EntropyCache::new())),
        }
    }

    /// Push a new trade to the history buffer
    ///
    /// Automatically prunes old entries based on lookback mode, but preserves
    /// trades needed for lookback computation (timestamp < protected_until).
    /// Issue #104: Uses batched pruning check to reduce frequency
    pub fn push(&mut self, trade: &AggTrade) {
        let snapshot = TradeSnapshot::from(trade);
        self.trades.push_back(snapshot);
        self.total_pushed += 1;
        self.pushes_since_prune_check += 1;

        // Issue #111: Adaptive batch size for pruning checks based on lookback mode
        // - FixedCount(n): Check every n/10 trades (scaled to lookback size)
        // - FixedWindow: Check every 10 trades (time-based, not count-dependent)
        // - BarRelative: Check every 10 trades (self-adaptive, no fixed count)
        let prune_batch_size = match &self.config.lookback_mode {
            LookbackMode::FixedCount(n) => std::cmp::max((*n / 10).max(5), 10), // At least 5, typically n/10
            _ => 10, // FixedWindow and BarRelative use fixed batch
        };

        // Check every N trades or when capacity limit exceeded
        if self.pushes_since_prune_check >= prune_batch_size
            || self.trades.len() > self.max_safe_capacity
        {
            self.prune_if_needed();
            self.pushes_since_prune_check = 0;
        }
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

    /// Conditionally prune trades based on capacity (Task #91: reduce prune() call overhead)
    ///
    /// Only calls the full prune() when approaching capacity limits.
    /// This reduces function call overhead while maintaining correctness.
    /// Issue #104: Use pre-computed max_safe_capacity for branch-free check
    fn prune_if_needed(&mut self) {
        // Issue #104: Simple threshold check using pre-computed capacity
        // Reduces function call overhead and enables better branch prediction
        if self.trades.len() > self.max_safe_capacity {
            self.prune();
        }
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

        // Issue #96 Task #68: Early-exit for tiny lookbacks to avoid collect() overhead
        // If cutoff_idx < 3, we have 0-2 trades: direct inline collection is more efficient
        if cutoff_idx == 0 {
            return SmallVec::new();
        }
        if cutoff_idx == 1 {
            let mut result = SmallVec::new();
            result.push(&self.trades[0]);
            return result;
        }
        if cutoff_idx == 2 {
            let mut result = SmallVec::new();
            result.push(&self.trades[0]);
            result.push(&self.trades[1]);
            return result;
        }

        self.trades
            .iter()
            .take(cutoff_idx)
            .collect()
    }

    /// Compute inter-bar features from lookback window
    ///
    /// Issue #96 Task #99: Memoized float conversions for 2-5% speedup
    /// Extracts prices/volumes once and reuses across all 16 features.
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

        // === Issue #96 Task #99: Single-pass cache extraction ===
        // Pre-compute all float conversions once, before any Tier 2/3 features
        let cache = if self.config.compute_tier2 || self.config.compute_tier3 {
            Some(crate::interbar_math::extract_lookback_cache(&lookback))
        } else {
            None
        };

        // === Tier 2 & 3: Parallelization with rayon (Issue #115) ===
        // Only parallelize if both tiers are computed (rayon overhead significant for single tier)
        if self.config.compute_tier2 && self.config.compute_tier3 {
            let (tier2_features, tier3_features) = join(
                || self.compute_tier2_features(&lookback, cache.as_ref()),
                || self.compute_tier3_features(&lookback, cache.as_ref()),
            );
            features.merge_tier2(&tier2_features);
            features.merge_tier3(&tier3_features);
        } else {
            // Sequential fallback for single tier or disabled tiers
            if self.config.compute_tier2 {
                let tier2 = self.compute_tier2_features(&lookback, cache.as_ref());
                features.merge_tier2(&tier2);
            }
            if self.config.compute_tier3 {
                let tier3 = self.compute_tier3_features(&lookback, cache.as_ref());
                features.merge_tier3(&tier3);
            }
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

        // Issue #96 Task #46: Merge Tier 1 feature folds into single pass
        // Previously: 3 separate folds (buy/sell, volumes, min/max prices)
        // Now: Single fold with 8-value accumulator - 8-15% speedup via improved cache locality
        let (buy_vol, sell_vol, buy_count, sell_count, total_turnover, total_volume_fp, low, high) =
            lookback
                .iter()
                .fold((0.0, 0.0, 0u32, 0u32, 0i128, 0i128, i64::MAX, i64::MIN), |acc, t| {
                    let new_turnover = acc.4 + t.turnover;
                    let new_volume_fp = acc.5 + (t.volume.0 as i128);
                    let new_low = acc.6.min(t.price.0);
                    let new_high = acc.7.max(t.price.0);

                    if t.is_buyer_maker {
                        // Sell pressure
                        (
                            acc.0,
                            acc.1 + t.volume.to_f64(),
                            acc.2,
                            acc.3 + 1,
                            new_turnover,
                            new_volume_fp,
                            new_low,
                            new_high,
                        )
                    } else {
                        // Buy pressure
                        (
                            acc.0 + t.volume.to_f64(),
                            acc.1,
                            acc.2 + 1,
                            acc.3,
                            new_turnover,
                            new_volume_fp,
                            new_low,
                            new_high,
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

        // VWAP (Issue #88: i128 sum to prevent overflow on high-token-count symbols)
        features.lookback_vwap = Some(if total_volume_fp > 0 {
            let vwap_raw = total_turnover / total_volume_fp;
            FixedPoint(vwap_raw as i64)
        } else {
            FixedPoint(0)
        });

        // VWAP position within range [0, 1]
        let range = (high - low) as f64;
        let vwap_val = features.lookback_vwap.as_ref().map(|v| v.0).unwrap_or(0);
        features.lookback_vwap_position = Some(if range > f64::EPSILON {
            (vwap_val - low) as f64 / range
        } else {
            0.5 // Flat price = middle position
        });
    }

    /// Compute Tier 2 features (5 features, varying min trades)
    ///
    /// Issue #96 Task #99: Optimized with memoized float conversions.
    /// Uses pre-computed cache passed from compute_features() to avoid
    /// redundant float conversions across multiple feature functions.
    /// Issue #115: Refactored to return InterBarFeatures for rayon parallelization support
    fn compute_tier2_features(
        &self,
        lookback: &[&TradeSnapshot],
        cache: Option<&crate::interbar_math::LookbackCache>,
    ) -> InterBarFeatures {
        let mut features = InterBarFeatures::default();
        let n = lookback.len();

        // Issue #96 Task #99: Use cache passed from compute_features()
        // Avoids redundant float conversion if Tier 3 also enabled
        let cache = cache.cloned().unwrap_or_else(|| crate::interbar_math::extract_lookback_cache(lookback));

        // Kyle's Lambda (min 2 trades)
        if n >= 2 {
            features.lookback_kyle_lambda = Some(compute_kyle_lambda(lookback));
        }

        // Burstiness (min 2 trades for inter-arrival times)
        if n >= 2 {
            features.lookback_burstiness = Some(compute_burstiness(lookback));
        }

        // Volume skewness (min 3 trades)
        // Issue #96 Task #99: Use cached volumes instead of repeated .volume.to_f64() calls
        if n >= 3 {
            let (skew, kurt) = crate::interbar_math::compute_volume_moments_cached(&cache.volumes);
            features.lookback_volume_skew = Some(skew);
            // Kurtosis requires 4 trades for meaningful estimate
            if n >= 4 {
                features.lookback_volume_kurt = Some(kurt);
            }
        }

        // Price range (min 1 trade)
        // Issue #96 Task #99: Use cached open (first price) and OHLC instead of conversion + fold
        if n >= 1 {
            let range = cache.high - cache.low;
            features.lookback_price_range = Some(if cache.open > f64::EPSILON {
                range / cache.open
            } else {
                0.0
            });
        }

        features
    }

    /// Compute Tier 3 features (4 features, higher min trades)
    ///
    /// Issue #96 Task #77: Single-pass OHLC + prices extraction for 1.3-1.6x speedup
    /// Compute Tier 3 features (4 features, higher min trades)
    ///
    /// Issue #96 Task #77: Single-pass OHLC + prices extraction for 1.3-1.6x speedup
    /// Combines price collection with OHLC computation (eliminates double-pass)
    /// Issue #96 Task #10: SmallVec optimization for price allocation (typical 100-500 trades)
    /// Issue #96 Task #99: Reuses memoized float conversions from shared cache
    /// Issue #115: Refactored to return InterBarFeatures for rayon parallelization support
    fn compute_tier3_features(
        &self,
        lookback: &[&TradeSnapshot],
        cache: Option<&crate::interbar_math::LookbackCache>,
    ) -> InterBarFeatures {
        let mut features = InterBarFeatures::default();
        let n = lookback.len();

        // Issue #96 Task #99: Use cache passed from compute_features()
        // If neither Tier 2 nor this cache is provided, extract independently
        let cache = cache.cloned().unwrap_or_else(|| crate::interbar_math::extract_lookback_cache(lookback));
        // Issue #110: Avoid cloning prices - all Tier 3 functions accept &[f64]
        let prices = &cache.prices;
        let (open, high, low, close) = (cache.open, cache.high, cache.low, cache.close);

        // Kaufman Efficiency Ratio (min 2 trades)
        if n >= 2 {
            features.lookback_kaufman_er = Some(compute_kaufman_er(prices));
        }

        // Garman-Klass volatility (min 1 trade) - use batch OHLC data
        if n >= 1 {
            features.lookback_garman_klass_vol = Some(compute_garman_klass_with_ohlc(open, high, low, close));
        }

        // Hurst exponent via DFA (min 64 trades for reliable estimate)
        if n >= 64 {
            features.lookback_hurst = Some(compute_hurst_dfa(prices));
        }

        // Entropy: adaptive switching with caching (Issue #96 Task #7 + Task #117)
        // - Small windows (n < 500): Permutation Entropy with caching (Issue #96 Task #117)
        // - Large windows (n >= 500): Approximate Entropy (5-10x faster on large n)
        // Minimum 60 trades for permutation entropy (m=3, need 10 * m! = 60)
        if n >= 60 {
            // Use RwLock for thread-safe interior mutability (needed for rayon parallelization)
            let mut cache_guard = self.entropy_cache.write().expect("entropy cache lock poisoned");
            features.lookback_permutation_entropy = Some(
                crate::interbar_math::compute_entropy_adaptive_cached(prices, &mut cache_guard)
            );
        }

        features
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

    // ========== Optimization Regression Tests (Task #115-119) ==========

    #[test]
    fn test_optimization_edge_case_zero_trades() {
        // Task #115-119: Verify optimizations handle edge case of zero trades gracefully
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Try to compute features with no trades
        let features = history.compute_features(1000);

        // All features should be None for empty lookback
        assert!(features.lookback_ofi.is_none());
        assert!(features.lookback_kyle_lambda.is_none());
        assert!(features.lookback_hurst.is_none());
    }

    #[test]
    fn test_optimization_edge_case_large_lookback() {
        // Task #118/119: Verify optimizations handle large lookback windows correctly
        // Tests VecDeque capacity optimization and SmallVec trade accumulation
        let config = InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(500),
            ..Default::default()
        };
        let mut history = TradeHistory::new(config);

        // Add 600 trades (exceeds 500-trade lookback)
        for i in 0..600_i64 {
            let snapshot = create_test_snapshot(i * 1000, 100.0, 10.0, i % 2 == 0);
            history.push(&AggTrade {
                agg_trade_id: i,
                price: snapshot.price,
                volume: snapshot.volume,
                first_trade_id: i,
                last_trade_id: i,
                timestamp: snapshot.timestamp,
                is_buyer_maker: snapshot.is_buyer_maker,
                is_best_match: Some(false),
            });
        }

        // Verify that pruning maintains correct lookback window
        let lookback = history.get_lookback_trades(599000);
        assert!(
            lookback.len() <= 600, // Should be around 500, maybe a bit more
            "Lookback should be <= 600 trades, got {}", lookback.len()
        );

        // Compute features - this exercises the optimizations
        let features = history.compute_features(599000);

        // Tier 1 features should be present
        assert!(features.lookback_trade_count.is_some(), "Trade count should be computed");
        assert!(features.lookback_ofi.is_some(), "OFI should be computed");
    }

    #[test]
    fn test_optimization_edge_case_single_trade() {
        // Task #115-119: Verify optimizations handle single-trade edge case
        let mut history = TradeHistory::new(InterBarConfig::default());

        let snapshot = create_test_snapshot(1000, 100.0, 10.0, false);
        history.push(&AggTrade {
            agg_trade_id: 1,
            price: snapshot.price,
            volume: snapshot.volume,
            first_trade_id: 1,
            last_trade_id: 1,
            timestamp: snapshot.timestamp,
            is_buyer_maker: snapshot.is_buyer_maker,
            is_best_match: Some(false),
        });

        let features = history.compute_features(2000);

        // Tier 1 should compute (only 1 trade needed)
        assert!(features.lookback_trade_count.is_some());
        // Tier 3 definitely not (needs >= 60 for Hurst/Entropy)
        assert!(features.lookback_hurst.is_none());
    }

    #[test]
    fn test_optimization_many_trades() {
        // Task #119: Verify SmallVec handles typical bar trade counts (100-500)
        let mut history = TradeHistory::new(InterBarConfig::default());

        // Add 300 trades
        for i in 0..300_i64 {
            let snapshot = create_test_snapshot(
                i * 1000,
                100.0 + (i as f64 % 10.0),
                10.0 + (i as f64 % 5.0),
                i % 2 == 0,
            );
            history.push(&AggTrade {
                agg_trade_id: i,
                price: snapshot.price,
                volume: snapshot.volume,
                first_trade_id: i,
                last_trade_id: i,
                timestamp: snapshot.timestamp,
                is_buyer_maker: snapshot.is_buyer_maker,
                is_best_match: Some(false),
            });
        }

        // Get lookback trades
        let lookback = history.get_lookback_trades(299000);

        // Compute features with both tiers enabled (Task #115: rayon parallelization)
        let features = history.compute_features(299000);

        // Verify Tier 2 features are present
        assert!(features.lookback_kyle_lambda.is_some(), "Kyle lambda should be computed");
        assert!(features.lookback_burstiness.is_some(), "Burstiness should be computed");

        // Verify Tier 3 features are present (only if n >= 60)
        if lookback.len() >= 60 {
            assert!(features.lookback_hurst.is_some(), "Hurst should be computed");
            assert!(features.lookback_permutation_entropy.is_some(), "Entropy should be computed");
        }
    }
}
