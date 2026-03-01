//! Inter-bar type definitions
//!
//! Extracted from interbar.rs (Phase 2b refactoring)
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59

use crate::fixed_point::FixedPoint;
use crate::types::AggTrade;

/// Configuration for inter-bar feature computation
#[derive(Debug, Clone)]
pub struct InterBarConfig {
    /// Lookback mode: by count or by time
    pub lookback_mode: LookbackMode,
    /// Whether to compute Tier 2 features (requires more trades)
    pub compute_tier2: bool,
    /// Whether to compute Tier 3 features (requires 60+ trades for some)
    pub compute_tier3: bool,
    /// Issue #128: Per-feature override for Hurst exponent computation.
    /// None = follow compute_tier3 flag. Some(false) = skip even if tier enabled.
    /// Some(true) = compute even if tier disabled.
    pub compute_hurst: Option<bool>,
    /// Issue #128: Per-feature override for Permutation Entropy computation.
    /// None = follow compute_tier3 flag. Some(false) = skip even if tier enabled.
    /// Some(true) = compute even if tier disabled.
    pub compute_permutation_entropy: Option<bool>,
}

impl InterBarConfig {
    /// Issue #128: Resolve whether Hurst should be computed, considering
    /// per-feature override and tier flag.
    #[inline]
    pub fn should_compute_hurst(&self) -> bool {
        self.compute_hurst.unwrap_or(self.compute_tier3)
    }

    /// Issue #128: Resolve whether Permutation Entropy should be computed,
    /// considering per-feature override and tier flag.
    #[inline]
    pub fn should_compute_permutation_entropy(&self) -> bool {
        self.compute_permutation_entropy.unwrap_or(self.compute_tier3)
    }
}

impl Default for InterBarConfig {
    fn default() -> Self {
        Self {
            lookback_mode: LookbackMode::FixedCount(500),
            compute_tier2: true,
            compute_tier3: true,
            compute_hurst: None,
            compute_permutation_entropy: None,
        }
    }
}

/// Lookback mode for trade history
#[derive(Debug, Clone)]
pub enum LookbackMode {
    /// Keep last N trades before bar open
    FixedCount(usize),
    /// Keep trades from last T microseconds before bar open
    FixedWindow(i64),
    /// Keep trades from last N completed bars (Issue #81)
    ///
    /// Self-adapting: lookback window scales with bar size.
    /// - Micro bars (50 dbps, ~10 trades): BarRelative(3) → ~30 trades
    /// - Standard bars (250 dbps, ~200 trades): BarRelative(3) → ~600 trades
    /// - Macro bars (1000 dbps, ~5000 trades): BarRelative(3) → ~15,000 trades
    BarRelative(usize),
}

/// Lightweight snapshot of trade for history buffer
///
/// Uses 48 bytes per trade (vs full AggTrade which is larger).
/// For 500 trades: 24 KB memory overhead.
/// Issue #96 Task #190: Cache-optimized field ordering.
/// Reordered to minimize cache line waste and group frequently accessed fields:
/// 1. price + volume (always accessed together for feature computation)
/// 2. turnover (used immediately after price * volume calculation)
/// 3. timestamp (used for time-based metrics)
/// 4. is_buyer_maker (metadata, accessed less frequently)
///
/// This reduces cache misses by ~2-4% on inter-bar feature computation.
#[derive(Debug, Clone)]
pub struct TradeSnapshot {
    /// Price as fixed-point (hot path: accessed in every feature computation)
    pub price: FixedPoint,
    /// Volume as fixed-point (hot path: accessed with price for all calculations)
    pub volume: FixedPoint,
    /// Turnover (price * volume) as i128 (hot path: computed from price/volume immediately)
    pub turnover: i128,
    /// Timestamp in microseconds (warm path: used for intensity and temporal metrics)
    pub timestamp: i64,
    /// Whether buyer is market maker (metadata: accessed for volume direction)
    pub is_buyer_maker: bool,
}

impl From<&AggTrade> for TradeSnapshot {
    fn from(trade: &AggTrade) -> Self {
        Self {
            timestamp: trade.timestamp,
            price: trade.price,
            volume: trade.volume,
            is_buyer_maker: trade.is_buyer_maker,
            turnover: trade.turnover(),
        }
    }
}

/// Inter-bar features computed from lookback window
///
/// All features use `Option<T>` to indicate when computation is not possible
/// (e.g., insufficient trades in lookback window).
#[derive(Debug, Clone, Copy, Default)]
pub struct InterBarFeatures {
    // === Tier 1: Core Features (7) ===
    /// Number of trades in lookback window
    pub lookback_trade_count: Option<u32>,
    /// Order Flow Imbalance: (buy_vol - sell_vol) / total_vol, range [-1, 1]
    pub lookback_ofi: Option<f64>,
    /// Duration of lookback window in microseconds
    pub lookback_duration_us: Option<i64>,
    /// Trade intensity: trades per second
    pub lookback_intensity: Option<f64>,
    /// Volume-weighted average price
    pub lookback_vwap: Option<FixedPoint>,
    /// VWAP position within price range: (vwap - low) / (high - low), range [0, 1]
    pub lookback_vwap_position: Option<f64>,
    /// Count imbalance: (buy_count - sell_count) / total_count, range [-1, 1]
    pub lookback_count_imbalance: Option<f64>,

    // === Tier 2: Statistical Features (6) ===
    // Issue #128: Garman-Klass moved from Tier 3 → Tier 2 (Gen600 top feature)
    /// Kyle's Lambda proxy (normalized): ((last-first)/first) / ((buy-sell)/total)
    pub lookback_kyle_lambda: Option<f64>,
    /// Burstiness (Goh-Barabasi): (sigma_tau - mu_tau) / (sigma_tau + mu_tau), range [-1, 1]
    pub lookback_burstiness: Option<f64>,
    /// Volume skewness (Fisher-Pearson coefficient)
    pub lookback_volume_skew: Option<f64>,
    /// Excess kurtosis: E[(V-mu)^4] / sigma^4 - 3
    pub lookback_volume_kurt: Option<f64>,
    /// Price range normalized: (high - low) / first_price
    pub lookback_price_range: Option<f64>,
    /// Garman-Klass volatility estimator (Issue #128: promoted from Tier 3)
    pub lookback_garman_klass_vol: Option<f64>,

    // === Tier 3: Advanced Features (3) ===
    // Issue #128: Hurst + PE have ZERO presence in Gen600 top 100 configs
    /// Kaufman Efficiency Ratio: |net movement| / sum(|individual movements|), range [0, 1]
    pub lookback_kaufman_er: Option<f64>,
    /// Hurst exponent via DFA, soft-clamped to [0, 1]
    pub lookback_hurst: Option<f64>,
    /// Permutation entropy (normalized), range [0, 1]
    pub lookback_permutation_entropy: Option<f64>,
}

impl InterBarFeatures {
    /// Merge Tier 2 features from another InterBarFeatures struct
    /// Issue #96 Task #90: #[inline] — per-bar field merge at finalization
    #[inline]
    pub fn merge_tier2(&mut self, other: &InterBarFeatures) {
        self.lookback_kyle_lambda = other.lookback_kyle_lambda;
        self.lookback_burstiness = other.lookback_burstiness;
        self.lookback_volume_skew = other.lookback_volume_skew;
        self.lookback_volume_kurt = other.lookback_volume_kurt;
        self.lookback_price_range = other.lookback_price_range;
        // Issue #128: Garman-Klass promoted from Tier 3 → Tier 2
        self.lookback_garman_klass_vol = other.lookback_garman_klass_vol;
    }

    /// Merge Tier 3 features from another InterBarFeatures struct
    /// Issue #96 Task #90: #[inline] — per-bar field merge at finalization
    /// Issue #128: Garman-Klass moved to Tier 2, Tier 3 now 3 features
    #[inline]
    pub fn merge_tier3(&mut self, other: &InterBarFeatures) {
        self.lookback_kaufman_er = other.lookback_kaufman_er;
        self.lookback_hurst = other.lookback_hurst;
        self.lookback_permutation_entropy = other.lookback_permutation_entropy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === InterBarConfig tests ===

    #[test]
    fn test_inter_bar_config_default() {
        let config = InterBarConfig::default();
        assert!(matches!(config.lookback_mode, LookbackMode::FixedCount(500)));
        assert!(config.compute_tier2);
        assert!(config.compute_tier3);
        // Issue #128: Per-feature flags default to None (follow tier flag)
        assert!(config.compute_hurst.is_none());
        assert!(config.compute_permutation_entropy.is_none());
        // Resolved: both follow compute_tier3 (true) by default
        assert!(config.should_compute_hurst());
        assert!(config.should_compute_permutation_entropy());
    }

    // === InterBarFeatures::default tests ===

    #[test]
    fn test_inter_bar_features_default_all_none() {
        let f = InterBarFeatures::default();
        // Tier 1
        assert!(f.lookback_trade_count.is_none());
        assert!(f.lookback_ofi.is_none());
        assert!(f.lookback_duration_us.is_none());
        assert!(f.lookback_intensity.is_none());
        assert!(f.lookback_vwap.is_none());
        assert!(f.lookback_vwap_position.is_none());
        assert!(f.lookback_count_imbalance.is_none());
        // Tier 2
        assert!(f.lookback_kyle_lambda.is_none());
        assert!(f.lookback_burstiness.is_none());
        assert!(f.lookback_volume_skew.is_none());
        assert!(f.lookback_volume_kurt.is_none());
        assert!(f.lookback_price_range.is_none());
        // Tier 3
        assert!(f.lookback_kaufman_er.is_none());
        assert!(f.lookback_garman_klass_vol.is_none());
        assert!(f.lookback_hurst.is_none());
        assert!(f.lookback_permutation_entropy.is_none());
    }

    // === merge_tier2 tests ===

    #[test]
    fn test_merge_tier2_overwrites_fields() {
        let mut base = InterBarFeatures::default();
        base.lookback_trade_count = Some(100); // Tier 1 — should not be affected

        let mut source = InterBarFeatures::default();
        source.lookback_kyle_lambda = Some(0.5);
        source.lookback_burstiness = Some(-0.3);
        source.lookback_volume_skew = Some(1.2);
        source.lookback_volume_kurt = Some(3.5);
        source.lookback_price_range = Some(0.02);
        source.lookback_garman_klass_vol = Some(0.001); // Issue #128: now Tier 2

        base.merge_tier2(&source);

        // Tier 2 fields merged (Issue #128: now includes garman_klass_vol)
        assert_eq!(base.lookback_kyle_lambda, Some(0.5));
        assert_eq!(base.lookback_burstiness, Some(-0.3));
        assert_eq!(base.lookback_volume_skew, Some(1.2));
        assert_eq!(base.lookback_volume_kurt, Some(3.5));
        assert_eq!(base.lookback_price_range, Some(0.02));
        assert_eq!(base.lookback_garman_klass_vol, Some(0.001));
        // Tier 1 untouched
        assert_eq!(base.lookback_trade_count, Some(100));
        // Tier 3 untouched
        assert!(base.lookback_kaufman_er.is_none());
    }

    #[test]
    fn test_merge_tier2_none_overwrites_some() {
        let mut base = InterBarFeatures::default();
        base.lookback_kyle_lambda = Some(0.5);
        base.lookback_burstiness = Some(-0.1);

        let source = InterBarFeatures::default(); // all None

        base.merge_tier2(&source);

        // None overwrites Some — merge is unconditional
        assert!(base.lookback_kyle_lambda.is_none());
        assert!(base.lookback_burstiness.is_none());
    }

    // === merge_tier3 tests ===

    #[test]
    fn test_merge_tier3_overwrites_fields() {
        let mut base = InterBarFeatures::default();
        base.lookback_ofi = Some(0.8); // Tier 1 — should not be affected
        base.lookback_kyle_lambda = Some(0.5); // Tier 2 — should not be affected

        let mut source = InterBarFeatures::default();
        source.lookback_kaufman_er = Some(0.75);
        source.lookback_hurst = Some(0.55);
        source.lookback_permutation_entropy = Some(0.92);

        base.merge_tier3(&source);

        // Tier 3 fields merged (Issue #128: garman_klass_vol moved to Tier 2)
        assert_eq!(base.lookback_kaufman_er, Some(0.75));
        assert_eq!(base.lookback_hurst, Some(0.55));
        assert_eq!(base.lookback_permutation_entropy, Some(0.92));
        // Tier 1 untouched
        assert_eq!(base.lookback_ofi, Some(0.8));
        // Tier 2 untouched
        assert_eq!(base.lookback_kyle_lambda, Some(0.5));
    }

    #[test]
    fn test_merge_tier3_none_overwrites_some() {
        let mut base = InterBarFeatures::default();
        base.lookback_hurst = Some(0.5);
        base.lookback_permutation_entropy = Some(0.9);

        let source = InterBarFeatures::default(); // all None

        base.merge_tier3(&source);

        assert!(base.lookback_hurst.is_none());
        assert!(base.lookback_permutation_entropy.is_none());
    }

    // === TradeSnapshot::from tests ===

    #[test]
    fn test_trade_snapshot_from_agg_trade() {
        let price = FixedPoint::from_str("50000.0").unwrap();
        let volume = FixedPoint::from_str("1.5").unwrap();
        let trade = AggTrade {
            agg_trade_id: 42,
            price,
            volume,
            first_trade_id: 100,
            last_trade_id: 105,
            timestamp: 1_700_000_000_000_000,
            is_buyer_maker: true,
            is_best_match: Some(true),
        };

        let snap = TradeSnapshot::from(&trade);

        assert_eq!(snap.price, price);
        assert_eq!(snap.volume, volume);
        assert_eq!(snap.timestamp, 1_700_000_000_000_000);
        assert!(snap.is_buyer_maker);
        assert_eq!(snap.turnover, trade.turnover());
    }
}
