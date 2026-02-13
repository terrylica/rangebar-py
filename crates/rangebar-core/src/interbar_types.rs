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
}

impl Default for InterBarConfig {
    fn default() -> Self {
        Self {
            lookback_mode: LookbackMode::FixedCount(500),
            compute_tier2: true,
            compute_tier3: true,
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
#[derive(Debug, Clone)]
pub struct TradeSnapshot {
    /// Timestamp in microseconds (matches AggTrade)
    pub timestamp: i64,
    /// Price as fixed-point
    pub price: FixedPoint,
    /// Volume as fixed-point
    pub volume: FixedPoint,
    /// Whether buyer is market maker (true = sell pressure)
    pub is_buyer_maker: bool,
    /// Turnover (price * volume) as i128 to prevent overflow
    pub turnover: i128,
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
#[derive(Debug, Clone, Default)]
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

    // === Tier 2: Statistical Features (5) ===
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

    // === Tier 3: Advanced Features (4) ===
    /// Kaufman Efficiency Ratio: |net movement| / sum(|individual movements|), range [0, 1]
    pub lookback_kaufman_er: Option<f64>,
    /// Garman-Klass volatility estimator
    pub lookback_garman_klass_vol: Option<f64>,
    /// Hurst exponent via DFA, soft-clamped to [0, 1]
    pub lookback_hurst: Option<f64>,
    /// Permutation entropy (normalized), range [0, 1]
    pub lookback_permutation_entropy: Option<f64>,
}
