//! Type definitions for range bar processing

// Issue #88: SCALE imported for i128 volume→f64 conversion
use crate::fixed_point::{FixedPoint, SCALE};
use serde::{Deserialize, Serialize};

// Re-export AggTrade and DataSource from trade.rs (Phase 2c extraction)
pub use crate::trade::{AggTrade, DataSource};

/// Range bar with OHLCV data and market microstructure enhancements
///
/// Field ordering optimized for cache locality (Issue #96 Task #85):
/// - Tier 1: OHLCV Core (48B, 1 CL)
/// - Tier 2: Volume Accumulators (96B, 1.5 CL)
/// - Tier 3: Trade Tracking (48B, 1 CL)
/// - Tier 4: Price Context (24B, partial CL)
/// - Tier 5: Microstructure (80B, 1.25 CL)
/// - Tier 6-7: Inter-Bar & Intra-Bar Features
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RangeBar {
    // === TIER 1: OHLCV CORE (48 bytes, 1 cache line) ===
    /// Opening timestamp in microseconds (first trade)
    pub open_time: i64,

    /// Closing timestamp in microseconds (last trade)
    pub close_time: i64,

    /// Opening price (first trade price)
    pub open: FixedPoint,

    /// Highest price in bar
    pub high: FixedPoint,

    /// Lowest price in bar
    pub low: FixedPoint,

    /// Closing price (breach trade price)
    pub close: FixedPoint,

    // === TIER 2: VOLUME ACCUMULATORS (96 bytes, 1.5 cache lines) ===
    /// Total volume (i128 accumulator to prevent overflow, Issue #88)
    pub volume: i128,

    /// Total turnover (sum of price * volume)
    pub turnover: i128,

    /// Volume from buy-side trades (is_buyer_maker = false)
    /// Represents aggressive buying pressure (i128 accumulator, Issue #88)
    pub buy_volume: i128,

    /// Volume from sell-side trades (is_buyer_maker = true)
    /// Represents aggressive selling pressure (i128 accumulator, Issue #88)
    pub sell_volume: i128,

    /// Turnover from buy-side trades (buy pressure)
    pub buy_turnover: i128,

    /// Turnover from sell-side trades (sell pressure)
    pub sell_turnover: i128,

    // === TIER 3: TRADE TRACKING (48 bytes, 1 cache line) ===
    /// First individual trade ID in this range bar
    pub first_trade_id: i64,

    /// Last individual trade ID in this range bar
    pub last_trade_id: i64,

    /// First aggregate trade ID in this range bar (Issue #72)
    /// Tracks the first AggTrade record that opened this bar
    pub first_agg_trade_id: i64,

    /// Last aggregate trade ID in this range bar (Issue #72)
    /// Tracks the last AggTrade record processed in this bar
    pub last_agg_trade_id: i64,

    /// Total number of individual exchange trades in this range bar
    /// Sum of individual_trade_count() from all processed AggTrade records
    pub individual_trade_count: u32,

    /// Number of AggTrade records processed to create this range bar
    /// NEW: Enables tracking of aggregation efficiency
    pub agg_record_count: u32,

    /// Number of individual buy-side trades (aggressive buying)
    pub buy_trade_count: u32,

    /// Number of individual sell-side trades (aggressive selling)
    pub sell_trade_count: u32,

    // === TIER 4: PRICE CONTEXT (24 bytes, partial cache line) ===
    /// Volume Weighted Average Price for the bar
    /// Calculated incrementally as: sum(price * volume) / sum(volume)
    pub vwap: FixedPoint,

    /// Data source this range bar was created from
    pub data_source: DataSource,

    // === TIER 5: MICROSTRUCTURE (80 bytes, 1.25 cache lines) ===
    /// Bar duration in microseconds (close_time - open_time)
    /// Reference: Easley et al. (2012) "Volume Clock"
    #[serde(default)]
    pub duration_us: i64,

    /// Order Flow Imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    /// Range: [-1.0, +1.0], Reference: Chordia et al. (2002)
    #[serde(default)]
    pub ofi: f64,

    /// VWAP-Close Deviation: (close - vwap) / (high - low)
    /// Measures intra-bar momentum, Reference: Berkowitz et al. (1988)
    #[serde(default)]
    pub vwap_close_deviation: f64,

    /// Price Impact (Amihud-style): abs(close - open) / volume
    /// Reference: Amihud (2002) illiquidity ratio
    #[serde(default)]
    pub price_impact: f64,

    /// Kyle's Lambda Proxy: (close - open) / (buy_vol - sell_vol)
    /// Market depth measure, Reference: Kyle (1985)
    #[serde(default)]
    pub kyle_lambda_proxy: f64,

    /// Trade Intensity: individual_trade_count / duration_seconds
    /// Reference: Engle & Russell (1998) ACD models
    #[serde(default)]
    pub trade_intensity: f64,

    /// Average trade size: volume / individual_trade_count
    /// Reference: Barclay & Warner (1993) stealth trading
    #[serde(default)]
    pub volume_per_trade: f64,

    /// Aggression Ratio: buy_trade_count / sell_trade_count
    /// Capped at 100.0, Reference: Lee & Ready (1991)
    #[serde(default)]
    pub aggression_ratio: f64,

    /// Aggregation Density: individual_trade_count / agg_record_count
    /// Average number of individual trades per AggTrade record (Issue #32 rename)
    /// Higher values = more fragmented trades; Lower values = more consolidated orders
    #[serde(default)]
    pub aggregation_density_f64: f64,

    /// Turnover Imbalance: (buy_turnover - sell_turnover) / total_turnover
    /// Dollar-weighted OFI, Range: [-1.0, +1.0]
    #[serde(default)]
    pub turnover_imbalance: f64,

    // === TIER 6: INTER-BAR FEATURES (Issue #59) ===
    // Computed from lookback trade window BEFORE each bar opens.
    // All fields are Option<T> to indicate when computation wasn't possible.

    // --- Tier 1: Core Features ---
    /// Number of trades in lookback window before bar opened
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_trade_count: Option<u32>,

    /// Order Flow Imbalance from lookback window: [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_ofi: Option<f64>,

    /// Duration of lookback window in microseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_duration_us: Option<i64>,

    /// Trade intensity in lookback: trades per second
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_intensity: Option<f64>,

    /// VWAP from lookback window (stored as FixedPoint raw value for serialization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_vwap_raw: Option<i64>,

    /// VWAP position within price range: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_vwap_position: Option<f64>,

    /// Count imbalance: (buy_count - sell_count) / total_count, [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_count_imbalance: Option<f64>,

    // --- Tier 2: Statistical Features ---
    /// Kyle's Lambda proxy from lookback (normalized)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_kyle_lambda: Option<f64>,

    /// Burstiness (Goh-Barabasi): [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_burstiness: Option<f64>,

    /// Volume skewness (Fisher-Pearson coefficient)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_volume_skew: Option<f64>,

    /// Excess kurtosis of volume distribution
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_volume_kurt: Option<f64>,

    /// Price range normalized by first price
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_price_range: Option<f64>,

    // --- Tier 3: Advanced Features ---
    /// Kaufman Efficiency Ratio: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_kaufman_er: Option<f64>,

    /// Garman-Klass volatility estimator
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_garman_klass_vol: Option<f64>,

    /// Hurst exponent via DFA, soft-clamped to [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_hurst: Option<f64>,

    /// Permutation entropy (normalized): [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookback_permutation_entropy: Option<f64>,

    // === TIER 7: INTRA-BAR FEATURES (Issue #59) ===
    // Computed from trades WITHIN each bar (open_time to close_time).
    // All fields are Option<T> to indicate when computation wasn't possible.

    // --- ITH Features (8) - All bounded [0, 1] ---
    /// Bull epoch density: sigmoid(epochs/trade_count, 0.5, 10)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bull_epoch_density: Option<f64>,

    /// Bear epoch density: sigmoid(epochs/trade_count, 0.5, 10)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bear_epoch_density: Option<f64>,

    /// Bull excess gain (sum): tanh-normalized to [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bull_excess_gain: Option<f64>,

    /// Bear excess gain (sum): tanh-normalized to [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bear_excess_gain: Option<f64>,

    /// Bull intervals CV: sigmoid-normalized to [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bull_cv: Option<f64>,

    /// Bear intervals CV: sigmoid-normalized to [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_bear_cv: Option<f64>,

    /// Max drawdown within bar: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_max_drawdown: Option<f64>,

    /// Max runup within bar: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_max_runup: Option<f64>,

    // --- Statistical Features (12) ---
    /// Number of trades within the bar
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_trade_count: Option<u32>,

    /// Order Flow Imbalance within bar: [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_ofi: Option<f64>,

    /// Duration of bar in microseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_duration_us: Option<i64>,

    /// Trade intensity: trades per second
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_intensity: Option<f64>,

    /// VWAP position within price range: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_vwap_position: Option<f64>,

    /// Count imbalance: (buy_count - sell_count) / total_count, [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_count_imbalance: Option<f64>,

    /// Kyle's Lambda proxy (normalized)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_kyle_lambda: Option<f64>,

    /// Burstiness (Goh-Barabasi): [-1, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_burstiness: Option<f64>,

    /// Volume skewness
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_volume_skew: Option<f64>,

    /// Volume excess kurtosis
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_volume_kurt: Option<f64>,

    /// Kaufman Efficiency Ratio: [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_kaufman_er: Option<f64>,

    /// Garman-Klass volatility estimator
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_garman_klass_vol: Option<f64>,

    // --- Complexity Features (2) ---
    /// Hurst exponent via DFA (requires >= 64 trades): [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_hurst: Option<f64>,

    /// Permutation entropy (requires >= 60 trades): [0, 1]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_permutation_entropy: Option<f64>,
}

impl RangeBar {
    /// Create new range bar from opening AggTrade record
    pub fn new(trade: &AggTrade) -> Self {
        let trade_turnover = trade.turnover();
        let individual_trades = trade.individual_trade_count() as u32;

        // Segregate order flow based on is_buyer_maker (Issue #88: i128 accumulators)
        let (buy_volume, sell_volume) = if trade.is_buyer_maker {
            (0i128, trade.volume.0 as i128) // Seller aggressive = sell pressure
        } else {
            (trade.volume.0 as i128, 0i128) // Buyer aggressive = buy pressure
        };

        let (buy_trade_count, sell_trade_count) = if trade.is_buyer_maker {
            (0, individual_trades)
        } else {
            (individual_trades, 0)
        };

        let (buy_turnover, sell_turnover) = if trade.is_buyer_maker {
            (0, trade_turnover)
        } else {
            (trade_turnover, 0)
        };

        Self {
            open_time: trade.timestamp,
            close_time: trade.timestamp,
            open: trade.price,
            high: trade.price,
            low: trade.price,
            close: trade.price,
            volume: trade.volume.0 as i128,
            turnover: trade_turnover,

            // NEW: Enhanced counting
            individual_trade_count: individual_trades,
            agg_record_count: 1, // This is the first AggTrade record
            first_trade_id: trade.first_trade_id,
            last_trade_id: trade.last_trade_id,
            // Issue #72: Track aggregate trade IDs for data integrity verification
            first_agg_trade_id: trade.agg_trade_id,
            last_agg_trade_id: trade.agg_trade_id,
            data_source: DataSource::default(),

            // Market microstructure fields
            buy_volume,
            sell_volume,
            buy_trade_count,
            sell_trade_count,
            vwap: trade.price, // Initial VWAP equals opening price
            buy_turnover,
            sell_turnover,

            // Microstructure features (Issue #25) - computed at finalization
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

            // Inter-bar features (Issue #59) - computed from lookback window
            // All None until explicitly set via set_inter_bar_features()
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

            // Intra-bar features (Issue #59) - computed from trades within bar
            // All None until explicitly set via set_intra_bar_features()
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
        }
    }

    /// Average number of individual trades per AggTrade record (aggregation density)
    pub fn aggregation_density(&self) -> f64 {
        if self.agg_record_count == 0 {
            0.0
        } else {
            self.individual_trade_count as f64 / self.agg_record_count as f64
        }
    }

    /// Update bar with new AggTrade record (always call before checking breach)
    /// Maintains market microstructure metrics incrementally
    pub fn update_with_trade(&mut self, trade: &AggTrade) {
        // Update price extremes
        if trade.price > self.high {
            self.high = trade.price;
        }
        if trade.price < self.low {
            self.low = trade.price;
        }

        // Update closing data
        self.close = trade.price;
        self.close_time = trade.timestamp;
        self.last_trade_id = trade.last_trade_id; // NEW: Track individual trade ID
        self.last_agg_trade_id = trade.agg_trade_id; // Issue #72: Track aggregate trade ID

        // Cache trade metrics for efficiency
        let trade_turnover = trade.turnover();
        let individual_trades = trade.individual_trade_count() as u32;

        // Update totals (Issue #88: i128 prevents overflow for high-volume tokens)
        self.volume += trade.volume.0 as i128;
        self.turnover += trade_turnover;

        // Enhanced counting
        self.individual_trade_count += individual_trades;
        self.agg_record_count += 1; // Track number of AggTrade records

        // === MARKET MICROSTRUCTURE INCREMENTAL UPDATES ===

        // Update order flow segregation (Issue #88: i128 accumulators)
        if trade.is_buyer_maker {
            // Seller aggressive = sell pressure
            self.sell_volume += trade.volume.0 as i128;
            self.sell_trade_count += individual_trades;
            self.sell_turnover += trade_turnover;
        } else {
            // Buyer aggressive = buy pressure
            self.buy_volume += trade.volume.0 as i128;
            self.buy_trade_count += individual_trades;
            self.buy_turnover += trade_turnover;
        }

        // Update VWAP incrementally: VWAP = total_turnover / total_volume
        // Issue #88: both turnover and volume are i128, no cast needed
        if self.volume > 0 {
            let vwap_raw = self.turnover / self.volume;
            self.vwap = FixedPoint(vwap_raw as i64); // Safe: VWAP is a price, fits i64
        }
    }

    /// Compute all microstructure features at bar finalization (Issue #25)
    ///
    /// This method computes 10 derived features from the accumulated bar state.
    /// All features use only data with timestamps <= bar.close_time (no lookahead).
    ///
    /// # Features Computed
    ///
    /// | Feature | Formula | Academic Reference |
    /// |---------|---------|-------------------|
    /// | duration_us | close_time - open_time | Easley et al. (2012) |
    /// | ofi | (buy_vol - sell_vol) / total | Chordia et al. (2002) |
    /// | vwap_close_deviation | (close - vwap) / (high - low) | Berkowitz et al. (1988) |
    /// | price_impact | abs(close - open) / volume | Amihud (2002) |
    /// | kyle_lambda_proxy | (close - open) / (buy_vol - sell_vol) | Kyle (1985) |
    /// | trade_intensity | trade_count / duration_seconds | Engle & Russell (1998) |
    /// | volume_per_trade | volume / trade_count | Barclay & Warner (1993) |
    /// | aggression_ratio | buy_trades / sell_trades | Lee & Ready (1991) |
    /// | aggregation_density_f64 | trade_count / agg_count | (proxy) |
    /// | turnover_imbalance | (buy_turn - sell_turn) / total_turn | (proxy) |
    pub fn compute_microstructure_features(&mut self) {
        // Extract values for computation (Issue #88: i128→f64 via SCALE)
        let buy_vol = self.buy_volume as f64 / SCALE as f64;
        let sell_vol = self.sell_volume as f64 / SCALE as f64;
        let total_vol = buy_vol + sell_vol;
        let volume = self.volume as f64 / SCALE as f64;
        let open = self.open.to_f64();
        let close = self.close.to_f64();
        let high = self.high.to_f64();
        let low = self.low.to_f64();
        let vwap = self.vwap.to_f64();
        let duration_us_raw = self.close_time - self.open_time;
        let trade_count = self.individual_trade_count as f64;
        let agg_count = self.agg_record_count as f64;
        let buy_turn = self.buy_turnover as f64;
        let sell_turn = self.sell_turnover as f64;
        let total_turn = (self.buy_turnover + self.sell_turnover) as f64;

        // 1. Duration (already in microseconds)
        self.duration_us = duration_us_raw;

        // 2. Order Flow Imbalance [-1, 1]
        self.ofi = if total_vol > f64::EPSILON {
            (buy_vol - sell_vol) / total_vol
        } else {
            0.0
        };

        // 3. VWAP-Close Deviation (normalized by price range)
        let range = high - low;
        self.vwap_close_deviation = if range > f64::EPSILON {
            (close - vwap) / range
        } else {
            0.0
        };

        // 4. Price Impact (Amihud-style)
        self.price_impact = if volume > f64::EPSILON {
            (close - open).abs() / volume
        } else {
            0.0
        };

        // 5. Kyle's Lambda Proxy (percentage returns formula - Issue #32)
        // Formula: ((close - open) / open) / ((buy_vol - sell_vol) / total_vol)
        // Creates dimensionally consistent ratio: percentage return per unit of normalized imbalance
        // Reference: Kyle (1985), normalized for cross-asset comparability
        let imbalance = buy_vol - sell_vol;
        let normalized_imbalance = if total_vol > f64::EPSILON {
            imbalance / total_vol
        } else {
            0.0
        };
        self.kyle_lambda_proxy =
            if normalized_imbalance.abs() > f64::EPSILON && open.abs() > f64::EPSILON {
                ((close - open) / open) / normalized_imbalance
            } else {
                0.0
            };

        // 6. Trade Intensity (trades per second)
        // Note: duration_us is in microseconds, convert to seconds
        // Issue #96: Multiply by reciprocal instead of dividing
        let duration_sec = duration_us_raw as f64 * 1e-6;
        self.trade_intensity = if duration_sec > f64::EPSILON {
            trade_count / duration_sec
        } else {
            trade_count // Instant bar = all trades at once
        };

        // 7. Volume per Trade (average trade size)
        self.volume_per_trade = if trade_count > f64::EPSILON {
            volume / trade_count
        } else {
            0.0
        };

        // 8. Aggression Ratio [0, 100] (capped)
        let sell_count = self.sell_trade_count as f64;
        self.aggression_ratio = if sell_count > f64::EPSILON {
            (self.buy_trade_count as f64 / sell_count).min(100.0)
        } else if self.buy_trade_count > 0 {
            100.0 // All buys, cap at 100
        } else {
            1.0 // No trades = neutral
        };

        // 9. Aggregation Density (Issue #32 rename: was aggregation_efficiency)
        // Measures average trades per AggTrade record (higher = more fragmented)
        self.aggregation_density_f64 = if agg_count > f64::EPSILON {
            trade_count / agg_count
        } else {
            1.0
        };

        // 10. Turnover Imbalance (dollar-weighted OFI) [-1, 1]
        self.turnover_imbalance = if total_turn.abs() > f64::EPSILON {
            (buy_turn - sell_turn) / total_turn
        } else {
            0.0
        };
    }

    /// Check if price breaches the range thresholds
    ///
    /// # Arguments
    ///
    /// * `price` - Current price to check
    /// * `upper_threshold` - Upper breach threshold (from bar open)
    /// * `lower_threshold` - Lower breach threshold (from bar open)
    ///
    /// # Returns
    ///
    /// `true` if price breaches either threshold
    /// Issue #96: #[inline] for per-trade hot path
    #[inline]
    pub fn is_breach(
        &self,
        price: FixedPoint,
        upper_threshold: FixedPoint,
        lower_threshold: FixedPoint,
    ) -> bool {
        price >= upper_threshold || price <= lower_threshold
    }

    /// Set inter-bar features from computed InterBarFeatures struct (Issue #59)
    ///
    /// This method is called when a bar is finalized with inter-bar feature
    /// computation enabled. The features are computed from trades that occurred
    /// BEFORE the bar opened, ensuring no lookahead bias.
    pub fn set_inter_bar_features(&mut self, features: &crate::interbar::InterBarFeatures) {
        // Tier 1: Core features
        self.lookback_trade_count = features.lookback_trade_count;
        self.lookback_ofi = features.lookback_ofi;
        self.lookback_duration_us = features.lookback_duration_us;
        self.lookback_intensity = features.lookback_intensity;
        self.lookback_vwap_raw = features.lookback_vwap.map(|v| v.0);
        self.lookback_vwap_position = features.lookback_vwap_position;
        self.lookback_count_imbalance = features.lookback_count_imbalance;

        // Tier 2: Statistical features
        self.lookback_kyle_lambda = features.lookback_kyle_lambda;
        self.lookback_burstiness = features.lookback_burstiness;
        self.lookback_volume_skew = features.lookback_volume_skew;
        self.lookback_volume_kurt = features.lookback_volume_kurt;
        self.lookback_price_range = features.lookback_price_range;

        // Tier 3: Advanced features
        self.lookback_kaufman_er = features.lookback_kaufman_er;
        self.lookback_garman_klass_vol = features.lookback_garman_klass_vol;
        self.lookback_hurst = features.lookback_hurst;
        self.lookback_permutation_entropy = features.lookback_permutation_entropy;
    }

    /// Set intra-bar features from computed IntraBarFeatures struct (Issue #59)
    ///
    /// This method is called when a bar is finalized with intra-bar feature
    /// computation enabled. The features are computed from trades WITHIN the bar,
    /// from open_time to close_time.
    pub fn set_intra_bar_features(&mut self, features: &crate::intrabar::IntraBarFeatures) {
        // ITH features (8)
        self.intra_bull_epoch_density = features.intra_bull_epoch_density;
        self.intra_bear_epoch_density = features.intra_bear_epoch_density;
        self.intra_bull_excess_gain = features.intra_bull_excess_gain;
        self.intra_bear_excess_gain = features.intra_bear_excess_gain;
        self.intra_bull_cv = features.intra_bull_cv;
        self.intra_bear_cv = features.intra_bear_cv;
        self.intra_max_drawdown = features.intra_max_drawdown;
        self.intra_max_runup = features.intra_max_runup;

        // Statistical features (12)
        self.intra_trade_count = features.intra_trade_count;
        self.intra_ofi = features.intra_ofi;
        self.intra_duration_us = features.intra_duration_us;
        self.intra_intensity = features.intra_intensity;
        self.intra_vwap_position = features.intra_vwap_position;
        self.intra_count_imbalance = features.intra_count_imbalance;
        self.intra_kyle_lambda = features.intra_kyle_lambda;
        self.intra_burstiness = features.intra_burstiness;
        self.intra_volume_skew = features.intra_volume_skew;
        self.intra_volume_kurt = features.intra_volume_kurt;
        self.intra_kaufman_er = features.intra_kaufman_er;
        self.intra_garman_klass_vol = features.intra_garman_klass_vol;

        // Complexity features (2)
        self.intra_hurst = features.intra_hurst;
        self.intra_permutation_entropy = features.intra_permutation_entropy;
    }
}

// Tests moved to crates/rangebar-core/tests/types_tests.rs (Phase 1a refactoring)

/// Issue #96: Edge case tests for compute_microstructure_features() epsilon guards
#[cfg(test)]
mod microstructure_edge_tests {
    use super::*;
    use crate::fixed_point::FixedPoint;

    /// Helper: create a minimal bar with given values for edge case testing
    fn bar_with(
        open: &str, close: &str, high: &str, low: &str,
        volume_raw: i128, buy_vol: i128, sell_vol: i128,
        trade_count: u32, agg_count: u32,
        buy_count: u32, sell_count: u32,
        buy_turn: i128, sell_turn: i128,
        open_time: i64, close_time: i64,
    ) -> RangeBar {
        RangeBar {
            open: FixedPoint::from_str(open).unwrap(),
            close: FixedPoint::from_str(close).unwrap(),
            high: FixedPoint::from_str(high).unwrap(),
            low: FixedPoint::from_str(low).unwrap(),
            vwap: FixedPoint::from_str(open).unwrap(),
            volume: volume_raw,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            individual_trade_count: trade_count,
            agg_record_count: agg_count,
            buy_trade_count: buy_count,
            sell_trade_count: sell_count,
            buy_turnover: buy_turn,
            sell_turnover: sell_turn,
            open_time,
            close_time,
            ..Default::default()
        }
    }

    #[test]
    fn test_zero_volume_ofi_is_zero() {
        let mut bar = bar_with(
            "50000.0", "50000.0", "50000.0", "50000.0",
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.ofi, 0.0, "OFI must be 0 when total_vol=0");
    }

    #[test]
    fn test_zero_range_vwap_deviation_is_zero() {
        // high == low → range = 0
        let mut bar = bar_with(
            "50000.0", "50000.0", "50000.0", "50000.0",
            100_000_000, 50_000_000, 50_000_000,
            2, 1, 1, 1, 100, 100, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.vwap_close_deviation, 0.0, "VWAP deviation must be 0 when range=0");
    }

    #[test]
    fn test_zero_volume_price_impact_is_zero() {
        let mut bar = bar_with(
            "50000.0", "50100.0", "50100.0", "50000.0",
            0, 0, 0, 2, 1, 1, 1, 0, 0, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.price_impact, 0.0, "Price impact must be 0 when volume=0");
    }

    #[test]
    fn test_kyle_lambda_zero_imbalance() {
        // Equal buy/sell volume → normalized_imbalance ≈ 0
        let vol = 100_000_000i128;
        let mut bar = bar_with(
            "50000.0", "50100.0", "50100.0", "50000.0",
            vol * 2, vol, vol,
            2, 1, 1, 1, 100, 100, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.kyle_lambda_proxy, 0.0, "Kyle lambda must be 0 when imbalance=0");
    }

    #[test]
    fn test_zero_duration_trade_intensity() {
        // open_time == close_time → duration_sec = 0
        let mut bar = bar_with(
            "50000.0", "50100.0", "50100.0", "50000.0",
            100_000_000, 100_000_000, 0,
            5, 1, 5, 0, 100, 0, 1000, 1000,
        );
        bar.compute_microstructure_features();
        // When duration=0, trade_intensity = trade_count (instant bar)
        assert_eq!(bar.trade_intensity, 5.0, "Intensity should be trade_count when duration=0");
    }

    #[test]
    fn test_zero_trade_count_volume_per_trade() {
        let mut bar = bar_with(
            "50000.0", "50000.0", "50000.0", "50000.0",
            100_000_000, 50_000_000, 50_000_000,
            0, 0, 0, 0, 0, 0, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.volume_per_trade, 0.0, "Volume per trade must be 0 when no trades");
    }

    #[test]
    fn test_zero_sell_count_aggression_ratio_capped() {
        let mut bar = bar_with(
            "50000.0", "50100.0", "50100.0", "50000.0",
            200_000_000, 200_000_000, 0,
            5, 1, 5, 0, 100, 0, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.aggression_ratio, 100.0, "Aggression ratio must be capped at 100 with 0 sells");
    }

    #[test]
    fn test_no_trades_aggression_ratio_neutral() {
        let mut bar = bar_with(
            "50000.0", "50000.0", "50000.0", "50000.0",
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.aggression_ratio, 1.0, "Aggression ratio must be 1.0 (neutral) with 0 trades");
    }

    #[test]
    fn test_zero_agg_count_density_defaults() {
        let mut bar = bar_with(
            "50000.0", "50000.0", "50000.0", "50000.0",
            100_000_000, 50_000_000, 50_000_000,
            2, 0, 1, 1, 100, 100, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.aggregation_density_f64, 1.0, "Density must be 1.0 when agg_count=0");
    }

    #[test]
    fn test_zero_turnover_imbalance_is_zero() {
        let mut bar = bar_with(
            "50000.0", "50100.0", "50100.0", "50000.0",
            100_000_000, 50_000_000, 50_000_000,
            2, 1, 1, 1, 0, 0, 1000, 2000,
        );
        bar.compute_microstructure_features();
        assert_eq!(bar.turnover_imbalance, 0.0, "Turnover imbalance must be 0 with zero turnover");
    }

    #[test]
    fn test_all_epsilon_guards_no_panic() {
        // Worst case: completely empty bar
        let mut bar = RangeBar::default();
        bar.compute_microstructure_features();

        // Nothing should panic; all features should be finite
        assert!(bar.ofi.is_finite(), "OFI must be finite");
        assert!(bar.vwap_close_deviation.is_finite(), "VWAP deviation must be finite");
        assert!(bar.price_impact.is_finite(), "Price impact must be finite");
        assert!(bar.kyle_lambda_proxy.is_finite(), "Kyle lambda must be finite");
        assert!(bar.trade_intensity.is_finite(), "Trade intensity must be finite");
        assert!(bar.volume_per_trade.is_finite(), "Volume per trade must be finite");
        assert!(bar.aggression_ratio.is_finite(), "Aggression ratio must be finite");
        assert!(bar.aggregation_density_f64.is_finite(), "Agg density must be finite");
        assert!(bar.turnover_imbalance.is_finite(), "Turnover imbalance must be finite");
    }
}
