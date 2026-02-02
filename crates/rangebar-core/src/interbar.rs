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
use crate::types::AggTrade;
use std::collections::VecDeque;

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
    /// Burstiness (Goh-Barabási): (σ_τ - μ_τ) / (σ_τ + μ_τ), range [-1, 1]
    pub lookback_burstiness: Option<f64>,
    /// Volume skewness (Fisher-Pearson coefficient)
    pub lookback_volume_skew: Option<f64>,
    /// Excess kurtosis: E[(V-μ)⁴] / σ⁴ - 3
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

/// Trade history ring buffer for inter-bar feature computation
#[derive(Debug, Clone)]
pub struct TradeHistory {
    /// Ring buffer of recent trades
    trades: VecDeque<TradeSnapshot>,
    /// Configuration for lookback
    config: InterBarConfig,
}

impl TradeHistory {
    /// Create new trade history with given configuration
    pub fn new(config: InterBarConfig) -> Self {
        let capacity = match &config.lookback_mode {
            LookbackMode::FixedCount(n) => *n,
            LookbackMode::FixedWindow(_) => 1000, // Initial capacity for time-based
        };
        Self {
            trades: VecDeque::with_capacity(capacity),
            config,
        }
    }

    /// Push a new trade to the history buffer
    ///
    /// Automatically prunes old entries based on lookback mode.
    pub fn push(&mut self, trade: &AggTrade) {
        let snapshot = TradeSnapshot::from(trade);
        self.trades.push_back(snapshot);
        self.prune(trade.timestamp);
    }

    /// Prune old trades based on lookback configuration
    fn prune(&mut self, current_timestamp: i64) {
        match &self.config.lookback_mode {
            LookbackMode::FixedCount(n) => {
                while self.trades.len() > *n {
                    self.trades.pop_front();
                }
            }
            LookbackMode::FixedWindow(window_us) => {
                let cutoff = current_timestamp - window_us;
                while let Some(front) = self.trades.front() {
                    if front.timestamp < cutoff {
                        self.trades.pop_front();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Get trades for lookback computation (excludes trades at or after bar_open_time)
    ///
    /// This is CRITICAL for temporal integrity - we only use trades that
    /// occurred BEFORE the current bar opened.
    pub fn get_lookback_trades(&self, bar_open_time: i64) -> Vec<&TradeSnapshot> {
        self.trades
            .iter()
            .filter(|t| t.timestamp < bar_open_time)
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
        let lookback: Vec<&TradeSnapshot> = self.get_lookback_trades(bar_open_time);

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
            lookback.iter().fold((0.0, 0.0, 0u32, 0u32, 0i128), |acc, t| {
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
        let total_volume_fp: i64 = lookback.iter().map(|t| t.volume.0).sum();
        features.lookback_vwap = Some(if total_volume_fp > 0 {
            let vwap_raw = total_turnover / (total_volume_fp as i128);
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
    fn compute_tier3_features(&self, lookback: &[&TradeSnapshot], features: &mut InterBarFeatures) {
        let n = lookback.len();

        // Collect prices for advanced features
        let prices: Vec<f64> = lookback.iter().map(|t| t.price.to_f64()).collect();

        // Kaufman Efficiency Ratio (min 2 trades)
        if n >= 2 {
            features.lookback_kaufman_er = Some(compute_kaufman_er(&prices));
        }

        // Garman-Klass volatility (min 1 trade, needs OHLC)
        if n >= 1 {
            features.lookback_garman_klass_vol = Some(compute_garman_klass(lookback));
        }

        // Hurst exponent via DFA (min 64 trades for reliable estimate)
        if n >= 64 {
            features.lookback_hurst = Some(compute_hurst_dfa(&prices));
        }

        // Permutation entropy (min 60 trades for m=3, need 10 × m! = 10 × 6 = 60)
        if n >= 60 {
            features.lookback_permutation_entropy = Some(compute_permutation_entropy(&prices));
        }
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

// ============================================================================
// Tier 2 Feature Computation Functions
// ============================================================================

/// Compute Kyle's Lambda (normalized version)
///
/// Formula: λ = ((price_end - price_start) / price_start) / ((buy_vol - sell_vol) / total_vol)
///
/// Reference: Kyle (1985), Hasbrouck (2009)
///
/// Interpretation:
/// - λ > 0: Price moves in direction of order flow (normal)
/// - λ < 0: Price moves against order flow (unusual)
/// - |λ| high: Large price impact per unit imbalance (illiquid)
fn compute_kyle_lambda(lookback: &[&TradeSnapshot]) -> f64 {
    if lookback.len() < 2 {
        return 0.0;
    }

    let first_price = lookback.first().unwrap().price.to_f64();
    let last_price = lookback.last().unwrap().price.to_f64();

    let (buy_vol, sell_vol): (f64, f64) = lookback.iter().fold((0.0, 0.0), |acc, t| {
        if t.is_buyer_maker {
            (acc.0, acc.1 + t.volume.to_f64())
        } else {
            (acc.0 + t.volume.to_f64(), acc.1)
        }
    });

    let total_vol = buy_vol + sell_vol;
    let normalized_imbalance = if total_vol > f64::EPSILON {
        (buy_vol - sell_vol) / total_vol
    } else {
        0.0
    };

    // Division by zero guards (matches existing codebase pattern)
    if normalized_imbalance.abs() > f64::EPSILON && first_price.abs() > f64::EPSILON {
        ((last_price - first_price) / first_price) / normalized_imbalance
    } else {
        0.0 // No information when imbalance is zero
    }
}

/// Compute Burstiness (Goh-Barabási)
///
/// Formula: B = (σ_τ - μ_τ) / (σ_τ + μ_τ)
///
/// Reference: Goh & Barabási (2008), EPL, Vol. 81, 48002
///
/// Interpretation:
/// - B = -1: Perfectly regular (periodic) arrivals
/// - B = 0: Poisson process
/// - B = +1: Maximally bursty
fn compute_burstiness(lookback: &[&TradeSnapshot]) -> f64 {
    if lookback.len() < 2 {
        return 0.0;
    }

    // Compute inter-arrival times (microseconds)
    let inter_arrivals: Vec<f64> = lookback
        .windows(2)
        .map(|w| (w[1].timestamp - w[0].timestamp) as f64)
        .collect();

    let n = inter_arrivals.len() as f64;
    let mu = inter_arrivals.iter().sum::<f64>() / n;

    let variance = inter_arrivals.iter().map(|t| (t - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();

    let denominator = sigma + mu;
    if denominator > f64::EPSILON {
        (sigma - mu) / denominator
    } else {
        0.0 // All trades at same timestamp
    }
}

/// Compute volume moments (skewness and excess kurtosis)
///
/// Skewness: E[(V-μ)³] / σ³ (Fisher-Pearson coefficient)
/// Excess Kurtosis: E[(V-μ)⁴] / σ⁴ - 3 (normal distribution = 0)
fn compute_volume_moments(lookback: &[&TradeSnapshot]) -> (f64, f64) {
    let volumes: Vec<f64> = lookback.iter().map(|t| t.volume.to_f64()).collect();
    let n = volumes.len() as f64;

    if n < 3.0 {
        return (0.0, 0.0);
    }

    let mu = volumes.iter().sum::<f64>() / n;

    // Central moments
    let m2 = volumes.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
    let m3 = volumes.iter().map(|v| (v - mu).powi(3)).sum::<f64>() / n;
    let m4 = volumes.iter().map(|v| (v - mu).powi(4)).sum::<f64>() / n;

    let sigma = m2.sqrt();

    if sigma < f64::EPSILON {
        return (0.0, 0.0); // All same volume
    }

    let skewness = m3 / sigma.powi(3);
    let kurtosis = m4 / sigma.powi(4) - 3.0; // Excess kurtosis

    (skewness, kurtosis)
}

// ============================================================================
// Tier 3 Feature Computation Functions
// ============================================================================

/// Compute Kaufman Efficiency Ratio
///
/// Formula: ER = |net movement| / sum(|individual movements|)
///
/// Reference: Kaufman (1995) - Smarter Trading
///
/// Range: [0, 1] where 1 = perfect trend, 0 = pure noise
fn compute_kaufman_er(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let net_movement = (prices.last().unwrap() - prices.first().unwrap()).abs();

    let volatility: f64 = prices.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

    if volatility > f64::EPSILON {
        net_movement / volatility
    } else {
        0.0 // No movement
    }
}

/// Compute Garman-Klass volatility estimator
///
/// Formula: σ² = 0.5 × ln(H/L)² - (2×ln(2) - 1) × ln(C/O)²
///
/// Reference: Garman & Klass (1980), Journal of Business, vol. 53, no. 1
///
/// Exact coefficient: (2×ln(2) - 1) ≈ 0.386294 (computed, not hardcoded)
fn compute_garman_klass(lookback: &[&TradeSnapshot]) -> f64 {
    if lookback.is_empty() {
        return 0.0;
    }

    // Compute OHLC from lookback window
    let o = lookback.first().unwrap().price.to_f64();
    let c = lookback.last().unwrap().price.to_f64();
    let (l, h) = lookback.iter().fold((f64::MAX, f64::MIN), |acc, t| {
        let p = t.price.to_f64();
        (acc.0.min(p), acc.1.max(p))
    });

    // Guard: prices must be positive
    if o <= f64::EPSILON || l <= f64::EPSILON || h <= f64::EPSILON {
        return 0.0;
    }

    let log_hl = (h / l).ln();
    let log_co = (c / o).ln();

    // Use exact coefficient derivation, not magic number
    let coef = 2.0 * 2.0_f64.ln() - 1.0; // ≈ 0.386294

    let variance = 0.5 * log_hl.powi(2) - coef * log_co.powi(2);

    // Variance can be negative due to the subtractive term
    if variance > 0.0 {
        variance.sqrt()
    } else {
        0.0 // Return 0 for unreliable estimate
    }
}

/// Compute Hurst exponent via Detrended Fluctuation Analysis (DFA)
///
/// Reference: Peng et al. (1994), Nature, 356, 168-170
///
/// Interpretation:
/// - H < 0.5: Anti-correlated (mean-reverting)
/// - H = 0.5: Random walk
/// - H > 0.5: Positively correlated (trending)
///
/// Output: soft-clamped to [0, 1] for ML consumption
fn compute_hurst_dfa(prices: &[f64]) -> f64 {
    const MIN_SAMPLES: usize = 64;
    if prices.len() < MIN_SAMPLES {
        return 0.5; // Neutral (insufficient data)
    }

    // Step 1: Compute profile (cumulative deviation from mean)
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let profile: Vec<f64> = prices
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x - mean;
            Some(*acc)
        })
        .collect();

    let n = profile.len();

    // Step 2-5: Compute F(n) for multiple box sizes
    let min_box = 4;
    let max_box = n / 4;
    if max_box < min_box {
        return 0.5;
    }

    let mut log_n = Vec::new();
    let mut log_f = Vec::new();

    // Generate ~10-20 box sizes logarithmically spaced
    let num_scales = ((max_box as f64).ln() - (min_box as f64).ln()) / 0.25;
    let num_scales = (num_scales as usize).max(4).min(20);

    for i in 0..num_scales {
        let box_size = (min_box as f64
            * ((max_box as f64 / min_box as f64).powf(i as f64 / (num_scales - 1) as f64)))
            as usize;
        let box_size = box_size.max(min_box).min(max_box);

        let f_n = compute_dfa_fluctuation(&profile, box_size);
        if f_n > f64::EPSILON {
            log_n.push((box_size as f64).ln());
            log_f.push(f_n.ln());
        }
    }

    if log_n.len() < 4 {
        return 0.5;
    }

    // Step 6: Linear regression to get slope (Hurst exponent)
    let hurst = linear_regression_slope(&log_n, &log_f);

    // Soft clamp to [0, 1] using tanh (from trading-fitness pattern)
    soft_clamp_hurst(hurst)
}

/// Compute DFA fluctuation for given box size
fn compute_dfa_fluctuation(profile: &[f64], box_size: usize) -> f64 {
    let n = profile.len();
    let num_boxes = n / box_size;
    if num_boxes == 0 {
        return 0.0;
    }

    let mut total_variance = 0.0;

    for i in 0..num_boxes {
        let start = i * box_size;
        let end = start + box_size;
        let segment = &profile[start..end];

        // Fit linear trend: y = a + b*x
        let (a, b) = linear_fit(segment);

        // Compute variance of detrended segment
        let variance: f64 = segment
            .iter()
            .enumerate()
            .map(|(j, &y)| {
                let trend = a + b * (j as f64);
                (y - trend).powi(2)
            })
            .sum::<f64>()
            / box_size as f64;

        total_variance += variance;
    }

    (total_variance / num_boxes as f64).sqrt()
}

/// Least squares fit: y = a + b*x where x = 0, 1, 2, ...
fn linear_fit(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let sum_x = (n - 1.0) * n / 2.0;
    let sum_x2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = y.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return (sum_y / n, 0.0); // Flat line
    }

    let b = (n * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n;
    (a, b)
}

/// Simple least squares slope
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let num: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let denom: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();

    if denom.abs() < f64::EPSILON {
        0.5
    } else {
        num / denom
    }
}

/// Soft clamp Hurst to [0, 1] using tanh
///
/// Formula: 0.5 + 0.5 × tanh((x - 0.5) × 4)
///
/// Maps 0.5 → 0.5, and asymptotically approaches 0 or 1 for extreme values
fn soft_clamp_hurst(h: f64) -> f64 {
    0.5 + 0.5 * ((h - 0.5) * 4.0).tanh()
}

/// Compute Permutation Entropy
///
/// Formula: H_PE = -Σ p_π × ln(p_π) / ln(m!)
///
/// Reference: Bandt & Pompe (2002), Phys. Rev. Lett. 88, 174102
///
/// Output range: [0, 1] where 0 = deterministic, 1 = completely random
fn compute_permutation_entropy(prices: &[f64]) -> f64 {
    const M: usize = 3; // Embedding dimension (Bandt & Pompe recommend 3-7)
    const MIN_SAMPLES: usize = 60; // Rule of thumb: 10 × m! = 10 × 6 = 60 for m=3

    if prices.len() < MIN_SAMPLES {
        return 1.0; // Insufficient data → max entropy (no information)
    }

    // Count occurrences of each permutation pattern
    // For m=3, there are 3! = 6 possible patterns
    let mut pattern_counts: [usize; 6] = [0; 6];
    let n_patterns = prices.len() - M + 1;

    for i in 0..n_patterns {
        let window = &prices[i..i + M];
        let pattern_idx = ordinal_pattern_index_m3(window[0], window[1], window[2]);
        pattern_counts[pattern_idx] += 1;
    }

    // Compute Shannon entropy of pattern distribution
    let total = n_patterns as f64;
    let entropy: f64 = pattern_counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum();

    // Normalize by maximum possible entropy: ln(3!) = ln(6)
    let max_entropy = 6.0_f64.ln(); // ≈ 1.7918

    entropy / max_entropy
}

/// Get ordinal pattern index for m=3 (0-5)
///
/// Patterns (lexicographic order):
/// 0: 012 (a ≤ b ≤ c)
/// 1: 021 (a ≤ c < b)
/// 2: 102 (b < a ≤ c)
/// 3: 120 (b ≤ c < a)
/// 4: 201 (c < a ≤ b)
/// 5: 210 (c < b < a)
fn ordinal_pattern_index_m3(a: f64, b: f64, c: f64) -> usize {
    if a <= b {
        if b <= c {
            0
        } else if a <= c {
            1
        } else {
            4
        }
    } else if a <= c {
        2
    } else if b <= c {
        3
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test trades
    fn create_test_snapshot(timestamp: i64, price: f64, volume: f64, is_buyer_maker: bool) -> TradeSnapshot {
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
                volume: FixedPoint(100000000), // 1.0
                first_trade_id: i,
                last_trade_id: i,
                timestamp: i * 1000,
                is_buyer_maker: false, // Buy
                is_best_match: None,
            };
            history.push(&trade);
        }

        let features = history.compute_features(10000);

        assert!((features.lookback_ofi.unwrap() - 1.0).abs() < f64::EPSILON,
            "OFI should be 1.0 for all buys, got {}", features.lookback_ofi.unwrap());
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

        assert!((features.lookback_ofi.unwrap() - (-1.0)).abs() < f64::EPSILON,
            "OFI should be -1.0 for all sells, got {}", features.lookback_ofi.unwrap());
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

        assert!(features.lookback_ofi.unwrap().abs() < f64::EPSILON,
            "OFI should be 0.0 for balanced volumes, got {}", features.lookback_ofi.unwrap());
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

        // Perfectly regular: σ = 0 → B = -1
        assert!((b - (-1.0)).abs() < 0.01,
            "Burstiness should be -1 for regular intervals, got {}", b);
    }

    // ========== Kaufman ER Tests ==========

    #[test]
    fn test_kaufman_er_perfect_trend() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let er = compute_kaufman_er(&prices);

        assert!((er - 1.0).abs() < f64::EPSILON,
            "Kaufman ER should be 1.0 for perfect trend, got {}", er);
    }

    #[test]
    fn test_kaufman_er_round_trip() {
        let prices = vec![100.0, 102.0, 104.0, 102.0, 100.0];
        let er = compute_kaufman_er(&prices);

        assert!(er.abs() < f64::EPSILON,
            "Kaufman ER should be 0.0 for round trip, got {}", er);
    }

    // ========== Permutation Entropy Tests ==========

    #[test]
    fn test_permutation_entropy_monotonic() {
        // Strictly increasing: only pattern 012 appears → H = 0
        let prices: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pe = compute_permutation_entropy(&prices);

        assert!(pe.abs() < f64::EPSILON,
            "PE should be 0 for monotonic, got {}", pe);
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
            assert!(trade.timestamp < 2000,
                "Trade at {} should be before bar open at 2000", trade.timestamp);
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
                volume: FixedPoint(((i % 10 + 1) as i64) * 100000000),
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

        assert!(imb >= -1.0 && imb <= 1.0,
            "Count imbalance should be in [-1, 1], got {}", imb);
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

        assert!(pos >= 0.0 && pos <= 1.0,
            "VWAP position should be in [0, 1], got {}", pos);
    }

    #[test]
    fn test_hurst_soft_clamp_bounded() {
        // Test with extreme input values
        // Note: tanh approaches 0 and 1 asymptotically, so we use >= and <=
        for raw_h in [-10.0, -1.0, 0.0, 0.5, 1.0, 2.0, 10.0] {
            let clamped = soft_clamp_hurst(raw_h);
            assert!(clamped >= 0.0 && clamped <= 1.0,
                "Hurst {} soft-clamped to {} should be in [0, 1]", raw_h, clamped);
        }

        // Verify 0.5 maps to 0.5 exactly
        let h_half = soft_clamp_hurst(0.5);
        assert!((h_half - 0.5).abs() < f64::EPSILON,
            "Hurst 0.5 should map to 0.5, got {}", h_half);
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_empty_lookback() {
        let history = TradeHistory::new(InterBarConfig::default());
        let features = history.compute_features(1000);

        assert!(features.lookback_trade_count.is_none() || features.lookback_trade_count == Some(0));
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
        // Equal buy/sell → imbalance = 0 → should return 0, not infinity
        let t0 = create_test_snapshot(0, 100.0, 1.0, false); // buy
        let t1 = create_test_snapshot(1000, 102.0, 1.0, true); // sell
        let lookback: Vec<&TradeSnapshot> = vec![&t0, &t1];

        let lambda = compute_kyle_lambda(&lookback);

        assert!(lambda.is_finite(), "Kyle lambda should be finite, got {}", lambda);
        assert!(lambda.abs() < f64::EPSILON, "Kyle lambda should be 0 for zero imbalance");
    }
}
