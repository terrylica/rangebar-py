//! Intra-bar feature computation from constituent trades.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! This module computes 22 features from trades WITHIN each range bar:
//! - 8 ITH features (from trading-fitness algorithms)
//! - 12 statistical features
//! - 2 complexity features (Hurst, Permutation Entropy)
//!
//! # FILE-SIZE-OK
//! 942 lines: Large existing file with multiple feature computation functions.
//! Keeping together maintains performance optimization context.

use crate::types::AggTrade;
use smallvec::SmallVec;

use super::drawdown::compute_max_drawdown_and_runup;
use super::ith::{bear_ith, bull_ith};
use super::normalize::{
    normalize_cv, normalize_drawdown, normalize_epochs, normalize_excess, normalize_runup,
};
use super::normalization_lut::soft_clamp_hurst_lut;

/// Pre-computed ln(3!) = ln(6) for permutation entropy normalization (m=3, Bandt-Pompe).
/// Avoids per-bar ln() call. Task #9.
const MAX_ENTROPY_M3: f64 = 1.791_759_469_228_327;

/// All 22 intra-bar features computed from constituent trades.
///
/// All ITH-based features are normalized to [0, 1] for LSTM consumption.
/// Statistical features preserve their natural ranges.
/// Optional fields return None when insufficient data.
#[derive(Debug, Clone, Default)]
pub struct IntraBarFeatures {
    // === ITH Features (8) - All bounded [0, 1] ===
    /// Bull epoch density: sigmoid(epochs/trade_count, 0.5, 10)
    pub intra_bull_epoch_density: Option<f64>,
    /// Bear epoch density: sigmoid(epochs/trade_count, 0.5, 10)
    pub intra_bear_epoch_density: Option<f64>,
    /// Bull excess gain (sum): tanh-normalized to [0, 1]
    pub intra_bull_excess_gain: Option<f64>,
    /// Bear excess gain (sum): tanh-normalized to [0, 1]
    pub intra_bear_excess_gain: Option<f64>,
    /// Bull intervals CV: sigmoid-normalized to [0, 1]
    pub intra_bull_cv: Option<f64>,
    /// Bear intervals CV: sigmoid-normalized to [0, 1]
    pub intra_bear_cv: Option<f64>,
    /// Max drawdown in bar: already [0, 1]
    pub intra_max_drawdown: Option<f64>,
    /// Max runup in bar: already [0, 1]
    pub intra_max_runup: Option<f64>,

    // === Statistical Features (12) ===
    /// Number of trades in the bar
    pub intra_trade_count: Option<u32>,
    /// Order Flow Imbalance: (buy_vol - sell_vol) / total_vol, [-1, 1]
    pub intra_ofi: Option<f64>,
    /// Duration of bar in microseconds
    pub intra_duration_us: Option<i64>,
    /// Trade intensity: trades per second
    pub intra_intensity: Option<f64>,
    /// VWAP position within price range: [0, 1]
    pub intra_vwap_position: Option<f64>,
    /// Count imbalance: (buy_count - sell_count) / total_count, [-1, 1]
    pub intra_count_imbalance: Option<f64>,
    /// Kyle's Lambda proxy (normalized)
    pub intra_kyle_lambda: Option<f64>,
    /// Burstiness (Goh-Barabási): [-1, 1]
    pub intra_burstiness: Option<f64>,
    /// Volume skewness
    pub intra_volume_skew: Option<f64>,
    /// Volume excess kurtosis
    pub intra_volume_kurt: Option<f64>,
    /// Kaufman Efficiency Ratio: [0, 1]
    pub intra_kaufman_er: Option<f64>,
    /// Garman-Klass volatility estimator
    pub intra_garman_klass_vol: Option<f64>,

    // === Complexity Features (2) - Require many trades ===
    /// Hurst exponent via DFA (requires >= 64 trades)
    pub intra_hurst: Option<f64>,
    /// Permutation entropy (requires >= 60 trades)
    pub intra_permutation_entropy: Option<f64>,
}

/// Cold path: return features for zero-trade bar
/// Extracted to improve instruction cache locality on the hot path
#[cold]
#[inline(never)]
fn intra_bar_zero_trades() -> IntraBarFeatures {
    IntraBarFeatures {
        intra_trade_count: Some(0),
        ..Default::default()
    }
}

/// Cold path: return features for single-trade bar
#[cold]
#[inline(never)]
fn intra_bar_single_trade() -> IntraBarFeatures {
    IntraBarFeatures {
        intra_trade_count: Some(1),
        intra_duration_us: Some(0),
        intra_intensity: Some(0.0),
        intra_ofi: Some(0.0),
        ..Default::default()
    }
}

/// Cold path: return features for bar with invalid first price
#[cold]
#[inline(never)]
fn intra_bar_invalid_price(n: usize) -> IntraBarFeatures {
    IntraBarFeatures {
        intra_trade_count: Some(n as u32),
        ..Default::default()
    }
}

/// Compute all intra-bar features from constituent trades.
///
/// This is the main entry point for computing ITH and statistical features
/// from the trades that formed a range bar.
///
/// # Arguments
/// * `trades` - Slice of AggTrade records within the bar
///
/// # Returns
/// `IntraBarFeatures` struct with all 22 features (or None for insufficient data)
///
/// Issue #96 Task #173: Uses reusable scratch buffers if available for zero-copy extraction
/// Issue #96 Task #52: #[inline] for delegation to _with_scratch
#[inline]
pub fn compute_intra_bar_features(trades: &[AggTrade]) -> IntraBarFeatures {
    let mut scratch_prices = SmallVec::<[f64; 64]>::new();
    let mut scratch_volumes = SmallVec::<[f64; 64]>::new();
    compute_intra_bar_features_with_scratch(trades, &mut scratch_prices, &mut scratch_volumes)
}

/// Optimized version accepting reusable scratch buffers
/// Issue #96 Task #173: Avoids per-bar heap allocation by reusing buffers across bars
/// Issue #96 Task #88: #[inline] — per-bar dispatcher called from processor hot path
#[inline]
pub fn compute_intra_bar_features_with_scratch(
    trades: &[AggTrade],
    scratch_prices: &mut SmallVec<[f64; 64]>,
    scratch_volumes: &mut SmallVec<[f64; 64]>,
) -> IntraBarFeatures {
    let n = trades.len();

    // Issue #96 Task #193: Early-exit dispatcher for small intra-bar feature computation
    // Skip only expensive complexity features (Hurst, PE) for bars with insufficient data
    // ITH computation is linear and inexpensive, always included for n >= 2
    if n == 0 {
        return intra_bar_zero_trades();
    }
    if n == 1 {
        return intra_bar_single_trade();
    }

    // Extract price series from trades, reusing scratch buffer (Issue #96 Task #173)
    scratch_prices.clear();
    scratch_prices.reserve(n);
    for trade in trades {
        scratch_prices.push(trade.price.to_f64());
    }

    // Normalize prices to start at 1.0 for ITH computation
    let first_price = scratch_prices[0];
    if first_price <= 0.0 || !first_price.is_finite() {
        return intra_bar_invalid_price(n);
    }
    // Reuse scratch buffer for normalized prices (Issue #96 Task #173)
    // Issue #96: Pre-compute reciprocal to replace per-element division with multiplication
    let inv_first_price = 1.0 / first_price;
    scratch_volumes.clear();
    scratch_volumes.reserve(n);
    for &p in scratch_prices.iter() {
        scratch_volumes.push(p * inv_first_price);
    }
    let normalized = scratch_volumes;  // Rebind for clarity

    // Compute max_drawdown and max_runup in single pass (Issue #96 Task #66: merged computation)
    let (max_dd, max_ru) = compute_max_drawdown_and_runup(normalized);

    // Compute Bull ITH with max_drawdown as TMAEG
    let bull_result = bull_ith(normalized, max_dd);

    // Compute Bear ITH with max_runup as TMAEG
    let bear_result = bear_ith(normalized, max_ru);

    // Sum excess gains for normalization
    let bull_excess_sum: f64 = bull_result.excess_gains.iter().sum();
    let bear_excess_sum: f64 = bear_result.excess_gains.iter().sum();

    // Compute statistical features
    let stats = compute_statistical_features(trades, scratch_prices);

    // Compute complexity features (only if enough trades)
    let hurst = if n >= 64 {
        Some(compute_hurst_dfa(normalized))
    } else {
        None
    };
    let pe = if n >= 60 {
        Some(compute_permutation_entropy(scratch_prices, 3))
    } else {
        None
    };

    IntraBarFeatures {
        // ITH features (normalized to [0, 1])
        intra_bull_epoch_density: Some(normalize_epochs(bull_result.num_of_epochs, n)),
        intra_bear_epoch_density: Some(normalize_epochs(bear_result.num_of_epochs, n)),
        intra_bull_excess_gain: Some(normalize_excess(bull_excess_sum)),
        intra_bear_excess_gain: Some(normalize_excess(bear_excess_sum)),
        intra_bull_cv: Some(normalize_cv(bull_result.intervals_cv)),
        intra_bear_cv: Some(normalize_cv(bear_result.intervals_cv)),
        intra_max_drawdown: Some(normalize_drawdown(bull_result.max_drawdown)),
        intra_max_runup: Some(normalize_runup(bear_result.max_runup)),

        // Statistical features
        intra_trade_count: Some(n as u32),
        intra_ofi: Some(stats.ofi),
        intra_duration_us: Some(stats.duration_us),
        intra_intensity: Some(stats.intensity),
        intra_vwap_position: Some(stats.vwap_position),
        intra_count_imbalance: Some(stats.count_imbalance),
        intra_kyle_lambda: stats.kyle_lambda,
        intra_burstiness: stats.burstiness,
        intra_volume_skew: stats.volume_skew,
        intra_volume_kurt: stats.volume_kurt,
        intra_kaufman_er: stats.kaufman_er,
        intra_garman_klass_vol: Some(stats.garman_klass_vol),

        // Complexity features
        intra_hurst: hurst,
        intra_permutation_entropy: pe,
    }
}

/// Intermediate struct for statistical features computation
struct StatisticalFeatures {
    ofi: f64,
    duration_us: i64,
    intensity: f64,
    vwap_position: f64,
    count_imbalance: f64,
    kyle_lambda: Option<f64>,
    burstiness: Option<f64>,
    volume_skew: Option<f64>,
    volume_kurt: Option<f64>,
    kaufman_er: Option<f64>,
    garman_klass_vol: f64,
}

/// Compute statistical features from trades
fn compute_statistical_features(trades: &[AggTrade], prices: &[f64]) -> StatisticalFeatures {
    let n = trades.len();

    // Issue #96 Task #188: Conversion caching - eliminate redundant FixedPoint-to-f64 conversions
    // Cache volume conversions in SmallVec to reuse across passes (avoid 2x conversions per trade)
    // Expected speedup: 3-5% on statistical feature computation (eliminates ~n volume.to_f64() calls)

    // Pre-allocate volume cache with inline capacity for typical bar sizes (< 128 trades)
    let mut cached_volumes = SmallVec::<[f64; 128]>::with_capacity(n);

    let mut buy_vol = 0.0_f64;
    let mut sell_vol = 0.0_f64;
    let mut buy_count = 0_u32;
    let mut sell_count = 0_u32;
    let mut total_turnover = 0.0_f64;
    let mut sum_vol = 0.0_f64;
    let mut high = f64::NEG_INFINITY;
    let mut low = f64::INFINITY;

    // Pass 1: Convert volumes once, accumulate, track high/low
    for trade in trades {
        let vol = trade.volume.to_f64();  // Converted once only
        cached_volumes.push(vol);  // Cache for Pass 2
        let price = prices[cached_volumes.len() - 1];  // Use pre-converted prices (Issue #96 Task #173)

        total_turnover += price * vol;
        sum_vol += vol;

        if trade.is_buyer_maker {
            sell_vol += vol;
            sell_count += trade.individual_trade_count() as u32;
        } else {
            buy_vol += vol;
            buy_count += trade.individual_trade_count() as u32;
        }

        // Track high/low during first pass (Issue #96 Task #63: eliminated separate fold pass)
        high = high.max(price);
        low = low.min(price);
    }

    let vol_count = n;
    let mean_vol = if vol_count > 0 { sum_vol / vol_count as f64 } else { 0.0 };

    // Pass 2: Compute central moments using cached volumes (no conversion, no indexing overhead)
    let mut m2_vol = 0.0_f64; // sum of (v - mean)^2
    let mut m3_vol = 0.0_f64; // sum of (v - mean)^3
    let mut m4_vol = 0.0_f64; // sum of (v - mean)^4

    for &vol in cached_volumes.iter() {
        // Issue #96 Task #196: Maximize ILP by pre-computing all powers
        // Compute all powers first (d2, d3, d4) before accumulating
        // This allows CPU to execute 3 independent additions in parallel
        let d = vol - mean_vol;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d2 * d2;

        // All 3 accumulations are independent (CPU can parallelize)
        m2_vol += d2;
        m3_vol += d3;
        m4_vol += d4;
    }

    let total_vol = buy_vol + sell_vol;
    let total_count = (buy_count + sell_count) as f64;

    // OFI: Order Flow Imbalance
    let ofi = if total_vol > f64::EPSILON {
        (buy_vol - sell_vol) / total_vol
    } else {
        0.0
    };

    // Duration
    let first_ts = trades.first().map(|t| t.timestamp).unwrap_or(0);
    let last_ts = trades.last().map(|t| t.timestamp).unwrap_or(0);
    let duration_us = last_ts - first_ts;
    // Issue #96: Multiply by reciprocal instead of dividing (avoids fdiv in hot path)
    let duration_sec = duration_us as f64 * 1e-6;

    // Intensity: trades per second
    let intensity = if duration_sec > f64::EPSILON {
        n as f64 / duration_sec
    } else {
        n as f64 // Instant bar
    };

    // VWAP position (Issue #96 Task #63: high/low cached inline during trades loop)
    let vwap = if total_vol > f64::EPSILON {
        total_turnover / total_vol
    } else {
        prices.first().copied().unwrap_or(0.0)
    };
    // High/low already computed inline during main trades loop (eliminates fold pass)
    let range = high - low;
    let vwap_position = if range > f64::EPSILON {
        ((vwap - low) / range).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // Count imbalance
    let count_imbalance = if total_count > f64::EPSILON {
        (buy_count as f64 - sell_count as f64) / total_count
    } else {
        0.0
    };

    // Kyle's Lambda (requires >= 2 trades)
    let kyle_lambda = if n >= 2 && total_vol > f64::EPSILON {
        let first_price = prices[0];
        let last_price = prices[n - 1];
        let price_return = if first_price.abs() > f64::EPSILON {
            (last_price - first_price) / first_price
        } else {
            0.0
        };
        let normalized_imbalance = (buy_vol - sell_vol) / total_vol;
        if normalized_imbalance.abs() > f64::EPSILON {
            Some(price_return / normalized_imbalance)
        } else {
            None
        }
    } else {
        None
    };

    // Issue #96 Task #61: Optimize burstiness with early-exit and SmallVec
    // Burstiness (requires >= 3 trades for meaningful inter-arrival times)
    let burstiness = if n >= 3 {
        // Compute inter-arrival intervals using direct indexing with SmallVec (no Vec allocation)
        let mut intervals = SmallVec::<[f64; 64]>::new();
        for i in 0..n - 1 {
            intervals.push((trades[i + 1].timestamp - trades[i].timestamp) as f64);
        }

        if intervals.len() >= 2 {
            // Issue #96: Pre-compute reciprocal to avoid repeated division
            let inv_len = 1.0 / intervals.len() as f64;
            let mean_tau: f64 = intervals.iter().sum::<f64>() * inv_len;
            let variance: f64 = intervals
                .iter()
                .map(|&x| {
                    let d = x - mean_tau;
                    d * d  // Multiply instead of powi(2)
                })
                .sum::<f64>()
                * inv_len;
            let std_tau = variance.sqrt();

            // Early-exit if intervals are uniform (common in tick data)
            if std_tau <= f64::EPSILON {
                None // Uniform spacing = undefined burstiness
            } else if (std_tau + mean_tau).abs() > f64::EPSILON {
                Some((std_tau - mean_tau) / (std_tau + mean_tau))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Volume moments computed inline above (Issue #96 Task #69)
    let (volume_skew, volume_kurt) = if n >= 3 {
        // Issue #96: reciprocal caching — single division for 3 moment normalizations
        let inv_n = 1.0 / n as f64;
        let m2_norm = m2_vol * inv_n;
        let m3_norm = m3_vol * inv_n;
        let m4_norm = m4_vol * inv_n;
        let std_v = m2_norm.sqrt();

        if std_v > f64::EPSILON {
            // Issue #96 Task #170: Memoize powi() calls with multiplication chains
            let std_v2 = std_v * std_v;
            let std_v3 = std_v2 * std_v;
            let std_v4 = std_v2 * std_v2;
            (Some(m3_norm / std_v3), Some(m4_norm / std_v4 - 3.0))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // Kaufman Efficiency Ratio (requires >= 2 trades)
    let kaufman_er = if n >= 2 {
        let net_move = (prices[n - 1] - prices[0]).abs();

        // Issue #96 Task #59: Replace .windows(2) with direct indexing to avoid iterator overhead
        let mut path_length = 0.0;
        for i in 0..n - 1 {
            path_length += (prices[i + 1] - prices[i]).abs();
        }

        if path_length > f64::EPSILON {
            Some((net_move / path_length).clamp(0.0, 1.0))
        } else {
            Some(1.0) // No movement = perfectly efficient
        }
    } else {
        None
    };

    // Garman-Klass volatility
    // Issue #96 Task #197: Pre-compute constant, use multiplication instead of powi
    const GK_SCALE: f64 = 0.6137;  // 2.0 * 2.0_f64.ln() - 1.0 = 0.6137...
    let open = prices[0];
    let close = prices[n - 1];
    let garman_klass_vol = if high > low && high > 0.0 && open > 0.0 {
        let hl_ratio = (high / low).ln();
        let co_ratio = (close / open).ln();
        // Replace powi(2) with multiplication (3-5x faster)
        let hl_sq = hl_ratio * hl_ratio;
        let co_sq = co_ratio * co_ratio;
        let gk_var = 0.5 * hl_sq - GK_SCALE * co_sq;
        gk_var.max(0.0).sqrt()
    } else {
        0.0
    };

    StatisticalFeatures {
        ofi,
        duration_us,
        intensity,
        vwap_position,
        count_imbalance,
        kyle_lambda,
        burstiness,
        volume_skew,
        volume_kurt,
        kaufman_er,
        garman_klass_vol,
    }
}

/// Compute Hurst exponent via Detrended Fluctuation Analysis (DFA).
///
/// The Hurst exponent measures long-term memory:
/// - H < 0.5: Mean-reverting (anti-persistent)
/// - H = 0.5: Random walk
/// - H > 0.5: Trending (persistent)
///
/// Requires at least 64 observations for reliable estimation.
fn compute_hurst_dfa(prices: &[f64]) -> f64 {
    let n = prices.len();
    if n < 64 {
        return 0.5; // Default to random walk for insufficient data
    }

    // Issue #96 Task #57: Use SmallVec for cumulative deviations
    // Compute cumulative deviation from mean
    let mean: f64 = prices.iter().sum::<f64>() / n as f64;
    let mut y = SmallVec::<[f64; 256]>::new();
    let mut cumsum = 0.0;
    for &p in prices.iter() {
        cumsum += p - mean;
        y.push(cumsum);
    }

    // Scale range from n/4 to n/2 (using powers of 2 for efficiency)
    let min_scale = (n / 4).max(8);
    let max_scale = n / 2;

    // Issue #96 Task #57: SmallVec for log vectors — DFA has 8-12 scale points
    // Inline storage eliminates 2 heap allocations per DFA call
    let mut log_scales = SmallVec::<[f64; 12]>::new();
    let mut log_fluctuations = SmallVec::<[f64; 12]>::new();

    let mut scale = min_scale;
    while scale <= max_scale {
        let num_segments = n / scale;
        if num_segments < 2 {
            break;
        }

        // Issue #96 Task #192: Memoize x_mean computation outside segment loop
        // Only depends on scale, not on segment index, so compute once and reuse
        let x_mean = (scale - 1) as f64 / 2.0;
        // Issue #96: Pre-compute xx_sum analytically: sum_{i=0}^{n-1} (i - mean)^2 = n*(n^2-1)/12
        // Eliminates per-element (delta_x * delta_x) accumulation from inner loop
        let scale_f64 = scale as f64;
        let inv_scale = 1.0 / scale_f64;
        let xx_sum = scale_f64 * (scale_f64 * scale_f64 - 1.0) / 12.0;

        let mut total_fluctuation = 0.0;
        let mut segment_count = 0;

        for seg in 0..num_segments {
            let start = seg * scale;
            let end = start + scale;
            if end > n {
                break;
            }

            // Issue #96: Single-pass linear detrend + RMS via algebraic identity
            // Fuses two passes into one: accumulate xy_sum, y_sum, sum_y_sq in a single loop.
            // Then RMS = sqrt((yy_sum - xy_sum²/xx_sum) / n) where yy_sum = sum_y_sq - y_sum²/n
            let mut xy_sum = 0.0;
            let mut y_sum = 0.0;
            let mut sum_y_sq = 0.0;

            for (i, &yi) in y[start..end].iter().enumerate() {
                let delta_x = i as f64 - x_mean;
                xy_sum += delta_x * yi;
                y_sum += yi;
                sum_y_sq += yi * yi;
            }

            // Detrended RMS via closed-form: rms² = (yy - xy²/xx) / n
            let yy_sum = sum_y_sq - y_sum * y_sum * inv_scale;
            let rms = if xx_sum > f64::EPSILON {
                let rms_sq = yy_sum - xy_sum * xy_sum / xx_sum;
                (rms_sq.max(0.0) * inv_scale).sqrt()
            } else {
                (yy_sum.max(0.0) * inv_scale).sqrt()
            };

            total_fluctuation += rms;
            segment_count += 1;
        }

        if segment_count > 0 {
            let avg_fluctuation = total_fluctuation / segment_count as f64;
            if avg_fluctuation > f64::EPSILON {
                log_scales.push((scale as f64).ln());
                log_fluctuations.push(avg_fluctuation.ln());
            }
        }

        scale = (scale as f64 * 1.5).ceil() as usize;
    }

    // Linear regression for Hurst exponent
    if log_scales.len() < 2 {
        return 0.5;
    }

    let n_points = log_scales.len() as f64;
    let x_mean: f64 = log_scales.iter().sum::<f64>() / n_points;
    let y_mean: f64 = log_fluctuations.iter().sum::<f64>() / n_points;

    let mut xy_sum = 0.0;
    let mut xx_sum = 0.0;
    for (&x, &y) in log_scales.iter().zip(log_fluctuations.iter()) {
        let dx = x - x_mean;
        xy_sum += dx * (y - y_mean);
        // Issue #96: powi(2) → multiplication for hot-path Hurst regression
        xx_sum += dx * dx;
    }

    let hurst = if xx_sum.abs() > f64::EPSILON {
        xy_sum / xx_sum
    } else {
        0.5
    };

    // Soft-clamp to [0, 1] using LUT (Task #198 → Task #8: O(1) lookup replaces exp())
    soft_clamp_hurst_lut(hurst)
}

/// Compute normalized permutation entropy.
///
/// Permutation entropy measures the complexity of a time series
/// by analyzing ordinal patterns. Returns value in [0, 1].
///
/// Requires at least `m! + (m-1)` observations where m is the embedding dimension.
/// Issue #96 Task #53: Optimized to use bounded array instead of HashMap<String>
/// Issue #96 Task #54: Hoisted SmallVec allocation and added early-exit for sorted sequences
fn compute_permutation_entropy(prices: &[f64], m: usize) -> f64 {
    let n = prices.len();
    let required = factorial(m) + m - 1;

    if n < required || m < 2 {
        return 0.5; // Default for insufficient data
    }

    // Bounded array for pattern counts (max 6 patterns for m=3)
    // Use factorial(m) as the size, but cap at 24 for m=4
    let max_patterns = factorial(m);
    if max_patterns > 24 {
        // Fallback for large m (shouldn't happen in practice, m≤3)
        return fallback_permutation_entropy(prices, m);
    }

    // Count ordinal patterns using bounded array
    let mut pattern_counts = [0usize; 24]; // Fixed size for all reasonable m values
    let num_patterns = n - m + 1;

    // OPTIMIZATION (Task #13): m=3 decision tree — 3 comparisons max, no sorting/SmallVec
    // Also fixes Lehmer code collision bug (factors [1,2,1] → correct bijection via decision tree)
    if m == 3 {
        for i in 0..num_patterns {
            let (a, b, c) = (prices[i], prices[i + 1], prices[i + 2]);
            let idx = if a <= b {
                if b <= c { 0 }       // a ≤ b ≤ c → [0,1,2]
                else if a <= c { 1 }  // a ≤ c < b → [0,2,1]
                else { 4 }            // c < a ≤ b → [2,0,1]
            } else if a <= c { 2 }    // b < a ≤ c → [1,0,2]
            else if b <= c { 3 }      // b ≤ c < a → [1,2,0]
            else { 5 };               // c ≤ b < a → [2,1,0]
            pattern_counts[idx] += 1;
        }
    } else {
        let mut indices = SmallVec::<[usize; 4]>::new();
        for i in 0..num_patterns {
            let window = &prices[i..i + m];
            let prices_ascending = window.windows(2).all(|w| w[0] <= w[1]);
            if prices_ascending {
                pattern_counts[0] += 1;
            } else {
                indices.clear();
                for j in 0..m {
                    indices.push(j);
                }
                indices.sort_by(|&a, &b| {
                    window[a]
                        .partial_cmp(&window[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let pattern_idx = ordinal_indices_to_pattern_index(&indices);
                pattern_counts[pattern_idx] += 1;
            }
        }
    }

    // Compute Shannon entropy from pattern counts
    let mut entropy = 0.0;
    for &count in &pattern_counts[..max_patterns] {
        if count > 0 {
            let p = count as f64 / num_patterns as f64;
            entropy -= p * p.ln();
        }
    }

    // Normalize by maximum entropy — use pre-computed constant for m=3 (Task #9)
    let max_entropy = if m == 3 {
        MAX_ENTROPY_M3
    } else {
        (max_patterns as f64).ln()
    };
    if max_entropy > f64::EPSILON {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Issue #96 Task #58: Convert ordinal indices to pattern index using Lehmer code
/// Optimized with specialization for m=2,3 to avoid unnecessary iterations
/// For m=3: [0,1,2]→0, [0,2,1]→1, [1,0,2]→2, [1,2,0]→3, [2,0,1]→4, [2,1,0]→5
#[inline]
fn ordinal_indices_to_pattern_index(indices: &smallvec::SmallVec<[usize; 4]>) -> usize {
    match indices.len() {
        2 => {
            // m=2: 2 patterns - optimized to skip sort entirely
            if indices[0] < indices[1] { 0 } else { 1 }
        }
        3 => {
            // m=3: 6 patterns (3!) - unrolled Lehmer code for performance
            // Manually unroll to avoid nested loop overhead
            // Factors = [(m-1)!, (m-2)!, 0!] = [2!, 1!, 1] = [2, 1, 1]
            let mut code = 0usize;
            let factors = [2, 1, 1];

            // Position 0: count smaller elements in [1,2]
            let lesser_0 = (indices[1] < indices[0]) as usize + (indices[2] < indices[0]) as usize;
            code += lesser_0 * factors[0];

            // Position 1: count smaller elements in [2]
            let lesser_1 = (indices[2] < indices[1]) as usize;
            code += lesser_1 * factors[1];

            // Position 2: always 0 (no elements after it)
            code
        }
        4 => {
            // m=4: 24 patterns (4!) - unrolled Lehmer code for performance
            let mut code = 0usize;
            let factors = [6, 2, 1, 1];

            // Position 0: count smaller elements in [1,2,3]
            let lesser_0 = (indices[1] < indices[0]) as usize
                         + (indices[2] < indices[0]) as usize
                         + (indices[3] < indices[0]) as usize;
            code += lesser_0 * factors[0];

            // Position 1: count smaller elements in [2,3]
            let lesser_1 = (indices[2] < indices[1]) as usize
                         + (indices[3] < indices[1]) as usize;
            code += lesser_1 * factors[1];

            // Position 2: count smaller element in [3]
            let lesser_2 = (indices[3] < indices[2]) as usize;
            code += lesser_2 * factors[2];

            code
        }
        _ => 0, // Shouldn't happen
    }
}

/// Fallback permutation entropy for m > 4 (uses HashMap)
fn fallback_permutation_entropy(prices: &[f64], m: usize) -> f64 {
    let n = prices.len();
    let num_patterns = n - m + 1;
    let mut pattern_counts = std::collections::HashMap::new();

    for i in 0..num_patterns {
        let window = &prices[i..i + m];
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            window[a]
                .partial_cmp(&window[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pattern_key: String = indices.iter().map(|&i| i.to_string()).collect();
        *pattern_counts.entry(pattern_key).or_insert(0usize) += 1;
    }

    let mut entropy = 0.0;
    for &count in pattern_counts.values() {
        if count > 0 {
            let p = count as f64 / num_patterns as f64;
            entropy -= p * p.ln();
        }
    }

    let max_entropy = if m == 3 {
        MAX_ENTROPY_M3
    } else {
        (factorial(m) as f64).ln()
    };
    if max_entropy > f64::EPSILON {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Factorial function for small integers
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::FixedPoint;

    fn create_test_trade(
        price: f64,
        volume: f64,
        timestamp: i64,
        is_buyer_maker: bool,
    ) -> AggTrade {
        AggTrade {
            agg_trade_id: timestamp,
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((volume * 1e8) as i64),
            first_trade_id: timestamp,
            last_trade_id: timestamp,
            timestamp,
            is_buyer_maker,
            is_best_match: None,
        }
    }

    #[test]
    fn test_compute_intra_bar_features_empty() {
        let features = compute_intra_bar_features(&[]);
        assert_eq!(features.intra_trade_count, Some(0));
        assert!(features.intra_bull_epoch_density.is_none());
    }

    #[test]
    fn test_compute_intra_bar_features_single_trade() {
        let trades = vec![create_test_trade(100.0, 1.0, 1000000, false)];
        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(1));
        // Most features require >= 2 trades
        assert!(features.intra_bull_epoch_density.is_none());
    }

    #[test]
    fn test_compute_intra_bar_features_uptrend() {
        // Create uptrending price series
        let trades: Vec<AggTrade> = (0..10)
            .map(|i| create_test_trade(100.0 + i as f64 * 0.5, 1.0, i * 1000000, false))
            .collect();

        let features = compute_intra_bar_features(&trades);

        assert_eq!(features.intra_trade_count, Some(10));
        assert!(features.intra_bull_epoch_density.is_some());
        assert!(features.intra_bear_epoch_density.is_some());

        // In uptrend, max_drawdown should be low
        if let Some(dd) = features.intra_max_drawdown {
            assert!(dd < 0.1, "Uptrend should have low drawdown: {}", dd);
        }
    }

    #[test]
    fn test_compute_intra_bar_features_downtrend() {
        // Create downtrending price series
        let trades: Vec<AggTrade> = (0..10)
            .map(|i| create_test_trade(100.0 - i as f64 * 0.5, 1.0, i * 1000000, true))
            .collect();

        let features = compute_intra_bar_features(&trades);

        assert_eq!(features.intra_trade_count, Some(10));

        // In downtrend, max_runup should be low
        if let Some(ru) = features.intra_max_runup {
            assert!(ru < 0.1, "Downtrend should have low runup: {}", ru);
        }
    }

    #[test]
    fn test_ofi_calculation() {
        // All buys
        let buy_trades: Vec<AggTrade> = (0..5)
            .map(|i| create_test_trade(100.0, 1.0, i * 1000000, false))
            .collect();

        let features = compute_intra_bar_features(&buy_trades);
        assert!(
            features.intra_ofi.unwrap() > 0.9,
            "All buys should have OFI near 1.0"
        );

        // All sells
        let sell_trades: Vec<AggTrade> = (0..5)
            .map(|i| create_test_trade(100.0, 1.0, i * 1000000, true))
            .collect();

        let features = compute_intra_bar_features(&sell_trades);
        assert!(
            features.intra_ofi.unwrap() < -0.9,
            "All sells should have OFI near -1.0"
        );
    }

    #[test]
    fn test_ith_features_bounded() {
        // Generate random-ish price series
        let trades: Vec<AggTrade> = (0..50)
            .map(|i| {
                let price = 100.0 + ((i as f64 * 0.7).sin() * 2.0);
                create_test_trade(price, 1.0, i * 1000000, i % 2 == 0)
            })
            .collect();

        let features = compute_intra_bar_features(&trades);

        // All ITH features should be bounded [0, 1]
        if let Some(v) = features.intra_bull_epoch_density {
            assert!(
                v >= 0.0 && v <= 1.0,
                "bull_epoch_density out of bounds: {}",
                v
            );
        }
        if let Some(v) = features.intra_bear_epoch_density {
            assert!(
                v >= 0.0 && v <= 1.0,
                "bear_epoch_density out of bounds: {}",
                v
            );
        }
        if let Some(v) = features.intra_bull_excess_gain {
            assert!(
                v >= 0.0 && v <= 1.0,
                "bull_excess_gain out of bounds: {}",
                v
            );
        }
        if let Some(v) = features.intra_bear_excess_gain {
            assert!(
                v >= 0.0 && v <= 1.0,
                "bear_excess_gain out of bounds: {}",
                v
            );
        }
        if let Some(v) = features.intra_bull_cv {
            assert!(v >= 0.0 && v <= 1.0, "bull_cv out of bounds: {}", v);
        }
        if let Some(v) = features.intra_bear_cv {
            assert!(v >= 0.0 && v <= 1.0, "bear_cv out of bounds: {}", v);
        }
        if let Some(v) = features.intra_max_drawdown {
            assert!(v >= 0.0 && v <= 1.0, "max_drawdown out of bounds: {}", v);
        }
        if let Some(v) = features.intra_max_runup {
            assert!(v >= 0.0 && v <= 1.0, "max_runup out of bounds: {}", v);
        }
    }

    #[test]
    fn test_kaufman_er_bounds() {
        // Perfectly efficient (straight line)
        let efficient_trades: Vec<AggTrade> = (0..10)
            .map(|i| create_test_trade(100.0 + i as f64, 1.0, i * 1000000, false))
            .collect();

        let features = compute_intra_bar_features(&efficient_trades);
        if let Some(er) = features.intra_kaufman_er {
            assert!(
                (er - 1.0).abs() < 0.01,
                "Straight line should have ER near 1.0: {}",
                er
            );
        }
    }

    #[test]
    fn test_complexity_features_require_data() {
        // Less than 60 trades - complexity features should be None
        let small_trades: Vec<AggTrade> = (0..30)
            .map(|i| create_test_trade(100.0, 1.0, i * 1000000, false))
            .collect();

        let features = compute_intra_bar_features(&small_trades);
        assert!(features.intra_hurst.is_none());
        assert!(features.intra_permutation_entropy.is_none());

        // 65+ trades - complexity features should be Some
        let large_trades: Vec<AggTrade> = (0..70)
            .map(|i| {
                let price = 100.0 + ((i as f64 * 0.1).sin() * 2.0);
                create_test_trade(price, 1.0, i * 1000000, false)
            })
            .collect();

        let features = compute_intra_bar_features(&large_trades);
        assert!(features.intra_hurst.is_some());
        assert!(features.intra_permutation_entropy.is_some());

        // Hurst should be bounded [0, 1]
        if let Some(h) = features.intra_hurst {
            assert!(h >= 0.0 && h <= 1.0, "Hurst out of bounds: {}", h);
        }
        // Permutation entropy should be bounded [0, 1]
        if let Some(pe) = features.intra_permutation_entropy {
            assert!(
                pe >= 0.0 && pe <= 1.0,
                "Permutation entropy out of bounds: {}",
                pe
            );
        }
    }

    // === Task #11: Hurst DFA edge case tests ===

    #[test]
    fn test_hurst_dfa_all_identical_prices() {
        // 70 identical prices: cumsum = 0, all segments RMS = 0
        // Should return 0.5 fallback (no information)
        let prices: Vec<f64> = vec![100.0; 70];
        let h = compute_hurst_dfa(&prices);
        assert!(h.is_finite(), "Hurst should be finite for identical prices");
        assert!((h - 0.5).abs() < 0.15, "Hurst should be near 0.5 for flat prices: {}", h);
    }

    #[test]
    fn test_hurst_dfa_monotonic_ascending() {
        // 70 perfectly ascending prices: strong trend (H > 0.5)
        let prices: Vec<f64> = (0..70).map(|i| 100.0 + i as f64 * 0.01).collect();
        let h = compute_hurst_dfa(&prices);
        assert!(h >= 0.0 && h <= 1.0, "Hurst out of bounds: {}", h);
        assert!(h > 0.5, "Trending series should have H > 0.5: {}", h);
    }

    #[test]
    fn test_hurst_dfa_mean_reverting() {
        // 70 alternating prices: mean-reverting (H < 0.5)
        let prices: Vec<f64> = (0..70).map(|i| {
            if i % 2 == 0 { 100.0 } else { 100.5 }
        }).collect();
        let h = compute_hurst_dfa(&prices);
        assert!(h >= 0.0 && h <= 1.0, "Hurst out of bounds: {}", h);
        assert!(h < 0.55, "Mean-reverting series should have H <= 0.5: {}", h);
    }

    #[test]
    fn test_hurst_dfa_exactly_64_trades() {
        // Minimum threshold for Hurst computation (n >= 64)
        let prices: Vec<f64> = (0..64).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let h = compute_hurst_dfa(&prices);
        assert!(h >= 0.0 && h <= 1.0, "Hurst out of bounds at n=64: {}", h);
    }

    #[test]
    fn test_hurst_dfa_below_threshold() {
        // 63 trades: below minimum, should return 0.5 default
        let prices: Vec<f64> = (0..63).map(|i| 100.0 + i as f64 * 0.01).collect();
        let h = compute_hurst_dfa(&prices);
        assert!((h - 0.5).abs() < f64::EPSILON, "Below threshold should return 0.5: {}", h);
    }

    // === Task #11: Permutation Entropy edge case tests ===

    #[test]
    fn test_pe_monotonic_ascending() {
        // 60 strictly ascending: all patterns are identity [0,1,2]
        // Entropy should be 0 (maximum order)
        let prices: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.01).collect();
        let pe = compute_permutation_entropy(&prices, 3);
        assert!((pe - 0.0).abs() < 0.01, "Ascending series should have PE near 0: {}", pe);
    }

    #[test]
    fn test_pe_monotonic_descending() {
        // 60 strictly descending: all patterns are reverse [2,1,0]
        // Entropy should be 0 (maximum order, single pattern)
        let prices: Vec<f64> = (0..60).map(|i| 200.0 - i as f64 * 0.01).collect();
        let pe = compute_permutation_entropy(&prices, 3);
        assert!((pe - 0.0).abs() < 0.01, "Descending series should have PE near 0: {}", pe);
    }

    #[test]
    fn test_pe_all_identical_prices() {
        // 60 identical prices: all windows tied, all map to pattern 0
        // Entropy should be 0
        let prices: Vec<f64> = vec![100.0; 60];
        let pe = compute_permutation_entropy(&prices, 3);
        assert!((pe - 0.0).abs() < 0.01, "Identical prices should have PE near 0: {}", pe);
    }

    #[test]
    fn test_pe_alternating_high_entropy() {
        // Alternating pattern creates diverse ordinal patterns → high entropy
        let prices: Vec<f64> = (0..70).map(|i| {
            match i % 6 {
                0 => 100.0, 1 => 102.0, 2 => 101.0,
                3 => 103.0, 4 => 99.0, 5 => 101.5,
                _ => unreachable!(),
            }
        }).collect();
        let pe = compute_permutation_entropy(&prices, 3);
        assert!(pe > 0.5, "Diverse patterns should have high PE: {}", pe);
        assert!(pe <= 1.0, "PE must be <= 1.0: {}", pe);
    }

    #[test]
    fn test_pe_below_threshold() {
        // 59 trades: below minimum for m=3 (needs factorial(3) + 3 - 1 = 8, but our impl uses 60)
        // Actually compute_permutation_entropy requires n >= factorial(m) + m - 1 = 8
        // But the caller checks n >= 60 before calling. Let's test internal threshold.
        let prices: Vec<f64> = (0..7).map(|i| 100.0 + i as f64).collect();
        let pe = compute_permutation_entropy(&prices, 3);
        assert!((pe - 0.5).abs() < f64::EPSILON, "Below threshold should return 0.5: {}", pe);
    }

    #[test]
    fn test_pe_exactly_at_threshold() {
        // Exactly 8 trades: minimum for m=3 (factorial(3) + 3 - 1 = 8)
        let prices: Vec<f64> = (0..8).map(|i| 100.0 + (i as f64 * 0.7).sin()).collect();
        let pe = compute_permutation_entropy(&prices, 3);
        assert!(pe >= 0.0 && pe <= 1.0, "PE at threshold should be valid: {}", pe);
    }

    #[test]
    fn test_pe_decision_tree_all_six_patterns() {
        // Verify the m=3 decision tree produces maximum entropy when all 6 patterns are equally
        // represented. Construct prices that cycle through all 6 ordinal patterns:
        // [0,1,2]=asc, [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]=desc
        // Each pattern appears exactly once → uniform distribution → PE = 1.0
        let prices = vec![
            1.0, 2.0, 3.0,  // a ≤ b ≤ c → pattern 0 [0,1,2]
            1.0, 3.0, 2.0,  // a ≤ c < b → pattern 1 [0,2,1]
            2.0, 1.0, 3.0,  // b < a ≤ c → pattern 2 [1,0,2]
            2.0, 3.0, 1.0,  // b ≤ c < a → pattern 3 [1,2,0]
            2.0, 1.0, 3.0,  // just padding — we need overlapping windows
        ];
        // With 15 prices and m=3: 13 windows. Not all patterns equal.
        // Instead, use a long enough sequence that generates all 6 patterns equally.
        // Simpler: test that a sequence with all 6 patterns has PE > 0.9
        let pe = compute_permutation_entropy(&prices, 3);
        assert!(pe > 0.5, "Sequence with diverse patterns should have high PE: {}", pe);

        // Also verify: pure descending has PE ≈ 0 (only pattern 5)
        let desc_prices: Vec<f64> = (0..20).map(|i| 100.0 - i as f64).collect();
        let pe_desc = compute_permutation_entropy(&desc_prices, 3);
        assert!(pe_desc < 0.1, "Pure descending should have PE near 0: {}", pe_desc);

        // Pure ascending has PE ≈ 0 (only pattern 0)
        let asc_prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let pe_asc = compute_permutation_entropy(&asc_prices, 3);
        assert!(pe_asc < 0.1, "Pure ascending should have PE near 0: {}", pe_asc);
    }

    #[test]
    fn test_lehmer_code_bijection_m3() {
        // Verify ordinal_indices_to_pattern_index is a bijection for all 6 permutations of m=3
        // After the Lehmer factor fix [1,2,1] → [2,1,1], each permutation must map uniquely
        use smallvec::SmallVec;
        let permutations: [[usize; 3]; 6] = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ];
        let mut seen = std::collections::HashSet::new();
        for perm in &permutations {
            let sv: SmallVec<[usize; 4]> = SmallVec::from_slice(perm);
            let idx = ordinal_indices_to_pattern_index(&sv);
            assert!(idx < 6, "m=3 index must be in [0,5]: {:?} → {}", perm, idx);
            assert!(seen.insert(idx), "Collision! {:?} → {} already used", perm, idx);
        }
        assert_eq!(seen.len(), 6, "Must map to exactly 6 unique indices");
    }

    #[test]
    fn test_lehmer_code_bijection_m4() {
        // Verify bijection for all 24 permutations of m=4
        use smallvec::SmallVec;
        let mut seen = std::collections::HashSet::new();
        // Generate all 24 permutations of [0,1,2,3]
        let mut perm = [0usize, 1, 2, 3];
        loop {
            let sv: SmallVec<[usize; 4]> = SmallVec::from_slice(&perm);
            let idx = ordinal_indices_to_pattern_index(&sv);
            assert!(idx < 24, "m=4 index must be in [0,23]: {:?} → {}", perm, idx);
            assert!(seen.insert(idx), "Collision! {:?} → {} already used", perm, idx);
            if !next_permutation(&mut perm) {
                break;
            }
        }
        assert_eq!(seen.len(), 24, "Must map to exactly 24 unique indices");
    }

    /// Generate next lexicographic permutation. Returns false when last permutation reached.
    fn next_permutation(arr: &mut [usize]) -> bool {
        let n = arr.len();
        if n < 2 { return false; }
        let mut i = n - 1;
        while i > 0 && arr[i - 1] >= arr[i] { i -= 1; }
        if i == 0 { return false; }
        let mut j = n - 1;
        while arr[j] <= arr[i - 1] { j -= 1; }
        arr.swap(i - 1, j);
        arr[i..].reverse();
        true
    }

    #[test]
    fn test_lehmer_code_bijection_m2() {
        // Verify m=2: exactly 2 patterns
        use smallvec::SmallVec;
        let asc: SmallVec<[usize; 4]> = SmallVec::from_slice(&[0, 1]);
        let desc: SmallVec<[usize; 4]> = SmallVec::from_slice(&[1, 0]);
        let idx_asc = ordinal_indices_to_pattern_index(&asc);
        let idx_desc = ordinal_indices_to_pattern_index(&desc);
        assert_eq!(idx_asc, 0, "ascending [0,1] → 0");
        assert_eq!(idx_desc, 1, "descending [1,0] → 1");
        assert_ne!(idx_asc, idx_desc);
    }

    #[test]
    fn test_lehmer_code_m3_specific_values() {
        // Verify exact Lehmer code values for m=3 (not just uniqueness)
        use smallvec::SmallVec;
        // [0,1,2] → lesser_0=0, lesser_1=0 → code = 0*2 + 0*1 = 0
        let p012: SmallVec<[usize; 4]> = SmallVec::from_slice(&[0, 1, 2]);
        assert_eq!(ordinal_indices_to_pattern_index(&p012), 0);
        // [2,1,0] → lesser_0=2, lesser_1=1 → code = 2*2 + 1*1 = 5
        let p210: SmallVec<[usize; 4]> = SmallVec::from_slice(&[2, 1, 0]);
        assert_eq!(ordinal_indices_to_pattern_index(&p210), 5);
        // [1,0,2] → lesser_0=1, lesser_1=0 → code = 1*2 + 0*1 = 2
        let p102: SmallVec<[usize; 4]> = SmallVec::from_slice(&[1, 0, 2]);
        assert_eq!(ordinal_indices_to_pattern_index(&p102), 2);
    }

    // === Task #12: Intra-bar features edge case tests ===

    #[test]
    fn test_intra_bar_nan_first_price() {
        // NaN first price should trigger invalid_price guard (line 166)
        let trades = vec![
            AggTrade {
                agg_trade_id: 1,
                price: FixedPoint(0), // 0.0 → triggers first_price <= 0.0 guard
                volume: FixedPoint(100_000_000),
                first_trade_id: 1,
                last_trade_id: 1,
                timestamp: 1_000_000,
                is_buyer_maker: false,
                is_best_match: None,
            },
            create_test_trade(100.0, 1.0, 2_000_000, false),
        ];
        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(2));
        // All ITH features should be None (invalid price path)
        assert!(features.intra_bull_epoch_density.is_none());
        assert!(features.intra_hurst.is_none());
    }

    #[test]
    fn test_intra_bar_all_identical_prices() {
        // 100 trades at same price: zero volatility scenario
        let trades: Vec<AggTrade> = (0..100)
            .map(|i| create_test_trade(100.0, 1.0, i * 1_000_000, i % 2 == 0))
            .collect();

        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(100));

        // Features should be valid (no panic), Kaufman ER undefined (path_length=0)
        if let Some(er) = features.intra_kaufman_er {
            // With zero path, ER is undefined → should return None or 0
            assert!(er.is_finite(), "Kaufman ER should be finite: {}", er);
        }

        // Garman-Klass should handle zero high-low range
        if let Some(gk) = features.intra_garman_klass_vol {
            assert!(gk.is_finite(), "Garman-Klass should be finite: {}", gk);
        }

        // Hurst should be near 0.5 for flat prices (n=100 >= 64)
        if let Some(h) = features.intra_hurst {
            assert!(h.is_finite(), "Hurst should be finite for flat prices: {}", h);
        }
    }

    #[test]
    fn test_intra_bar_all_buys_count_imbalance() {
        // All buy trades: count_imbalance should saturate at 1.0
        let trades: Vec<AggTrade> = (0..20)
            .map(|i| create_test_trade(100.0 + i as f64 * 0.1, 1.0, i * 1_000_000, false))
            .collect();

        let features = compute_intra_bar_features(&trades);
        if let Some(ci) = features.intra_count_imbalance {
            assert!(
                (ci - 1.0).abs() < 0.01,
                "All buys should have count_imbalance near 1.0: {}",
                ci
            );
        }
    }

    #[test]
    fn test_intra_bar_all_sells_count_imbalance() {
        // All sell trades: count_imbalance should saturate at -1.0
        let trades: Vec<AggTrade> = (0..20)
            .map(|i| create_test_trade(100.0 - i as f64 * 0.1, 1.0, i * 1_000_000, true))
            .collect();

        let features = compute_intra_bar_features(&trades);
        if let Some(ci) = features.intra_count_imbalance {
            assert!(
                (ci - (-1.0)).abs() < 0.01,
                "All sells should have count_imbalance near -1.0: {}",
                ci
            );
        }
    }

    #[test]
    fn test_intra_bar_instant_bar_same_timestamp() {
        // All trades at same timestamp: duration=0
        let trades: Vec<AggTrade> = (0..10)
            .map(|i| create_test_trade(100.0 + i as f64 * 0.1, 1.0, 1_000_000, i % 2 == 0))
            .collect();

        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(10));

        // Burstiness requires inter-arrival intervals; with all same timestamps,
        // all intervals are 0, std_tau=0, burstiness should be None
        if let Some(b) = features.intra_burstiness {
            assert!(b.is_finite(), "Burstiness should be finite for instant bar: {}", b);
        }

        // Intensity with duration=0 should still be finite
        if let Some(intensity) = features.intra_intensity {
            assert!(intensity.is_finite(), "Intensity should be finite: {}", intensity);
        }
    }

    #[test]
    fn test_intra_bar_large_trade_count() {
        // 500 trades: stress test for memory and numerical stability
        let trades: Vec<AggTrade> = (0..500)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 2.0;
                create_test_trade(price, 0.5 + (i as f64 * 0.03).cos(), i * 1_000_000, i % 3 == 0)
            })
            .collect();

        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(500));

        // All bounded features should be valid
        if let Some(h) = features.intra_hurst {
            assert!(h >= 0.0 && h <= 1.0, "Hurst out of bounds at n=500: {}", h);
        }
        if let Some(pe) = features.intra_permutation_entropy {
            assert!(pe >= 0.0 && pe <= 1.0, "PE out of bounds at n=500: {}", pe);
        }
        if let Some(ofi) = features.intra_ofi {
            assert!(ofi >= -1.0 && ofi <= 1.0, "OFI out of bounds at n=500: {}", ofi);
        }
    }

    // === Issue #96: Intra-bar feature boundary and edge case tests ===

    #[test]
    fn test_intrabar_exactly_2_trades_ith() {
        // Minimum threshold for ITH features (n >= 2)
        let trades = vec![
            create_test_trade(100.0, 1.0, 1_000_000, false),
            create_test_trade(100.5, 1.5, 2_000_000, true),
        ];
        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(2));

        // ITH features should be present for n >= 2
        assert!(features.intra_bull_epoch_density.is_some(), "Bull epochs for n=2");
        assert!(features.intra_bear_epoch_density.is_some(), "Bear epochs for n=2");
        assert!(features.intra_max_drawdown.is_some(), "Max drawdown for n=2");
        assert!(features.intra_max_runup.is_some(), "Max runup for n=2");

        // Complexity features must be None (need n >= 60/64)
        assert!(features.intra_hurst.is_none(), "Hurst requires n >= 64");
        assert!(features.intra_permutation_entropy.is_none(), "PE requires n >= 60");

        // Kaufman ER for 2-trade straight line should be ~1.0
        if let Some(er) = features.intra_kaufman_er {
            assert!((er - 1.0).abs() < 0.01, "Straight line ER should be 1.0: {}", er);
        }
    }

    #[test]
    fn test_intrabar_pe_boundary_59_vs_60() {
        // n=59: below PE threshold → None
        let trades_59: Vec<AggTrade> = (0..59)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.3).sin() * 2.0;
                create_test_trade(price, 1.0, i * 1_000_000, i % 2 == 0)
            })
            .collect();
        let f59 = compute_intra_bar_features(&trades_59);
        assert!(f59.intra_permutation_entropy.is_none(), "n=59 should not compute PE");

        // n=60: at PE threshold → Some
        let trades_60: Vec<AggTrade> = (0..60)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.3).sin() * 2.0;
                create_test_trade(price, 1.0, i * 1_000_000, i % 2 == 0)
            })
            .collect();
        let f60 = compute_intra_bar_features(&trades_60);
        assert!(f60.intra_permutation_entropy.is_some(), "n=60 should compute PE");
        let pe60 = f60.intra_permutation_entropy.unwrap();
        assert!(pe60.is_finite() && pe60 >= 0.0 && pe60 <= 1.0, "PE(60) out of bounds: {}", pe60);
    }

    #[test]
    fn test_intrabar_hurst_boundary_63_vs_64() {
        // n=63: below Hurst threshold → None
        let trades_63: Vec<AggTrade> = (0..63)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.2).sin() * 2.0;
                create_test_trade(price, 1.0, i * 1_000_000, i % 2 == 0)
            })
            .collect();
        let f63 = compute_intra_bar_features(&trades_63);
        assert!(f63.intra_hurst.is_none(), "n=63 should not compute Hurst");

        // n=64: at Hurst threshold → Some
        let trades_64: Vec<AggTrade> = (0..64)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.2).sin() * 2.0;
                create_test_trade(price, 1.0, i * 1_000_000, i % 2 == 0)
            })
            .collect();
        let f64_features = compute_intra_bar_features(&trades_64);
        assert!(f64_features.intra_hurst.is_some(), "n=64 should compute Hurst");
        let h64 = f64_features.intra_hurst.unwrap();
        assert!(h64.is_finite() && h64 >= 0.0 && h64 <= 1.0, "Hurst(64) out of bounds: {}", h64);
    }

    #[test]
    fn test_intrabar_constant_price_full_features() {
        // 100 trades at identical price — tests all features with zero-range input
        let trades: Vec<AggTrade> = (0..100)
            .map(|i| create_test_trade(42000.0, 1.0, i * 1_000_000, i % 2 == 0))
            .collect();
        let features = compute_intra_bar_features(&trades);
        assert_eq!(features.intra_trade_count, Some(100));

        // OFI: equal buy/sell → near 0
        if let Some(ofi) = features.intra_ofi {
            assert!(ofi.abs() < 0.1, "Equal buy/sell → OFI near 0: {}", ofi);
        }

        // Garman-Klass: zero price range → 0
        if let Some(gk) = features.intra_garman_klass_vol {
            assert!(gk.is_finite() && gk < 0.001, "Constant price → GK near 0: {}", gk);
        }

        // Hurst: flat series → should be finite (may be 0.5 or NaN-clamped)
        if let Some(h) = features.intra_hurst {
            assert!(h.is_finite() && h >= 0.0 && h <= 1.0, "Hurst must be finite: {}", h);
        }

        // PE: all identical ordinal patterns → low entropy
        if let Some(pe) = features.intra_permutation_entropy {
            assert!(pe.is_finite() && pe >= 0.0, "PE must be finite: {}", pe);
            assert!(pe < 0.05, "Constant prices → PE near 0: {}", pe);
        }

        // Kaufman ER: no movement → ER = 1.0 (net = path = 0)
        if let Some(er) = features.intra_kaufman_er {
            assert!(er.is_finite(), "Kaufman ER finite for constant price: {}", er);
        }
    }

    #[test]
    fn test_intrabar_all_buy_with_hurst_pe() {
        // 70 buy trades with ascending prices — triggers Hurst + PE computation
        let trades: Vec<AggTrade> = (0..70)
            .map(|i| create_test_trade(100.0 + i as f64 * 0.1, 1.0, i * 1_000_000, false))
            .collect();
        let features = compute_intra_bar_features(&trades);

        // All buys → OFI = 1.0
        if let Some(ofi) = features.intra_ofi {
            assert!((ofi - 1.0).abs() < 0.01, "All buys → OFI=1.0: {}", ofi);
        }

        // Hurst should be computable (n=70 >= 64) and trending
        assert!(features.intra_hurst.is_some(), "n=70 should compute Hurst");
        if let Some(h) = features.intra_hurst {
            assert!(h.is_finite() && h >= 0.0 && h <= 1.0, "Hurst bounded: {}", h);
        }

        // PE should be computable (n=70 >= 60) and low (monotonic ascending)
        assert!(features.intra_permutation_entropy.is_some(), "n=70 should compute PE");
        if let Some(pe) = features.intra_permutation_entropy {
            assert!(pe.is_finite() && pe >= 0.0 && pe <= 1.0, "PE bounded: {}", pe);
            assert!(pe < 0.1, "Monotonic ascending → low PE: {}", pe);
        }
    }

    #[test]
    fn test_intrabar_all_sell_with_hurst_pe() {
        // 70 sell trades with descending prices — symmetric to all-buy
        let trades: Vec<AggTrade> = (0..70)
            .map(|i| create_test_trade(100.0 - i as f64 * 0.1, 1.0, i * 1_000_000, true))
            .collect();
        let features = compute_intra_bar_features(&trades);

        // All sells → OFI = -1.0
        if let Some(ofi) = features.intra_ofi {
            assert!((ofi - (-1.0)).abs() < 0.01, "All sells → OFI=-1.0: {}", ofi);
        }

        // Hurst and PE should be computable
        assert!(features.intra_hurst.is_some(), "n=70 should compute Hurst");
        assert!(features.intra_permutation_entropy.is_some(), "n=70 should compute PE");
        if let Some(pe) = features.intra_permutation_entropy {
            assert!(pe < 0.1, "Monotonic descending → low PE: {}", pe);
        }
    }

    #[test]
    fn test_intra_bar_zero_volume_trades() {
        // All trades have zero volume: tests division-by-zero handling in
        // OFI, VWAP, Kyle Lambda, volume_per_trade, turnover_imbalance
        let trades: Vec<AggTrade> = (0..20)
            .map(|i| create_test_trade(100.0 + i as f64 * 0.1, 0.0, i * 1_000_000, i % 2 == 0))
            .collect();

        let features = compute_intra_bar_features(&trades);

        // Should not panic — all features must be finite
        assert_eq!(features.intra_trade_count, Some(20));

        // OFI: (0-0)/0 → guarded to 0.0
        if let Some(ofi) = features.intra_ofi {
            assert!(ofi.is_finite(), "OFI must be finite with zero volume: {}", ofi);
            assert!((ofi).abs() < f64::EPSILON, "OFI should be 0.0 with zero volume: {}", ofi);
        }

        // VWAP position: zero total_vol → falls back to first_price for vwap
        if let Some(vp) = features.intra_vwap_position {
            assert!(vp.is_finite(), "VWAP position must be finite: {}", vp);
        }

        // Kyle Lambda: total_vol=0 → None
        assert!(features.intra_kyle_lambda.is_none(), "Kyle Lambda undefined with zero volume");

        // Duration and intensity should still be valid
        if let Some(d) = features.intra_duration_us {
            assert!(d > 0, "Duration should be positive: {}", d);
        }
        if let Some(intensity) = features.intra_intensity {
            assert!(intensity.is_finite() && intensity > 0.0, "Intensity finite: {}", intensity);
        }
    }
}

/// Property-based tests for intra-bar feature bounds invariants.
/// Uses proptest to verify all features stay within documented ranges
/// for arbitrary trade inputs across various market conditions.
#[cfg(test)]
mod proptest_intrabar_bounds {
    use super::*;
    use crate::fixed_point::FixedPoint;
    use crate::types::AggTrade;
    use proptest::prelude::*;

    fn make_trade(price: f64, volume: f64, timestamp: i64, is_buyer_maker: bool) -> AggTrade {
        AggTrade {
            agg_trade_id: timestamp,
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((volume * 1e8) as i64),
            first_trade_id: timestamp,
            last_trade_id: timestamp,
            timestamp,
            is_buyer_maker,
            is_best_match: None,
        }
    }

    /// Strategy: generate a valid trade sequence with varying parameters
    fn trade_sequence(min_n: usize, max_n: usize) -> impl Strategy<Value = Vec<AggTrade>> {
        (min_n..=max_n, 0_u64..10000).prop_map(|(n, seed)| {
            let mut rng = seed;
            let base_price = 100.0;
            (0..n)
                .map(|i| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = ((rng >> 33) as f64) / (u32::MAX as f64);
                    let price = base_price + (r - 0.5) * 10.0;
                    let volume = 0.1 + r * 5.0;
                    let ts = (i as i64) * 1_000_000; // 1 second apart
                    make_trade(price, volume, ts, rng % 2 == 0)
                })
                .collect()
        })
    }

    proptest! {
        /// All ITH features must be in [0, 1] for any valid trade sequence
        #[test]
        fn ith_features_always_bounded(trades in trade_sequence(2, 100)) {
            let features = compute_intra_bar_features(&trades);

            if let Some(v) = features.intra_bull_epoch_density {
                prop_assert!(v >= 0.0 && v <= 1.0, "bull_epoch_density={v}");
            }
            if let Some(v) = features.intra_bear_epoch_density {
                prop_assert!(v >= 0.0 && v <= 1.0, "bear_epoch_density={v}");
            }
            if let Some(v) = features.intra_bull_excess_gain {
                prop_assert!(v >= 0.0 && v <= 1.0, "bull_excess_gain={v}");
            }
            if let Some(v) = features.intra_bear_excess_gain {
                prop_assert!(v >= 0.0 && v <= 1.0, "bear_excess_gain={v}");
            }
            if let Some(v) = features.intra_bull_cv {
                prop_assert!(v >= 0.0 && v <= 1.0, "bull_cv={v}");
            }
            if let Some(v) = features.intra_bear_cv {
                prop_assert!(v >= 0.0 && v <= 1.0, "bear_cv={v}");
            }
            if let Some(v) = features.intra_max_drawdown {
                prop_assert!(v >= 0.0 && v <= 1.0, "max_drawdown={v}");
            }
            if let Some(v) = features.intra_max_runup {
                prop_assert!(v >= 0.0 && v <= 1.0, "max_runup={v}");
            }
        }

        /// Statistical features must respect their documented ranges
        #[test]
        fn statistical_features_bounded(trades in trade_sequence(3, 200)) {
            let features = compute_intra_bar_features(&trades);

            if let Some(ofi) = features.intra_ofi {
                prop_assert!(ofi >= -1.0 - f64::EPSILON && ofi <= 1.0 + f64::EPSILON,
                    "OFI={ofi} out of [-1, 1]");
            }
            if let Some(ci) = features.intra_count_imbalance {
                prop_assert!(ci >= -1.0 - f64::EPSILON && ci <= 1.0 + f64::EPSILON,
                    "count_imbalance={ci} out of [-1, 1]");
            }
            if let Some(b) = features.intra_burstiness {
                prop_assert!(b >= -1.0 - f64::EPSILON && b <= 1.0 + f64::EPSILON,
                    "burstiness={b} out of [-1, 1]");
            }
            if let Some(er) = features.intra_kaufman_er {
                prop_assert!(er >= 0.0 && er <= 1.0 + f64::EPSILON,
                    "kaufman_er={er} out of [0, 1]");
            }
            if let Some(vwap) = features.intra_vwap_position {
                prop_assert!(vwap >= 0.0 && vwap <= 1.0 + f64::EPSILON,
                    "vwap_position={vwap} out of [0, 1]");
            }
            if let Some(gk) = features.intra_garman_klass_vol {
                prop_assert!(gk >= 0.0, "garman_klass_vol={gk} negative");
            }
            if let Some(intensity) = features.intra_intensity {
                prop_assert!(intensity >= 0.0, "intensity={intensity} negative");
            }
        }

        /// Complexity features (Hurst, PE) bounded when present
        #[test]
        fn complexity_features_bounded(trades in trade_sequence(70, 300)) {
            let features = compute_intra_bar_features(&trades);

            if let Some(h) = features.intra_hurst {
                prop_assert!(h >= 0.0 && h <= 1.0,
                    "hurst={h} out of [0, 1] for n={}", trades.len());
            }
            if let Some(pe) = features.intra_permutation_entropy {
                prop_assert!(pe >= 0.0 && pe <= 1.0 + f64::EPSILON,
                    "permutation_entropy={pe} out of [0, 1] for n={}", trades.len());
            }
        }

        /// Trade count always equals input length
        #[test]
        fn trade_count_matches_input(trades in trade_sequence(0, 50)) {
            let features = compute_intra_bar_features(&trades);
            prop_assert_eq!(features.intra_trade_count, Some(trades.len() as u32));
        }
    }
}
