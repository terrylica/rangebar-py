//! Intra-bar feature computation from constituent trades.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! This module computes 22 features from trades WITHIN each range bar:
//! - 8 ITH features (from trading-fitness algorithms)
//! - 12 statistical features
//! - 2 complexity features (Hurst, Permutation Entropy)

use crate::types::AggTrade;
use smallvec::SmallVec;

use super::drawdown::{compute_max_drawdown, compute_max_runup};
use super::ith::{bear_ith, bull_ith};
use super::normalize::{
    normalize_cv, normalize_drawdown, normalize_epochs, normalize_excess, normalize_runup,
};

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

/// Issue #96 Task #56: Compute volume moments (skewness and kurtosis) in single pass.
/// Helper function that encapsulates volume statistics accumulation for reusability.
///
/// Computes all volume moments (mean, m2, m3, m4) with Welford's online algorithm.
/// Returns (volume_skewness, volume_kurtosis) where each is Option<f64>.
fn compute_volume_moments(volumes: &[f64]) -> (Option<f64>, Option<f64>) {
    let n = volumes.len();

    if n < 3 {
        return (None, None);
    }

    // Phase 1: Compute mean
    let sum_vol = volumes.iter().sum::<f64>();
    let mean_v = sum_vol / n as f64;

    // Phase 2: Compute central moments (m2, m3, m4) in single pass
    let (m2, m3, m4) = volumes.iter().fold((0.0, 0.0, 0.0), |(m2, m3, m4), &v| {
        let d = v - mean_v;
        let d2 = d * d;
        (m2 + d2, m3 + d2 * d, m4 + d2 * d2)
    });

    let m2_norm = m2 / n as f64;
    let m3_norm = m3 / n as f64;
    let m4_norm = m4 / n as f64;

    let std_v = m2_norm.sqrt();

    if std_v > f64::EPSILON {
        let skew = Some(m3_norm / std_v.powi(3));
        let kurt = Some(m4_norm / std_v.powi(4) - 3.0); // Excess kurtosis
        (skew, kurt)
    } else {
        (None, None)
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
pub fn compute_intra_bar_features(trades: &[AggTrade]) -> IntraBarFeatures {
    let n = trades.len();

    if n < 2 {
        return IntraBarFeatures {
            intra_trade_count: Some(n as u32),
            ..Default::default()
        };
    }

    // Extract price series from trades
    let prices: Vec<f64> = trades.iter().map(|t| t.price.to_f64()).collect();

    // Normalize prices to start at 1.0 for ITH computation
    let first_price = prices[0];
    if first_price <= 0.0 || !first_price.is_finite() {
        return IntraBarFeatures {
            intra_trade_count: Some(n as u32),
            ..Default::default()
        };
    }
    let normalized: Vec<f64> = prices.iter().map(|p| p / first_price).collect();

    // Compute max_drawdown and max_runup (used as TMAEG - no magic numbers)
    let max_dd = compute_max_drawdown(&normalized);
    let max_ru = compute_max_runup(&normalized);

    // Compute Bull ITH with max_drawdown as TMAEG
    let bull_result = bull_ith(&normalized, max_dd);

    // Compute Bear ITH with max_runup as TMAEG
    let bear_result = bear_ith(&normalized, max_ru);

    // Sum excess gains for normalization
    let bull_excess_sum: f64 = bull_result.excess_gains.iter().sum();
    let bear_excess_sum: f64 = bear_result.excess_gains.iter().sum();

    // Compute statistical features
    let stats = compute_statistical_features(trades, &prices);

    // Compute complexity features (only if enough trades)
    let hurst = if n >= 64 {
        Some(compute_hurst_dfa(&normalized))
    } else {
        None
    };
    let pe = if n >= 60 {
        Some(compute_permutation_entropy(&prices, 3))
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

    // Volume aggregation
    let mut buy_vol = 0.0_f64;
    let mut sell_vol = 0.0_f64;
    let mut buy_count = 0_u32;
    let mut sell_count = 0_u32;
    let mut total_turnover = 0.0_f64;
    let volumes: Vec<f64> = trades.iter().map(|t| t.volume.to_f64()).collect();

    for trade in trades {
        let vol = trade.volume.to_f64();
        let price = trade.price.to_f64();
        total_turnover += price * vol;

        if trade.is_buyer_maker {
            sell_vol += vol;
            sell_count += trade.individual_trade_count() as u32;
        } else {
            buy_vol += vol;
            buy_count += trade.individual_trade_count() as u32;
        }
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
    let duration_sec = duration_us as f64 / 1_000_000.0;

    // Intensity: trades per second
    let intensity = if duration_sec > f64::EPSILON {
        n as f64 / duration_sec
    } else {
        n as f64 // Instant bar
    };

    // VWAP position (Issue #96 Task #51: single-pass high/low computation)
    let vwap = if total_vol > f64::EPSILON {
        total_turnover / total_vol
    } else {
        prices.first().copied().unwrap_or(0.0)
    };
    // Single pass for both high and low (instead of two folds)
    let (high, low) = prices.iter().fold((f64::NEG_INFINITY, f64::INFINITY), |(h, l), &p| {
        (h.max(p), l.min(p))
    });
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

    // Burstiness (requires >= 2 trades for inter-arrival times)
    let burstiness = if n >= 2 {
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp).collect();
        let intervals: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .collect();

        if !intervals.is_empty() {
            let mean_tau: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance: f64 = intervals
                .iter()
                .map(|&x| (x - mean_tau).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;
            let std_tau = variance.sqrt();

            if (std_tau + mean_tau).abs() > f64::EPSILON {
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

    // Issue #96 Task #55+56: Use consolidated helper for volume moments
    // Encapsulates 2-pass computation (mean + all central moments)
    let (volume_skew, volume_kurt) = compute_volume_moments(&volumes);

    // Kaufman Efficiency Ratio (requires >= 2 trades)
    let kaufman_er = if n >= 2 {
        let net_move = (prices[n - 1] - prices[0]).abs();
        let path_length: f64 = prices.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

        if path_length > f64::EPSILON {
            Some((net_move / path_length).clamp(0.0, 1.0))
        } else {
            Some(1.0) // No movement = perfectly efficient
        }
    } else {
        None
    };

    // Garman-Klass volatility
    let open = prices[0];
    let close = prices[n - 1];
    let garman_klass_vol = if high > low && high > 0.0 && open > 0.0 {
        let hl_ratio = (high / low).ln();
        let co_ratio = (close / open).ln();
        let gk_var = 0.5 * hl_ratio.powi(2) - (2.0 * 2.0_f64.ln() - 1.0) * co_ratio.powi(2);
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

    // Compute cumulative deviation from mean
    let mean: f64 = prices.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = prices
        .iter()
        .scan(0.0, |acc, &p| {
            *acc += p - mean;
            Some(*acc)
        })
        .collect();

    // Scale range from n/4 to n/2 (using powers of 2 for efficiency)
    let min_scale = (n / 4).max(8);
    let max_scale = n / 2;

    let mut log_scales = Vec::new();
    let mut log_fluctuations = Vec::new();

    let mut scale = min_scale;
    while scale <= max_scale {
        let num_segments = n / scale;
        if num_segments < 2 {
            break;
        }

        let mut total_fluctuation = 0.0;
        let mut segment_count = 0;

        for seg in 0..num_segments {
            let start = seg * scale;
            let end = start + scale;
            if end > n {
                break;
            }

            // Linear detrend via least squares
            let x_mean = (scale - 1) as f64 / 2.0;
            let mut xy_sum = 0.0;
            let mut xx_sum = 0.0;
            let mut y_sum = 0.0;

            for (i, &yi) in y[start..end].iter().enumerate() {
                let xi = i as f64;
                xy_sum += (xi - x_mean) * yi;
                xx_sum += (xi - x_mean).powi(2);
                y_sum += yi;
            }

            let y_mean = y_sum / scale as f64;
            let slope = if xx_sum.abs() > f64::EPSILON {
                xy_sum / xx_sum
            } else {
                0.0
            };

            // Compute RMS of detrended segment
            let mut rms = 0.0;
            for (i, &yi) in y[start..end].iter().enumerate() {
                let trend = y_mean + slope * (i as f64 - x_mean);
                rms += (yi - trend).powi(2);
            }
            rms = (rms / scale as f64).sqrt();

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
        xy_sum += (x - x_mean) * (y - y_mean);
        xx_sum += (x - x_mean).powi(2);
    }

    let hurst = if xx_sum.abs() > f64::EPSILON {
        xy_sum / xx_sum
    } else {
        0.5
    };

    // Soft-clamp to [0, 1] using sigmoid
    1.0 / (1.0 + (-4.0 * (hurst - 0.5)).exp())
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

    // OPTIMIZATION: Hoist SmallVec allocation outside loop for reuse with .clear()
    let mut indices = SmallVec::<[usize; 4]>::new();

    for i in 0..num_patterns {
        let window = &prices[i..i + m];

        // Create sorted indices (ordinal pattern) using SmallVec
        indices.clear();
        for j in 0..m {
            indices.push(j);
        }

        // OPTIMIZATION: Early-exit if already sorted (common in trending data)
        // Check if indices == [0, 1, 2, ..., m-1] without allocation
        let is_sorted = indices.iter().enumerate().all(|(pos, &idx)| idx == pos);
        if is_sorted {
            // Pattern index is 0 (identity permutation - perfect ascending order)
            pattern_counts[0] += 1;
        } else {
            // Perform sort and compute pattern index
            indices.sort_by(|&a, &b| {
                window[a]
                    .partial_cmp(&window[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Convert sorted indices to pattern index (0 to m!-1)
            let pattern_idx = ordinal_indices_to_pattern_index(&indices);
            pattern_counts[pattern_idx] += 1;
        }
    }

    // Compute Shannon entropy from pattern counts
    let mut entropy = 0.0;
    for i in 0..max_patterns {
        let count = pattern_counts[i];
        if count > 0 {
            let p = count as f64 / num_patterns as f64;
            entropy -= p * p.ln();
        }
    }

    // Normalize by maximum entropy (log(m!))
    let max_entropy = (max_patterns as f64).ln();
    if max_entropy > f64::EPSILON {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Convert ordinal indices to pattern index using Lehmer code
/// For m=3: [0,1,2]→0, [0,2,1]→1, [1,0,2]→2, [1,2,0]→3, [2,0,1]→4, [2,1,0]→5
#[inline]
fn ordinal_indices_to_pattern_index(indices: &smallvec::SmallVec<[usize; 4]>) -> usize {
    match indices.len() {
        2 => {
            // m=2: 2 patterns
            if indices[0] < indices[1] { 0 } else { 1 }
        }
        3 => {
            // m=3: 6 patterns (3!) - compute Lehmer code
            let mut code = 0usize;
            let factors = [1, 2, 1];
            for (i, &idx) in indices.iter().enumerate() {
                let mut lesser = 0;
                for j in (i + 1)..3 {
                    if indices[j] < idx {
                        lesser += 1;
                    }
                }
                code += lesser * factors[i];
            }
            code
        }
        4 => {
            // m=4: 24 patterns (4!) - compute Lehmer code
            let mut code = 0usize;
            let factors = [6, 2, 1, 1];
            for (i, &idx) in indices.iter().enumerate() {
                let mut lesser = 0;
                for j in (i + 1)..4 {
                    if indices[j] < idx {
                        lesser += 1;
                    }
                }
                code += lesser * factors[i];
            }
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

    let max_entropy = (factorial(m) as f64).ln();
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
}
