//! Inter-bar math helper functions
//! Extracted from interbar.rs (Phase 2e refactoring)
//!
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59

use crate::interbar_types::TradeSnapshot;

/// Compute Kyle's Lambda (normalized version)
///
/// Formula: lambda = ((price_end - price_start) / price_start) / ((buy_vol - sell_vol) / total_vol)
///
/// Reference: Kyle (1985), Hasbrouck (2009)
///
/// Interpretation:
/// - lambda > 0: Price moves in direction of order flow (normal)
/// - lambda < 0: Price moves against order flow (unusual)
/// - |lambda| high: Large price impact per unit imbalance (illiquid)
pub(crate) fn compute_kyle_lambda(lookback: &[&TradeSnapshot]) -> f64 {
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

/// Compute Burstiness (Goh-Barabasi)
///
/// Formula: B = (sigma_tau - mu_tau) / (sigma_tau + mu_tau)
///
/// Reference: Goh & Barabasi (2008), EPL, Vol. 81, 48002
///
/// Interpretation:
/// - B = -1: Perfectly regular (periodic) arrivals
/// - B = 0: Poisson process
/// - B = +1: Maximally bursty
pub(crate) fn compute_burstiness(lookback: &[&TradeSnapshot]) -> f64 {
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
/// Skewness: E[(V-mu)^3] / sigma^3 (Fisher-Pearson coefficient)
/// Excess Kurtosis: E[(V-mu)^4] / sigma^4 - 3 (normal distribution = 0)
pub(crate) fn compute_volume_moments(lookback: &[&TradeSnapshot]) -> (f64, f64) {
    let volumes: Vec<f64> = lookback.iter().map(|t| t.volume.to_f64()).collect();
    let n = volumes.len() as f64;

    if n < 3.0 {
        return (0.0, 0.0);
    }

    let mu = volumes.iter().sum::<f64>() / n;

    // Central moments — single pass for m2/m3/m4 (was 3 separate passes)
    let (m2, m3, m4) = volumes.iter().fold((0.0, 0.0, 0.0), |(m2, m3, m4), v| {
        let d = v - mu;
        let d2 = d * d;
        (m2 + d2, m3 + d2 * d, m4 + d2 * d2)
    });
    let m2 = m2 / n;
    let m3 = m3 / n;
    let m4 = m4 / n;

    let sigma = m2.sqrt();

    if sigma < f64::EPSILON {
        return (0.0, 0.0); // All same volume
    }

    let skewness = m3 / sigma.powi(3);
    let kurtosis = m4 / sigma.powi(4) - 3.0; // Excess kurtosis

    (skewness, kurtosis)
}

/// Compute Kaufman Efficiency Ratio
///
/// Formula: ER = |net movement| / sum(|individual movements|)
///
/// Reference: Kaufman (1995) - Smarter Trading
///
/// Range: [0, 1] where 1 = perfect trend, 0 = pure noise
pub(crate) fn compute_kaufman_er(prices: &[f64]) -> f64 {
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

/// Garman-Klass volatility coefficient: 2*ln(2) - 1
/// Precomputed to avoid repeated calculation in every call
/// Exact value: 0.3862943611198906
const GARMAN_KLASS_COEFFICIENT: f64 = 0.3862943611198906;

/// Compute Garman-Klass volatility estimator
///
/// Formula: sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
///
/// Reference: Garman & Klass (1980), Journal of Business, vol. 53, no. 1
///
/// Coefficient precomputed: (2*ln(2) - 1) = 0.386294...
pub(crate) fn compute_garman_klass(lookback: &[&TradeSnapshot]) -> f64 {
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

    let variance = 0.5 * log_hl.powi(2) - GARMAN_KLASS_COEFFICIENT * log_co.powi(2);

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
pub fn compute_hurst_dfa(prices: &[f64]) -> f64 {
    // Issue #96 Phase 3b: Integrate evrom/hurst for 4-5x speedup
    // Rescaled Range (R/S) Analysis: O(n log n) vs DFA O(n²)

    const MIN_SAMPLES: usize = 64;
    if prices.len() < MIN_SAMPLES {
        return 0.5; // Neutral (insufficient data)
    }

    // Use evrom/hurst R/S Analysis (O(n log n), 4-5x faster than DFA)
    // Note: hurst::rssimple() takes owned Vec, so clone prices
    let h = hurst::rssimple(prices.to_vec());

    // Soft clamp to [0, 1] using tanh (matches DFA output normalization)
    soft_clamp_hurst(h)
}

/// Soft clamp Hurst to [0, 1] using tanh
///
/// Formula: 0.5 + 0.5 * tanh((x - 0.5) * 4)
///
/// Maps 0.5 -> 0.5, and asymptotically approaches 0 or 1 for extreme values
#[inline]
pub(crate) fn soft_clamp_hurst(h: f64) -> f64 {
    0.5 + 0.5 * ((h - 0.5) * 4.0).tanh()
}

/// Compute Permutation Entropy
///
/// Formula: H_PE = -sum p_pi * ln(p_pi) / ln(m!)
///
/// Reference: Bandt & Pompe (2002), Phys. Rev. Lett. 88, 174102
///
/// Output range: [0, 1] where 0 = deterministic, 1 = completely random
pub(crate) fn compute_permutation_entropy(prices: &[f64]) -> f64 {
    const M: usize = 3; // Embedding dimension (Bandt & Pompe recommend 3-7)
    const MIN_SAMPLES: usize = 60; // Rule of thumb: 10 * m! = 10 * 6 = 60 for m=3

    if prices.len() < MIN_SAMPLES {
        return 1.0; // Insufficient data -> max entropy (no information)
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
    let max_entropy = 6.0_f64.ln(); // ~ 1.7918

    entropy / max_entropy
}

/// Get ordinal pattern index for m=3 (0-5)
///
/// Patterns (lexicographic order):
/// 0: 012 (a <= b <= c)
/// 1: 021 (a <= c < b)
/// 2: 102 (b < a <= c)
/// 3: 120 (b <= c < a)
/// 4: 201 (c < a <= b)
/// 5: 210 (c < b < a)
pub(crate) fn ordinal_pattern_index_m3(a: f64, b: f64, c: f64) -> usize {
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
mod hurst_accuracy_tests {
    use super::*;

    #[test]
    fn test_hurst_accuracy_trending() {
        // Strongly trending series (H > 0.5)
        let mut prices = vec![0.0; 256];
        for i in 0..256 {
            prices[i] = i as f64 * 1.0; // Linear trend
        }

        let dfa_h = compute_hurst_dfa(&prices);
        let rs_h = hurst::rssimple(prices.clone());

        println!("Trending series:");
        println!("  DFA H = {:.4}", dfa_h);
        println!("  R/S H = {:.4}", rs_h);
        println!("  Both > 0.5? DFA={}, RS={}", dfa_h > 0.5, rs_h > 0.5);

        // Both should agree on trending direction (H > 0.5)
        assert!(dfa_h > 0.5, "DFA should detect trending");
        assert!(rs_h > 0.5, "R/S should detect trending");
    }

    #[test]
    fn test_hurst_accuracy_mean_reverting() {
        // Mean-reverting series (H < 0.5)
        let mut prices = vec![0.5; 256];
        for i in 0..256 {
            prices[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        }

        let dfa_h = compute_hurst_dfa(&prices);
        let rs_h = hurst::rssimple(prices.clone());

        println!("Mean-reverting series:");
        println!("  DFA H = {:.4}", dfa_h);
        println!("  R/S H = {:.4}", rs_h);
        println!("  Both < 0.5? DFA={}, RS={}", dfa_h < 0.5, rs_h < 0.5);

        // Both should agree on mean-reversion (H < 0.5)
        assert!(dfa_h < 0.5, "DFA should detect mean-reversion");
        assert!(rs_h < 0.5, "R/S should detect mean-reversion");
    }

    #[test]
    fn test_hurst_accuracy_random_walk() {
        // Brownian motion / random walk (H ≈ 0.5)
        let mut prices = vec![0.0; 256];
        let mut rng = 12345u64;
        prices[0] = 0.0;

        for i in 1..256 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let step = if (rng >> 16) & 1 == 0 { 1.0 } else { -1.0 };
            prices[i] = prices[i - 1] + step;
        }

        let dfa_h = compute_hurst_dfa(&prices);
        let rs_h = hurst::rssimple(prices.clone());

        println!("Random walk series:");
        println!("  DFA H = {:.4}", dfa_h);
        println!("  R/S H = {:.4}", rs_h);
        println!("  Both ≈ 0.5? DFA={:.2}, RS={:.2}", dfa_h, rs_h);
    }
}
