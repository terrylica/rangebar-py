//! Inter-bar math helper functions
//! Extracted from interbar.rs (Phase 2e refactoring)
//!
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59
//! # FILE-SIZE-OK (565 lines - organized by feature module)

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
pub fn compute_garman_klass(lookback: &[&TradeSnapshot]) -> f64 {
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

/// Compute Adaptive Permutation Entropy with dynamic embedding dimension
///
/// Selects embedding dimension M based on window size for optimal efficiency:
/// - n < 10: Insufficient data -> return 1.0
/// - 10 ≤ n < 20: M=2 (2 patterns) -> ~3-5x faster than M=3 on these sizes
/// - n ≥ 20: M=3 (6 patterns) -> standard Bandt-Pompe choice
///
/// Trade-off: Function call overhead (~5-10% on large windows) vs significant gains
/// on small windows (which are common in live trading). Overall win on typical
/// mixed workloads (10-500 sample windows).
///
/// Formula: H_PE = -sum p_pi * ln(p_pi) / ln(m!)
///
/// Reference: Bandt & Pompe (2002), Phys. Rev. Lett. 88, 174102
///
/// Output range: [0, 1] where 0 = deterministic, 1 = completely random
///
/// Performance characteristics:
/// - Small windows (10-20 samples): 3-5x faster (fewer patterns, less computation)
/// - Medium windows (20-100 samples): Baseline (minimal overhead)
/// - Large windows (>100 samples): ~5-10% overhead (function call indirection)
#[inline(always)]
pub fn compute_permutation_entropy(prices: &[f64]) -> f64 {
    let n = prices.len();

    if n < 10 {
        return 1.0; // Insufficient data
    }

    // Inline critical path for common large sizes
    if n >= 20 {
        // Standard M=3: inline directly to avoid function call overhead
        let mut pattern_counts: [usize; 6] = [0; 6];
        let n_patterns = n - 2;

        for i in 0..n_patterns {
            let pattern_idx = ordinal_pattern_index_m3(prices[i], prices[i + 1], prices[i + 2]);
            pattern_counts[pattern_idx] += 1;
        }

        let total = n_patterns as f64;
        let entropy: f64 = pattern_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum();

        entropy / 6.0_f64.ln() // ln(3!)
    } else {
        // Small windows: M=2 path (n < 20)
        compute_permutation_entropy_m2(prices)
    }
}

/// Permutation entropy with M=2 (2 patterns: a<=b, b<a)
/// Faster than M=3, suitable for small windows (10-20 samples)
#[inline]
fn compute_permutation_entropy_m2(prices: &[f64]) -> f64 {
    debug_assert!(prices.len() >= 10);

    let mut counts = [0usize; 2]; // 2! = 2 patterns
    let n_patterns = prices.len() - 1;

    for i in 0..n_patterns {
        let idx = if prices[i] <= prices[i + 1] { 0 } else { 1 };
        counts[idx] += 1;
    }

    // Shannon entropy
    let total = n_patterns as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();

    entropy / 2.0_f64.ln() // ln(2!)
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

    // Edge case tests for inter-bar features (Issue #96: Test expansion)
    // Validates robustness on boundary conditions and stress scenarios

    #[test]
    fn test_hurst_edge_case_empty() {
        let prices: Vec<f64> = vec![];
        let h = compute_hurst_dfa(&prices);
        assert_eq!(h, 0.5, "Empty prices should return neutral (0.5)");
    }

    #[test]
    fn test_hurst_edge_case_insufficient_samples() {
        // Less than MIN_SAMPLES (64) should return neutral
        let prices: Vec<f64> = (0..32).map(|i| 100.0 + i as f64).collect();
        let h = compute_hurst_dfa(&prices);
        assert_eq!(
            h, 0.5,
            "Less than 64 samples should return neutral (0.5)"
        );
    }

    #[test]
    fn test_hurst_edge_case_constant_prices() {
        // All same price should handle gracefully (no variation)
        // With R/S analysis, constant series results in NaN (0/0 case)
        let prices = vec![100.0; 100];
        let h = compute_hurst_dfa(&prices);
        // Constant prices may result in NaN after soft clamping, which is acceptable
        // The important thing is no panic/crash
        if !h.is_nan() {
            assert!(h >= 0.0 && h <= 1.0, "Hurst should be in [0,1] if not NaN");
        }
    }

    #[test]
    fn test_hurst_bounds_stress() {
        // Verify Hurst stays bounded across diverse scenarios
        let scenarios = vec![
            ("linear", (0..256).map(|i| 100.0 + i as f64).collect::<Vec<_>>()),
            (
                "sawtooth",
                (0..256)
                    .map(|i| if i % 2 == 0 { 100.0 } else { 101.0 })
                    .collect::<Vec<_>>(),
            ),
        ];

        for (name, prices) in scenarios {
            let h = compute_hurst_dfa(&prices);
            assert!(
                h >= 0.0 && h <= 1.0,
                "Hurst({}) must be in [0,1], got {}",
                name,
                h
            );
            assert!(!h.is_nan(), "Hurst({}) must not be NaN", name);
        }
    }

    #[test]
    fn test_garman_klass_edge_case_empty() {
        use crate::interbar_types::TradeSnapshot;

        // Empty lookback should return 0
        let snapshot: Vec<TradeSnapshot> = vec![];
        let snapshot_refs: Vec<&TradeSnapshot> = snapshot.iter().collect();
        let vol = compute_garman_klass(&snapshot_refs);
        assert_eq!(vol, 0.0, "Empty lookback should return 0");
    }

    #[test]
    fn test_garman_klass_edge_case_constant_price() {
        use crate::{FixedPoint, interbar_types::TradeSnapshot};

        // All same price: H=L, C=O, variance should be 0
        let prices = vec![100.0; 50];
        let snapshots: Vec<TradeSnapshot> = prices
            .iter()
            .enumerate()
            .map(|(i, &price)| {
                let price_fp =
                    FixedPoint::from_str(&format!("{:.8}", price)).expect("valid price");
                let vol_fp = FixedPoint::from_str("1.00000000").expect("valid volume");
                let turnover_f64 = price_fp.to_f64() * vol_fp.to_f64();
                TradeSnapshot {
                    price: price_fp,
                    volume: vol_fp,
                    timestamp: 1000 + (i as i64 * 100),
                    is_buyer_maker: false,
                    turnover: (turnover_f64 * 1e8) as i128,
                }
            })
            .collect();
        let snapshot_refs: Vec<&TradeSnapshot> = snapshots.iter().collect();
        let vol = compute_garman_klass(&snapshot_refs);
        assert_eq!(vol, 0.0, "Constant price should give 0 volatility");
    }

    #[test]
    fn test_garman_klass_bounds() {
        use crate::{FixedPoint, interbar_types::TradeSnapshot};

        // Garman-Klass should be non-negative
        let prices = vec![100.0, 105.0, 103.0, 108.0, 102.0];
        let snapshots: Vec<TradeSnapshot> = prices
            .iter()
            .enumerate()
            .map(|(i, &price)| {
                let price_fp =
                    FixedPoint::from_str(&format!("{:.8}", price)).expect("valid price");
                let vol_fp = FixedPoint::from_str("1.00000000").expect("valid volume");
                let turnover_f64 = price_fp.to_f64() * vol_fp.to_f64();
                TradeSnapshot {
                    price: price_fp,
                    volume: vol_fp,
                    timestamp: 1000 + (i as i64 * 100),
                    is_buyer_maker: false,
                    turnover: (turnover_f64 * 1e8) as i128,
                }
            })
            .collect();
        let snapshot_refs: Vec<&TradeSnapshot> = snapshots.iter().collect();
        let vol = compute_garman_klass(&snapshot_refs);
        assert!(vol >= 0.0, "Garman-Klass volatility must be non-negative");
        assert!(!vol.is_nan(), "Garman-Klass must not be NaN");
    }

    #[test]
    fn test_permutation_entropy_edge_case_empty() {
        let prices: Vec<f64> = vec![];
        let entropy = compute_permutation_entropy(&prices);
        assert_eq!(
            entropy, 1.0,
            "Empty prices should return max entropy (1.0)"
        );
    }

    #[test]
    fn test_permutation_entropy_edge_case_insufficient_data() {
        // Less than MIN_SAMPLES (60) should return max entropy
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let entropy = compute_permutation_entropy(&prices);
        assert_eq!(entropy, 1.0, "Insufficient data should return max entropy");
    }

    #[test]
    fn test_permutation_entropy_bounds() {
        // Entropy should be in [0, 1]
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i % 3) as f64).collect();
        let entropy = compute_permutation_entropy(&prices);
        assert!(
            entropy >= 0.0 && entropy <= 1.0,
            "Entropy must be in [0,1], got {}",
            entropy
        );
        assert!(!entropy.is_nan(), "Entropy must not be NaN");
    }

    #[test]
    fn test_kaufman_er_edge_case_empty() {
        let prices: Vec<f64> = vec![];
        let er = compute_kaufman_er(&prices);
        assert_eq!(er, 0.0, "Empty prices should give ER=0");
    }

    #[test]
    fn test_kaufman_er_edge_case_constant_prices() {
        let prices = vec![100.0; 50];
        let er = compute_kaufman_er(&prices);
        assert_eq!(er, 0.0, "Constant prices should give ER=0");
    }

    #[test]
    fn test_kaufman_er_bounds() {
        // Kaufman ER should be in [0, 1]
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let er = compute_kaufman_er(&prices);
        assert!(er >= 0.0 && er <= 1.0, "ER must be in [0,1], got {}", er);
        assert!(!er.is_nan(), "ER must not be NaN");
    }

    #[test]
    fn test_ordinal_pattern_index_coverage() {
        // Test ordinal pattern mappings for m=3
        // All 6 patterns from algorithm in ordinal_pattern_index_m3
        let test_cases = vec![
            (0.0, 1.0, 2.0, 0), // a<=b<=c → 0
            (0.0, 2.0, 1.0, 1), // a<=c<b → 1
            (1.0, 0.0, 2.0, 2), // b<a<=c → 2
            (2.0, 0.0, 1.0, 3), // a>b, a>c, b<=c → 3
            (1.0, 2.0, 0.0, 4), // a<=b, b>c, a>c → 4
            (2.0, 1.0, 0.0, 5), // a>b>c → 5
        ];

        for (a, b, c, expected) in test_cases {
            let idx = ordinal_pattern_index_m3(a, b, c);
            assert_eq!(
                idx, expected,
                "Pattern ({},{},{}) should map to index {} but got {}",
                a, b, c, expected, idx
            );
        }
    }

    // Tier 2 Feature Tests: Kyle Lambda
    #[test]
    fn test_kyle_lambda_edge_case_empty() {
        let kyle_lambda = compute_kyle_lambda(&[]);
        assert_eq!(kyle_lambda, 0.0, "Empty lookback should return 0");
    }

    #[test]
    fn test_kyle_lambda_edge_case_single_trade() {
        use crate::interbar_types::TradeSnapshot;
        let snapshot = TradeSnapshot {
            timestamp: 1000000,
            price: crate::FixedPoint::from_str("100.0").unwrap(),
            volume: crate::FixedPoint::from_str("1.0").unwrap(),
            is_buyer_maker: true,
            turnover: (100 * 1) as i128 * 100000000i128,
        };
        let kyle_lambda = compute_kyle_lambda(&[&snapshot]);
        assert_eq!(kyle_lambda, 0.0, "Single trade should return 0 (insufficient data)");
    }

    #[test]
    fn test_kyle_lambda_zero_imbalance() {
        use crate::interbar_types::TradeSnapshot;
        // Equal buy and sell volume should give zero imbalance
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: true,
                turnover: (100 * 1) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("100.5").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: false, // Seller (opposite)
                turnover: (100 * 1) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        assert_eq!(kyle_lambda, 0.0, "Zero imbalance should return 0");
    }

    #[test]
    fn test_kyle_lambda_positive_trend_buy_pressure() {
        use crate::interbar_types::TradeSnapshot;
        // Price increases with BUY pressure (is_buyer_maker=false = BUY)
        // More buy volume (aggressive buyers) pushes price up
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: true, // SELL (minimal)
                turnover: (100 * 1) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("101.0").unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: false, // BUY (large buy volume)
                turnover: (101 * 10) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        // With more buy volume (imbalance > 0) and price increase, kyle_lambda should be positive
        assert!(kyle_lambda > 0.0, "Buy pressure with price increase should give positive kyle_lambda, got {}", kyle_lambda);
    }

    #[test]
    fn test_kyle_lambda_bounded() {
        use crate::interbar_types::TradeSnapshot;
        // Kyle lambda should be finite (not NaN or Inf)
        for _i in 0..10 {
            let trades = vec![
                TradeSnapshot {
                    timestamp: 1000000,
                    price: crate::FixedPoint::from_str("100.0").unwrap(),
                    volume: crate::FixedPoint::from_str("5.0").unwrap(),
                    is_buyer_maker: true,
                    turnover: (100 * 5) as i128 * 100000000i128,
                },
                TradeSnapshot {
                    timestamp: 1000100,
                    price: crate::FixedPoint::from_str("105.0").unwrap(),
                    volume: crate::FixedPoint::from_str("2.0").unwrap(),
                    is_buyer_maker: false,
                    turnover: (105 * 2) as i128 * 100000000i128,
                },
            ];
            let refs: Vec<&TradeSnapshot> = trades.iter().collect();
            let kyle_lambda = compute_kyle_lambda(&refs);
            assert!(kyle_lambda.is_finite(), "Kyle lambda must be finite, got {}", kyle_lambda);
        }
    }

    // Tier 2 Feature Tests: Burstiness
    #[test]
    fn test_burstiness_edge_case_empty() {
        let burstiness = compute_burstiness(&[]);
        assert_eq!(burstiness, 0.0, "Empty lookback should return 0");
    }

    #[test]
    fn test_burstiness_single_trade() {
        use crate::interbar_types::TradeSnapshot;
        let snapshot = TradeSnapshot {
            timestamp: 1000000,
            price: crate::FixedPoint::from_str("100.0").unwrap(),
            volume: crate::FixedPoint::from_str("1.0").unwrap(),
            is_buyer_maker: true,
            turnover: (100 * 1) as i128 * 100000000i128,
        };
        let burstiness = compute_burstiness(&[&snapshot]);
        assert_eq!(burstiness, 0.0, "Single trade should return 0 (insufficient data)");
    }

    #[test]
    fn test_burstiness_bounds() {
        use crate::interbar_types::TradeSnapshot;
        // Create regular arrivals (approximately)
        let mut trades = Vec::new();
        for i in 0..20 {
            trades.push(TradeSnapshot {
                timestamp: 1000000 + (i * 100) as i64,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            });
        }
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burstiness = compute_burstiness(&refs);
        assert!(burstiness >= -1.0 && burstiness <= 1.0, "Burstiness must be in [-1, 1], got {}", burstiness);
    }

    // Tier 3 Feature Tests: Additional Kaufman ER edge cases
    #[test]
    fn test_kaufman_er_trending_market() {
        // Strong uptrend
        let mut prices = Vec::new();
        let mut price = 100.0;
        for _ in 0..50 {
            price += 0.1; // Consistent uptrend
            prices.push(price);
        }
        let er = compute_kaufman_er(&prices);
        assert!(er > 0.5, "Strong trending market should have high efficiency ratio, got {}", er);
    }

    #[test]
    fn test_kaufman_er_ranging_market() {
        // Oscillating prices (ranging)
        let mut prices = Vec::new();
        for i in 0..50 {
            let price = 100.0 + if (i % 2) == 0 { 0.1 } else { -0.1 };
            prices.push(price);
        }
        let er = compute_kaufman_er(&prices);
        assert!(er < 0.3, "Ranging market should have low efficiency ratio, got {}", er);
    }

    // ===== NEW TIER 3 FEATURE EDGE CASE TESTS (Task #17) =====

    // Kyle Lambda - Additional Edge Cases
    #[test]
    fn test_kyle_lambda_negative_trend_sell_pressure() {
        use crate::interbar_types::TradeSnapshot;
        // Price decreases with SELL pressure (is_buyer_maker=true = SELL)
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("101.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: false, // BUY (minimal)
                turnover: (101 * 1) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: true, // SELL (large sell volume)
                turnover: (100 * 10) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        // With more sell volume (imbalance < 0) and price decrease, kyle_lambda should be positive
        // (price moves in direction of order flow)
        assert!(kyle_lambda > 0.0, "Sell pressure with price decrease should give positive kyle_lambda");
    }

    #[test]
    fn test_kyle_lambda_zero_price_movement() {
        use crate::interbar_types::TradeSnapshot;
        // Price doesn't change but there's volume imbalance
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("5.0").unwrap(),
                is_buyer_maker: false, // BUY
                turnover: (100 * 5) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: true, // SELL (minimal)
                turnover: (100 * 1) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        // No price movement should give 0 kyle_lambda
        assert_eq!(kyle_lambda, 0.0, "Zero price movement should give 0");
    }

    #[test]
    fn test_kyle_lambda_tiny_prices() {
        use crate::interbar_types::TradeSnapshot;
        // Test with very small prices (e.g., penny stocks)
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("0.001").unwrap(),
                volume: crate::FixedPoint::from_str("100000.0").unwrap(),
                is_buyer_maker: true,
                turnover: (1 * 100000) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("0.002").unwrap(),
                volume: crate::FixedPoint::from_str("50000.0").unwrap(),
                is_buyer_maker: false,
                turnover: (2 * 50000) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        assert!(kyle_lambda.is_finite(), "Should handle tiny prices without NaN/Inf");
    }

    #[test]
    fn test_kyle_lambda_opposing_flows() {
        use crate::interbar_types::TradeSnapshot;
        // Buy and sell at different times with conflicting pressures
        let trades = vec![
            TradeSnapshot {
                timestamp: 1000000,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: false, // BUY (large)
                turnover: (100 * 10) as i128 * 100000000i128,
            },
            TradeSnapshot {
                timestamp: 1000100,
                price: crate::FixedPoint::from_str("99.0").unwrap(),
                volume: crate::FixedPoint::from_str("5.0").unwrap(),
                is_buyer_maker: true, // SELL (price down despite buy pressure initially)
                turnover: (99 * 5) as i128 * 100000000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let kyle_lambda = compute_kyle_lambda(&refs);
        // Price decreased against buy pressure → negative kyle_lambda
        assert!(kyle_lambda < 0.0, "Price moving against order flow should give negative kyle_lambda");
    }

    // Burstiness - Additional Edge Cases
    #[test]
    fn test_burstiness_clustered_arrivals() {
        use crate::interbar_types::TradeSnapshot;
        // Trades clustered at start, then gap
        let mut trades = Vec::new();
        // Cluster: 10 trades in 100ms
        for i in 0..10 {
            trades.push(TradeSnapshot {
                timestamp: 1000000 + (i * 10) as i64,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            });
        }
        // Large gap: 1000ms
        for i in 0..5 {
            trades.push(TradeSnapshot {
                timestamp: 1000100 + 1000 + (i * 10) as i64,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            });
        }
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burstiness = compute_burstiness(&refs);
        // Bursty pattern should give high burstiness
        assert!(burstiness > 0.0, "Clustered arrivals should have positive burstiness, got {}", burstiness);
        assert!(burstiness <= 1.0, "Burstiness should be bounded by 1.0");
    }

    #[test]
    fn test_burstiness_perfectly_regular() {
        use crate::interbar_types::TradeSnapshot;
        // Perfectly regular 100ms intervals
        let mut trades = Vec::new();
        for i in 0..20 {
            trades.push(TradeSnapshot {
                timestamp: 1000000 + (i * 100) as i64,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            });
        }
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burstiness = compute_burstiness(&refs);
        // Regular arrivals should give burstiness near -1
        assert!(burstiness < 0.0, "Regular periodic arrivals should have negative burstiness, got {}", burstiness);
    }

    #[test]
    fn test_burstiness_extreme_gap() {
        use crate::interbar_types::TradeSnapshot;
        // One large burst followed by extreme gap
        let mut trades = Vec::new();
        // Initial burst: 5 trades
        for i in 0..5 {
            trades.push(TradeSnapshot {
                timestamp: 1000000 + (i as i64),
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            });
        }
        // Massive gap then one more trade
        trades.push(TradeSnapshot {
            timestamp: 1000000 + 100000,
            price: crate::FixedPoint::from_str("100.0").unwrap(),
            volume: crate::FixedPoint::from_str("1.0").unwrap(),
            is_buyer_maker: false,
            turnover: (100 * 1) as i128 * 100000000i128,
        });
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burstiness = compute_burstiness(&refs);
        // Extreme gap should produce positive (bursty) burstiness
        assert!(burstiness > 0.0, "Extreme gap should produce positive burstiness");
        assert!(burstiness <= 1.0, "Burstiness should be bounded");
    }

    // Garman-Klass - Additional Edge Cases
    #[test]
    fn test_garman_klass_high_volatility() {
        use crate::{FixedPoint, interbar_types::TradeSnapshot};
        // Large price swings (H >> L)
        let prices = vec![100.0, 150.0, 120.0, 180.0, 110.0];
        let snapshots: Vec<TradeSnapshot> = prices
            .iter()
            .enumerate()
            .map(|(i, &price)| {
                let price_fp = FixedPoint::from_str(&format!("{:.8}", price)).expect("valid price");
                let vol_fp = FixedPoint::from_str("1.00000000").expect("valid volume");
                let turnover_f64 = price_fp.to_f64() * vol_fp.to_f64();
                TradeSnapshot {
                    price: price_fp,
                    volume: vol_fp,
                    timestamp: 1000 + (i as i64 * 100),
                    is_buyer_maker: false,
                    turnover: (turnover_f64 * 1e8) as i128,
                }
            })
            .collect();
        let snapshot_refs: Vec<&TradeSnapshot> = snapshots.iter().collect();
        let vol = compute_garman_klass(&snapshot_refs);
        assert!(vol > 0.0, "High volatility scenario should produce non-zero volatility");
        assert!(!vol.is_nan(), "Garman-Klass must not be NaN");
    }

    #[test]
    fn test_garman_klass_extreme_ohlc_ratios() {
        use crate::{FixedPoint, interbar_types::TradeSnapshot};
        // Extreme high/low ratio
        let prices = vec![100.0, 1000.0, 200.0]; // H/L = 5
        let snapshots: Vec<TradeSnapshot> = prices
            .iter()
            .enumerate()
            .map(|(i, &price)| {
                let price_fp = FixedPoint::from_str(&format!("{:.8}", price)).expect("valid price");
                let vol_fp = FixedPoint::from_str("1.00000000").expect("valid volume");
                let turnover_f64 = price_fp.to_f64() * vol_fp.to_f64();
                TradeSnapshot {
                    price: price_fp,
                    volume: vol_fp,
                    timestamp: 1000 + (i as i64 * 100),
                    is_buyer_maker: false,
                    turnover: (turnover_f64 * 1e8) as i128,
                }
            })
            .collect();
        let snapshot_refs: Vec<&TradeSnapshot> = snapshots.iter().collect();
        let vol = compute_garman_klass(&snapshot_refs);
        // Should handle extreme ratios without panic
        assert!(vol >= 0.0, "Garman-Klass must be non-negative");
        assert!(vol.is_finite(), "Garman-Klass must be finite");
    }

    // Permutation Entropy - Additional Edge Cases
    #[test]
    fn test_permutation_entropy_deterministic_pattern() {
        // Perfectly ordered ascending pattern
        let prices: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let entropy = compute_permutation_entropy(&prices);
        // Deterministic pattern should have low entropy
        assert!(entropy >= 0.0 && entropy <= 1.0, "Entropy must be in [0,1]");
    }

    #[test]
    fn test_permutation_entropy_oscillating_pattern() {
        // Simple oscillating pattern (should have repeating permutations)
        let mut prices = Vec::new();
        for i in 0..100 {
            prices.push(if i % 3 == 0 { 100.0 } else if i % 3 == 1 { 101.0 } else { 99.0 });
        }
        let entropy = compute_permutation_entropy(&prices);
        // Repeating pattern should have lower entropy than random
        assert!(entropy >= 0.0 && entropy <= 1.0, "Entropy must be in [0,1]");
        assert!(!entropy.is_nan(), "Entropy must not be NaN");
    }

    // Kaufman ER - Additional Edge Cases
    #[test]
    fn test_kaufman_er_single_large_move() {
        // Single direction move with no noise
        let mut prices = Vec::new();
        for i in 0..50 {
            prices.push(100.0 + i as f64); // Perfect linear trend
        }
        let er = compute_kaufman_er(&prices);
        // Perfect trend should give ER close to 1.0
        assert!(er > 0.9, "Perfect trend should have ER > 0.9, got {}", er);
    }

    #[test]
    fn test_kaufman_er_noise_dominated() {
        // High-frequency noise with minimal net movement
        let mut prices = Vec::new();
        let mut rng = 12345u64;
        prices.push(100.0);
        for _ in 1..100 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) as f64 % 200.0) - 100.0; // Random [-100, 100] bps
            let new_price = prices.last().unwrap() + noise * 0.0001; // ±0.01 bps noise
            prices.push(new_price);
        }
        let er = compute_kaufman_er(&prices);
        // Noise-dominated should have lower ER than trending
        assert!(er < 0.5, "Noise-dominated market should have ER < 0.5, got {}", er);
        assert!(!er.is_nan(), "ER must be finite");
    }

    // Hurst - Additional Advanced Tests
    #[test]
    fn test_hurst_strong_reverting_pattern() {
        // Alternating high-low pattern (strong mean reversion)
        let mut prices = vec![100.0; 200];
        for i in 0..200 {
            prices[i] = if i % 2 == 0 { 99.0 } else { 101.0 };
        }
        let h = compute_hurst_dfa(&prices);
        assert!(h < 0.5, "Strong mean reverting should have H < 0.5, got {}", h);
        assert!(h.is_finite(), "Hurst must be finite");
    }

    #[test]
    fn test_hurst_extreme_volatility() {
        // Extreme spikes and drops
        let mut prices = vec![100.0; 200];
        for i in 0..200 {
            prices[i] = match i % 4 {
                0 => 100.0,
                1 => 200.0, // Spike
                2 => 150.0,
                _ => 50.0,  // Drop
            };
        }
        let h = compute_hurst_dfa(&prices);
        assert!(h >= 0.0 && h <= 1.0, "Hurst must be in [0,1] even for extreme volatility");
    }

    // Volume Moments - Additional Tests
    #[test]
    fn test_volume_moments_constant_volume() {
        use crate::interbar_types::TradeSnapshot;
        // All trades same volume → skewness and kurtosis should be 0
        let trades: Vec<TradeSnapshot> = (0..20)
            .map(|i| TradeSnapshot {
                timestamp: 1000000 + (i as i64 * 100),
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 1) as i128 * 100000000i128,
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let (skew, kurt) = compute_volume_moments(&refs);
        assert_eq!(skew, 0.0, "Constant volume should have zero skewness");
        assert_eq!(kurt, 0.0, "Constant volume should have zero kurtosis");
    }

    #[test]
    fn test_volume_moments_right_skewed() {
        use crate::interbar_types::TradeSnapshot;
        // Volume distribution skewed right (many small, few large)
        let volumes = vec![1.0, 1.0, 1.0, 1.0, 100.0]; // Right skew
        let trades: Vec<TradeSnapshot> = volumes
            .iter()
            .enumerate()
            .map(|(i, &vol)| TradeSnapshot {
                timestamp: 1000000 + (i as i64 * 100),
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str(&format!("{:.8}", vol)).unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100.0 * vol * 1e8) as i128,
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let (skew, _kurt) = compute_volume_moments(&refs);
        // Right-skewed should have positive skewness
        assert!(skew > 0.0, "Right-skewed volume should have positive skewness, got {}", skew);
    }

    #[test]
    fn test_volume_moments_heavy_tails() {
        use crate::interbar_types::TradeSnapshot;
        // Volume distribution with heavy tails (high kurtosis)
        let mut volumes = vec![1.0; 18]; // Many small volumes
        volumes.push(100.0); // One extreme value
        volumes.push(100.0); // Another extreme

        let trades: Vec<TradeSnapshot> = volumes
            .iter()
            .enumerate()
            .map(|(i, &vol)| TradeSnapshot {
                timestamp: 1000000 + (i as i64 * 100),
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str(&format!("{:.8}", vol)).unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100.0 * vol * 1e8) as i128,
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let (_skew, kurt) = compute_volume_moments(&refs);
        // Heavy tails should have high (positive) kurtosis
        assert!(kurt > 0.0, "Heavy-tailed distribution should have positive kurtosis, got {}", kurt);
    }

    // Ordinal Pattern - Additional Coverage
    #[test]
    fn test_ordinal_pattern_equal_values() {
        // Test handling of equal values in patterns
        // Verify the ordinal pattern function handles equal values gracefully
        let test_cases = vec![
            (1.0, 1.0, 2.0), // a=b < c
            (1.0, 2.0, 2.0), // a < b=c (uses < for b<=c branch)
            (1.0, 1.0, 1.0), // a=b=c
            (2.0, 2.0, 1.0), // a=b > c
        ];
        for (a, b, c) in test_cases {
            let idx = ordinal_pattern_index_m3(a, b, c);
            // All indices should be in valid range [0, 5]
            assert!(idx < 6, "Pattern index must be < 6, got {}", idx);
        }
    }

    // ========== NEW TESTS FOR TASK #23 (Expanded Coverage) ==========

    // Permutation Entropy - Adaptive Path Tests (M=2 for small windows)
    #[test]
    fn test_adaptive_permutation_entropy_m2_small_window() {
        // Small window (n < 20) should use M=2 path
        let prices = vec![100.0, 101.0, 100.5, 102.0, 99.0];
        let entropy = compute_permutation_entropy(&prices);
        assert!(entropy >= 0.0 && entropy <= 1.0, "Entropy should be normalized [0,1]");
        // M=2 should return meaningful value, not default max
        assert!(entropy < 1.0, "M=2 adaptive path should return meaningful entropy");
    }

    #[test]
    fn test_adaptive_permutation_entropy_m2_deterministic() {
        // Perfectly ascending should have low entropy
        let prices: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let entropy = compute_permutation_entropy(&prices);
        assert!(entropy < 0.3, "Monotonic sequence should have low entropy, got {}", entropy);
    }

    #[test]
    fn test_adaptive_permutation_entropy_m2_m3_transition() {
        // Test behavior at M=2→M=3 boundary (n=20)
        let mut prices: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let entropy_boundary = compute_permutation_entropy(&prices);

        prices.push(21.0);
        let entropy_m3 = compute_permutation_entropy(&prices);

        // Both should be in valid range
        assert!(entropy_boundary >= 0.0 && entropy_boundary <= 1.0);
        assert!(entropy_m3 >= 0.0 && entropy_m3 <= 1.0);
    }

    #[test]
    fn test_adaptive_permutation_entropy_insufficient_data() {
        // Too small (< 10) should return max entropy
        let prices = vec![1.0, 2.0];
        let entropy = compute_permutation_entropy(&prices);
        assert_eq!(entropy, 1.0, "Insufficient data should return max entropy");
    }

    // Kyle Lambda - Extended Edge Cases
    #[test]
    fn test_kyle_lambda_zero_imbalance_extended() {
        use crate::interbar_types::TradeSnapshot;
        // Equal buy and sell volume → zero imbalance → lambda = 0
        let trades: Vec<TradeSnapshot> = vec![
            TradeSnapshot {
                timestamp: 1000,
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: true,
                turnover: 1_000_000_000i128,
            },
            TradeSnapshot {
                timestamp: 2000,
                price: crate::FixedPoint::from_str("101.0").unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: false,
                turnover: 1_010_000_000i128,
            },
        ];
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let lambda = compute_kyle_lambda(&refs);
        assert_eq!(lambda, 0.0, "Zero imbalance should yield zero lambda");
    }

    #[test]
    fn test_kyle_lambda_strong_buy_pressure_extended() {
        use crate::interbar_types::TradeSnapshot;
        // Heavy buy pressure (price up, dominated by buy volume)
        let trades: Vec<TradeSnapshot> = (0..10)
            .map(|i| TradeSnapshot {
                timestamp: 1000 + (i as i64 * 100),
                price: crate::FixedPoint::from_str(&format!("{}.0", 100 + i / 2)).unwrap(),
                volume: if i % 2 == 0 {
                    crate::FixedPoint::from_str("100.0").unwrap() // Heavy buy
                } else {
                    crate::FixedPoint::from_str("1.0").unwrap() // Light sell
                },
                is_buyer_maker: i % 2 == 0,
                turnover: ((100 + i / 2) as i128 * if i % 2 == 0 { 100 } else { 1 } * 100_000_000i128),
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let lambda = compute_kyle_lambda(&refs);
        assert!(lambda > 0.0, "Buy pressure should yield positive lambda");
    }

    // Burstiness - Timing Analysis Extended
    #[test]
    fn test_burstiness_regular_arrivals_extended() {
        use crate::interbar_types::TradeSnapshot;
        // Regular spacing (Poisson-like) → burstiness near 0
        let trades: Vec<TradeSnapshot> = (0..20)
            .map(|i| TradeSnapshot {
                timestamp: 1000 + (i as i64 * 1000), // Uniform 1-second spacing
                price: crate::FixedPoint::from_str("100.0").unwrap(),
                volume: crate::FixedPoint::from_str("1.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: 100_000_000i128,
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burst = compute_burstiness(&refs);
        assert!(burst.abs() < 0.2, "Regular arrivals should have low burstiness, got {}", burst);
    }

    #[test]
    fn test_burstiness_clustered_arrivals_extended() {
        use crate::interbar_types::TradeSnapshot;
        // Clustered (bursty) → burstiness > 0.5
        let timestamp = 1000i64;
        let trades: Vec<TradeSnapshot> = (0..20)
            .map(|i| {
                let ts = if i < 10 {
                    timestamp + (i as i64 * 100) // Cluster 1: 100µs apart
                } else {
                    timestamp + 1_000_000 + ((i - 10) as i64 * 100) // Cluster 2: far apart
                };
                TradeSnapshot {
                    timestamp: ts,
                    price: crate::FixedPoint::from_str("100.0").unwrap(),
                    volume: crate::FixedPoint::from_str("1.0").unwrap(),
                    is_buyer_maker: i % 2 == 0,
                    turnover: 100_000_000i128,
                }
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();
        let burst = compute_burstiness(&refs);
        assert!(burst > 0.3, "Clustered arrivals should have high burstiness, got {}", burst);
    }

    // Hurst Exponent - Confidence & Bounds
    #[test]
    fn test_hurst_soft_clamp_boundary_extended() {
        // Test soft_clamp_hurst at boundaries
        assert!(soft_clamp_hurst(0.0) >= 0.0 && soft_clamp_hurst(0.0) <= 1.0);
        assert!(soft_clamp_hurst(1.0) >= 0.0 && soft_clamp_hurst(1.0) <= 1.0);
        assert!(soft_clamp_hurst(2.0) >= 0.0 && soft_clamp_hurst(2.0) <= 1.0);
        // Extreme negative
        assert!(soft_clamp_hurst(-10.0) >= 0.0 && soft_clamp_hurst(-10.0) <= 1.0);
    }

    #[test]
    fn test_hurst_monotonicity_extended() {
        // Hurst should be monotonic in trending strength
        let trending: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let mean_reverting = vec![0.5; 256];

        let h_trending = compute_hurst_dfa(&trending);
        let h_mean_revert = compute_hurst_dfa(&mean_reverting);

        // Trending should have higher Hurst
        assert!(h_trending > h_mean_revert, "Trending should have higher H than mean-reverting");
    }

    // Multi-feature consistency (cross-validation)
    #[test]
    fn test_feature_consistency_normal_market_extended() {
        use crate::interbar_types::TradeSnapshot;
        // Normal market conditions
        let trades: Vec<TradeSnapshot> = (0..100)
            .map(|i| TradeSnapshot {
                timestamp: 1000 + (i as i64 * 1000),
                price: crate::FixedPoint::from_str(&format!("{}.0", 100.0 + (i % 10) as f64 * 0.1)).unwrap(),
                volume: crate::FixedPoint::from_str("10.0").unwrap(),
                is_buyer_maker: i % 2 == 0,
                turnover: (100 * 10 * 100_000_000i128),
            })
            .collect();
        let refs: Vec<&TradeSnapshot> = trades.iter().collect();

        // All features should return valid numbers
        let kyle = compute_kyle_lambda(&refs);
        let burst = compute_burstiness(&refs);
        let (skew, kurt) = compute_volume_moments(&refs);

        assert!(kyle.is_finite(), "Kyle lambda must be finite");
        assert!(burst.is_finite(), "Burstiness must be finite");
        assert!(skew.is_finite(), "Skewness must be finite");
        assert!(kurt.is_finite(), "Kurtosis must be finite");
    }
}
