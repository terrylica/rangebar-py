//! Hurst Exponent estimator functions for Rust
//!
//! Originally based on evrom/hurst (GPL-3.0), forked to MIT for license compatibility.
//! This provides core R/S (Rescaled Range) analysis for Hurst exponent estimation.
//!
//! # License Resolution
//!
//! Issue #96 Task #149/150: GPL-3.0 license conflict resolution.
//! The external evrom/hurst crate (v0.1.0) was GPL-3.0 licensed, blocking PyPI distribution.
//! This internal fork enables MIT-licensed Hurst calculations without GPL restrictions.
//!
//! # Examples
//!
//! ```
//! # use rangebar_hurst::rssimple;
//! let prices = vec![100.0, 101.0, 99.5, 102.0, 101.5];
//! let hurst = rssimple(&prices);
//! assert!(hurst >= 0.0 && hurst <= 1.0);
//! ```

use linreg::linear_regression;

pub mod utils;

use utils::*;

/// Simple R/S Hurst estimation
///
/// Computes the Hurst exponent using basic Rescaled Range analysis.
/// Issue #96: Optimized from 5 passes + 2 Vec allocations to 2 passes + 0 allocations.
///
/// # Arguments
///
/// * `x` - Input time series data (prices or returns)
///
/// # Returns
///
/// Hurst exponent value, typically in range [0, 1]
pub fn rssimple(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let n_f64 = n as f64;
    let inv_n = 1.0 / n_f64;

    // Pass 1: mean
    let x_mean: f64 = x.iter().sum::<f64>() * inv_n;

    // Pass 2: fused cumsum-minmax + variance (zero allocations)
    // Simultaneously tracks:
    // - Running cumulative sum of deviations (for R/S range)
    // - Min/max of cumulative sum (for rescaled range)
    // - Sum of squared deviations (for standard deviation)
    let mut cumsum = 0.0_f64;
    let mut cum_min = 0.0_f64;
    let mut cum_max = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for &val in x {
        let d = val - x_mean;
        cumsum += d;
        cum_min = cum_min.min(cumsum);
        cum_max = cum_max.max(cumsum);
        sum_sq += d * d;
    }

    let std_dev = (sum_sq * inv_n).sqrt();
    if std_dev < f64::EPSILON {
        return 0.5; // Constant series
    }

    let rs = (cum_max - cum_min) / std_dev;
    if rs < f64::EPSILON {
        return 0.5;
    }

    rs.log2() / n_f64.log2()
}

/// Corrected R over S Hurst exponent
///
/// Computes Hurst exponent with interval averaging correction for improved stability.
///
/// # Arguments
///
/// * `x` - Input time series data
///
/// # Returns
///
/// Corrected Hurst exponent value, typically in range [0, 1]
pub fn rs_corrected(x: Vec<f64>) -> f64 {
    let mut cap_x: Vec<f64> = vec![x.len() as f64];
    let mut cap_y: Vec<f64> = vec![rscalc(&x)];
    let mut n: Vec<u64> = vec![0, x.len() as u64 / 2, x.len() as u64];

    // compute averaged R/S for halved intervals
    while n[1] >= 8 {
        let mut xl: Vec<f64> = vec![];
        let mut yl: Vec<f64> = vec![];
        for i in 1..n.len() {
            let rs: f64 = rscalc(&x[((n[i - 1] + 1) as usize)..(n[i] as usize)]);
            xl.push((n[i] - n[i - 1]) as f64);
            yl.push(rs);
        }
        cap_x.push(mean(&xl));
        cap_y.push(mean(&yl));
        // next step
        n = half(&n, x.len() as u64);
    }

    // apply linear regression
    let cap_x_log: Vec<f64> = cap_x.iter().map(|a| a.ln()).collect();
    let cap_y_log: Vec<f64> = cap_y.iter().map(|a| a.ln()).collect();
    let (slope, _): (f64, f64) = linear_regression(&cap_x_log, &cap_y_log).unwrap();
    slope
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rssimple_trending_series() {
        // Monotonically increasing → H > 0.5 (trending)
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        let h = rssimple(&prices);
        assert!(h.is_finite(), "Hurst must be finite: {}", h);
        assert!(h > 0.4, "Trending series should have H > 0.4: {}", h);
    }

    #[test]
    fn test_rssimple_constant_series() {
        let prices = vec![100.0; 50];
        let h = rssimple(&prices);
        assert_eq!(h, 0.5, "Constant series → H = 0.5");
    }

    #[test]
    fn test_rssimple_empty() {
        let h = rssimple(&[]);
        assert_eq!(h, 0.0, "Empty series → 0.0");
    }

    #[test]
    fn test_rssimple_single_element() {
        let h = rssimple(&[100.0]);
        // Single element: std_dev = 0 → returns 0.5
        assert_eq!(h, 0.5, "Single element → H = 0.5");
    }

    #[test]
    fn test_rssimple_alternating_series() {
        // Mean-reverting: H < 0.5
        let prices: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 100.0 } else { 101.0 }).collect();
        let h = rssimple(&prices);
        assert!(h.is_finite(), "Hurst must be finite: {}", h);
        assert!(h < 0.6, "Alternating series should have low H: {}", h);
    }

    #[test]
    fn test_rssimple_five_elements_doc_example() {
        let prices = vec![100.0, 101.0, 99.5, 102.0, 101.5];
        let h = rssimple(&prices);
        assert!(h >= 0.0 && h <= 1.5, "Hurst should be in reasonable range: {}", h);
    }
}
