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
//! let hurst = rssimple(prices);
//! assert!(hurst >= 0.0 && hurst <= 1.0);
//! ```

use linreg::linear_regression;

pub mod utils;

use utils::*;

/// Simple R/S Hurst estimation
///
/// Computes the Hurst exponent using basic Rescaled Range analysis.
///
/// # Arguments
///
/// * `x` - Input time series data (prices or returns)
///
/// # Returns
///
/// Hurst exponent value, typically in range [0, 1]
pub fn rssimple(x: &[f64]) -> f64 {
    let n: f64 = x.len() as f64;
    let x_mean: f64 = mean(x);
    let y: Vec<f64> = x
        .iter()
        .map(|x| x - x_mean)
        .collect();
    let s: Vec<f64> = cumsum(&y);
    let (min, max) = minmax(&s);
    let rs: f64 = (max - min) / standard_deviation(x);
    rs.log2() / n.log2()
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
