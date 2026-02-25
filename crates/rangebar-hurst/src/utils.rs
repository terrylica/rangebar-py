//! Utility functions for Hurst exponent calculation
//!
//! Issue #96 Task #149/150: Internal MIT-licensed utilities for R/S analysis

/// Calculate mean of a slice (used by rs_corrected path only)
pub(crate) fn mean(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().sum();
    let n: f64 = x.len() as f64;
    sum / n
}

/// Calculate standard deviation (used by rs_corrected path only)
pub(crate) fn standard_deviation(x: &[f64]) -> f64 {
    let mean_x: f64 = mean(x);
    // Issue #96 Task #209: Replace expensive .powi(2) with direct multiplication
    // powi(2) costs ~30 CPU cycles, direct multiplication costs ~1-2 cycles
    let sum_x_minus_mean: f64 = x.iter().map(|a| {
        let diff = a - mean_x;
        diff * diff  // Faster than .powi(2)
    }).sum();
    (sum_x_minus_mean / (x.len() as f64)).sqrt()
}

/// Calculate cumulative sum (used by rs_corrected path only)
pub(crate) fn cumsum(x: &[f64]) -> Vec<f64> {
    x.iter()
        .scan(0f64, |acc, &a| {
            *acc += a;
            Some(*acc)
        })
        .collect()
}

/// Find min and max values in a slice (used by rs_corrected path only)
pub(crate) fn minmax(x: &[f64]) -> (f64, f64) {
    x.iter()
        .fold((x[0], x[0]), |acc, &x| (acc.0.min(x), acc.1.max(x)))
}

/// Define the R/S scale for a time series (used by rs_corrected path only)
pub(crate) fn rscalc(x: &[f64]) -> f64 {
    let x_mean: f64 = mean(x);
    let x_minus_mean: Vec<f64> = x.iter().map(|x| x - x_mean).collect();
    let y: Vec<f64> = cumsum(&x_minus_mean);
    let (min_y, max_y) = minmax(&y);
    let r: f64 = (max_y - min_y).abs();
    let s: f64 = standard_deviation(x);
    r / s
}

/// Half intervals of indices for multi-scale R/S analysis (used by rs_corrected path only)
pub(crate) fn half(n: &[u64], original_length: u64) -> Vec<u64> {
    let previous_step: u64 = n[1];
    let next_step: u64 = previous_step / 2;
    let length: u64 = original_length / next_step;
    let range: Vec<u64> = (0..=length).collect();
    range.iter().map(|a| a * next_step).collect()
}
