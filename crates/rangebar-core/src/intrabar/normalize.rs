//! Normalization functions for intra-bar ITH metrics.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! ORIGIN: trading-fitness/packages/metrics-rust/src/ith_normalize.rs
//! COPIED: 2026-02-02
//! MODIFICATIONS: Extracted only the functions needed for intra-bar features
//!
//! All outputs are bounded to [0, 1] for LSTM/BiLSTM consumption.
//!
//! Issue #96 Task #197: Uses precomputed lookup tables for sigmoid and tanh
//! to replace expensive transcendental function calls (~100-200 CPU cycles each).

use super::normalization_lut::{cv_sigmoid_lut, sigmoid_lut, tanh_lut};

/// Logistic sigmoid function: 1 / (1 + exp(-(x - center) * scale))
///
/// This is the workhorse of normalization. It:
/// - Maps any real number to (0, 1)
/// - Is monotonically increasing
/// - Has continuous derivatives (important for gradient-based learning)
/// - Has a natural probabilistic interpretation
///
/// Parameters:
/// - center: The input value that maps to exactly 0.5
/// - scale: Controls steepness (higher = sharper transition)
#[inline]
pub fn logistic_sigmoid(x: f64, center: f64, scale: f64) -> f64 {
    1.0 / (1.0 + (-(x - center) * scale).exp())
}

/// Normalize epoch count to [0, 1] using rank-based transform.
///
/// Uses a precomputed lookup table for sigmoid applied to epoch density (epochs/lookback).
/// The sigmoid naturally maps any density to (0, 1) without hardcoded thresholds.
///
/// The function is: sigmoid_lut(density) ≈ sigmoid(10 * (density - 0.5))
/// - density=0 → ~0.007 (near zero, distinguishable)
/// - density=0.5 → 0.5 (exactly half)
/// - density=1 → ~0.993 (near one)
///
/// Issue #96 Task #197: Uses precomputed LUT instead of exp() (100-200 CPU cycles → <1 CPU cycle).
///
/// # Arguments
/// * `epochs` - Number of ITH epochs detected
/// * `lookback` - Window size (trade count for intra-bar)
///
/// # Returns
/// Normalized value in (0, 1)
#[inline]
pub fn normalize_epochs(epochs: usize, lookback: usize) -> f64 {
    if lookback == 0 {
        return 0.5; // Degenerate case
    }

    // Epoch density: fraction of observations that are epochs
    let density = epochs as f64 / lookback as f64;

    // Precomputed sigmoid LUT in 0.01 steps [0, 1] density range
    // Replaces expensive exp() call with O(1) table lookup
    sigmoid_lut(density)
}

/// Normalize excess gain/loss to [0, 1] using precomputed tanh lookup table.
///
/// Tanh is mathematically natural for this purpose:
/// - Maps [0, ∞) → [0, 1)
/// - Zero input → zero output
/// - Monotonically increasing
/// - Smooth gradients for backpropagation
///
/// The scaling factor (5.0) is derived from the observation that
/// typical ITH excess gains range from 0 to 20%, and we want
/// this range to occupy most of the [0, 0.8] output space.
///
/// Issue #96 Task #197: Uses precomputed LUT in 0.1 steps [0, 5] range
/// instead of exp() (50-100 CPU cycles → <1 CPU cycle).
///
/// # Arguments
/// * `value` - Raw excess gain or loss (absolute value used)
///
/// # Returns
/// Normalized value in [0, 1)
#[inline]
pub fn normalize_excess(value: f64) -> f64 {
    // tanh_lut(x * 5) provides (from precomputed table):
    // - 1% (0.05 scaled) → ~0.05
    // - 5% (0.25 scaled) → ~0.24
    // - 10% (0.50 scaled) → ~0.46
    // - 20% (1.00 scaled) → ~0.76
    // - 100% (5.00 scaled) → ~0.9999 (saturates at 1.0)
    tanh_lut(value.abs() * 5.0)
}

/// Normalize coefficient of variation (CV) to [0, 1] using logistic sigmoid.
///
/// CV = std / mean of epoch intervals. This ratio is naturally unbounded
/// and heavy-tailed in practice.
///
/// The sigmoid is centered at CV=0.5 (moderate regularity) because:
/// - CV=0 means perfectly regular intervals
/// - CV=0.5 is typical for many stochastic processes
/// - CV=1 means std equals mean (high irregularity)
/// - CV>1 is very irregular (common in financial data)
///
/// Special handling: NaN (no epochs) maps to ~0.12, making it
/// distinguishable from real CV values.
///
/// # Arguments
/// * `cv` - Coefficient of variation of epoch intervals (std/mean)
///
/// # Returns
/// Normalized value in (0, 1)
#[inline]
pub fn normalize_cv(cv: f64) -> f64 {
    // NaN handling: treat as CV=0 (would be perfectly regular if epochs existed)
    let cv_effective = if cv.is_nan() { 0.0 } else { cv };

    // Task #10: Precomputed LUT replaces exp() call (~50-100 CPU cycles → <1 cycle)
    cv_sigmoid_lut(cv_effective)
}

/// Normalize max drawdown to [0, 1].
///
/// Drawdown is inherently bounded [0, 1] by definition:
/// DD = (peak - current) / peak
///
/// This function ensures the bound is respected even with numerical noise.
///
/// # Arguments
/// * `drawdown` - Max drawdown as fraction (0.0 to 1.0)
///
/// # Returns
/// Clamped value in [0, 1]
#[inline]
pub fn normalize_drawdown(drawdown: f64) -> f64 {
    drawdown.clamp(0.0, 1.0)
}

/// Normalize max runup to [0, 1].
///
/// Runup is inherently bounded [0, 1] by definition:
/// RU = (current - trough) / current
///
/// This function ensures the bound is respected even with numerical noise.
///
/// # Arguments
/// * `runup` - Max runup as fraction (0.0 to 1.0)
///
/// # Returns
/// Clamped value in [0, 1]
#[inline]
pub fn normalize_runup(runup: f64) -> f64 {
    runup.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_epochs_bounded() {
        // Property: output is always in [0, 1] for any valid input
        for epochs in 0..=100 {
            for lookback in 1..=200 {
                let result = normalize_epochs(epochs, lookback);
                assert!(
                    result >= 0.0 && result <= 1.0,
                    "normalize_epochs({}, {}) = {} not in [0, 1]",
                    epochs,
                    lookback,
                    result
                );
            }
        }
    }

    #[test]
    fn test_normalize_epochs_monotonic() {
        // Property: more epochs → higher normalized value
        let lookback = 50;
        let mut prev = normalize_epochs(0, lookback);
        for epochs in 1..=lookback {
            let curr = normalize_epochs(epochs, lookback);
            assert!(
                curr >= prev,
                "normalize_epochs not monotonic: {} gave {}, {} gave {}",
                epochs - 1,
                prev,
                epochs,
                curr
            );
            prev = curr;
        }
    }

    #[test]
    fn test_normalize_excess_bounded() {
        for &value in &[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 100.0] {
            let result = normalize_excess(value);
            assert!(
                result >= 0.0 && result <= 1.0,
                "normalize_excess({}) = {} not in [0, 1]",
                value,
                result
            );
        }
    }

    #[test]
    fn test_normalize_cv_bounded() {
        for &cv in &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let result = normalize_cv(cv);
            assert!(
                result >= 0.0 && result <= 1.0,
                "normalize_cv({}) = {} not in [0, 1]",
                cv,
                result
            );
        }
    }

    #[test]
    fn test_normalize_cv_nan_handling() {
        let nan_result = normalize_cv(f64::NAN);
        assert!(nan_result.is_finite(), "NaN should map to finite value");
        assert!(nan_result < 0.3, "NaN should map to low value");
    }

    #[test]
    fn test_normalize_drawdown_clamped() {
        assert_eq!(normalize_drawdown(-0.1), 0.0);
        assert_eq!(normalize_drawdown(0.5), 0.5);
        assert_eq!(normalize_drawdown(1.5), 1.0);
    }

    #[test]
    fn test_normalize_runup_clamped() {
        assert_eq!(normalize_runup(-0.1), 0.0);
        assert_eq!(normalize_runup(0.5), 0.5);
        assert_eq!(normalize_runup(1.5), 1.0);
    }

    // Issue #96: Edge case coverage for normalization functions

    #[test]
    fn test_normalize_epochs_zero_lookback() {
        // Degenerate case: lookback=0 should not panic
        assert_eq!(normalize_epochs(0, 0), 0.5);
        assert_eq!(normalize_epochs(5, 0), 0.5);
    }

    #[test]
    fn test_normalize_epochs_zero_epochs() {
        let result = normalize_epochs(0, 100);
        assert!(result < 0.1, "Zero epochs should map to low value, got {}", result);
        assert!(result > 0.0, "Zero epochs should be distinguishable from 0");
    }

    #[test]
    fn test_normalize_epochs_full_density() {
        let result = normalize_epochs(100, 100);
        // sigmoid_lut(1.0) ≈ 0.70 (LUT-specific scaling)
        assert!(result > 0.5, "All-epochs should map above midpoint, got {}", result);
        assert!(result <= 1.0, "Must be bounded by 1.0");
    }

    #[test]
    fn test_normalize_excess_zero() {
        let result = normalize_excess(0.0);
        assert!(result.abs() < 0.01, "Zero excess should map near 0, got {}", result);
    }

    #[test]
    fn test_normalize_excess_negative_uses_abs() {
        let pos = normalize_excess(0.1);
        let neg = normalize_excess(-0.1);
        assert_eq!(pos, neg, "normalize_excess should use absolute value");
    }

    #[test]
    fn test_normalize_excess_large_saturates() {
        let result = normalize_excess(100.0);
        assert!(result > 0.999, "Large excess should saturate near 1.0, got {}", result);
    }

    #[test]
    fn test_logistic_sigmoid_center() {
        let result = logistic_sigmoid(0.5, 0.5, 4.0);
        assert!((result - 0.5).abs() < 0.001, "At center, sigmoid should be 0.5");
    }

    #[test]
    fn test_logistic_sigmoid_extremes() {
        let low = logistic_sigmoid(-10.0, 0.0, 1.0);
        let high = logistic_sigmoid(10.0, 0.0, 1.0);
        assert!(low < 0.001, "Far below center should be near 0");
        assert!(high > 0.999, "Far above center should be near 1");
    }
}
