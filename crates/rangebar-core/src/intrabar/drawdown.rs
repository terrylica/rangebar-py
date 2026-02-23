//! Max drawdown and max runup computation for TMAEG calculation.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! ORIGIN: trading-fitness/packages/metrics-rust/src/ith_rolling.rs:79-142
//! COPIED: 2026-02-02
//! MODIFICATIONS: Extracted as standalone functions for intra-bar use
//!
//! These functions compute the Maximum Drawdown and Maximum Runup which are used
//! as the TMAEG (Target Maximum Acceptable Excess Gain) threshold for ITH analysis.
//!
//! # Key Design Decision: TMAEG = Max Drawdown / Max Runup
//!
//! This approach is elegant because:
//! 1. TMAEG is derived directly from the window's own extremes
//! 2. No arbitrary parameters or percentile tuning
//! 3. Epochs trigger when gains exceed the maximum adverse movement
//! 4. Mathematically symmetric: drawdown ↔ runup

/// Compute Maximum Drawdown for Bull ITH TMAEG.
///
/// Maximum Drawdown = 1 - (trough / peak)
///
/// This is the simplest, most mathematically pure definition:
/// - An epoch triggers when excess gain exceeds the maximum adverse movement
/// - No arbitrary parameters, no percentile tuning
/// - TMAEG is derived directly from the window's own extremes
///
/// # Arguments
/// * `window` - Normalized price window (first value = 1.0 recommended)
///
/// # Returns
/// Maximum drawdown as a fraction [0, 1). Returns `f64::EPSILON` for windows < 2.
pub fn compute_max_drawdown(window: &[f64]) -> f64 {
    if window.len() < 2 {
        return f64::EPSILON;
    }

    let mut running_max = window[0];
    let mut max_drawdown = 0.0;

    for &val in window.iter().skip(1) {
        if val > running_max {
            running_max = val;
        }
        if running_max > 0.0 && val.is_finite() {
            let drawdown = 1.0 - val / running_max;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
    }

    // Ensure a minimum threshold to avoid division issues
    max_drawdown.max(f64::EPSILON)
}

/// Compute both max drawdown and max runup in a single pass (Issue #96 Task #66).
///
/// This combined function eliminates a redundant pass through the window
/// by computing both extrema simultaneously.
pub fn compute_max_drawdown_and_runup(window: &[f64]) -> (f64, f64) {
    if window.len() < 2 {
        return (f64::EPSILON, f64::EPSILON);
    }

    let mut running_max = window[0];
    let mut running_min = window[0];
    let mut max_drawdown = 0.0;
    let mut max_runup = 0.0;

    for &val in window.iter().skip(1) {
        // Update extrema
        if val > running_max {
            running_max = val;
        }
        if val < running_min {
            running_min = val;
        }

        // Compute drawdown (adverse move for longs)
        if running_max > 0.0 && val.is_finite() {
            let drawdown = 1.0 - val / running_max;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Compute runup (adverse move for shorts)
        if val > 0.0 && running_min > 0.0 && val.is_finite() {
            let runup = 1.0 - running_min / val;
            if runup > max_runup {
                max_runup = runup;
            }
        }
    }

    (
        max_drawdown.max(f64::EPSILON),
        max_runup.max(f64::EPSILON),
    )
}

/// Compute Maximum Runup for Bear ITH TMAEG.
///
/// Maximum Runup = 1 - (trough / peak) where we track the inverse:
/// - Running minimum (trough)
/// - Then measure how much price rises from that trough
///
/// This is the symmetric counterpart to Maximum Drawdown:
/// - Drawdown: how much price falls from peak (adverse for longs)
/// - Runup: how much price rises from trough (adverse for shorts)
///
/// # Arguments
/// * `window` - Normalized price window (first value = 1.0 recommended)
///
/// # Returns
/// Maximum runup as a fraction [0, 1). Returns `f64::EPSILON` for windows < 2.
pub fn compute_max_runup(window: &[f64]) -> f64 {
    if window.len() < 2 {
        return f64::EPSILON;
    }

    let mut running_min = window[0];
    let mut max_runup = 0.0;

    for &val in window.iter().skip(1) {
        if val < running_min {
            running_min = val;
        }
        if val > 0.0 && running_min > 0.0 && val.is_finite() {
            // Runup = how much price has risen from the trough
            // Formula: 1 - (trough / current) = (current - trough) / current
            let runup = 1.0 - running_min / val;
            if runup > max_runup {
                max_runup = runup;
            }
        }
    }

    // Ensure a minimum threshold to avoid division issues
    max_runup.max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown_uptrend() {
        // Pure uptrend should have zero drawdown
        let prices = vec![1.0, 1.01, 1.02, 1.03, 1.04, 1.05];
        let dd = compute_max_drawdown(&prices);
        assert!(dd < 0.001, "Pure uptrend should have near-zero drawdown");
    }

    #[test]
    fn test_max_drawdown_downtrend() {
        // Decline from 1.0 to 0.8 = 20% drawdown
        let prices = vec![1.0, 0.95, 0.9, 0.85, 0.8];
        let dd = compute_max_drawdown(&prices);
        assert!((dd - 0.2).abs() < 0.01, "Expected 20% drawdown, got {}", dd);
    }

    #[test]
    fn test_max_drawdown_recovery() {
        // Peak at 1.1, trough at 0.9, max DD = 1 - 0.9/1.1 ≈ 18.2%
        let prices = vec![1.0, 1.1, 1.0, 0.9, 1.0, 1.1];
        let dd = compute_max_drawdown(&prices);
        assert!(
            (dd - 0.182).abs() < 0.01,
            "Expected ~18.2% drawdown, got {}",
            dd
        );
    }

    #[test]
    fn test_max_runup_downtrend() {
        // Pure downtrend should have zero runup
        let prices = vec![1.0, 0.99, 0.98, 0.97, 0.96, 0.95];
        let ru = compute_max_runup(&prices);
        assert!(ru < 0.001, "Pure downtrend should have near-zero runup");
    }

    #[test]
    fn test_max_runup_uptrend() {
        // Rise from 1.0 to 1.25 = 1 - 1.0/1.25 = 20% runup
        let prices = vec![1.0, 1.05, 1.1, 1.15, 1.2, 1.25];
        let ru = compute_max_runup(&prices);
        assert!((ru - 0.2).abs() < 0.01, "Expected 20% runup, got {}", ru);
    }

    #[test]
    fn test_max_runup_recovery() {
        // Trough at 0.9, peak at 1.1, max RU = 1 - 0.9/1.1 ≈ 18.2%
        let prices = vec![1.0, 0.9, 0.95, 1.0, 1.05, 1.1];
        let ru = compute_max_runup(&prices);
        assert!(
            (ru - 0.182).abs() < 0.01,
            "Expected ~18.2% runup, got {}",
            ru
        );
    }

    #[test]
    fn test_empty_window() {
        assert_eq!(compute_max_drawdown(&[]), f64::EPSILON);
        assert_eq!(compute_max_runup(&[]), f64::EPSILON);
    }

    #[test]
    fn test_single_element() {
        assert_eq!(compute_max_drawdown(&[1.0]), f64::EPSILON);
        assert_eq!(compute_max_runup(&[1.0]), f64::EPSILON);
    }
}
