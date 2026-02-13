//! Investment Time Horizon (ITH) analysis for intra-bar features.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! ORIGIN: trading-fitness/packages/metrics-rust/src/ith.rs
//! COPIED: 2026-02-02
//! MODIFICATIONS: Use crate-local types, remove serde derives
//!
//! ITH analysis evaluates price movement efficiency using TMAEG (Target Maximum
//! Acceptable Excess Gain) thresholds to count epochs where price movement
//! exceeds performance hurdles.
//!
//! This implementation is aligned with the Numba Python reference implementation
//! in `ith-python/src/ith_python/bull_ith_numba.py` and `bear_ith_numba.py`.

use super::types::{BearIthResult, BullIthResult};

// ============================================================================
// Bull ITH (Long Position Analysis)
// ============================================================================

/// Calculate Bull ITH (long position) analysis.
///
/// Bull ITH tracks excess gains (upside) and excess losses (drawdowns) for
/// a long-only strategy using a state machine with dynamic baseline tracking.
///
/// The algorithm uses:
/// - `endorsing_crest`: Confirmed HIGH we track performance FROM
/// - `candidate_crest`: Potential new HIGH (favorable for longs)
/// - `candidate_nadir`: Potential new LOW (drawdown = adverse for longs)
/// - Epoch condition: `excess_gain > excess_loss AND excess_gain > hurdle AND new_high`
///
/// # Arguments
///
/// * `nav` - Net Asset Value series (normalized prices)
/// * `tmaeg` - Target Maximum Acceptable Excess Gain threshold (e.g., 0.05 for 5%)
///
/// # Returns
///
/// `BullIthResult` containing excess gains, excess losses, epoch count, and statistics.
pub fn bull_ith(nav: &[f64], tmaeg: f64) -> BullIthResult {
    if nav.is_empty() {
        return BullIthResult {
            excess_gains: vec![],
            excess_losses: vec![],
            num_of_epochs: 0,
            epochs: vec![],
            intervals_cv: f64::NAN,
            max_drawdown: 0.0,
        };
    }

    let n = nav.len();
    let mut excess_gains = vec![0.0; n];
    let mut excess_losses = vec![0.0; n];
    let mut epochs = vec![false; n];

    // State machine variables (aligned with Numba implementation)
    let mut excess_gain = 0.0;
    let mut excess_loss = 0.0;
    let mut endorsing_crest = nav[0]; // Confirmed HIGH we track performance FROM
    let mut endorsing_nadir = nav[0]; // Lowest point since endorsing_crest
    let mut candidate_crest = nav[0]; // Potential new HIGH
    let mut candidate_nadir = nav[0]; // Potential new LOW (drawdown)

    // For max_drawdown calculation (separate from state machine)
    let mut running_max = nav[0];
    let mut max_drawdown = 0.0;

    for i in 1..n {
        let equity = nav[i - 1];
        let next_equity = nav[i];

        // Track running max for max_drawdown (independent calculation)
        if next_equity > running_max {
            running_max = next_equity;
        }
        let current_drawdown = if running_max > 0.0 {
            1.0 - next_equity / running_max
        } else {
            0.0
        };
        if current_drawdown > max_drawdown {
            max_drawdown = current_drawdown;
        }

        // Track new HIGHS (favorable for longs)
        if next_equity > candidate_crest {
            if endorsing_crest != 0.0 && next_equity != 0.0 {
                // Excess gain = profit from price rally
                excess_gain = next_equity / endorsing_crest - 1.0;
            } else {
                excess_gain = 0.0;
            }
            candidate_crest = next_equity;
        }

        // Track new LOWS (drawdown = adverse for longs)
        if next_equity < candidate_nadir {
            // Excess loss = drawdown hurts longs
            if endorsing_crest != 0.0 {
                excess_loss = 1.0 - next_equity / endorsing_crest;
            } else {
                excess_loss = 0.0;
            }
            candidate_nadir = next_equity;
        }

        // Reset condition: gains exceed losses, exceed hurdle, AND new high
        let reset_condition = excess_gain > excess_loss.abs()
            && excess_gain > tmaeg
            && candidate_crest >= endorsing_crest;

        if reset_condition {
            endorsing_crest = candidate_crest;
            endorsing_nadir = equity;
            candidate_nadir = equity;
        } else {
            endorsing_nadir = endorsing_nadir.min(equity);
        }

        excess_gains[i] = excess_gain;
        excess_losses[i] = excess_loss;

        if reset_condition {
            excess_gain = 0.0;
            excess_loss = 0.0;
        }

        // Check bull epoch condition
        let bull_epoch_condition = excess_gains[i] > excess_losses[i] && excess_gains[i] > tmaeg;
        epochs[i] = bull_epoch_condition;
    }

    // Count epochs (sum of True values, matching Numba)
    let num_of_epochs = epochs.iter().filter(|&&e| e).count();

    // Calculate coefficient of variation of epoch intervals (matching Numba)
    let intervals_cv = calculate_intervals_cv_numba_style(&epochs, num_of_epochs);

    BullIthResult {
        excess_gains,
        excess_losses,
        num_of_epochs,
        epochs,
        intervals_cv,
        max_drawdown,
    }
}

// ============================================================================
// Bear ITH (Short Position Analysis)
// ============================================================================

/// Calculate Bear ITH (short position) analysis.
///
/// Bear ITH is the INVERSE of Bull ITH for short positions:
/// - `endorsing_trough`: Confirmed LOW we track performance FROM (short entry)
/// - `candidate_trough`: Potential new LOW (favorable for shorts)
/// - `candidate_peak`: Potential new HIGH (runup = adverse for shorts)
/// - Epoch condition: `excess_gain > excess_loss AND excess_gain > hurdle AND new_low`
///
/// # Arguments
///
/// * `nav` - Net Asset Value series (normalized prices)
/// * `tmaeg` - Target Maximum Acceptable Excess Gain threshold (e.g., 0.05 for 5%)
///
/// # Returns
///
/// `BearIthResult` containing excess gains, excess losses, epoch count, and statistics.
pub fn bear_ith(nav: &[f64], tmaeg: f64) -> BearIthResult {
    if nav.is_empty() {
        return BearIthResult {
            excess_gains: vec![],
            excess_losses: vec![],
            num_of_epochs: 0,
            epochs: vec![],
            intervals_cv: f64::NAN,
            max_runup: 0.0,
        };
    }

    let n = nav.len();
    let mut excess_gains = vec![0.0; n];
    let mut excess_losses = vec![0.0; n];
    let mut epochs = vec![false; n];

    // State machine variables (INVERTED from Bull, aligned with Numba)
    let mut excess_gain = 0.0;
    let mut excess_loss = 0.0;
    let mut endorsing_trough = nav[0]; // Confirmed LOW we track performance FROM
    let mut endorsing_peak = nav[0]; // Highest point since endorsing_trough
    let mut candidate_trough = nav[0]; // Potential new LOW (favorable for shorts)
    let mut candidate_peak = nav[0]; // Potential new HIGH (runup = adverse)

    // For max_runup calculation (separate from state machine)
    let mut running_min = nav[0];
    let mut max_runup = 0.0;

    for i in 1..n {
        let equity = nav[i - 1];
        let next_equity = nav[i];

        // Track running min for max_runup (independent calculation)
        if next_equity < running_min {
            running_min = next_equity;
        }
        let current_runup = if next_equity > 0.0 {
            1.0 - running_min / next_equity
        } else {
            0.0
        };
        if current_runup > max_runup {
            max_runup = current_runup;
        }

        // INVERTED: Track new LOWS (favorable for shorts)
        if next_equity < candidate_trough {
            if endorsing_trough != 0.0 && next_equity != 0.0 {
                // Excess gain = profit from price decline
                // SYMMETRIC with bull: (trough/new) - 1
                excess_gain = endorsing_trough / next_equity - 1.0;
            } else {
                excess_gain = 0.0;
            }
            candidate_trough = next_equity;
        }

        // INVERTED: Track new HIGHS (runup = adverse for shorts)
        if next_equity > candidate_peak {
            // Excess loss = runup hurts shorts
            // SYMMETRIC with bull: 1 - (trough/new)
            if next_equity != 0.0 {
                excess_loss = 1.0 - endorsing_trough / next_equity;
            } else {
                excess_loss = 0.0;
            }
            candidate_peak = next_equity;
        }

        // INVERTED reset condition: gains exceed losses, exceed hurdle, AND new low
        let reset_condition = excess_gain > excess_loss.abs()
            && excess_gain > tmaeg
            && candidate_trough <= endorsing_trough;

        if reset_condition {
            endorsing_trough = candidate_trough;
            endorsing_peak = equity;
            candidate_peak = equity;
        } else {
            endorsing_peak = endorsing_peak.max(equity);
        }

        excess_gains[i] = excess_gain;
        excess_losses[i] = excess_loss;

        if reset_condition {
            excess_gain = 0.0;
            excess_loss = 0.0;
        }

        // Check bear epoch condition
        let bear_epoch_condition = excess_gains[i] > excess_losses[i] && excess_gains[i] > tmaeg;
        epochs[i] = bear_epoch_condition;
    }

    // Count epochs (sum of True values, matching Numba)
    let num_of_epochs = epochs.iter().filter(|&&e| e).count();

    // Calculate coefficient of variation of epoch intervals (matching Numba)
    let intervals_cv = calculate_intervals_cv_numba_style(&epochs, num_of_epochs);

    BearIthResult {
        excess_gains,
        excess_losses,
        num_of_epochs,
        epochs,
        intervals_cv,
        max_runup,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate coefficient of variation of epoch intervals (Numba-aligned).
///
/// This matches the Numba implementation exactly:
/// 1. Create epoch_indices array starting with 0
/// 2. Append indices where epochs[i] = true
/// 3. Calculate intervals using np.diff
/// 4. Return std(intervals) / mean(intervals)
fn calculate_intervals_cv_numba_style(epochs: &[bool], num_of_epochs: usize) -> f64 {
    if num_of_epochs == 0 {
        return f64::NAN;
    }

    // Build epoch_indices: [0, epoch_idx1, epoch_idx2, ...]
    // This matches the Numba code exactly
    let mut epoch_indices = Vec::with_capacity(num_of_epochs + 1);
    epoch_indices.push(0); // Always start with 0

    for (i, &is_epoch) in epochs.iter().enumerate() {
        if is_epoch {
            epoch_indices.push(i);
        }
    }

    // Calculate intervals (np.diff equivalent)
    // epoch_indices[: num_of_epochs + 1] gives us [0, idx1, idx2, ..., idx_n]
    // np.diff gives us [idx1-0, idx2-idx1, ..., idx_n - idx_{n-1}]
    let intervals: Vec<f64> = epoch_indices
        .windows(2)
        .take(num_of_epochs) // Match Numba: epoch_indices[: num_of_epochs + 1]
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    if intervals.is_empty() {
        return f64::NAN;
    }

    let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean <= 0.0 {
        return f64::NAN;
    }

    // Calculate std (population std, matching numpy default)
    let variance: f64 =
        intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;

    let std_dev = variance.sqrt();

    std_dev / mean
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bull ITH tests
    #[test]
    fn test_bull_ith_empty() {
        let result = bull_ith(&[], 0.05);
        assert_eq!(result.num_of_epochs, 0);
        assert!(result.intervals_cv.is_nan());
    }

    #[test]
    fn test_bull_ith_no_epochs() {
        // Flat or declining NAV should have no epochs
        let nav = vec![1.0, 0.99, 0.98, 0.97, 0.96];
        let result = bull_ith(&nav, 0.05);
        assert_eq!(result.num_of_epochs, 0);
    }

    #[test]
    fn test_bull_ith_with_epochs() {
        // Rising NAV should have epochs
        let nav = vec![1.0, 1.02, 1.04, 1.06, 1.08, 1.10];
        let result = bull_ith(&nav, 0.05);
        assert!(result.num_of_epochs > 0);
    }

    #[test]
    fn test_bull_ith_max_drawdown() {
        let nav = vec![1.0, 1.10, 1.05, 1.15, 1.00];
        let result = bull_ith(&nav, 0.05);
        // Max drawdown from 1.15 to 1.00 = 13%
        assert!(result.max_drawdown > 0.10);
    }

    #[test]
    fn test_bull_ith_state_machine() {
        // Test the state machine resets correctly
        // NAV goes up 6% (triggers epoch), then drops, then up again
        let nav = vec![1.0, 1.06, 1.03, 1.09];
        let result = bull_ith(&nav, 0.05);
        // At index 1: +6% gain > 5% hurdle, epoch = true
        assert!(result.epochs[1]);
        // After reset, baseline is now 1.06, so at 1.09: gain = 1.09/1.06 - 1 ≈ 2.8%
        // which is < 5%, so no epoch
        assert!(!result.epochs[3]);
    }

    // Bear ITH tests
    #[test]
    fn test_bear_ith_empty() {
        let result = bear_ith(&[], 0.05);
        assert_eq!(result.num_of_epochs, 0);
        assert!(result.intervals_cv.is_nan());
    }

    #[test]
    fn test_bear_ith_no_epochs() {
        // Rising NAV should have no bear epochs
        let nav = vec![1.0, 1.01, 1.02, 1.03, 1.04];
        let result = bear_ith(&nav, 0.05);
        assert_eq!(result.num_of_epochs, 0);
    }

    #[test]
    fn test_bear_ith_with_epochs() {
        // Falling NAV should have bear epochs
        let nav = vec![1.0, 0.98, 0.96, 0.94, 0.92, 0.90];
        let result = bear_ith(&nav, 0.05);
        assert!(result.num_of_epochs > 0);
    }

    #[test]
    fn test_bear_ith_max_runup() {
        let nav = vec![1.0, 0.90, 0.95, 0.85, 1.00];
        let result = bear_ith(&nav, 0.05);
        // Max runup from 0.85 to 1.00 = 17.6%
        assert!(result.max_runup > 0.15);
    }

    #[test]
    fn test_bear_ith_state_machine() {
        // Test the bear state machine resets correctly
        // NAV drops 6% (triggers epoch), then up, then down again
        let nav = vec![1.0, 0.94, 0.97, 0.91];
        let result = bear_ith(&nav, 0.05);
        // At index 1: endorsing_trough/next - 1 = 1.0/0.94 - 1 ≈ 6.4% > 5%
        assert!(result.epochs[1]);
    }

    // Helper function tests
    #[test]
    fn test_intervals_cv_numba_style() {
        // epochs = [false, true, false, true] -> epoch_indices = [0, 1, 3]
        // intervals = [1-0, 3-1] = [1, 2]
        // mean = 1.5, std = 0.5, cv = 0.5/1.5 = 0.333...
        let epochs = vec![false, true, false, true];
        let num_epochs = 2;
        let cv = calculate_intervals_cv_numba_style(&epochs, num_epochs);
        assert!((cv - 0.3333).abs() < 0.01);
    }

    #[test]
    fn test_intervals_cv_no_epochs() {
        let epochs = vec![false, false, false];
        let cv = calculate_intervals_cv_numba_style(&epochs, 0);
        assert!(cv.is_nan());
    }

    #[test]
    fn test_intervals_cv_equal_spacing() {
        // epochs at indices 10, 20, 30 -> epoch_indices = [0, 10, 20, 30]
        // intervals = [10, 10, 10], cv = 0
        let mut epochs = vec![false; 31];
        epochs[10] = true;
        epochs[20] = true;
        epochs[30] = true;
        let cv = calculate_intervals_cv_numba_style(&epochs, 3);
        assert!((cv - 0.0).abs() < 0.001);
    }
}
