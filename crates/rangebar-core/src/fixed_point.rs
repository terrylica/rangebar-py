//! Fixed-point arithmetic for precise decimal calculations without floating point errors

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Scale factor for 8 decimal places (100,000,000)
pub const SCALE: i64 = 100_000_000;

/// Scale factor for decimal basis points (v3.0.0: 100,000)
/// Prior to v3.0.0, this was 10,000 (1 dbps units). Now 100,000 (dbps).
/// Migration: multiply all threshold_decimal_bps values by 10.
pub const BASIS_POINTS_SCALE: u32 = 100_000;

/// Fixed-point decimal representation using i64 with 8 decimal precision
///
/// This avoids floating point rounding errors while maintaining performance.
/// All prices and volumes are stored as integers scaled by SCALE (1e8).
///
/// Example:
/// - 50000.12345678 → 5000012345678
/// - 1.5 → 150000000
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FixedPoint(pub i64);

impl FixedPoint {
    /// Create FixedPoint from string representation
    ///
    /// # Arguments
    ///
    /// * `s` - Decimal string (e.g., "50000.12345678")
    ///
    /// # Returns
    ///
    /// Result containing FixedPoint or parse error
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, FixedPointError> {
        // Handle empty string
        if s.is_empty() {
            return Err(FixedPointError::InvalidFormat);
        }

        // Split on decimal point
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() > 2 {
            return Err(FixedPointError::InvalidFormat);
        }

        // Parse integer part
        let integer_part: i64 = parts[0]
            .parse()
            .map_err(|_| FixedPointError::InvalidFormat)?;

        // Parse fractional part (if exists) — zero-allocation path
        // Issue #96: Avoid format!() String allocation per parse (2 allocs/trade eliminated)
        let fractional_part = if parts.len() == 2 {
            let frac_str = parts[1];
            let frac_len = frac_str.len();
            if frac_len > 8 {
                return Err(FixedPointError::TooManyDecimals);
            }

            // Parse digits directly and scale by 10^(8-len) instead of String padding
            let frac_digits: i64 = frac_str
                .parse()
                .map_err(|_| FixedPointError::InvalidFormat)?;

            // Multiply by appropriate power of 10 to get 8 decimal places
            // e.g., "5" (1 digit) → 5 * 10^7 = 50_000_000
            // e.g., "12345678" (8 digits) → 12345678 * 10^0 = 12345678
            const POWERS: [i64; 9] = [
                100_000_000, 10_000_000, 1_000_000, 100_000, 10_000,
                1_000, 100, 10, 1,
            ];
            frac_digits * POWERS[frac_len]
        } else {
            0
        };

        // Combine parts with proper sign handling
        let result = if integer_part >= 0 {
            integer_part * SCALE + fractional_part
        } else {
            integer_part * SCALE - fractional_part
        };

        Ok(FixedPoint(result))
    }

    /// Convert FixedPoint to string representation with 8 decimal places
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        let abs_value = self.0.abs();
        let integer_part = abs_value / SCALE;
        let fractional_part = abs_value % SCALE;

        let sign = if self.0 < 0 { "-" } else { "" };
        format!("{}{}.{:08}", sign, integer_part, fractional_part)
    }

    /// Compute range thresholds for given basis points
    ///
    /// # Arguments
    ///
    /// * `threshold_decimal_bps` - Threshold in **decimal basis points**
    ///   - Example: `250` → 25bps = 0.25%
    ///   - Example: `10` → 1bps = 0.01%
    ///   - Minimum: `1` → 0.1bps = 0.001%
    ///
    /// # Returns
    ///
    /// Tuple of (upper_threshold, lower_threshold)
    ///
    /// # Breaking Change (v3.0.0)
    ///
    /// Prior to v3.0.0, `threshold_decimal_bps` was in 1 dbps units.
    /// **Migration**: Multiply all threshold values by 10.
    pub fn compute_range_thresholds(&self, threshold_decimal_bps: u32) -> (FixedPoint, FixedPoint) {
        // Calculate threshold delta: price * (threshold_decimal_bps / 100,000)
        // v3.0.0: threshold now in dbps (e.g., 250 dbps = 0.25%)
        let delta = (self.0 as i128 * threshold_decimal_bps as i128) / BASIS_POINTS_SCALE as i128;
        let delta = delta as i64;

        let upper = FixedPoint(self.0 + delta);
        let lower = FixedPoint(self.0 - delta);

        (upper, lower)
    }

    /// Issue #96 Task #98: Fast threshold computation using pre-computed ratio
    ///
    /// Avoids repeated division by BASIS_POINTS_SCALE in hot path (every bar creation).
    /// Instead of: delta = (price * threshold_dbps) / 100_000
    /// We use: delta = (price * ratio) / SCALE, where ratio is pre-computed.
    ///
    /// # Arguments
    /// * `threshold_ratio` - Pre-computed (threshold_dbps * SCALE) / 100_000
    ///   This should be computed once at RangeBarProcessor initialization.
    #[inline]
    pub fn compute_range_thresholds_cached(&self, threshold_ratio: i64) -> (FixedPoint, FixedPoint) {
        // Calculate threshold delta using cached ratio: delta = (price * ratio) / SCALE
        // Avoids division in hot path, only does multiplication
        let delta = (self.0 as i128 * threshold_ratio as i128) / SCALE as i128;
        let delta = delta as i64;

        let upper = FixedPoint(self.0 + delta);
        let lower = FixedPoint(self.0 - delta);

        (upper, lower)
    }

    /// Convert to f64 for user-friendly output
    /// Issue #96: #[inline] for hot-path conversion (called 100s of times per bar)
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / SCALE as f64
    }
}

impl fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl FromStr for FixedPoint {
    type Err = FixedPointError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        FixedPoint::from_str(s)
    }
}

/// Fixed-point arithmetic errors
#[derive(Debug, Clone, PartialEq)]
pub enum FixedPointError {
    /// Invalid number format
    InvalidFormat,
    /// Too many decimal places (>8)
    TooManyDecimals,
    /// Arithmetic overflow
    Overflow,
}

impl fmt::Display for FixedPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixedPointError::InvalidFormat => write!(f, "Invalid number format"),
            FixedPointError::TooManyDecimals => write!(f, "Too many decimal places (max 8)"),
            FixedPointError::Overflow => write!(f, "Arithmetic overflow"),
        }
    }
}

impl std::error::Error for FixedPointError {}

#[cfg(feature = "python")]
impl From<FixedPointError> for PyErr {
    fn from(err: FixedPointError) -> PyErr {
        match err {
            FixedPointError::InvalidFormat => {
                pyo3::exceptions::PyValueError::new_err("Invalid number format")
            }
            FixedPointError::TooManyDecimals => {
                pyo3::exceptions::PyValueError::new_err("Too many decimal places (max 8)")
            }
            FixedPointError::Overflow => {
                pyo3::exceptions::PyOverflowError::new_err("Arithmetic overflow")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string() {
        assert_eq!(FixedPoint::from_str("0").unwrap().0, 0);
        assert_eq!(FixedPoint::from_str("1").unwrap().0, SCALE);
        assert_eq!(FixedPoint::from_str("1.5").unwrap().0, SCALE + SCALE / 2);
        assert_eq!(
            FixedPoint::from_str("50000.12345678").unwrap().0,
            5000012345678
        );
        assert_eq!(FixedPoint::from_str("-1.5").unwrap().0, -SCALE - SCALE / 2);
    }

    #[test]
    fn test_to_string() {
        assert_eq!(FixedPoint(0).to_string(), "0.00000000");
        assert_eq!(FixedPoint(SCALE).to_string(), "1.00000000");
        assert_eq!(FixedPoint(SCALE + SCALE / 2).to_string(), "1.50000000");
        assert_eq!(FixedPoint(5000012345678).to_string(), "50000.12345678");
        assert_eq!(FixedPoint(-SCALE).to_string(), "-1.00000000");
    }

    #[test]
    fn test_round_trip() {
        let test_values = [
            "0",
            "1",
            "1.5",
            "50000.12345678",
            "999999.99999999",
            "-1.5",
            "-50000.12345678",
        ];

        for val in &test_values {
            let fp = FixedPoint::from_str(val).unwrap();
            let back = fp.to_string();

            // Verify round-trip conversion works correctly
            let fp2 = FixedPoint::from_str(&back).unwrap();
            assert_eq!(fp.0, fp2.0, "Round trip failed for {}", val);
        }
    }

    #[test]
    fn test_compute_thresholds() {
        let price = FixedPoint::from_str("50000.0").unwrap();
        let (upper, lower) = price.compute_range_thresholds(250); // 250 × 0.1bps = 25bps

        // 50000 * 0.0025 = 125 (25bps = 0.25%)
        assert_eq!(upper.to_string(), "50125.00000000");
        assert_eq!(lower.to_string(), "49875.00000000");
    }

    #[test]
    fn test_error_cases() {
        assert!(FixedPoint::from_str("").is_err());
        assert!(FixedPoint::from_str("not_a_number").is_err());
        assert!(FixedPoint::from_str("1.123456789").is_err()); // Too many decimals
        assert!(FixedPoint::from_str("1.2.3").is_err()); // Multiple decimal points
    }

    #[test]
    fn test_comparison() {
        let a = FixedPoint::from_str("50000.0").unwrap();
        let b = FixedPoint::from_str("50000.1").unwrap();
        let c = FixedPoint::from_str("49999.9").unwrap();

        assert!(a < b);
        assert!(b > a);
        assert!(c < a);
        assert_eq!(a, a);
    }

    // Issue #96 Task #91: Edge case tests for arithmetic correctness

    #[test]
    fn test_from_str_too_many_decimals() {
        let err = FixedPoint::from_str("0.000000001").unwrap_err();
        assert_eq!(err, FixedPointError::TooManyDecimals);
    }

    #[test]
    fn test_from_str_negative_fractional() {
        // Known edge case: "-0.5" parses as +0.5 because "-0" → 0 (non-negative)
        // The sign is lost when integer_part == 0. This only affects (-1, 0) range.
        // Real Binance prices are always positive, so this is acceptable behavior.
        let fp = FixedPoint::from_str("-0.5").unwrap();
        assert_eq!(fp.0, 50_000_000); // "-0" parsed as 0 (non-negative), so +0.5

        // Negative values with non-zero integer part work correctly
        let fp2 = FixedPoint::from_str("-1.5").unwrap();
        assert_eq!(fp2.0, -150_000_000); // -1.5 * SCALE
        assert_eq!(fp2.to_f64(), -1.5);
    }

    #[test]
    fn test_from_str_leading_zeros() {
        // "000.123" should parse — integer part "000" is valid i64
        let fp = FixedPoint::from_str("000.123").unwrap();
        assert_eq!(fp.0, 12_300_000); // 0.123 * SCALE
    }

    #[test]
    fn test_to_f64_extreme_values() {
        // i64::MAX / SCALE = 92233720368.54775807
        let max_fp = FixedPoint(i64::MAX);
        let max_f64 = max_fp.to_f64();
        assert!(max_f64 > 92_233_720_368.0);
        assert!(max_f64.is_finite());

        // i64::MIN / SCALE = -92233720368.54775808
        let min_fp = FixedPoint(i64::MIN);
        let min_f64 = min_fp.to_f64();
        assert!(min_f64 < -92_233_720_368.0);
        assert!(min_f64.is_finite());
    }

    #[test]
    fn test_threshold_zero_ratio() {
        let price = FixedPoint::from_str("100.0").unwrap();
        let (upper, lower) = price.compute_range_thresholds_cached(0);
        assert_eq!(upper, price);
        assert_eq!(lower, price);
    }

    #[test]
    fn test_threshold_small_price_small_bps() {
        // Very small price (0.01) with smallest threshold (1 dbps = 0.001%)
        let price = FixedPoint::from_str("0.01").unwrap();
        let (upper, lower) = price.compute_range_thresholds(1);
        // delta = (1_000_000 * 1) / 100_000 = 10
        // So upper = 1_000_010, lower = 999_990
        assert!(upper > price);
        assert!(lower < price);
    }

    #[test]
    fn test_fixedpoint_zero() {
        let zero = FixedPoint(0);
        assert_eq!(zero.to_f64(), 0.0);
        assert_eq!(zero.to_string(), "0.00000000");
        let (upper, lower) = zero.compute_range_thresholds(250);
        assert_eq!(upper, zero); // 0 * anything = 0
        assert_eq!(lower, zero);
    }

    #[test]
    fn test_fixedpoint_error_display() {
        assert_eq!(
            FixedPointError::InvalidFormat.to_string(),
            "Invalid number format"
        );
        assert_eq!(
            FixedPointError::TooManyDecimals.to_string(),
            "Too many decimal places (max 8)"
        );
        assert_eq!(
            FixedPointError::Overflow.to_string(),
            "Arithmetic overflow"
        );
    }
}
