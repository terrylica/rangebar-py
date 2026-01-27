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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

        // Parse fractional part (if exists)
        let fractional_part = if parts.len() == 2 {
            let frac_str = parts[1];
            if frac_str.len() > 8 {
                return Err(FixedPointError::TooManyDecimals);
            }

            // Pad with zeros to get exactly 8 decimals
            let padded = format!("{:0<8}", frac_str);
            padded
                .parse::<i64>()
                .map_err(|_| FixedPointError::InvalidFormat)?
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

    /// Convert to f64 for user-friendly output
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
}
