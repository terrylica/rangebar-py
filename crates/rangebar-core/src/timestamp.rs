//! Universal timestamp normalization utilities
//!
//! This module provides centralized timestamp handling to ensure all aggTrade data
//! uses consistent 16-digit microsecond precision regardless of source format.

/// Universal timestamp normalization threshold
/// Values below this are treated as 13-digit milliseconds and converted to microseconds
const MICROSECOND_THRESHOLD: u64 = 10_000_000_000_000;

/// Normalize any timestamp to 16-digit microseconds
///
/// # Arguments
/// * `raw_timestamp` - Raw timestamp that could be 13-digit millis or 16-digit micros
///
/// # Returns
/// * Normalized timestamp in microseconds (16-digit precision)
///
/// # Examples
/// ```rust
/// use rangebar_core::normalize_timestamp;
///
/// // 13-digit millisecond timestamp -> 16-digit microseconds
/// assert_eq!(normalize_timestamp(1609459200000), 1609459200000000);
///
/// // Already 16-digit microseconds -> unchanged
/// assert_eq!(normalize_timestamp(1609459200000000), 1609459200000000);
/// ```
pub fn normalize_timestamp(raw_timestamp: u64) -> i64 {
    if raw_timestamp < MICROSECOND_THRESHOLD {
        // 13-digit milliseconds -> convert to microseconds
        (raw_timestamp * 1_000) as i64
    } else {
        // Already microseconds (16+ digits)
        raw_timestamp as i64
    }
}

/// Validate timestamp is in expected microsecond range
///
/// Checks if timestamp falls within reasonable bounds for financial data.
/// Expanded range (2000-2035) covers historical Forex data (2003+)
/// and cryptocurrency data (2009+) while rejecting obviously invalid timestamps.
///
/// # Arguments
///
/// * `timestamp` - Timestamp in microseconds (16-digit precision)
///
/// # Returns
///
/// `true` if timestamp is within valid range, `false` otherwise
///
/// # Validation Range (Q16)
///
/// - MIN: 2000-01-01 (covers historical Forex from 2003)
/// - MAX: 2035-01-01 (future-proof for upcoming data)
/// - Rejects: Unix epoch (1970), far future (2100+), negative timestamps
pub fn validate_timestamp(timestamp: i64) -> bool {
    // Expanded bounds: 2000-01-01 to 2035-01-01 in microseconds (Q16)
    const MIN_TIMESTAMP: i64 = 946_684_800_000_000; // 2000-01-01 00:00:00 UTC
    const MAX_TIMESTAMP: i64 = 2_051_222_400_000_000; // 2035-01-01 00:00:00 UTC

    (MIN_TIMESTAMP..=MAX_TIMESTAMP).contains(&timestamp)
}

/// Create a normalized AggTrade with automatic timestamp conversion
///
/// This is the preferred way to create AggTrade instances to ensure
/// timestamp consistency across all data sources.
pub fn create_aggtrade_with_normalized_timestamp(
    agg_trade_id: i64,
    price: crate::FixedPoint,
    volume: crate::FixedPoint,
    first_trade_id: i64,
    last_trade_id: i64,
    raw_timestamp: u64,
    is_buyer_maker: bool,
) -> crate::types::AggTrade {
    use crate::types::AggTrade;

    AggTrade {
        agg_trade_id,
        price,
        volume,
        first_trade_id,
        last_trade_id,
        timestamp: normalize_timestamp(raw_timestamp),
        is_buyer_maker,
        is_best_match: None, // Not provided in this context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_13_digit_milliseconds() {
        // Common 13-digit timestamp (Jan 1, 2021 00:00:00 UTC)
        let millis = 1609459200000u64;
        let expected = 1609459200000000i64;
        assert_eq!(normalize_timestamp(millis), expected);
    }

    #[test]
    fn test_normalize_16_digit_microseconds() {
        // Already 16-digit microseconds
        let micros = 1609459200000000u64;
        let expected = 1609459200000000i64;
        assert_eq!(normalize_timestamp(micros), expected);
    }

    #[test]
    fn test_threshold_boundary() {
        // Right at the threshold
        let threshold_minus_one = MICROSECOND_THRESHOLD - 1;
        let threshold = MICROSECOND_THRESHOLD;

        // Below threshold: convert
        assert_eq!(
            normalize_timestamp(threshold_minus_one),
            (threshold_minus_one * 1000) as i64
        );

        // At threshold: no conversion
        assert_eq!(normalize_timestamp(threshold), threshold as i64);
    }

    #[test]
    fn test_validate_timestamp() {
        // Valid: 2024 timestamp (crypto era)
        assert!(validate_timestamp(1_704_067_200_000_000)); // 2024-01-01

        // Valid: 2003 timestamp (Forex historical data)
        assert!(validate_timestamp(1_041_379_200_000_000)); // 2003-01-01

        // Valid: 2000 timestamp (min boundary)
        assert!(validate_timestamp(946_684_800_000_000)); // 2000-01-01

        // Valid: 2034 timestamp (near max boundary)
        assert!(validate_timestamp(2_019_686_400_000_000)); // 2034-01-01

        // Invalid: 1999 (before historical Forex data)
        assert!(!validate_timestamp(915_148_800_000_000)); // 1999-01-01

        // Invalid: Unix epoch era (1970s)
        assert!(!validate_timestamp(1_000_000_000_000)); // 1970-01-12

        // Invalid: Far future (2050+)
        assert!(!validate_timestamp(2_524_608_000_000_000)); // 2050-01-01
    }
}
