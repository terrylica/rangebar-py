//! Tick validation and conversion to synthetic aggTrades
//!
//! Converts Exness ticks (quote data) to AggTrade format (trade data)
//! using mid-price as synthetic trade price. Preserves range bar algorithm
//! integrity while handling semantic differences between quotes and trades.

use crate::exness::types::{ConversionError, ExnessTick, ValidationStrictness};
use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::timestamp::normalize_timestamp;
use rangebar_core::types::AggTrade;

/// Validate tick data
///
/// Validation levels (configurable strictness):
/// - Permissive: Basic checks (bid > 0, ask > 0, bid < ask)
/// - Strict: + Spread < 10% (catches obvious errors) [DEFAULT]
/// - Paranoid: + Spread < 1% (flags suspicious patterns)
///
/// # Arguments
///
/// * `tick` - Exness tick to validate
/// * `strictness` - Validation level
///
/// # Returns
///
/// Ok(()) if valid, Err with specific validation failure otherwise
///
/// # Error Policy
///
/// Fail-fast: All validation errors propagated immediately to caller.
pub fn validate_tick(
    tick: &ExnessTick,
    strictness: ValidationStrictness,
) -> Result<(), ConversionError> {
    // Critical checks (all levels)
    if tick.bid <= 0.0 {
        return Err(ConversionError::InvalidBid { bid: tick.bid });
    }

    if tick.ask <= 0.0 {
        return Err(ConversionError::InvalidAsk { ask: tick.ask });
    }

    // Crossed market check: bid > ask (strictly greater)
    // Note: bid == ask is valid (zero spread, common in Raw_Spread data)
    if tick.bid > tick.ask {
        return Err(ConversionError::CrossedMarket {
            bid: tick.bid,
            ask: tick.ask,
        });
    }

    // Strictness-dependent checks
    match strictness {
        ValidationStrictness::Permissive => Ok(()),

        ValidationStrictness::Strict => {
            let spread_pct = ((tick.ask - tick.bid) / tick.bid) * 100.0;
            if spread_pct > 10.0 {
                return Err(ConversionError::ExcessiveSpread {
                    spread_pct,
                    threshold_pct: 10.0,
                });
            }
            Ok(())
        }

        ValidationStrictness::Paranoid => {
            let spread_pct = ((tick.ask - tick.bid) / tick.bid) * 100.0;
            if spread_pct > 1.0 {
                return Err(ConversionError::ExcessiveSpread {
                    spread_pct,
                    threshold_pct: 1.0,
                });
            }
            Ok(())
        }
    }
}

/// Convert Exness tick to synthetic AggTrade
///
/// Mid-price conversion (academic standard):
/// - price = (bid + ask) / 2.0
/// - volume = 0 (Exness Raw_Spread has no volume data)
/// - is_buyer_maker = false (direction unknown for quotes)
///
/// # Arguments
///
/// * `tick` - Exness tick (quote data)
/// * `_instrument` - Instrument symbol (unused, kept for API compatibility)
/// * `id` - Synthetic aggTrade ID
/// * `strictness` - Validation level
///
/// # Returns
///
/// AggTrade with mid-price as trade price, or validation error
///
/// # Error Handling
///
/// Raises errors immediately (no fallbacks, no defaults).
/// All validation failures propagated to caller.
pub fn tick_to_synthetic_trade(
    tick: &ExnessTick,
    _instrument: &str,
    id: i64,
    strictness: ValidationStrictness,
) -> Result<AggTrade, ConversionError> {
    // 1. Validate tick (raise on error)
    validate_tick(tick, strictness)?;

    // 2. Calculate mid-price (market consensus price)
    let mid_price = (tick.ask + tick.bid) / 2.0;

    // 3. Volume is always 0 (Exness Raw_Spread has no volume data)
    let total_volume = 0.0;

    // 4. Direction UNKNOWN for quote data (no buy/sell inference)
    let is_buyer_maker = false; // Arbitrary default (not used for segregation)

    // 5. Construct synthetic AggTrade
    // Format f64 with 8 decimals to match FixedPoint precision
    let price_str = format!("{:.8}", mid_price);
    let volume_str = format!("{:.8}", total_volume);

    Ok(AggTrade {
        agg_trade_id: id,
        price: FixedPoint::from_str(&price_str).map_err(|e| {
            ConversionError::FixedPointConversion {
                value: price_str.clone(),
                error: format!("{:?}", e),
            }
        })?,
        volume: FixedPoint::from_str(&volume_str).map_err(|e| {
            ConversionError::FixedPointConversion {
                value: volume_str.clone(),
                error: format!("{:?}", e),
            }
        })?,
        first_trade_id: id,
        last_trade_id: id,
        timestamp: normalize_timestamp(tick.timestamp_ms as u64),
        is_buyer_maker,
        is_best_match: None, // N/A for Exness
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tick_crossed_market() {
        let tick = ExnessTick {
            bid: 1.0815,
            ask: 1.0800, // bid > ask (crossed)
            timestamp_ms: 1000000,
        };

        let result = validate_tick(&tick, ValidationStrictness::Permissive);
        assert!(result.is_err());
        match result {
            Err(ConversionError::CrossedMarket { bid, ask }) => {
                assert_eq!(bid, 1.0815);
                assert_eq!(ask, 1.0800);
            }
            _ => panic!("Expected CrossedMarket error"),
        }
    }

    #[test]
    fn test_validate_tick_zero_spread() {
        // bid == ask is VALID (zero spread, common in Raw_Spread data)
        let tick = ExnessTick {
            bid: 1.0800,
            ask: 1.0800, // Zero spread
            timestamp_ms: 1000000,
        };

        let result = validate_tick(&tick, ValidationStrictness::Permissive);
        assert!(result.is_ok(), "Zero spread (bid==ask) should be valid");

        // Should also pass strict validation (spread = 0%)
        let result = validate_tick(&tick, ValidationStrictness::Strict);
        assert!(result.is_ok(), "Zero spread should pass strict validation");
    }

    #[test]
    fn test_validate_tick_excessive_spread() {
        let tick = ExnessTick {
            bid: 1.0000,
            ask: 1.2000, // 20% spread
            timestamp_ms: 1000000,
        };

        // Permissive: passes
        assert!(validate_tick(&tick, ValidationStrictness::Permissive).is_ok());

        // Strict: fails (>10%)
        let result = validate_tick(&tick, ValidationStrictness::Strict);
        assert!(result.is_err());
        match result {
            Err(ConversionError::ExcessiveSpread { spread_pct, .. }) => {
                assert!((spread_pct - 20.0).abs() < 0.01);
            }
            _ => panic!("Expected ExcessiveSpread error"),
        }
    }

    #[test]
    fn test_mid_price_conversion() {
        let tick = ExnessTick {
            bid: 1.0800,
            ask: 1.0820,
            timestamp_ms: 1_600_000_000_000, // ms
        };

        let trade =
            tick_to_synthetic_trade(&tick, "EURUSD_Raw_Spread", 1, ValidationStrictness::Strict)
                .unwrap();

        // Mid-price = (1.0800 + 1.0820) / 2 = 1.0810
        let expected_price = 1.0810;
        assert!((trade.price.to_f64() - expected_price).abs() < 0.0001);

        // Volume = 0 (Exness has no volume data)
        assert_eq!(trade.volume.0, 0);

        // Direction unknown
        assert!(!trade.is_buyer_maker);

        // Timestamp normalized to microseconds
        assert_eq!(trade.timestamp, 1_600_000_000_000_000);
    }

    #[test]
    fn test_zero_volume_semantics() {
        let tick = ExnessTick {
            bid: 1.0800,
            ask: 1.0815,
            timestamp_ms: 1_600_000_000_000,
        };

        let trade =
            tick_to_synthetic_trade(&tick, "EURUSD_Raw_Spread", 1, ValidationStrictness::Strict)
                .unwrap();

        // Volume should always be 0 (Exness Raw_Spread has no volume data)
        assert_eq!(trade.volume.0, 0);
    }

    #[test]
    fn test_invalid_bid() {
        let tick = ExnessTick {
            bid: -1.0, // Invalid
            ask: 1.0815,
            timestamp_ms: 1_600_000_000_000,
        };

        let result =
            tick_to_synthetic_trade(&tick, "EURUSD_Raw_Spread", 1, ValidationStrictness::Strict);

        assert!(result.is_err());
        match result {
            Err(ConversionError::InvalidBid { bid }) => {
                assert_eq!(bid, -1.0);
            }
            _ => panic!("Expected InvalidBid error"),
        }
    }

    #[test]
    fn test_invalid_ask() {
        let tick = ExnessTick {
            bid: 1.0800,
            ask: 0.0, // Invalid
            timestamp_ms: 1_600_000_000_000,
        };

        let result =
            tick_to_synthetic_trade(&tick, "EURUSD_Raw_Spread", 1, ValidationStrictness::Strict);

        assert!(result.is_err());
        match result {
            Err(ConversionError::InvalidAsk { ask }) => {
                assert_eq!(ask, 0.0);
            }
            _ => panic!("Expected InvalidAsk error"),
        }
    }
}
