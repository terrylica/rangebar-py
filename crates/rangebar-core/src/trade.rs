//! Trade and data source types
//!
//! Extracted from types.rs (Phase 2c refactoring)

use crate::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};

/// Data source for market data (future-proofing for multi-exchange support)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum DataSource {
    /// Binance Spot Market (8 fields including is_best_match)
    BinanceSpot,
    /// Binance USD-Margined Futures (7 fields without is_best_match)
    #[default]
    BinanceFuturesUM,
    /// Binance Coin-Margined Futures
    BinanceFuturesCM,
}

/// Aggregate trade data from Binance markets
///
/// Represents a single AggTrade record which aggregates multiple individual
/// exchange trades that occurred at the same price within ~100ms timeframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AggTrade {
    /// Aggregate trade ID (unique per AggTrade record)
    pub agg_trade_id: i64,

    /// Price as fixed-point integer
    pub price: FixedPoint,

    /// Volume as fixed-point integer (total quantity across all individual trades)
    pub volume: FixedPoint,

    /// First individual trade ID in this aggregation
    pub first_trade_id: i64,

    /// Last individual trade ID in this aggregation
    pub last_trade_id: i64,

    /// Timestamp in microseconds (preserves maximum precision)
    pub timestamp: i64,

    /// Whether buyer is market maker (true = sell pressure, false = buy pressure)
    /// Critical for order flow analysis and market microstructure
    pub is_buyer_maker: bool,

    /// Whether trade was best price match (Spot market only)
    /// None for futures markets, Some(bool) for spot markets
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_best_match: Option<bool>,
}

impl AggTrade {
    /// Number of individual exchange trades in this aggregated record
    ///
    /// Each AggTrade record represents multiple individual trades that occurred
    /// at the same price within the same ~100ms window on the exchange.
    /// Issue #96: #[inline] for per-trade hot path (called in RangeBar::new + update_with_trade)
    #[inline]
    pub fn individual_trade_count(&self) -> i64 {
        self.last_trade_id - self.first_trade_id + 1
    }

    /// Turnover (price * volume) as i128 to prevent overflow
    /// Issue #96: #[inline] for per-trade hot path (called in RangeBar::new + update_with_trade)
    #[inline]
    pub fn turnover(&self) -> i128 {
        (self.price.0 as i128) * (self.volume.0 as i128)
    }
}

// Issue #96: Test coverage for AggTrade methods
#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(price: &str, volume: &str, first_id: i64, last_id: i64) -> AggTrade {
        AggTrade {
            agg_trade_id: 1,
            price: FixedPoint::from_str(price).unwrap(),
            volume: FixedPoint::from_str(volume).unwrap(),
            first_trade_id: first_id,
            last_trade_id: last_id,
            timestamp: 1000,
            is_buyer_maker: false,
            is_best_match: None,
        }
    }

    #[test]
    fn test_individual_trade_count_single() {
        let trade = make_trade("100.0", "1.0", 5, 5);
        assert_eq!(trade.individual_trade_count(), 1);
    }

    #[test]
    fn test_individual_trade_count_multiple() {
        let trade = make_trade("100.0", "1.0", 100, 199);
        assert_eq!(trade.individual_trade_count(), 100);
    }

    #[test]
    fn test_individual_trade_count_large_range() {
        let trade = make_trade("100.0", "1.0", 0, 999_999);
        assert_eq!(trade.individual_trade_count(), 1_000_000);
    }

    #[test]
    fn test_turnover_basic() {
        // price=100.0 (FixedPoint=10_000_000_000), volume=2.0 (FixedPoint=200_000_000)
        let trade = make_trade("100.0", "2.0", 1, 1);
        let expected = 10_000_000_000i128 * 200_000_000i128;
        assert_eq!(trade.turnover(), expected);
    }

    #[test]
    fn test_turnover_zero_volume() {
        let trade = make_trade("100.0", "0.0", 1, 1);
        assert_eq!(trade.turnover(), 0);
    }

    #[test]
    fn test_turnover_large_values_no_overflow() {
        // Simulate high-volume token: price * volume would overflow i64 but fits i128
        // SHIBUSDT: price=0.00002, volume=10_000_000_000
        let trade = make_trade("0.00002", "10000000000.0", 1, 1);
        let turnover = trade.turnover();
        assert!(turnover > 0, "Turnover should be positive for valid trade");
        // Verify it's computable without panic
        let _as_f64 = turnover as f64;
    }

    #[test]
    fn test_turnover_tiny_price() {
        let trade = make_trade("0.00000001", "1.0", 1, 1);
        // price.0 = 1 (minimum non-zero FixedPoint), volume.0 = 100_000_000
        assert_eq!(trade.turnover(), 100_000_000);
    }
}
