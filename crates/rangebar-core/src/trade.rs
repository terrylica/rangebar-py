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
    pub fn individual_trade_count(&self) -> i64 {
        self.last_trade_id - self.first_trade_id + 1
    }

    /// Turnover (price * volume) as i128 to prevent overflow
    pub fn turnover(&self) -> i128 {
        (self.price.0 as i128) * (self.volume.0 as i128)
    }
}
