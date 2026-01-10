//! Type definitions for range bar processing

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

/// Range bar with OHLCV data and market microstructure enhancements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RangeBar {
    /// Opening timestamp in microseconds (first trade)
    pub open_time: i64,

    /// Closing timestamp in microseconds (last trade)
    pub close_time: i64,

    /// Opening price (first trade price)
    pub open: FixedPoint,

    /// Highest price in bar
    pub high: FixedPoint,

    /// Lowest price in bar
    pub low: FixedPoint,

    /// Closing price (breach trade price)
    pub close: FixedPoint,

    /// Total volume
    pub volume: FixedPoint,

    /// Total turnover (sum of price * volume)
    pub turnover: i128,

    /// Total number of individual exchange trades in this range bar
    /// Sum of individual_trade_count() from all processed AggTrade records
    pub individual_trade_count: u32,

    /// Number of AggTrade records processed to create this range bar
    /// NEW: Enables tracking of aggregation efficiency
    pub agg_record_count: u32,

    /// First individual trade ID in this range bar
    pub first_trade_id: i64,

    /// Last individual trade ID in this range bar
    pub last_trade_id: i64,

    /// Data source this range bar was created from
    pub data_source: DataSource,

    // === MARKET MICROSTRUCTURE ENHANCEMENTS ===
    /// Volume from buy-side trades (is_buyer_maker = false)
    /// Represents aggressive buying pressure
    pub buy_volume: FixedPoint,

    /// Volume from sell-side trades (is_buyer_maker = true)
    /// Represents aggressive selling pressure
    pub sell_volume: FixedPoint,

    /// Number of individual buy-side trades (aggressive buying)
    pub buy_trade_count: u32,

    /// Number of individual sell-side trades (aggressive selling)
    pub sell_trade_count: u32,

    /// Volume Weighted Average Price for the bar
    /// Calculated incrementally as: sum(price * volume) / sum(volume)
    pub vwap: FixedPoint,

    /// Turnover from buy-side trades (buy pressure)
    pub buy_turnover: i128,

    /// Turnover from sell-side trades (sell pressure)
    pub sell_turnover: i128,
}

impl RangeBar {
    /// Create new range bar from opening AggTrade record
    pub fn new(trade: &AggTrade) -> Self {
        let trade_turnover = trade.turnover();
        let individual_trades = trade.individual_trade_count() as u32;

        // Segregate order flow based on is_buyer_maker
        let (buy_volume, sell_volume) = if trade.is_buyer_maker {
            (FixedPoint(0), trade.volume) // Seller aggressive = sell pressure
        } else {
            (trade.volume, FixedPoint(0)) // Buyer aggressive = buy pressure
        };

        let (buy_trade_count, sell_trade_count) = if trade.is_buyer_maker {
            (0, individual_trades)
        } else {
            (individual_trades, 0)
        };

        let (buy_turnover, sell_turnover) = if trade.is_buyer_maker {
            (0, trade_turnover)
        } else {
            (trade_turnover, 0)
        };

        Self {
            open_time: trade.timestamp,
            close_time: trade.timestamp,
            open: trade.price,
            high: trade.price,
            low: trade.price,
            close: trade.price,
            volume: trade.volume,
            turnover: trade_turnover,

            // NEW: Enhanced counting
            individual_trade_count: individual_trades,
            agg_record_count: 1, // This is the first AggTrade record
            first_trade_id: trade.first_trade_id,
            last_trade_id: trade.last_trade_id,
            data_source: DataSource::default(),

            // Market microstructure fields
            buy_volume,
            sell_volume,
            buy_trade_count,
            sell_trade_count,
            vwap: trade.price, // Initial VWAP equals opening price
            buy_turnover,
            sell_turnover,
        }
    }

    /// Average number of individual trades per AggTrade record (aggregation efficiency)
    pub fn aggregation_efficiency(&self) -> f64 {
        if self.agg_record_count == 0 {
            0.0
        } else {
            self.individual_trade_count as f64 / self.agg_record_count as f64
        }
    }

    /// Update bar with new AggTrade record (always call before checking breach)
    /// Maintains market microstructure metrics incrementally
    pub fn update_with_trade(&mut self, trade: &AggTrade) {
        // Update price extremes
        if trade.price > self.high {
            self.high = trade.price;
        }
        if trade.price < self.low {
            self.low = trade.price;
        }

        // Update closing data
        self.close = trade.price;
        self.close_time = trade.timestamp;
        self.last_trade_id = trade.last_trade_id; // NEW: Track individual trade ID

        // Cache trade metrics for efficiency
        let trade_turnover = trade.turnover();
        let individual_trades = trade.individual_trade_count() as u32;

        // Update totals
        self.volume = FixedPoint(self.volume.0 + trade.volume.0);
        self.turnover += trade_turnover;

        // Enhanced counting
        self.individual_trade_count += individual_trades;
        self.agg_record_count += 1; // Track number of AggTrade records

        // === MARKET MICROSTRUCTURE INCREMENTAL UPDATES ===

        // Update order flow segregation
        if trade.is_buyer_maker {
            // Seller aggressive = sell pressure
            self.sell_volume = FixedPoint(self.sell_volume.0 + trade.volume.0);
            self.sell_trade_count += individual_trades;
            self.sell_turnover += trade_turnover;
        } else {
            // Buyer aggressive = buy pressure
            self.buy_volume = FixedPoint(self.buy_volume.0 + trade.volume.0);
            self.buy_trade_count += individual_trades;
            self.buy_turnover += trade_turnover;
        }

        // Update VWAP incrementally: VWAP = total_turnover / total_volume
        // Using integer arithmetic to maintain precision
        if self.volume.0 > 0 {
            // Calculate VWAP: turnover / volume, but maintain FixedPoint precision
            // turnover is in (price * volume) units, volume is in volume units
            // VWAP should be in price units
            let vwap_raw = self.turnover / (self.volume.0 as i128);
            self.vwap = FixedPoint(vwap_raw as i64);
        }
    }

    /// Check if price breaches the range thresholds
    ///
    /// # Arguments
    ///
    /// * `price` - Current price to check
    /// * `upper_threshold` - Upper breach threshold (from bar open)
    /// * `lower_threshold` - Lower breach threshold (from bar open)
    ///
    /// # Returns
    ///
    /// `true` if price breaches either threshold
    pub fn is_breach(
        &self,
        price: FixedPoint,
        upper_threshold: FixedPoint,
        lower_threshold: FixedPoint,
    ) -> bool {
        price >= upper_threshold || price <= lower_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;

    #[test]
    fn test_agg_trade_creation() {
        let trade = test_utils::create_test_agg_trade_with_range(
            12345,
            "50000.12345678",
            "1.5",
            1640995200000,
            100,
            102,
            false, // Buy pressure (taker buying from maker)
        );

        assert_eq!(trade.individual_trade_count(), 3); // 102 - 100 + 1
        assert!(trade.turnover() > 0);
    }

    #[test]
    fn test_range_bar_creation() {
        let trade = test_utils::create_test_agg_trade_with_range(
            12345,
            "50000.0",
            "1.0",
            1640995200000,
            100,
            100,
            true, // Sell pressure (taker selling to maker)
        );

        let bar = RangeBar::new(&trade);
        assert_eq!(bar.open, trade.price);
        assert_eq!(bar.high, trade.price);
        assert_eq!(bar.low, trade.price);
        assert_eq!(bar.close, trade.price);
    }

    #[test]
    fn test_range_bar_update() {
        let trade1 = test_utils::create_test_agg_trade_with_range(
            12345,
            "50000.0",
            "1.0",
            1640995200000,
            100,
            100,
            false, // Buy pressure
        );

        let mut bar = RangeBar::new(&trade1);

        let trade2 = test_utils::create_test_agg_trade_with_range(
            12346,
            "50100.0",
            "2.0",
            1640995201000,
            101,
            101,
            true, // Sell pressure
        );

        bar.update_with_trade(&trade2);

        assert_eq!(bar.open.to_string(), "50000.00000000");
        assert_eq!(bar.high.to_string(), "50100.00000000");
        assert_eq!(bar.low.to_string(), "50000.00000000");
        assert_eq!(bar.close.to_string(), "50100.00000000");
        assert_eq!(bar.volume.to_string(), "3.00000000");
        assert_eq!(bar.individual_trade_count, 2);
    }

    #[test]
    fn test_microstructure_segregation() {
        // Create buy trade (is_buyer_maker = false)
        let buy_trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.5",
            1640995200000,
            1,
            1,
            false, // Buy pressure (taker buying from maker)
        );

        let mut bar = RangeBar::new(&buy_trade);

        // Create sell trade (is_buyer_maker = true)
        let sell_trade = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "2.5",
            1640995201000,
            2,
            3,    // Multiple trades aggregated
            true, // Sell pressure (taker selling to maker)
        );

        bar.update_with_trade(&sell_trade);

        // Verify order flow segregation
        assert_eq!(bar.buy_volume.to_string(), "1.50000000"); // Only first trade
        assert_eq!(bar.sell_volume.to_string(), "2.50000000"); // Only second trade
        assert_eq!(bar.buy_trade_count, 1); // First trade count
        assert_eq!(bar.sell_trade_count, 2); // Second trade count (3 - 2 + 1 = 2)

        // Verify totals
        assert_eq!(bar.volume.to_string(), "4.00000000"); // 1.5 + 2.5
        assert_eq!(bar.individual_trade_count, 3); // 1 + 2

        // Verify VWAP calculation
        // VWAP = (50000 * 1.5 + 50050 * 2.5) / 4.0 = (75000 + 125125) / 4.0 = 50031.25
        assert_eq!(bar.vwap.to_string(), "50031.25000000");

        println!("✅ Microstructure segregation test passed:");
        println!(
            "   Buy volume: {}, Sell volume: {}",
            bar.buy_volume.to_string(),
            bar.sell_volume.to_string()
        );
        println!(
            "   Buy trades: {}, Sell trades: {}",
            bar.buy_trade_count, bar.sell_trade_count
        );
        println!("   VWAP: {}", bar.vwap.to_string());
    }
}
