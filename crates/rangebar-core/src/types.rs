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

    // === MICROSTRUCTURE FEATURES (Issue #25) ===
    // Computed at bar finalization via compute_microstructure_features()
    /// Bar duration in microseconds (close_time - open_time)
    /// Reference: Easley et al. (2012) "Volume Clock"
    #[serde(default)]
    pub duration_us: i64,

    /// Order Flow Imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    /// Range: [-1.0, +1.0], Reference: Chordia et al. (2002)
    #[serde(default)]
    pub ofi: f64,

    /// VWAP-Close Deviation: (close - vwap) / (high - low)
    /// Measures intra-bar momentum, Reference: Berkowitz et al. (1988)
    #[serde(default)]
    pub vwap_close_deviation: f64,

    /// Price Impact (Amihud-style): abs(close - open) / volume
    /// Reference: Amihud (2002) illiquidity ratio
    #[serde(default)]
    pub price_impact: f64,

    /// Kyle's Lambda Proxy: (close - open) / (buy_vol - sell_vol)
    /// Market depth measure, Reference: Kyle (1985)
    #[serde(default)]
    pub kyle_lambda_proxy: f64,

    /// Trade Intensity: individual_trade_count / duration_seconds
    /// Reference: Engle & Russell (1998) ACD models
    #[serde(default)]
    pub trade_intensity: f64,

    /// Average trade size: volume / individual_trade_count
    /// Reference: Barclay & Warner (1993) stealth trading
    #[serde(default)]
    pub volume_per_trade: f64,

    /// Aggression Ratio: buy_trade_count / sell_trade_count
    /// Capped at 100.0, Reference: Lee & Ready (1991)
    #[serde(default)]
    pub aggression_ratio: f64,

    /// Aggregation Density: individual_trade_count / agg_record_count
    /// Average number of individual trades per AggTrade record (Issue #32 rename)
    /// Higher values = more fragmented trades; Lower values = more consolidated orders
    #[serde(default)]
    pub aggregation_density_f64: f64,

    /// Turnover Imbalance: (buy_turnover - sell_turnover) / total_turnover
    /// Dollar-weighted OFI, Range: [-1.0, +1.0]
    #[serde(default)]
    pub turnover_imbalance: f64,
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

            // Microstructure features (Issue #25) - computed at finalization
            duration_us: 0,
            ofi: 0.0,
            vwap_close_deviation: 0.0,
            price_impact: 0.0,
            kyle_lambda_proxy: 0.0,
            trade_intensity: 0.0,
            volume_per_trade: 0.0,
            aggression_ratio: 0.0,
            aggregation_density_f64: 0.0,
            turnover_imbalance: 0.0,
        }
    }

    /// Average number of individual trades per AggTrade record (aggregation density)
    pub fn aggregation_density(&self) -> f64 {
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

    /// Compute all microstructure features at bar finalization (Issue #25)
    ///
    /// This method computes 10 derived features from the accumulated bar state.
    /// All features use only data with timestamps <= bar.close_time (no lookahead).
    ///
    /// # Features Computed
    ///
    /// | Feature | Formula | Academic Reference |
    /// |---------|---------|-------------------|
    /// | duration_us | close_time - open_time | Easley et al. (2012) |
    /// | ofi | (buy_vol - sell_vol) / total | Chordia et al. (2002) |
    /// | vwap_close_deviation | (close - vwap) / (high - low) | Berkowitz et al. (1988) |
    /// | price_impact | abs(close - open) / volume | Amihud (2002) |
    /// | kyle_lambda_proxy | (close - open) / (buy_vol - sell_vol) | Kyle (1985) |
    /// | trade_intensity | trade_count / duration_seconds | Engle & Russell (1998) |
    /// | volume_per_trade | volume / trade_count | Barclay & Warner (1993) |
    /// | aggression_ratio | buy_trades / sell_trades | Lee & Ready (1991) |
    /// | aggregation_density_f64 | trade_count / agg_count | (proxy) |
    /// | turnover_imbalance | (buy_turn - sell_turn) / total_turn | (proxy) |
    pub fn compute_microstructure_features(&mut self) {
        // Extract values for computation
        let buy_vol = self.buy_volume.to_f64();
        let sell_vol = self.sell_volume.to_f64();
        let total_vol = buy_vol + sell_vol;
        let volume = self.volume.to_f64();
        let open = self.open.to_f64();
        let close = self.close.to_f64();
        let high = self.high.to_f64();
        let low = self.low.to_f64();
        let vwap = self.vwap.to_f64();
        let duration_us_raw = self.close_time - self.open_time;
        let trade_count = self.individual_trade_count as f64;
        let agg_count = self.agg_record_count as f64;
        let buy_turn = self.buy_turnover as f64;
        let sell_turn = self.sell_turnover as f64;
        let total_turn = (self.buy_turnover + self.sell_turnover) as f64;

        // 1. Duration (already in microseconds)
        self.duration_us = duration_us_raw;

        // 2. Order Flow Imbalance [-1, 1]
        self.ofi = if total_vol > f64::EPSILON {
            (buy_vol - sell_vol) / total_vol
        } else {
            0.0
        };

        // 3. VWAP-Close Deviation (normalized by price range)
        let range = high - low;
        self.vwap_close_deviation = if range > f64::EPSILON {
            (close - vwap) / range
        } else {
            0.0
        };

        // 4. Price Impact (Amihud-style)
        self.price_impact = if volume > f64::EPSILON {
            (close - open).abs() / volume
        } else {
            0.0
        };

        // 5. Kyle's Lambda Proxy (percentage returns formula - Issue #32)
        // Formula: ((close - open) / open) / ((buy_vol - sell_vol) / total_vol)
        // Creates dimensionally consistent ratio: percentage return per unit of normalized imbalance
        // Reference: Kyle (1985), normalized for cross-asset comparability
        let imbalance = buy_vol - sell_vol;
        let normalized_imbalance = if total_vol > f64::EPSILON {
            imbalance / total_vol
        } else {
            0.0
        };
        self.kyle_lambda_proxy =
            if normalized_imbalance.abs() > f64::EPSILON && open.abs() > f64::EPSILON {
                ((close - open) / open) / normalized_imbalance
            } else {
                0.0
            };

        // 6. Trade Intensity (trades per second)
        // Note: duration_us is in microseconds, convert to seconds
        let duration_sec = duration_us_raw as f64 / 1_000_000.0;
        self.trade_intensity = if duration_sec > f64::EPSILON {
            trade_count / duration_sec
        } else {
            trade_count // Instant bar = all trades at once
        };

        // 7. Volume per Trade (average trade size)
        self.volume_per_trade = if trade_count > f64::EPSILON {
            volume / trade_count
        } else {
            0.0
        };

        // 8. Aggression Ratio [0, 100] (capped)
        let sell_count = self.sell_trade_count as f64;
        self.aggression_ratio = if sell_count > f64::EPSILON {
            (self.buy_trade_count as f64 / sell_count).min(100.0)
        } else if self.buy_trade_count > 0 {
            100.0 // All buys, cap at 100
        } else {
            1.0 // No trades = neutral
        };

        // 9. Aggregation Density (Issue #32 rename: was aggregation_efficiency)
        // Measures average trades per AggTrade record (higher = more fragmented)
        self.aggregation_density_f64 = if agg_count > f64::EPSILON {
            trade_count / agg_count
        } else {
            1.0
        };

        // 10. Turnover Imbalance (dollar-weighted OFI) [-1, 1]
        self.turnover_imbalance = if total_turn.abs() > f64::EPSILON {
            (buy_turn - sell_turn) / total_turn
        } else {
            0.0
        };
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

    // =========================================================================
    // Microstructure Features Tests (Issue #25)
    // =========================================================================

    #[test]
    fn test_ofi_balanced() {
        // Create a bar with equal buy and sell volumes
        let buy_trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000, // microseconds
            1,
            1,
            false, // Buy
        );

        let mut bar = RangeBar::new(&buy_trade);

        let sell_trade = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995201000000, // 1 second later
            2,
            2,
            true, // Sell
        );

        bar.update_with_trade(&sell_trade);
        bar.compute_microstructure_features();

        // OFI should be 0 when buy_volume == sell_volume
        assert!(
            bar.ofi.abs() < f64::EPSILON,
            "OFI should be 0 for balanced volumes, got {}",
            bar.ofi
        );
    }

    #[test]
    fn test_ofi_all_buys() {
        // Create a bar with only buy volume
        let buy_trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            false, // Buy
        );

        let mut bar = RangeBar::new(&buy_trade1);

        let buy_trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995201000000,
            2,
            2,
            false, // Buy
        );

        bar.update_with_trade(&buy_trade2);
        bar.compute_microstructure_features();

        // OFI should be 1.0 when all buys
        assert!(
            (bar.ofi - 1.0).abs() < f64::EPSILON,
            "OFI should be 1.0 for all buys, got {}",
            bar.ofi
        );
    }

    #[test]
    fn test_ofi_all_sells() {
        // Create a bar with only sell volume
        let sell_trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            true, // Sell
        );

        let mut bar = RangeBar::new(&sell_trade1);

        let sell_trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995201000000,
            2,
            2,
            true, // Sell
        );

        bar.update_with_trade(&sell_trade2);
        bar.compute_microstructure_features();

        // OFI should be -1.0 when all sells
        assert!(
            (bar.ofi - (-1.0)).abs() < f64::EPSILON,
            "OFI should be -1.0 for all sells, got {}",
            bar.ofi
        );
    }

    #[test]
    fn test_turnover_imbalance_bounded() {
        // Create a bar with mixed buy/sell turnover
        let buy_trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            false, // Buy
        );

        let mut bar = RangeBar::new(&buy_trade);

        let sell_trade = test_utils::create_test_agg_trade_with_range(
            2,
            "50100.0",
            "2.0",
            1640995201000000,
            2,
            2,
            true, // Sell
        );

        bar.update_with_trade(&sell_trade);
        bar.compute_microstructure_features();

        // Turnover imbalance should be in [-1, 1]
        assert!(
            bar.turnover_imbalance >= -1.0 && bar.turnover_imbalance <= 1.0,
            "Turnover imbalance should be in [-1, 1], got {}",
            bar.turnover_imbalance
        );
    }

    #[test]
    fn test_kyle_lambda_div_zero() {
        // Create a bar with equal buy/sell -> imbalance = 0
        let buy_trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            false, // Buy
        );

        let mut bar = RangeBar::new(&buy_trade);

        let sell_trade = test_utils::create_test_agg_trade_with_range(
            2,
            "50100.0",
            "1.0",
            1640995201000000,
            2,
            2,
            true, // Sell
        );

        bar.update_with_trade(&sell_trade);
        bar.compute_microstructure_features();

        // Kyle lambda should be 0 when imbalance is 0
        assert!(
            bar.kyle_lambda_proxy.abs() < f64::EPSILON,
            "Kyle lambda should be 0 when imbalance is 0, got {}",
            bar.kyle_lambda_proxy
        );
    }

    #[test]
    fn test_duration_positive() {
        let trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000, // microseconds
            1,
            1,
            false,
        );

        let mut bar = RangeBar::new(&trade1);

        let trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995205000000, // 5 seconds later
            2,
            2,
            true,
        );

        bar.update_with_trade(&trade2);
        bar.compute_microstructure_features();

        // Duration should be 5 seconds = 5,000,000 microseconds
        assert_eq!(
            bar.duration_us, 5_000_000,
            "Duration should be 5,000,000 microseconds, got {}",
            bar.duration_us
        );
    }

    #[test]
    fn test_trade_intensity() {
        let trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000, // microseconds
            1,
            1,
            false,
        );

        let mut bar = RangeBar::new(&trade1);

        let trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995202000000, // 2 seconds later
            2,
            2,
            true,
        );

        bar.update_with_trade(&trade2);
        bar.compute_microstructure_features();

        // Trade intensity = 2 trades / 2 seconds = 1 trade/sec
        assert!(
            (bar.trade_intensity - 1.0).abs() < 0.01,
            "Trade intensity should be ~1 trade/sec, got {}",
            bar.trade_intensity
        );
    }

    #[test]
    fn test_aggression_ratio_capped() {
        // Create a bar with only buy trades (no sells)
        let buy_trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            false, // Buy
        );

        let mut bar = RangeBar::new(&buy_trade1);

        let buy_trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "1.0",
            1640995201000000,
            2,
            2,
            false, // Buy
        );

        bar.update_with_trade(&buy_trade2);
        bar.compute_microstructure_features();

        // Aggression ratio should be capped at 100 when no sells
        assert_eq!(
            bar.aggression_ratio, 100.0,
            "Aggression ratio should be 100.0 when no sells, got {}",
            bar.aggression_ratio
        );
    }

    #[test]
    fn test_aggregation_density() {
        // Create a bar with multiple individual trades per agg record
        let trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "5.0",
            1640995200000000,
            1,
            10, // 10 individual trades in this agg record
            false,
        );

        let mut bar = RangeBar::new(&trade);
        bar.compute_microstructure_features();

        // individual_trade_count / agg_record_count = 10 / 1 = 10.0
        assert!(
            (bar.aggregation_density_f64 - 10.0).abs() < 0.01,
            "Aggregation efficiency should be 10.0, got {}",
            bar.aggregation_density_f64
        );
    }

    #[test]
    fn test_vwap_close_deviation_zero_range() {
        // Create a bar with high == low (zero range)
        let trade = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "1.0",
            1640995200000000,
            1,
            1,
            false,
        );

        let mut bar = RangeBar::new(&trade);
        bar.compute_microstructure_features();

        // VWAP close deviation should be 0 when high == low
        assert_eq!(
            bar.vwap_close_deviation, 0.0,
            "VWAP close deviation should be 0 when high == low, got {}",
            bar.vwap_close_deviation
        );
    }

    #[test]
    fn test_volume_per_trade() {
        let trade1 = test_utils::create_test_agg_trade_with_range(
            1,
            "50000.0",
            "3.0",
            1640995200000000,
            1,
            1,
            false,
        );

        let mut bar = RangeBar::new(&trade1);

        let trade2 = test_utils::create_test_agg_trade_with_range(
            2,
            "50050.0",
            "7.0",
            1640995201000000,
            2,
            2,
            true,
        );

        bar.update_with_trade(&trade2);
        bar.compute_microstructure_features();

        // volume_per_trade = 10 / 2 = 5.0
        assert!(
            (bar.volume_per_trade - 5.0).abs() < 0.01,
            "Volume per trade should be 5.0, got {}",
            bar.volume_per_trade
        );
    }
}
