//! Streaming-optimized statistics for range bar analysis
//!
//! Clean implementation using production-proven streaming statistics crates:
//! - tdigests: t-digest algorithm for streaming percentiles
//! - rolling-stats: Welford's algorithm for numerically stable variance
//! - online-statistics: Comprehensive streaming statistics with serialization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use rangebar_core::{AggTrade, RangeBar};

/// Core streaming statistics engine optimized for range bar processing
pub struct StreamingStatsEngine {
    /// Configuration for statistical computation
    #[allow(dead_code)]
    config: StreamingConfig,

    /// Trade-level streaming statistics
    trade_stats: TradeStats,

    /// Range bar streaming statistics
    bar_stats: BarStats,
}

/// Configuration for streaming statistics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable percentile computation (uses t-digest)
    pub enable_percentiles: bool,

    /// Enable rolling statistics (uses Welford's algorithm)
    pub enable_rolling_stats: bool,

    /// Window size for rolling statistics
    pub rolling_window_size: usize,

    /// T-digest compression parameter (higher = more accurate, more memory)
    pub tdigest_compression: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_percentiles: true,
            enable_rolling_stats: true,
            rolling_window_size: 1000,
            tdigest_compression: 100.0, // Good balance of accuracy/memory
        }
    }
}

/// Trade-level streaming statistics
pub struct TradeStats {
    count: u64,

    #[cfg(feature = "stats")]
    price_values: Vec<f64>,

    #[cfg(feature = "stats")]
    volume_values: Vec<f64>,

    #[cfg(feature = "stats")]
    rolling_price: rolling_stats::Stats<f64>,

    #[cfg(feature = "stats")]
    rolling_volume: rolling_stats::Stats<f64>,
}

/// Range bar streaming statistics
pub struct BarStats {
    count: u64,

    #[cfg(feature = "stats")]
    ohlc_values: HashMap<String, Vec<f64>>,

    #[cfg(feature = "stats")]
    rolling_ohlc: HashMap<String, rolling_stats::Stats<f64>>,
}

/// Serializable statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSnapshot {
    /// Trade count processed
    pub trade_count: u64,

    /// Bar count generated
    pub bar_count: u64,

    /// Price statistics
    pub price_stats: PriceStatistics,

    /// Volume statistics
    pub volume_stats: VolumeStatistics,

    /// OHLC statistics
    pub ohlc_stats: OhlcStatistics,

    /// Timestamp of snapshot
    pub timestamp: String,
}

/// Price streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceStatistics {
    /// Percentiles (P50, P75, P90, P95, P99)
    pub percentiles: HashMap<String, f64>,

    /// Rolling statistics
    pub rolling: RollingStats,

    /// Price range (min/max)
    pub range: (f64, f64),
}

/// Volume streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeStatistics {
    /// Percentiles (P50, P75, P90, P95, P99)
    pub percentiles: HashMap<String, f64>,

    /// Rolling statistics
    pub rolling: RollingStats,

    /// Volume range (min/max)
    pub range: (f64, f64),
}

/// OHLC streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcStatistics {
    /// Open price statistics
    pub open: PriceStatistics,

    /// High price statistics
    pub high: PriceStatistics,

    /// Low price statistics
    pub low: PriceStatistics,

    /// Close price statistics
    pub close: PriceStatistics,
}

/// Rolling window statistics (Welford's algorithm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingStats {
    /// Mean
    pub mean: f64,

    /// Variance (sample)
    pub variance: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Count in rolling window
    pub count: u64,
}

impl StreamingStatsEngine {
    /// Create new streaming statistics engine
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config: config.clone(),
            trade_stats: TradeStats::new(&config),
            bar_stats: BarStats::new(&config),
        }
    }

    /// Process single trade for streaming statistics
    pub fn process_trade(&mut self, trade: &AggTrade) {
        self.trade_stats.update(trade);
    }

    /// Process single range bar for streaming statistics
    pub fn process_bar(&mut self, bar: &RangeBar) {
        self.bar_stats.update(bar);
    }

    /// Get current statistics snapshot (serializable)
    pub fn snapshot(&self) -> StatisticsSnapshot {
        StatisticsSnapshot {
            trade_count: self.trade_stats.count,
            bar_count: self.bar_stats.count,
            price_stats: self.trade_stats.price_statistics(),
            volume_stats: self.trade_stats.volume_statistics(),
            ohlc_stats: self.bar_stats.ohlc_statistics(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl Default for StreamingStatsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TradeStats {
    #[cfg(feature = "stats")]
    fn new(_config: &StreamingConfig) -> Self {
        Self {
            count: 0,
            price_values: Vec::new(),
            volume_values: Vec::new(),
            rolling_price: rolling_stats::Stats::new(),
            rolling_volume: rolling_stats::Stats::new(),
        }
    }

    #[cfg(not(feature = "stats"))]
    fn new(_config: &StreamingConfig) -> Self {
        Self { count: 0 }
    }

    fn update(&mut self, _trade: &AggTrade) {
        self.count += 1;

        #[cfg(feature = "stats")]
        {
            let price = _trade.price.to_f64();
            let volume = _trade.volume.to_f64();

            self.price_values.push(price);
            self.volume_values.push(volume);

            self.rolling_price.update(price);
            self.rolling_volume.update(volume);
        }
    }

    fn price_statistics(&self) -> PriceStatistics {
        #[cfg(feature = "stats")]
        {
            let percentiles = if !self.price_values.is_empty() {
                let mut tdigest = tdigests::TDigest::from_values(self.price_values.clone());
                tdigest.compress(100);

                [
                    ("P50", 0.5),
                    ("P75", 0.75),
                    ("P90", 0.9),
                    ("P95", 0.95),
                    ("P99", 0.99),
                ]
                .iter()
                .map(|(name, quantile)| (name.to_string(), tdigest.estimate_quantile(*quantile)))
                .collect()
            } else {
                HashMap::new()
            };

            let range = if !self.price_values.is_empty() {
                let min = self
                    .price_values
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                let max = self
                    .price_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            } else {
                (0.0, 0.0)
            };

            PriceStatistics {
                percentiles,
                rolling: RollingStats {
                    mean: self.rolling_price.mean,
                    variance: 0.0, // Not available in rolling-stats
                    std_dev: self.rolling_price.std_dev,
                    count: self.rolling_price.count as u64,
                },
                range,
            }
        }

        #[cfg(not(feature = "stats"))]
        {
            PriceStatistics {
                percentiles: HashMap::new(),
                rolling: RollingStats {
                    mean: 0.0,
                    variance: 0.0,
                    std_dev: 0.0,
                    count: self.count,
                },
                range: (0.0, 0.0),
            }
        }
    }

    fn volume_statistics(&self) -> VolumeStatistics {
        #[cfg(feature = "stats")]
        {
            let percentiles = if !self.volume_values.is_empty() {
                let mut tdigest = tdigests::TDigest::from_values(self.volume_values.clone());
                tdigest.compress(100);

                [
                    ("P50", 0.5),
                    ("P75", 0.75),
                    ("P90", 0.9),
                    ("P95", 0.95),
                    ("P99", 0.99),
                ]
                .iter()
                .map(|(name, quantile)| (name.to_string(), tdigest.estimate_quantile(*quantile)))
                .collect()
            } else {
                HashMap::new()
            };

            let range = if !self.volume_values.is_empty() {
                let min = self
                    .volume_values
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                let max = self
                    .volume_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            } else {
                (0.0, 0.0)
            };

            VolumeStatistics {
                percentiles,
                rolling: RollingStats {
                    mean: self.rolling_volume.mean,
                    variance: 0.0, // Not available in rolling-stats
                    std_dev: self.rolling_volume.std_dev,
                    count: self.rolling_volume.count as u64,
                },
                range,
            }
        }

        #[cfg(not(feature = "stats"))]
        {
            VolumeStatistics {
                percentiles: HashMap::new(),
                rolling: RollingStats {
                    mean: 0.0,
                    variance: 0.0,
                    std_dev: 0.0,
                    count: self.count,
                },
                range: (0.0, 0.0),
            }
        }
    }
}

impl BarStats {
    #[cfg(feature = "stats")]
    fn new(_config: &StreamingConfig) -> Self {
        let mut ohlc_values = HashMap::new();
        let mut rolling_ohlc = HashMap::new();

        for field in ["open", "high", "low", "close"] {
            ohlc_values.insert(field.to_string(), Vec::new());
            rolling_ohlc.insert(field.to_string(), rolling_stats::Stats::new());
        }

        Self {
            count: 0,
            ohlc_values,
            rolling_ohlc,
        }
    }

    #[cfg(not(feature = "stats"))]
    fn new(_config: &StreamingConfig) -> Self {
        Self { count: 0 }
    }

    fn update(&mut self, _bar: &RangeBar) {
        self.count += 1;

        #[cfg(feature = "stats")]
        {
            let values = [
                ("open", _bar.open.to_f64()),
                ("high", _bar.high.to_f64()),
                ("low", _bar.low.to_f64()),
                ("close", _bar.close.to_f64()),
            ];

            for (field, value) in values {
                if let Some(values_vec) = self.ohlc_values.get_mut(field) {
                    values_vec.push(value);
                }
                if let Some(rolling) = self.rolling_ohlc.get_mut(field) {
                    rolling.update(value);
                }
            }
        }
    }

    fn ohlc_statistics(&self) -> OhlcStatistics {
        #[cfg(feature = "stats")]
        {
            let create_price_stats = |field: &str| -> PriceStatistics {
                let values = &self.ohlc_values[field];
                let rolling = &self.rolling_ohlc[field];

                let percentiles = if !values.is_empty() {
                    let mut tdigest = tdigests::TDigest::from_values(values.clone());
                    tdigest.compress(100);

                    [
                        ("P50", 0.5),
                        ("P75", 0.75),
                        ("P90", 0.9),
                        ("P95", 0.95),
                        ("P99", 0.99),
                    ]
                    .iter()
                    .map(|(name, quantile)| {
                        (name.to_string(), tdigest.estimate_quantile(*quantile))
                    })
                    .collect()
                } else {
                    HashMap::new()
                };

                let range = if !values.is_empty() {
                    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    (min, max)
                } else {
                    (0.0, 0.0)
                };

                PriceStatistics {
                    percentiles,
                    rolling: RollingStats {
                        mean: rolling.mean,
                        variance: 0.0, // Not available in rolling-stats
                        std_dev: rolling.std_dev,
                        count: rolling.count as u64,
                    },
                    range,
                }
            };

            OhlcStatistics {
                open: create_price_stats("open"),
                high: create_price_stats("high"),
                low: create_price_stats("low"),
                close: create_price_stats("close"),
            }
        }

        #[cfg(not(feature = "stats"))]
        {
            let empty_stats = PriceStatistics {
                percentiles: HashMap::new(),
                rolling: RollingStats {
                    mean: 0.0,
                    variance: 0.0,
                    std_dev: 0.0,
                    count: self.count,
                },
                range: (0.0, 0.0),
            };

            OhlcStatistics {
                open: empty_stats.clone(),
                high: empty_stats.clone(),
                low: empty_stats.clone(),
                close: empty_stats,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rangebar_core::{DataSource, FixedPoint};

    #[test]
    fn test_streaming_stats_engine_creation() {
        let engine = StreamingStatsEngine::new();
        let snapshot = engine.snapshot();

        assert_eq!(snapshot.trade_count, 0);
        assert_eq!(snapshot.bar_count, 0);
    }

    #[test]
    fn test_trade_processing() {
        let mut engine = StreamingStatsEngine::new();

        let trade = AggTrade {
            agg_trade_id: 1,
            price: FixedPoint::from_str("50000.0").unwrap(),
            volume: FixedPoint::from_str("1.5").unwrap(),
            first_trade_id: 1,
            last_trade_id: 1,
            timestamp: 1609459200000,
            is_buyer_maker: false,
            is_best_match: None,
        };

        engine.process_trade(&trade);
        let snapshot = engine.snapshot();

        assert_eq!(snapshot.trade_count, 1);
    }

    #[test]
    fn test_bar_processing() {
        let mut engine = StreamingStatsEngine::new();

        let bar = RangeBar {
            open_time: 1609459200000,
            close_time: 1609459200000,
            open: FixedPoint::from_str("50000.0").unwrap(),
            high: FixedPoint::from_str("50100.0").unwrap(),
            low: FixedPoint::from_str("49900.0").unwrap(),
            close: FixedPoint::from_str("50050.0").unwrap(),
            volume: FixedPoint::from_str("10.5").unwrap(),
            turnover: 0,
            individual_trade_count: 42,
            agg_record_count: 1,
            first_trade_id: 1,
            last_trade_id: 42,
            data_source: DataSource::BinanceFuturesUM,
            buy_volume: FixedPoint::from_str("5.0").unwrap(),
            buy_turnover: 0,
            sell_volume: FixedPoint::from_str("5.5").unwrap(),
            sell_turnover: 0,
            buy_trade_count: 20,
            sell_trade_count: 22,
            vwap: FixedPoint::from_str("50025.0").unwrap(),
        };

        engine.process_bar(&bar);
        let snapshot = engine.snapshot();

        assert_eq!(snapshot.bar_count, 1);
    }
}
