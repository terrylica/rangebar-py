//! Batch analysis engine powered by Polars
//!
//! High-performance batch processing for historical analysis,
//! backtesting, and research with exception-only failure handling.

use polars::frame::row::Row;
use polars::prelude::*;
use rangebar_core::RangeBar;
use rangebar_io::formats::{ConversionError, DataFrameConverter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Configuration for batch analysis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum memory usage in bytes (for streaming operations)
    pub max_memory_bytes: usize,

    /// Chunk size for processing large datasets
    pub chunk_size: usize,

    /// Number of parallel threads for computation
    pub parallel_threads: Option<usize>,

    /// Enable lazy evaluation for better memory efficiency
    pub use_lazy_evaluation: bool,

    /// Statistical computations configuration
    pub statistics_config: StatisticsConfig,
}

/// Statistical computations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Compute rolling statistics
    pub enable_rolling_stats: bool,

    /// Window size for rolling statistics
    pub rolling_window_size: usize,

    /// Compute quantiles and percentiles
    pub enable_quantiles: bool,

    /// Quantile levels to compute (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
    pub quantile_levels: Vec<f64>,

    /// Enable correlation analysis
    pub enable_correlations: bool,

    /// Enable time-series resampling
    pub enable_resampling: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1_000_000_000, // 1GB
            chunk_size: 100_000,
            parallel_threads: None, // Use all available cores
            use_lazy_evaluation: true,
            statistics_config: StatisticsConfig {
                enable_rolling_stats: true,
                rolling_window_size: 20,
                enable_quantiles: true,
                quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
                enable_correlations: true,
                enable_resampling: true,
            },
        }
    }
}

/// Main batch analysis engine
#[derive(Debug)]
pub struct BatchAnalysisEngine {
    config: BatchConfig,
}

impl BatchAnalysisEngine {
    /// Create new batch analysis engine with default configuration
    pub fn new() -> Self {
        Self {
            config: BatchConfig::default(),
        }
    }

    /// Create new batch analysis engine with custom configuration
    pub fn with_config(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Analyze single symbol range bar data
    pub fn analyze_single_symbol(
        &self,
        range_bars: &[RangeBar],
        symbol: &str,
    ) -> Result<BatchResult, BatchError> {
        if range_bars.is_empty() {
            return Err(BatchError::EmptyData {
                symbol: symbol.to_string(),
            });
        }

        // Convert to DataFrame for analysis
        let df = range_bars.to_vec().to_polars_dataframe().map_err(|e| {
            BatchError::ConversionFailed {
                symbol: symbol.to_string(),
                source: ConversionError::PolarsError(e),
            }
        })?;

        let mut analysis_report = AnalysisReport::new(symbol.to_string());

        // Basic statistics
        analysis_report.basic_stats = self.compute_basic_statistics(&df)?;

        // Rolling statistics (if enabled)
        if self.config.statistics_config.enable_rolling_stats {
            analysis_report.rolling_stats = Some(self.compute_rolling_statistics(&df)?);
        }

        // Quantile analysis (if enabled)
        if self.config.statistics_config.enable_quantiles {
            analysis_report.quantiles = Some(self.compute_quantiles(&df)?);
        }

        // Price movement analysis
        analysis_report.price_analysis = self.compute_price_analysis(&df)?;

        // Volume analysis
        analysis_report.volume_analysis = self.compute_volume_analysis(&df)?;

        // Microstructure analysis
        analysis_report.microstructure = self.compute_microstructure_analysis(&df)?;

        Ok(BatchResult {
            symbol: symbol.to_string(),
            records_processed: range_bars.len(),
            analysis: analysis_report,
        })
    }

    /// Analyze multiple symbols for cross-symbol analysis
    pub fn analyze_multiple_symbols(
        &self,
        symbol_data: HashMap<String, Vec<RangeBar>>,
    ) -> Result<Vec<BatchResult>, BatchError> {
        if symbol_data.is_empty() {
            return Err(BatchError::NoSymbolData);
        }

        let mut results = Vec::with_capacity(symbol_data.len());

        // Process each symbol
        for (symbol, range_bars) in symbol_data {
            let result = self.analyze_single_symbol(&range_bars, &symbol)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compute basic statistics for range bar data
    fn compute_basic_statistics(&self, df: &DataFrame) -> Result<BasicStatistics, BatchError> {
        let lazy_df = df.clone().lazy();

        // Compute basic aggregations using Polars expressions
        let stats_df = lazy_df
            .select([
                // Price statistics
                col("close").mean().alias("close_mean"),
                col("close").std(1).alias("close_std"),
                col("close").min().alias("close_min"),
                col("close").max().alias("close_max"),
                col("close").median().alias("close_median"),
                // Volume statistics
                col("volume").mean().alias("volume_mean"),
                col("volume").std(1).alias("volume_std"),
                col("volume").sum().alias("volume_total"),
                // Count statistics
                len().alias("total_bars"),
                col("individual_trade_count").sum().alias("total_trades"),
                // Time span
                col("open_time").min().alias("first_time"),
                col("close_time").max().alias("last_time"),
            ])
            .collect()
            .map_err(|e| BatchError::ComputationFailed {
                operation: "basic_statistics".to_string(),
                source: e.into(),
            })?;

        // Extract values from the result
        let row = stats_df
            .get_row(0)
            .map_err(|e| BatchError::ComputationFailed {
                operation: "extract_basic_stats".to_string(),
                source: e.into(),
            })?;

        Ok(BasicStatistics {
            close_mean: extract_f64_value(&row, 0)?,
            close_std: extract_f64_value(&row, 1)?,
            close_min: extract_f64_value(&row, 2)?,
            close_max: extract_f64_value(&row, 3)?,
            close_median: extract_f64_value(&row, 4)?,
            volume_mean: extract_f64_value(&row, 5)?,
            volume_std: extract_f64_value(&row, 6)?,
            volume_total: extract_f64_value(&row, 7)?,
            total_bars: extract_i64_value(&row, 8)? as usize,
            total_trades: extract_i64_value(&row, 9)? as usize,
            first_time: extract_i64_value(&row, 10)?,
            last_time: extract_i64_value(&row, 11)?,
        })
    }

    /// Compute rolling statistics
    fn compute_rolling_statistics(&self, df: &DataFrame) -> Result<RollingStatistics, BatchError> {
        let window = self.config.statistics_config.rolling_window_size;

        let rolling_df = df
            .clone()
            .lazy()
            .with_columns([
                // Rolling price statistics
                col("close")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: window,
                        min_periods: 1,
                        ..Default::default()
                    })
                    .alias("close_sma"),
                col("close")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: window,
                        min_periods: 1,
                        ..Default::default()
                    })
                    .alias("close_rolling_std"),
                // Rolling volume statistics
                col("volume")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: window,
                        min_periods: 1,
                        ..Default::default()
                    })
                    .alias("volume_sma"),
                col("volume")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: window,
                        min_periods: 1,
                        ..Default::default()
                    })
                    .alias("volume_rolling_std"),
                // Price returns
                (col("close") / col("close").shift(lit(1)) - lit(1.0)).alias("returns"),
            ])
            .select([
                col("close_sma").mean().alias("avg_sma"),
                col("close_rolling_std").mean().alias("avg_volatility"),
                col("volume_sma").mean().alias("avg_volume_sma"),
                col("returns").std(1).alias("returns_volatility"),
            ])
            .collect()
            .map_err(|e| BatchError::ComputationFailed {
                operation: "rolling_statistics".to_string(),
                source: e.into(),
            })?;

        let row = rolling_df
            .get_row(0)
            .map_err(|e| BatchError::ComputationFailed {
                operation: "extract_rolling_stats".to_string(),
                source: e.into(),
            })?;

        Ok(RollingStatistics {
            window_size: window,
            avg_sma: extract_f64_value(&row, 0)?,
            avg_volatility: extract_f64_value(&row, 1)?,
            avg_volume_sma: extract_f64_value(&row, 2)?,
            returns_volatility: extract_f64_value(&row, 3)?,
        })
    }

    /// Compute quantiles and percentiles
    fn compute_quantiles(&self, df: &DataFrame) -> Result<QuantileAnalysis, BatchError> {
        let quantile_levels = &self.config.statistics_config.quantile_levels;
        let mut price_quantiles = Vec::new();
        let mut volume_quantiles = Vec::new();

        for &level in quantile_levels {
            // Price quantiles
            let price_q = df
                .clone()
                .lazy()
                .select([col("close").quantile(lit(level), QuantileMethod::Linear)])
                .collect()
                .map_err(|e| BatchError::ComputationFailed {
                    operation: format!("price_quantile_{}", level),
                    source: e.into(),
                })?;

            let price_value = price_q
                .get_row(0)
                .and_then(|row| {
                    extract_f64_value(&row, 0)
                        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
                })
                .map_err(|e| BatchError::ComputationFailed {
                    operation: format!("extract_price_quantile_{}", level),
                    source: e.into(),
                })?;

            price_quantiles.push((level, price_value));

            // Volume quantiles
            let volume_q = df
                .clone()
                .lazy()
                .select([col("volume").quantile(lit(level), QuantileMethod::Linear)])
                .collect()
                .map_err(|e| BatchError::ComputationFailed {
                    operation: format!("volume_quantile_{}", level),
                    source: e.into(),
                })?;

            let volume_value = volume_q
                .get_row(0)
                .and_then(|row| {
                    extract_f64_value(&row, 0)
                        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
                })
                .map_err(|e| BatchError::ComputationFailed {
                    operation: format!("extract_volume_quantile_{}", level),
                    source: e.into(),
                })?;

            volume_quantiles.push((level, volume_value));
        }

        Ok(QuantileAnalysis {
            levels: quantile_levels.clone(),
            price_quantiles,
            volume_quantiles,
        })
    }

    /// Compute price movement analysis
    fn compute_price_analysis(&self, df: &DataFrame) -> Result<PriceAnalysis, BatchError> {
        let price_df = df
            .clone()
            .lazy()
            .with_columns([
                // Price ranges
                (col("high") - col("low")).alias("price_range"),
                (col("close") - col("open")).alias("price_change"),
                ((col("close") - col("open")) / col("open") * lit(100.0)).alias("price_change_pct"),
                // OHLC analysis
                col("close").gt(col("open")).alias("is_bullish"),
                col("high").eq(col("close")).alias("is_high_close"),
                col("low").eq(col("close")).alias("is_low_close"),
            ])
            .select([
                col("price_range").mean().alias("avg_range"),
                col("price_range").std(1).alias("range_volatility"),
                col("price_change").mean().alias("avg_change"),
                col("price_change_pct").std(1).alias("price_volatility"),
                col("is_bullish").mean().alias("bullish_ratio"),
                col("is_high_close").mean().alias("high_close_ratio"),
                col("is_low_close").mean().alias("low_close_ratio"),
            ])
            .collect()
            .map_err(|e| BatchError::ComputationFailed {
                operation: "price_analysis".to_string(),
                source: e.into(),
            })?;

        let row = price_df
            .get_row(0)
            .map_err(|e| BatchError::ComputationFailed {
                operation: "extract_price_analysis".to_string(),
                source: e.into(),
            })?;

        Ok(PriceAnalysis {
            avg_range: extract_f64_value(&row, 0)?,
            range_volatility: extract_f64_value(&row, 1)?,
            avg_change: extract_f64_value(&row, 2)?,
            price_volatility: extract_f64_value(&row, 3)?,
            bullish_ratio: extract_f64_value(&row, 4)?,
            high_close_ratio: extract_f64_value(&row, 5)?,
            low_close_ratio: extract_f64_value(&row, 6)?,
        })
    }

    /// Compute volume analysis
    fn compute_volume_analysis(&self, df: &DataFrame) -> Result<VolumeAnalysis, BatchError> {
        let volume_df = df
            .clone()
            .lazy()
            .with_columns([
                // Buy/sell analysis
                (col("buy_volume") / col("volume")).alias("buy_ratio"),
                (col("sell_volume") / col("volume")).alias("sell_ratio"),
                col("vwap").alias("volume_weighted_price"),
            ])
            .select([
                col("buy_ratio").mean().alias("avg_buy_ratio"),
                col("sell_ratio").mean().alias("avg_sell_ratio"),
                col("buy_ratio").std(1).alias("buy_ratio_volatility"),
                col("volume_weighted_price").mean().alias("avg_vwap"),
                col("volume").sum().alias("total_volume"),
                col("buy_volume").sum().alias("total_buy_volume"),
                col("sell_volume").sum().alias("total_sell_volume"),
            ])
            .collect()
            .map_err(|e| BatchError::ComputationFailed {
                operation: "volume_analysis".to_string(),
                source: e.into(),
            })?;

        let row = volume_df
            .get_row(0)
            .map_err(|e| BatchError::ComputationFailed {
                operation: "extract_volume_analysis".to_string(),
                source: e.into(),
            })?;

        Ok(VolumeAnalysis {
            avg_buy_ratio: extract_f64_value(&row, 0)?,
            avg_sell_ratio: extract_f64_value(&row, 1)?,
            buy_ratio_volatility: extract_f64_value(&row, 2)?,
            avg_vwap: extract_f64_value(&row, 3)?,
            total_volume: extract_f64_value(&row, 4)?,
            total_buy_volume: extract_f64_value(&row, 5)?,
            total_sell_volume: extract_f64_value(&row, 6)?,
        })
    }

    /// Compute microstructure analysis
    fn compute_microstructure_analysis(
        &self,
        df: &DataFrame,
    ) -> Result<MicrostructureAnalysis, BatchError> {
        let micro_df = df
            .clone()
            .lazy()
            .with_columns([
                // AggTrade intensity
                (col("individual_trade_count")
                    / ((col("close_time") - col("open_time")) / lit(1000.0)))
                .alias("aggtrades_per_second"),
                // Order flow imbalance
                ((col("buy_volume") - col("sell_volume")) / col("volume"))
                    .alias("order_flow_imbalance"),
                // VWAP deviation
                ((col("close") - col("vwap")) / col("vwap") * lit(100.0)).alias("vwap_deviation"),
            ])
            .select([
                col("aggtrades_per_second")
                    .mean()
                    .alias("avg_aggtrade_intensity"),
                col("order_flow_imbalance")
                    .mean()
                    .alias("avg_order_flow_imbalance"),
                col("order_flow_imbalance")
                    .std(1)
                    .alias("order_flow_volatility"),
                col("vwap_deviation").mean().alias("avg_vwap_deviation"),
                col("vwap_deviation")
                    .std(1)
                    .alias("vwap_deviation_volatility"),
            ])
            .collect()
            .map_err(|e| BatchError::ComputationFailed {
                operation: "microstructure_analysis".to_string(),
                source: e.into(),
            })?;

        let row = micro_df
            .get_row(0)
            .map_err(|e| BatchError::ComputationFailed {
                operation: "extract_microstructure_analysis".to_string(),
                source: e.into(),
            })?;

        Ok(MicrostructureAnalysis {
            avg_trade_intensity: extract_f64_value(&row, 0)?,
            avg_order_flow_imbalance: extract_f64_value(&row, 1)?,
            order_flow_volatility: extract_f64_value(&row, 2)?,
            avg_vwap_deviation: extract_f64_value(&row, 3)?,
            vwap_deviation_volatility: extract_f64_value(&row, 4)?,
        })
    }
}

impl Default for BatchAnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub symbol: String,
    pub records_processed: usize,
    pub analysis: AnalysisReport,
}

/// Comprehensive analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub symbol: String,
    pub basic_stats: BasicStatistics,
    pub rolling_stats: Option<RollingStatistics>,
    pub quantiles: Option<QuantileAnalysis>,
    pub price_analysis: PriceAnalysis,
    pub volume_analysis: VolumeAnalysis,
    pub microstructure: MicrostructureAnalysis,
}

impl AnalysisReport {
    fn new(symbol: String) -> Self {
        Self {
            symbol,
            basic_stats: BasicStatistics::default(),
            rolling_stats: None,
            quantiles: None,
            price_analysis: PriceAnalysis::default(),
            volume_analysis: VolumeAnalysis::default(),
            microstructure: MicrostructureAnalysis::default(),
        }
    }
}

/// Basic statistical measures
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicStatistics {
    pub close_mean: f64,
    pub close_std: f64,
    pub close_min: f64,
    pub close_max: f64,
    pub close_median: f64,
    pub volume_mean: f64,
    pub volume_std: f64,
    pub volume_total: f64,
    pub total_bars: usize,
    pub total_trades: usize,
    pub first_time: i64,
    pub last_time: i64,
}

/// Rolling window statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingStatistics {
    pub window_size: usize,
    pub avg_sma: f64,
    pub avg_volatility: f64,
    pub avg_volume_sma: f64,
    pub returns_volatility: f64,
}

/// Quantile analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileAnalysis {
    pub levels: Vec<f64>,
    pub price_quantiles: Vec<(f64, f64)>,
    pub volume_quantiles: Vec<(f64, f64)>,
}

/// Price movement analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriceAnalysis {
    pub avg_range: f64,
    pub range_volatility: f64,
    pub avg_change: f64,
    pub price_volatility: f64,
    pub bullish_ratio: f64,
    pub high_close_ratio: f64,
    pub low_close_ratio: f64,
}

/// Volume analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolumeAnalysis {
    pub avg_buy_ratio: f64,
    pub avg_sell_ratio: f64,
    pub buy_ratio_volatility: f64,
    pub avg_vwap: f64,
    pub total_volume: f64,
    pub total_buy_volume: f64,
    pub total_sell_volume: f64,
}

/// Market microstructure analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MicrostructureAnalysis {
    pub avg_trade_intensity: f64,
    pub avg_order_flow_imbalance: f64,
    pub order_flow_volatility: f64,
    pub avg_vwap_deviation: f64,
    pub vwap_deviation_volatility: f64,
}

/// Batch processing errors with rich context
#[derive(Debug, Error)]
pub enum BatchError {
    #[error("No data provided for symbol: {symbol}")]
    EmptyData { symbol: String },

    #[error("No symbol data provided")]
    NoSymbolData,

    #[error("Data conversion failed for symbol '{symbol}'")]
    ConversionFailed {
        symbol: String,
        #[source]
        source: ConversionError,
    },

    #[error("Computation failed for operation '{operation}'")]
    ComputationFailed {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Value extraction failed for {operation}")]
    ValueExtractionFailed {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Helper function to extract f64 values from Polars rows
fn extract_f64_value(row: &Row, index: usize) -> Result<f64, BatchError> {
    match row.0.get(index) {
        Some(AnyValue::Float64(val)) => Ok(*val),
        Some(AnyValue::Float32(val)) => Ok(*val as f64),
        Some(AnyValue::Int64(val)) => Ok(*val as f64),
        Some(AnyValue::Int32(val)) => Ok(*val as f64),
        Some(AnyValue::Null) => Ok(0.0), // Return 0.0 for null values
        Some(other) => Err(BatchError::ValueExtractionFailed {
            operation: format!("extract_f64_at_index_{}", index),
            source: format!("Unexpected type: {:?}", other).into(),
        }),
        None => Err(BatchError::ValueExtractionFailed {
            operation: format!("extract_f64_at_index_{}", index),
            source: "Value not found".into(),
        }),
    }
}

/// Helper function to extract i64 values from Polars rows
fn extract_i64_value(row: &Row, index: usize) -> Result<i64, BatchError> {
    match row.0.get(index) {
        Some(AnyValue::Int64(val)) => Ok(*val),
        Some(AnyValue::Int32(val)) => Ok(*val as i64),
        Some(AnyValue::UInt64(val)) => Ok(*val as i64),
        Some(AnyValue::UInt32(val)) => Ok(*val as i64),
        Some(other) => Err(BatchError::ValueExtractionFailed {
            operation: format!("extract_i64_at_index_{}", index),
            source: format!("Unexpected type: {:?}", other).into(),
        }),
        None => Err(BatchError::ValueExtractionFailed {
            operation: format!("extract_i64_at_index_{}", index),
            source: "Value not found".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rangebar_core::{DataSource, FixedPoint, RangeBar};

    fn create_test_range_bars() -> Vec<RangeBar> {
        vec![
            RangeBar {
                open_time: 1000000,
                close_time: 1000001,
                open: FixedPoint(100000000),
                high: FixedPoint(110000000),
                low: FixedPoint(90000000),
                close: FixedPoint(105000000),
                volume: FixedPoint(1000000000),
                turnover: 1050000000,
                individual_trade_count: 5,
                agg_record_count: 1,
                first_trade_id: 1,
                last_trade_id: 5,
                data_source: DataSource::default(),
                buy_volume: FixedPoint(600000000),
                sell_volume: FixedPoint(400000000),
                buy_trade_count: 3,
                sell_trade_count: 2,
                vwap: FixedPoint(105000000),
                buy_turnover: 630000000,
                sell_turnover: 420000000,
            },
            RangeBar {
                open_time: 1000002,
                close_time: 1000003,
                open: FixedPoint(105000000),
                high: FixedPoint(115000000),
                low: FixedPoint(95000000),
                close: FixedPoint(110000000),
                volume: FixedPoint(2000000000),
                turnover: 2200000000,
                individual_trade_count: 8,
                agg_record_count: 1,
                first_trade_id: 6,
                last_trade_id: 13,
                data_source: DataSource::default(),
                buy_volume: FixedPoint(1200000000),
                sell_volume: FixedPoint(800000000),
                buy_trade_count: 5,
                sell_trade_count: 3,
                vwap: FixedPoint(110000000),
                buy_turnover: 1320000000,
                sell_turnover: 880000000,
            },
        ]
    }

    #[test]
    fn test_batch_analysis_engine_creation() {
        let engine = BatchAnalysisEngine::new();
        assert_eq!(engine.config.chunk_size, 100_000);

        let custom_config = BatchConfig {
            chunk_size: 50_000,
            ..Default::default()
        };
        let custom_engine = BatchAnalysisEngine::with_config(custom_config);
        assert_eq!(custom_engine.config.chunk_size, 50_000);
    }

    #[test]
    fn test_single_symbol_analysis() {
        let engine = BatchAnalysisEngine::new();
        let range_bars = create_test_range_bars();

        let result = engine
            .analyze_single_symbol(&range_bars, "BTCUSDT")
            .unwrap();

        assert_eq!(result.symbol, "BTCUSDT");
        assert_eq!(result.records_processed, 2);
        assert_eq!(result.analysis.basic_stats.total_bars, 2);
        assert_eq!(result.analysis.basic_stats.total_trades, 13);
    }

    #[test]
    fn test_multiple_symbols_analysis() {
        let engine = BatchAnalysisEngine::new();
        let mut symbol_data = HashMap::new();
        symbol_data.insert("BTCUSDT".to_string(), create_test_range_bars());
        symbol_data.insert("ETHUSDT".to_string(), create_test_range_bars());

        let results = engine.analyze_multiple_symbols(symbol_data).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.symbol == "BTCUSDT"));
        assert!(results.iter().any(|r| r.symbol == "ETHUSDT"));
    }

    #[test]
    fn test_empty_data_error() {
        let engine = BatchAnalysisEngine::new();
        let empty_bars: Vec<RangeBar> = vec![];

        let result = engine.analyze_single_symbol(&empty_bars, "BTCUSDT");
        assert!(matches!(result, Err(BatchError::EmptyData { .. })));
    }

    #[test]
    fn test_no_symbol_data_error() {
        let engine = BatchAnalysisEngine::new();
        let empty_data = HashMap::new();

        let result = engine.analyze_multiple_symbols(empty_data);
        assert!(matches!(result, Err(BatchError::NoSymbolData)));
    }
}
