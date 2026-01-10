//! Format conversion utilities for range bar data
//!
//! Provides bidirectional conversion between RangeBar and Polars DataFrame
//! with exception-only failure handling.

use polars::prelude::*;
use rangebar_core::{AggTrade, FixedPoint, RangeBar};
use thiserror::Error;

/// Trait for converting between Rust types and Polars DataFrames
pub trait DataFrameConverter<T> {
    /// Convert to Polars DataFrame
    fn to_polars_dataframe(&self) -> PolarsResult<DataFrame>;

    /// Convert from Polars DataFrame
    fn from_polars_dataframe(df: DataFrame) -> Result<T, ConversionError>;
}

/// Conversion errors with rich context
#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("Missing required column: {column}")]
    MissingColumn { column: String },

    #[error("Invalid data type for column '{column}': expected {expected}, got {actual}")]
    InvalidDataType {
        column: String,
        expected: String,
        actual: String,
    },

    #[error("Data validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("FixedPoint conversion error: {0}")]
    FixedPointError(String),
}

/// Required columns for RangeBar DataFrame
pub const RANGEBAR_COLUMNS: &[&str] = &[
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "individual_trade_count",
    "agg_record_count",
    "first_trade_id",
    "last_trade_id",
    "data_source",
    "buy_volume",
    "sell_volume",
    "buy_trade_count",
    "sell_trade_count",
    "vwap",
    "buy_turnover",
    "sell_turnover",
];

/// Required columns for AggTrade DataFrame
pub const AGGTRADE_COLUMNS: &[&str] = &[
    "agg_trade_id",
    "price",
    "volume",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
];

impl DataFrameConverter<Vec<RangeBar>> for Vec<RangeBar> {
    fn to_polars_dataframe(&self) -> PolarsResult<DataFrame> {
        if self.is_empty() {
            return Err(PolarsError::NoData("Empty RangeBar vector".into()));
        }

        // Extract all fields into separate vectors for efficient columnar construction
        let open_times: Vec<i64> = self.iter().map(|bar| bar.open_time).collect();
        let close_times: Vec<i64> = self.iter().map(|bar| bar.close_time).collect();
        let opens: Vec<i64> = self.iter().map(|bar| bar.open.0).collect();
        let highs: Vec<i64> = self.iter().map(|bar| bar.high.0).collect();
        let lows: Vec<i64> = self.iter().map(|bar| bar.low.0).collect();
        let closes: Vec<i64> = self.iter().map(|bar| bar.close.0).collect();
        let volumes: Vec<i64> = self.iter().map(|bar| bar.volume.0).collect();
        let turnovers: Vec<i64> = self.iter().map(|bar| bar.turnover as i64).collect();
        let trade_counts: Vec<i64> = self
            .iter()
            .map(|bar| bar.individual_trade_count as i64)
            .collect();
        let first_ids: Vec<i64> = self.iter().map(|bar| bar.first_trade_id).collect();
        let last_ids: Vec<i64> = self.iter().map(|bar| bar.last_trade_id).collect();
        let buy_volumes: Vec<i64> = self.iter().map(|bar| bar.buy_volume.0).collect();
        let sell_volumes: Vec<i64> = self.iter().map(|bar| bar.sell_volume.0).collect();
        let buy_trade_counts: Vec<i64> =
            self.iter().map(|bar| bar.buy_trade_count as i64).collect();
        let sell_trade_counts: Vec<i64> =
            self.iter().map(|bar| bar.sell_trade_count as i64).collect();
        let vwaps: Vec<i64> = self.iter().map(|bar| bar.vwap.0).collect();
        let buy_turnovers: Vec<i64> = self.iter().map(|bar| bar.buy_turnover as i64).collect();
        let sell_turnovers: Vec<i64> = self.iter().map(|bar| bar.sell_turnover as i64).collect();

        // Create DataFrame with all columns (full names matching RANGEBAR_COLUMNS)
        DataFrame::new(vec![
            Column::new("open_time".into(), &open_times),
            Column::new("close_time".into(), &close_times),
            Column::new("open".into(), &opens),
            Column::new("high".into(), &highs),
            Column::new("low".into(), &lows),
            Column::new("close".into(), &closes),
            Column::new("volume".into(), &volumes),
            Column::new("turnover".into(), &turnovers),
            Column::new("individual_trade_count".into(), &trade_counts),
            Column::new(
                "agg_record_count".into(),
                &self
                    .iter()
                    .map(|bar| bar.agg_record_count as i64)
                    .collect::<Vec<i64>>(),
            ),
            Column::new("first_trade_id".into(), &first_ids),
            Column::new("last_trade_id".into(), &last_ids),
            Column::new(
                "data_source".into(),
                &self
                    .iter()
                    .map(|_| "BinanceFuturesUM")
                    .collect::<Vec<&str>>(),
            ),
            Column::new("buy_volume".into(), &buy_volumes),
            Column::new("sell_volume".into(), &sell_volumes),
            Column::new("buy_trade_count".into(), &buy_trade_counts),
            Column::new("sell_trade_count".into(), &sell_trade_counts),
            Column::new("vwap".into(), &vwaps),
            Column::new("buy_turnover".into(), &buy_turnovers),
            Column::new("sell_turnover".into(), &sell_turnovers),
        ])
    }

    fn from_polars_dataframe(df: DataFrame) -> Result<Vec<RangeBar>, ConversionError> {
        // Validate required columns are present
        validate_rangebar_columns(&df)?;

        let height = df.height();
        if height == 0 {
            return Ok(Vec::new());
        }

        let mut range_bars = Vec::with_capacity(height);

        // Extract all columns efficiently (using full column names)
        let open_times = extract_i64_column(&df, "open_time")?;
        let close_times = extract_i64_column(&df, "close_time")?;
        let opens = extract_i64_column(&df, "open")?;
        let highs = extract_i64_column(&df, "high")?;
        let lows = extract_i64_column(&df, "low")?;
        let closes = extract_i64_column(&df, "close")?;
        let volumes = extract_i64_column(&df, "volume")?;
        let turnovers = extract_i64_column(&df, "turnover")?;
        let trade_counts = extract_i64_column(&df, "individual_trade_count")?;
        let agg_record_counts = extract_i64_column(&df, "agg_record_count")?;
        let first_ids = extract_i64_column(&df, "first_trade_id")?;
        let last_ids = extract_i64_column(&df, "last_trade_id")?;
        let buy_volumes = extract_i64_column(&df, "buy_volume")?;
        let sell_volumes = extract_i64_column(&df, "sell_volume")?;
        let buy_trade_counts = extract_i64_column(&df, "buy_trade_count")?;
        let sell_trade_counts = extract_i64_column(&df, "sell_trade_count")?;
        let vwaps = extract_i64_column(&df, "vwap")?;
        let buy_turnovers = extract_i64_column(&df, "buy_turnover")?;
        let sell_turnovers = extract_i64_column(&df, "sell_turnover")?;

        // Construct RangeBar structs
        for i in 0..height {
            let range_bar = RangeBar {
                open_time: open_times[i],
                close_time: close_times[i],
                open: FixedPoint(opens[i]),
                high: FixedPoint(highs[i]),
                low: FixedPoint(lows[i]),
                close: FixedPoint(closes[i]),
                volume: FixedPoint(volumes[i]),
                turnover: turnovers[i] as i128,
                individual_trade_count: trade_counts[i] as u32,
                agg_record_count: agg_record_counts[i] as u32,
                first_trade_id: first_ids[i],
                last_trade_id: last_ids[i],
                data_source: rangebar_core::DataSource::default(),
                buy_volume: FixedPoint(buy_volumes[i]),
                sell_volume: FixedPoint(sell_volumes[i]),
                buy_trade_count: buy_trade_counts[i] as u32,
                sell_trade_count: sell_trade_counts[i] as u32,
                vwap: FixedPoint(vwaps[i]),
                buy_turnover: buy_turnovers[i] as i128,
                sell_turnover: sell_turnovers[i] as i128,
            };

            // Validate range bar data integrity
            validate_range_bar(&range_bar)?;
            range_bars.push(range_bar);
        }

        Ok(range_bars)
    }
}

impl DataFrameConverter<Vec<AggTrade>> for Vec<AggTrade> {
    fn to_polars_dataframe(&self) -> PolarsResult<DataFrame> {
        if self.is_empty() {
            return Err(PolarsError::NoData("Empty AggTrade vector".into()));
        }

        let agg_trade_ids: Vec<i64> = self.iter().map(|trade| trade.agg_trade_id).collect();
        let prices: Vec<i64> = self.iter().map(|trade| trade.price.0).collect();
        let volumes: Vec<i64> = self.iter().map(|trade| trade.volume.0).collect();
        let first_trade_ids: Vec<i64> = self.iter().map(|trade| trade.first_trade_id).collect();
        let last_trade_ids: Vec<i64> = self.iter().map(|trade| trade.last_trade_id).collect();
        let timestamps: Vec<i64> = self.iter().map(|trade| trade.timestamp).collect();
        let is_buyer_makers: Vec<bool> = self.iter().map(|trade| trade.is_buyer_maker).collect();

        DataFrame::new(vec![
            Column::new("agg_trade_id".into(), &agg_trade_ids),
            Column::new("price".into(), &prices),
            Column::new("volume".into(), &volumes),
            Column::new("first_trade_id".into(), &first_trade_ids),
            Column::new("last_trade_id".into(), &last_trade_ids),
            Column::new("timestamp".into(), &timestamps),
            Column::new("is_buyer_maker".into(), &is_buyer_makers),
        ])
    }

    fn from_polars_dataframe(df: DataFrame) -> Result<Vec<AggTrade>, ConversionError> {
        validate_aggtrade_columns(&df)?;

        let height = df.height();
        if height == 0 {
            return Ok(Vec::new());
        }

        let mut agg_trades = Vec::with_capacity(height);

        let agg_trade_ids = extract_i64_column(&df, "agg_trade_id")?;
        let prices = extract_i64_column(&df, "price")?;
        let volumes = extract_i64_column(&df, "volume")?;
        let first_trade_ids = extract_i64_column(&df, "first_trade_id")?;
        let last_trade_ids = extract_i64_column(&df, "last_trade_id")?;
        let timestamps = extract_i64_column(&df, "timestamp")?;
        let is_buyer_makers = extract_bool_column(&df, "is_buyer_maker")?;

        for i in 0..height {
            let agg_trade = AggTrade {
                agg_trade_id: agg_trade_ids[i],
                price: FixedPoint(prices[i]),
                volume: FixedPoint(volumes[i]),
                first_trade_id: first_trade_ids[i],
                last_trade_id: last_trade_ids[i],
                timestamp: timestamps[i],
                is_buyer_maker: is_buyer_makers[i],
                is_best_match: None,
            };

            validate_agg_trade(&agg_trade)?;
            agg_trades.push(agg_trade);
        }

        Ok(agg_trades)
    }
}

/// Validate RangeBar DataFrame has required columns
fn validate_rangebar_columns(df: &DataFrame) -> Result<(), ConversionError> {
    for &column in RANGEBAR_COLUMNS {
        if !df
            .get_column_names()
            .iter()
            .any(|name| name.as_str() == column)
        {
            return Err(ConversionError::MissingColumn {
                column: column.to_string(),
            });
        }
    }
    Ok(())
}

/// Validate AggTrade DataFrame has required columns
fn validate_aggtrade_columns(df: &DataFrame) -> Result<(), ConversionError> {
    for &column in AGGTRADE_COLUMNS {
        if !df
            .get_column_names()
            .iter()
            .any(|name| name.as_str() == column)
        {
            return Err(ConversionError::MissingColumn {
                column: column.to_string(),
            });
        }
    }
    Ok(())
}

/// Extract i64 column with error handling
fn extract_i64_column(df: &DataFrame, column_name: &str) -> Result<Vec<i64>, ConversionError> {
    let series = df
        .column(column_name)
        .map_err(|_| ConversionError::MissingColumn {
            column: column_name.to_string(),
        })?;

    Ok(series
        .i64()
        .map_err(|_| ConversionError::InvalidDataType {
            column: column_name.to_string(),
            expected: "i64".to_string(),
            actual: format!("{:?}", series.dtype()),
        })?
        .into_no_null_iter()
        .collect::<Vec<i64>>())
}

/// Extract boolean column with error handling
fn extract_bool_column(df: &DataFrame, column_name: &str) -> Result<Vec<bool>, ConversionError> {
    let series = df
        .column(column_name)
        .map_err(|_| ConversionError::MissingColumn {
            column: column_name.to_string(),
        })?;

    Ok(series
        .bool()
        .map_err(|_| ConversionError::InvalidDataType {
            column: column_name.to_string(),
            expected: "bool".to_string(),
            actual: format!("{:?}", series.dtype()),
        })?
        .into_no_null_iter()
        .collect::<Vec<bool>>())
}

/// Validate individual RangeBar data integrity
fn validate_range_bar(bar: &RangeBar) -> Result<(), ConversionError> {
    // Temporal validation
    if bar.close_time <= bar.open_time {
        return Err(ConversionError::ValidationFailed {
            message: format!(
                "Invalid time sequence: close_time ({}) <= open_time ({})",
                bar.close_time, bar.open_time
            ),
        });
    }

    // OHLC validation
    if bar.high < bar.low {
        return Err(ConversionError::ValidationFailed {
            message: format!("Invalid OHLC: high ({}) < low ({})", bar.high, bar.low),
        });
    }

    if bar.open > bar.high || bar.open < bar.low {
        return Err(ConversionError::ValidationFailed {
            message: "Open price outside high-low range".to_string(),
        });
    }

    if bar.close > bar.high || bar.close < bar.low {
        return Err(ConversionError::ValidationFailed {
            message: "Close price outside high-low range".to_string(),
        });
    }

    // Volume validation
    if bar.volume.0 <= 0 {
        return Err(ConversionError::ValidationFailed {
            message: "Volume must be positive".to_string(),
        });
    }

    // Trade count validation
    if bar.individual_trade_count == 0 {
        return Err(ConversionError::ValidationFailed {
            message: "Trade count must be positive".to_string(),
        });
    }

    // ID validation
    if bar.last_trade_id < bar.first_trade_id {
        return Err(ConversionError::ValidationFailed {
            message: "last_id must be >= first_id".to_string(),
        });
    }

    Ok(())
}

/// Validate individual AggTrade data integrity
fn validate_agg_trade(trade: &AggTrade) -> Result<(), ConversionError> {
    // Price validation
    if trade.price.0 <= 0 {
        return Err(ConversionError::ValidationFailed {
            message: "Price must be positive".to_string(),
        });
    }

    // Volume validation
    if trade.volume.0 <= 0 {
        return Err(ConversionError::ValidationFailed {
            message: "Volume must be positive".to_string(),
        });
    }

    // Timestamp validation (basic sanity check)
    if trade.timestamp <= 0 {
        return Err(ConversionError::ValidationFailed {
            message: "Timestamp must be positive".to_string(),
        });
    }

    // Trade ID validation
    if trade.last_trade_id < trade.first_trade_id {
        return Err(ConversionError::ValidationFailed {
            message: "last_trade_id must be >= first_trade_id".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rangebar_core::{DataSource, FixedPoint};

    fn create_test_range_bar() -> RangeBar {
        RangeBar {
            open_time: 1000000,
            close_time: 1000001,
            open: FixedPoint(100000000),    // 1.0
            high: FixedPoint(110000000),    // 1.1
            low: FixedPoint(90000000),      // 0.9
            close: FixedPoint(105000000),   // 1.05
            volume: FixedPoint(1000000000), // 10.0
            turnover: 1050000000,
            individual_trade_count: 5,
            agg_record_count: 1,
            first_trade_id: 1,
            last_trade_id: 5,
            data_source: DataSource::BinanceFuturesUM,
            buy_volume: FixedPoint(600000000),
            sell_volume: FixedPoint(400000000),
            buy_trade_count: 3,
            sell_trade_count: 2,
            vwap: FixedPoint(105000000),
            buy_turnover: 630000000,
            sell_turnover: 420000000,
        }
    }

    #[test]
    fn test_rangebar_to_dataframe_conversion() {
        let bars = vec![create_test_range_bar()];
        let df = bars.to_polars_dataframe().unwrap();

        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), RANGEBAR_COLUMNS.len());

        // Verify specific values
        let open_series = df.column("open").unwrap();
        assert_eq!(open_series.i64().unwrap().get(0), Some(100000000));
    }

    #[test]
    fn test_dataframe_to_rangebar_conversion() {
        let original_bars = vec![create_test_range_bar()];
        let df = original_bars.to_polars_dataframe().unwrap();
        let converted_bars = Vec::<RangeBar>::from_polars_dataframe(df).unwrap();

        assert_eq!(converted_bars.len(), 1);
        let bar = &converted_bars[0];
        let original = &original_bars[0];

        assert_eq!(bar.open_time, original.open_time);
        assert_eq!(bar.close_time, original.close_time);
        assert_eq!(bar.open, original.open);
        assert_eq!(bar.high, original.high);
        assert_eq!(bar.low, original.low);
        assert_eq!(bar.close, original.close);
    }

    #[test]
    fn test_empty_vector_conversion() {
        let empty_bars: Vec<RangeBar> = vec![];
        let result = empty_bars.to_polars_dataframe();
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_column_validation() {
        let df = DataFrame::new(vec![
            Column::new("open_time".into(), vec![1000000i64]),
            // Missing other required columns
        ])
        .unwrap();

        let result = Vec::<RangeBar>::from_polars_dataframe(df);
        assert!(matches!(result, Err(ConversionError::MissingColumn { .. })));
    }

    #[test]
    fn test_invalid_range_bar_validation() {
        let mut invalid_bar = create_test_range_bar();
        invalid_bar.high = FixedPoint(50000000); // Invalid: high < low

        let result = validate_range_bar(&invalid_bar);
        assert!(matches!(
            result,
            Err(ConversionError::ValidationFailed { .. })
        ));
    }
}
