//! Polars-based I/O operations for range bar data
//!
//! High-performance export capabilities using Polars for efficient
//! file format conversion and streaming operations.

use crate::formats::{ConversionError, DataFrameConverter};
use polars::prelude::*;
use rangebar_core::RangeBar;
use std::path::Path;
use thiserror::Error;

/// Polars exporter configuration
#[derive(Debug, Clone)]
pub struct PolarsExporterConfig {
    /// Compression level for Parquet files (0-22, higher = better compression)
    pub parquet_compression_level: Option<u8>,

    /// Row group size for Parquet files
    pub parquet_row_group_size: Option<usize>,

    /// Whether to use statistics in Parquet files
    pub parquet_statistics: bool,

    /// Buffer size for streaming operations
    pub streaming_buffer_size: usize,
}

impl Default for PolarsExporterConfig {
    fn default() -> Self {
        Self {
            parquet_compression_level: Some(6), // Good balance of speed/compression
            parquet_row_group_size: Some(100_000), // Optimal for range bar data
            parquet_statistics: true,
            streaming_buffer_size: 8192,
        }
    }
}

/// Main Polars exporter for range bar data
#[derive(Debug)]
pub struct PolarsExporter {
    config: PolarsExporterConfig,
}

impl PolarsExporter {
    /// Create new exporter with default configuration
    pub fn new() -> Self {
        Self {
            config: PolarsExporterConfig::default(),
        }
    }

    /// Create new exporter with custom configuration
    pub fn with_config(config: PolarsExporterConfig) -> Self {
        Self { config }
    }

    /// Export range bars to Parquet format
    pub fn export_parquet<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<ParquetExportResult, ExportError> {
        if range_bars.is_empty() {
            return Err(ExportError::EmptyData);
        }

        // Convert to DataFrame
        let df = range_bars.to_vec().to_polars_dataframe().map_err(|e| {
            ExportError::ConversionFailed {
                source: ConversionError::PolarsError(e),
            }
        })?;

        // Write Parquet file directly
        let mut file =
            std::fs::File::create(path.as_ref()).map_err(|e| ExportError::WriteFailed {
                format: "parquet".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Snappy)
            .with_statistics(if self.config.parquet_statistics {
                StatisticsOptions::default()
            } else {
                StatisticsOptions::empty()
            })
            .finish(&mut df.clone())
            .map_err(|e| ExportError::WriteFailed {
                format: "parquet".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        Ok(ParquetExportResult {
            records_written: range_bars.len(),
            file_path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    /// Export range bars to Arrow IPC format (zero-copy for Python)
    pub fn export_arrow_ipc<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<ArrowExportResult, ExportError> {
        if range_bars.is_empty() {
            return Err(ExportError::EmptyData);
        }

        let df = range_bars.to_vec().to_polars_dataframe().map_err(|e| {
            ExportError::ConversionFailed {
                source: ConversionError::PolarsError(e),
            }
        })?;

        // Write Arrow IPC file directly
        let mut file =
            std::fs::File::create(path.as_ref()).map_err(|e| ExportError::WriteFailed {
                format: "arrow".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        polars::io::ipc::IpcWriter::new(&mut file)
            .finish(&mut df.clone())
            .map_err(|e| ExportError::WriteFailed {
                format: "arrow".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        Ok(ArrowExportResult {
            records_written: range_bars.len(),
            file_path: path.as_ref().to_string_lossy().to_string(),
        })
    }

    /// Export range bars to CSV using Polars streaming writer
    pub fn export_streaming_csv<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<CsvExportResult, ExportError> {
        if range_bars.is_empty() {
            return Err(ExportError::EmptyData);
        }

        let df = range_bars.to_vec().to_polars_dataframe().map_err(|e| {
            ExportError::ConversionFailed {
                source: ConversionError::PolarsError(e),
            }
        })?;

        // Write CSV file directly
        let mut file =
            std::fs::File::create(path.as_ref()).map_err(|e| ExportError::WriteFailed {
                format: "csv".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        CsvWriter::new(&mut file)
            .include_header(true)
            .with_separator(b',')
            .finish(&mut df.clone())
            .map_err(|e| ExportError::WriteFailed {
                format: "csv".to_string(),
                path: path.as_ref().to_string_lossy().to_string(),
                source: e.into(),
            })?;

        Ok(CsvExportResult {
            records_written: range_bars.len(),
            file_path: path.as_ref().to_string_lossy().to_string(),
        })
    }
}

impl Default for PolarsExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized Parquet exporter
#[derive(Debug)]
pub struct ParquetExporter {
    exporter: PolarsExporter,
}

impl ParquetExporter {
    pub fn new() -> Self {
        Self {
            exporter: PolarsExporter::new(),
        }
    }

    pub fn export<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<ParquetExportResult, ExportError> {
        self.exporter.export_parquet(range_bars, path)
    }
}

impl Default for ParquetExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized Arrow exporter
#[derive(Debug)]
pub struct ArrowExporter {
    exporter: PolarsExporter,
}

impl ArrowExporter {
    pub fn new() -> Self {
        Self {
            exporter: PolarsExporter::new(),
        }
    }

    pub fn export<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<ArrowExportResult, ExportError> {
        self.exporter.export_arrow_ipc(range_bars, path)
    }
}

impl Default for ArrowExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized streaming CSV exporter
#[derive(Debug)]
pub struct StreamingCsvExporter {
    exporter: PolarsExporter,
}

impl StreamingCsvExporter {
    pub fn new() -> Self {
        Self {
            exporter: PolarsExporter::new(),
        }
    }

    pub fn export<P: AsRef<Path>>(
        &self,
        range_bars: &[RangeBar],
        path: P,
    ) -> Result<CsvExportResult, ExportError> {
        self.exporter.export_streaming_csv(range_bars, path)
    }
}

impl Default for StreamingCsvExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parquet export result
#[derive(Debug, Clone)]
pub struct ParquetExportResult {
    pub records_written: usize,
    pub file_path: String,
}

/// Arrow export result
#[derive(Debug, Clone)]
pub struct ArrowExportResult {
    pub records_written: usize,
    pub file_path: String,
}

/// CSV export result
#[derive(Debug, Clone)]
pub struct CsvExportResult {
    pub records_written: usize,
    pub file_path: String,
}

/// Export operation errors
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("No data to export")]
    EmptyData,

    #[error("Data conversion failed")]
    ConversionFailed {
        #[source]
        source: ConversionError,
    },

    #[error("Failed to write {format} file to '{path}'")]
    WriteFailed {
        format: String,
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rangebar_core::{DataSource, FixedPoint, RangeBar};
    use tempfile::tempdir;

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
                data_source: DataSource::BinanceFuturesUM,
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
                data_source: DataSource::BinanceFuturesUM,
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
    fn test_parquet_export() {
        let range_bars = create_test_range_bars();
        let exporter = PolarsExporter::new();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.parquet");

        let result = exporter.export_parquet(&range_bars, &file_path).unwrap();

        assert_eq!(result.records_written, 2);
        assert!(file_path.exists());
    }

    #[test]
    fn test_arrow_export() {
        let range_bars = create_test_range_bars();
        let exporter = PolarsExporter::new();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.arrow");

        let result = exporter.export_arrow_ipc(&range_bars, &file_path).unwrap();

        assert_eq!(result.records_written, 2);
        assert!(file_path.exists());
    }

    #[test]
    fn test_streaming_csv_export() {
        let range_bars = create_test_range_bars();
        let exporter = PolarsExporter::new();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.csv");

        let result = exporter
            .export_streaming_csv(&range_bars, &file_path)
            .unwrap();

        assert_eq!(result.records_written, 2);
        assert!(file_path.exists());
    }

    #[test]
    fn test_empty_data_export() {
        let empty_bars: Vec<RangeBar> = vec![];
        let exporter = PolarsExporter::new();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.parquet");

        let result = exporter.export_parquet(&empty_bars, &file_path);
        assert!(matches!(result, Err(ExportError::EmptyData)));
    }

    #[test]
    fn test_specialized_exporters() {
        let range_bars = create_test_range_bars();
        let temp_dir = tempdir().unwrap();

        // Test ParquetExporter
        let parquet_exporter = ParquetExporter::new();
        let parquet_path = temp_dir.path().join("specialized.parquet");
        let parquet_result = parquet_exporter.export(&range_bars, &parquet_path).unwrap();
        assert_eq!(parquet_result.records_written, 2);

        // Test ArrowExporter
        let arrow_exporter = ArrowExporter::new();
        let arrow_path = temp_dir.path().join("specialized.arrow");
        let arrow_result = arrow_exporter.export(&range_bars, &arrow_path).unwrap();
        assert_eq!(arrow_result.records_written, 2);

        // Test StreamingCsvExporter
        let csv_exporter = StreamingCsvExporter::new();
        let csv_path = temp_dir.path().join("specialized.csv");
        let csv_result = csv_exporter.export(&range_bars, &csv_path).unwrap();
        assert_eq!(csv_result.records_written, 2);
    }
}
