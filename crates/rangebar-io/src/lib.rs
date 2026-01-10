//! Input/output operations for range bar data
//!
//! This module provides efficient I/O operations for reading and writing
//! range bar data in various formats including CSV, Parquet, and Arrow.

#[cfg(feature = "parquet")]
pub mod polars_io;

#[cfg(feature = "parquet")]
pub mod formats;

// Re-export commonly used types when parquet feature is enabled
#[cfg(feature = "parquet")]
pub use formats::{ConversionError, DataFrameConverter};

#[cfg(feature = "parquet")]
pub use polars_io::{
    ArrowExporter, ExportError, ParquetExporter, PolarsExporter, PolarsExporterConfig,
    StreamingCsvExporter,
};
