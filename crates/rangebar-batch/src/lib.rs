//! Batch processing and analysis using Polars
//!
//! This module provides batch analytics capabilities powered by Polars
//! for research, backtesting, and advanced statistical analysis.

pub mod engine;

// Re-export commonly used types
pub use engine::{AnalysisReport, BatchAnalysisEngine, BatchConfig, BatchError, BatchResult};
