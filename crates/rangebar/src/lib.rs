//! Non-lookahead range bar construction for cryptocurrency and forex trading.
//!
//! [![Crates.io](https://img.shields.io/crates/v/rangebar.svg)](https://crates.io/crates/rangebar)
//! [![Documentation](https://docs.rs/rangebar/badge.svg)](https://docs.rs/rangebar)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//!
//! This crate provides algorithms for constructing range bars from tick data
//! with temporal integrity guarantees, ensuring no lookahead bias in financial backtesting.
//!
//! ## Installation
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! rangebar = "6.1"
//! ```
//!
//! ## Meta-Crate
//!
//! This is a meta-crate that re-exports all rangebar sub-crates for backward compatibility
//! with v4.0.0. New code should depend on specific sub-crates directly:
//!
//! - `rangebar-core` - Core algorithm and types
//! - `rangebar-providers` - Data providers (Binance, Exness)
//! - `rangebar-config` - Configuration management
//! - `rangebar-io` - I/O operations and Polars integration
//! - `rangebar-streaming` - Real-time streaming processor
//! - `rangebar-batch` - Batch analytics engine
//! - `rangebar-cli` - Command-line tools
//!
//! ## Features
//!
//! - `core` - Core algorithm (always enabled)
//! - `providers` - Data providers (Binance, Exness)
//! - `config` - Configuration management
//! - `io` - I/O operations and Polars integration
//! - `streaming` - Real-time streaming processor
//! - `batch` - Batch analytics engine
//! - `full` - Enable all features
//!
//! ## Basic Usage
//!
//! ```rust
//! use rangebar::{RangeBarProcessor, AggTrade, FixedPoint};
//!
//! // Create processor with 25 basis points threshold (0.25%)
//! let mut processor = RangeBarProcessor::new(250).unwrap();
//!
//! // Create sample aggTrade
//! let trade = AggTrade {
//!     agg_trade_id: 1,
//!     price: FixedPoint::from_str("50000.0").unwrap(),
//!     volume: FixedPoint::from_str("1.0").unwrap(),
//!     first_trade_id: 1,
//!     last_trade_id: 1,
//!     timestamp: 1609459200000,
//!     is_buyer_maker: false,
//!     is_best_match: None,
//! };
//!
//! // Process aggTrade records into range bars
//! let agg_trade_records = vec![trade];
//! let bars = processor.process_agg_trade_records(&agg_trade_records).unwrap();
//! ```
//!
//! ## Dual-Path Architecture
//!
//! ### Streaming Mode (Real-time)
//! ```rust
//! # #[cfg(feature = "streaming")]
//! # {
//! use rangebar::streaming::StreamingProcessor;
//!
//! let threshold_decimal_bps = 250; // 25 bps = 0.25% range bars
//! let processor = StreamingProcessor::new(threshold_decimal_bps);
//! // Real-time processing with bounded memory
//! # }
//! ```
//!
//! ### Batch Mode (Analytics)
//! ```rust
//! # #[cfg(feature = "batch")]
//! # {
//! use rangebar::batch::BatchAnalysisEngine;
//! use rangebar::core::types::RangeBar;
//!
//! let range_bars: Vec<RangeBar> = vec![]; // Your range bar data
//! let engine = BatchAnalysisEngine::new();
//! // let result = engine.analyze_single_symbol(&range_bars, "BTCUSDT").unwrap();
//! # }
//! ```
//!
//! ## Links
//!
//! - [GitHub Repository](https://github.com/terrylica/rangebar-py)
//! - [API Documentation](https://github.com/terrylica/rangebar-py/blob/main/docs/api.md)
//! - [Changelog](https://github.com/terrylica/rangebar-py/blob/main/CHANGELOG.md)
//! - [Core API Reference](https://github.com/terrylica/rangebar-py/blob/main/docs/rangebar_core_api.md)

// Re-export core (always available)
pub use rangebar_core as core;

// Re-export optional crates
#[cfg(feature = "providers")]
pub use rangebar_providers as providers;

#[cfg(feature = "config")]
pub use rangebar_config as config;

#[cfg(feature = "io")]
pub use rangebar_io as io;

#[cfg(feature = "streaming")]
pub use rangebar_streaming as streaming;

#[cfg(feature = "batch")]
pub use rangebar_batch as batch;

// Legacy compatibility modules for v4.0.0 API
pub mod fixed_point {
    //! Fixed-point arithmetic module (legacy compatibility)
    pub use rangebar_core::fixed_point::*;
}

pub mod range_bars {
    //! Range bar processor module (legacy compatibility)
    pub use rangebar_core::processor::*;
}

pub mod types {
    //! Core types module (legacy compatibility)
    pub use rangebar_core::types::*;
}

#[cfg(feature = "providers")]
pub mod tier1 {
    //! Tier-1 symbol discovery (legacy compatibility)
    pub use rangebar_providers::binance::symbols::*;
}

#[cfg(feature = "providers")]
pub mod data {
    //! Historical data loading (legacy compatibility)
    pub use rangebar_providers::binance::HistoricalDataLoader;
}

#[cfg(feature = "config")]
pub mod infrastructure {
    //! Infrastructure modules (legacy compatibility)

    #[cfg(feature = "config")]
    pub use rangebar_config as config;

    #[cfg(feature = "io")]
    pub use rangebar_io as io;
}

#[cfg(feature = "streaming")]
pub mod engines {
    //! Engine modules (legacy compatibility)

    #[cfg(feature = "streaming")]
    pub use rangebar_streaming as streaming;

    #[cfg(feature = "batch")]
    pub use rangebar_batch as batch;
}

// Re-export commonly used types at crate root for convenience
pub use rangebar_core::{
    AggTrade, ExportRangeBarProcessor, FixedPoint, ProcessingError, RangeBar, RangeBarProcessor,
};

#[cfg(feature = "config")]
pub use rangebar_config::Settings;

#[cfg(feature = "providers")]
pub use rangebar_providers::binance::{
    get_tier1_symbols, get_tier1_usdt_pairs, is_tier1_symbol, TIER1_SYMBOLS,
};

#[cfg(feature = "streaming")]
pub use rangebar_streaming::processor::{
    StreamingError, StreamingMetrics, StreamingProcessor, StreamingProcessorConfig,
};

#[cfg(feature = "batch")]
pub use rangebar_batch::{AnalysisReport, BatchAnalysisEngine, BatchConfig, BatchResult};

// Re-export I/O types when feature is enabled
#[cfg(feature = "io")]
pub use rangebar_io::{ArrowExporter, ParquetExporter, PolarsExporter, StreamingCsvExporter};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Library initialization and configuration
pub fn init() {
    // Future: Initialize logging, metrics, etc.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert!(!NAME.is_empty());
        assert!(!DESCRIPTION.is_empty());
    }

    #[test]
    fn test_types_export() {
        // Test that we can create and use exported types
        let fp = FixedPoint::from_str("123.456").unwrap();
        assert_eq!(fp.to_string(), "123.45600000");
    }

    #[test]
    fn test_legacy_fixed_point_module() {
        // Test backward compatibility via legacy module path
        let fp = fixed_point::FixedPoint::from_str("100.0").unwrap();
        assert_eq!(fp.to_string(), "100.00000000");
    }

    #[test]
    fn test_legacy_types_module() {
        // Test backward compatibility via legacy types module
        use types::DataSource;
        let _data_source = DataSource::BinanceSpot;
    }

    #[cfg(feature = "providers")]
    #[test]
    fn test_tier1_symbols_export() {
        // Test that tier1 symbols are accessible
        let symbols = get_tier1_symbols();
        assert!(!symbols.is_empty());
        assert!(symbols.contains(&"BTC".to_string()));
    }

    #[cfg(feature = "config")]
    #[test]
    fn test_settings_export() {
        // Test that Settings type is accessible
        let settings = Settings::default();
        assert!(!settings.app.name.is_empty());
    }

    #[cfg(feature = "providers")]
    #[test]
    fn test_data_module_export() {
        // Test backward compatibility via legacy data module path
        // This verifies that `use rangebar::data::HistoricalDataLoader` works
        use data::HistoricalDataLoader;

        // Just verify the type is accessible
        let _loader = HistoricalDataLoader::new("BTCUSDT");
    }
}
