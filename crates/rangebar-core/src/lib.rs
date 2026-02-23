//! Core range bar processing algorithms
//!
//! Non-lookahead bias range bar construction with temporal integrity guarantees.
//!
//! ## Features
//!
//! - Non-lookahead bias: Thresholds computed only from bar open price
//! - Breach inclusion: Breaching trade included in closing bar
//! - Fixed thresholds: Never recalculated during bar lifetime
//! - Temporal integrity: Guaranteed correct historical simulation
//! - **Cross-file checkpoints**: Seamless continuation across file boundaries (v6.1.0+)
//! - **Arrow export**: Zero-copy streaming to Python via Arrow RecordBatch (v8.0.0+)

pub mod checkpoint;
pub mod errors;
pub mod export_processor; // Export-oriented processor (extracted Phase 2d)
pub mod fixed_point;
pub mod interbar; // Issue #59: Inter-bar microstructure features (lookback window BEFORE bar)
pub mod interbar_math; // Issue #59: Inter-bar math helpers (extracted Phase 2e) - public for profiling/benchmarking
pub mod interbar_types; // Issue #59: Inter-bar type definitions (extracted Phase 2b)
pub mod intrabar; // Issue #59: Intra-bar features (trades WITHIN bar)
pub mod processor;
pub mod timestamp;
pub mod trade; // Trade/DataSource types (extracted Phase 2c)
pub mod types;

// Arrow export (only available with arrow feature)
#[cfg(feature = "arrow")]
pub mod arrow_export;

// Test utilities (only available in test builds or with test-utils feature)
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_data_loader;

/// Feature manifest TOML, embedded at compile time.
/// SSoT for all microstructure feature metadata (Issue #95).
/// Exposed to Python via PyO3 `get_feature_manifest_raw()`.
pub const FEATURE_MANIFEST_TOML: &str = include_str!("../data/feature_manifest.toml");

// Re-export commonly used types
pub use checkpoint::{AnomalySummary, Checkpoint, CheckpointError, PositionVerification};
pub use fixed_point::FixedPoint;
pub use interbar::{InterBarConfig, InterBarFeatures, LookbackMode, TradeHistory, TradeSnapshot};
pub use intrabar::{IntraBarFeatures, compute_intra_bar_features};
pub use processor::{ExportRangeBarProcessor, ProcessingError, RangeBarProcessor};
pub use timestamp::{
    create_aggtrade_with_normalized_timestamp, normalize_timestamp, validate_timestamp,
};
pub use types::{AggTrade, DataSource, RangeBar};

// Arrow export re-exports (only available with arrow feature)
#[cfg(feature = "arrow")]
pub use arrow_export::{
    ArrowImportError,
    aggtrade_schema,
    aggtrades_to_record_batch,
    rangebar_schema,
    rangebar_vec_to_record_batch,
    // Issue #88: Arrow-native input path
    record_batch_to_aggtrades,
};
