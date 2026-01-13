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
pub mod fixed_point;
pub mod processor;
pub mod timestamp;
pub mod types;

// Arrow export (only available with arrow feature)
#[cfg(feature = "arrow")]
pub mod arrow_export;

// Test utilities (only available in test builds or with test-utils feature)
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_data_loader;

// Re-export commonly used types
pub use checkpoint::{AnomalySummary, Checkpoint, CheckpointError, PositionVerification};
pub use fixed_point::FixedPoint;
pub use processor::{ExportRangeBarProcessor, ProcessingError, RangeBarProcessor};
pub use timestamp::{
    create_aggtrade_with_normalized_timestamp, normalize_timestamp, validate_timestamp,
};
pub use types::{AggTrade, DataSource, RangeBar};

// Arrow export re-exports (only available with arrow feature)
#[cfg(feature = "arrow")]
pub use arrow_export::{
    aggtrade_schema, aggtrades_to_record_batch, rangebar_schema, rangebar_vec_to_record_batch,
};
