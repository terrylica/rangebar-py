//! Processing error types
//!
//! Extracted from processor.rs (Phase 2a refactoring)

#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

/// Processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error(
        "Trades not sorted at index {index}: prev=({prev_time}, {prev_id}), curr=({curr_time}, {curr_id})"
    )]
    UnsortedTrades {
        index: usize,
        prev_time: i64,
        prev_id: i64,
        curr_time: i64,
        curr_id: i64,
    },

    #[error("Empty trade data")]
    EmptyData,

    #[error(
        "Invalid threshold: {threshold_decimal_bps} dbps. Valid range: 1-100,000 dbps (0.001%-100%)"
    )]
    InvalidThreshold { threshold_decimal_bps: u32 },
}

#[cfg(feature = "python")]
impl From<ProcessingError> for PyErr {
    fn from(err: ProcessingError) -> PyErr {
        match err {
            ProcessingError::UnsortedTrades {
                index,
                prev_time,
                prev_id,
                curr_time,
                curr_id,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "Trades not sorted at index {}: prev=({}, {}), curr=({}, {})",
                index, prev_time, prev_id, curr_time, curr_id
            )),
            ProcessingError::EmptyData => {
                pyo3::exceptions::PyValueError::new_err("Empty trade data")
            }
            ProcessingError::InvalidThreshold {
                threshold_decimal_bps,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid threshold: {} dbps. Valid range: 1-100,000 dbps (0.001%-100%)",
                threshold_decimal_bps
            )),
        }
    }
}

// Issue #96 Task #87: Test coverage for error Display formatting
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsorted_trades_display() {
        let err = ProcessingError::UnsortedTrades {
            index: 42,
            prev_time: 1000,
            prev_id: 100,
            curr_time: 999,
            curr_id: 101,
        };
        let msg = err.to_string();
        assert!(msg.contains("index 42"));
        assert!(msg.contains("prev=(1000, 100)"));
        assert!(msg.contains("curr=(999, 101)"));
    }

    #[test]
    fn test_empty_data_display() {
        let err = ProcessingError::EmptyData;
        assert_eq!(err.to_string(), "Empty trade data");
    }

    #[test]
    fn test_invalid_threshold_display() {
        let err = ProcessingError::InvalidThreshold {
            threshold_decimal_bps: 0,
        };
        let msg = err.to_string();
        assert!(msg.contains("0 dbps"));
        assert!(msg.contains("Valid range"));
    }

    #[test]
    fn test_invalid_threshold_large_value() {
        let err = ProcessingError::InvalidThreshold {
            threshold_decimal_bps: 999_999,
        };
        assert!(err.to_string().contains("999999 dbps"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        // ProcessingError must be Send+Sync for cross-thread use
        assert_send::<ProcessingError>();
        assert_sync::<ProcessingError>();
    }

    #[test]
    fn test_error_debug_impl() {
        let err = ProcessingError::EmptyData;
        let debug = format!("{:?}", err);
        assert!(debug.contains("EmptyData"));
    }

    #[test]
    fn test_unsorted_trades_boundary_values() {
        let err = ProcessingError::UnsortedTrades {
            index: usize::MAX,
            prev_time: i64::MIN,
            prev_id: i64::MAX,
            curr_time: 0,
            curr_id: 0,
        };
        // Should not panic on extreme values
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }
}
