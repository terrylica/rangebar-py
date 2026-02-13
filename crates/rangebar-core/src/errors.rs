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
