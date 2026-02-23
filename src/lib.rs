// =============================================================================
// PyO3 Module Entry Point
// =============================================================================
// This is the thin orchestrator for the rangebar Python extension module.
// All implementation lives in domain-specific submodules.
// Issue #94: Refactored from monolithic 2999-line file.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rangebar_core::{
    AggTrade, AnomalySummary, Checkpoint, CheckpointError, FixedPoint, InterBarConfig, LookbackMode,
    PositionVerification, RangeBar, RangeBarProcessor,
};

// Arrow export support (feature-gated)
#[cfg(feature = "arrow-export")]
use pyo3_arrow::PyRecordBatch;
// Issue #88: Arrow-native input path
#[cfg(feature = "arrow-export")]
use rangebar_core::{
    aggtrades_to_record_batch, record_batch_to_aggtrades, rangebar_vec_to_record_batch,
};

#[cfg(feature = "data-providers")]
use rangebar_providers::exness::{
    ExnessInstrument, ExnessRangeBar, ExnessRangeBarBuilder, ExnessTick, SpreadStats,
    ValidationStrictness,
};

#[cfg(feature = "data-providers")]
use rangebar_providers::binance::{HistoricalDataLoader, IntraDayChunkIterator};

// =============================================================================
// Domain Modules
// =============================================================================

mod helpers;
use helpers::*;

mod core_bindings;
use core_bindings::*;

#[cfg(feature = "data-providers")]
mod exness_bindings;

#[cfg(feature = "arrow-export")]
mod arrow_bindings;

#[cfg(feature = "data-providers")]
mod binance_bindings;

#[cfg(feature = "streaming")]
mod streaming_bindings;

mod thread_pool_init;

// =============================================================================
// Python Module Registration
// =============================================================================

/// Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize rayon global thread pool (Task #90, Issue #96)
    // This eliminates lazy initialization overhead on first parallel operation
    let _ = thread_pool_init::initialize_rayon_pool();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRangeBarProcessor>()?;
    m.add_class::<PyPositionVerification>()?;
    m.add_function(wrap_pyfunction!(
        core_bindings::get_feature_manifest_raw,
        m
    )?)?;

    // Add Arrow export functions if feature enabled
    #[cfg(feature = "arrow-export")]
    {
        m.add_function(wrap_pyfunction!(arrow_bindings::bars_to_arrow, m)?)?;
        m.add_function(wrap_pyfunction!(arrow_bindings::trades_to_arrow, m)?)?;
    }

    // Add Exness classes if feature enabled
    #[cfg(feature = "data-providers")]
    {
        m.add_class::<exness_bindings::PyExnessInstrument>()?;
        m.add_class::<exness_bindings::PyValidationStrictness>()?;
        m.add_class::<exness_bindings::PyExnessRangeBarBuilder>()?;
        m.add_class::<binance_bindings::PyMarketType>()?;
        m.add_class::<binance_bindings::PyBinanceTradeStream>()?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::fetch_binance_aggtrades,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::stream_binance_trades,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            binance_bindings::fetch_aggtrades_rest,
            m
        )?)?;

        // Phase 3: Arrow-native stream (requires both data-providers and arrow-export)
        #[cfg(feature = "arrow-export")]
        {
            m.add_class::<binance_bindings::PyBinanceTradeStreamArrow>()?;
            m.add_function(wrap_pyfunction!(
                binance_bindings::stream_binance_trades_arrow,
                m
            )?)?;
        }
    }

    // Add streaming classes if feature enabled
    // ADR: docs/adr/2026-01-31-realtime-streaming-api.md
    #[cfg(feature = "streaming")]
    {
        m.add_class::<streaming_bindings::PyBinanceLiveStream>()?;
        m.add_class::<streaming_bindings::PyLiveBarEngine>()?;
        m.add_class::<streaming_bindings::PyStreamingConfig>()?;
        m.add_class::<streaming_bindings::PyStreamingMetrics>()?;
        m.add_class::<streaming_bindings::PyStreamingRangeBarProcessor>()?;
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_to_fixed_point() {
        let fp = f64_to_fixed_point(42000.12345678);
        assert_eq!(fp.to_f64(), 42000.12345678);

        let fp2 = f64_to_fixed_point(1.5);
        assert_eq!(fp2.to_f64(), 1.5);
    }

    #[test]
    fn test_processor_creation() {
        let processor = PyRangeBarProcessor::new(250, None, true, None, false, None);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().threshold_decimal_bps, 250);
    }

    #[test]
    fn test_invalid_threshold() {
        let processor = PyRangeBarProcessor::new(0, None, true, None, false, None);
        assert!(processor.is_err());

        let processor = PyRangeBarProcessor::new(200_000, None, true, None, false, None);
        assert!(processor.is_err());
    }

    #[test]
    fn test_f64_to_fixed_point_extremes() {
        // Test zero
        let fp_zero = f64_to_fixed_point(0.0);
        assert_eq!(fp_zero.to_f64(), 0.0);

        // Test negative values
        let fp_neg = f64_to_fixed_point(-42000.12345678);
        assert_eq!(fp_neg.to_f64(), -42000.12345678);

        // Test very small values (precision boundary)
        let fp_small = f64_to_fixed_point(0.00000001);
        assert_eq!(fp_small.to_f64(), 0.00000001);

        // Test very large values
        let fp_large = f64_to_fixed_point(1_000_000.12345678);
        assert_eq!(fp_large.to_f64(), 1_000_000.12345678);

        // Test maximum precision (8 decimal places)
        let fp_precision = f64_to_fixed_point(123.45678901);
        // Should round to 8 decimal places
        let rounded = fp_precision.to_f64();
        assert!((rounded - 123.45678901).abs() < 0.000001);
    }

    #[test]
    fn test_processor_boundary_thresholds() {
        // Test minimum valid threshold (1 = 0.1 basis points)
        let processor_min = PyRangeBarProcessor::new(1, None, true, None, false, None);
        assert!(processor_min.is_ok());
        assert_eq!(processor_min.unwrap().threshold_decimal_bps, 1);

        // Test maximum valid threshold (100_000 = 10,000 basis points = 100%)
        let processor_max = PyRangeBarProcessor::new(100_000, None, true, None, false, None);
        assert!(processor_max.is_ok());
        assert_eq!(processor_max.unwrap().threshold_decimal_bps, 100_000);

        // Test common valid thresholds
        for threshold in [10, 100, 250, 500, 1000, 10_000] {
            let processor = PyRangeBarProcessor::new(threshold, None, true, None, false, None);
            assert!(processor.is_ok(), "Threshold {} should be valid", threshold);
        }
    }
}
