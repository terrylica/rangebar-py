//! Integration tests for critical algorithm invariants
//!
//! This test suite validates the **Breach Consistency Invariant**:
//! - If `high - open >= threshold` → `close` must equal `high`
//! - If `open - low >= threshold` → `close` must equal `low`
//!
//! **Why This Matters**: Ensures non-lookahead bias and algorithmic correctness
//! across all scenarios (edge cases, large datasets, multiple thresholds).
//!
//! **Reference**: `/docs/rangebar_core_api.md`

use rangebar_core::{FixedPoint, RangeBar, RangeBarProcessor};

/// Breach consistency invariant validator
///
/// Validates that every completed range bar satisfies the breach consistency invariant:
/// - If high breached (high - open >= threshold), close must equal high
/// - If low breached (open - low >= threshold), close must equal low
///
/// # Arguments
///
/// * `bars` - Completed range bars to validate
/// * `threshold_decimal_bps` - Threshold in dbps (250 dbps = 0.25%)
///
/// # Returns
///
/// `Ok(())` if all bars pass invariant checks, `Err(String)` with detailed diagnostics otherwise
fn validate_breach_consistency_invariant(
    bars: &[RangeBar],
    threshold_decimal_bps: u32,
) -> Result<(), String> {
    const BASIS_POINTS_SCALE: i64 = 100_000; // v3.0.0: dbps

    for (i, bar) in bars.iter().enumerate() {
        // Compute thresholds from open (fixed throughout bar lifetime)
        let open_val = bar.open.0;
        let threshold_delta = (open_val * threshold_decimal_bps as i64) / BASIS_POINTS_SCALE;
        let upper_threshold = open_val + threshold_delta;
        let lower_threshold = open_val - threshold_delta;

        let high_val = bar.high.0;
        let low_val = bar.low.0;
        let close_val = bar.close.0;

        // Check if high breached threshold
        let high_breached = high_val >= upper_threshold;
        // Check if low breached threshold
        let low_breached = low_val <= lower_threshold;

        // Invariant: If high breached, close must be at high (upward breach)
        if high_breached && close_val != high_val {
            return Err(format!(
                "Breach Consistency Invariant VIOLATION at bar {}: \
                 High breached (high={} >= upper_threshold={}), \
                 but close={} != high. \
                 Bar: open={}, high={}, low={}, close={}, threshold={} dbps",
                i,
                FixedPoint(high_val),
                FixedPoint(upper_threshold),
                FixedPoint(close_val),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                threshold_decimal_bps
            ));
        }

        // Invariant: If low breached, close must be at low (downward breach)
        if low_breached && close_val != low_val {
            return Err(format!(
                "Breach Consistency Invariant VIOLATION at bar {}: \
                 Low breached (low={} <= lower_threshold={}), \
                 but close={} != low. \
                 Bar: open={}, high={}, low={}, close={}, threshold={} dbps",
                i,
                FixedPoint(low_val),
                FixedPoint(lower_threshold),
                FixedPoint(close_val),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                threshold_decimal_bps
            ));
        }

        // Additional sanity checks
        if high_val < open_val {
            return Err(format!(
                "Sanity check FAILED at bar {}: high < open (high={}, open={})",
                i,
                FixedPoint(high_val),
                FixedPoint(open_val)
            ));
        }

        if low_val > open_val {
            return Err(format!(
                "Sanity check FAILED at bar {}: low > open (low={}, open={})",
                i,
                FixedPoint(low_val),
                FixedPoint(open_val)
            ));
        }

        if high_val < close_val {
            return Err(format!(
                "Sanity check FAILED at bar {}: high < close (high={}, close={})",
                i,
                FixedPoint(high_val),
                FixedPoint(close_val)
            ));
        }

        if low_val > close_val {
            return Err(format!(
                "Sanity check FAILED at bar {}: low > close (low={}, close={})",
                i,
                FixedPoint(low_val),
                FixedPoint(close_val)
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod invariant_tests {
    use super::*;
    use rangebar_core::test_utils::{generators, scenarios, AggTradeBuilder};

    /// Test breach consistency invariant on exact upward breach scenario
    #[test]
    fn test_invariant_exact_breach_upward() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = scenarios::exact_breach_upward(threshold_decimal_bps);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Exact upward breach should satisfy invariant");
    }

    /// Test breach consistency invariant on exact downward breach scenario
    #[test]
    fn test_invariant_exact_breach_downward() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = scenarios::exact_breach_downward(threshold_decimal_bps);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Exact downward breach should satisfy invariant");
    }

    /// Test breach consistency invariant on large price gap scenario
    #[test]
    fn test_invariant_large_gap() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = scenarios::large_gap_sequence();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Large gap scenario should satisfy invariant");
    }

    /// Test breach consistency invariant on single breach sequence
    #[test]
    fn test_invariant_single_breach() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = scenarios::single_breach_sequence(threshold_decimal_bps);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Single breach scenario should satisfy invariant");
    }

    /// Test breach consistency invariant on large random dataset (1M ticks)
    #[test]
    fn test_invariant_massive_realistic_dataset() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_massive_realistic_dataset(1_000_000);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Processed {} trades into {} bars (threshold={} dbps)",
            trades.len(),
            bars.len(),
            threshold_decimal_bps
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Massive realistic dataset should satisfy invariant");
    }

    /// Test breach consistency invariant with multiple threshold values
    #[test]
    fn test_invariant_multiple_thresholds() {
        // Test thresholds: 2 dbps (HFT), 10 dbps, 100 dbps, 250 dbps, 1000 dbps
        let thresholds = vec![2, 10, 100, 250, 1000];

        let trades = generators::create_massive_realistic_dataset(100_000);

        for threshold_decimal_bps in thresholds {
            let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();
            let bars = processor.process_agg_trade_records(&trades).unwrap();

            validate_breach_consistency_invariant(&bars, threshold_decimal_bps).unwrap_or_else(
                |err| {
                    panic!(
                        "Invariant violation at threshold {} dbps: {}",
                        threshold_decimal_bps, err
                    )
                },
            );

            println!(
                "✓ Threshold {} dbps: {} bars generated, all satisfy invariant",
                threshold_decimal_bps,
                bars.len()
            );
        }
    }

    /// Test breach consistency invariant on multi-day boundary dataset
    #[test]
    fn test_invariant_multi_day_boundaries() {
        let threshold_decimal_bps = 250; // 250 dbps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_multi_day_boundary_dataset(7); // 7 days
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Multi-day dataset: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Multi-day boundary dataset should satisfy invariant");
    }

    /// Test breach consistency invariant on volatile market conditions
    #[test]
    fn test_invariant_volatile_conditions() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let base_time = 1609459200000; // 2021-01-01 00:00:00
        let trades = generators::create_volatile_day_data(base_time, 10_000);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Volatile day: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Volatile market conditions should satisfy invariant");
    }

    /// Test breach consistency invariant on stable market conditions
    #[test]
    fn test_invariant_stable_conditions() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let base_time = 1609459200000; // 2021-01-01 00:00:00
        let trades = generators::create_stable_day_data(base_time, 10_000);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!("Stable day: {} trades → {} bars", trades.len(), bars.len());

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Stable market conditions should satisfy invariant");
    }

    /// Test breach consistency invariant on trending market conditions
    #[test]
    fn test_invariant_trending_conditions() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let base_time = 1609459200000; // 2021-01-01 00:00:00
        let trades = generators::create_trending_day_data(base_time, 10_000);
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Trending day: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Trending market conditions should satisfy invariant");
    }

    /// Test breach consistency invariant on high-frequency data
    #[test]
    fn test_invariant_high_frequency() {
        let threshold_decimal_bps = 100; // 10bps = 0.1% (tight for HFT)
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_high_frequency_data(10); // 10ms interval
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "High-frequency (10ms): {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("High-frequency data should satisfy invariant");
    }

    /// Test breach consistency invariant on low-frequency data
    #[test]
    fn test_invariant_low_frequency() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_low_frequency_data(60_000); // 1min interval
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Low-frequency (1min): {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Low-frequency data should satisfy invariant");
    }

    /// Test breach consistency invariant on mixed-frequency data
    #[test]
    fn test_invariant_mixed_frequency() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_mixed_frequency_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Mixed-frequency: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Mixed-frequency data should satisfy invariant");
    }

    /// Test breach consistency invariant on rapid threshold hits
    #[test]
    fn test_invariant_rapid_threshold_hits() {
        let threshold_decimal_bps = 50; // 5bps = 0.05% (tight threshold)
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_rapid_threshold_hit_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Rapid threshold hits: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Rapid threshold hits should satisfy invariant");
    }

    /// Test breach consistency invariant on precision limit edge cases
    #[test]
    fn test_invariant_precision_limits() {
        let threshold_decimal_bps = 1; // 0.1bps = 0.001% (minimum threshold)
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_precision_limit_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Precision limits (0.1bps): {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Precision limit data should satisfy invariant");
    }

    /// Test breach consistency invariant on volume extreme edge cases
    #[test]
    fn test_invariant_volume_extremes() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_volume_extreme_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Volume extremes: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Volume extreme data should satisfy invariant");
    }

    /// Test breach consistency invariant on timestamp edge cases
    #[test]
    fn test_invariant_timestamp_edges() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_timestamp_edge_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Timestamp edges: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Timestamp edge data should satisfy invariant");
    }

    /// Test breach consistency invariant on floating-point stress data
    #[test]
    fn test_invariant_floating_point_stress() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        let trades = generators::create_floating_point_stress_data();
        let bars = processor.process_agg_trade_records(&trades).unwrap();

        println!(
            "Floating-point stress: {} trades → {} bars",
            trades.len(),
            bars.len()
        );

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Floating-point stress data should satisfy invariant");
    }

    /// Test breach consistency invariant on custom oscillating sequence
    #[test]
    fn test_invariant_custom_oscillation() {
        let threshold_decimal_bps = 250; // 25bps = 0.25%
        let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();

        // Create oscillating sequence that repeatedly approaches but doesn't breach
        let trades = AggTradeBuilder::new()
            .with_base_price(50000.0)
            .with_base_timestamp(1609459200000)
            .add_trade(1, 1.0, 1000) // 50000.00
            .add_trade(2, 1.002, 2000) // +0.2% (no breach at 25bps threshold)
            .add_trade(3, 0.998, 3000) // -0.4% from #2
            .add_trade(4, 1.002, 4000) // +0.4% from #3
            .add_trade(5, 0.998, 5000) // -0.4% from #4
            .add_trade(6, 1.0025, 6000) // +0.25% - EXACT BREACH
            .build();

        let bars = processor.process_agg_trade_records(&trades).unwrap();

        assert_eq!(bars.len(), 1, "Should create exactly 1 bar on exact breach");

        validate_breach_consistency_invariant(&bars, threshold_decimal_bps)
            .expect("Custom oscillation should satisfy invariant");
    }

    /// Test breach consistency invariant on boundary threshold values
    #[test]
    fn test_invariant_boundary_thresholds() {
        // Test minimum and maximum valid thresholds
        let boundary_thresholds = vec![
            1,       // 0.1bps = 0.001% (minimum)
            100_000, // 10,000bps = 100% (maximum)
        ];

        let trades = generators::create_massive_realistic_dataset(10_000);

        for threshold_decimal_bps in boundary_thresholds {
            let mut processor = RangeBarProcessor::new(threshold_decimal_bps).unwrap();
            let bars = processor.process_agg_trade_records(&trades).unwrap();

            validate_breach_consistency_invariant(&bars, threshold_decimal_bps).unwrap_or_else(
                |err| {
                    panic!(
                        "Invariant violation at boundary threshold {} dbps: {}",
                        threshold_decimal_bps, err
                    )
                },
            );

            println!(
                "✓ Boundary threshold {} dbps: {} bars, all valid",
                threshold_decimal_bps,
                bars.len()
            );
        }
    }
}
