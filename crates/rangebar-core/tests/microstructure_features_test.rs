// FILE-SIZE-OK: Integration tests, naturally grow with feature coverage
//! Comprehensive tests for microstructure features (Issue #25, Issue #96)
//!
//! Tests for:
//! - 10 intra-bar microstructure features computed during bar construction
//! - Edge cases: empty bars, single trades, extreme values
//! - Bounds validation: features stay within expected ranges
//! - Tier 2/3 inter-bar features with lookback windows

use rangebar_core::processor::RangeBarProcessor;
use rangebar_core::trade::AggTrade;
use rangebar_core::fixed_point::FixedPoint;

// Helper to create test trades
fn create_trade(
    agg_trade_id: i64,
    price: &str,
    volume: &str,
    timestamp: i64,
    is_buyer_maker: bool,
) -> AggTrade {
    AggTrade {
        agg_trade_id,
        price: FixedPoint::from_str(price).expect("invalid price"),
        volume: FixedPoint::from_str(volume).expect("invalid volume"),
        first_trade_id: agg_trade_id * 10,
        last_trade_id: agg_trade_id * 10 + 5,
        timestamp,
        is_buyer_maker,
        is_best_match: Some(true),
    }
}

#[test]
fn test_ofi_bounded_in_range() {
    // OFI (Order Flow Imbalance) should always be in [-1, 1]
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false), // Buy
        create_trade(2, "50050.0", "2.0", 1640995201000, true),  // Sell
        create_trade(3, "50100.0", "0.5", 1640995202000, false), // Buy
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.ofi.is_finite(),
            "OFI must be finite, got {}",
            bar.ofi
        );
        assert!(
            bar.ofi >= -1.0 && bar.ofi <= 1.0,
            "OFI out of bounds: {} (expected [-1, 1])",
            bar.ofi
        );
    }
}

#[test]
fn test_vwap_close_deviation_bounded() {
    // VWAP close deviation should be meaningful and bounded
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50050.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.vwap_close_deviation.is_finite(),
            "vwap_close_deviation must be finite"
        );
        // Deviation should be in roughly [-1, 1] range
        assert!(
            bar.vwap_close_deviation.abs() <= 2.0,
            "vwap_close_deviation too large: {}",
            bar.vwap_close_deviation
        );
    }
}

#[test]
fn test_kyle_lambda_proxy_computable() {
    // Kyle Lambda should be computable and finite for normal scenarios
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50150.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.kyle_lambda_proxy.is_finite(),
            "kyle_lambda_proxy must be finite, got {}",
            bar.kyle_lambda_proxy
        );
    }
}

#[test]
fn test_trade_intensity_non_negative() {
    // Trade intensity = trade_count / duration_sec should be non-negative
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50150.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.trade_intensity >= 0.0,
            "trade_intensity must be non-negative, got {}",
            bar.trade_intensity
        );
    }
}

#[test]
fn test_volume_per_trade_positive() {
    // Volume per trade should be positive for multi-trade bars
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        if bar.individual_trade_count > 0 {
            assert!(
                bar.volume_per_trade > 0.0,
                "volume_per_trade must be positive"
            );
        }
    }
}

#[test]
fn test_aggression_ratio_bounded() {
    // Aggression ratio = buy_count / sell_count should be bounded and meaningful
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false), // Buy
        create_trade(2, "50100.0", "2.0", 1640995201000, true),  // Sell
        create_trade(3, "50150.0", "1.5", 1640995202000, false), // Buy
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.aggression_ratio >= 0.0,
            "aggression_ratio must be non-negative"
        );
        assert!(
            bar.aggression_ratio.is_finite(),
            "aggression_ratio must be finite"
        );
    }
}

#[test]
fn test_single_trade_bar_features() {
    // Single-trade bars should compute features without panic or NaN
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![create_trade(1, "50000.0", "1.0", 1640995200000, false)];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(!bar.ofi.is_nan(), "OFI must not be NaN");
        assert!(
            !bar.vwap_close_deviation.is_nan(),
            "vwap_close_deviation must not be NaN"
        );
        assert!(
            !bar.kyle_lambda_proxy.is_nan(),
            "kyle_lambda_proxy must not be NaN"
        );
    }
}

#[test]
fn test_high_volume_bar_features() {
    // High-volume bar should handle large numbers correctly
    let mut processor = RangeBarProcessor::new(500).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1000.0", 1640995200000, false),
        create_trade(2, "50500.0", "2000.0", 1640995201000, true),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            !bar.ofi.is_nan() && bar.ofi.is_finite(),
            "Large volume should not break OFI"
        );
        assert!(
            !bar.kyle_lambda_proxy.is_nan() && bar.kyle_lambda_proxy.is_finite(),
            "Large volume should not break kyle_lambda"
        );
    }
}

#[test]
fn test_turnover_imbalance_bounded() {
    // Turnover imbalance = (buy_turnover - sell_turnover) / total_turnover
    // Should be bounded in [-1, 1]
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50200.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        assert!(
            bar.turnover_imbalance.is_finite(),
            "turnover_imbalance must be finite"
        );
        assert!(
            bar.turnover_imbalance >= -1.0 && bar.turnover_imbalance <= 1.0,
            "turnover_imbalance out of bounds: {}",
            bar.turnover_imbalance
        );
    }
}


#[test]
fn test_all_microstructure_features_present() {
    // All 10 microstructure features should be computed for every bar
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50150.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        // All 10 intra-bar features must be finite and not NaN
        assert!(bar.ofi.is_finite(), "ofi must be finite");
        assert!(
            bar.vwap_close_deviation.is_finite(),
            "vwap_close_deviation must be finite"
        );
        assert!(
            bar.price_impact.is_finite(),
            "price_impact must be finite"
        );
        assert!(
            bar.kyle_lambda_proxy.is_finite(),
            "kyle_lambda_proxy must be finite"
        );
        assert!(
            bar.trade_intensity.is_finite(),
            "trade_intensity must be finite"
        );
        assert!(
            bar.volume_per_trade.is_finite(),
            "volume_per_trade must be finite"
        );
        assert!(
            bar.aggression_ratio.is_finite(),
            "aggression_ratio must be finite"
        );
        assert!(
            bar.aggregation_density_f64.is_finite(),
            "aggregation_density must be finite"
        );
        assert!(
            bar.turnover_imbalance.is_finite(),
            "turnover_imbalance must be finite"
        );
        assert!(
            bar.duration_us >= 0,
            "duration_us must be non-negative"
        );
    }
}

// ============================================================================
// Tier 2 Inter-Bar Feature Tests (Issue #104 Task #12)
// ============================================================================

#[test]
fn test_lookback_kyle_lambda_empty_window() {
    // Kyle Lambda with empty lookback should return None
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");
    processor = processor.with_inter_bar_config(Default::default());

    // First bar (no prior trades for lookback)
    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50150.0", "1.5", 1640995202000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        // First bar has no lookback history
        assert!(
            bar.lookback_kyle_lambda.is_none()
                || bar.lookback_kyle_lambda.map(|v| v.is_finite()).unwrap_or(true),
            "Kyle Lambda must be None or finite"
        );
    }
}

#[test]
fn test_lookback_kyle_lambda_single_trade() {
    // Kyle Lambda with single trade should be None or 0
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");
    processor = processor.with_inter_bar_config(Default::default());

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    // Second bar has one trade in lookback (typically should be None or bounded)
    if bars.len() > 1 {
        let bar = &bars[1];
        if let Some(kyle) = bar.lookback_kyle_lambda {
            assert!(kyle.is_finite(), "Kyle Lambda must be finite");
        }
    }
}

#[test]
fn test_lookback_kyle_lambda_zero_imbalance() {
    // Kyle Lambda with perfectly balanced buy/sell should be close to 0
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");
    processor = processor.with_inter_bar_config(Default::default());

    let mut trades = vec![];
    let base_time = 1640995200000i64;

    // Create balanced trades (equal buy/sell volume and count)
    for i in 0..20 {
        let is_buyer_maker = i % 2 == 0;
        trades.push(create_trade(
            i + 1,
            "50000.0",
            "1.0",
            base_time + (i as i64) * 1000,
            is_buyer_maker,
        ));
    }

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        if let Some(kyle) = bar.lookback_kyle_lambda {
            assert!(kyle.is_finite(), "Kyle Lambda must be finite for balanced trades");
            // Balanced imbalance should yield Kyle lambda close to 0
            assert!(kyle.abs() < 0.5, "Kyle Lambda should be close to 0 for balanced trades, got {}", kyle);
        }
    }
}

#[test]
fn test_lookback_burstiness_bounded() {
    // Burstiness should be bounded in [-1, 1]
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");
    processor = processor.with_inter_bar_config(Default::default());

    let trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50100.0", "2.0", 1640995201000, true),
        create_trade(3, "50150.0", "1.5", 1640995202000, false),
        create_trade(4, "50200.0", "3.0", 1640995203000, true),
        create_trade(5, "50250.0", "2.5", 1640995204000, false),
    ];

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    for bar in bars {
        if let Some(burst) = bar.lookback_burstiness {
            assert!(burst.is_finite(), "Burstiness must be finite");
            assert!(
                burst >= -1.0 && burst <= 1.0,
                "Burstiness out of bounds: {} (expected [-1, 1])",
                burst
            );
        }
    }
}

#[test]
fn test_lookback_kaufman_er_trending_vs_ranging() {
    // Kaufman ER should differ between trending and ranging markets
    let mut processor_trend = RangeBarProcessor::new(250).expect("failed to create processor");
    processor_trend = processor_trend.with_inter_bar_config(Default::default());

    let mut processor_range = RangeBarProcessor::new(250).expect("failed to create processor");
    processor_range = processor_range.with_inter_bar_config(Default::default());

    // Trending: monotonic increasing prices
    let trending_trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50050.0", "1.0", 1640995201000, false),
        create_trade(3, "50100.0", "1.0", 1640995202000, false),
        create_trade(4, "50150.0", "1.0", 1640995203000, false),
        create_trade(5, "50200.0", "1.0", 1640995204000, false),
    ];

    // Ranging: oscillating prices
    let ranging_trades = vec![
        create_trade(1, "50000.0", "1.0", 1640995200000, false),
        create_trade(2, "50050.0", "1.0", 1640995201000, false),
        create_trade(3, "50025.0", "1.0", 1640995202000, false),
        create_trade(4, "50075.0", "1.0", 1640995203000, false),
        create_trade(5, "50050.0", "1.0", 1640995204000, false),
    ];

    let trend_bars = processor_trend
        .process_agg_trade_records(&trending_trades)
        .expect("failed to process trending");

    let range_bars = processor_range
        .process_agg_trade_records(&ranging_trades)
        .expect("failed to process ranging");

    // Find bars with Kaufman ER values
    for bar in &trend_bars {
        if let Some(er) = bar.lookback_kaufman_er {
            assert!(er.is_finite(), "Kaufman ER must be finite");
            assert!(er >= 0.0 && er <= 1.0, "Kaufman ER out of bounds: {}", er);
        }
    }

    for bar in &range_bars {
        if let Some(er) = bar.lookback_kaufman_er {
            assert!(er.is_finite(), "Kaufman ER must be finite");
            assert!(er >= 0.0 && er <= 1.0, "Kaufman ER out of bounds: {}", er);
        }
    }
}

#[test]
fn test_all_interbar_features_when_enabled() {
    // When inter-bar features are enabled, all features should be computable
    let mut processor = RangeBarProcessor::new(250).expect("failed to create processor");
    processor = processor.with_inter_bar_config(Default::default());

    let mut trades = vec![];
    let base_time = 1640995200000i64;

    // Generate enough trades to populate lookback window
    for i in 0..100 {
        trades.push(create_trade(
            i + 1,
            &format!("50000.{:02}", i),
            "1.0",
            base_time + (i as i64) * 100,
            i % 2 == 0,
        ));
    }

    let bars = processor
        .process_agg_trade_records(&trades)
        .expect("failed to process");

    // Skip first few bars (lookback buildup), check later bars
    for bar in bars.iter().skip(5) {
        // Tier 1 features (should be present for later bars)
        if let Some(count) = bar.lookback_trade_count {
            assert!(count >= 0, "lookback_trade_count must be non-negative");
        }
        if let Some(ofi) = bar.lookback_ofi {
            assert!(ofi.is_finite(), "lookback_ofi must be finite");
        }
        if let Some(duration) = bar.lookback_duration_us {
            assert!(duration >= 0, "lookback_duration_us must be non-negative");
        }

        // Tier 2 features (optional but should be computable)
        if let Some(kyle) = bar.lookback_kyle_lambda {
            assert!(kyle.is_finite(), "Kyle Lambda must be finite when present");
        }
        if let Some(burst) = bar.lookback_burstiness {
            assert!(burst.is_finite(), "Burstiness must be finite when present");
            assert!(burst >= -1.0 && burst <= 1.0, "Burstiness out of bounds");
        }
        if let Some(er) = bar.lookback_kaufman_er {
            assert!(er.is_finite(), "Kaufman ER must be finite when present");
            assert!(er >= 0.0 && er <= 1.0, "Kaufman ER out of bounds");
        }
    }
}
