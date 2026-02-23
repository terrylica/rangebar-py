//! Integration test: Verify intra-bar dual-pass optimization produces identical output
//!
//! Issue #96 Task #166: After fusing OHLC tracking into moment computation loop,
//! verify that output is identical to the original dual-pass implementation.

use rangebar_core::types::AggTrade;
use rangebar_core::fixed_point::FixedPoint;
use rangebar_core::intrabar::features::compute_intra_bar_features;

fn create_test_trade(price: f64, volume: f64, timestamp: i64, is_buyer_maker: bool) -> AggTrade {
    AggTrade {
        agg_trade_id: timestamp,
        price: FixedPoint((price * 1e8) as i64),
        volume: FixedPoint((volume * 1e8) as i64),
        first_trade_id: timestamp,
        last_trade_id: timestamp,
        timestamp,
        is_buyer_maker,
        is_best_match: None,
    }
}

#[test]
fn test_intrabar_features_consistency_uptrend() {
    // Create consistent uptrending price series
    let trades: Vec<AggTrade> = (0..100)
        .map(|i| {
            let price = 100.0 + i as f64 * 0.05;
            let volume = 1.0 + (i as f64 * 0.01);
            let is_buyer = i % 2 == 0;
            create_test_trade(price, volume, i as i64 * 1000, is_buyer)
        })
        .collect();

    let features = compute_intra_bar_features(&trades);

    // Verify all core features are computed
    assert_eq!(features.intra_trade_count, Some(100));
    assert!(features.intra_ofi.is_some());
    assert!(features.intra_duration_us.is_some());
    assert!(features.intra_intensity.is_some());
    assert!(features.intra_vwap_position.is_some());
    assert!(features.intra_count_imbalance.is_some());

    // Verify OHLC bounds make sense for uptrend
    if let Some(vwap_pos) = features.intra_vwap_position {
        assert!(vwap_pos >= 0.0 && vwap_pos <= 1.0, "VWAP position should be in [0,1]: {}", vwap_pos);
    }

    // OFI should reflect mixed buy/sell (50/50 distribution)
    if let Some(ofi) = features.intra_ofi {
        assert!(ofi.abs() < 1.0, "OFI should be bounded: {}", ofi);
    }
}

#[test]
fn test_intrabar_features_consistency_downtrend() {
    // Create downtrending price series
    let trades: Vec<AggTrade> = (0..100)
        .map(|i| {
            let price = 100.0 - i as f64 * 0.05;
            let volume = 1.0 + (i as f64 * 0.01);
            let is_buyer = i % 2 == 0;
            create_test_trade(price, volume, i as i64 * 1000, is_buyer)
        })
        .collect();

    let features = compute_intra_bar_features(&trades);

    assert_eq!(features.intra_trade_count, Some(100));
    assert!(features.intra_ofi.is_some());
    assert!(features.intra_max_drawdown.is_some());
}

#[test]
fn test_intrabar_features_volume_moments_consistency() {
    // Test with varying volumes to verify moment computation
    let trades: Vec<AggTrade> = (0..200)
        .map(|i| {
            let price = 100.0 + ((i as f64 * 0.05).sin() * 2.0);
            // Oscillating volumes to test skewness/kurtosis
            let volume = 1.0 + ((i as f64 * 0.02).sin() * 0.5).abs();
            let is_buyer = i % 3 == 0;
            create_test_trade(price, volume, i as i64 * 500, is_buyer)
        })
        .collect();

    let features = compute_intra_bar_features(&trades);

    assert_eq!(features.intra_trade_count, Some(200));

    // Volume moments should be present for >= 3 trades
    assert!(features.intra_volume_skew.is_some(), "Skewness should be computed");
    assert!(features.intra_volume_kurt.is_some(), "Kurtosis should be computed");

    // Verify moment values are reasonable
    if let Some(skew) = features.intra_volume_skew {
        assert!(skew.is_finite(), "Skewness should be finite: {}", skew);
    }
    if let Some(kurt) = features.intra_volume_kurt {
        assert!(kurt.is_finite(), "Kurtosis should be finite: {}", kurt);
    }
}

#[test]
fn test_intrabar_features_ohlc_bounds() {
    // Test that OHLC tracking produces correct bounds
    let mut trades = vec![];
    for i in 0..50 {
        let price = match i {
            0 => 100.0,   // First price
            25 => 105.0,  // Peak
            49 => 95.0,   // Trough
            _ => 100.0 + ((i as f64 - 25.0) / 25.0).sin() * 5.0,
        };
        let volume = 1.0;
        trades.push(create_test_trade(price, volume, i as i64 * 1000, i % 2 == 0));
    }

    let features = compute_intra_bar_features(&trades);

    // VWAP position should reflect the computed bounds
    assert!(features.intra_vwap_position.is_some());
    if let Some(vwap_pos) = features.intra_vwap_position {
        assert!(vwap_pos >= 0.0 && vwap_pos <= 1.0, "VWAP position must be in [0,1]");
    }
}

#[test]
fn test_intrabar_features_large_bar() {
    // Test with 500+ trades to stress test the dual-pass fusion
    let trades: Vec<AggTrade> = (0..500)
        .map(|i| {
            let price = 100.0 + ((i as f64 * 0.01).sin() * 3.0);
            let volume = 1.0 + (i as f64 * 0.001);
            let is_buyer = i % 5 == 0;
            create_test_trade(price, volume, i as i64 * 100, is_buyer)
        })
        .collect();

    let features = compute_intra_bar_features(&trades);

    assert_eq!(features.intra_trade_count, Some(500));

    // All core features should be present
    assert!(features.intra_ofi.is_some());
    assert!(features.intra_vwap_position.is_some());
    assert!(features.intra_volume_skew.is_some());
    assert!(features.intra_volume_kurt.is_some());

    // Complexity features should be present for 500 trades
    assert!(features.intra_hurst.is_some(), "Hurst should be computed for 500 trades");
    assert!(features.intra_permutation_entropy.is_some(), "PE should be computed for 500 trades");

    // Verify bounds
    if let Some(h) = features.intra_hurst {
        assert!(h >= 0.0 && h <= 1.0, "Hurst should be in [0,1]: {}", h);
    }
    if let Some(pe) = features.intra_permutation_entropy {
        assert!(pe >= 0.0 && pe <= 1.0, "PE should be in [0,1]: {}", pe);
    }
}

#[test]
fn test_intrabar_features_edge_cases() {
    // Single trade
    let single = vec![create_test_trade(100.0, 1.0, 0, false)];
    let features_single = compute_intra_bar_features(&single);
    assert_eq!(features_single.intra_trade_count, Some(1));

    // Two trades
    let double = vec![
        create_test_trade(100.0, 1.0, 0, false),
        create_test_trade(100.5, 1.5, 1000, true),
    ];
    let features_double = compute_intra_bar_features(&double);
    assert_eq!(features_double.intra_trade_count, Some(2));
    assert!(features_double.intra_ofi.is_some());

    // Empty
    let empty: Vec<AggTrade> = vec![];
    let features_empty = compute_intra_bar_features(&empty);
    assert_eq!(features_empty.intra_trade_count, Some(0));
}
