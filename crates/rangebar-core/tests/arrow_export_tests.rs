//! Tests for arrow_export.rs - Arrow RecordBatch conversion
//!
//! Extracted from crates/rangebar-core/src/arrow_export.rs (Phase 1b refactoring)
//! Also fixes pre-existing compile errors: FixedPoint → i128 for volume fields (Issue #88)
// FILE-SIZE-OK: extracted test block, not a source module

#![cfg(feature = "arrow")]

use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use rangebar_core::arrow_export::{
    ArrowImportError, aggtrades_to_record_batch, rangebar_vec_to_record_batch,
    record_batch_to_aggtrades,
};
use rangebar_core::fixed_point::{FixedPoint, SCALE};
use rangebar_core::types::{AggTrade, DataSource, RangeBar};

fn create_test_bar() -> RangeBar {
    RangeBar {
        open_time: 1640995200000000,
        close_time: 1640995201000000,
        open: FixedPoint::from_str("50000.0").unwrap(),
        high: FixedPoint::from_str("50100.0").unwrap(),
        low: FixedPoint::from_str("49900.0").unwrap(),
        close: FixedPoint::from_str("50050.0").unwrap(),
        // Issue #88: volume fields are i128, use FixedPoint.0 as i128
        volume: FixedPoint::from_str("10.5").unwrap().0 as i128,
        turnover: 525_250_000_000_000_i128,
        individual_trade_count: 100,
        agg_record_count: 10,
        first_trade_id: 1,
        last_trade_id: 100,
        first_agg_trade_id: 1000, // Issue #72
        last_agg_trade_id: 1009,  // Issue #72
        data_source: DataSource::BinanceFuturesUM,
        // Issue #88: volume fields are i128
        buy_volume: FixedPoint::from_str("6.0").unwrap().0 as i128,
        sell_volume: FixedPoint::from_str("4.5").unwrap().0 as i128,
        buy_trade_count: 60,
        sell_trade_count: 40,
        vwap: FixedPoint::from_str("50025.0").unwrap(),
        buy_turnover: 300_150_000_000_000_i128,
        sell_turnover: 225_100_000_000_000_i128,
        duration_us: 1_000_000,
        ofi: 0.142857,
        vwap_close_deviation: 0.125,
        price_impact: 0.000476,
        kyle_lambda_proxy: 0.007,
        trade_intensity: 100.0,
        volume_per_trade: 0.105,
        aggression_ratio: 1.5,
        aggregation_density_f64: 10.0,
        turnover_imbalance: 0.142857,
        // Inter-bar features (Issue #59) - test defaults
        lookback_trade_count: None,
        lookback_ofi: None,
        lookback_duration_us: None,
        lookback_intensity: None,
        lookback_vwap_raw: None,
        lookback_vwap_position: None,
        lookback_count_imbalance: None,
        lookback_kyle_lambda: None,
        lookback_burstiness: None,
        lookback_volume_skew: None,
        lookback_volume_kurt: None,
        lookback_price_range: None,
        lookback_kaufman_er: None,
        lookback_garman_klass_vol: None,
        lookback_hurst: None,
        lookback_permutation_entropy: None,
        // Intra-bar features (Issue #59) - test defaults
        intra_bull_epoch_density: None,
        intra_bear_epoch_density: None,
        intra_bull_excess_gain: None,
        intra_bear_excess_gain: None,
        intra_bull_cv: None,
        intra_bear_cv: None,
        intra_max_drawdown: None,
        intra_max_runup: None,
        intra_trade_count: None,
        intra_ofi: None,
        intra_duration_us: None,
        intra_intensity: None,
        intra_vwap_position: None,
        intra_count_imbalance: None,
        intra_kyle_lambda: None,
        intra_burstiness: None,
        intra_volume_skew: None,
        intra_volume_kurt: None,
        intra_kaufman_er: None,
        intra_garman_klass_vol: None,
        intra_hurst: None,
        intra_permutation_entropy: None,
    }
}

fn create_test_trade() -> AggTrade {
    AggTrade {
        agg_trade_id: 12345,
        price: FixedPoint::from_str("50000.12345678").unwrap(),
        volume: FixedPoint::from_str("1.5").unwrap(),
        first_trade_id: 100,
        last_trade_id: 102,
        timestamp: 1640995200000000,
        is_buyer_maker: false,
        is_best_match: Some(true),
    }
}

#[test]
fn test_rangebar_to_record_batch_single() {
    let bar = create_test_bar();
    let batch = rangebar_vec_to_record_batch(&[bar]);

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields

    // Verify schema
    assert_eq!(batch.schema().field(0).name(), "open_time");
    assert_eq!(batch.schema().field(31).name(), "turnover_imbalance");
}

#[test]
fn test_rangebar_to_record_batch_multiple() {
    let bars: Vec<RangeBar> = (0..1000).map(|_| create_test_bar()).collect();
    let batch = rangebar_vec_to_record_batch(&bars);

    assert_eq!(batch.num_rows(), 1000);
    assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields
}

#[test]
fn test_rangebar_to_record_batch_empty() {
    let batch = rangebar_vec_to_record_batch(&[]);
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(batch.num_columns(), 32); // Issue #72: +2 for agg_trade_id fields
}

#[test]
fn test_aggtrade_to_record_batch() {
    let trade = create_test_trade();
    let batch = aggtrades_to_record_batch(&[trade]);

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 8);

    // Verify schema
    assert_eq!(batch.schema().field(0).name(), "agg_trade_id");
    assert_eq!(batch.schema().field(7).name(), "is_best_match");
}

#[test]
fn test_data_source_encoding() {
    let mut bar = create_test_bar();
    bar.data_source = DataSource::BinanceSpot;
    let batch = rangebar_vec_to_record_batch(&[bar]);

    // data_source is at index 14 (after Issue #72 added first/last_agg_trade_id at 12/13)
    let data_source_col = batch
        .column(14)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(data_source_col.value(0), "BinanceSpot");
}

// Issue #88: Arrow import tests

#[test]
fn test_record_batch_to_aggtrades_roundtrip() {
    // Export → Import roundtrip: fields must be bit-exact
    let original = vec![create_test_trade()];
    let batch = aggtrades_to_record_batch(&original);
    let imported = record_batch_to_aggtrades(&batch, false).unwrap();

    assert_eq!(imported.len(), 1);
    let t = &imported[0];
    let o = &original[0];

    assert_eq!(t.agg_trade_id, o.agg_trade_id);
    // aggtrades_to_record_batch exports price as f64 via to_f64(),
    // record_batch_to_aggtrades imports via (f64 * SCALE).round() as i64.
    // Roundtrip preserves exact value for 8-decimal prices.
    assert_eq!(t.price.0, o.price.0);
    assert_eq!(t.volume.0, o.volume.0);
    assert_eq!(t.first_trade_id, o.first_trade_id);
    assert_eq!(t.last_trade_id, o.last_trade_id);
    // aggtrades_to_record_batch exports timestamp in microseconds,
    // record_batch_to_aggtrades(false) expects milliseconds and converts via * 1000.
    // So the roundtrip with false changes the value: exported μs → imported as ms → * 1000 = μs * 1000.
    // For a true roundtrip, use timestamp_is_microseconds=true — see
    // test_record_batch_to_aggtrades_export_import_roundtrip_microseconds below.
    assert_eq!(t.is_buyer_maker, o.is_buyer_maker);
    assert_eq!(t.is_best_match, o.is_best_match);
}

#[test]
fn test_record_batch_to_aggtrades_empty() {
    let batch = aggtrades_to_record_batch(&[]);
    let result = record_batch_to_aggtrades(&batch, false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_record_batch_to_aggtrades_timestamp_conversion() {
    // Verify ms → μs conversion: 1704067200000 ms → 1704067200000000 μs
    let timestamp_ms: i64 = 1704067200000; // 2024-01-01 00:00:00 UTC
    let expected_us: i64 = 1704067200000000;

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![timestamp_ms])),
            Arc::new(Float64Array::from(vec![50000.0])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, false).unwrap();
    assert_eq!(trades[0].timestamp, expected_us);

    // Also test edge cases: 0 and 1
    let batch_edge = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1_i64])),
            Arc::new(Float64Array::from(vec![100.0, 100.0])),
            Arc::new(Float64Array::from(vec![1.0, 1.0])),
        ],
    )
    .unwrap();

    let trades_edge = record_batch_to_aggtrades(&batch_edge, false).unwrap();
    assert_eq!(trades_edge[0].timestamp, 0); // 0 ms → 0 μs
    assert_eq!(trades_edge[1].timestamp, 1000); // 1 ms → 1000 μs
}

#[test]
fn test_record_batch_to_aggtrades_fixed_point_precision() {
    // Hand-computed: 50000.12345678 * 100_000_000 = 5000012345678
    let price = 50000.12345678_f64;
    let expected_fixed = (price * SCALE as f64).round() as i64;
    assert_eq!(expected_fixed, 5_000_012_345_678_i64);

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1704067200000_i64])),
            Arc::new(Float64Array::from(vec![price])),
            Arc::new(Float64Array::from(vec![1.5])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, false).unwrap();
    assert_eq!(trades[0].price.0, expected_fixed);
    // Verify roundtrip: FixedPoint → f64 → FixedPoint
    assert_eq!(trades[0].price.to_f64(), price);

    // Edge case: very small price
    let small_price = 0.00000001_f64;
    let expected_small = (small_price * SCALE as f64).round() as i64;
    assert_eq!(expected_small, 1);

    // Edge case: zero
    let zero_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1000_i64])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ],
    )
    .unwrap();

    let zero_trades = record_batch_to_aggtrades(&zero_batch, false).unwrap();
    assert_eq!(zero_trades[0].price.0, 0);
    assert_eq!(zero_trades[0].volume.0, 0);
}

#[test]
fn test_record_batch_to_aggtrades_missing_column() {
    // Missing "price" → ArrowImportError::MissingColumn
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1000_i64])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    let result = record_batch_to_aggtrades(&batch, false);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, ArrowImportError::MissingColumn { column: "price" }),
        "Expected MissingColumn for 'price', got: {err:?}"
    );

    // Missing both "volume" and "quantity" → error
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int64Array::from(vec![1000_i64])),
            Arc::new(Float64Array::from(vec![50000.0])),
        ],
    )
    .unwrap();

    let result2 = record_batch_to_aggtrades(&batch2, false);
    assert!(result2.is_err());
    assert!(
        matches!(
            result2.unwrap_err(),
            ArrowImportError::MissingColumn { column: "volume" }
        ),
        "Expected MissingColumn for 'volume'"
    );
}

#[test]
fn test_record_batch_to_aggtrades_type_mismatch() {
    // String column where Int64 expected for timestamp
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["not_a_number"])),
            Arc::new(Float64Array::from(vec![50000.0])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    let result = record_batch_to_aggtrades(&batch, false);
    assert!(result.is_err());
    match result.unwrap_err() {
        ArrowImportError::TypeMismatch {
            column: "timestamp",
            expected: "Int64",
            ..
        } => {} // expected
        other => panic!("Expected TypeMismatch for 'timestamp', got: {other:?}"),
    }
}

#[test]
fn test_record_batch_to_aggtrades_volume_quantity_fallback() {
    // Test that "quantity" column name works as fallback for "volume"
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1704067200000_i64])),
            Arc::new(Float64Array::from(vec![50000.0])),
            Arc::new(Float64Array::from(vec![2.5])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, false).unwrap();
    assert_eq!(trades.len(), 1);
    // 2.5 * 100_000_000 = 250_000_000
    assert_eq!(trades[0].volume.0, 250_000_000);
}

#[test]
fn test_record_batch_to_aggtrades_optional_defaults() {
    // Only required columns — verify defaults for optional fields
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1000_i64, 2000_i64])),
            Arc::new(Float64Array::from(vec![100.0, 200.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, false).unwrap();
    assert_eq!(trades.len(), 2);

    // agg_trade_id defaults to row index
    assert_eq!(trades[0].agg_trade_id, 0);
    assert_eq!(trades[1].agg_trade_id, 1);

    // first_trade_id and last_trade_id default to agg_trade_id
    assert_eq!(trades[0].first_trade_id, 0);
    assert_eq!(trades[0].last_trade_id, 0);
    assert_eq!(trades[1].first_trade_id, 1);
    assert_eq!(trades[1].last_trade_id, 1);

    // is_buyer_maker defaults to false
    assert!(!trades[0].is_buyer_maker);
    assert!(!trades[1].is_buyer_maker);

    // is_best_match defaults to None
    assert_eq!(trades[0].is_best_match, None);
    assert_eq!(trades[1].is_best_match, None);
}

// Phase 3: timestamp_is_microseconds tests

#[test]
fn test_record_batch_to_aggtrades_microseconds() {
    // timestamp_is_microseconds=true: input is already microseconds, no *1000
    let timestamp_us: i64 = 1704067200000000; // 2024-01-01 00:00:00 UTC in μs

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![timestamp_us])),
            Arc::new(Float64Array::from(vec![50000.0])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, true).unwrap();
    assert_eq!(trades.len(), 1);
    // With timestamp_is_microseconds=true, the value passes through unchanged
    assert_eq!(trades[0].timestamp, timestamp_us);
    // Verify it's NOT multiplied by 1000 (which would be year ~52000)
    assert!(
        trades[0].timestamp < 2_051_222_400_000_000, // 2035 in μs
        "timestamp {} exceeds 2035 — multiplication applied when it shouldn't be",
        trades[0].timestamp
    );
}

#[test]
fn test_record_batch_to_aggtrades_milliseconds_unchanged() {
    // Regression gate: timestamp_is_microseconds=false preserves existing ms→μs behavior
    let timestamp_ms: i64 = 1704067200000; // 2024-01-01 00:00:00 UTC in ms
    let expected_us: i64 = 1704067200000000; // same instant in μs

    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![timestamp_ms])),
            Arc::new(Float64Array::from(vec![50000.0])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    let trades = record_batch_to_aggtrades(&batch, false).unwrap();
    assert_eq!(trades[0].timestamp, expected_us);
}

#[test]
fn test_record_batch_to_aggtrades_export_import_roundtrip_microseconds() {
    // Full roundtrip: Vec<AggTrade> → aggtrades_to_record_batch → record_batch_to_aggtrades(true)
    // This is the Phase 3 internal path where timestamps never leave microsecond format
    let original = vec![
        AggTrade {
            agg_trade_id: 1,
            price: FixedPoint::from_str("42000.50").unwrap(),
            volume: FixedPoint::from_str("3.75").unwrap(),
            first_trade_id: 100,
            last_trade_id: 105,
            timestamp: 1704067200123456, // μs with sub-ms precision
            is_buyer_maker: true,
            is_best_match: Some(false),
        },
        AggTrade {
            agg_trade_id: 2,
            price: FixedPoint::from_str("42001.12345678").unwrap(),
            volume: FixedPoint::from_str("0.00000001").unwrap(), // minimum precision
            first_trade_id: 106,
            last_trade_id: 106,
            timestamp: 1704067200999999, // μs near second boundary
            is_buyer_maker: false,
            is_best_match: None,
        },
    ];

    let batch = aggtrades_to_record_batch(&original);
    // timestamp_is_microseconds=true because aggtrades_to_record_batch exports μs
    let imported = record_batch_to_aggtrades(&batch, true).unwrap();

    assert_eq!(imported.len(), 2);

    // Bit-exact field-by-field verification
    for (i, (imp, orig)) in imported.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            imp.agg_trade_id, orig.agg_trade_id,
            "trade {i}: agg_trade_id"
        );
        assert_eq!(imp.price.0, orig.price.0, "trade {i}: price");
        assert_eq!(imp.volume.0, orig.volume.0, "trade {i}: volume");
        assert_eq!(
            imp.first_trade_id, orig.first_trade_id,
            "trade {i}: first_trade_id"
        );
        assert_eq!(
            imp.last_trade_id, orig.last_trade_id,
            "trade {i}: last_trade_id"
        );
        assert_eq!(imp.timestamp, orig.timestamp, "trade {i}: timestamp");
        assert_eq!(
            imp.is_buyer_maker, orig.is_buyer_maker,
            "trade {i}: is_buyer_maker"
        );
        assert_eq!(
            imp.is_best_match, orig.is_best_match,
            "trade {i}: is_best_match"
        );
    }
}

#[test]
fn test_timestamp_cross_year_sweep() {
    // Verify both paths produce valid timestamps across multiple years
    let test_cases = vec![
        (1514764800000_i64, "2018-01-01"), // ms, 13 digits
        (1577836800000_i64, "2020-01-01"), // ms, 13 digits
        (1704067200000_i64, "2024-01-01"), // ms, 13 digits
        (1735689600000_i64, "2025-01-01"), // ms, 13 digits
    ];

    for (timestamp_ms, label) in &test_cases {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        // Test ms path (timestamp_is_microseconds=false)
        let batch_ms = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![*timestamp_ms])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let trades_ms = record_batch_to_aggtrades(&batch_ms, false).unwrap();
        let ts_us_from_ms = trades_ms[0].timestamp;
        assert_eq!(
            ts_us_from_ms,
            timestamp_ms * 1000,
            "{label}: ms path should multiply by 1000"
        );

        // Test μs path (timestamp_is_microseconds=true)
        let timestamp_us = timestamp_ms * 1000;
        let batch_us = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![timestamp_us])),
                Arc::new(Float64Array::from(vec![50000.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let trades_us = record_batch_to_aggtrades(&batch_us, true).unwrap();
        let ts_us_from_us = trades_us[0].timestamp;
        assert_eq!(
            ts_us_from_us, timestamp_us,
            "{label}: μs path should pass through unchanged"
        );

        // Both paths should produce identical output for equivalent timestamps
        assert_eq!(
            ts_us_from_ms, ts_us_from_us,
            "{label}: both paths must converge to same μs value"
        );

        // Sanity: all timestamps in valid range [2000, 2035]
        let year_2000_us: i64 = 946684800000000;
        let year_2035_us: i64 = 2051222400000000;
        assert!(
            ts_us_from_ms >= year_2000_us && ts_us_from_ms <= year_2035_us,
            "{label}: timestamp {ts_us_from_ms} outside valid range"
        );
    }
}
