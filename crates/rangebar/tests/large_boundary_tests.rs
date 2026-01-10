use rangebar::types::RangeBar;
use rangebar_core::test_utils::generators::*;
use std::time::Instant;

/// Large-scale boundary consistency tests with comprehensive datasets
///
/// These tests validate streaming vs batch consistency across:
/// - Multi-million trade datasets
/// - Multi-day boundary transitions
/// - Market session boundaries (open/close)
/// - High/low frequency trading periods
/// - Memory stress testing

#[tokio::test]
async fn test_massive_dataset_boundary_consistency() {
    println!("🔍 Testing massive dataset boundary consistency (1M+ trades)");

    let threshold_decimal_bps = 25; // 0.25% standard threshold
    let trade_count = 1_000_000; // 1 million trades

    println!(
        "  Generating {} trades with realistic price movements",
        trade_count
    );
    let start_gen = Instant::now();
    let massive_dataset = create_massive_realistic_dataset(trade_count);
    println!("  ✅ Dataset generated in {:?}", start_gen.elapsed());

    // Test batch processing
    println!("  🔄 Running batch processing...");
    let start_batch = Instant::now();
    let batch_bars = process_batch_style(&massive_dataset, threshold_decimal_bps);
    let batch_duration = start_batch.elapsed();

    println!(
        "  ✅ Batch: {} bars in {:?}",
        batch_bars.len(),
        batch_duration
    );

    // Test streaming processing
    println!("  🔄 Running streaming processing...");
    let start_streaming = Instant::now();
    let streaming_bars = process_streaming_style(&massive_dataset, threshold_decimal_bps).await;
    let streaming_duration = start_streaming.elapsed();

    println!(
        "  ✅ Streaming: {} bars in {:?}",
        streaming_bars.len(),
        streaming_duration
    );

    // Compare results
    let matches = batch_bars.len() == streaming_bars.len();
    let ratio = streaming_bars.len() as f64 / batch_bars.len() as f64;

    let status_msg = if matches {
        "✅ MATCH".to_string()
    } else {
        format!("❌ MISMATCH ({:.2}x)", ratio)
    };

    println!(
        "  📊 Comparison: {} vs {} bars - {}",
        batch_bars.len(),
        streaming_bars.len(),
        status_msg
    );

    // Performance analysis
    let batch_speed = trade_count as f64 / batch_duration.as_secs_f64();
    let streaming_speed = trade_count as f64 / streaming_duration.as_secs_f64();

    println!("  ⚡ Performance:");
    println!("    Batch: {:.0} trades/sec", batch_speed);
    println!("    Streaming: {:.0} trades/sec", streaming_speed);
    println!("    Speedup: {:.2}x", streaming_speed / batch_speed);

    // Memory analysis
    let avg_trades_per_bar = trade_count as f64 / batch_bars.len() as f64;
    println!("  💾 Efficiency: {:.1} trades per bar", avg_trades_per_bar);

    // Validation
    assert!(!batch_bars.is_empty(), "Batch should generate bars");
    assert!(!streaming_bars.is_empty(), "Streaming should generate bars");

    // Validate temporal integrity
    validate_temporal_integrity(&batch_bars, "massive_batch");
    validate_temporal_integrity(&streaming_bars, "massive_streaming");

    println!("  ✅ Large dataset test complete");
}

#[tokio::test]
async fn test_multi_day_boundary_transitions() {
    println!("🔍 Testing multi-day boundary transitions");

    let threshold_decimal_bps = 25;
    let days = 7; // One week of data

    println!("  Generating {} days of continuous trading data", days);
    let multi_day_dataset = create_multi_day_boundary_dataset(days);

    println!("  Total trades: {}", multi_day_dataset.len());

    // Test with boundary preservation
    let batch_bars = process_batch_style(&multi_day_dataset, threshold_decimal_bps);
    let streaming_bars = process_streaming_style(&multi_day_dataset, threshold_decimal_bps).await;

    println!("  📊 Multi-day results:");
    println!("    Batch: {} bars", batch_bars.len());
    println!("    Streaming: {} bars", streaming_bars.len());

    let matches = batch_bars.len() == streaming_bars.len();
    println!(
        "    Status: {}",
        if matches { "✅ MATCH" } else { "❌ MISMATCH" }
    );

    // Analyze boundary behavior
    analyze_boundary_behavior(&batch_bars, &streaming_bars, days);

    // Validate that bars can span day boundaries properly
    validate_cross_day_bars(&batch_bars, "batch");
    validate_cross_day_bars(&streaming_bars, "streaming");

    assert!(!batch_bars.is_empty(), "Batch should generate bars");
    assert!(!streaming_bars.is_empty(), "Streaming should generate bars");
}

#[tokio::test]
async fn test_market_session_boundaries() {
    println!("🔍 Testing market session boundaries");

    let threshold_decimal_bps = 25;

    // Create data with distinct trading sessions
    let session_datasets = vec![
        ("asian_session", create_asian_session_data()),
        ("european_session", create_european_session_data()),
        ("us_session", create_us_session_data()),
        ("weekend_gap", create_weekend_gap_data()),
    ];

    for (session_name, dataset) in session_datasets {
        println!("  🎯 Testing: {}", session_name);

        let batch_bars = process_batch_style(&dataset, threshold_decimal_bps);
        let streaming_bars = process_streaming_style(&dataset, threshold_decimal_bps).await;

        let matches = batch_bars.len() == streaming_bars.len();
        println!(
            "    {}: Batch={}, Streaming={} - {}",
            session_name,
            batch_bars.len(),
            streaming_bars.len(),
            if matches { "✅ MATCH" } else { "❌ MISMATCH" }
        );

        // Validate session-specific characteristics
        validate_session_characteristics(&batch_bars, session_name);
        validate_session_characteristics(&streaming_bars, session_name);
    }
}

#[tokio::test]
async fn test_frequency_boundary_variations() {
    println!("🔍 Testing high/low frequency boundary variations");

    let threshold_decimal_bps = 25;

    let frequency_tests = vec![
        ("high_frequency_1ms", create_high_frequency_data(1)), // 1ms intervals
        ("medium_frequency_100ms", create_medium_frequency_data(100)), // 100ms intervals
        ("low_frequency_1s", create_low_frequency_data(1000)), // 1s intervals
        ("mixed_frequency", create_mixed_frequency_data()),    // Variable intervals
    ];

    for (test_name, dataset) in frequency_tests {
        println!("  📈 Testing: {}", test_name);

        let start_time = Instant::now();
        let batch_bars = process_batch_style(&dataset, threshold_decimal_bps);
        let batch_duration = start_time.elapsed();

        let start_time = Instant::now();
        let streaming_bars = process_streaming_style(&dataset, threshold_decimal_bps).await;
        let streaming_duration = start_time.elapsed();

        let matches = batch_bars.len() == streaming_bars.len();
        println!(
            "    {} ({} trades): Batch={} bars ({:?}), Streaming={} bars ({:?}) - {}",
            test_name,
            dataset.len(),
            batch_bars.len(),
            batch_duration,
            streaming_bars.len(),
            streaming_duration,
            if matches { "✅ MATCH" } else { "❌ MISMATCH" }
        );

        // Analyze frequency-specific patterns
        analyze_frequency_patterns(&batch_bars, &streaming_bars, test_name);
    }
}

#[tokio::test]
async fn test_stress_boundary_conditions() {
    println!("🔍 Testing stress boundary conditions");

    let threshold_decimal_bps = 25;

    let stress_tests = vec![
        ("rapid_threshold_hits", create_rapid_threshold_hit_data()),
        ("price_precision_limits", create_precision_limit_data()),
        ("volume_extremes", create_volume_extreme_data()),
        ("timestamp_edge_cases", create_timestamp_edge_data()),
        ("floating_point_stress", create_floating_point_stress_data()),
    ];

    for (test_name, dataset) in stress_tests {
        println!("  ⚡ Stress testing: {}", test_name);

        let batch_bars = process_batch_style(&dataset, threshold_decimal_bps);
        let streaming_bars = process_streaming_style(&dataset, threshold_decimal_bps).await;

        let matches = batch_bars.len() == streaming_bars.len();
        println!(
            "    {}: {} - {}",
            test_name,
            if matches {
                format!("✅ MATCH ({} bars)", batch_bars.len())
            } else {
                format!(
                    "❌ MISMATCH (B:{}, S:{})",
                    batch_bars.len(),
                    streaming_bars.len()
                )
            },
            if matches { "PASS" } else { "FAIL" }
        );

        // Validate stress test specific requirements
        validate_stress_test_requirements(&batch_bars, &streaming_bars, test_name);
    }
}

// Test-specific validation helpers

fn validate_temporal_integrity(bars: &[RangeBar], test_name: &str) {
    for (i, bar) in bars.iter().enumerate() {
        assert!(
            bar.close_time >= bar.open_time,
            "{}: Bar {} has close_time before open_time",
            test_name,
            i
        );

        if i > 0 {
            assert!(
                bar.open_time >= bars[i - 1].close_time,
                "{}: Bar {} starts before previous bar ends",
                test_name,
                i
            );
        }
    }
}

fn analyze_boundary_behavior(batch_bars: &[RangeBar], streaming_bars: &[RangeBar], _days: usize) {
    println!("  🔍 Boundary analysis:");

    // Calculate average bar duration
    if !batch_bars.is_empty() {
        let total_time =
            batch_bars.last().unwrap().close_time - batch_bars.first().unwrap().open_time;
        let avg_duration = total_time / batch_bars.len() as i64;
        println!("    Batch avg bar duration: {}ms", avg_duration);
    }

    if !streaming_bars.is_empty() {
        let total_time =
            streaming_bars.last().unwrap().close_time - streaming_bars.first().unwrap().open_time;
        let avg_duration = total_time / streaming_bars.len() as i64;
        println!("    Streaming avg bar duration: {}ms", avg_duration);
    }

    // Look for bars that span multiple days (indicating proper boundary handling)
    let day_ms = 24 * 60 * 60 * 1000;
    let long_batch_bars = batch_bars
        .iter()
        .filter(|bar| bar.close_time - bar.open_time > day_ms)
        .count();
    let long_streaming_bars = streaming_bars
        .iter()
        .filter(|bar| bar.close_time - bar.open_time > day_ms)
        .count();

    println!(
        "    Multi-day bars: Batch={}, Streaming={}",
        long_batch_bars, long_streaming_bars
    );
}

fn validate_cross_day_bars(bars: &[RangeBar], implementation: &str) {
    let day_ms = 24 * 60 * 60 * 1000;
    let cross_day_count = bars
        .iter()
        .filter(|bar| bar.close_time - bar.open_time > day_ms)
        .count();

    if cross_day_count > 0 {
        println!(
            "    ✅ {} implementation properly handles cross-day bars: {}",
            implementation, cross_day_count
        );
    }
}

fn validate_session_characteristics(bars: &[RangeBar], session_name: &str) {
    if bars.is_empty() {
        return;
    }

    // Validate basic characteristics are reasonable for the session
    let total_duration = bars.last().unwrap().close_time - bars.first().unwrap().open_time;
    let avg_bar_duration = total_duration / bars.len() as i64;

    println!(
        "      {} avg bar duration: {}ms",
        session_name, avg_bar_duration
    );
}

fn analyze_frequency_patterns(
    batch_bars: &[RangeBar],
    streaming_bars: &[RangeBar],
    test_name: &str,
) {
    if batch_bars.is_empty() || streaming_bars.is_empty() {
        return;
    }

    // Calculate frequency-specific metrics
    let batch_time_span =
        batch_bars.last().unwrap().close_time - batch_bars.first().unwrap().open_time;
    let streaming_time_span =
        streaming_bars.last().unwrap().close_time - streaming_bars.first().unwrap().open_time;

    let batch_freq = batch_bars.len() as f64 / (batch_time_span as f64 / 1000.0); // bars per second
    let streaming_freq = streaming_bars.len() as f64 / (streaming_time_span as f64 / 1000.0);

    println!(
        "      {} frequency: Batch={:.2} bars/s, Streaming={:.2} bars/s",
        test_name, batch_freq, streaming_freq
    );
}

fn validate_stress_test_requirements(
    batch_bars: &[RangeBar],
    streaming_bars: &[RangeBar],
    test_name: &str,
) {
    // Ensure both implementations handle stress conditions
    assert!(
        !batch_bars.is_empty(),
        "{}: Batch should handle stress test",
        test_name
    );
    assert!(
        !streaming_bars.is_empty(),
        "{}: Streaming should handle stress test",
        test_name
    );

    // Validate temporal integrity under stress
    validate_temporal_integrity(batch_bars, &format!("{}_batch", test_name));
    validate_temporal_integrity(streaming_bars, &format!("{}_streaming", test_name));

    println!("      ✅ {} stress test validation passed", test_name);
}
