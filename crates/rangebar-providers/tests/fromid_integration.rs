// Issue #92: fromId pagination integration tests (live Binance API)
//
// These tests validate the new fromId-based REST methods against live data.
// Each test emits structured NDJSON telemetry to target/test-telemetry/ for post-analysis.
//
// Run: cargo nextest run -p rangebar-providers --test fromid_integration --no-fail-fast

use rangebar_providers::binance::HistoricalDataLoader;
use std::fs::File;
use std::io::Write;

// ============================================================================
// Telemetry Infrastructure
// ============================================================================

struct TestTelemetry {
    log: File,
}

impl TestTelemetry {
    fn new(test_name: &str) -> Self {
        let dir = "target/test-telemetry";
        std::fs::create_dir_all(dir).ok();
        let path = format!("{dir}/{test_name}.ndjson");
        Self {
            log: File::create(path).unwrap(),
        }
    }

    fn emit(&mut self, event: &str, data: serde_json::Value) {
        let entry = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "event": event,
            "data": data,
        });
        writeln!(self.log, "{}", serde_json::to_string(&entry).unwrap()).ok();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn test_fetch_latest_returns_recent_trade() {
    let mut tel = TestTelemetry::new("fetch_latest_returns_recent_trade");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    let trade = loader.fetch_latest_aggtrade().await.unwrap();

    tel.emit(
        "latest_trade",
        serde_json::json!({
            "agg_trade_id": trade.agg_trade_id,
            "timestamp_us": trade.timestamp,
            "price": trade.price.to_f64(),
        }),
    );

    // Trade should have a positive ID and recent timestamp
    assert!(trade.agg_trade_id > 0, "agg_trade_id should be positive");
    assert!(trade.timestamp > 0, "timestamp should be positive");

    // Timestamp should be within the last 60 seconds (in microseconds)
    let now_us = chrono::Utc::now().timestamp_micros();
    let age_us = now_us - trade.timestamp;
    tel.emit(
        "timestamp_freshness",
        serde_json::json!({
            "now_us": now_us,
            "trade_us": trade.timestamp,
            "age_seconds": age_us as f64 / 1_000_000.0,
        }),
    );
    assert!(
        age_us < 60_000_000,
        "Trade should be within last 60s, age was {}s",
        age_us as f64 / 1_000_000.0
    );
}

#[tokio::test]
async fn test_fetch_by_id_returns_exact_range() {
    let mut tel = TestTelemetry::new("fetch_by_id_returns_exact_range");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Get anchor point
    let latest = loader.fetch_latest_aggtrade().await.unwrap();
    let from_id = latest.agg_trade_id - 100;

    let trades = loader.fetch_aggtrades_by_id(from_id, 100).await.unwrap();

    tel.emit(
        "by_id_fetch",
        serde_json::json!({
            "requested_from_id": from_id,
            "first_returned_id": trades.first().map(|t| t.agg_trade_id),
            "last_returned_id": trades.last().map(|t| t.agg_trade_id),
            "count": trades.len(),
        }),
    );

    assert!(!trades.is_empty(), "Should return trades");
    assert_eq!(
        trades[0].agg_trade_id, from_id,
        "First trade should match requested fromId"
    );

    tel.emit(
        "first_id_match",
        serde_json::json!({
            "expected": from_id,
            "actual": trades[0].agg_trade_id,
            "match": trades[0].agg_trade_id == from_id,
        }),
    );
}

#[tokio::test]
async fn test_fetch_by_id_contiguity() {
    let mut tel = TestTelemetry::new("fetch_by_id_contiguity");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Get anchor point
    let latest = loader.fetch_latest_aggtrade().await.unwrap();
    let start_id = latest.agg_trade_id - 2000;

    // Page 1
    let page1 = loader.fetch_aggtrades_by_id(start_id, 1000).await.unwrap();
    let page1_last = page1.last().unwrap().agg_trade_id;

    tel.emit(
        "page1_range",
        serde_json::json!({
            "first_id": page1[0].agg_trade_id,
            "last_id": page1_last,
            "count": page1.len(),
        }),
    );

    // Page 2: fromId = page1_last + 1
    let page2 = loader
        .fetch_aggtrades_by_id(page1_last + 1, 1000)
        .await
        .unwrap();

    tel.emit(
        "page2_range",
        serde_json::json!({
            "first_id": page2[0].agg_trade_id,
            "last_id": page2.last().unwrap().agg_trade_id,
            "count": page2.len(),
        }),
    );

    // Verify contiguity: page2 first = page1 last + 1
    let delta = page2[0].agg_trade_id - page1_last;
    tel.emit(
        "junction_delta",
        serde_json::json!({
            "delta": delta,
            "seamless": delta == 1,
        }),
    );

    assert_eq!(delta, 1, "Gap between pages: delta={delta}");

    // Verify internal contiguity within each page
    for window in page1.windows(2) {
        assert_eq!(
            window[1].agg_trade_id - window[0].agg_trade_id,
            1,
            "Internal gap in page1 at id {}",
            window[0].agg_trade_id
        );
    }
    for window in page2.windows(2) {
        assert_eq!(
            window[1].agg_trade_id - window[0].agg_trade_id,
            1,
            "Internal gap in page2 at id {}",
            window[0].agg_trade_id
        );
    }
}

#[tokio::test]
async fn test_fromid_vs_starttime_gap_comparison() {
    let mut tel = TestTelemetry::new("fromid_vs_starttime_gap_comparison");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Get anchor and fetch a range by fromId
    let latest = loader.fetch_latest_aggtrade().await.unwrap();
    let start_id = latest.agg_trade_id - 5000;

    // fromId: fetch 5 pages of 1000
    let mut fromid_trades = Vec::new();
    let mut cursor = start_id;
    for page_num in 0..5 {
        let page = loader.fetch_aggtrades_by_id(cursor, 1000).await.unwrap();
        if page.is_empty() {
            break;
        }
        cursor = page.last().unwrap().agg_trade_id + 1;
        fromid_trades.extend(page);
        tel.emit(
            "fromid_page",
            serde_json::json!({ "page": page_num, "cursor": cursor }),
        );
    }

    // Count gaps in fromId sequence
    let fromid_gaps: Vec<_> = fromid_trades
        .windows(2)
        .filter(|w| w[1].agg_trade_id - w[0].agg_trade_id != 1)
        .map(|w| {
            serde_json::json!({
                "before": w[0].agg_trade_id,
                "after": w[1].agg_trade_id,
                "delta": w[1].agg_trade_id - w[0].agg_trade_id,
            })
        })
        .collect();

    tel.emit(
        "fromid_gaps",
        serde_json::json!({
            "total_trades": fromid_trades.len(),
            "gap_count": fromid_gaps.len(),
            "gaps": fromid_gaps,
        }),
    );

    // startTime: fetch the same time range
    if let (Some(first), Some(last)) = (fromid_trades.first(), fromid_trades.last()) {
        let start_ms = first.timestamp / 1000; // us → ms
        let end_ms = last.timestamp / 1000;

        let starttime_trades = loader
            .fetch_aggtrades_rest(start_ms, end_ms)
            .await
            .unwrap();

        let starttime_gaps: Vec<_> = starttime_trades
            .windows(2)
            .filter(|w| w[1].agg_trade_id - w[0].agg_trade_id != 1)
            .map(|w| {
                serde_json::json!({
                    "before": w[0].agg_trade_id,
                    "after": w[1].agg_trade_id,
                    "delta": w[1].agg_trade_id - w[0].agg_trade_id,
                })
            })
            .collect();

        tel.emit(
            "starttime_gaps",
            serde_json::json!({
                "total_trades": starttime_trades.len(),
                "gap_count": starttime_gaps.len(),
                "gaps": starttime_gaps,
            }),
        );
    }

    // fromId should have zero gaps
    assert_eq!(
        fromid_gaps.len(),
        0,
        "fromId pagination should have zero gaps, found {}",
        fromid_gaps.len()
    );
}

#[tokio::test]
async fn test_fetch_by_id_field_identity() {
    let mut tel = TestTelemetry::new("fetch_by_id_field_identity");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Get latest via fetch_latest
    let latest = loader.fetch_latest_aggtrade().await.unwrap();

    // Fetch the same trade via fromId
    let by_id = loader
        .fetch_aggtrades_by_id(latest.agg_trade_id, 1)
        .await
        .unwrap();

    assert!(!by_id.is_empty(), "Should return at least one trade");
    let by_id_trade = &by_id[0];

    let fields_match = latest.agg_trade_id == by_id_trade.agg_trade_id
        && latest.price == by_id_trade.price
        && latest.volume == by_id_trade.volume
        && latest.timestamp == by_id_trade.timestamp
        && latest.is_buyer_maker == by_id_trade.is_buyer_maker
        && latest.first_trade_id == by_id_trade.first_trade_id
        && latest.last_trade_id == by_id_trade.last_trade_id;

    tel.emit(
        "field_comparison",
        serde_json::json!({
            "agg_trade_id_match": latest.agg_trade_id == by_id_trade.agg_trade_id,
            "price_match": latest.price == by_id_trade.price,
            "volume_match": latest.volume == by_id_trade.volume,
            "timestamp_match": latest.timestamp == by_id_trade.timestamp,
            "is_buyer_maker_match": latest.is_buyer_maker == by_id_trade.is_buyer_maker,
            "all_match": fields_match,
        }),
    );

    assert!(
        fields_match,
        "fetch_latest and fetch_by_id should return identical data for the same agg_trade_id"
    );
}

#[tokio::test]
async fn test_rate_limit_backoff_does_not_crash() {
    let mut tel = TestTelemetry::new("rate_limit_backoff_does_not_crash");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Make 5 rapid sequential calls — 429 handling should prevent crashes
    let mut success_count = 0;
    for i in 0..5 {
        match loader.fetch_latest_aggtrade().await {
            Ok(trade) => {
                success_count += 1;
                tel.emit(
                    "call_success",
                    serde_json::json!({
                        "call": i,
                        "agg_trade_id": trade.agg_trade_id,
                    }),
                );
            }
            Err(e) => {
                tel.emit(
                    "call_error",
                    serde_json::json!({
                        "call": i,
                        "error": e.to_string(),
                    }),
                );
            }
        }
    }

    tel.emit(
        "summary",
        serde_json::json!({
            "total_calls": 5,
            "success_count": success_count,
            "recovered": success_count > 0,
        }),
    );

    assert!(
        success_count > 0,
        "At least one call should succeed with backoff"
    );
}

#[tokio::test]
async fn test_fetch_by_id_large_cursor_advance() {
    let mut tel = TestTelemetry::new("fetch_by_id_large_cursor_advance");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    let latest = loader.fetch_latest_aggtrade().await.unwrap();
    let start_id = latest.agg_trade_id - 5000;

    let mut all_trades = Vec::new();
    let mut cursor = start_id;

    for page_num in 0..5 {
        let page = loader.fetch_aggtrades_by_id(cursor, 1000).await.unwrap();
        if page.is_empty() {
            break;
        }
        cursor = page.last().unwrap().agg_trade_id + 1;
        all_trades.extend(page);

        tel.emit(
            "page",
            serde_json::json!({
                "page": page_num,
                "count": all_trades.len(),
                "cursor": cursor,
            }),
        );
    }

    // Count gaps
    let gaps: usize = all_trades
        .windows(2)
        .filter(|w| w[1].agg_trade_id - w[0].agg_trade_id != 1)
        .count();

    tel.emit(
        "summary",
        serde_json::json!({
            "total_trades": all_trades.len(),
            "total_gaps": gaps,
            "pages": 5,
            "contiguous": gaps == 0,
        }),
    );

    assert!(
        all_trades.len() >= 4000,
        "Should have fetched at least 4000 trades, got {}",
        all_trades.len()
    );
    assert_eq!(gaps, 0, "All 5 pages should be contiguous, found {gaps} gaps");
}

#[tokio::test]
async fn test_vision_to_rest_junction() {
    let mut tel = TestTelemetry::new("vision_to_rest_junction");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Load latest available Vision day
    let recent_day = match loader.load_recent_day().await {
        Ok(trades) => trades,
        Err(e) => {
            tel.emit(
                "skip",
                serde_json::json!({ "reason": format!("No Vision data: {e}") }),
            );
            // Skip test if no Vision data available (network, timing, etc.)
            return;
        }
    };

    if recent_day.is_empty() {
        tel.emit("skip", serde_json::json!({ "reason": "Empty Vision day" }));
        return;
    }

    let vision_last_id = recent_day.last().unwrap().agg_trade_id;

    tel.emit(
        "vision_data",
        serde_json::json!({
            "trade_count": recent_day.len(),
            "vision_last_id": vision_last_id,
        }),
    );

    // Fetch REST starting from vision_last_id + 1
    let rest_trades = loader
        .fetch_aggtrades_by_id(vision_last_id + 1, 10)
        .await
        .unwrap();

    if rest_trades.is_empty() {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "No REST trades after Vision last ID" }),
        );
        return;
    }

    let rest_first_id = rest_trades[0].agg_trade_id;
    let junction_delta = rest_first_id - vision_last_id;

    tel.emit(
        "junction",
        serde_json::json!({
            "vision_last_id": vision_last_id,
            "rest_first_id": rest_first_id,
            "junction_delta": junction_delta,
            "junction_seamless": junction_delta == 1,
        }),
    );

    assert_eq!(
        junction_delta, 1,
        "Vision→REST junction should be seamless (delta=1), got delta={junction_delta}"
    );
}
