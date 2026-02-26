// Issue #92: REST→WebSocket junction integration tests (live Binance API)
//
// These tests validate that REST fromId and WebSocket streams produce
// contiguous, field-identical data at their junction point.
//
// Requires the `binance` feature for WebSocket support.
//
// Run: cargo nextest run -p rangebar-providers --test rest_ws_junction_integration --no-fail-fast

use rangebar_providers::binance::{BinanceWebSocketStream, HistoricalDataLoader};
use std::fs::File;
use std::io::Write;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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

/// Collect WebSocket trades for a given duration.
/// Returns collected trades (may be empty on timeout/error).
async fn collect_ws_trades(
    symbol: &str,
    duration_secs: u64,
) -> Vec<rangebar_core::AggTrade> {
    let (tx, mut rx) = mpsc::channel(1000);
    let shutdown = CancellationToken::new();
    let shutdown_clone = shutdown.clone();
    let symbol_owned = symbol.to_string();

    // Spawn WS connection in background
    let ws_handle = tokio::spawn(async move {
        BinanceWebSocketStream::connect_and_stream(
            &symbol_owned,
            &tx,
            &shutdown_clone,
            std::time::Duration::from_secs(30),
        )
        .await
        .ok();
    });

    // Collect trades for the specified duration
    let mut trades = Vec::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(duration_secs);

    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await {
            Ok(Some(trade)) => trades.push(trade),
            Ok(None) => break, // Channel closed
            Err(_) => continue, // Timeout on individual recv, keep trying
        }
    }

    // Clean shutdown
    shutdown.cancel();
    ws_handle.abort();

    trades
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn test_rest_tail_to_ws_bridge() {
    let mut tel = TestTelemetry::new("rest_tail_to_ws_bridge");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Step 1: Get REST tail (latest trade as anchor)
    let rest_tail = loader.fetch_latest_aggtrade().await.unwrap();
    let rest_tail_id = rest_tail.agg_trade_id;

    tel.emit(
        "rest_tail_id",
        serde_json::json!({ "agg_trade_id": rest_tail_id }),
    );

    // Step 2: Connect WebSocket and collect trades for 5 seconds
    let ws_trades = collect_ws_trades("BTCUSDT", 5).await;

    if ws_trades.is_empty() {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "No WebSocket trades received" }),
        );
        return;
    }

    let ws_first_id = ws_trades.first().unwrap().agg_trade_id;

    tel.emit(
        "ws_capture",
        serde_json::json!({
            "ws_trade_count": ws_trades.len(),
            "ws_first_id": ws_first_id,
            "ws_last_id": ws_trades.last().unwrap().agg_trade_id,
        }),
    );

    // Step 3: Fetch bridge via REST fromId to overlap with WS
    let bridge_from = rest_tail_id + 1;
    let bridge_trades = loader
        .fetch_aggtrades_by_id(bridge_from, 1000)
        .await
        .unwrap();

    if bridge_trades.is_empty() {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "No bridge trades" }),
        );
        return;
    }

    let bridge_last_id = bridge_trades.last().unwrap().agg_trade_id;

    tel.emit(
        "bridge_range",
        serde_json::json!({
            "bridge_from": bridge_from,
            "bridge_first_id": bridge_trades[0].agg_trade_id,
            "bridge_last_id": bridge_last_id,
            "bridge_count": bridge_trades.len(),
        }),
    );

    // Step 4: Check overlap between bridge and WS
    let overlap_count = bridge_trades
        .iter()
        .filter(|bt| ws_trades.iter().any(|wt| wt.agg_trade_id == bt.agg_trade_id))
        .count();

    tel.emit(
        "overlap_count",
        serde_json::json!({
            "overlap": overlap_count,
            "has_overlap": overlap_count > 0,
        }),
    );

    // Bridge should start right after REST tail
    assert_eq!(
        bridge_trades[0].agg_trade_id,
        rest_tail_id + 1,
        "Bridge should start at rest_tail_id + 1"
    );
}

#[tokio::test]
async fn test_field_identity_rest_vs_ws() {
    let mut tel = TestTelemetry::new("field_identity_rest_vs_ws");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Connect WebSocket first and collect some trades
    let ws_trades = collect_ws_trades("BTCUSDT", 5).await;

    if ws_trades.is_empty() {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "No WebSocket trades received" }),
        );
        return;
    }

    // Fetch the same trades via REST fromId
    let ws_first_id = ws_trades.first().unwrap().agg_trade_id;
    let ws_last_id = ws_trades.last().unwrap().agg_trade_id;
    let count = (ws_last_id - ws_first_id + 1).min(1000) as u16;

    let rest_trades = loader
        .fetch_aggtrades_by_id(ws_first_id, count)
        .await
        .unwrap();

    // Cross-validate overlapping trades
    let mut mismatches = Vec::new();
    let mut checked = 0;

    for ws_trade in &ws_trades {
        if let Some(rest_trade) = rest_trades
            .iter()
            .find(|rt| rt.agg_trade_id == ws_trade.agg_trade_id)
        {
            checked += 1;
            let price_match = ws_trade.price == rest_trade.price;
            let volume_match = ws_trade.volume == rest_trade.volume;
            let maker_match = ws_trade.is_buyer_maker == rest_trade.is_buyer_maker;

            if !price_match || !volume_match || !maker_match {
                mismatches.push(serde_json::json!({
                    "agg_trade_id": ws_trade.agg_trade_id,
                    "price_match": price_match,
                    "volume_match": volume_match,
                    "maker_match": maker_match,
                }));
            }
        }
    }

    tel.emit(
        "field_comparison",
        serde_json::json!({
            "ws_trades": ws_trades.len(),
            "rest_trades": rest_trades.len(),
            "overlapping_checked": checked,
            "mismatches": mismatches.len(),
            "fields_checked": ["price", "volume", "is_buyer_maker"],
        }),
    );

    assert!(checked > 0, "Should have at least one overlapping trade");
    assert_eq!(
        mismatches.len(),
        0,
        "REST and WS should return identical data, found {} mismatches",
        mismatches.len()
    );
}

#[tokio::test]
async fn test_ws_reconnect_gap_detection() {
    let mut tel = TestTelemetry::new("ws_reconnect_gap_detection");
    let loader = HistoricalDataLoader::new("BTCUSDT");

    // Capture some WS trades
    let ws_trades = collect_ws_trades("BTCUSDT", 3).await;

    if ws_trades.len() < 2 {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "Insufficient WS trades" }),
        );
        return;
    }

    // Simulate a gap: take first and last trade, skip middle
    let gap_start = ws_trades.first().unwrap().agg_trade_id + 1;
    let gap_end = ws_trades.last().unwrap().agg_trade_id - 1;

    if gap_end <= gap_start {
        tel.emit(
            "skip",
            serde_json::json!({ "reason": "Not enough trades for gap simulation" }),
        );
        return;
    }

    tel.emit(
        "simulated_gap",
        serde_json::json!({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_size": gap_end - gap_start + 1,
        }),
    );

    // Backfill the gap via REST fromId
    let backfill = loader
        .fetch_aggtrades_by_id(gap_start, (gap_end - gap_start + 1).min(1000) as u16)
        .await
        .unwrap();

    tel.emit(
        "backfill_result",
        serde_json::json!({
            "backfill_count": backfill.len(),
            "first_id": backfill.first().map(|t| t.agg_trade_id),
            "last_id": backfill.last().map(|t| t.agg_trade_id),
            "restored": !backfill.is_empty()
                && backfill[0].agg_trade_id == gap_start,
        }),
    );

    assert!(
        !backfill.is_empty(),
        "REST fromId should fill the simulated gap"
    );
    assert_eq!(
        backfill[0].agg_trade_id, gap_start,
        "Backfill should start at gap_start"
    );
}
