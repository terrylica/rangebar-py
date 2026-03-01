//! Live bar engine for real-time range bar construction (Issue #91)
//! # FILE-SIZE-OK
//!
//! Multiplexes Binance WebSocket streams across symbols and fans out trades
//! to canonical `RangeBarProcessor` instances (full 58-column microstructure).
//!
//! Architecture:
//! - One tokio task per symbol (independent reconnection via barter-rs pattern)
//! - One `RangeBarProcessor` per (symbol, threshold) pair
//! - Completed bars emitted to a bounded channel for Python consumption
//! - Graceful shutdown via `CancellationToken`

use rangebar_core::checkpoint::Checkpoint;
use rangebar_core::processor::{ProcessingError, RangeBarProcessor};
use rangebar_core::interbar::InterBarConfig;
use rangebar_core::IntraBarConfig;
use rangebar_core::{AggTrade, RangeBar};
use rangebar_providers::binance::{BinanceWebSocketStream, ReconnectionPolicy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

// Issue #96 Task #9: Ring buffer replaces mpsc channel for memory efficiency
use crate::ring_buffer::ConcurrentRingBuffer;

/// A completed bar with metadata identifying its source.
#[derive(Debug, Clone)]
pub struct CompletedBar {
    pub symbol: String,
    pub threshold_decimal_bps: u32,
    pub bar: RangeBar,
}

/// Metrics for the live bar engine.
/// Issue #96 Task #6: Added backpressure metrics for monitoring queue behavior
/// Issue #96 Task #12: Expose ring buffer metrics (dropped bars, queue depth)
#[derive(Debug)]
pub struct LiveEngineMetrics {
    pub trades_received: AtomicU64,
    pub bars_emitted: AtomicU64,
    pub reconnections: AtomicU64,
    pub backpressure_events: AtomicU64,  // Times ring buffer was full and bar dropped
    pub dropped_bars: AtomicU64,         // Total bars dropped due to full ring buffer
    pub max_queue_depth: AtomicU64,      // Maximum observed queue depth
    pub total_block_time_ms: AtomicU64,  // Accumulated time producer waited for queue space
}

impl Default for LiveEngineMetrics {
    fn default() -> Self {
        Self {
            trades_received: AtomicU64::new(0),
            bars_emitted: AtomicU64::new(0),
            reconnections: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
            dropped_bars: AtomicU64::new(0),
            max_queue_depth: AtomicU64::new(0),
            total_block_time_ms: AtomicU64::new(0),
        }
    }
}

/// Configuration for the live bar engine.
#[derive(Debug, Clone)]
pub struct LiveEngineConfig {
    /// Symbols to stream (e.g., ["BTCUSDT", "ETHUSDT"])
    pub symbols: Vec<String>,
    /// Thresholds in decimal basis points (e.g., [250, 500, 750, 1000])
    pub thresholds: Vec<u32>,
    /// Whether to compute inter-bar + intra-bar microstructure features
    pub include_microstructure: bool,
    /// Channel capacity for completed bars (backpressure)
    pub bar_channel_capacity: usize,
    /// Reconnection policy for WebSocket connections
    pub reconnection_policy: ReconnectionPolicy,
    /// Initial checkpoints keyed by (symbol, threshold) for resuming incomplete bars
    pub initial_checkpoints: HashMap<(String, u32), Checkpoint>,
    /// Issue #128: Per-feature computation toggles
    pub compute_tier2: bool,
    pub compute_tier3: bool,
    pub compute_hurst: Option<bool>,
    pub compute_permutation_entropy: Option<bool>,
}

impl LiveEngineConfig {
    /// Get bar channel capacity from environment or use default (10K)
    /// Issue #96 Task #6: RANGEBAR_MAX_PENDING_BARS env var support
    fn get_bar_channel_capacity() -> usize {
        std::env::var("RANGEBAR_MAX_PENDING_BARS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10_000)
    }

    /// Create config with sensible defaults.
    pub fn new(symbols: Vec<String>, thresholds: Vec<u32>) -> Self {
        Self {
            symbols,
            thresholds,
            include_microstructure: true,
            bar_channel_capacity: Self::get_bar_channel_capacity(),
            reconnection_policy: ReconnectionPolicy::default(),
            initial_checkpoints: HashMap::new(),
            compute_tier2: true,
            compute_tier3: false,
            compute_hurst: None,
            compute_permutation_entropy: None,
        }
    }

    /// Inject a checkpoint for a specific (symbol, threshold) pair.
    /// Must be called before `LiveBarEngine::start()`.
    pub fn with_checkpoint(mut self, symbol: String, threshold: u32, checkpoint: Checkpoint) -> Self {
        self.initial_checkpoints.insert((symbol, threshold), checkpoint);
        self
    }

    /// Set bar channel capacity explicitly (overrides env var)
    pub fn with_bar_channel_capacity(mut self, capacity: usize) -> Self {
        self.bar_channel_capacity = capacity;
        self
    }
}

/// Live bar engine: multiplexes WebSocket streams → canonical processors → completed bars.
///
/// Issue #91: Uses `RangeBarProcessor` (NOT `ExportRangeBarProcessor`) for full
/// 3-step feature finalization producing all 58 columns.
pub struct LiveBarEngine {
    config: LiveEngineConfig,
    /// Issue #96 Task #9: Ring buffer replaces mpsc for 10-20% memory savings
    bar_buffer: ConcurrentRingBuffer<CompletedBar>,
    shutdown: CancellationToken,
    metrics: Arc<LiveEngineMetrics>,
    started: bool,
    /// Channel for receiving processor checkpoints on shutdown
    checkpoint_rx: Option<mpsc::Receiver<(String, u32, Checkpoint)>>,
    checkpoint_tx: mpsc::Sender<(String, u32, Checkpoint)>,
}

impl LiveBarEngine {
    /// Create a new live bar engine.
    ///
    /// Does NOT start streaming — call `start()` after creation.
    pub fn new(config: LiveEngineConfig) -> Self {
        // Issue #96 Task #9: Initialize ring buffer instead of mpsc channel
        let bar_buffer = ConcurrentRingBuffer::new(config.bar_channel_capacity);
        // Channel for extracting checkpoints on shutdown (one per symbol×threshold)
        let max_checkpoints = config.symbols.len() * config.thresholds.len();
        let (checkpoint_tx, checkpoint_rx) = mpsc::channel(max_checkpoints.max(1));
        Self {
            config,
            bar_buffer,
            shutdown: CancellationToken::new(),
            metrics: Arc::new(LiveEngineMetrics::default()),
            started: false,
            checkpoint_rx: Some(checkpoint_rx),
            checkpoint_tx,
        }
    }

    /// Inject a checkpoint for a specific (symbol, threshold) pair.
    /// Must be called before `start()`.
    pub fn set_initial_checkpoint(&mut self, symbol: &str, threshold: u32, checkpoint: Checkpoint) {
        self.config.initial_checkpoints.insert(
            (symbol.to_uppercase(), threshold),
            checkpoint,
        );
    }

    /// Start all WebSocket connections and processing loops.
    ///
    /// Spawns one tokio task per symbol. Each task:
    /// 1. Connects via `run_with_reconnect()` (independent backoff per symbol)
    /// 2. Fans trades out to N `RangeBarProcessor` instances (one per threshold)
    /// 3. Sends completed bars to the shared output channel
    pub fn start(&mut self) -> Result<(), ProcessingError> {
        if self.started {
            return Ok(());
        }

        for symbol in &self.config.symbols {
            let symbol = symbol.to_uppercase();
            let thresholds = self.config.thresholds.clone();
            let include_microstructure = self.config.include_microstructure;
            // Issue #128: Build tier-aware InterBarConfig + IntraBarConfig
            let inter_bar_config = InterBarConfig {
                compute_tier2: self.config.compute_tier2,
                compute_tier3: self.config.compute_tier3,
                compute_hurst: self.config.compute_hurst,
                compute_permutation_entropy: self.config.compute_permutation_entropy,
                ..Default::default()
            };
            let intra_hurst = self.config.compute_hurst.unwrap_or(self.config.compute_tier3);
            let intra_pe = self.config.compute_permutation_entropy.unwrap_or(self.config.compute_tier3);
            let intra_bar_config = IntraBarConfig {
                compute_hurst: intra_hurst,
                compute_permutation_entropy: intra_pe,
            };
            // Issue #96 Task #9: Clone ring buffer reference for symbol task
            let bar_buffer = self.bar_buffer.clone_ref();
            let shutdown = self.shutdown.clone();
            let policy = self.config.reconnection_policy.clone();
            let metrics = Arc::clone(&self.metrics);
            let checkpoint_tx = self.checkpoint_tx.clone();

            // Create processors for this symbol (one per threshold)
            let mut processors: Vec<(u32, RangeBarProcessor)> = Vec::with_capacity(thresholds.len());
            for &threshold in &thresholds {
                // Try to restore from checkpoint, fall back to fresh processor
                let key = (symbol.clone(), threshold);
                let p = if let Some(cp) = self.config.initial_checkpoints.remove(&key) {
                    match RangeBarProcessor::from_checkpoint(cp) {
                        Ok(mut restored) => {
                            // Re-enable microstructure features after checkpoint restore
                            if include_microstructure {
                                restored = restored
                                    .with_inter_bar_config(inter_bar_config.clone())
                                    .with_intra_bar_features()
                                    .with_intra_bar_config(intra_bar_config.clone());
                            }
                            tracing::info!(
                                %symbol, threshold,
                                has_incomplete_bar = restored.get_incomplete_bar().is_some(),
                                "processor restored from checkpoint"
                            );
                            restored
                        }
                        Err(e) => {
                            tracing::warn!(
                                %symbol, threshold, ?e,
                                "checkpoint restore failed, starting fresh"
                            );
                            let mut fresh = RangeBarProcessor::new(threshold)?;
                            if include_microstructure {
                                fresh = fresh
                                    .with_inter_bar_config(inter_bar_config.clone())
                                    .with_intra_bar_features()
                                    .with_intra_bar_config(intra_bar_config.clone());
                            }
                            fresh
                        }
                    }
                } else {
                    let mut fresh = RangeBarProcessor::new(threshold)?;
                    if include_microstructure {
                        fresh = fresh
                            .with_inter_bar_config(inter_bar_config.clone())
                            .with_intra_bar_features()
                            .with_intra_bar_config(intra_bar_config.clone());
                    }
                    fresh
                };
                processors.push((threshold, p));
            }

            // Spawn per-symbol task
            tokio::spawn(symbol_task(
                symbol,
                processors,
                bar_buffer,
                checkpoint_tx,
                policy,
                shutdown,
                metrics,
            ));
        }

        self.started = true;
        tracing::info!(
            symbols = ?self.config.symbols,
            thresholds = ?self.config.thresholds,
            microstructure = self.config.include_microstructure,
            "live bar engine started"
        );
        Ok(())
    }

    /// Receive next completed bar. Returns `None` on timeout or shutdown.
    /// Issue #96 Task #9: Poll ring buffer with timeout, replacing async channel
    pub async fn next_bar(&mut self, timeout: Duration) -> Option<CompletedBar> {
        let start = tokio::time::Instant::now();
        let poll_interval = Duration::from_micros(100);  // Poll every 100μs

        loop {
            // Try to pop from ring buffer (non-blocking)
            if let Some(bar) = self.bar_buffer.pop() {
                return Some(bar);
            }

            // Check timeout
            if start.elapsed() >= timeout {
                return None;
            }

            // Check shutdown
            if self.shutdown.is_cancelled() {
                return None;
            }

            // Sleep briefly before polling again
            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Graceful shutdown — cancels all WebSocket tasks.
    pub fn stop(&self) {
        tracing::info!("live bar engine stopping");
        self.shutdown.cancel();
    }

    /// Get engine metrics snapshot.
    pub fn metrics(&self) -> &LiveEngineMetrics {
        &self.metrics
    }

    /// Whether the engine has been started.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Collect checkpoints from all processors after shutdown.
    ///
    /// Call after `stop()`. Each symbol task sends its processor checkpoints
    /// through the channel before exiting. Returns a map of `"SYMBOL:THRESHOLD"` → `Checkpoint`.
    pub async fn collect_checkpoints(&mut self, timeout: Duration) -> HashMap<String, Checkpoint> {
        let mut result = HashMap::new();
        let rx = match self.checkpoint_rx.take() {
            Some(rx) => rx,
            None => return result,
        };

        // Drain all available checkpoints within timeout
        let deadline = tokio::time::Instant::now() + timeout;
        let mut rx = rx;
        loop {
            tokio::select! {
                item = rx.recv() => {
                    match item {
                        Some((symbol, threshold, cp)) => {
                            let key = format!("{symbol}:{threshold}");
                            result.insert(key, cp);
                        }
                        None => break, // Channel closed, all senders dropped
                    }
                }
                () = tokio::time::sleep_until(deadline) => {
                    tracing::warn!(
                        collected = result.len(),
                        "checkpoint collection timed out"
                    );
                    break;
                }
            }
        }

        tracing::info!(count = result.len(), "checkpoints collected");
        result
    }

    /// Get the shutdown token (for external coordination).
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }
}

impl LiveEngineMetrics {
    /// Snapshot of current metrics.
    pub fn snapshot(&self) -> LiveEngineMetricsSnapshot {
        LiveEngineMetricsSnapshot {
            trades_received: self.trades_received.load(Ordering::Relaxed),
            bars_emitted: self.bars_emitted.load(Ordering::Relaxed),
            reconnections: self.reconnections.load(Ordering::Relaxed),
            dropped_bars: self.dropped_bars.load(Ordering::Relaxed),
            max_queue_depth: self.max_queue_depth.load(Ordering::Relaxed),
            backpressure_events: self.backpressure_events.load(Ordering::Relaxed),
        }
    }

    /// Get ring buffer queue depth estimate (from snapshot timing).
    /// Note: This is approximate due to concurrent updates.
    pub fn estimate_queue_depth(&self) -> u64 {
        let emitted = self.bars_emitted.load(Ordering::Relaxed);
        let dropped = self.dropped_bars.load(Ordering::Relaxed);
        let received = self.trades_received.load(Ordering::Relaxed);
        // Rough estimate: bars emitted + dropped vs trades received
        // This is not exact but gives a sense of queue pressure
        if emitted + dropped > received {
            0
        } else {
            received - emitted - dropped
        }
    }
}

/// Immutable metrics snapshot for reporting.
#[derive(Debug, Clone)]
pub struct LiveEngineMetricsSnapshot {
    pub trades_received: u64,
    pub bars_emitted: u64,
    pub reconnections: u64,
    pub dropped_bars: u64,
    pub max_queue_depth: u64,
    pub backpressure_events: u64,
}

/// Per-symbol WebSocket task with independent reconnection.
/// Issue #91: Each symbol gets its own backoff state (barter-rs pattern).
async fn symbol_task(
    symbol: String,
    mut processors: Vec<(u32, RangeBarProcessor)>,
    bar_buffer: ConcurrentRingBuffer<CompletedBar>,  // Issue #96 Task #9: Ring buffer replaces mpsc
    checkpoint_tx: mpsc::Sender<(String, u32, Checkpoint)>,
    policy: ReconnectionPolicy,
    shutdown: CancellationToken,
    metrics: Arc<LiveEngineMetrics>,
) {
    // Trade channel for this symbol's WebSocket
    let (trade_tx, mut trade_rx) = mpsc::channel::<AggTrade>(1000);

    // Spawn reconnecting WebSocket connection
    let ws_shutdown = shutdown.clone();
    let ws_symbol = symbol.clone();
    tokio::spawn(async move {
        BinanceWebSocketStream::run_with_reconnect(
            &ws_symbol,
            trade_tx,
            policy,
            ws_shutdown,
        )
        .await;
    });

    tracing::info!(%symbol, thresholds = ?processors.iter().map(|(t, _)| *t).collect::<Vec<_>>(), "symbol task started");

    // Process trades → bars
    loop {
        tokio::select! {
            trade = trade_rx.recv() => {
                match trade {
                    Some(trade) => {
                        metrics.trades_received.fetch_add(1, Ordering::Relaxed);

                        // Issue #96 Task #78: Fan out using borrowed trade (0 clones)
                        // Previously cloned trade for each threshold processor (~57 bytes each).
                        // With &trade, eliminates 4-8x unnecessary allocations per trade.
                        for (threshold, processor) in &mut processors {
                            match processor.process_single_trade(&trade) {
                                Ok(Some(bar)) => {
                                    metrics.bars_emitted.fetch_add(1, Ordering::Relaxed);
                                    let completed = CompletedBar {
                                        symbol: symbol.clone(),
                                        threshold_decimal_bps: *threshold,
                                        bar,
                                    };
                                    // Issue #96 Task #9: Push to ring buffer (non-blocking, may drop old bars)
                                    let was_added = bar_buffer.push(completed);
                                    if !was_added {
                                        // Ring buffer was full, old bar was dropped
                                        metrics.dropped_bars.fetch_add(1, Ordering::Relaxed);
                                        metrics.backpressure_events.fetch_add(1, Ordering::Relaxed);
                                        tracing::warn!(%symbol, threshold, "ring buffer full, old bar dropped");
                                    }
                                    // Issue #96 Task #12: Track queue depth metrics
                                    let current_depth = bar_buffer.len() as u64;
                                    let _ = metrics.max_queue_depth.fetch_max(current_depth, Ordering::Relaxed);
                                }
                                Ok(None) => {
                                    // Trade absorbed, no bar completed yet
                                }
                                Err(e) => {
                                    tracing::warn!(%symbol, threshold = *threshold, ?e, "trade processing error");
                                }
                            }
                        }
                    }
                    None => {
                        // WebSocket channel closed (terminal error or shutdown)
                        tracing::info!(%symbol, "trade channel closed");
                        break;
                    }
                }
            }
            () = shutdown.cancelled() => {
                tracing::info!(%symbol, "shutdown requested");
                break;
            }
        }
    }

    // Extract checkpoints from all processors before task exits
    for (threshold, processor) in &processors {
        let cp = processor.create_checkpoint(&symbol);
        tracing::info!(
            %symbol, threshold,
            has_incomplete_bar = cp.has_incomplete_bar(),
            "checkpoint extracted on shutdown"
        );
        if checkpoint_tx.send((symbol.clone(), *threshold, cp)).await.is_err() {
            tracing::warn!(%symbol, threshold, "checkpoint channel closed");
        }
    }

    tracing::info!(%symbol, "symbol task ended");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rangebar_core::FixedPoint;

    fn make_trade(id: i64, price: f64, timestamp_ms: u64, is_buyer_maker: bool) -> AggTrade {
        let price_str = format!("{price:.8}");
        AggTrade {
            agg_trade_id: id,
            price: FixedPoint::from_str(&price_str).unwrap(),
            volume: FixedPoint::from_str("1.00000000").unwrap(),
            first_trade_id: id,
            last_trade_id: id,
            timestamp: rangebar_core::normalize_timestamp(timestamp_ms),
            is_buyer_maker,
            is_best_match: None,
        }
    }

    #[test]
    fn test_live_engine_config_defaults() {
        let config = LiveEngineConfig::new(
            vec!["BTCUSDT".into(), "ETHUSDT".into()],
            vec![250, 500, 1000],
        );
        assert_eq!(config.symbols.len(), 2);
        assert_eq!(config.thresholds.len(), 3);
        assert!(config.include_microstructure);
        // Default from get_bar_channel_capacity() when RANGEBAR_MAX_PENDING_BARS is unset
        assert_eq!(config.bar_channel_capacity, 10_000);
        // Issue #128: Verify tier config defaults
        assert!(config.compute_tier2);
        assert!(!config.compute_tier3);
        assert_eq!(config.compute_hurst, None);
        assert_eq!(config.compute_permutation_entropy, None);
    }

    #[test]
    fn test_completed_bar_metadata() {
        let trade = make_trade(1, 50000.0, 1700000000000, false);
        let bar = RangeBar::new(&trade);
        let completed = CompletedBar {
            symbol: "BTCUSDT".into(),
            threshold_decimal_bps: 250,
            bar,
        };
        assert_eq!(completed.symbol, "BTCUSDT");
        assert_eq!(completed.threshold_decimal_bps, 250);
    }

    #[test]
    fn test_live_engine_creation() {
        let config = LiveEngineConfig::new(
            vec!["BTCUSDT".into()],
            vec![250],
        );
        let engine = LiveBarEngine::new(config);
        assert!(!engine.is_started());
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = LiveEngineMetrics::default();
        metrics.trades_received.store(100, Ordering::Relaxed);
        metrics.bars_emitted.store(5, Ordering::Relaxed);
        metrics.reconnections.store(2, Ordering::Relaxed);
        metrics.dropped_bars.store(1, Ordering::Relaxed);
        metrics.max_queue_depth.store(50, Ordering::Relaxed);
        metrics.backpressure_events.store(3, Ordering::Relaxed);

        let snap = metrics.snapshot();
        assert_eq!(snap.trades_received, 100);
        assert_eq!(snap.bars_emitted, 5);
        assert_eq!(snap.reconnections, 2);
        assert_eq!(snap.dropped_bars, 1);
        assert_eq!(snap.max_queue_depth, 50);
        assert_eq!(snap.backpressure_events, 3);
    }

    #[test]
    fn test_ring_buffer_metrics() {
        // Issue #96 Task #12: Verify ring buffer metrics tracking
        let metrics = LiveEngineMetrics::default();

        // Initially all zeros
        assert_eq!(metrics.dropped_bars.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 0);

        // Simulate ring buffer drops
        metrics.dropped_bars.fetch_add(5, Ordering::Relaxed);
        metrics.backpressure_events.fetch_add(5, Ordering::Relaxed);
        let _ = metrics.max_queue_depth.fetch_max(100, Ordering::Relaxed);

        assert_eq!(metrics.dropped_bars.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 100);

        // Verify snapshot captures all metrics
        let snap = metrics.snapshot();
        assert_eq!(snap.dropped_bars, 5);
        assert_eq!(snap.backpressure_events, 5);
        assert_eq!(snap.max_queue_depth, 100);
    }

    #[tokio::test]
    async fn test_engine_start_and_stop() {
        let config = LiveEngineConfig::new(
            vec!["BTCUSDT".into()],
            vec![250],
        );
        let mut engine = LiveBarEngine::new(config);

        // Start should succeed (spawns tasks but WS won't connect in test)
        assert!(engine.start().is_ok());
        assert!(engine.is_started());

        // Double-start is idempotent
        assert!(engine.start().is_ok());

        // Stop
        engine.stop();

        // next_bar should return None after stop
        let bar = engine.next_bar(Duration::from_millis(50)).await;
        assert!(bar.is_none());
    }

    #[tokio::test]
    async fn test_processor_fan_out() {
        // Verify that trades fan out to all threshold processors correctly
        let thresholds = vec![250u32, 500, 1000];
        let mut processors: Vec<(u32, RangeBarProcessor)> = thresholds
            .iter()
            .map(|&t| (t, RangeBarProcessor::new(t).unwrap()))
            .collect();

        // Process a series of trades — each processor should maintain independent state
        let base_price = 50000.0;
        for i in 0..10 {
            let trade = make_trade(
                i,
                base_price + (i as f64) * 0.01,
                1700000000000 + (i as u64) * 1000,
                i % 2 == 0,
            );
            for (_, processor) in &mut processors {
                let _ = processor.process_single_trade(&trade);
            }
        }

        // All processors should have processed trades independently
        for (threshold, processor) in &processors {
            let incomplete = processor.get_incomplete_bar();
            assert!(incomplete.is_some(), "threshold {threshold} should have an incomplete bar");
        }
    }

    #[tokio::test]
    async fn test_microstructure_processor_creation() {
        // Verify processors can be created with full microstructure features
        let threshold = 250u32;
        let p = RangeBarProcessor::new(threshold)
            .unwrap()
            .with_inter_bar_config(InterBarConfig::default())
            .with_intra_bar_features();

        assert!(p.inter_bar_enabled());
        assert!(p.intra_bar_enabled());
    }

    #[test]
    fn test_ring_buffer_drop_tracking() {
        // Issue #96 Task #12: Verify ring buffer correctly tracks dropped bars
        use crate::ring_buffer::ConcurrentRingBuffer;

        let ring_buf = ConcurrentRingBuffer::new(3);
        let metrics = LiveEngineMetrics::default();

        // Fill the buffer
        let bar1 = CompletedBar {
            symbol: "BTC".into(),
            threshold_decimal_bps: 250,
            bar: RangeBar::new(&make_trade(1, 50000.0, 1700000000000, false)),
        };
        let bar2 = CompletedBar {
            symbol: "BTC".into(),
            threshold_decimal_bps: 250,
            bar: RangeBar::new(&make_trade(2, 50001.0, 1700000000001, false)),
        };
        let bar3 = CompletedBar {
            symbol: "BTC".into(),
            threshold_decimal_bps: 250,
            bar: RangeBar::new(&make_trade(3, 50002.0, 1700000000002, false)),
        };
        let bar4 = CompletedBar {
            symbol: "BTC".into(),
            threshold_decimal_bps: 250,
            bar: RangeBar::new(&make_trade(4, 50003.0, 1700000000003, false)),
        };

        assert!(ring_buf.push(bar1));
        assert!(ring_buf.push(bar2));
        assert!(ring_buf.push(bar3));

        // Buffer is full (capacity=3), so next push should fail and drop oldest
        assert!(!ring_buf.push(bar4));

        // Simulate metrics tracking (as symbol_task would do)
        metrics.dropped_bars.fetch_add(1, Ordering::Relaxed);
        metrics.backpressure_events.fetch_add(1, Ordering::Relaxed);
        let current_depth = ring_buf.len() as u64;
        let _ = metrics.max_queue_depth.fetch_max(current_depth, Ordering::Relaxed);

        assert_eq!(metrics.dropped_bars.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 3);

        // Verify we can still retrieve bars
        assert_eq!(ring_buf.pop().is_some(), true); // bar2
        assert_eq!(ring_buf.pop().is_some(), true); // bar3
        assert_eq!(ring_buf.pop().is_some(), true); // bar4
        assert_eq!(ring_buf.pop().is_none(), true);  // empty
    }

    #[test]
    fn test_live_engine_metrics_estimate_queue_depth() {
        // Issue #96 Task #12: Verify queue depth estimation
        let metrics = LiveEngineMetrics::default();

        metrics.trades_received.store(1000, Ordering::Relaxed);
        metrics.bars_emitted.store(500, Ordering::Relaxed);
        metrics.dropped_bars.store(0, Ordering::Relaxed);

        let estimated = metrics.estimate_queue_depth();
        assert_eq!(estimated, 500); // 1000 - 500 - 0
    }
}
