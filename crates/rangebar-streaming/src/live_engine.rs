//! Live bar engine for real-time range bar construction (Issue #91)
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
use rangebar_core::{AggTrade, RangeBar};
use rangebar_providers::binance::{BinanceWebSocketStream, ReconnectionPolicy};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

/// A completed bar with metadata identifying its source.
#[derive(Debug, Clone)]
pub struct CompletedBar {
    pub symbol: String,
    pub threshold_decimal_bps: u32,
    pub bar: RangeBar,
}

/// Metrics for the live bar engine.
#[derive(Debug, Default)]
pub struct LiveEngineMetrics {
    pub trades_received: AtomicU64,
    pub bars_emitted: AtomicU64,
    pub reconnections: AtomicU64,
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
}

impl LiveEngineConfig {
    /// Create config with sensible defaults.
    pub fn new(symbols: Vec<String>, thresholds: Vec<u32>) -> Self {
        Self {
            symbols,
            thresholds,
            include_microstructure: true,
            bar_channel_capacity: 500,
            reconnection_policy: ReconnectionPolicy::default(),
            initial_checkpoints: HashMap::new(),
        }
    }

    /// Inject a checkpoint for a specific (symbol, threshold) pair.
    /// Must be called before `LiveBarEngine::start()`.
    pub fn with_checkpoint(mut self, symbol: String, threshold: u32, checkpoint: Checkpoint) -> Self {
        self.initial_checkpoints.insert((symbol, threshold), checkpoint);
        self
    }
}

/// Live bar engine: multiplexes WebSocket streams → canonical processors → completed bars.
///
/// Issue #91: Uses `RangeBarProcessor` (NOT `ExportRangeBarProcessor`) for full
/// 3-step feature finalization producing all 58 columns.
pub struct LiveBarEngine {
    config: LiveEngineConfig,
    bar_rx: mpsc::Receiver<CompletedBar>,
    bar_tx: mpsc::Sender<CompletedBar>,
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
        let (bar_tx, bar_rx) = mpsc::channel(config.bar_channel_capacity);
        // Channel for extracting checkpoints on shutdown (one per symbol×threshold)
        let max_checkpoints = config.symbols.len() * config.thresholds.len();
        let (checkpoint_tx, checkpoint_rx) = mpsc::channel(max_checkpoints.max(1));
        Self {
            config,
            bar_rx,
            bar_tx,
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
            let bar_tx = self.bar_tx.clone();
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
                                    .with_inter_bar_config(InterBarConfig::default())
                                    .with_intra_bar_features();
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
                                    .with_inter_bar_config(InterBarConfig::default())
                                    .with_intra_bar_features();
                            }
                            fresh
                        }
                    }
                } else {
                    let mut fresh = RangeBarProcessor::new(threshold)?;
                    if include_microstructure {
                        fresh = fresh
                            .with_inter_bar_config(InterBarConfig::default())
                            .with_intra_bar_features();
                    }
                    fresh
                };
                processors.push((threshold, p));
            }

            // Spawn per-symbol task
            tokio::spawn(symbol_task(
                symbol,
                processors,
                bar_tx,
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
    pub async fn next_bar(&mut self, timeout: Duration) -> Option<CompletedBar> {
        tokio::select! {
            result = self.bar_rx.recv() => result,
            () = tokio::time::sleep(timeout) => None,
            () = self.shutdown.cancelled() => None,
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
        }
    }
}

/// Immutable metrics snapshot for reporting.
#[derive(Debug, Clone)]
pub struct LiveEngineMetricsSnapshot {
    pub trades_received: u64,
    pub bars_emitted: u64,
    pub reconnections: u64,
}

/// Per-symbol WebSocket task with independent reconnection.
/// Issue #91: Each symbol gets its own backoff state (barter-rs pattern).
async fn symbol_task(
    symbol: String,
    mut processors: Vec<(u32, RangeBarProcessor)>,
    bar_tx: mpsc::Sender<CompletedBar>,
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

                        // Fan out to all threshold processors
                        for (threshold, processor) in &mut processors {
                            match processor.process_single_trade(trade.clone()) {
                                Ok(Some(bar)) => {
                                    metrics.bars_emitted.fetch_add(1, Ordering::Relaxed);
                                    let completed = CompletedBar {
                                        symbol: symbol.clone(),
                                        threshold_decimal_bps: *threshold,
                                        bar,
                                    };
                                    if bar_tx.send(completed).await.is_err() {
                                        tracing::warn!(%symbol, "bar channel closed, stopping symbol task");
                                        // Still extract checkpoints before returning
                                        break;
                                    }
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
        assert_eq!(config.bar_channel_capacity, 500);
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

        let snap = metrics.snapshot();
        assert_eq!(snap.trades_received, 100);
        assert_eq!(snap.bars_emitted, 5);
        assert_eq!(snap.reconnections, 2);
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
                let _ = processor.process_single_trade(trade.clone());
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
}
