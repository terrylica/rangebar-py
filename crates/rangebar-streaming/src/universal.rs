//! Universal stream interface for all trade data sources (Issue #91)
//!
//! This module provides a unified interface for streaming trade data from
//! different sources (WebSocket, replay buffer, historical files) with
//! consistent speed control and mode switching.

use super::ReplayBuffer;
use async_trait::async_trait;
use rangebar_core::AggTrade;
#[cfg(feature = "binance-integration")]
use rangebar_providers::binance::{
    BinanceWebSocketStream, ReconnectionPolicy, WebSocketError,
};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use std::time::Duration;
#[cfg(feature = "binance-integration")]
use tokio::sync::mpsc;
#[cfg(feature = "binance-integration")]
use tokio_util::sync::CancellationToken;

/// Different modes for streaming trade data
#[derive(Debug, Clone, PartialEq)]
pub enum StreamMode {
    /// Real-time live data from WebSocket
    Live,
    /// Replay from buffer starting N minutes ago
    Replay { minutes_ago: u32, speed: f32 },
    /// Paused state (no new trades)
    Paused,
}

impl std::fmt::Display for StreamMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamMode::Live => write!(f, "LIVE"),
            StreamMode::Replay { minutes_ago, speed } => {
                write!(f, "REPLAY {}m @ {:.1}x", minutes_ago, speed)
            }
            StreamMode::Paused => write!(f, "PAUSED"),
        }
    }
}

/// Trait for unified trade streaming interface
#[async_trait]
pub trait TradeStream: Send {
    /// Get the next trade from the stream
    async fn next_trade(&mut self) -> Option<AggTrade>;

    /// Set the streaming mode
    async fn set_mode(&mut self, mode: StreamMode) -> Result<(), StreamError>;

    /// Get the current mode
    fn mode(&self) -> StreamMode;

    /// Check if the stream is connected/active
    fn is_active(&self) -> bool;

    /// Get progress information (current, total) for replay mode
    fn progress(&self) -> (usize, usize);

    /// Get the symbol this stream is for
    fn symbol(&self) -> &str;
}

/// Errors that can occur during streaming
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[cfg(feature = "binance-integration")]
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] WebSocketError),

    #[error("No data available for replay")]
    NoReplayData,

    #[error("Invalid mode transition from {from} to {to}")]
    InvalidModeTransition { from: String, to: String },

    #[error("Stream is not connected")]
    NotConnected,
}

/// Universal stream that can switch between live and replay modes.
/// Issue #91: Refactored to use channel-based WebSocket with auto-reconnection.
pub struct UniversalStream {
    symbol: String,
    #[cfg(feature = "binance-integration")]
    trade_rx: Option<mpsc::Receiver<AggTrade>>,
    #[cfg(feature = "binance-integration")]
    shutdown: Option<CancellationToken>,
    replay_buffer: ReplayBuffer,
    mode: StreamMode,
    replay_stream: Option<super::replay_buffer::ReplayStream>,
    speed_multiplier: Arc<AtomicU32>, // Fixed-point: 1000 = 1.0x
    connected: bool,
    trade_count: usize,
}

impl UniversalStream {
    /// Create a new universal stream for the given symbol
    pub async fn new(symbol: &str) -> Result<Self, StreamError> {
        let symbol = symbol.to_uppercase();
        let replay_buffer = ReplayBuffer::new(Duration::from_secs(3600)); // 1 hour buffer

        Ok(Self {
            symbol,
            #[cfg(feature = "binance-integration")]
            trade_rx: None,
            #[cfg(feature = "binance-integration")]
            shutdown: None,
            replay_buffer,
            mode: StreamMode::Live,
            replay_stream: None,
            speed_multiplier: Arc::new(AtomicU32::new(1000)), // 1.0x
            connected: false,
            trade_count: 0,
        })
    }

    /// Connect to the live WebSocket stream.
    /// Issue #91: Uses channel-based `run_with_reconnect()` with auto-reconnection.
    #[cfg(feature = "binance-integration")]
    pub fn connect(&mut self) -> Result<(), StreamError> {
        let _ = BinanceWebSocketStream::new(&self.symbol)?;

        let (trade_tx, trade_rx) = mpsc::channel(1000);
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();
        let symbol = self.symbol.clone();

        tokio::spawn(async move {
            BinanceWebSocketStream::run_with_reconnect(
                &symbol,
                trade_tx,
                ReconnectionPolicy::default(),
                shutdown_clone,
            )
            .await;
        });

        self.trade_rx = Some(trade_rx);
        self.shutdown = Some(shutdown);
        self.connected = true;
        tracing::info!(symbol = %self.symbol, "universal stream connected");

        Ok(())
    }

    /// Start background task to populate replay buffer from the trade channel.
    /// Issue #91: Reads from mpsc::Receiver instead of locking WebSocket directly.
    #[cfg(feature = "binance-integration")]
    pub fn start_buffer_population(&mut self) {
        // Take the receiver — buffer population task owns it exclusively.
        // After this, Live mode next_trade() returns None (trades flow through buffer).
        if let Some(mut rx) = self.trade_rx.take() {
            let replay_buffer = self.replay_buffer.clone();
            let symbol = self.symbol.clone();

            tokio::spawn(async move {
                tracing::info!(%symbol, "starting buffer population");

                while let Some(trade) = rx.recv().await {
                    replay_buffer.push(trade);
                }

                tracing::info!(%symbol, "buffer population ended");
            });
        }
    }

    /// Set the replay speed (only affects replay mode)
    pub fn set_speed(&self, multiplier: f32) {
        let fixed_point = (multiplier * 1000.0) as u32;
        self.speed_multiplier.store(fixed_point, Ordering::Relaxed);
    }

    /// Get the current speed multiplier
    pub fn speed(&self) -> f32 {
        self.speed_multiplier.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Get buffer statistics
    pub fn buffer_stats(&self) -> super::replay_buffer::ReplayBufferStats {
        self.replay_buffer.stats()
    }
}

#[async_trait]
impl TradeStream for UniversalStream {
    async fn next_trade(&mut self) -> Option<AggTrade> {
        match &self.mode {
            StreamMode::Live => {
                #[cfg(feature = "binance-integration")]
                {
                    if let Some(ref mut rx) = self.trade_rx {
                        let trade = rx.recv().await;
                        if trade.is_some() {
                            self.trade_count += 1;
                        }
                        trade
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "binance-integration"))]
                {
                    None
                }
            }
            StreamMode::Replay { .. } => {
                if let Some(ref mut stream) = self.replay_stream {
                    use tokio_stream::StreamExt;
                    let trade = stream.next().await;
                    if trade.is_some() {
                        self.trade_count += 1;
                    }
                    trade
                } else {
                    None
                }
            }
            StreamMode::Paused => {
                // In paused mode, don't emit any trades
                tokio::time::sleep(Duration::from_millis(100)).await;
                None
            }
        }
    }

    async fn set_mode(&mut self, mode: StreamMode) -> Result<(), StreamError> {
        match (&self.mode, &mode) {
            (StreamMode::Live, StreamMode::Replay { minutes_ago, speed }) => {
                // Switch from live to replay
                let trades = self.replay_buffer.get_trades_from(*minutes_ago);
                if trades.is_empty() {
                    return Err(StreamError::NoReplayData);
                }

                self.replay_stream = Some(self.replay_buffer.replay_from(*minutes_ago, *speed));
                tracing::info!(minutes_ago, speed, "switched to replay mode");
                self.mode = mode;
            }
            (StreamMode::Replay { .. }, StreamMode::Live) => {
                // Switch from replay to live
                self.replay_stream = None;
                self.mode = mode;
                tracing::info!("switched to live mode");
            }
            (_, StreamMode::Paused) => {
                // Can always pause
                self.mode = mode;
                tracing::info!("stream paused");
            }
            (StreamMode::Paused, _) => {
                // Resume to the new mode
                if let StreamMode::Replay { minutes_ago, speed } = &mode {
                    let trades = self.replay_buffer.get_trades_from(*minutes_ago);
                    if !trades.is_empty() {
                        self.replay_stream =
                            Some(self.replay_buffer.replay_from(*minutes_ago, *speed));
                    }
                }
                self.mode = mode;
                tracing::info!(mode = %self.mode, "stream resumed");
            }
            _ => {
                // Same mode or other transitions
                self.mode = mode;
            }
        }

        Ok(())
    }

    fn mode(&self) -> StreamMode {
        self.mode.clone()
    }

    fn is_active(&self) -> bool {
        self.connected && !matches!(self.mode, StreamMode::Paused)
    }

    fn progress(&self) -> (usize, usize) {
        if let Some(ref stream) = self.replay_stream {
            let total = stream.total();
            let remaining = stream.remaining();
            (total - remaining, total)
        } else {
            (self.trade_count, self.trade_count)
        }
    }

    fn symbol(&self) -> &str {
        &self.symbol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_universal_stream_creation() {
        let stream = UniversalStream::new("BTCUSDT").await;
        assert!(stream.is_ok());

        let stream = stream.unwrap();
        assert_eq!(stream.symbol(), "BTCUSDT");
        assert_eq!(stream.mode(), StreamMode::Live);
        assert!(!stream.is_active()); // Not connected yet
    }

    #[tokio::test]
    async fn test_mode_switching() {
        let mut stream = UniversalStream::new("BTCUSDT").await.unwrap();

        // Test pausing
        let result = stream.set_mode(StreamMode::Paused).await;
        assert!(result.is_ok());
        assert_eq!(stream.mode(), StreamMode::Paused);

        // Test resuming to live
        let result = stream.set_mode(StreamMode::Live).await;
        assert!(result.is_ok());
        assert_eq!(stream.mode(), StreamMode::Live);
    }

    #[test]
    fn test_speed_control() {
        let stream = UniversalStream::new("BTCUSDT");
        let stream = futures::executor::block_on(stream).unwrap();

        // Test setting and getting speed
        stream.set_speed(2.5);
        assert!((stream.speed() - 2.5).abs() < 0.1);

        stream.set_speed(0.5);
        assert!((stream.speed() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_stream_mode_display() {
        assert_eq!(StreamMode::Live.to_string(), "LIVE");
        assert_eq!(
            StreamMode::Replay {
                minutes_ago: 5,
                speed: 2.0
            }
            .to_string(),
            "REPLAY 5m @ 2.0x"
        );
        assert_eq!(StreamMode::Paused.to_string(), "PAUSED");
    }
}
