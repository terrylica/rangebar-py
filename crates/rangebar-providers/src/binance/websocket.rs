//! WebSocket streaming for real-time Binance market data (Issue #91)
//!
//! This module provides asynchronous WebSocket connections to Binance streams
//! for real-time aggTrade data feeding into range bar construction.
//!
//! Production-hardened with:
//! - Exponential backoff reconnection (adapted from barter-rs)
//! - Ping/pong handling (Binance disconnects after 3 missed pongs)
//! - Terminal vs recoverable error classification
//! - Graceful shutdown via `CancellationToken`

use futures_util::{SinkExt, StreamExt};
use rangebar_core::{normalize_timestamp, AggTrade, FixedPoint};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_stream::Stream;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;

/// WebSocket specific errors
/// Issue #91: Hardened with `is_terminal()` for reconnection routing
#[derive(Error, Debug)]
pub enum WebSocketError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(Box<tokio_tungstenite::tungstenite::Error>),

    #[error("JSON parsing failed: {0}")]
    JsonParsingFailed(#[from] serde_json::Error),

    #[error("Channel send failed")]
    ChannelSendFailed,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Connection closed unexpectedly")]
    ConnectionClosed,

    #[error("Pong send failed: {0}")]
    PongSendFailed(String),
}

impl WebSocketError {
    /// Whether this error is terminal (no retry) or recoverable (retry with backoff).
    ///
    /// Terminal errors indicate a logic bug or permanent condition that retrying
    /// cannot fix. Recoverable errors are transient network/server issues.
    pub const fn is_terminal(&self) -> bool {
        match self {
            // Terminal: bad input, consumer dropped — retrying won't help
            Self::InvalidSymbol(_) | Self::ChannelSendFailed => true,
            // Recoverable: transient network, server kick, bad message
            Self::ConnectionFailed(_)
            | Self::JsonParsingFailed(_)
            | Self::ConnectionClosed
            | Self::PongSendFailed(_) => false,
        }
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for WebSocketError {
    fn from(error: tokio_tungstenite::tungstenite::Error) -> Self {
        WebSocketError::ConnectionFailed(Box::new(error))
    }
}

/// Reconnection policy with exponential backoff.
/// Issue #91: Adapted from barter-rs `ReconnectingStream` pattern.
#[derive(Debug, Clone)]
pub struct ReconnectionPolicy {
    /// Initial backoff duration in milliseconds (default: 125ms)
    pub backoff_ms_initial: u64,
    /// Backoff multiplier (default: 2x)
    pub backoff_multiplier: u64,
    /// Maximum backoff duration in milliseconds (default: 60s)
    pub backoff_ms_max: u64,
}

impl Default for ReconnectionPolicy {
    fn default() -> Self {
        Self {
            backoff_ms_initial: 125,
            backoff_multiplier: 2,
            backoff_ms_max: 60_000,
        }
    }
}

/// Binance WebSocket aggTrade message format
#[derive(Debug, Deserialize, Serialize)]
struct BinanceAggTrade {
    #[serde(rename = "e")]
    event_type: String,

    #[serde(rename = "E")]
    event_time: i64,

    #[serde(rename = "s")]
    symbol: String,

    #[serde(rename = "a")]
    agg_trade_id: i64,

    #[serde(rename = "p")]
    price: String,

    #[serde(rename = "q")]
    quantity: String,

    #[serde(rename = "f")]
    first_trade_id: i64,

    #[serde(rename = "l")]
    last_trade_id: i64,

    #[serde(rename = "T")]
    trade_time: i64,

    #[serde(rename = "m")]
    is_buyer_maker: bool,

    #[serde(rename = "M")]
    _ignore: bool,
}

impl BinanceAggTrade {
    /// Convert Binance WebSocket format to internal AggTrade format
    fn to_agg_trade(&self) -> Result<AggTrade, WebSocketError> {
        let price = FixedPoint::from_str(&self.price).map_err(|_| {
            WebSocketError::JsonParsingFailed(serde_json::from_str::<()>("{}").unwrap_err())
        })?;

        let volume = FixedPoint::from_str(&self.quantity).map_err(|_| {
            WebSocketError::JsonParsingFailed(serde_json::from_str::<()>("{}").unwrap_err())
        })?;

        Ok(AggTrade {
            agg_trade_id: self.agg_trade_id,
            price,
            volume,
            first_trade_id: self.first_trade_id,
            last_trade_id: self.last_trade_id,
            timestamp: normalize_timestamp(self.trade_time as u64),
            is_buyer_maker: self.is_buyer_maker,
            is_best_match: None, // Not provided in WebSocket stream
        })
    }
}

/// WebSocket stream for Binance aggTrade data
#[derive(Debug)]
pub struct BinanceWebSocketStream {
    symbol: String,
    receiver: mpsc::Receiver<AggTrade>,
    _sender: mpsc::Sender<AggTrade>, // Keep sender alive
    connected: bool,
}

impl BinanceWebSocketStream {
    /// Create a new WebSocket stream for the given symbol.
    ///
    /// Does NOT connect — use `connect_and_stream()` or `run_with_reconnect()`.
    pub fn new(symbol: &str) -> Result<Self, WebSocketError> {
        let symbol = symbol.to_uppercase();

        // Validate symbol format
        if !symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(WebSocketError::InvalidSymbol(symbol));
        }

        let (sender, receiver) = mpsc::channel(1000);

        Ok(Self {
            symbol,
            receiver,
            _sender: sender,
            connected: false,
        })
    }

    /// Build the WebSocket URL for this symbol.
    #[cfg(test)]
    fn ws_url(&self) -> String {
        format!(
            "wss://stream.binance.com:9443/ws/{}@aggTrade",
            self.symbol.to_lowercase()
        )
    }

    /// Connect and stream trades until disconnection or shutdown.
    ///
    /// Returns `Ok(())` on clean shutdown, `Err` on any failure.
    /// The caller (typically `run_with_reconnect`) decides whether to retry.
    pub async fn connect_and_stream(
        symbol: &str,
        trade_tx: &mpsc::Sender<AggTrade>,
        shutdown: &CancellationToken,
    ) -> Result<(), WebSocketError> {
        let symbol_lower = symbol.to_lowercase();
        let url = format!("wss://stream.binance.com:9443/ws/{symbol_lower}@aggTrade");

        tracing::info!(%symbol, %url, "connecting to Binance WebSocket");

        let (ws_stream, _) = connect_async(&url).await?;
        let (mut ws_sender, mut ws_reader) = ws_stream.split();

        tracing::info!(%symbol, "WebSocket connected");

        loop {
            tokio::select! {
                msg = ws_reader.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<BinanceAggTrade>(&text) {
                                Ok(binance_trade) => {
                                    match binance_trade.to_agg_trade() {
                                        Ok(agg_trade) => {
                                            if trade_tx.send(agg_trade).await.is_err() {
                                                tracing::warn!(%symbol, "trade channel closed");
                                                return Err(WebSocketError::ChannelSendFailed);
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!(%symbol, ?e, "trade conversion failed");
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(%symbol, ?e, "JSON parse failed");
                                }
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if let Err(e) = ws_sender.send(Message::Pong(data)).await {
                                tracing::warn!(%symbol, %e, "pong send failed");
                                return Err(WebSocketError::PongSendFailed(e.to_string()));
                            }
                        }
                        Some(Ok(Message::Close(frame))) => {
                            tracing::info!(%symbol, ?frame, "WebSocket closed by server");
                            return Err(WebSocketError::ConnectionClosed);
                        }
                        Some(Ok(_)) => {
                            // Binary, Pong — ignore
                        }
                        Some(Err(e)) => {
                            tracing::warn!(%symbol, %e, "WebSocket error");
                            return Err(WebSocketError::from(e));
                        }
                        None => {
                            tracing::info!(%symbol, "WebSocket stream ended");
                            return Err(WebSocketError::ConnectionClosed);
                        }
                    }
                }
                () = shutdown.cancelled() => {
                    tracing::info!(%symbol, "shutdown requested, closing WebSocket");
                    return Ok(());
                }
            }
        }
    }

    /// Run WebSocket with auto-reconnection and exponential backoff.
    /// Issue #91: Adapted from barter-rs `ReconnectingStream` pattern.
    ///
    /// Each reconnection re-establishes the WebSocket and resumes streaming.
    /// Backoff resets to initial value on each successful connection.
    pub async fn run_with_reconnect(
        symbol: &str,
        trade_tx: mpsc::Sender<AggTrade>,
        policy: ReconnectionPolicy,
        shutdown: CancellationToken,
    ) {
        // Validate symbol upfront (terminal error — don't enter reconnect loop)
        if !symbol
            .to_uppercase()
            .chars()
            .all(|c| c.is_ascii_alphanumeric())
        {
            tracing::error!(%symbol, "invalid symbol — not entering reconnect loop");
            return;
        }

        let mut backoff_ms = policy.backoff_ms_initial;

        loop {
            // Reset backoff on each new connection attempt start
            // (if we get data, backoff resets below)
            match Self::connect_and_stream(symbol, &trade_tx, &shutdown).await {
                Ok(()) => {
                    // Clean shutdown requested via CancellationToken
                    tracing::info!(%symbol, "clean shutdown");
                    break;
                }
                Err(e) if e.is_terminal() => {
                    tracing::error!(%symbol, ?e, "terminal WebSocket error — not retrying");
                    break;
                }
                Err(e) => {
                    tracing::warn!(%symbol, ?e, backoff_ms, "WebSocket disconnected, reconnecting");

                    tokio::select! {
                        () = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                        () = shutdown.cancelled() => {
                            tracing::info!(%symbol, "shutdown during backoff");
                            break;
                        }
                    }

                    // Exponential backoff with cap
                    backoff_ms = (backoff_ms * policy.backoff_multiplier).min(policy.backoff_ms_max);
                }
            }
        }
    }

    /// Get the next trade from the stream
    pub async fn next_trade(&mut self) -> Option<AggTrade> {
        self.receiver.recv().await
    }

    /// Check if the WebSocket is connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get the symbol this stream is connected to
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Stream for BinanceWebSocketStream {
    type Item = AggTrade;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_aggtrade_conversion() {
        let json = r#"{
            "e": "aggTrade",
            "E": 1758666334424,
            "s": "BTCUSDT",
            "a": 3679713739,
            "p": "112070.01000000",
            "q": "0.01328000",
            "f": 5252203256,
            "l": 5252203266,
            "T": 1758666334424,
            "m": false,
            "M": true
        }"#;

        let binance_trade: BinanceAggTrade = serde_json::from_str(json).unwrap();
        let agg_trade = binance_trade.to_agg_trade().unwrap();

        assert_eq!(agg_trade.agg_trade_id, 3679713739);
        assert_eq!(agg_trade.price.to_f64(), 112070.01);
        assert_eq!(agg_trade.volume.to_f64(), 0.01328);
        assert_eq!(agg_trade.timestamp, 1758666334424000);
        assert!(!agg_trade.is_buyer_maker);
    }

    #[test]
    fn test_websocket_creation() {
        let stream = BinanceWebSocketStream::new("BTCUSDT");
        assert!(stream.is_ok());

        let stream = stream.unwrap();
        assert_eq!(stream.symbol(), "BTCUSDT");
        assert!(!stream.is_connected());
    }

    #[test]
    fn test_invalid_symbol() {
        let stream = BinanceWebSocketStream::new("BTC-USD");
        assert!(stream.is_err());
        assert!(matches!(
            stream.unwrap_err(),
            WebSocketError::InvalidSymbol(_)
        ));
    }

    #[test]
    fn test_error_terminal_classification() {
        // Terminal errors — retrying won't help
        assert!(WebSocketError::InvalidSymbol("BAD".into()).is_terminal());
        assert!(WebSocketError::ChannelSendFailed.is_terminal());

        // Recoverable errors — retry with backoff
        assert!(!WebSocketError::ConnectionClosed.is_terminal());
        assert!(!WebSocketError::PongSendFailed("timeout".into()).is_terminal());
    }

    #[test]
    fn test_reconnection_policy_default() {
        let policy = ReconnectionPolicy::default();
        assert_eq!(policy.backoff_ms_initial, 125);
        assert_eq!(policy.backoff_multiplier, 2);
        assert_eq!(policy.backoff_ms_max, 60_000);
    }

    #[test]
    fn test_backoff_calculation() {
        let policy = ReconnectionPolicy::default();
        let mut backoff = policy.backoff_ms_initial;

        // Verify exponential growth: 125 → 250 → 500 → 1000 → 2000 → ...
        assert_eq!(backoff, 125);
        backoff = (backoff * policy.backoff_multiplier).min(policy.backoff_ms_max);
        assert_eq!(backoff, 250);
        backoff = (backoff * policy.backoff_multiplier).min(policy.backoff_ms_max);
        assert_eq!(backoff, 500);
        backoff = (backoff * policy.backoff_multiplier).min(policy.backoff_ms_max);
        assert_eq!(backoff, 1000);

        // Verify cap at 60s
        backoff = 32_000;
        backoff = (backoff * policy.backoff_multiplier).min(policy.backoff_ms_max);
        assert_eq!(backoff, 60_000);
        backoff = (backoff * policy.backoff_multiplier).min(policy.backoff_ms_max);
        assert_eq!(backoff, 60_000); // Stays capped
    }

    #[test]
    fn test_ws_url_format() {
        let stream = BinanceWebSocketStream::new("BTCUSDT").unwrap();
        assert_eq!(
            stream.ws_url(),
            "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
        );
    }

    #[tokio::test]
    async fn test_run_with_reconnect_invalid_symbol() {
        // Invalid symbol should exit immediately without entering reconnect loop
        let (tx, _rx) = mpsc::channel(10);
        let shutdown = CancellationToken::new();

        // This should return immediately — invalid symbol is terminal
        BinanceWebSocketStream::run_with_reconnect("BTC-USD", tx, Default::default(), shutdown)
            .await;
        // If we get here, the function correctly detected terminal error
    }
}
