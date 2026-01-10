//! WebSocket streaming for real-time Binance market data
//!
//! This module provides asynchronous WebSocket connections to Binance streams
//! for real-time aggTrade data feeding into range bar construction.

use futures_util::StreamExt;
use rangebar_core::{normalize_timestamp, AggTrade, FixedPoint};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};

/// WebSocket specific errors
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
}

impl From<tokio_tungstenite::tungstenite::Error> for WebSocketError {
    fn from(error: tokio_tungstenite::tungstenite::Error) -> Self {
        WebSocketError::ConnectionFailed(Box::new(error))
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
    ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    receiver: mpsc::Receiver<AggTrade>,
    _sender: mpsc::Sender<AggTrade>, // Keep sender alive
    connected: bool,
}

impl BinanceWebSocketStream {
    /// Create a new WebSocket stream for the given symbol
    pub async fn new(symbol: &str) -> Result<Self, WebSocketError> {
        let symbol = symbol.to_uppercase();

        // Validate symbol format
        if !symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(WebSocketError::InvalidSymbol(symbol));
        }

        let (sender, receiver) = mpsc::channel(1000);

        Ok(Self {
            symbol,
            ws_stream: None,
            receiver,
            _sender: sender,
            connected: false,
        })
    }

    /// Connect to the Binance WebSocket stream
    pub async fn connect(&mut self) -> Result<(), WebSocketError> {
        let url = format!(
            "wss://stream.binance.com:9443/ws/{}@aggTrade",
            self.symbol.to_lowercase()
        );

        println!("🔌 Connecting to WebSocket: {}", url);

        let (ws_stream, _) = connect_async(&url).await?;
        self.ws_stream = Some(ws_stream);
        self.connected = true;

        println!("✅ Connected to Binance WebSocket for {}", self.symbol);

        Ok(())
    }

    /// Start the message processing loop in a background task
    pub async fn start_processing(&mut self) -> Result<(), WebSocketError> {
        if let Some(mut ws_stream) = self.ws_stream.take() {
            let sender = self._sender.clone();
            let symbol = self.symbol.clone();

            tokio::spawn(async move {
                println!("🚀 Starting WebSocket message processing for {}", symbol);

                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(binance_trade) =
                                serde_json::from_str::<BinanceAggTrade>(&text)
                            {
                                if let Ok(agg_trade) = binance_trade.to_agg_trade() {
                                    if sender.send(agg_trade).await.is_err() {
                                        println!(
                                            "❌ Channel closed, stopping WebSocket processing"
                                        );
                                        break;
                                    }
                                } else {
                                    println!("⚠️ Failed to convert trade data");
                                }
                            } else {
                                println!("⚠️ Failed to parse JSON: {}", text);
                            }
                        }
                        Ok(Message::Close(_)) => {
                            println!("🔌 WebSocket connection closed by server");
                            break;
                        }
                        Ok(_) => {
                            // Ignore other message types (ping, pong, binary)
                        }
                        Err(e) => {
                            println!("❌ WebSocket error: {}", e);
                            break;
                        }
                    }
                }

                println!("🔚 WebSocket processing ended for {}", symbol);
            });
        }

        Ok(())
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

    #[tokio::test]
    async fn test_websocket_creation() {
        let stream = BinanceWebSocketStream::new("BTCUSDT").await;
        assert!(stream.is_ok());

        let stream = stream.unwrap();
        assert_eq!(stream.symbol(), "BTCUSDT");
        assert!(!stream.is_connected());
    }

    #[tokio::test]
    async fn test_invalid_symbol() {
        let stream = BinanceWebSocketStream::new("BTC-USD").await;
        assert!(stream.is_err());
        assert!(matches!(
            stream.unwrap_err(),
            WebSocketError::InvalidSymbol(_)
        ));
    }
}
