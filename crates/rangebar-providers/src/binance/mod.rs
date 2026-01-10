//! Binance data provider
//!
//! Integrations for Binance spot and futures markets:
//! - Historical aggTrades data (CSV/ZIP format)
//! - Tier-1 symbol discovery (multi-market analysis)
//! - Real-time WebSocket streams (aggTrades)
//!
//! ## Architecture
//!
//! - `historical` - CSV/ZIP historical data loader
//! - `symbols` - Tier-1 cryptocurrency symbol discovery
//! - `websocket` - Real-time aggTrades WebSocket client
//!
//! ## Data Sources
//!
//! - **Spot Market**: https://data.binance.vision/
//! - **UM Futures**: USDT-margined perpetuals
//! - **CM Futures**: Coin-margined perpetuals
//!
//! ## Tier-1 Instruments
//!
//! Cryptocurrencies available across ALL THREE Binance futures markets.
//! Current count: 18 symbols (BTC, ETH, SOL, ADA, etc.)

pub mod historical;
pub mod symbols;
pub mod websocket;

// Re-export commonly used types
pub use historical::{detect_csv_headers, python_bool, CsvAggTrade, HistoricalDataLoader};
pub use symbols::{get_tier1_symbols, get_tier1_usdt_pairs, is_tier1_symbol, TIER1_SYMBOLS};
pub use websocket::{BinanceWebSocketStream, WebSocketError};
