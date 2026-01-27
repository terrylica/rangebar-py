//! Binance data provider
//!
//! Integrations for Binance spot and futures markets:
//! - Historical aggTrades data (CSV/ZIP format)
//! - Tier-1 symbol discovery (multi-market analysis)
//! - Real-time WebSocket streams (aggTrades)
//! - SHA-256 checksum verification (Issue #43)
//!
//! ## Architecture
//!
//! - `historical` - CSV/ZIP historical data loader
//! - `symbols` - Tier-1 cryptocurrency symbol discovery
//! - `websocket` - Real-time aggTrades WebSocket client
//! - `checksum` - SHA-256 verification for data integrity
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

pub mod checksum;
pub mod historical;
pub mod symbols;
pub mod websocket;

// Re-export commonly used types
pub use checksum::{
    compute_sha256, fetch_and_verify, fetch_checksum, parse_checksum_file, verify_checksum,
    ChecksumError, ChecksumResult,
};
pub use historical::{
    detect_csv_headers, python_bool, CsvAggTrade, HistoricalDataLoader, HistoricalError,
    IntraDayChunkIterator,
};
pub use symbols::{get_tier1_symbols, get_tier1_usdt_pairs, is_tier1_symbol, TIER1_SYMBOLS};
pub use websocket::{BinanceWebSocketStream, WebSocketError};
