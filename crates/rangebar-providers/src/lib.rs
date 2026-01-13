//! Data provider integrations
//!
//! Source-specific adapters for fetching and processing tick/trade data.
//!
//! ## Supported Providers
//!
//! - `binance` - Binance spot and futures markets (primary - crypto)
//! - `exness` - Exness EURUSD Standard tick data (primary - forex)
//!
//! ## Provider Selection
//!
//! | Asset Class | Provider | Rationale |
//! |-------------|----------|-----------|
//! | Crypto | Binance | Official data, high volume, REST + WebSocket |
//! | Forex | Exness | Zero rate limiting, 100% reliability, simple format |
//!
//! ## Adding New Providers
//!
//! Follow the established pattern:
//!
//! ```text
//! providers/
//! └── [provider_name]/
//!     ├── mod.rs          # Public API and documentation
//!     ├── client.rs       # HTTP client or WebSocket
//!     ├── types.rs        # Provider-specific data structures
//!     ├── builder.rs      # Range bar builder (if custom logic needed)
//!     └── conversion.rs   # Convert to AggTrade format
//! ```
//!
//! ## Design Principles
//!
//! 1. **Adapter pattern**: Convert provider format → AggTrade (core format)
//! 2. **Error propagation**: Raise immediately, no silent failures
//! 3. **Stateless where possible**: Cache externally, not in provider
//! 4. **Documented edge cases**: Timezone handling, decimal factors, etc.
//! 5. **Out-of-box dependencies**: Use standard crates (zip, csv, chrono)
//!
//! ## Quick Start
//!
//! Top-level re-exports allow shorter import paths:
//!
//! ```rust,ignore
//! // Top-level imports (recommended)
//! use rangebar_providers::{
//!     HistoricalDataLoader,
//!     get_tier1_symbols,
//!     ExnessFetcher,
//! };
//! ```
//!
//! Or use submodule paths directly:
//!
//! ```rust,ignore
//! // Submodule imports (also supported)
//! use rangebar_providers::binance::HistoricalDataLoader;
//! use rangebar_providers::exness::ExnessFetcher;
//! ```

#[cfg(feature = "binance")]
pub mod binance;

#[cfg(feature = "exness")]
pub mod exness;

// ============================================================================
// Public API Re-exports
// ============================================================================
//
// Commonly used types re-exported at the top level for convenience.
// Users can import from either:
//   - Top level: `use rangebar_providers::HistoricalDataLoader;`
//   - Submodule: `use rangebar_providers::binance::HistoricalDataLoader;`
//
// Both paths are supported and equivalent.

// Binance provider re-exports (alphabetically sorted)
// Includes: historical data loading, Tier-1 symbol discovery, WebSocket streaming, intra-day chunking
#[cfg(feature = "binance")]
pub use binance::{
    detect_csv_headers, get_tier1_symbols, get_tier1_usdt_pairs, is_tier1_symbol, python_bool,
    BinanceWebSocketStream, CsvAggTrade, HistoricalDataLoader, HistoricalError,
    IntraDayChunkIterator, WebSocketError, TIER1_SYMBOLS,
};

// Exness provider re-exports (alphabetically sorted)
// Includes: client, builder, types, and errors
#[cfg(feature = "exness")]
pub use exness::{
    ConversionError, ExnessError, ExnessFetcher, ExnessInstrument, ExnessRangeBar,
    ExnessRangeBarBuilder, ExnessTick, SpreadStats, ValidationStrictness,
};
