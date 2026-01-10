//! # Tier-1 Symbol Discovery and Multi-Market Analysis
//!
//! This module provides functionality for discovering and analyzing Tier-1 cryptocurrency symbols
//! across multiple Binance futures markets.
//!
//! ## What are Tier-1 Instruments?
//!
//! Tier-1 instruments are cryptocurrency assets that Binance lists across **ALL THREE** futures markets:
//!
//! 1. **UM Futures (USDT-margined)**: Traditional perpetual contracts (e.g., BTCUSDT)
//! 2. **UM Futures (USDC-margined)**: Stablecoin-margined perpetuals (e.g., BTCUSDC)
//! 3. **CM Futures (Coin-margined)**: Inverse perpetual contracts (e.g., BTCUSD_PERP)
//!
//! ## Current Tier-1 Symbols (18 total)
//!
//! - **BTC** - Bitcoin
//! - **ETH** - Ethereum
//! - **SOL** - Solana
//! - **ADA** - Cardano
//! - **AVAX** - Avalanche
//! - **BCH** - Bitcoin Cash
//! - **BNB** - Binance Coin
//! - **DOGE** - Dogecoin
//! - **FIL** - Filecoin
//! - **LINK** - Chainlink
//! - **LTC** - Litecoin
//! - **NEAR** - Near Protocol
//! - **UNI** - Uniswap
//! - **XRP** - Ripple
//! - **AAVE** - Aave
//! - **SUI** - Sui
//! - **WIF** - Dogwifhat
//! - **WLD** - Worldcoin
//!
//! ## Key Characteristics
//!
//! - **Multi-market availability** indicates Binance's highest confidence in the asset
//! - **Premium liquidity** across all three settlement currencies
//! - **Institutional interest** and professional trading focus
//! - **Cross-market arbitrage** opportunities
//! - **Reliability analysis** across different margining systems
//!
//! ## Usage with `tier1-symbol-discovery` Binary
//!
//! The main functionality is provided through the `tier1-symbol-discovery` binary:
//!
//! ```bash
//! # Discover all Tier-1 symbols (comprehensive database)
//! cargo run --bin tier1-symbol-discovery
//!
//! # Generate minimal output for pipeline integration
//! cargo run --bin tier1-symbol-discovery -- --format minimal
//!
//! # Include single-market symbols in analysis
//! cargo run --bin tier1-symbol-discovery -- --include-single-market
//!
//! # Add custom suffix to output files
//! cargo run --bin tier1-symbol-discovery -- --custom-suffix range_bar_ready
//! ```
//!
//! ## Output Files
//!
//! ### JSON Databases
//!
//! Generated in `output/symbol_analysis/current/`:
//!
//! - **Comprehensive**: Complete multi-market database with all contract details
//! - **Minimal**: Focused on Tier-1 symbols for lightweight processing
//! - **Spot-equivalent**: Mapping between futures and spot market symbols
//!
//! ### Pipeline Integration
//!
//! - **`/tmp/tier1_usdt_pairs.txt`**: Simple list of USDT pairs for downstream processing
//! - Machine-readable format for integration with `rangebar-analyze` and other tools
//!
//! ## Integration with Range Bar Analysis
//!
//! Tier-1 symbols are particularly suitable for range bar construction due to:
//!
//! - **High liquidity**: Consistent tick flow for reliable bar formation
//! - **Low spreads**: Minimal price gaps that could affect range calculations
//! - **24/7 trading**: Continuous market data availability
//! - **Multi-market validation**: Cross-reference data quality across markets
//!
//! ## Performance
//!
//! - **Symbol Discovery**: 577 UM + 59 CM symbols analyzed in ~1 second
//! - **Live API Integration**: Direct Binance API connectivity
//! - **Real-time Analysis**: Up-to-date symbol availability checking
//! - **Efficient Processing**: Pure Rust implementation with minimal dependencies

/// Tier-1 symbol list (as of the latest analysis)
pub const TIER1_SYMBOLS: &[&str] = &[
    "AAVE", "ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "ETH", "FIL", "LINK", "LTC", "NEAR", "SOL",
    "SUI", "UNI", "WIF", "WLD", "XRP",
];

/// Check if a symbol is a Tier-1 instrument
///
/// # Examples
///
/// ```
/// use rangebar_providers::binance::symbols::is_tier1_symbol;
///
/// assert!(is_tier1_symbol("BTC"));
/// assert!(is_tier1_symbol("ETH"));
/// assert!(!is_tier1_symbol("SHIB"));
/// ```
pub fn is_tier1_symbol(symbol: &str) -> bool {
    TIER1_SYMBOLS.contains(&symbol.to_uppercase().as_str())
}

/// Get all Tier-1 symbols as a vector
///
/// # Examples
///
/// ```
/// use rangebar_providers::binance::symbols::get_tier1_symbols;
///
/// let symbols = get_tier1_symbols();
/// assert_eq!(symbols.len(), 18);
/// assert!(symbols.contains(&"BTC".to_string()));
/// ```
pub fn get_tier1_symbols() -> Vec<String> {
    TIER1_SYMBOLS.iter().map(|s| s.to_string()).collect()
}

/// Get Tier-1 USDT perpetual pairs
///
/// # Examples
///
/// ```
/// use rangebar_providers::binance::symbols::get_tier1_usdt_pairs;
///
/// let pairs = get_tier1_usdt_pairs();
/// assert!(pairs.contains(&"BTCUSDT".to_string()));
/// assert!(pairs.contains(&"ETHUSDT".to_string()));
/// ```
pub fn get_tier1_usdt_pairs() -> Vec<String> {
    TIER1_SYMBOLS.iter().map(|s| format!("{}USDT", s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier1_symbol_count() {
        assert_eq!(TIER1_SYMBOLS.len(), 18);
    }

    #[test]
    fn test_is_tier1_symbol() {
        assert!(is_tier1_symbol("BTC"));
        assert!(is_tier1_symbol("btc"));
        assert!(is_tier1_symbol("ETH"));
        assert!(!is_tier1_symbol("SHIB"));
        assert!(!is_tier1_symbol("PEPE"));
    }

    #[test]
    fn test_get_tier1_symbols() {
        let symbols = get_tier1_symbols();
        assert_eq!(symbols.len(), 18);
        assert!(symbols.contains(&"BTC".to_string()));
        assert!(symbols.contains(&"ETH".to_string()));
    }

    #[test]
    fn test_get_tier1_usdt_pairs() {
        let pairs = get_tier1_usdt_pairs();
        assert_eq!(pairs.len(), 18);
        assert!(pairs.contains(&"BTCUSDT".to_string()));
        assert!(pairs.contains(&"ETHUSDT".to_string()));
    }
}
