# rangebar-providers

Data providers for multiple financial markets with unified `AggTrade` interface.

## Overview

`rangebar-providers` implements provider adapters for different data sources, normalizing them into the common `AggTrade` format required by `rangebar-core`. Currently supports Binance (crypto) and Exness (forex).

## Supported Providers

### Binance

Cryptocurrency aggTrades across multiple markets:

- **Spot Market**: Standard cryptocurrency spot trading pairs (default)
- **UM Futures (USD-M)**: USDT/USDC-margined perpetual futures
- **CM Futures (Coin-M)**: Coin-margined perpetual futures

#### Features

- Historical data loading via `binance_historical_data` crate
- Automatic CSV/Parquet format detection
- Tier-1 symbol discovery (18 symbols across all 3 markets)
- Timestamp normalization (spot: 16-digit μs, futures: 13-digit ms → 16-digit μs)

#### Usage

```rust
use rangebar_providers::binance::HistoricalDataLoader;

// Load spot market data (default)
let loader = HistoricalDataLoader::new("BTCUSDT");
let trades = loader.load_recent_day().await?;

// Load UM futures market data
let loader = HistoricalDataLoader::new_with_market("BTCUSDT", "um");
let trades = loader.load_historical_range(7).await?;
```

#### Tier-1 Symbols

Discover symbols available across all three Binance markets:

```rust
use rangebar_providers::binance::{get_tier1_symbols, get_tier1_usdt_pairs};

// Get symbol list (18 symbols)
let symbols = get_tier1_symbols(); // ["BTC", "ETH", "SOL", ...]

// Get USDT pairs
let pairs = get_tier1_usdt_pairs(); // ["BTCUSDT", "ETHUSDT", ...]
```

### Exness

Forex tick data for EURUSD Standard:

- **Variant**: EURUSD Standard (1.26M ticks/month, SNR=1.90)
- **Historical Range**: 2019-2025 validated
- **Tick Rate**: ~1.6M ticks/month average

#### Features

- ZIP archive fetching and extraction
- Bid/Ask/Timestamp CSV parsing
- Automatic timestamp conversion to microseconds
- Spread-based event detection

#### Usage

```rust
use rangebar_providers::exness::{ExnessFetcher, ExnessRangeBarBuilder};

// Fetch month of tick data
let fetcher = ExnessFetcher::new("EURUSD");
let ticks = fetcher.fetch_month(2024, 10).await?;

// Build range bars from ticks
let mut builder = ExnessRangeBarBuilder::new(
    5,  // 0.5 BPS threshold (5 × 0.1 BPS)
    "EURUSD",
    ValidationStrictness::Standard
)?;

for tick in ticks {
    if let Some(bar) = builder.process_tick(&tick)? {
        // Process completed bar
    }
}
```

## Provider Pattern

All providers normalize to the unified `AggTrade` interface from `rangebar-core`:

```rust
pub struct AggTrade {
    pub agg_trade_id: i64,
    pub price: FixedPoint,
    pub volume: FixedPoint,
    pub timestamp: i64,  // microseconds (normalized)
    // ... other fields
}
```

This enables:

- Unified processing pipeline across all data sources
- Easy addition of new providers (Deribit, Kraken, etc.)
- Temporal integrity via timestamp normalization

## Data Validation

### Binance Data Structure

Use `data-structure-validator` binary to verify schema:

```bash
cargo run --bin data-structure-validator
```

**Key Differences Detected**:

- **Spot**: No headers, short columns (`a,p,q,f,l,T,m`), 16-digit μs timestamps
- **UM Futures**: Headers, descriptive columns, 13-digit ms timestamps
- **Auto-normalization**: Parser detects format and normalizes to microseconds

### Exness Data Quality

**Validated Periods** (2019-2025):

- 2019-04, 2021-01, 2022-07
- 2023-04, 2024-10, 2025-01, 2025-02

**Best Trading Hours** (UTC):

- Hours 0-12, 14-17 (avoid hour 22 rollover, Sunday gaps)

**Event Detection**:

- Spread >1.4 pips = exit signal
- Use ±15min windows for event boundaries

## Dependencies

- **rangebar-core** - Core types and algorithm
- **binance_historical_data** - Binance data fetching
- **tokio** - Async runtime
- **reqwest** - HTTP client for Exness
- **zip** - ZIP archive extraction
- **csv** - CSV parsing

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Tier-1 symbols: Use `tier1-symbol-discovery` binary
- Data validation: Use `data-structure-validator` binary
- Architecture: `../../docs/ARCHITECTURE.md`
- Exness decision: `../../docs/planning/exness-eurusd-standard-final-decision.md`

## License

See LICENSE file in the repository root.
