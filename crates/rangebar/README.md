# rangebar (meta-crate)

Unified interface for the rangebar workspace with backward compatibility for v4.0.0 API.

## Overview

The `rangebar` meta-crate provides a unified entry point to all rangebar functionality by re-exporting all sub-crates. It also maintains backward compatibility with the v4.0.0 monolithic API through legacy module paths.

## Quick Start

```rust
use rangebar::prelude::*;

// Core types and algorithm
let mut processor = RangeBarProcessor::new(250)?; // 250 = 25 BPS = 0.25%
let bars = processor.process_agg_trade_records(&trades)?;

// Data providers
let loader = HistoricalDataLoader::new("BTCUSDT");
let trades = loader.load_recent_day().await?;

// Export to multiple formats
let exporter = PolarsExporter::new();
exporter.export_parquet(&bars, "output.parquet")?;
```

## Module Structure

### Direct Crate Access

```rust
use rangebar::core::*;          // rangebar-core
use rangebar::providers::*;     // rangebar-providers
use rangebar::config::*;        // rangebar-config
use rangebar::io::*;            // rangebar-io
use rangebar::streaming::*;     // rangebar-streaming
use rangebar::batch::*;         // rangebar-batch
```

### Legacy v4.0.0 Compatibility

For backward compatibility with existing code:

```rust
// Legacy module paths (v4.0.0)
use rangebar::fixed_point::FixedPoint;
use rangebar::range_bars::ExportRangeBarProcessor;
use rangebar::types::{AggTrade, RangeBar};
use rangebar::tier1::{get_tier1_symbols, get_tier1_usdt_pairs};
use rangebar::data::HistoricalDataLoader;
```

## Feature Flags

The meta-crate supports optional features for selective compilation:

```toml
[dependencies]
rangebar = { version = "6.1.0", features = ["full"] }
```

### Available Features

- **`core`** (default): Core algorithm and types
- **`providers`**: Data providers (Binance, Exness)
- **`config`**: Configuration management
- **`io`**: Export formats and Polars integration
- **`streaming`**: Real-time streaming processor
- **`batch`**: Batch analytics engine
- **`full`**: All features enabled

### Feature Combinations

```toml
# Minimal: Core algorithm only
rangebar = { version = "6.1.0" }

# Core + providers
rangebar = { version = "6.1.0", features = ["providers"] }

# Core + providers + streaming
rangebar = { version = "6.1.0", features = ["streaming"] }

# Everything
rangebar = { version = "6.1.0", features = ["full"] }
```

## Architecture

The rangebar workspace consists of 8 specialized crates:

### Core Crates

- **rangebar-core**: Algorithm, types, fixed-point arithmetic
- **rangebar-providers**: Binance, Exness data providers
- **rangebar-config**: Configuration management
- **rangebar-io**: Export formats, Polars integration

### Engine Crates

- **rangebar-streaming**: Real-time streaming processor
- **rangebar-batch**: Batch analytics engine

### Tools & Compatibility

- **rangebar-cli**: Command-line tools
- **rangebar**: This meta-crate (unified interface)

## Migrating from v4.0.0

The v5.0.0 release introduced a modular crate architecture. Your existing code should continue to work due to backward compatibility shims, but consider migrating to direct crate imports:

### Before (v4.0.0)

```rust
use rangebar::fixed_point::FixedPoint;
use rangebar::range_bars::ExportRangeBarProcessor;
use rangebar::data::HistoricalDataLoader;
```

### After (v5.0.0)

```rust
use rangebar_core::FixedPoint;
use rangebar_core::RangeBarProcessor;
use rangebar_providers::binance::HistoricalDataLoader;
```

**Migration Guide**: See `../../docs/planning/` for detailed migration instructions.

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

**Breaking Changes in v3.0.0**:

- Threshold values changed from 1 BPS to 0.1 BPS units (multiply by 10)
- Example: 25 BPS now requires `new(250)` instead of `new(25)`

## Common Use Cases

### Historical Analysis

```rust
use rangebar::prelude::*;

// Load historical data
let loader = HistoricalDataLoader::new("BTCUSDT");
let trades = loader.load_historical_range(7).await?;

// Generate range bars
let mut processor = RangeBarProcessor::new(250)?;
let bars = processor.process_agg_trade_records(&trades)?;

// Export to Parquet
let exporter = ParquetExporter::new();
exporter.export(&bars, "btcusdt_bars.parquet")?;
```

### Real-Time Streaming

```rust
use rangebar::prelude::*;

let mut processor = StreamingProcessor::new(250)?;
let metrics = processor.process_stream(websocket_stream).await?;

println!("Processed {} trades", metrics.trades_processed);
```

### Multi-Symbol Batch Analysis

```rust
use rangebar::prelude::*;

let engine = BatchAnalysisEngine::new();
let reports = engine.analyze_multiple_symbols(multi_symbol_data)?;
```

## Documentation

- **Architecture**: `../../docs/ARCHITECTURE.md`
- **Examples**: `../../examples/`
- **CLI Tools**: See `rangebar-cli` crate
- **Individual Crates**: Each crate has its own README.md

## Dependencies

The meta-crate re-exports all workspace crates based on enabled features.

## License

See LICENSE file in the repository root.
