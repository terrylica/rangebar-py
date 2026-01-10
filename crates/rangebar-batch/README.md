# rangebar-batch

High-throughput batch analytics engine with parallel multi-symbol analysis.

## Overview

`rangebar-batch` provides batch processing capabilities for historical range bar analysis with parallel execution via Rayon. Optimized for research, backtesting, and comprehensive market analysis across multiple symbols.

## Features

- **Parallel Processing**: Multi-symbol analysis using Rayon thread pool
- **Comprehensive Statistics**: Price, volume, and trade count analytics
- **Multi-Symbol Support**: Analyze entire market segments simultaneously
- **High Throughput**: Scales linearly with CPU cores
- **In-Memory Operations**: Fast processing with full dataset in memory

## Usage

### Single Symbol Analysis

```rust
use rangebar_batch::BatchAnalysisEngine;

let engine = BatchAnalysisEngine::new();
let report = engine.analyze_single_symbol(&range_bars, "BTCUSDT")?;

println!("Total bars: {}", report.total_bars);
println!("Mean price: {}", report.price_statistics.mean);
println!("Total volume: {}", report.volume_statistics.total);
```

### Multi-Symbol Analysis

```rust
use rangebar_batch::BatchAnalysisEngine;
use std::collections::HashMap;

let engine = BatchAnalysisEngine::new();

// Prepare multi-symbol data
let mut data: HashMap<String, Vec<RangeBar>> = HashMap::new();
data.insert("BTCUSDT".to_string(), btc_bars);
data.insert("ETHUSDT".to_string(), eth_bars);
data.insert("SOLUSDT".to_string(), sol_bars);

// Parallel analysis
let reports = engine.analyze_multiple_symbols(data)?;

for report in reports {
    println!("{}: {} bars, ${:.2} mean",
        report.symbol,
        report.total_bars,
        report.price_statistics.mean
    );
}
```

## Analysis Reports

```rust
pub struct AnalysisReport {
    pub symbol: String,
    pub total_bars: usize,
    pub price_statistics: PriceStats,
    pub volume_statistics: VolumeStats,
    pub time_statistics: TimeStats,
    pub trade_statistics: TradeStats,
}

pub struct PriceStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

pub struct VolumeStats {
    pub total: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
}
```

## Tier-1 Batch Analysis

Use `rangebar-analyze` binary for comprehensive Tier-1 symbol analysis:

```bash
cargo run --bin rangebar-analyze --release
```

This processes all 18 Tier-1 symbols in parallel with detailed statistics.

## Performance Characteristics

- **Memory Usage**: O(N) for N bars - full dataset in memory
- **Throughput**: High (multi-threaded Rayon parallelism)
- **Parallelism**: Scales linearly with CPU cores (uses N-1 cores)
- **Use Case**: Historical backtesting, research, multi-market analysis

## Batch Configuration

```rust
pub struct BatchConfig {
    pub num_threads: usize,           // Rayon thread pool size
    pub chunk_size: usize,            // Processing chunk size
    pub enable_statistics: bool,      // Compute full statistics
}
```

## Comparison: Batch vs Streaming

| Feature      | Batch                  | Streaming       |
| ------------ | ---------------------- | --------------- |
| Memory Usage | Unbounded              | Bounded         |
| Throughput   | High                   | Moderate        |
| Parallelism  | Multi-threaded (Rayon) | Single-threaded |
| Real-time    | No                     | Yes             |
| Use Case     | Historical analysis    | Live trading    |

## Integration with Polars

`rangebar-batch` integrates with `rangebar-io` for DataFrame operations:

```rust
use rangebar_io::PolarsExporter;

// Export analysis results to Parquet
let exporter = PolarsExporter::new();
exporter.export_parquet(&range_bars, "analysis_output.parquet")?;
```

## Dependencies

- **rangebar-core** - Core algorithm and types
- **rangebar-io** - Polars integration
- **rayon** - Parallel processing
- **serde** - Serialization support

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Architecture: `../../docs/ARCHITECTURE.md`
- CLI tool: Run `rangebar-analyze` binary

## License

See LICENSE file in the repository root.
