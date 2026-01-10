# rangebar-cli

Command-line tools for range bar processing, analysis, and validation.

## Overview

`rangebar-cli` consolidates all command-line binaries for the rangebar workspace. Provides tools for symbol discovery, data validation, batch analysis, benchmarking, and temporal validation.

## Available Tools (6 binaries)

### tier1-symbol-discovery

Discover Tier-1 cryptocurrency symbols across Binance markets:

```bash
# Comprehensive output with market matrix
cargo run --bin tier1-symbol-discovery -- --format comprehensive

# Minimal output (symbols only)
cargo run --bin tier1-symbol-discovery -- --format minimal
```

**Output**: 18 Tier-1 symbols available across spot, UM futures, and CM futures markets.

### data-structure-validator

Validate Binance aggTrades data structure across markets:

```bash
# Validate all Tier-1 symbols across spot/um/cm markets
cargo run --bin data-structure-validator --release

# Validate specific markets
cargo run --bin data-structure-validator -- --markets spot,um

# Custom date range
cargo run --bin data-structure-validator -- \
  --start-date 2024-01-01 --end-date 2024-12-31
```

**Features**:

- Cross-market schema detection (spot vs futures differences)
- Timestamp precision validation (16-digit μs vs 13-digit ms)
- SHA256 checksum verification (optional)
- Parallel processing with configurable workers

### rangebar-analyze

Parallel Tier-1 batch analysis (formerly `parallel-tier1-analysis`):

```bash
# Analyze with default settings
cargo run --bin rangebar-analyze --release -- \
  --symbol BTCUSDT --threshold 250

# Multi-symbol parallel analysis
cargo run --bin rangebar-analyze --release -- \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --threshold 250
```

**Features**:

- Multi-symbol parallel analysis using Rayon
- Comprehensive statistics generation
- JSON output with analysis reports

### spot-tier1-processor

Batch processor for all Tier-1 spot symbols:

```bash
# Process all 18 Tier-1 symbols in parallel
cargo run --bin spot-tier1-processor --release -- \
  --start-date 2024-07-01 --end-date 2024-10-31 --threshold-decimal-bps 250

# Custom parallelism
cargo run --bin spot-tier1-processor -- --workers 16
```

**Features**:

- Parallel execution using Rayon (default: 8 workers)
- Comprehensive execution statistics
- JSON metadata with symbol performance rankings
- Automatic output file naming

### polars-benchmark

Benchmark Polars integration performance:

```bash
cargo run --bin polars-benchmark -- \
  --input ./data/BTCUSDT_bars.csv \
  --output-dir ./benchmark_output
```

**Tests**:

- Parquet export (70%+ compression target)
- Arrow IPC export (zero-copy Python)
- Streaming CSV export (2x-5x speedup target)
- General Polars performance

### temporal-integrity-validator

Validate temporal integrity of range bar data:

```bash
cargo run --bin temporal-integrity-validator -- \
  --input ./data/BTCUSDT_bars.csv
```

**Validates**:

- Monotonic timestamp ordering
- DataFrame operation safety
- Export readiness without round-trip conversion

## Tool Categories

### Discovery & Validation

- `tier1-symbol-discovery` - Symbol discovery
- `data-structure-validator` - Data validation
- `temporal-integrity-validator` - Temporal validation

### Processing & Analysis

- `rangebar-analyze` - Parallel batch analysis
- `spot-tier1-processor` - Batch Tier-1 processing

### Benchmarking

- `polars-benchmark` - Performance benchmarks

## Common Flags

All tools support standard flags:

```bash
--help              # Show comprehensive help
--version           # Show version
--verbose, -v       # Verbose output
```

## Dependencies

`rangebar-cli` uses all workspace crates:

- **rangebar-core** - Core algorithm
- **rangebar-providers** - Data providers
- **rangebar-config** - Configuration
- **rangebar-io** - Export formats
- **rangebar-streaming** - Streaming processing
- **rangebar-batch** - Batch processing

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Architecture: [`/docs/ARCHITECTURE.md`](/docs/ARCHITECTURE.md)
- Algorithm spec: [`/docs/specifications/algorithm-spec.md`](/docs/specifications/algorithm-spec.md)
- Each tool has comprehensive `--help` documentation

## License

See LICENSE file in the repository root.
