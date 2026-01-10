# Crates Workspace

Detailed context for the 8-crate modular workspace architecture.

**Parent**: [`/CLAUDE.md`](/CLAUDE.md) | **Architecture**: [`/docs/ARCHITECTURE.md`](/docs/ARCHITECTURE.md)

## Dependency Graph

```
                    rangebar (meta-crate, v4.0 compat)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   rangebar-cli          rangebar-batch      rangebar-streaming
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   │                     │
             rangebar-io          rangebar-providers
                   │                     │
                   └──────────┬──────────┘
                              │
                        rangebar-core
                              │
                        rangebar-config
```

## Crate Details

### Layer 0: Foundation

**rangebar-config** - Configuration management

- Dependencies: `config`, `serde`
- Public API: `Settings::load()`, `Settings::default()`

**rangebar-core** - Core algorithm (minimal deps)

- Dependencies: `chrono`, `serde`, `serde_json`, `thiserror`, `ahash`
- Features: `test-utils` (CSV loading), `python` (PyO3), `api` (utoipa)
- Public types: `AggTrade`, `RangeBar`, `FixedPoint`, `RangeBarProcessor`
- Key: 8-decimal fixed-point arithmetic, non-lookahead breach detection

### Layer 1: Data Access

**rangebar-providers** - Data sources

- Depends on: `rangebar-core`
- Features: `binance` (default), `exness`, `all-providers`
- Binance: `HistoricalDataLoader`, spot/UM/CM futures, aggTrades
- Exness: `ExnessFetcher`, `ExnessInstrument` enum (10 forex pairs)

**rangebar-io** - I/O operations

- Depends on: `rangebar-core`
- Features: `parquet` (enables Polars), `all`
- Polars integration for DataFrame ops
- Export: CSV, Parquet, Arrow IPC

### Layer 2: Engines

**rangebar-streaming** - Real-time processor

- Depends on: `rangebar-core`, `rangebar-providers` (optional)
- Features: `binance-integration`, `stats`, `indicators`, `all`
- Bounded memory, circuit breaker pattern
- Public API: `StreamingProcessor`, `StreamingConfig`

**rangebar-batch** - Batch analytics

- Depends on: `rangebar-core`, `rangebar-io`
- High-throughput parallel processing (Rayon)
- Public API: `BatchAnalysisEngine`, `BatchConfig`, `AnalysisReport`

### Layer 3: Tools

**rangebar-cli** - All binaries (6 tools)

- Depends on: ALL other crates
- Location: `src/bin/`

| Binary | Purpose |
|--------|---------|
| `tier1-symbol-discovery` | Multi-market symbol analysis |
| `rangebar-analyze` | Parallel tier-1 batch analysis |
| `data-structure-validator` | Cross-market format verification |
| `spot-tier1-processor` | Spot market processing |
| `polars-benchmark` | Performance benchmarking |
| `temporal-integrity-validator` | Temporal integrity tests |

**rangebar** - Meta-crate (v4.0 backward compat)

- Re-exports all sub-crates
- Legacy module paths: `fixed_point`, `range_bars`, `types`, `tier1`, `data`

## Common Patterns

### Adding a New Binary

1. Create `crates/rangebar-cli/src/bin/my_tool.rs`
2. Add to `Cargo.toml`:
   ```toml
   [[bin]]
   name = "my-tool"
   path = "src/bin/my_tool.rs"
   ```

### Feature Flags

```bash
# Build with specific features
cargo build -p rangebar-providers --features exness
cargo build -p rangebar-core --features test-utils

# Build CLI with all features
cargo build -p rangebar-cli --release
```

### Running Binaries

```bash
cargo run --bin tier1-symbol-discovery -- --format comprehensive
cargo run --bin rangebar-analyze -- --symbol BTCUSDT --threshold 250
cargo run --bin data-structure-validator
```

## Testing

```bash
# Test specific crate
cargo nextest run -p rangebar-core

# Test with features
cargo nextest run -p rangebar-core --features test-utils

# Test all crates
cargo nextest run --workspace
```
