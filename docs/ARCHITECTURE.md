# Architecture Overview

**Parent**: [/CLAUDE.md](/CLAUDE.md)

**rangebar-py** is a unified Rust workspace with Python bindings. This document describes the 8-crate modular architecture.

## Workspace Structure

```
rangebar-py/
├── Cargo.toml                 # Workspace root + PyO3 cdylib
├── src/lib.rs                 # PyO3 bindings (Python entry point)
├── crates/                    # 8 Rust crates (all publish = false)
│   ├── rangebar-core/         # Layer 0: Core algorithm
│   ├── rangebar-config/       # Layer 0: Configuration
│   ├── rangebar-providers/    # Layer 1: Data sources
│   ├── rangebar-io/           # Layer 1: I/O operations
│   ├── rangebar-streaming/    # Layer 2: Real-time processing
│   ├── rangebar-batch/        # Layer 2: Batch analytics
│   ├── rangebar-cli/          # Layer 3: CLI tools (disabled)
│   └── rangebar/              # Layer 3: Meta-crate
├── python/rangebar/           # Python API layer
└── pyproject.toml             # Maturin configuration
```

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

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| **rangebar-config** | Configuration management | `Settings` |
| **rangebar-core** | Core algorithm (minimal deps) | `AggTrade`, `RangeBar`, `FixedPoint`, `RangeBarProcessor` |

**rangebar-core** features:
- 8-decimal fixed-point arithmetic (avoids floating-point errors)
- Non-lookahead breach detection (temporal integrity)
- `test-utils` feature for CSV loading
- `python` feature for PyO3 types
- `api` feature for utoipa OpenAPI

### Layer 1: Data Access

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| **rangebar-providers** | Data sources | `HistoricalDataLoader`, `ExnessFetcher` |
| **rangebar-io** | I/O operations | Polars DataFrame integration |

**rangebar-providers** features:
- `binance` - Binance spot/UM/CM futures (aggTrades)
- `exness` - 10 forex pairs (EURUSD, GBPUSD, etc.)
- `all-providers` - Enable all

**rangebar-io** features:
- `parquet` - Polars Parquet I/O
- Export to CSV, Parquet, Arrow IPC

### Layer 2: Processing Engines

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| **rangebar-streaming** | Real-time processor | `StreamingProcessor`, `StreamingConfig` |
| **rangebar-batch** | Batch analytics | `BatchAnalysisEngine`, `AnalysisReport` |

**rangebar-streaming** features:
- Bounded memory (configurable limits)
- Circuit breaker pattern
- `stats` - Streaming statistics
- `indicators` - Technical indicators

**rangebar-batch** features:
- Rayon parallel processing
- High-throughput batch analysis

### Layer 3: Applications

| Crate | Purpose | Status |
|-------|---------|--------|
| **rangebar-cli** | 6 CLI tools | Disabled for PyPI |
| **rangebar** | Meta-crate | v4.0 backward compatibility |

**CLI binaries** (source preserved, not built):
- `tier1-symbol-discovery`
- `rangebar-analyze`
- `data-structure-validator`
- `spot-tier1-processor`
- `polars-benchmark`
- `temporal-integrity-validator`

## Python Integration

The Python layer sits on top of the Rust workspace:

```
Python API (python/rangebar/)
├── __init__.py          # get_range_bars(), process_trades_*()
├── clickhouse/          # Tier 2 cache (ClickHouse)
└── storage/             # Tier 1 cache (Parquet)
        │
        ▼
PyO3 Bindings (src/lib.rs)
├── PyRangeBarProcessor  # Wraps rangebar-core::RangeBarProcessor
├── PyAggTrade           # Wraps rangebar-core::AggTrade
└── Data fetching        # Uses rangebar-providers (optional feature)
        │
        ▼
Local Crates (crates/)
└── rangebar-core, rangebar-providers, etc.
```

## Build Configuration

### Workspace Inheritance

All crates inherit from workspace:
```toml
[package]
publish.workspace = true  # Inherits false - no crates.io
version.workspace = true  # Inherits 6.1.1
edition.workspace = true  # Inherits 2024
```

### Optimization Profiles

```toml
[profile.release]
lto = "thin"           # Cross-platform LTO
codegen-units = 1      # Maximum optimization
# panic = "abort"      # FORBIDDEN with PyO3!

[profile.wheel]
inherits = "release"
strip = "symbols"      # Minimize wheel size
```

### PyO3 Configuration

```toml
[lib]
crate-type = ["cdylib"]  # Python extension module

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }
```

The `abi3-py39` feature enables stable ABI, producing **one wheel per platform** instead of one per Python version.

## Development Workflow

```bash
# Install tools
mise install

# Build Python extension
mise run build           # maturin develop

# Run tests
mise run test            # Rust tests (excludes PyO3 crate)
mise run test-py         # Python tests

# Quality checks
mise run check-full      # fmt + lint + test + deny

# Benchmarks
mise run bench:run       # Full benchmarks
mise run bench:validate  # Verify 1M ticks < 100ms
```

## Design Decisions

1. **PyPI-only publishing**: All crates have `publish = false`. Only Python wheels are distributed.

2. **Modular architecture**: 8 specialized crates enable:
   - Independent testing
   - Optional features (e.g., streaming without batch)
   - Clear separation of concerns

3. **Local crates**: No external crates.io dependencies for rangebar-* crates. All source is in this repo.

4. **CLI disabled**: CLI binaries are preserved as source but not built, focusing the project on Python API.

5. **Thin LTO**: Chosen over fat LTO for cross-platform compatibility (macOS + Linux builds).

## Related Documentation

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/crates/CLAUDE.md](/crates/CLAUDE.md) - Detailed crate documentation
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python layer details
- [/docs/api.md](/docs/api.md) - Python API reference
- [/docs/development/RELEASE.md](/docs/development/RELEASE.md) - Release workflow
