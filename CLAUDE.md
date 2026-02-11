# CLAUDE.md - Project Memory

**rangebar-py**: Rust workspace with Python bindings via PyO3/maturin. Publishes to PyPI only (not crates.io).

---

## Quick Reference

| Task                    | Entry Point                                                                   | Details                                            |
| ----------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------- |
| Generate range bars     | `get_range_bars()`                                                            | [Python API](#python-api)                          |
| Understand architecture | [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)                                 | 8-crate workspace                                  |
| Work with Rust crates   | [crates/CLAUDE.md](/crates/CLAUDE.md)                                         | Core algorithm, microstructure, inter-bar features |
| Work with Python layer  | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)                       | API, caching, validation, symbol registry          |
| ClickHouse cache ops    | [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) | Schema, population, remote setup                   |
| Operations & scripts    | [scripts/CLAUDE.md](/scripts/CLAUDE.md)                                       | Pueue, cache population, per-year parallelism      |
| Release workflow        | [docs/development/RELEASE.md](/docs/development/RELEASE.md)                   | Zig cross-compile, mise tasks                      |
| Performance monitoring  | [docs/development/PERFORMANCE.md](/docs/development/PERFORMANCE.md)           | Benchmarks, metrics                                |
| Project context         | [docs/CONTEXT.md](/docs/CONTEXT.md)                                           | Why this project exists                            |
| API reference           | [docs/api/INDEX.md](/docs/api/INDEX.md)                                       | Full Python API docs                               |
| Research                | [docs/research/INDEX.md](/docs/research/INDEX.md)                             | ML labeling, regime patterns, TDA                  |
| Oracle verification     | [docs/verification/](/docs/verification/)                                     | Bit-exact cross-reference reports                  |

---

## Critical Principle: Leverage Rust

**ALWAYS use local Rust crates for heavy lifting.** Python handles I/O only.

1. **Check Rust first**: Before writing Python code, check if `rangebar-core` already provides the capability
2. **Stream to Rust**: The `RangeBarProcessor` maintains state between `process_trades()` calls
3. **Checkpoint API**: Use `create_checkpoint()` and `from_checkpoint()` for cross-session continuity
4. **No reinventing**: Don't reimplement range bar logic in Python

```python
# CORRECT: Stream to Rust (maintains state automatically)
for chunk in data_stream:
    bars = processor.process_trades(chunk.to_dicts())

# WRONG: Buffer in Python (OOM risk)
all_data = []
for chunk in data_stream:
    all_data.extend(chunk)
```

---

## Python API

```python
from rangebar import get_range_bars, get_n_range_bars, process_trades_polars

# Date-bounded (backtesting)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded (ML training)
df = get_n_range_bars("BTCUSDT", n_bars=10000)

# Polars users (2-3x faster)
import polars as pl
trades = pl.scan_parquet("trades.parquet")
df = process_trades_polars(trades, threshold_decimal_bps=250)

# With microstructure features (v7.0+)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", include_microstructure=True)

# Long ranges (>30 days) — must populate cache first
from rangebar import populate_cache_resumable
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

| API                          | Use Case                   | Details                                                   |
| ---------------------------- | -------------------------- | --------------------------------------------------------- |
| `get_range_bars()`           | Date range, backtesting    | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `get_n_range_bars()`         | Exact N bars, ML           | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `populate_cache_resumable()` | Long ranges (>30 days)     | [docs/api/cache-api.md](/docs/api/cache-api.md)           |
| `process_trades_polars()`    | Polars DataFrames          | [docs/api/processing-api.md](/docs/api/processing-api.md) |
| `process_trades_chunked()`   | Large datasets >10M trades | [docs/api/processing-api.md](/docs/api/processing-api.md) |

---

## Architecture

```
rangebar-py/
├── crates/                    8 Rust crates (publish = false)
│   └── rangebar-core/         Core algorithm, microstructure features
├── src/lib.rs                 PyO3 bindings (Rust→Python bridge)
├── python/rangebar/           Python API layer
│   ├── clickhouse/            ClickHouse cache (bigblack)
│   ├── validation/            Microstructure validation (v7.0+)
│   └── storage/               Tier 1 cache (Parquet)
├── scripts/                   Pueue jobs, validation, cache population
├── .cargo/config.toml         Cross-compile friendly rustflags
└── pyproject.toml             Maturin config
```

**Key files**: `src/lib.rs` (PyO3 bindings), `python/rangebar/__init__.py` (public API), `crates/rangebar-core/src/processor.rs` (core algorithm)

**Full architecture**: [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)

---

## Development Commands

```bash
# Setup (mise manages all tools including zig)
mise install

# Build & test
mise run build              # maturin develop
mise run test               # Rust tests (cargo nextest)
mise run test-py            # Python tests (pytest)

# Quality
mise run check-full         # fmt + lint + test + deny

# Release (see docs/development/RELEASE.md)
mise run release:full       # Full 4-phase workflow
mise run release:linux      # Zig cross-compile (~55 sec)
mise run publish            # Upload to PyPI

# Benchmarks
mise run bench:run          # Full benchmarks
mise run bench:validate     # Verify 1M ticks < 100ms
```

---

## Build System

Zig cross-compilation from macOS (no remote SSH needed).

| Platform     | Strategy          | Time    | Command                        |
| ------------ | ----------------- | ------- | ------------------------------ |
| macOS ARM64  | Native maturin    | ~10 sec | `mise run release:macos-arm64` |
| Linux x86_64 | Zig cross-compile | ~55 sec | `mise run release:linux`       |

**Why rustls**: Eliminates OpenSSL dependency, enabling pure Rust cross-compilation.

**Full release workflow**: [docs/development/RELEASE.md](/docs/development/RELEASE.md)

---

## Symbol Registry (Issue #79)

**Every symbol must be registered before processing.** Unregistered symbols raise `SymbolNotRegisteredError`.

**SSoT**: `symbols.toml` (repo root symlink → `python/rangebar/data/symbols.toml`)

**Adding new symbols**: Edit `symbols.toml`, run `maturin develop`, then process.

**Override gate** (dev only): `export RANGEBAR_SYMBOL_GATE=off`

**Full details**: [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md#symbol-registry-issue-79) (field schema, gate table, telemetry)

---

## Microstructure Features (v7.0+)

10 intra-bar features + 16 inter-bar lookback features computed in Rust during bar construction.

**Intra-bar**: `ofi`, `vwap_close_deviation`, `kyle_lambda_proxy`, `trade_intensity`, `volume_per_trade`, `aggression_ratio`, `aggregation_density`, `turnover_imbalance`, `duration_us`, `price_impact`

**Inter-bar** (lookback window): `lookback_ofi`, `lookback_intensity`, `lookback_kyle_lambda`, `lookback_burstiness`, `lookback_hurst`, `lookback_permutation_entropy`, + 10 more

**Full details**: [crates/CLAUDE.md](/crates/CLAUDE.md#microstructure-features-v70) (formulas, ranges, academic backing)

**Validation**: [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md#validation-framework-v70) (Tier 1 smoke, Tier 2 statistical)

---

## Memory Guards (MEM-\*)

| Guard   | Description                                   | Location                      |
| ------- | --------------------------------------------- | ----------------------------- |
| MEM-001 | Memory estimation before fetch                | `resource_guard.py`           |
| MEM-002 | Chunked dict conversion (100K trades)         | `helpers.py`                  |
| MEM-006 | Polars concat instead of pandas               | `conversion.py`               |
| MEM-011 | Adaptive chunk size (50K with microstructure) | `helpers.py:304`              |
| MEM-013 | Force ClickHouse-first for >30 day ranges     | `orchestration/range_bars.py` |

---

## Common Errors

| Error                                   | Cause                  | Fix                                        |
| --------------------------------------- | ---------------------- | ------------------------------------------ |
| `RangeBarProcessor has no attribute X`  | Outdated binding       | `maturin develop`                          |
| `Invalid threshold_decimal_bps`         | Wrong units            | Use 250 for 0.25%                          |
| `High < Low` assertion                  | Bad input data         | Check sorting                              |
| `target-cpu=native` cross-compile error | RUSTFLAGS pollution    | Use `RUSTFLAGS=""` or `.cargo/config.toml` |
| OOM with `include_microstructure=True`  | Large date range       | MEM-011 adaptive chunk size                |
| Duplicate ticks in Parquet cache        | Pre-v12.8 bug          | `scripts/deduplicate_parquet_cache.py`     |
| `SymbolNotRegisteredError`              | Symbol not in registry | Edit `symbols.toml`, `maturin develop`     |

---

## ClickHouse Infrastructure

**Architecture**: All range bar data served from **bigblack** (remote GPU host) via SSH tunnel. No local ClickHouse.

**Connection mode**: `RANGEBAR_MODE=remote` (set in `.mise.toml`). Skips localhost, always routes through `RANGEBAR_CH_HOSTS` via direct connection or SSH tunnel. Preflight: `mise run db:ensure` verifies bigblack reachable before tests.

| Host        | Thresholds (dbps)   | ClickHouse | Total Bars |
| ----------- | ------------------- | ---------- | ---------- |
| bigblack    | 250, 500, 750, 1000 | Native     | 260M+      |
| littleblack | 100                 | Docker     | —          |

**Cache operations**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md)

**Distributed jobs**: [scripts/CLAUDE.md](/scripts/CLAUDE.md) (pueue, per-year parallelization, autoscaler)

**Dedup hardening**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md#dedup-hardening-issue-90) (5-layer idempotent write protection)

---

## Dedup Hardening (Issue #90)

Five-layer protection against duplicate rows from OPTIMIZE timeout crashes and retry storms:

| Layer | Mechanism                               | Scope        |
| ----- | --------------------------------------- | ------------ |
| 1     | `non_replicated_deduplication_window`   | Engine-level |
| 2     | INSERT dedup token (`cache_key` hash)   | Per-INSERT   |
| 3     | Fire-and-forget `OPTIMIZE` + merge poll | Post-write   |
| 4     | `FINAL` read with partition parallelism | Query-time   |
| 5     | Schema migration in `_ensure_schema()`  | Bootstrap    |

All three write methods (`store_range_bars`, `store_bars_bulk`, `store_bars_batch`) emit INSERT dedup tokens.

**Validation**: `mise run cache:validate-dedup` (runs on bigblack, ALL 4 LAYERS PASS)

**Full details**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md#dedup-hardening-issue-90)

---

## CLAUDE.md Network (Hub-and-Spoke)

| Directory                      | CLAUDE.md                                                                     | Purpose                           |
| ------------------------------ | ----------------------------------------------------------------------------- | --------------------------------- |
| `/`                            | This file                                                                     | Hub, quick reference              |
| `/crates/`                     | [crates/CLAUDE.md](/crates/CLAUDE.md)                                         | Rust workspace, microstructure    |
| `/python/rangebar/`            | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)                       | Python API, caching, validation   |
| `/python/rangebar/clickhouse/` | [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) | Cache layer, schema, remote setup |
| `/scripts/`                    | [scripts/CLAUDE.md](/scripts/CLAUDE.md)                                       | Pueue jobs, per-year parallelism  |

---

## Terminology

| Term                     | Acronym | Definition                                                                                                           |
| ------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------- |
| **Decimal Basis Points** | dbps    | 1 dbps = 0.001% = 0.00001. Example: 250 dbps = 0.25%. **All threshold values use dbps.**                             |
| **Ouroboros**            | —       | Cyclical reset boundary (year/month/week). Resets processor state for reproducibility and cache-friendly processing. |
| **Orphaned Bar**         | —       | Incomplete bar at an ouroboros boundary. Marked `is_orphan=True` with metadata.                                      |
| **Dynamic Ouroboros**    | —       | Forex-specific: reset at first tick after weekend gap, auto-handling DST shifts.                                     |
