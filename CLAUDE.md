# CLAUDE.md - Project Memory

**rangebar-py**: Rust workspace with Python bindings via PyO3/maturin. Publishes to PyPI only (not crates.io).

---

## Quick Reference

| Task                    | Entry Point                                                         | Details                       |
| ----------------------- | ------------------------------------------------------------------- | ----------------------------- |
| Generate range bars     | `get_range_bars()`                                                  | [Python API](#python-api)     |
| Understand architecture | [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)                       | 8-crate workspace             |
| Work with Rust crates   | [crates/CLAUDE.md](/crates/CLAUDE.md)                               | Crate details, microstructure |
| Work with Python layer  | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)             | API, caching, validation      |
| Release workflow        | [docs/development/RELEASE.md](/docs/development/RELEASE.md)         | mise tasks, PyPI              |
| Performance monitoring  | [docs/development/PERFORMANCE.md](/docs/development/PERFORMANCE.md) | Benchmarks, metrics           |
| Project context         | [docs/CONTEXT.md](/docs/CONTEXT.md)                                 | Why this project exists       |
| API reference           | [docs/api.md](/docs/api.md)                                         | Full Python API docs          |

---

## Critical Principle: Leverage Rust

**ALWAYS use local Rust crates for heavy lifting.** Python handles I/O only.

When implementing features or fixing issues:

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

**Why**: Rust handles threshold breach, OHLCV aggregation, temporal integrity, fixed-point arithmetic.

---

## Python API

```python
from rangebar import get_range_bars, get_n_range_bars, process_trades_polars

# Date-bounded (backtesting)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded (ML training)
df = get_n_range_bars("BTCUSDT", n_bars=10000)

# Polars users (2-3x faster) - recommended for existing Polars data
import polars as pl
trades = pl.scan_parquet("trades.parquet")  # LazyFrame for predicate pushdown
df = process_trades_polars(trades, threshold_decimal_bps=250)

# With microstructure features (v7.0+)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", include_microstructure=True)
```

| API                        | Use Case                       | Details                                            |
| -------------------------- | ------------------------------ | -------------------------------------------------- |
| `get_range_bars()`         | Date range, backtesting        | [docs/api.md](/docs/api.md#get_range_bars)         |
| `get_n_range_bars()`       | Exact N bars, ML               | [docs/api.md](/docs/api.md#get_n_range_bars)       |
| `process_trades_polars()`  | Polars DataFrames, 2-3x faster | [docs/api.md](/docs/api.md#process_trades_polars)  |
| `process_trades_chunked()` | Large datasets >10M trades     | [docs/api.md](/docs/api.md#process_trades_chunked) |

**Full API reference**: [docs/api.md](/docs/api.md)

---

## Architecture

```
rangebar-py/
├── crates/                    8 Rust crates (publish = false)
│   └── rangebar-core/         Core algorithm, microstructure features
├── src/lib.rs                 PyO3 bindings
├── python/rangebar/           Python API layer
│   ├── clickhouse/            Tier 2 cache
│   ├── validation/            Microstructure validation (v7.0+)
│   └── storage/               Tier 1 cache (Parquet)
└── pyproject.toml             Maturin config
```

**Key files**:

- `src/lib.rs` - PyO3 bindings (Rust→Python bridge)
- `python/rangebar/__init__.py` - Public Python API
- `crates/rangebar-core/src/processor.rs` - Core algorithm

**Full architecture**: [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)

---

## Development Commands

**mise patterns**: `Skill(itp:mise-configuration)` | `Skill(itp:mise-tasks)`

```bash
# Setup (mise manages all tools)
mise install

# Build & test
mise run build              # maturin develop
mise run test               # Rust tests (cargo nextest)
mise run test-py            # Python tests (pytest)

# Quality
mise run check-full         # fmt + lint + test + deny

# Release (see docs/development/RELEASE.md)
mise run release:full       # Full 4-phase workflow
mise run publish            # Upload to PyPI

# Benchmarks
mise run bench:run          # Full benchmarks
mise run bench:validate     # Verify 1M ticks < 100ms
```

---

## Version 7.0 Features (Issue #25)

10 market microstructure features computed in Rust during bar construction:

| Feature                | Formula                                     | Range        |
| ---------------------- | ------------------------------------------- | ------------ |
| `duration_us`          | (close_time - open_time) \* 1000            | [0, +inf)    |
| `ofi`                  | (buy_vol - sell_vol) / total                | [-1, 1]      |
| `vwap_close_deviation` | (close - vwap) / range                      | ~[-1, 1]     |
| `price_impact`         | abs(close - open) / volume                  | [0, +inf)    |
| `kyle_lambda_proxy`    | ((close-open)/open) / (imbalance/total_vol) | (-inf, +inf) |
| `trade_intensity`      | trades / duration_sec                       | [0, +inf)    |
| `volume_per_trade`     | volume / trade_count                        | [0, +inf)    |
| `aggression_ratio`     | buy_count / sell_count                      | [0, 100]     |
| `aggregation_density`  | individual_count / agg_count                | [1, +inf)    |
| `turnover_imbalance`   | (buy_turn - sell_turn) / volume             | [-1, 1]      |

**Full details**: [crates/CLAUDE.md](/crates/CLAUDE.md#microstructure-features-v70)

**Validation**: `python/rangebar/validation/` - Tier 1 (auto), Tier 2 (before ML)

---

## Common Errors

| Error                                         | Cause            | Fix               |
| --------------------------------------------- | ---------------- | ----------------- |
| `RangeBarProcessor has no attribute X`        | Outdated binding | `maturin develop` |
| `Invalid threshold_decimal_bps`               | Wrong units      | Use 250 for 0.25% |
| `High < Low` assertion                        | Bad input data   | Check sorting     |
| `dtype mismatch (double[pyarrow] vs float64)` | Cache issue      | Fixed in v2.2.0   |

---

## Cache Population

To populate ClickHouse cache with range bar data (for remote hosts or new thresholds):

```python
from rangebar import get_range_bars

df = get_range_bars(
    "BTCUSDT",
    start_date="2023-06-01",
    end_date="2025-12-01",
    threshold_decimal_bps=100,
    use_cache=True,
    fetch_if_missing=True,
)
```

**Full documentation**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md)

**Host-specific cache status**:

| Host        | Thresholds | Notes            |
| ----------- | ---------- | ---------------- |
| bigblack    | 100, 700   | Primary GPU host |
| littleblack | 700        | Secondary host   |

---

## Navigation

### CLAUDE.md Files (Hub-and-Spoke)

| Directory                      | CLAUDE.md                                                                     | Purpose                         |
| ------------------------------ | ----------------------------------------------------------------------------- | ------------------------------- |
| `/`                            | This file                                                                     | Hub, quick reference            |
| `/crates/`                     | [crates/CLAUDE.md](/crates/CLAUDE.md)                                         | Rust workspace, microstructure  |
| `/python/rangebar/`            | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)                       | Python API, caching, validation |
| `/python/rangebar/clickhouse/` | [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) | Cache population, status        |

### Documentation Index

| Document                                                            | Purpose                                 |
| ------------------------------------------------------------------- | --------------------------------------- |
| [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)                       | System design, dependency graph         |
| [docs/CONTEXT.md](/docs/CONTEXT.md)                                 | Why this project exists, backtesting.py |
| [docs/api.md](/docs/api.md)                                         | Python API reference                    |
| [docs/development/RELEASE.md](/docs/development/RELEASE.md)         | Release workflow, mise tasks            |
| [docs/development/PERFORMANCE.md](/docs/development/PERFORMANCE.md) | Benchmarks, metrics, viability          |

---

## Terminology

| Term                          | Acronym | Definition                                                                                                                                                                                                                                           | Unit/Range        |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| **Decimal Basis Points**      | dbps    | 1 dbps = 0.001% = 0.00001 (one-tenth of a basis point). Fine-grained threshold precision unit. Example: 250 dbps = 0.25%. Used throughout as `threshold_decimal_bps`. **All threshold values in this codebase use dbps.**                            | 1 dbps = 0.00001  |
| **Ouroboros**                 | -       | Cyclical reset boundary for range bar construction. Named after the Greek serpent eating its tail (οὐροβόρος), representing cyclical nature. Resets processor state at year/month/week boundaries for reproducibility and cache-friendly processing. | year, month, week |
| **Ouroboros Boundary**        | -       | Specific timestamp where processor state resets: year (Jan 1 00:00 UTC), month (1st 00:00 UTC), or week (Sunday 00:00 UTC for crypto, dynamic first tick for Forex).                                                                                 | timestamp         |
| **Orphaned Bar**              | -       | Incomplete range bar at an ouroboros boundary. Marked with `is_orphan=True` and includes metadata: `ouroboros_boundary`, `expected_duration_us`, `reason`.                                                                                           | -                 |
| **Dynamic Ouroboros**         | -       | Forex-specific ouroboros mode where the reset point is the first available tick after weekend market gap, automatically handling DST shifts.                                                                                                         | timestamp         |
| **Exchange Session Sydney**   | -       | Boolean flag indicating Sydney exchange market session was active during bar construction. Column: `exchange_session_sydney`.                                                                                                                        | bool              |
| **Exchange Session Tokyo**    | -       | Boolean flag indicating Tokyo exchange market session was active during bar construction. Column: `exchange_session_tokyo`.                                                                                                                          | bool              |
| **Exchange Session London**   | -       | Boolean flag indicating London exchange market session was active during bar construction. Column: `exchange_session_london`.                                                                                                                        | bool              |
| **Exchange Session New York** | -       | Boolean flag indicating New York exchange market session was active during bar construction. Column: `exchange_session_newyork`.                                                                                                                     | bool              |
