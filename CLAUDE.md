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
| Release workflow        | [docs/development/RELEASE.md](/docs/development/RELEASE.md)         | Zig cross-compile, mise tasks |
| Performance monitoring  | [docs/development/PERFORMANCE.md](/docs/development/PERFORMANCE.md) | Benchmarks, metrics           |
| Project context         | [docs/CONTEXT.md](/docs/CONTEXT.md)                                 | Why this project exists       |
| API reference           | [docs/api/INDEX.md](/docs/api/INDEX.md)                             | Full Python API docs          |

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

| API                          | Use Case                       | Details                                                   |
| ---------------------------- | ------------------------------ | --------------------------------------------------------- |
| `get_range_bars()`           | Date range, backtesting        | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `get_n_range_bars()`         | Exact N bars, ML               | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `populate_cache_resumable()` | Long ranges (>30 days)         | [docs/api/cache-api.md](/docs/api/cache-api.md)           |
| `process_trades_polars()`    | Polars DataFrames, 2-3x faster | [docs/api/processing-api.md](/docs/api/processing-api.md) |
| `process_trades_chunked()`   | Large datasets >10M trades     | [docs/api/processing-api.md](/docs/api/processing-api.md) |

**Full API reference**: [docs/api/INDEX.md](/docs/api/INDEX.md)

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
├── .cargo/config.toml         Cross-compile friendly rustflags
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

### Cross-Platform Wheel Building

**Strategy**: Zig cross-compilation from macOS (no remote SSH needed).

| Platform         | Strategy          | Time    | Details                        |
| ---------------- | ----------------- | ------- | ------------------------------ |
| macOS ARM64      | Native maturin    | ~10 sec | `mise run release:macos-arm64` |
| Linux x86_64     | Zig cross-compile | ~55 sec | `mise run release:linux`       |
| Linux (fallback) | SSH + Docker      | ~5 min  | `LINUX_BUILD_STRATEGY=remote`  |

**Key configuration**:

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `.mise.toml`         | Tools (zig, rust, python) + release tasks        |
| `.cargo/config.toml` | Cross-compile rustflags (no `target-cpu=native`) |
| `Cargo.toml`         | `rustls-tls` (no OpenSSL dependency)             |

**Why rustls**: Eliminates OpenSSL dependency, enabling pure Rust cross-compilation.

**Full release workflow**: [docs/development/RELEASE.md](/docs/development/RELEASE.md)

---

## Memory Guards (MEM-\*)

Production memory optimization infrastructure:

| Guard   | Description                                   | Location                      |
| ------- | --------------------------------------------- | ----------------------------- |
| MEM-001 | Memory estimation before fetch                | `resource_guard.py`           |
| MEM-002 | Chunked dict conversion (100K trades)         | `helpers.py`                  |
| MEM-006 | Polars concat instead of pandas               | `conversion.py`               |
| MEM-011 | Adaptive chunk size (50K with microstructure) | `helpers.py:304`              |
| MEM-013 | Force ClickHouse-first for >30 day ranges     | `orchestration/range_bars.py` |

**Issue #65 Fix**: MEM-011 reduces chunk size from 100K to 50K when `include_microstructure=True`, preventing OOM on large date ranges.

**Issue #69 Fix**: MEM-013 requires ClickHouse cache for date ranges >30 days. See [Long Date Ranges](#long-date-ranges-30-days) below.

---

## Version 7.0 Features (Issue #25)

10 market microstructure features computed in Rust during bar construction:

| Feature                | Formula                                     | Range        |
| ---------------------- | ------------------------------------------- | ------------ |
| `duration_us`          | close_time - open_time                      | [0, +inf)    |
| `ofi`                  | (buy_vol - sell_vol) / total                | [-1, 1]      |
| `vwap_close_deviation` | (close - vwap) / (high - low)               | ~[-1, 1]     |
| `price_impact`         | abs(close - open) / volume                  | [0, +inf)    |
| `kyle_lambda_proxy`    | ((close-open)/open) / (imbalance/total_vol) | (-inf, +inf) |
| `trade_intensity`      | trades / duration_sec                       | [0, +inf)    |
| `volume_per_trade`     | volume / trade_count                        | [0, +inf)    |
| `aggression_ratio`     | buy_count / sell_count                      | [0, 100]     |
| `aggregation_density`  | individual_count / agg_count                | [1, +inf)    |
| `turnover_imbalance`   | (buy_turn - sell_turn) / total_turnover     | [-1, 1]      |

**Full details**: [crates/CLAUDE.md](/crates/CLAUDE.md#microstructure-features-v70)

**Validation**: `python/rangebar/validation/` - Tier 1 (auto), Tier 2 (before ML)

---

## Trade ID Tracking (v12.4+, Issue #72)

Each `RangeBar` tracks aggregate trade ID range for data integrity verification:

| Field                | Type  | Description                            |
| -------------------- | ----- | -------------------------------------- |
| `first_agg_trade_id` | `i64` | First AggTrade ID that opened this bar |
| `last_agg_trade_id`  | `i64` | Last AggTrade ID processed in this bar |

**Use cases**:

- **Gap detection**: `bars[i].first_agg_trade_id == bars[i-1].last_agg_trade_id + 1`
- **Data integrity**: Verify no trades were dropped during processing
- **Checkpoint validation**: Confirm resume alignment after interruption

```python
from rangebar import TRADE_ID_RANGE_COLUMNS

# Constant for column selection
TRADE_ID_RANGE_COLUMNS  # ("first_agg_trade_id", "last_agg_trade_id")

# Gap detection pattern
for i in range(1, len(bars)):
    expected = bars[i-1]["last_agg_trade_id"] + 1
    actual = bars[i]["first_agg_trade_id"]
    if actual != expected:
        print(f"Gap detected: expected {expected}, got {actual}")
```

**Full details**: [crates/CLAUDE.md](/crates/CLAUDE.md#trade-id-tracking-issue-72-v124)

---

## Common Errors

| Error                                   | Cause               | Fix                                        |
| --------------------------------------- | ------------------- | ------------------------------------------ |
| `RangeBarProcessor has no attribute X`  | Outdated binding    | `maturin develop`                          |
| `Invalid threshold_decimal_bps`         | Wrong units         | Use 250 for 0.25%                          |
| `High < Low` assertion                  | Bad input data      | Check sorting                              |
| `target-cpu=native` cross-compile error | RUSTFLAGS pollution | Use `RUSTFLAGS=""` or `.cargo/config.toml` |
| OOM with `include_microstructure=True`  | Large date range    | Fixed by MEM-011 adaptive chunk size       |

---

## Long Date Ranges (>30 days)

For date ranges exceeding 30 days, ClickHouse cache is **required** (MEM-013 guard):

```python
from rangebar import populate_cache_resumable, get_range_bars

# Step 1: Populate cache incrementally (memory-safe, resumable)
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")

# Step 2: Read from cache (fast, memory-safe)
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

**Why**: Direct processing of multi-year ranges causes OOM. The `populate_cache_resumable()` function processes day-by-day with checkpoints, enabling:

- **Cross-machine resume**: Checkpoints stored in both local filesystem and ClickHouse
- **Bar-level accuracy**: Incomplete bars preserved across interrupts
- **Force refresh**: `force_refresh=True` wipes cache and restarts

| Parameter       | Default | Description                        |
| --------------- | ------- | ---------------------------------- |
| `force_refresh` | `False` | Wipe existing cache and checkpoint |
| `notify`        | `True`  | Send progress notifications        |

**Full API**: [docs/api/cache-api.md](/docs/api/cache-api.md#populate_cache_resumable)

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

| Host        | Thresholds | ClickHouse | Notes            |
| ----------- | ---------- | ---------- | ---------------- |
| bigblack    | 100, 700   | Native     | Primary GPU host |
| littleblack | 100        | Docker     | Secondary host   |

---

## Remote Host Setup (ClickHouse)

**CRITICAL gotchas** when setting up rangebar on remote hosts:

| Issue                   | Root Cause                                | Solution                                           |
| ----------------------- | ----------------------------------------- | -------------------------------------------------- |
| Connection refused      | `mise.toml` hardcodes `RANGEBAR_CH_HOSTS` | Override: `export RANGEBAR_CH_HOSTS=localhost`     |
| ThresholdError          | Default minimum threshold validation      | Override: `export RANGEBAR_CRYPTO_MIN_THRESHOLD=1` |
| pip install fails       | Python 3.13 externally-managed-env        | Use `uv pip install rangebar` instead              |
| ClickHouse auth failure | Docker container without password flag    | Add `-e CLICKHOUSE_PASSWORD=` to docker run        |
| mise not found          | mise not installed on remote              | Use explicit paths: `~/.local/bin/uv run python`   |

**mise SSoT pattern**: Environment variables in `.mise.toml` are the source of truth. Remote hosts without mise must manually export required variables.

**Full setup guide**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md#populating-remote-hosts)

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
| [docs/api/INDEX.md](/docs/api/INDEX.md)                             | Python API reference                    |
| [docs/development/RELEASE.md](/docs/development/RELEASE.md)         | Release workflow, zig cross-compile     |
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
