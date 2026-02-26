# Python Layer

**Parent**: [/CLAUDE.md](/CLAUDE.md) | **API Reference**: [/docs/api/INDEX.md](/docs/api/INDEX.md)

This directory contains the Python API layer for rangebar-py.

---

## AI Agent Quick Reference

### Common Tasks & Entry Points

| When Claude is asked to...              | Primary File          | Function/Class                         |
| --------------------------------------- | --------------------- | -------------------------------------- |
| Generate range bars (date-bounded)      | `__init__.py`         | `get_range_bars()`                     |
| Generate range bars (count-bounded, ML) | `__init__.py`         | `get_n_range_bars()`                   |
| Generate range bars (existing data)     | `__init__.py`         | `process_trades_to_dataframe()`        |
| Generate range bars (Polars)            | `__init__.py`         | `process_trades_polars()`              |
| Process large datasets                  | `__init__.py`         | `process_trades_chunked()`             |
| Run streaming sidecar                   | `sidecar.py`          | `run_sidecar()`, `SidecarConfig`       |
| Populate cache for long ranges          | `checkpoint.py`       | `populate_cache_resumable()`           |
| Read/write tick data                    | `storage/parquet.py`  | `TickStorage` class                    |
| Bar-count cache operations              | `clickhouse/cache.py` | `count_bars()`, `get_n_bars()`         |
| Validate microstructure features        | `validation/tier1.py` | `validate_tier1()`                     |
| Self-healing gap reconciliation         | `kintsugi.py`         | `kintsugi_pass()`, `kintsugi_daemon()` |

### API Selection Guide

```
Starting Point?
├── Date range > 30 days? → populate_cache_resumable() first, then get_range_bars()
├── Need data fetching (date range ≤ 30 days)? → get_range_bars() [DATE-BOUNDED]
├── Need exactly N bars (ML/walk-forward)? → get_n_range_bars() [COUNT-BOUNDED]
├── Have pandas DataFrame → process_trades_to_dataframe()
├── Have Polars DataFrame/LazyFrame → process_trades_polars() [2-3x faster]
└── Have Iterator (large data) → process_trades_chunked()
```

### File-to-Responsibility Mapping

| File                    | Responsibility                                  |
| ----------------------- | ----------------------------------------------- |
| `__init__.py`           | Public Python API                               |
| `__init__.pyi`          | Re-export index (stubs)                         |
| `sidecar.py`            | Streaming sidecar orchestrator (v12.20+)        |
| `sidecar.pyi`           | Sidecar type stubs (SidecarConfig, run_sidecar) |
| `storage/parquet.py`    | Tier 1 cache (local Parquet)                    |
| `clickhouse/cache.py`   | Tier 2 cache (ClickHouse)                       |
| `clickhouse/schema.sql` | ClickHouse table schema                         |
| `validation/tier1.py`   | Fast validation (<30 sec)                       |
| `validation/tier2.py`   | Statistical validation (~10 min)                |
| `exness.py`             | Exness data source utilities                    |
| `kintsugi.py`           | Self-healing gap reconciliation (Issue #115)    |

### Performance Optimization Checklist

When optimizing data processing:

1. Use `pl.scan_parquet()` instead of `pl.read_parquet()` (lazy loading)
2. Apply filters on LazyFrame before `.collect()` (predicate pushdown)
3. Select only required columns before `.to_dicts()` (minimal conversion)
4. Use `process_trades_chunked()` for datasets >10M trades

---

## Structure

```
python/rangebar/
├── __init__.py          # Public API (get_range_bars, process_trades_*)
├── __init__.pyi         # Type stubs for IDE/AI
├── _core.abi3.so        # PyO3 binary extension (built by maturin)
├── sidecar.py           # Streaming sidecar orchestrator (v12.20+)
├── sidecar.pyi          # Sidecar type stubs
├── clickhouse/          # Tier 2 cache (ClickHouse, bigblack)
│   ├── cache.py         # Range bar cache operations
│   └── schema.sql       # Table schema (v7.0: 10 microstructure columns)
├── data/                # Package data files (Issue #79)
│   ├── __init__.py      # importlib.resources marker
│   └── symbols.toml     # Unified Symbol Registry (SSoT)
├── symbol_registry.py   # Registry loader + mandatory gate (Issue #79)
├── storage/             # Tier 1 cache (local Parquet)
│   └── parquet.py       # TickStorage class
├── validation/          # Microstructure feature validation (v7.0+)
│   ├── tier1.py         # Fast validation (<30 sec)
│   └── tier2.py         # Statistical validation (~10 min)
└── exness.py            # Exness data source utilities
```

---

## Key Files

### `__init__.py` - Public API

**Entry points**:

| Function                        | Purpose                           |
| ------------------------------- | --------------------------------- |
| `get_range_bars()`              | Date-bounded, auto-fetch, caching |
| `get_n_range_bars()`            | Count-bounded, ML training        |
| `process_trades_to_dataframe()` | From existing pandas DataFrame    |
| `process_trades_polars()`       | From Polars, 2-3x faster          |
| `process_trades_chunked()`      | Streaming, memory-safe            |
| `precompute_range_bars()`       | Batch precompute to ClickHouse    |

**Constants**:

- `TIER1_SYMBOLS` - 18 high-liquidity symbols
- `THRESHOLD_PRESETS` - Named thresholds (micro, tight, standard, etc.)

### `.pyi` Type Stubs (PEP 561 Per-Module Layout)

Type stubs follow PEP 561 per-module layout. Each `.pyi` file lives alongside its `.py` source.

- `__init__.pyi` — **Re-export index only** (no definitions). Uses `from .x import Y as Y` pattern.
- Per-module stubs (e.g., `constants.pyi`, `ouroboros.pyi`, `orchestration/range_bars.pyi`) — Contain actual type definitions.

**Enforced by**: `pretooluse-pyi-stub-guard.ts` hook (blocks definitions in `__init__.pyi`).

### `_core.abi3.so` - PyO3 Extension

Binary built by `maturin develop`. Contains:

- `PyRangeBarProcessor` - Wraps Rust processor
- `PyAggTrade` - Trade data type
- Data fetching (when providers feature enabled)

**Rebuild after Rust changes**: `maturin develop`

---

## Symbol Registry (Issue #79)

**Every symbol must be registered in `symbols.toml` before processing.** Unregistered symbols raise `SymbolNotRegisteredError`.

### Files

| File                       | Purpose                                      |
| -------------------------- | -------------------------------------------- |
| `data/symbols.toml`        | TOML registry (SSoT for all symbol metadata) |
| `symbol_registry.py`       | Loader module + gate + query functions       |
| `exceptions.py`            | `SymbolNotRegisteredError` exception         |
| `symbols.toml` (repo root) | Symlink for developer convenience            |

### Adding New Symbols

1. Edit `python/rangebar/data/symbols.toml`
2. Run `maturin develop`
3. Process the symbol

### Gate Integration Points

Gates are inserted at the top of each entry point function, **before** threshold validation:

```python
from rangebar.symbol_registry import validate_symbol_registered, validate_and_clamp_start_date

validate_symbol_registered(symbol, operation="function_name")
start_date = validate_and_clamp_start_date(symbol, start_date)  # only if start_date param exists
```

### Environment Variable

`RANGEBAR_SYMBOL_GATE` controls gate behavior:

- `"strict"` (default): `SymbolNotRegisteredError`
- `"warn"`: `UserWarning` + continue
- `"off"`: No gating (dev/testing)

### Telemetry

Gate violations emit:

1. **NDJSON log**: `logs/events.jsonl` with `component="symbol_registry"`
2. **Pushover alert**: Emergency priority (repeats every 30s until acknowledged)

---

## Caching Architecture

### Tier 1: Local Parquet

**Location**: `storage/parquet.py`

- Stores raw tick data locally
- Used by `get_range_bars()` with `use_cache=True`
- Fast retrieval, no network dependency

```python
from rangebar.storage.parquet import TickStorage
storage = TickStorage()
ticks = storage.get_ticks("BTCUSDT", start, end)
```

### Tier 2: ClickHouse

**Location**: `clickhouse/cache.py`

- Stores precomputed range bars
- Used by `get_range_bars()` and `get_n_range_bars()`
- Requires ClickHouse connection (bigblack via SSH tunnel)

```python
from rangebar.clickhouse.cache import get_cached_bars, count_bars
bars_df = get_cached_bars("BTCUSDT", threshold=250, start, end)
n = count_bars("BTCUSDT", threshold=250)
```

**Schema**: `clickhouse/schema.sql` (v7.0 adds 10 microstructure columns)

---

## Validation Framework (v7.0+)

**Location**: `validation/`

Validates microstructure features for ML readiness.

### Tier 1: Smoke Test (`tier1.py`)

- Runtime: <30 seconds
- Runs automatically on every `precompute_range_bars()`
- Checks: NaN, Inf, bounds, basic correlations

```python
from rangebar.validation.tier1 import validate_tier1
result = validate_tier1(df)
assert result["tier1_passed"]
```

### Tier 2: Statistical (`tier2.py`)

- Runtime: ~10 minutes
- **Mandatory before production ML training**
- Checks: Stationarity, predictive power, mutual information

```python
from rangebar.validation.tier2 import validate_tier2
df["forward_return"] = df["Close"].shift(-1) / df["Close"] - 1
result = validate_tier2(df)
assert result["tier2_passed"]
```

---

## Notifications (`notify/`)

Telegram and Pushover notification modules for cache operations and monitoring.

| Module        | Bot          | Use Case                            |
| ------------- | ------------ | ----------------------------------- |
| `telegram.py` | @rangebarbot | Hook events, gap alerts, heartbeats |
| `pushover.py` | —            | Critical alerts (checksum failures) |

### Telegram Integration

```python
from rangebar.notify.telegram import send_telegram, enable_telegram_notifications

# Enable for all hook events (populate, cache write, validation, etc.)
enable_telegram_notifications()

# Or failures only
from rangebar.notify.telegram import enable_failure_notifications
enable_failure_notifications()

# Direct message
send_telegram("<b>Alert:</b> Custom message", disable_notification=False)
```

**Env vars**: `RANGEBAR_TELEGRAM_TOKEN` + `RANGEBAR_TELEGRAM_CHAT_ID` (set in `.mise.toml`)

### Safe Populate Pattern (Post-Incident)

`populate_cache_resumable()` uses `_fatal_cache_write()` (not `try_cache_write()`) to ensure ClickHouse failures raise exceptions instead of being silently swallowed. The checkpoint only advances for days where ClickHouse confirmed the write.

| Guard                          | What It Prevents                          |
| ------------------------------ | ----------------------------------------- |
| Fatal cache write              | Silent loss of bars when SSH tunnel drops |
| T-1 date clamping              | Crash on unavailable Binance Vision data  |
| `bars_written` from ClickHouse | Checkpoint/ClickHouse count divergence    |

---

## Common Patterns

### Date-Bounded Bars (Backtesting)

```python
from rangebar import get_range_bars

df = get_range_bars(
    "BTCUSDT",
    "2024-01-01",
    "2024-06-30",
    threshold_decimal_bps=250,  # 0.25%
)
```

### Count-Bounded Bars (ML)

```python
from rangebar import get_n_range_bars

df = get_n_range_bars(
    "BTCUSDT",
    n_bars=10000,
    threshold_decimal_bps=250,
)
assert len(df) == 10000  # Exact count guaranteed
```

### With Microstructure Features (v7.0+)

```python
df = get_range_bars(
    "BTCUSDT",
    "2024-01-01",
    "2024-06-30",
    include_microstructure=True,
)
# Includes: ofi, vwap_close_deviation, kyle_lambda_proxy, etc.
```

### Large Dataset Processing

```python
from rangebar import process_trades_chunked

for bars_chunk in process_trades_chunked(trades_iterator, chunk_size=100_000):
    # Process incrementally, bounded memory
    save_to_database(bars_chunk)
```

### backtesting.py Integration

```python
from backtesting import Backtest, Strategy
from rangebar import get_range_bars

# Fetch data and generate range bars in one call
data = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Use directly with backtesting.py
bt = Backtest(data, MyStrategy, cash=10000, commission=0.0002)
stats = bt.run()
bt.plot()
```

**Output Format** (backtesting.py compatible):

```python
# DataFrame with DatetimeIndex and OHLCV columns
                          Open      High       Low     Close  Volume
timestamp
2024-01-01 00:00:15  42000.00  42105.00  41980.00  42100.00   15.43
2024-01-01 00:03:42  42100.00  42220.00  42100.00  42215.00    8.72
```

### backtesting.py Compatibility Checklist

- [x] **OHLCV column names**: Capitalized (Open, High, Low, Close, Volume)
- [x] **DatetimeIndex**: Pandas DatetimeIndex with timezone-naive timestamps
- [x] **No NaN values**: All bars complete (backtesting.py raises on NaN)
- [x] **Sorted chronologically**: Timestamps in ascending order
- [x] **OHLC invariants**: High ≥ max(Open, Close), Low ≤ min(Open, Close)

**Validation Script**:

```python
def validate_for_backtesting_py(df: pd.DataFrame) -> bool:
    """Validate DataFrame is compatible with backtesting.py."""
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.isnull().any().any()
    assert (df["High"] >= df["Open"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()
    return True
```

---

## Long Range Processing (MEM-013)

For date ranges > 30 days, direct `get_range_bars()` is blocked to prevent OOM.

### Workflow

```python
from rangebar import populate_cache_resumable, get_range_bars

# Step 1: Populate cache (resumable, memory-safe)
populate_cache_resumable(
    "BTCUSDT",
    "2019-01-01",
    "2025-12-31",
    threshold_decimal_bps=250,
    force_refresh=False,  # Resume from last checkpoint
)

# Step 2: Read from cache
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

### Parameters

| Parameter                | Default  | Description                        |
| ------------------------ | -------- | ---------------------------------- |
| `force_refresh`          | `False`  | Wipe existing cache and checkpoint |
| `threshold_decimal_bps`  | `250`    | Threshold (same as get_range_bars) |
| `include_microstructure` | `False`  | Include microstructure features    |
| `ouroboros`              | `"year"` | Reset mode for reproducibility     |
| `notify`                 | `True`   | Send progress notifications        |
| `verbose`                | `True`   | Show tqdm progress bar and logging |

### Resumability

- **Local checkpoint**: `~/.cache/rangebar/checkpoints/` (fast)
- **ClickHouse checkpoint**: `population_checkpoints` table (cross-machine)
- **Bar-level**: Incomplete bars preserved across interrupts

---

## Error Handling

| Exception                                              | When                         | Fix                       |
| ------------------------------------------------------ | ---------------------------- | ------------------------- |
| `ValueError`                                           | Invalid threshold, bad dates | Check parameters          |
| `RuntimeError`                                         | Processing failure           | Check data sorting        |
| `ConnectionError`                                      | ClickHouse unavailable       | Check network/credentials |
| `FileNotFoundError`                                    | Tick data not cached         | Set `use_cache=False`     |
| `AttributeError: RangeBarProcessor has no attribute X` | Rust binding outdated        | Run `maturin develop`     |
| `AssertionError: High < Low`                           | OHLC invariant violation     | Check input data sorting  |

---

## Development

### Adding New Features

1. Update `__init__.py` with new function
2. Add type stub to the per-module `.pyi` file (e.g., `orchestration/range_bars.pyi`)
3. Add re-export to `__init__.pyi` (e.g., `from .orchestration.range_bars import new_func as new_func`)
4. Add tests in `/tests/`
5. Update `/docs/api/INDEX.md`

### Testing

```bash
# Run Python tests
mise run test-py

# Test specific file
pytest tests/test_microstructure_features.py -v

# E2E tests
pytest tests/test_e2e_optimized.py -v
pytest tests/test_get_n_range_bars.py -v
```

### Portable Validation Scripts

For GPU workstations without full dev environment:

- `scripts/validate_n_range_bars.py` - Count-bounded API validation
- `scripts/validate_microstructure_features.py` - v7.0 feature validation

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub (architecture, principles, terminology)
- [/docs/api/INDEX.md](/docs/api/INDEX.md) - Full API reference
- [/crates/CLAUDE.md](/crates/CLAUDE.md) - Rust crates, microstructure feature formulas
- [/python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) - ClickHouse cache, dedup hardening
- [/python/rangebar/plugins/CLAUDE.md](/python/rangebar/plugins/CLAUDE.md) - FeatureProvider plugin system, entry-point discovery
- [/scripts/CLAUDE.md](/scripts/CLAUDE.md) - Pueue ops, per-year parallelization (default strategy), anti-patterns
- [/src/lib.rs](/src/lib.rs) - PyO3 bindings source
