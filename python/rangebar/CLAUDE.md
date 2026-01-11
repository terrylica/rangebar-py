# Python Layer

**Parent**: [/CLAUDE.md](/CLAUDE.md) | **API Reference**: [/docs/api.md](/docs/api.md)

This directory contains the Python API layer for rangebar-py.

---

## AI Agent Quick Reference

### Common Tasks & Entry Points

| When Claude is asked to... | Primary File | Function/Class |
|---------------------------|--------------|----------------|
| Generate range bars (date-bounded) | `__init__.py` | `get_range_bars()` |
| Generate range bars (count-bounded, ML) | `__init__.py` | `get_n_range_bars()` |
| Generate range bars (existing data) | `__init__.py` | `process_trades_to_dataframe()` |
| Generate range bars (Polars) | `__init__.py` | `process_trades_polars()` |
| Process large datasets | `__init__.py` | `process_trades_chunked()` |
| Read/write tick data | `storage/parquet.py` | `TickStorage` class |
| Bar-count cache operations | `clickhouse/cache.py` | `count_bars()`, `get_n_bars()` |
| Validate microstructure features | `validation/tier1.py` | `validate_tier1()` |

### API Selection Guide

```
Starting Point?
├── Need data fetching (date range)? → get_range_bars() [DATE-BOUNDED]
├── Need exactly N bars (ML/walk-forward)? → get_n_range_bars() [COUNT-BOUNDED]
├── Have pandas DataFrame → process_trades_to_dataframe()
├── Have Polars DataFrame/LazyFrame → process_trades_polars() [2-3x faster]
└── Have Iterator (large data) → process_trades_chunked()
```

### File-to-Responsibility Mapping

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public Python API |
| `__init__.pyi` | Type stubs for IDE/AI |
| `storage/parquet.py` | Tier 1 cache (local Parquet) |
| `clickhouse/cache.py` | Tier 2 cache (ClickHouse) |
| `clickhouse/schema.sql` | ClickHouse table schema |
| `validation/tier1.py` | Fast validation (<30 sec) |
| `validation/tier2.py` | Statistical validation (~10 min) |
| `exness.py` | Exness data source utilities |

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
├── clickhouse/          # Tier 2 cache (ClickHouse Cloud)
│   ├── cache.py         # Range bar cache operations
│   └── schema.sql       # Table schema (v7.0: 10 microstructure columns)
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

| Function | Purpose |
|----------|---------|
| `get_range_bars()` | Date-bounded, auto-fetch, caching |
| `get_n_range_bars()` | Count-bounded, ML training |
| `process_trades_to_dataframe()` | From existing pandas DataFrame |
| `process_trades_polars()` | From Polars, 2-3x faster |
| `process_trades_chunked()` | Streaming, memory-safe |
| `precompute_range_bars()` | Batch precompute to ClickHouse |

**Constants**:
- `TIER1_SYMBOLS` - 18 high-liquidity symbols
- `THRESHOLD_PRESETS` - Named thresholds (micro, tight, standard, etc.)

### `__init__.pyi` - Type Stubs

Provides type hints for IDE autocompletion and AI assistants. Keep in sync with `__init__.py`.

### `_core.abi3.so` - PyO3 Extension

Binary built by `maturin develop`. Contains:
- `PyRangeBarProcessor` - Wraps Rust processor
- `PyAggTrade` - Trade data type
- Data fetching (when providers feature enabled)

**Rebuild after Rust changes**: `maturin develop`

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
- Requires ClickHouse Cloud connection

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

## Error Handling

| Exception | When | Fix |
|-----------|------|-----|
| `ValueError` | Invalid threshold, bad dates | Check parameters |
| `RuntimeError` | Processing failure | Check data sorting |
| `ConnectionError` | ClickHouse unavailable | Check network/credentials |
| `FileNotFoundError` | Tick data not cached | Set `use_cache=False` |
| `AttributeError: RangeBarProcessor has no attribute X` | Rust binding outdated | Run `maturin develop` |
| `AssertionError: High < Low` | OHLC invariant violation | Check input data sorting |

---

## Development

### Adding New Features

1. Update `__init__.py` with new function
2. Update `__init__.pyi` with type stub
3. Add tests in `/tests/`
4. Update `/docs/api.md`

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

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/docs/api.md](/docs/api.md) - Full API reference
- [/crates/CLAUDE.md](/crates/CLAUDE.md) - Rust crate details
- [/src/lib.rs](/src/lib.rs) - PyO3 bindings source
