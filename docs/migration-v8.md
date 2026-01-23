# Migration Guide: v7.x to v8.0

This guide helps you migrate from rangebar-py v7.x to v8.0.

## Breaking Changes

### 1. `get_range_bars()` Now Returns Iterator by Default

**Before (v7.x):**

```python
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")
# Returns: pd.DataFrame
```

**After (v8.0):**

```python
# Default: Returns Iterator[pl.DataFrame]
for batch in get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30"):
    process(batch)  # Each batch is ~10,000 bars

# Or use materialize=True for single DataFrame (old behavior)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", materialize=True)
# Returns: pd.DataFrame (same as v7.x)
```

### 2. Polars DataFrames in Streaming Mode

When `materialize=False` (default), each batch is a `polars.DataFrame`:

```python
for batch in get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30"):
    # batch is pl.DataFrame
    pandas_batch = batch.to_pandas()
```

### 3. Per-Python-Version Wheels

v8.0 removes `abi3` (Python stable ABI) to enable Arrow export features.
This means:

- Wheels are built for Python 3.13
- Slightly larger download for each Python version
- No change to API or functionality

## Quick Migration Checklist

| v7.x Code                  | v8.0 Equivalent                                             |
| -------------------------- | ----------------------------------------------------------- |
| `df = get_range_bars(...)` | `df = get_range_bars(..., materialize=True)`                |
| Process full dataset       | Iterate over batches                                        |
| `pd.DataFrame` everywhere  | `pl.DataFrame` in batches, `pd.DataFrame` when materialized |

## Backward Compatibility Shim

For gradual migration, use the deprecated compatibility function:

```python
from rangebar import get_range_bars_pandas

# This works like v7.x but emits DeprecationWarning
df = get_range_bars_pandas("BTCUSDT", "2024-01-01", "2024-06-30")
```

**Note:** `get_range_bars_pandas()` will be removed in v9.0.

## Memory Improvements

The streaming architecture in v8.0 dramatically reduces memory usage:

| Workload        | v7.x Peak | v8.0 Peak |
| --------------- | --------- | --------- |
| 1-month BTCUSDT | 5.6 GB    | ~50 MB    |
| 6-month BTCUSDT | ~34 GB    | ~50 MB    |
| 1-year BTCUSDT  | ~68 GB    | ~50 MB    |

Peak memory is now independent of date range.

## New Features

### Streaming Trade Fetch

```python
from rangebar._core import stream_binance_trades

# Iterate over 6-hour trade chunks
for trades in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-31"):
    process(trades)  # Arrow RecordBatch
```

### Arrow Export

```python
from rangebar._core import bars_to_arrow, trades_to_arrow
import polars as pl

# Zero-copy conversion to Polars
arrow_batch = bars_to_arrow(bar_dicts)
df = pl.from_arrow(arrow_batch)
```

### Incremental Cache Writes

```python
from rangebar.clickhouse.cache import RangeBarCache

cache = RangeBarCache()
for batch in get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30"):
    cache.store_bars_batch("BTCUSDT", 250, batch)
```

## Need Help?

- [Full API Reference](/docs/api.md)
- [Architecture Documentation](/docs/ARCHITECTURE.md)
- [Report Issues](https://github.com/terrylica/rangebar-py/issues)
