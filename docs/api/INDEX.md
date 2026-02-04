# API Reference Index

**Navigation**: [CLAUDE.md](/CLAUDE.md) | [Architecture](/docs/ARCHITECTURE.md)

---

## Quick Route (AI Agent Decision Tree)

| User Question / Task                | Read This File                                             |
| ----------------------------------- | ---------------------------------------------------------- |
| Generate range bars from date range | [primary-api.md](./primary-api.md#get_range_bars)          |
| Get exactly N bars for ML           | [primary-api.md](./primary-api.md#get_n_range_bars)        |
| Long backfills (>30 days)           | [cache-api.md](./cache-api.md#populate_cache_resumable)    |
| Process existing trades             | [processing-api.md](./processing-api.md)                   |
| RangeBarProcessor class             | [processing-api.md](./processing-api.md#rangebarprocessor) |
| Validate bar continuity             | [validation-api.md](./validation-api.md)                   |
| Constants/presets                   | [utilities.md](./utilities.md)                             |
| Error/troubleshooting               | [faq.md](./faq.md)                                         |

---

## File Summary

| File                                     | Lines | Content                                                                |
| ---------------------------------------- | ----- | ---------------------------------------------------------------------- |
| [primary-api.md](./primary-api.md)       | ~350  | `get_range_bars()`, `get_n_range_bars()`, ouroboros, exchange sessions |
| [cache-api.md](./cache-api.md)           | ~150  | `populate_cache_resumable()`, ClickHouse cache workflow                |
| [processing-api.md](./processing-api.md) | ~350  | `process_trades_*`, `RangeBarProcessor` class                          |
| [validation-api.md](./validation-api.md) | ~200  | Data formats, error handling, continuity validation                    |
| [utilities.md](./utilities.md)           | ~150  | Constants, conversion utilities, module architecture                   |
| [faq.md](./faq.md)                       | ~250  | Common questions, troubleshooting, comparisons                         |

---

## Quick Reference

```python
from rangebar import get_range_bars, get_n_range_bars, process_trades_polars

# Date-bounded API - fetch and convert with date range
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded API - get exactly N bars (ML training, walk-forward)
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# Long date ranges (>30 days) - use cache workflow
from rangebar import populate_cache_resumable

# Step 1: Populate cache (resumable, memory-safe)
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")

# Step 2: Read from cache
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")

# Polars API - 2-3x faster for Polars users
import polars as pl
trades = pl.scan_parquet("trades.parquet")
df = process_trades_polars(trades, threshold_decimal_bps=250)
```

---

## API Selection Guide

```
Starting Point?
├── Date range > 30 days? → populate_cache_resumable() first, then get_range_bars()
├── Need data fetching (date range ≤ 30 days)? → get_range_bars() [DATE-BOUNDED]
├── Need exactly N bars (ML/walk-forward)? → get_n_range_bars() [COUNT-BOUNDED]
├── Have pandas DataFrame → process_trades_to_dataframe()
├── Have Polars DataFrame/LazyFrame → process_trades_polars() [2-3x faster]
└── Have Iterator (large data) → process_trades_chunked()
```

---

## Constants Quick Reference

```python
from rangebar import (
    TIER1_SYMBOLS,           # 18 high-liquidity symbols
    THRESHOLD_PRESETS,       # {'micro': 10, 'tight': 50, 'standard': 100, ...}
    THRESHOLD_DECIMAL_MIN,   # 1 (0.001%)
    THRESHOLD_DECIMAL_MAX,   # 100000 (100%)
    LONG_RANGE_DAYS,         # 30 (MEM-013 threshold)
    MICROSTRUCTURE_COLUMNS,  # v7.0+ feature columns
)
```

See [utilities.md](./utilities.md) for full constant documentation.

---

**Last Updated**: 2026-02-03
