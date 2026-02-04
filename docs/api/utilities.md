# Utilities & Constants API Reference

**Navigation**: [INDEX.md](./INDEX.md) | [Primary API](./primary-api.md) | [Processing API](./processing-api.md)

---

## Configuration Constants

All constants are centralized in `rangebar.constants` (SSoT) and re-exported from the main package.

```python
from rangebar import (
    TIER1_SYMBOLS,
    THRESHOLD_PRESETS,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_DECIMAL_MAX,
    LONG_RANGE_DAYS,
    MICROSTRUCTURE_COLUMNS,
    EXCHANGE_SESSION_COLUMNS,
    TRADE_ID_RANGE_COLUMNS,
    ALL_OPTIONAL_COLUMNS,
)

# 18 high-liquidity symbols available on all Binance markets
TIER1_SYMBOLS  # ('AAVE', 'ADA', 'AVAX', 'BCH', 'BNB', 'BTC', ...)

# Named threshold presets (in decimal basis points)
THRESHOLD_PRESETS  # {'micro': 10, 'tight': 50, 'standard': 100, 'medium': 250, 'wide': 500, 'macro': 1000}

# Valid threshold range
THRESHOLD_DECIMAL_MIN  # 1 (0.001%)
THRESHOLD_DECIMAL_MAX  # 100000 (100%)

# MEM-013 guard threshold
LONG_RANGE_DAYS  # 30 (days before cache workflow required)

# Microstructure feature columns (v7.0+)
MICROSTRUCTURE_COLUMNS  # ('vwap', 'buy_volume', 'sell_volume', 'duration_us', 'ofi', ...)

# Exchange session columns (Ouroboros feature)
EXCHANGE_SESSION_COLUMNS  # ('exchange_session_sydney', 'exchange_session_tokyo', ...)

# Trade ID range columns for data integrity (v12.4+, Issue #72)
TRADE_ID_RANGE_COLUMNS  # ('first_agg_trade_id', 'last_agg_trade_id')

# All optional columns (microstructure + exchange sessions)
ALL_OPTIONAL_COLUMNS  # Union of MICROSTRUCTURE_COLUMNS and EXCHANGE_SESSION_COLUMNS
```

### Direct Module Access

For explicit imports, you can access constants directly from submodules:

```python
# Direct access to constants module
from rangebar.constants import MICROSTRUCTURE_COLUMNS, THRESHOLD_PRESETS

# Direct access to conversion utilities
from rangebar.conversion import normalize_arrow_dtypes, normalize_temporal_precision
```

---

## Threshold Presets

| Preset       | Value | Use Case         |
| ------------ | ----- | ---------------- |
| `"micro"`    | 10    | Scalping         |
| `"tight"`    | 50    | Day trading      |
| `"standard"` | 100   | Swing trading    |
| `"medium"`   | 250   | Default          |
| `"wide"`     | 500   | Position trading |
| `"macro"`    | 1000  | Long-term        |

**Usage**:

```python
from rangebar import get_range_bars

# Using preset name
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold_decimal_bps="tight")

# Using explicit value
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold_decimal_bps=50)
```

---

## DataFrame Conversion Utilities

These utilities are available from `rangebar.conversion` and re-exported from the main package.

### normalize_arrow_dtypes()

Convert PyArrow dtypes to numpy for compatibility between ClickHouse cache and fresh computation.

```python
from rangebar import normalize_arrow_dtypes

# ClickHouse returns double[pyarrow], process_trades returns float64
# This function normalizes them to float64
df = normalize_arrow_dtypes(df)

# Normalize specific columns (e.g., microstructure features)
from rangebar import MICROSTRUCTURE_COLUMNS
df = normalize_arrow_dtypes(df, columns=list(MICROSTRUCTURE_COLUMNS))
```

**Parameters**:

- **df**: `pd.DataFrame` - DataFrame potentially containing PyArrow dtypes
- **columns**: `list[str] | None` - Columns to normalize. If None, normalizes OHLCV columns.

**Returns**: `pd.DataFrame` with numpy dtypes

### normalize_temporal_precision()

Normalize datetime columns to microsecond precision to prevent Polars `SchemaError` when concatenating DataFrames with mixed precision (Issue #44).

```python
from rangebar.conversion import normalize_temporal_precision
import polars as pl

# Before concatenating DataFrames from different sources
normalized_dfs = [normalize_temporal_precision(df) for df in polars_dfs]
combined = pl.concat(normalized_dfs)  # No SchemaError!
```

**Parameters**:

- **pldf**: `pl.DataFrame` - Polars DataFrame to normalize

**Returns**: `pl.DataFrame` with all datetime columns cast to microsecond precision

---

## Module Architecture

rangebar-py follows a modular SSoT (Single Source of Truth) architecture:

```
rangebar/
├── __init__.py              # Public API (~200 lines, re-exports only)
├── constants.py             # SSoT: All constants (MICROSTRUCTURE_COLUMNS, PRESETS, etc.)
├── conversion.py            # SSoT: DataFrame conversion utilities
├── ouroboros.py             # Cyclical reset boundaries + exchange sessions
├── checkpoint.py            # Cache population with resume support
├── processors/              # Range bar processing
│   ├── core.py              # RangeBarProcessor class
│   └── api.py               # process_trades_* functions
├── orchestration/           # High-level data pipelines
│   ├── range_bars.py        # get_range_bars() orchestration
│   ├── count_bounded.py     # get_n_range_bars() + gap filling
│   ├── precompute.py        # precompute_range_bars() pipeline
│   ├── helpers.py           # Binance/Exness fetching helpers
│   └── models.py            # PrecomputeProgress, PrecomputeResult
├── clickhouse/              # ClickHouse cache layer
│   ├── cache.py             # RangeBarCache class
│   ├── bulk_operations.py   # BulkStoreMixin (store_bars_bulk/batch)
│   ├── query_operations.py  # QueryOperationsMixin (get_n_bars, get_bars_by_timestamp_range)
│   └── schema.sql           # Table schema
└── validation/              # Microstructure feature validation
    ├── continuity.py        # Continuity validation (8 dataclasses + functions)
    ├── tier1.py             # Fast validation (<30 sec)
    └── tier2.py             # Statistical validation (~10 min)
```

### Import Patterns

```python
# Recommended: Import from main package (re-exports)
from rangebar import (
    get_range_bars,
    MICROSTRUCTURE_COLUMNS,
    normalize_arrow_dtypes,
)

# Alternative: Import directly from submodules
from rangebar.constants import THRESHOLD_PRESETS
from rangebar.conversion import normalize_temporal_precision
from rangebar.ouroboros import get_ouroboros_boundaries
from rangebar.processors import RangeBarProcessor
from rangebar.orchestration import get_range_bars, precompute_range_bars
from rangebar.validation.continuity import validate_continuity
```

---

## Performance Considerations

### Throughput

- **Rust backend**: >1M trades/sec (synthetic benchmark)
- **DataFrame conversion**: ~100k trades/sec (pandas overhead)
- **Bottleneck**: pandas operations, not Rust processing

### Memory Usage

- **Input**: ~80 bytes/trade (Python dict)
- **Output**: ~40 bytes/bar (DataFrame row)
- **Peak**: ~2x input size during processing

### Optimization Tips

1. **Pre-sort data**: Sort trades before processing (avoid runtime error)
2. **Use DataFrame input**: Slightly faster than list of dicts
3. **Batch processing**: Process in chunks for large datasets (>10M trades)
4. **Threshold selection**: Higher thresholds = fewer bars = faster

---

## Type Hints

Full type annotations available via `.pyi` stubs:

```python
from typing import List, Dict, Union
import pandas as pd

# Fully typed
def process_trades_to_dataframe(
    trades: Union[List[Dict[str, Union[int, float]]], pd.DataFrame],
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame: ...
```

IDE support:

- PyCharm: Full autocomplete
- VS Code: Full IntelliSense
- mypy: Type checking passes

---

## Compatibility

### Python Versions

- **Supported**: 3.13
- **Tested**: 3.13 (development)

### pandas Versions

- **Minimum**: 2.0.0
- **Tested**: 2.0.3, 2.1.4
- **Known issues**: None

### backtesting.py Versions

- **Minimum**: 0.3.0
- **Tested**: 0.3.3
- **Compatibility**: 100% (OHLCV format validated)

---

**Last Updated**: 2026-02-03
