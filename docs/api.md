# rangebar-py API Reference

<!-- Version controlled by semantic-release via pyproject.toml -->

**Last Updated**: 2026-01-29

---

## Overview

rangebar-py provides a high-level Python API for converting trade data to range bars, optimized for backtesting.py integration. The package offers multiple usage patterns:

1. **Date-bounded**: `get_range_bars()` - Fetch and convert with date range (recommended for backtesting)
2. **Count-bounded**: `get_n_range_bars()` - Get exactly N bars (recommended for ML training)
3. **High-level**: `process_trades_to_dataframe()` - Process existing trade data
4. **Polars-optimized**: `process_trades_polars()` - 2-3x faster with Polars
5. **Streaming**: `process_trades_chunked()` - Memory-safe for large datasets

---

## Quick Reference

```python
from rangebar import get_range_bars, get_n_range_bars, process_trades_polars

# Date-bounded API - fetch and convert with date range
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded API - get exactly N bars (ML training, walk-forward)
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# With preset threshold
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold_decimal_bps="tight")

# Polars API - 2-3x faster for Polars users (recommended)
import polars as pl
trades = pl.scan_parquet("trades.parquet")  # LazyFrame for predicate pushdown
df = process_trades_polars(trades, threshold_decimal_bps=250)

# Streaming mode - memory-efficient Iterator[pl.DataFrame]
for batch in get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", materialize=False):
    process_batch(batch)  # Each batch is a pl.DataFrame
```

---

## Configuration Constants

All constants are centralized in `rangebar.constants` (SSoT) and re-exported from the main package.

```python
from rangebar import (
    TIER1_SYMBOLS,
    THRESHOLD_PRESETS,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_DECIMAL_MAX,
    MICROSTRUCTURE_COLUMNS,
    EXCHANGE_SESSION_COLUMNS,
    ALL_OPTIONAL_COLUMNS,
)

# 18 high-liquidity symbols available on all Binance markets
TIER1_SYMBOLS  # ('AAVE', 'ADA', 'AVAX', 'BCH', 'BNB', 'BTC', ...)

# Named threshold presets (in decimal basis points)
THRESHOLD_PRESETS  # {'micro': 10, 'tight': 50, 'standard': 100, 'medium': 250, 'wide': 500, 'macro': 1000}

# Valid threshold range
THRESHOLD_DECIMAL_MIN  # 1 (0.001%)
THRESHOLD_DECIMAL_MAX  # 100000 (100%)

# Microstructure feature columns (v7.0+)
MICROSTRUCTURE_COLUMNS  # ('vwap', 'buy_volume', 'sell_volume', 'duration_us', 'ofi', ...)

# Exchange session columns (Ouroboros feature)
EXCHANGE_SESSION_COLUMNS  # ('exchange_session_sydney', 'exchange_session_tokyo', ...)

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

## Primary API: `get_range_bars()`

**The recommended single entry point for all range bar generation.**

```python
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = "week",
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,
    include_orphaned_bars: bool = False,
    verify_checksum: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame
```

### Parameters

- **symbol**: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
- **start_date**: Start date in YYYY-MM-DD format
- **end_date**: End date in YYYY-MM-DD format
- **threshold_decimal_bps**: Threshold in decimal basis points or preset name (default: 250)
  - Integer: `250` = 25bps = 0.25%
  - Preset: `"micro"`, `"tight"`, `"standard"`, `"medium"`, `"wide"`, `"macro"`
- **ouroboros**: Reset mode for reproducible bar construction (default: `"week"`)
  - `"week"`: Reset at Sunday 00:00:00 UTC (smallest granularity, default)
  - `"month"`: Reset at 1st of each month 00:00:00 UTC
  - `"year"`: Reset at January 1st 00:00:00 UTC
- **source**: Data source - `"binance"` or `"exness"` (default: `"binance"`)
- **market**: Market type - `"spot"`, `"futures-um"`, `"futures-cm"` (default: `"spot"`)
- **include_microstructure**: Include vwap, buy_volume, sell_volume (default: False)
- **include_exchange_sessions**: Include exchange session flags (default: False). See [Exchange Sessions](#exchange-sessions) below.
- **include_orphaned_bars**: Include incomplete bars at ouroboros boundaries (default: False)
- **verify_checksum**: Verify SHA-256 checksum of downloaded data (default: True). Enabled by default for data integrity. Set to False for faster downloads when integrity is verified elsewhere. (Issue #43)
- **use_cache**: Cache tick data locally (default: True)

### Returns

- **pd.DataFrame**: OHLCV DataFrame with DatetimeIndex and columns `Open`, `High`, `Low`, `Close`, `Volume`

### Example

```python
from rangebar import get_range_bars

# Basic usage
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Using preset threshold
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", threshold_decimal_bps="tight")

# Binance USD-M Futures
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

# With microstructure data
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)

# With year-based ouroboros (reproducible across researchers)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")

# With exchange session flags
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_exchange_sessions=True)
# Adds: exchange_session_sydney, exchange_session_tokyo, exchange_session_london, exchange_session_newyork
```

---

## Exchange Sessions

When `include_exchange_sessions=True`, the output includes boolean columns indicating which traditional exchange market sessions were active at each bar's close time:

| Column                     | Exchange | Local Hours           | Notes                          |
| -------------------------- | -------- | --------------------- | ------------------------------ |
| `exchange_session_sydney`  | ASX      | 10:00-16:00 AEDT/AEST | Australian Securities Exchange |
| `exchange_session_tokyo`   | TSE      | 09:00-15:00 JST       | Tokyo Stock Exchange           |
| `exchange_session_london`  | LSE      | 08:00-17:00 GMT/BST   | London Stock Exchange          |
| `exchange_session_newyork` | NYSE     | 10:00-16:00 EST/EDT   | New York Stock Exchange        |

**Use Cases**:

- Analyze crypto volatility during traditional market hours
- Identify session overlaps (e.g., London/NY overlap)
- Feature engineering for ML models

**Example**:

```python
df = get_range_bars("BTCUSDT", "2024-01-15", "2024-01-16", include_exchange_sessions=True)

# Filter bars during London session only
london_bars = df[df["exchange_session_london"]]

# Find bars during London/NY overlap
overlap_bars = df[df["exchange_session_london"] & df["exchange_session_newyork"]]
```

**Note**: Session detection uses `zoneinfo` for DST-aware timezone conversion. Session boundaries are hour-granularity approximations of actual exchange trading hours.

---

## Ouroboros: Reproducible Range Bar Construction

The **Ouroboros** feature (named after the serpent eating its tail, representing cyclical boundaries) enables reproducible range bar construction by resetting processor state at deterministic boundaries.

### Why Ouroboros?

| Problem                    | How Ouroboros Helps                                    |
| -------------------------- | ------------------------------------------------------ |
| **Reproducibility**        | Two researchers starting from Jan 1 get identical bars |
| **Cache granularity**      | Entire years/months/weeks can be cached independently  |
| **Cross-study comparison** | Standardized starting points for academic/research use |

### Ouroboros Modes

| Mode      | Boundary                  | Use Case                                         |
| --------- | ------------------------- | ------------------------------------------------ |
| `"week"`  | Sunday 00:00:00 UTC       | Default, smallest granularity                    |
| `"month"` | 1st of month 00:00:00 UTC | Monthly analysis                                 |
| `"year"`  | January 1 00:00:00 UTC    | Annual reports, cross-researcher reproducibility |

### Example: Year-Based Reproducibility

```python
from rangebar import get_range_bars

# Two researchers with the same parameters get identical results
df1 = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")
df2 = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")
assert df1.equals(df2)  # Guaranteed reproducibility!
```

### Orphaned Bars

At ouroboros boundaries, incomplete bars are "orphaned" (reset). You can include these for analysis:

```python
df = get_range_bars(
    "BTCUSDT",
    "2023-12-01",
    "2024-01-31",
    ouroboros="year",
    include_orphaned_bars=True,
)

# Orphaned bars have metadata
orphans = df[df.get("is_orphan", False)]
print(f"Orphaned bars at year boundary: {len(orphans)}")
```

### Programmatic Boundary Access

```python
from rangebar import get_ouroboros_boundaries
from datetime import date

# Get all week boundaries in a date range
boundaries = get_ouroboros_boundaries(
    start=date(2024, 1, 1),
    end=date(2024, 1, 31),
    mode="week",
)
# [datetime(2024, 1, 7), datetime(2024, 1, 14), datetime(2024, 1, 21), datetime(2024, 1, 28)]
```

---

## Count-Bounded API: `get_n_range_bars()`

**Get exactly N range bars - useful for ML training, walk-forward optimization, and consistent backtest comparisons.**

Unlike `get_range_bars()` which uses date bounds (producing variable bar counts), this function returns a deterministic number of bars.

```python
def get_n_range_bars(
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: int | str = 250,
    *,
    end_date: str | None = None,
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame
```

### Parameters

- **symbol**: Trading symbol (e.g., "BTCUSDT")
- **n_bars**: Number of bars to retrieve (must be > 0)
- **threshold_decimal_bps**: Threshold in decimal basis points or preset name (default: 250)
- **end_date**: End date in YYYY-MM-DD format, or None for most recent data
- **source**: Data source - `"binance"` or `"exness"` (default: `"binance"`)
- **market**: Market type - `"spot"`, `"futures-um"`, `"futures-cm"` (default: `"spot"`)
- **include_microstructure**: Include vwap, buy_volume, sell_volume (default: False)
- **use_cache**: Use ClickHouse cache for bar retrieval/storage (default: True)
- **fetch_if_missing**: Fetch and process new data if cache doesn't have enough bars (default: True)
- **max_lookback_days**: Safety limit - maximum days to look back when fetching (default: 90)
- **warn_if_fewer**: Emit UserWarning if returning fewer bars than requested (default: True)
- **cache_dir**: Custom cache directory for tick data (default: platform-specific)

### Returns

- **pd.DataFrame**: OHLCV DataFrame with exactly `n_bars` rows (or fewer if insufficient data), sorted chronologically (oldest first)

### Cache Behavior

- **Fast path**: If ClickHouse cache has >= n_bars, returns immediately (~50ms)
- **Slow path**: If cache has < n_bars and `fetch_if_missing=True`, fetches additional data, computes bars, stores in cache, returns

### Example

```python
from rangebar import get_n_range_bars

# Get last 10,000 bars for ML training
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# Get 5,000 bars ending at specific date for walk-forward
df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

# With safety limit (won't fetch more than 30 days of data)
df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)

# Using preset threshold
df = get_n_range_bars("BTCUSDT", n_bars=500, threshold_decimal_bps="tight")
```

### Raises

- **ValueError**: If `n_bars <= 0`, invalid threshold, or invalid date format
- **RuntimeError**: If ClickHouse not available when `use_cache=True`, or data fetching failed

---

## Module: `rangebar`

### `process_trades_to_dataframe()`

**Convenience function for one-step conversion to backtesting.py-compatible DataFrame.**

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict[str, Union[int, float]]], pd.DataFrame],
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame
```

#### Parameters

- **trades**: `List[Dict]` or `pd.DataFrame`
  - Trade data with required columns: `timestamp`, `price`, `quantity`
  - If `List[Dict]`:
    - `timestamp`: Unix timestamp in milliseconds (int)
    - `price`: Trade price (float)
    - `quantity`: Trade volume (float) - can also use `volume` key
  - If `pd.DataFrame`:
    - Must have columns: `timestamp`, `price`, `quantity`
    - `timestamp` can be datetime (auto-converted to ms) or int
    - Other columns ignored

- **threshold_decimal_bps**: `int`, default=250
  - Range bar threshold in decimal basis point (dbps) units
  - Examples:
    - `100` = 100 dbps = 0.1%
    - `250` = 250 dbps = 0.25% (recommended default)
    - `500` = 500 dbps = 0.5%
    - `1000` = 1000 dbps = 1.0%
  - Must be positive (>0)

#### Returns

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Index: `DatetimeIndex` (timezone-naive UTC)
  - Compatible with backtesting.py

#### Raises

- **ValueError**: If `trades` is empty, missing required columns, or `threshold_decimal_bps` is invalid
- **RuntimeError**: If processing fails (e.g., trades not sorted, internal error)
- **KeyError**: If trade dict missing required keys

#### Example

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Example 1: List of dicts
trades = [
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    {"timestamp": 1704067220000, "price": 42000.0, "quantity": 1.8},
]

df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
print(df.head())
#                          Open      High       Low     Close  Volume
# timestamp
# 2024-01-01 00:00:10  42000.0  42105.0  42000.0  42105.0    3.8

# Example 2: DataFrame input
trades_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
    "price": [42000 + i*10 for i in range(100)],
    "quantity": [1.5] * 100,
})

df = process_trades_to_dataframe(trades_df, threshold_decimal_bps=250)
print(f"Generated {len(df)} range bars from {len(trades_df)} trades")
```

#### Notes

- Trades must be in chronological order (sorted by timestamp)
- If no range bars are generated (price movement < threshold), returns empty DataFrame
- Timestamp precision: microseconds (inherited from rangebar-core)
- Price precision: 8 decimal places (fixed-point arithmetic)

---

### `process_trades_polars()`

**Optimized API for Polars users - 2-3x faster than `process_trades_to_dataframe()`.**

```python
def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame
```

#### Parameters

- **trades**: `polars.DataFrame` or `polars.LazyFrame`
  - Trade data with columns:
    - `timestamp`: Unix timestamp in milliseconds (int64)
    - `price`: Trade price (float)
    - `quantity` or `volume`: Trade volume (float)
    - `is_buyer_maker` (optional): For microstructure features

- **threshold_decimal_bps**: `int`, default=250
  - Range bar threshold in decimal basis points (dbps)

#### Returns

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Compatible with backtesting.py

#### Performance Benefits

| Feature                | Benefit                                   |
| ---------------------- | ----------------------------------------- |
| **LazyFrame support**  | Predicate pushdown - filter at I/O layer  |
| **Minimal conversion** | Only required columns extracted           |
| **Chunked processing** | 100K records per batch                    |
| **Memory efficient**   | 10-100x reduction vs full materialization |

#### Example

```python
import polars as pl
from rangebar import process_trades_polars

# With LazyFrame (predicate pushdown for efficient filtering)
lazy_df = pl.scan_parquet("trades.parquet")
lazy_filtered = lazy_df.filter(pl.col("timestamp") >= 1704067200000)
df = process_trades_polars(lazy_filtered, threshold_decimal_bps=250)

# With DataFrame
df = pl.read_parquet("trades.parquet")
bars = process_trades_polars(df)

# With explicit column selection (minimal memory)
lazy_df = pl.scan_parquet("trades/*.parquet").select([
    pl.col("timestamp"),
    pl.col("price"),
    pl.col("quantity"),
])
bars = process_trades_polars(lazy_df.filter(pl.col("price") > 40000))
```

#### When to Use

| Scenario              | Recommended API                            |
| --------------------- | ------------------------------------------ |
| Have Polars DataFrame | `process_trades_polars()`                  |
| Have pandas DataFrame | `process_trades_to_dataframe()`            |
| Have list of dicts    | `process_trades_to_dataframe()`            |
| Large Parquet files   | `process_trades_polars()` with `LazyFrame` |
| Streaming pipeline    | `get_range_bars(..., materialize=False)`   |

#### Notes

- Returns pandas DataFrame for backtesting.py compatibility
- To get Polars output, use `get_range_bars(..., materialize=False)` which returns `Iterator[pl.DataFrame]`
- Trades must be in chronological order

---

### Class: `RangeBarProcessor`

**Low-level API for range bar processing with state management.**

```python
class RangeBarProcessor:
    def __init__(self, threshold_decimal_bps: int) -> None: ...
    def process_trades(self, trades: List[Dict[str, Union[int, float]]]) -> List[Dict[str, float]]: ...
    def to_dataframe(self, bars: List[Dict[str, float]]) -> pd.DataFrame: ...
    def reset(self) -> None: ...
```

#### `__init__(threshold_decimal_bps)`

Create a new range bar processor.

**Parameters**:

- **threshold_decimal_bps**: `int`
  - Range bar threshold in decimal basis point units
  - Must be positive (>0)

**Raises**:

- **ValueError**: If `threshold_decimal_bps` ≤ 0

**Example**:

```python
from rangebar import RangeBarProcessor

# Create processor with 0.25% threshold
processor = RangeBarProcessor(threshold_decimal_bps=250)

# Access threshold
print(processor.threshold_decimal_bps)  # 250
```

---

#### `process_trades(trades)`

Process trades into range bars (intermediate format).

**Parameters**:

- **trades**: `List[Dict[str, Union[int, float]]]`
  - List of trade dictionaries with keys:
    - `timestamp`: Unix timestamp in milliseconds (int)
    - `price`: Trade price (float)
    - `quantity`: Trade volume (float) - or use `volume` key

**Returns**:

- **List[Dict[str, float]]**
  - List of range bar dictionaries with keys:
    - `timestamp`: RFC3339 timestamp string
    - `open`: Bar open price
    - `high`: Bar high price
    - `low`: Bar low price
    - `close`: Bar close price
    - `volume`: Bar total volume

**Raises**:

- **ValueError**: If trades empty or missing required fields
- **RuntimeError**: If processing fails
- **KeyError**: If trade missing required keys

**Example**:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)

trades = [
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
]

bars = processor.process_trades(trades)
print(bars[0])
# {'timestamp': '2024-01-01T00:00:10+00:00',
#  'open': 42000.0,
#  'high': 42105.0,
#  'low': 42000.0,
#  'close': 42105.0,
#  'volume': 3.8}
```

---

#### `to_dataframe(bars)`

Convert range bars to backtesting.py-compatible DataFrame.

**Parameters**:

- **bars**: `List[Dict[str, float]]`
  - List of range bar dictionaries (from `process_trades()`)

**Returns**:

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Empty DataFrame if `bars` is empty

**Example**:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)
bars = processor.process_trades(trades)
df = processor.to_dataframe(bars)

print(df.head())
#                          Open      High       Low     Close  Volume
# timestamp
# 2024-01-01 00:00:10  42000.0  42105.0  42000.0  42105.0    3.8
```

---

#### `reset()`

Reset processor state (clears internal bar accumulation).

**Returns**: `None`

**Example**:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)

# Process first batch
bars1 = processor.process_trades(trades1)

# Reset before processing new batch
processor.reset()

# Process second batch (independent)
bars2 = processor.process_trades(trades2)
```

**Note**: Use `reset()` when processing unrelated datasets. Not needed if creating new processor instance.

---

#### Attribute: `threshold_decimal_bps`

**Type**: `int` (read-only)

The threshold value in decimal basis point units.

**Example**:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)
print(processor.threshold_decimal_bps)  # 250
print(processor.threshold_decimal_bps * 0.001)  # 0.25 (percentage)
```

---

## Data Formats

### Input: Trade Data

**List of Dicts Format**:

```python
[
    {
        "timestamp": 1704067200000,  # Unix ms (required)
        "price": 42000.0,            # Trade price (required)
        "quantity": 1.5,             # Trade volume (required, or use "volume")
    },
    # ... more trades
]
```

**DataFrame Format**:

```python
pd.DataFrame({
    "timestamp": [1704067200000, 1704067210000, ...],  # Unix ms or datetime
    "price": [42000.0, 42105.0, ...],
    "quantity": [1.5, 2.3, ...],  # or "volume" column
})
```

**Field Requirements**:

- `timestamp`:
  - Must be Unix timestamp in milliseconds (int)
  - Or pandas datetime (auto-converted)
  - Must be monotonically increasing (sorted)
- `price`:
  - Must be positive float
  - Precision preserved to 8 decimal places
- `quantity` (or `volume`):
  - Must be positive float
  - Represents trade volume

---

### Output: Range Bar DataFrame

**Structure**:

```python
                          Open      High       Low     Close  Volume
timestamp
2024-01-01 00:00:10  42000.0  42105.0  42000.0  42105.0    3.8
2024-01-01 00:03:25  42105.0  42210.0  42105.0  42210.0    7.2
...
```

**Properties**:

- **Index**: `pd.DatetimeIndex`
  - Timezone-naive UTC timestamps
  - Monotonically increasing
  - No duplicates
  - Bar closing timestamp
- **Columns**: `["Open", "High", "Low", "Close", "Volume"]`
  - Capitalized (backtesting.py convention)
  - All float64 dtype
  - No NaN values
- **OHLC Invariants**:
  - `High >= max(Open, Close)`
  - `Low <= min(Open, Close)`
  - `Volume > 0`

---

## Error Handling

### ValueError

**Causes**:

- Empty trades list
- Invalid threshold_decimal_bps (≤0)
- Missing required columns/keys
- Negative price/quantity

**Example**:

```python
# Empty trades
process_trades_to_dataframe([])
# ValueError: Trades list is empty

# Invalid threshold
RangeBarProcessor(threshold_decimal_bps=0)
# ValueError: threshold_decimal_bps must be positive

# Missing fields
process_trades_to_dataframe([{"timestamp": 123, "price": 42000}])
# ValueError: Trade 0 missing: {'quantity'}
```

---

### RuntimeError

**Causes**:

- Trades not sorted chronologically
- Internal processing error
- Rust-level failure

**Example**:

```python
# Unsorted trades
trades = [
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 1.0},
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},  # Earlier!
]
process_trades_to_dataframe(trades)
# RuntimeError: Processing failed: trades not sorted by timestamp
```

**Fix**: Sort trades before processing:

```python
df = df.sort_values("timestamp")
trades = sorted(trades, key=lambda t: t["timestamp"])
```

---

### KeyError

**Causes**:

- Trade dict missing required keys (`timestamp`, `price`, `quantity`)

**Example**:

```python
trades = [{"timestamp": 123, "price": 42000}]  # Missing 'quantity'
process_trades_to_dataframe(trades)
# KeyError: Trade 0: missing 'quantity'
```

---

## Usage Patterns

### Pattern 1: Quick Conversion

For simple use cases with Binance CSV:

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Load CSV
trades = pd.read_csv("BTCUSDT-aggTrades.csv")

# Convert to range bars
df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)

# Use with backtesting.py
from backtesting import Backtest
bt = Backtest(df, MyStrategy, cash=10000)
stats = bt.run()
```

---

### Pattern 2: Custom Thresholds

Testing multiple thresholds:

```python
thresholds = [100, 250, 500, 1000]

for threshold in thresholds:
    df = process_trades_to_dataframe(trades, threshold_decimal_bps=threshold)
    print(f"Threshold {threshold}: {len(df)} bars")
```

---

### Pattern 3: Incremental Processing

Processing large datasets in chunks:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)

all_bars = []
for chunk in pd.read_csv("large_file.csv", chunksize=100_000):
    bars = processor.process_trades(chunk.to_dict("records"))
    all_bars.extend(bars)

df = processor.to_dataframe(all_bars)
```

**Note**: Do NOT call `reset()` between chunks if you want continuous processing.

---

### Pattern 4: Multi-Symbol Processing

Processing multiple symbols independently:

```python
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

results = {}
for symbol in symbols:
    trades = pd.read_csv(f"{symbol}-aggTrades.csv")
    results[symbol] = process_trades_to_dataframe(trades, threshold_decimal_bps=250)

# Compare bar counts
for symbol, df in results.items():
    print(f"{symbol}: {len(df)} bars")
```

---

## Utility Functions

### DataFrame Conversion Utilities

These utilities are available from `rangebar.conversion` and re-exported from the main package.

#### `normalize_arrow_dtypes()`

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

#### `normalize_temporal_precision()`

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

## Comparison with Alternatives

### vs Time-Based Bars

**Range Bars**:

- ✅ Market-adaptive (bars form based on volatility)
- ✅ No lookahead bias
- ✅ Equal information per bar (fixed price movement)
- ❌ Variable time intervals (harder to align with external events)

**Time Bars**:

- ✅ Fixed time intervals (easy to align with news, etc.)
- ❌ Variable information per bar (volatile periods compressed)
- ❌ Potential lookahead bias (if not careful)
- ❌ Inverse timeframe effect (shorter timeframes ≠ better)

### vs Tick Bars

**Range Bars**:

- ✅ Filters noise (only price movements matter)
- ✅ Fewer bars (more efficient)
- ✅ Better for trend-following strategies

**Tick Bars**:

- ✅ Fixed number of trades per bar
- ❌ Ignores price movement (1 cent move = 10% move)
- ❌ More bars (slower backtests)

### vs Volume Bars

**Range Bars**:

- ✅ Focus on price movement (more intuitive for trading)
- ✅ Works well with MA crossover strategies

**Volume Bars**:

- ✅ Fixed volume per bar
- ❌ Ignores price movement (large volume with small price change)

---

## FAQ

### Q: What threshold should I use?

**A**: Start with `threshold_decimal_bps=250` (0.25%). This generates ~200 bars/day for BTC/USDT, similar to 1-hour time bars. Adjust based on:

- Higher volatility → higher threshold (avoid too many bars)
- Lower volatility → lower threshold (avoid too few bars)
- Shorter strategies → lower threshold (more bars)
- Longer strategies → higher threshold (fewer bars)

### Q: Why are no range bars generated?

**A**: Price movement is below threshold. Either:

1. Lower threshold (e.g., 250 → 100)
2. Use more volatile data
3. Check data timespan (need sufficient trades)

### Q: Can I use this with live trading?

**A**: rangebar-py is designed for backtesting, not live trading. For live trading, you'd need:

- Streaming API (process trades incrementally)
- Real-time bar completion detection
- State persistence (recover from failures)

These features are planned for future releases.

### Q: How do I handle multiple symbols?

**A**: Create separate processors for each symbol:

```python
processors = {
    "BTCUSDT": RangeBarProcessor(threshold_decimal_bps=250),
    "ETHUSDT": RangeBarProcessor(threshold_decimal_bps=250),
}

for symbol, processor in processors.items():
    trades = load_trades(symbol)
    bars = processor.process_trades(trades)
    save_bars(symbol, processor.to_dataframe(bars))
```

### Q: Can I process tick data?

**A**: Yes, as long as you have timestamp, price, and quantity. Convert to trade format:

```python
ticks = pd.read_csv("ticks.csv")  # time, bid, ask, bid_size, ask_size

# Convert to trades (midpoint)
trades = pd.DataFrame({
    "timestamp": ticks["time"],
    "price": (ticks["bid"] + ticks["ask"]) / 2,
    "quantity": (ticks["bid_size"] + ticks["ask_size"]) / 2,
})

df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
```

### Q: Why did I get 97K bars during a flash crash? (Issue #36)

**A**: This is expected behavior for range bars during extreme volatility events.

During a flash crash (e.g., BTC dropping 15% in minutes), the price rapidly crosses the threshold many times. Each threshold breach creates a new bar. For example:

- **Normal day**: BTC moves 2% total → ~80 bars with 250bps threshold
- **Flash crash**: BTC moves 15% in 10 minutes → hundreds of threshold breaches → thousands of bars

This is actually a **feature, not a bug**:

1. **Information-preserving**: Each bar represents equal price movement (threshold size)
2. **No lookahead bias**: Bars close at threshold breach, not at arbitrary times
3. **Volatility-adaptive**: More bars during volatile periods = more granular data for backtesting

**Practical implications for backtesting**:

```python
# Flash crash day will have many more bars than quiet day
quiet_day = get_range_bars("BTCUSDT", "2024-01-15", "2024-01-15")  # ~200 bars
crash_day = get_range_bars("BTCUSDT", "2024-03-05", "2024-03-05")  # ~5,000 bars (hypothetical crash)
```

**References**:

- Mandelbrot & Hudson (2004): "The (Mis)behavior of Markets" - fat tails and volatility clustering
- Easley, López de Prado & O'Hara (2012): "The Volume Clock" - information-based sampling

**If you want fewer bars during volatile periods**: Increase threshold or use time-based bars (but lose the benefits of information-based sampling).

### Q: What are the expected ranges for microstructure features? (Issue #32)

**A**: Range bar microstructure features have documented value ranges:

| Feature                | Expected Range | Interpretation                                       |
| ---------------------- | -------------- | ---------------------------------------------------- |
| `duration_us`          | [0, +∞)        | Microseconds; 0 = instantaneous bar (rapid trades)   |
| `ofi`                  | [-1, 1]        | Order Flow Imbalance; -1 = all sells, +1 = all buys  |
| `vwap_close_deviation` | ~[-1, 1]       | Can exceed ±1 during gaps; near 0 = close near VWAP  |
| `price_impact`         | [0, +∞)        | Price change per unit volume; higher = more impact   |
| `kyle_lambda_proxy`    | (-∞, +∞)       | Market impact coefficient; bounded for typical data  |
| `trade_intensity`      | [0, +∞)        | Trades per second; higher = more active              |
| `volume_per_trade`     | [0, +∞)        | Average trade size; institutional vs retail          |
| `aggression_ratio`     | [0, 100]       | Buy count / sell count (capped at 100)               |
| `aggregation_density`  | [1, +∞)        | Individual trades per agg record; 1 = no aggregation |
| `turnover_imbalance`   | [-1, 1]        | Buy-sell turnover difference normalized              |

**Out-of-range values indicate data issues**:

```python
from rangebar.validation.tier1 import validate_tier1

df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)
result = validate_tier1(df)

if not result["tier1_passed"]:
    print("Validation failed:", result)
    # Check specific bounds violations
    if not result["ofi_bounded"]:
        print("OFI values outside [-1, 1] - possible data corruption")
```

**Academic references for feature formulas**:

- **OFI**: Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- **Kyle Lambda**: Kyle (1985) - "Continuous Auctions and Insider Trading"
- **Trade Intensity**: Hasbrouck (1991) - "Measuring the Information Content of Stock Trades"
- **Aggression Ratio**: Biais, Hillion & Spatt (1995) - "An Empirical Analysis of the Limit Order Book"

---

## Changelog

See [CHANGELOG.md](/CHANGELOG.md) for version history.

---

## See Also

- [README.md](/README.md) - User guide and quick start
- [examples/README.md](/examples/README.md) - Usage examples
- [rangebar_core_api.md](/docs/rangebar_core_api.md) - Rust API documentation
- [backtesting.py docs](https://kernc.github.io/backtesting.py/) - Target framework

---

**Last Updated**: 2026-01-29

<!-- API Version: See pyproject.toml (SSoT controlled by semantic-release) -->

**License**: MIT
