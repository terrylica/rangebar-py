# Processing API Reference

**Navigation**: [INDEX.md](./INDEX.md) | [Primary API](./primary-api.md) | [Cache API](./cache-api.md)

---

## process_trades_to_dataframe()

**Convenience function for one-step conversion to backtesting.py-compatible DataFrame.**

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict[str, Union[int, float]]], pd.DataFrame],
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame
```

### Parameters

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

### Returns

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Index: `DatetimeIndex` (timezone-naive UTC)
  - Compatible with backtesting.py

### Raises

- **ValueError**: If `trades` is empty, missing required columns, or `threshold_decimal_bps` is invalid
- **RuntimeError**: If processing fails (e.g., trades not sorted, internal error)
- **KeyError**: If trade dict missing required keys

### Example

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

### Notes

- Trades must be in chronological order (sorted by timestamp)
- If no range bars are generated (price movement < threshold), returns empty DataFrame
- Timestamp precision: microseconds (inherited from rangebar-core)
- Price precision: 8 decimal places (fixed-point arithmetic)

---

## process_trades_polars()

**Optimized API for Polars users - 2-3x faster than `process_trades_to_dataframe()`.**

```python
def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame
```

### Parameters

- **trades**: `polars.DataFrame` or `polars.LazyFrame`
  - Trade data with columns:
    - `timestamp`: Unix timestamp in milliseconds (int64)
    - `price`: Trade price (float)
    - `quantity` or `volume`: Trade volume (float)
    - `is_buyer_maker` (optional): For microstructure features

- **threshold_decimal_bps**: `int`, default=250
  - Range bar threshold in decimal basis points (dbps)

### Returns

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Compatible with backtesting.py

### Performance Benefits

| Feature                | Benefit                                   |
| ---------------------- | ----------------------------------------- |
| **LazyFrame support**  | Predicate pushdown - filter at I/O layer  |
| **Minimal conversion** | Only required columns extracted           |
| **Chunked processing** | 100K records per batch                    |
| **Memory efficient**   | 10-100x reduction vs full materialization |

### Example

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

### When to Use

| Scenario              | Recommended API                            |
| --------------------- | ------------------------------------------ |
| Have Polars DataFrame | `process_trades_polars()`                  |
| Have pandas DataFrame | `process_trades_to_dataframe()`            |
| Have list of dicts    | `process_trades_to_dataframe()`            |
| Large Parquet files   | `process_trades_polars()` with `LazyFrame` |
| Streaming pipeline    | `get_range_bars(..., materialize=False)`   |

### Notes

- Returns pandas DataFrame for backtesting.py compatibility
- To get Polars output, use `get_range_bars(..., materialize=False)` which returns `Iterator[pl.DataFrame]`
- Trades must be in chronological order

---

## RangeBarProcessor

**Low-level API for range bar processing with state management.**

```python
class RangeBarProcessor:
    def __init__(self, threshold_decimal_bps: int) -> None: ...
    def process_trades(self, trades: List[Dict[str, Union[int, float]]]) -> List[Dict[str, float]]: ...
    def to_dataframe(self, bars: List[Dict[str, float]]) -> pd.DataFrame: ...
    def reset(self) -> None: ...
```

### `__init__(threshold_decimal_bps)`

Create a new range bar processor.

**Parameters**:

- **threshold_decimal_bps**: `int`
  - Range bar threshold in decimal basis point units
  - Must be positive (>0)

**Raises**:

- **ValueError**: If `threshold_decimal_bps` <= 0

**Example**:

```python
from rangebar import RangeBarProcessor

# Create processor with 0.25% threshold
processor = RangeBarProcessor(threshold_decimal_bps=250)

# Access threshold
print(processor.threshold_decimal_bps)  # 250
```

---

### `process_trades(trades)`

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

### `to_dataframe(bars)`

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

### `reset()`

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

### Attribute: `threshold_decimal_bps`

**Type**: `int` (read-only)

The threshold value in decimal basis point units.

**Example**:

```python
processor = RangeBarProcessor(threshold_decimal_bps=250)
print(processor.threshold_decimal_bps)  # 250
print(processor.threshold_decimal_bps * 0.001)  # 0.25 (percentage)
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

**Last Updated**: 2026-02-03
