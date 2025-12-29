# rangebar-py API Reference

**Version**: 0.1.0
**Last Updated**: 2024-11-15

---

## Overview

rangebar-py provides a high-level Python API for converting trade data to range bars, optimized for backtesting.py integration. The package offers two usage patterns:

1. **High-level**: `process_trades_to_dataframe()` - One-step conversion
2. **Low-level**: `RangeBarProcessor` class - More control over processing

---

## Quick Reference

```python
from rangebar import process_trades_to_dataframe, RangeBarProcessor

# Quick conversion
df = process_trades_to_dataframe(trades, threshold_bps=250)

# Custom processing
processor = RangeBarProcessor(threshold_bps=250)
bars = processor.process_trades(trades)
df = processor.to_dataframe(bars)
```

---

## Module: `rangebar`

### `process_trades_to_dataframe()`

**Convenience function for one-step conversion to backtesting.py-compatible DataFrame.**

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict[str, Union[int, float]]], pd.DataFrame],
    threshold_bps: int = 250,
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

- **threshold_bps**: `int`, default=250
  - Range bar threshold in 0.1 basis point units
  - Examples:
    - `100` = 10 bps = 0.1%
    - `250` = 25 bps = 0.25% (recommended default)
    - `500` = 50 bps = 0.5%
    - `1000` = 100 bps = 1.0%
  - Must be positive (>0)

#### Returns

- **pd.DataFrame**
  - OHLCV DataFrame with DatetimeIndex
  - Columns: `["Open", "High", "Low", "Close", "Volume"]`
  - Index: `DatetimeIndex` (timezone-naive UTC)
  - Compatible with backtesting.py

#### Raises

- **ValueError**: If `trades` is empty, missing required columns, or `threshold_bps` is invalid
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

df = process_trades_to_dataframe(trades, threshold_bps=250)
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

df = process_trades_to_dataframe(trades_df, threshold_bps=250)
print(f"Generated {len(df)} range bars from {len(trades_df)} trades")
```

#### Notes

- Trades must be in chronological order (sorted by timestamp)
- If no range bars are generated (price movement < threshold), returns empty DataFrame
- Timestamp precision: microseconds (inherited from rangebar-core)
- Price precision: 8 decimal places (fixed-point arithmetic)

---

### Class: `RangeBarProcessor`

**Low-level API for range bar processing with state management.**

```python
class RangeBarProcessor:
    def __init__(self, threshold_bps: int) -> None: ...
    def process_trades(self, trades: List[Dict[str, Union[int, float]]]) -> List[Dict[str, float]]: ...
    def to_dataframe(self, bars: List[Dict[str, float]]) -> pd.DataFrame: ...
    def reset(self) -> None: ...
```

#### `__init__(threshold_bps)`

Create a new range bar processor.

**Parameters**:

- **threshold_bps**: `int`
  - Range bar threshold in 0.1 basis point units
  - Must be positive (>0)

**Raises**:

- **ValueError**: If `threshold_bps` ≤ 0

**Example**:

```python
from rangebar import RangeBarProcessor

# Create processor with 0.25% threshold
processor = RangeBarProcessor(threshold_bps=250)

# Access threshold
print(processor.threshold_bps)  # 250
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
processor = RangeBarProcessor(threshold_bps=250)

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
processor = RangeBarProcessor(threshold_bps=250)
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
processor = RangeBarProcessor(threshold_bps=250)

# Process first batch
bars1 = processor.process_trades(trades1)

# Reset before processing new batch
processor.reset()

# Process second batch (independent)
bars2 = processor.process_trades(trades2)
```

**Note**: Use `reset()` when processing unrelated datasets. Not needed if creating new processor instance.

---

#### Attribute: `threshold_bps`

**Type**: `int` (read-only)

The threshold value in 0.1 basis point units.

**Example**:

```python
processor = RangeBarProcessor(threshold_bps=250)
print(processor.threshold_bps)  # 250
print(processor.threshold_bps * 0.001)  # 0.25 (percentage)
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
- Invalid threshold_bps (≤0)
- Missing required columns/keys
- Negative price/quantity

**Example**:

```python
# Empty trades
process_trades_to_dataframe([])
# ValueError: Trades list is empty

# Invalid threshold
RangeBarProcessor(threshold_bps=0)
# ValueError: threshold_bps must be positive

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
df = process_trades_to_dataframe(trades, threshold_bps=250)

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
    df = process_trades_to_dataframe(trades, threshold_bps=threshold)
    print(f"Threshold {threshold}: {len(df)} bars")
```

---

### Pattern 3: Incremental Processing

Processing large datasets in chunks:

```python
processor = RangeBarProcessor(threshold_bps=250)

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
    results[symbol] = process_trades_to_dataframe(trades, threshold_bps=250)

# Compare bar counts
for symbol, df in results.items():
    print(f"{symbol}: {len(df)} bars")
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
    threshold_bps: int = 250,
) -> pd.DataFrame: ...
```

IDE support:

- PyCharm: Full autocomplete
- VS Code: Full IntelliSense
- mypy: Type checking passes

---

## Compatibility

### Python Versions

- **Supported**: 3.9, 3.10, 3.11, 3.12
- **Tested**: 3.11 (development), 3.12 (CI)

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

**A**: Start with `threshold_bps=250` (0.25%). This generates ~200 bars/day for BTC/USDT, similar to 1-hour time bars. Adjust based on:

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
    "BTCUSDT": RangeBarProcessor(threshold_bps=250),
    "ETHUSDT": RangeBarProcessor(threshold_bps=250),
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

df = process_trades_to_dataframe(trades, threshold_bps=250)
```

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

**Last Updated**: 2024-11-15
**API Version**: 0.1.0
**License**: MIT
