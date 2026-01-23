# rangebar-py API Reference

<!-- Version controlled by semantic-release via pyproject.toml -->

**Last Updated**: 2026-01-08

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
from rangebar import get_range_bars, get_n_range_bars, process_trades_to_dataframe

# Date-bounded API - fetch and convert with date range
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded API - get exactly N bars (ML training, walk-forward)
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# With preset threshold
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold_decimal_bps="tight")

# Using existing trade data
df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
```

---

## Configuration Constants

```python
from rangebar import TIER1_SYMBOLS, THRESHOLD_PRESETS, THRESHOLD_DECIMAL_MIN, THRESHOLD_DECIMAL_MAX

# 18 high-liquidity symbols available on all Binance markets
TIER1_SYMBOLS  # ('AAVE', 'ADA', 'AVAX', 'BCH', 'BNB', 'BTC', ...)

# Named threshold presets (in decimal basis points)
THRESHOLD_PRESETS  # {'micro': 10, 'tight': 50, 'standard': 100, 'medium': 250, 'wide': 500, 'macro': 1000}

# Valid threshold range
THRESHOLD_DECIMAL_MIN  # 1 (0.001%)
THRESHOLD_DECIMAL_MAX  # 100000 (100%)
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
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
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
- **source**: Data source - `"binance"` or `"exness"` (default: `"binance"`)
- **market**: Market type - `"spot"`, `"futures-um"`, `"futures-cm"` (default: `"spot"`)
- **include_microstructure**: Include vwap, buy_volume, sell_volume (default: False)
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
  - Range bar threshold in decimal basis point units
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

**Last Updated**: 2026-01-08

<!-- API Version: See pyproject.toml (SSoT controlled by semantic-release) -->

**License**: MIT
