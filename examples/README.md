# rangebar-py Examples

This directory contains comprehensive examples demonstrating how to use rangebar-py for cryptocurrency trading backtesting.

## Quick Start

```bash
# Install rangebar-py
pip install rangebar  # (or use uv pip install)

# Run basic example
python examples/basic_usage.py

# Run with Binance CSV (creates sample data if no file provided)
python examples/binance_csv_example.py

# Run backtesting integration (requires backtesting.py)
pip install backtesting.py
python examples/backtesting_integration.py

# Validate output format
python examples/validate_output.py
```

## Examples Overview

### 1. basic_usage.py

**Purpose**: Demonstrates the most straightforward use case

**What it does**:

- Creates synthetic trade data (100 trades)
- Converts to range bars using `process_trades_to_dataframe()`
- Displays results and statistics
- Validates OHLCV invariants

**Run**:

```bash
python examples/basic_usage.py
```

**Output**:

- Console output with range bar statistics
- OHLCV invariant checks
- Temporal distribution analysis

**Use this when**: You want to understand the basic API and see range bars in action.

---

### 2. binance_csv_example.py

**Purpose**: Load and convert Binance aggTrades CSV files

**What it does**:

- Loads Binance aggTrades CSV format
- Validates required columns
- Converts to range bars
- Saves output CSV

**Run**:

```bash
# With your own Binance CSV
python examples/binance_csv_example.py path/to/BTCUSDT-aggTrades-2024-01.csv

# Create and use sample data
python examples/binance_csv_example.py
```

**Binance CSV Format**:

```
agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,is_buyer_maker,is_best_match
1000000,42000.50,1.234,2000000,2000009,1704067200000,true,true
```

**Download Binance Data**:

- https://data.binance.vision/?prefix=data/spot/monthly/aggTrades/
- https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades/

**Output**:

- `range_bars_output.csv` - OHLCV data ready for backtesting

**Use this when**: You have real Binance historical data to convert.

---

### 3. backtesting_integration.py

**Purpose**: Complete backtesting.py integration example

**What it does**:

- Generates realistic synthetic trade data (10,000 trades)
- Converts to range bars
- Defines MA crossover strategy
- Runs backtest with backtesting.py
- Displays performance metrics
- Opens interactive plot in browser

**Run**:

```bash
# Install backtesting.py first
pip install backtesting.py

# Run example
python examples/backtesting_integration.py
```

**Strategy**: Moving Average Crossover on Range Bars

- **Buy**: When fast MA (20) crosses above slow MA (50)
- **Sell**: When fast MA crosses below slow MA

**Output**:

- Backtest statistics (return, Sharpe ratio, drawdown, etc.)
- `backtest_stats.csv` - Saved statistics
- Interactive plot (opens in browser)

**Key Metrics Displayed**:

- Total Return
- Buy & Hold Return
- Sharpe Ratio
- Max Drawdown
- Number of Trades
- Win Rate

**Use this when**: You want to see complete backtesting.py integration.

---

### 4. validate_output.py

**Purpose**: Validate OHLCV format for backtesting.py compatibility

**What it does**:

- Runs comprehensive validation checks
- Tests with valid data
- Tests with intentionally invalid data
- Reports errors and warnings

**Validation Checks**:

1. ✅ DatetimeIndex
2. ✅ Column names: `[Open, High, Low, Close, Volume]`
3. ✅ Numeric data types
4. ✅ No NaN values
5. ✅ OHLC invariants (`High >= max(Open, Close)`, etc.)
6. ✅ Chronological ordering
7. ✅ Positive volume
8. ✅ Precision preservation

**Run**:

```bash
python examples/validate_output.py
```

**Output**:

- Validation results (pass/fail)
- Error details (if any)
- Warning messages
- DataFrame summary

**Use this when**: You want to verify output format correctness.

---

## Common Patterns

### Pattern 1: Basic Conversion

```python
from rangebar import process_trades_to_dataframe

trades = [
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
]

df = process_trades_to_dataframe(trades, threshold_bps=250)
# Returns: DataFrame with columns [Open, High, Low, Close, Volume]
```

### Pattern 2: DataFrame Input

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Load CSV
trades_df = pd.read_csv("trades.csv")

# Convert to range bars
df = process_trades_to_dataframe(
    trades_df[["timestamp", "price", "quantity"]],
    threshold_bps=250
)
```

### Pattern 3: Custom Processor

```python
from rangebar import RangeBarProcessor

processor = RangeBarProcessor(threshold_bps=100)  # 0.1%

# Process trades
bars = processor.process_trades(trades)

# Convert to DataFrame
df = processor.to_dataframe(bars)
```

## Threshold Selection Guide

| threshold_bps | Percentage | Use Case                     |
| ------------- | ---------- | ---------------------------- |
| 100           | 0.1%       | High-frequency, scalping     |
| 250           | 0.25%      | **Default**, general purpose |
| 500           | 0.5%       | Swing trading                |
| 1000          | 1.0%       | Position trading             |

**Recommendation**: Start with `threshold_bps=250` (0.25%) and adjust based on:

- Market volatility
- Trading timeframe
- Strategy requirements

## Troubleshooting

### Error: "KeyError: missing 'timestamp'"

**Cause**: Trade dict missing required fields

**Fix**:

```python
# Ensure trades have: timestamp, price, quantity (or volume)
trades = [
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    # ✅ All required fields present
]
```

### Error: "RuntimeError: Processing failed...not sorted"

**Cause**: Trades not in chronological order

**Fix**:

```python
# Sort by timestamp before processing
trades_df = trades_df.sort_values("timestamp")
df = process_trades_to_dataframe(trades_df, threshold_bps=250)
```

### Error: "DataFrame missing required columns"

**Cause**: DataFrame missing timestamp/price/quantity columns

**Fix**:

```python
# Rename columns if needed
trades_df = trades_df.rename(columns={
    "time": "timestamp",  # or convert datetime to ms
    "qty": "quantity",    # or use 'volume'
})

df = process_trades_to_dataframe(trades_df, threshold_bps=250)
```

### Warning: "Index must be DatetimeIndex"

**Cause**: Using process_trades() without to_dataframe()

**Fix**:

```python
# Option 1: Use convenience function (recommended)
df = process_trades_to_dataframe(trades, threshold_bps=250)

# Option 2: Manual conversion
processor = RangeBarProcessor(threshold_bps=250)
bars = processor.process_trades(trades)
df = processor.to_dataframe(bars)  # ✅ Converts to DatetimeIndex
```

## Performance Tips

1. **Pre-sort data**: Sort trades by timestamp before processing
2. **Batch processing**: Process large datasets in chunks if memory-limited
3. **Threshold selection**: Larger thresholds = fewer bars = faster processing
4. **CSV optimization**: Use `pd.read_csv(usecols=[...])` to load only needed columns

## Next Steps

1. **Try with real data**: Download Binance CSV and run `binance_csv_example.py`
2. **Experiment with thresholds**: Test different `threshold_bps` values
3. **Compare to time bars**: Run same strategy on time-based bars vs range bars
4. **Customize strategies**: Modify `backtesting_integration.py` with your own strategy

## Additional Resources

- **Main README**: See `../README.md` for installation and overview
- **API Documentation**: See `../docs/api.md` for detailed API reference
- **rangebar-core**: https://github.com/terrylica/rangebar (upstream Rust crate)
- **backtesting.py**: https://kernc.github.io/backtesting.py/

## Support

- **Issues**: https://github.com/terrylica/rangebar-py/issues
- **Discussions**: https://github.com/terrylica/rangebar-py/discussions
- **Documentation**: https://github.com/terrylica/rangebar-py#readme
