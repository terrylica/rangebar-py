# rangebar-py

Python bindings for the [rangebar](https://github.com/terrylica/rangebar-py/tree/main/crates) Rust crates via PyO3/maturin.

[![PyPI](https://img.shields.io/pypi/v/rangebar.svg)](https://pypi.org/project/rangebar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/terrylica/rangebar-py/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/rangebar.svg)](https://pypi.org/project/rangebar/)

## Links

| Resource              | URL                                                         |
| --------------------- | ----------------------------------------------------------- |
| PyPI                  | <https://pypi.org/project/rangebar/>                        |
| Repository            | <https://github.com/terrylica/rangebar-py>                  |
| Performance Dashboard | <https://terrylica.github.io/rangebar-py/>                  |
| Rust Crates           | <https://github.com/terrylica/rangebar-py/tree/main/crates> |
| Issues                | <https://github.com/terrylica/rangebar-py/issues>           |

## Installation

```bash
pip install rangebar
```

Pre-built wheels: Linux (x86_64), macOS (ARM64), Python 3.13

Source build requires Rust toolchain and maturin.

## Overview

Converts trade tick data into range bars for backtesting. Range bars close when price moves a fixed percentage from the opening price, adapting to market volatility rather than fixed time intervals.

**Output format**: pandas DataFrame with DatetimeIndex and OHLCV columns, compatible with [backtesting.py](https://github.com/kernc/backtesting.py).

## Quick Start

```python
from rangebar import get_range_bars

# Fetch data and generate range bars in one call
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Use with backtesting.py
from backtesting import Backtest, Strategy
bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
stats = bt.run()
```

## API Reference

### get_range_bars (Date-Bounded)

The entry point for date-bounded range bar generation with automatic data fetching and caching.

```python
from rangebar import get_range_bars

# Basic usage - Binance spot
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Using threshold presets
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", threshold_decimal_bps="tight")

# Binance USD-M Futures
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

# With microstructure data (vwap, buy_volume, sell_volume)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)
```

**Parameters**:

- `symbol`: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `threshold_decimal_bps`: Threshold in decimal basis points or preset name (default: 250 = 25bps = 0.25%)
- `source`: Data source - "binance" or "exness" (default: "binance")
- `market`: Market type - "spot", "futures-um"/"um", "futures-cm"/"cm" (default: "spot")
- `include_microstructure`: Include vwap, buy_volume, sell_volume columns (default: False)
- `use_cache`: Cache tick data locally (default: True)

**Returns**: pandas DataFrame with DatetimeIndex and columns `Open`, `High`, `Low`, `Close`, `Volume`

### get_n_range_bars (Count-Bounded)

Get exactly N range bars - useful for ML training, walk-forward optimization, and consistent backtest comparisons.

```python
from rangebar import get_n_range_bars

# Get last 10,000 bars for ML training
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# Get 5,000 bars ending at specific date
df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

# With safety limit (won't fetch more than 30 days of data)
df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)
```

**Parameters**:

- `symbol`: Trading symbol (e.g., "BTCUSDT")
- `n_bars`: Number of bars to retrieve (must be > 0)
- `threshold_decimal_bps`: Threshold in decimal basis points or preset name (default: 250)
- `end_date`: End date (YYYY-MM-DD) or None for most recent data
- `source`: Data source - "binance" or "exness" (default: "binance")
- `market`: Market type (default: "spot")
- `use_cache`: Use ClickHouse cache (default: True)
- `fetch_if_missing`: Fetch data if cache doesn't have enough bars (default: True)
- `max_lookback_days`: Safety limit for data fetching (default: 90)
- `warn_if_fewer`: Emit warning if returning fewer bars than requested (default: True)

**Returns**: pandas DataFrame with exactly `n_bars` rows (or fewer if insufficient data), sorted chronologically

**Cache behavior**:

- **Fast path**: If ClickHouse cache has >= n_bars, returns immediately (~50ms)
- **Slow path**: Fetches additional data, computes bars, stores in cache

### Threshold Presets

Use string presets for common threshold values:

| Preset       | Value | Percentage    | Use Case         |
| ------------ | ----- | ------------- | ---------------- |
| `"micro"`    | 10    | 0.01% (1bps)  | Scalping         |
| `"tight"`    | 50    | 0.05% (5bps)  | Day trading      |
| `"standard"` | 100   | 0.1% (10bps)  | Swing trading    |
| `"medium"`   | 250   | 0.25% (25bps) | Default          |
| `"wide"`     | 500   | 0.5% (50bps)  | Position trading |
| `"macro"`    | 1000  | 1% (100bps)   | Long-term        |

```python
from rangebar import get_range_bars, THRESHOLD_PRESETS

# Using preset string
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps="tight")

# Or numeric value
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps=50)

# View all presets
print(THRESHOLD_PRESETS)
# {'micro': 10, 'tight': 50, 'standard': 100, 'medium': 250, 'wide': 500, 'macro': 1000}
```

### Configuration Constants

```python
from rangebar import TIER1_SYMBOLS, THRESHOLD_PRESETS, THRESHOLD_DECIMAL_MIN, THRESHOLD_DECIMAL_MAX

# 18 high-liquidity symbols available on all Binance markets
print(TIER1_SYMBOLS)
# ('AAVE', 'ADA', 'AVAX', 'BCH', 'BNB', 'BTC', 'DOGE', 'ETH', 'FIL',
#  'LINK', 'LTC', 'NEAR', 'SOL', 'SUI', 'UNI', 'WIF', 'WLD', 'XRP')

# Valid threshold range
print(f"Min: {THRESHOLD_DECIMAL_MIN}, Max: {THRESHOLD_DECIMAL_MAX}")
# Min: 1, Max: 100000
```

### Advanced APIs

For users with existing tick data (not fetching from Binance):

#### process_trades_to_dataframe

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Load your own trade data
trades = pd.read_csv("BTCUSDT-aggTrades.csv")

# Convert to range bars
df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
```

#### process_trades_polars (Recommended for Polars Users)

**2-3x faster** than `process_trades_to_dataframe()` with lazy evaluation and predicate pushdown:

```python
import polars as pl
from rangebar import process_trades_polars

# LazyFrame - predicates pushed to I/O layer (10-100x memory reduction)
lazy_df = pl.scan_parquet("trades/*.parquet").filter(pl.col("timestamp") >= start_ts)
bars = process_trades_polars(lazy_df, threshold_decimal_bps=250)

# Or with eager DataFrame
df = pl.read_parquet("trades.parquet")
bars = process_trades_polars(df)
```

**Performance benefits**:

- Predicate pushdown: filters applied at Parquet read, not after
- Minimal conversion: only required columns extracted
- Chunked processing: 100K records per batch
- Memory efficient: avoids materializing full dataset

#### process_trades_chunked

Memory-safe processing for large datasets:

```python
from rangebar import process_trades_chunked

# Process 10M+ trades without OOM
for bars_df in process_trades_chunked(iter(trades), chunk_size=50_000):
    process_batch(bars_df)
```

## Requirements

**Runtime**: Python >= 3.13, pandas >= 2.0, numpy >= 1.24, polars >= 1.0

**Optional**: backtesting >= 0.3 (for backtesting integration)

**Build**: Rust toolchain, maturin >= 1.7

## Architecture

```
rangebar (Rust crate)
    ↓ Cargo dependency
rangebar-py (this package)
    ├── src/lib.rs (PyO3 bindings)
    └── python/rangebar/ (Python API)
    ↓ pip install
Python applications
```

## Development

```bash
git clone https://github.com/terrylica/rangebar-py.git
cd rangebar-py
pip install maturin
maturin develop --features data-providers
pytest tests/
```

## Documentation

| Document                                                                                                | Description           |
| ------------------------------------------------------------------------------------------------------- | --------------------- |
| [Examples](https://github.com/terrylica/rangebar-py/tree/main/examples)                                 | Usage examples        |
| [CLAUDE.md](https://github.com/terrylica/rangebar-py/blob/main/CLAUDE.md)                               | Project context       |
| [Architecture](https://github.com/terrylica/rangebar-py/blob/main/docs/ARCHITECTURE.md)                 | System design         |
| [Performance Guide](https://github.com/terrylica/rangebar-py/blob/main/docs/development/PERFORMANCE.md) | Benchmark methodology |

## License

MIT License. See [LICENSE](https://github.com/terrylica/rangebar-py/blob/main/LICENSE).

## Citation

```bibtex
@software{rangebar-py,
  title = {rangebar-py: Python bindings for range bar construction},
  author = {Terry Li},
  url = {https://github.com/terrylica/rangebar-py}
}
```
