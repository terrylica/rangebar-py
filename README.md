[//]: # SSoT-OK

# rangebar-py

High-performance range bar construction for quantitative trading, with Python bindings via PyO3/maturin.

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

## Minimum Threshold Enforcement (v12.0+)

**Breaking change in v12.0**: Crypto symbols now enforce a minimum threshold of 1000 dbps (1%) by default. Thresholds below this cannot overcome trading costs sufficiently for crypto assets.

```python
from rangebar import get_range_bars, ThresholdError

# This raises ThresholdError for crypto (below 1000 dbps minimum)
try:
    df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps=250)
except ThresholdError as e:
    print(e)  # Threshold 250 dbps below minimum 1000 dbps for crypto symbol 'BTCUSDT'

# Use valid threshold for crypto (>= 1000 dbps)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps=1000)

# Forex allows lower thresholds (minimum 50 dbps)
df = get_range_bars("EURUSD", "2024-01-01", "2024-01-31", threshold_decimal_bps=50, source="exness")
```

**Override for research/testing**:

```bash
# Lower crypto minimum for research
RANGEBAR_CRYPTO_MIN_THRESHOLD=100 python my_research_script.py

# Override specific symbol
RANGEBAR_MIN_THRESHOLD_BTCUSDT=250 python btc_analysis.py
```

**Default minimums by asset class**:

| Asset Class | Minimum   | Env Var                           |
| ----------- | --------- | --------------------------------- |
| Crypto      | 1000 dbps | `RANGEBAR_CRYPTO_MIN_THRESHOLD`   |
| Forex       | 50 dbps   | `RANGEBAR_FOREX_MIN_THRESHOLD`    |
| Equities    | 100 dbps  | `RANGEBAR_EQUITIES_MIN_THRESHOLD` |
| Unknown     | 1 dbps    | `RANGEBAR_UNKNOWN_MIN_THRESHOLD`  |

## API Reference

### get_range_bars (Date-Bounded)

The entry point for date-bounded range bar generation with automatic data fetching and caching.

```python
from rangebar import get_range_bars

# Basic usage - Binance spot (crypto requires threshold >= 1000 dbps)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", threshold_decimal_bps=1000)

# Using threshold presets (only "macro" and "wide" valid for crypto)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", threshold_decimal_bps="macro")

# Binance USD-M Futures
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um", threshold_decimal_bps=1000)

# With microstructure features (53 total columns)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps=1000, include_microstructure=True)
```

**Parameters**:

- `symbol`: Trading symbol (e.g., "BTCUSDT", "ETHUSDT", "EURUSD")
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `threshold_decimal_bps`: Threshold in decimal basis points or preset name (default: 250, but crypto requires >= 1000)
- `source`: Data source - "binance" or "exness" (default: "binance")
- `market`: Market type - "spot", "futures-um"/"um", "futures-cm"/"cm" (default: "spot")
- `include_microstructure`: Include 53 microstructure feature columns (default: False)
- `use_cache`: Cache tick data locally (default: True)

**Returns**: pandas DataFrame with DatetimeIndex and columns `Open`, `High`, `Low`, `Close`, `Volume`

### get_n_range_bars (Count-Bounded)

Get exactly N range bars - useful for ML training, walk-forward optimization, and consistent backtest comparisons.

```python
from rangebar import get_n_range_bars

# Get last 10,000 bars for ML training
df = get_n_range_bars("BTCUSDT", n_bars=10000, threshold_decimal_bps=1000)
assert len(df) == 10000

# Get 5,000 bars ending at specific date
df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01", threshold_decimal_bps=1000)

# With safety limit (won't fetch more than 30 days of data)
df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30, threshold_decimal_bps=1000)
```

**Parameters**:

- `symbol`: Trading symbol (e.g., "BTCUSDT")
- `n_bars`: Number of bars to retrieve (must be > 0)
- `threshold_decimal_bps`: Threshold in decimal basis points or preset name (default: 250, but crypto requires >= 1000)
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

| Preset       | Value | Percentage    | Crypto | Forex | Use Case         |
| ------------ | ----- | ------------- | ------ | ----- | ---------------- |
| `"micro"`    | 10    | 0.01% (1bps)  | No     | No    | Scalping         |
| `"tight"`    | 50    | 0.05% (5bps)  | No     | Yes   | Day trading      |
| `"standard"` | 100   | 0.1% (10bps)  | No     | Yes   | Swing trading    |
| `"medium"`   | 250   | 0.25% (25bps) | No     | Yes   | Default          |
| `"wide"`     | 500   | 0.5% (50bps)  | No     | Yes   | Position trading |
| `"macro"`    | 1000  | 1% (100bps)   | Yes    | Yes   | Long-term        |

```python
from rangebar import get_range_bars, THRESHOLD_PRESETS

# Using preset string (only "macro" valid for crypto)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps="macro")

# Forex allows all presets
df = get_range_bars("EURUSD", "2024-01-01", "2024-01-31", threshold_decimal_bps="tight", source="exness")

# View all presets
print(THRESHOLD_PRESETS)
# {'micro': 10, 'tight': 50, 'standard': 100, 'medium': 250, 'wide': 500, 'macro': 1000}
```

### Threshold Validation Utilities (v12.0+)

```python
from rangebar import (
    ThresholdError,
    get_min_threshold,
    get_min_threshold_for_symbol,
    resolve_and_validate_threshold,
    clear_threshold_cache,
    detect_asset_class,
    AssetClass,
)

# Get minimum threshold for asset class
crypto_min = get_min_threshold(AssetClass.CRYPTO)  # 1000
forex_min = get_min_threshold(AssetClass.FOREX)    # 50

# Get minimum threshold for specific symbol
btc_min = get_min_threshold_for_symbol("BTCUSDT")  # 1000
eur_min = get_min_threshold_for_symbol("EURUSD")   # 50

# Validate and resolve threshold (raises ThresholdError if invalid)
threshold = resolve_and_validate_threshold("BTCUSDT", "macro")  # 1000

# Detect asset class
assert detect_asset_class("BTCUSDT") == AssetClass.CRYPTO
assert detect_asset_class("EURUSD") == AssetClass.FOREX

# Clear cache after changing env vars at runtime
clear_threshold_cache()
```

### Microstructure Features (v7.0+)

When `include_microstructure=True`, the DataFrame includes 53 feature columns organized into three categories:

**Base Microstructure (15 columns)**:

- `vwap`, `buy_volume`, `sell_volume`, `individual_trade_count`, `agg_record_count`
- `duration_us`, `ofi`, `vwap_close_deviation`, `price_impact`, `kyle_lambda_proxy`
- `trade_intensity`, `volume_per_trade`, `aggression_ratio`, `aggregation_density`, `turnover_imbalance`

**Intra-Bar Features (22 columns, Issue #59)** - Computed from trades WITHIN each bar:

- Epoch analysis: `intra_bull_epoch_density`, `intra_bear_epoch_density`, `intra_bull_excess_gain`, `intra_bear_excess_gain`, `intra_bull_cv`, `intra_bear_cv`
- Price path: `intra_max_drawdown`, `intra_max_runup`
- Volume/trade: `intra_trade_count`, `intra_ofi`, `intra_duration_us`, `intra_intensity`
- Advanced: `intra_vwap_position`, `intra_count_imbalance`, `intra_kyle_lambda`, `intra_burstiness`
- Statistical: `intra_volume_skew`, `intra_volume_kurt`, `intra_kaufman_er`, `intra_garman_klass_vol`, `intra_hurst`, `intra_permutation_entropy`

**Inter-Bar Features (16 columns, Issue #59)** - Computed from lookback window BEFORE each bar:

- `lookback_trade_count`, `lookback_ofi`, `lookback_duration_us`, `lookback_intensity`
- `lookback_vwap_raw`, `lookback_vwap_position`, `lookback_count_imbalance`, `lookback_kyle_lambda`
- `lookback_burstiness`, `lookback_volume_skew`, `lookback_volume_kurt`, `lookback_price_range`
- `lookback_kaufman_er`, `lookback_garman_klass_vol`, `lookback_hurst`, `lookback_permutation_entropy`

```python
from rangebar import get_range_bars, MICROSTRUCTURE_COLUMNS, INTRA_BAR_FEATURE_COLUMNS, INTER_BAR_FEATURE_COLUMNS

df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-07", threshold_decimal_bps=1000, include_microstructure=True)

print(f"Base microstructure: {len(MICROSTRUCTURE_COLUMNS)} columns")
print(f"Intra-bar features: {len(INTRA_BAR_FEATURE_COLUMNS)} columns")
print(f"Inter-bar features: {len(INTER_BAR_FEATURE_COLUMNS)} columns")
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

# Convert to range bars (symbol required for threshold validation)
df = process_trades_to_dataframe(trades, threshold_decimal_bps=1000, symbol="BTCUSDT")

# Without symbol, no asset-class validation (use for custom data)
df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
```

#### process_trades_polars (Recommended for Polars Users)

**2-3x faster** than `process_trades_to_dataframe()` with lazy evaluation and predicate pushdown:

```python
import polars as pl
from rangebar import process_trades_polars

# LazyFrame - predicates pushed to I/O layer (10-100x memory reduction)
lazy_df = pl.scan_parquet("trades/*.parquet").filter(pl.col("timestamp") >= start_ts)
bars = process_trades_polars(lazy_df, threshold_decimal_bps=1000, symbol="BTCUSDT")

# Or with eager DataFrame
df = pl.read_parquet("trades.parquet")
bars = process_trades_polars(df, threshold_decimal_bps=1000, symbol="BTCUSDT")
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
for bars_df in process_trades_chunked(iter(trades), chunk_size=50_000, symbol="BTCUSDT", threshold_decimal_bps=1000):
    process_batch(bars_df)
```

### Streaming API

Real-time range bar generation from Binance WebSocket:

```python
from rangebar import stream_binance_live, StreamingConfig, ReconnectionConfig

# Basic streaming
async for bar in stream_binance_live("BTCUSDT", threshold_decimal_bps=1000):
    print(f"New bar: {bar}")

# With configuration
config = StreamingConfig(
    buffer_size=1000,
    emit_partial_bars=False,
    reconnection=ReconnectionConfig(max_retries=5, base_delay_ms=1000),
)
async for bar in stream_binance_live("BTCUSDT", threshold_decimal_bps=1000, config=config):
    process_bar(bar)
```

## Requirements

**Runtime**: Python >= 3.13, pandas >= 2.0, numpy >= 1.24, polars >= 1.0

**Optional**: backtesting >= 0.3 (for backtesting integration)

**Build**: Rust toolchain, maturin >= 1.7

## Architecture

```
rangebar-py/
├── crates/                    8 Rust crates (core algorithm)
│   └── rangebar-core/         Core processor, microstructure features
├── src/lib.rs                 PyO3 bindings
├── python/rangebar/           Python API layer
│   ├── clickhouse/            ClickHouse cache
│   ├── validation/            Microstructure validation
│   └── threshold.py           Threshold validation (v12.0+)
└── pyproject.toml             Maturin config
```

## Development

```bash
git clone https://github.com/terrylica/rangebar-py.git
cd rangebar-py
pip install maturin
maturin develop --features data-providers
pytest tests/
```

## Migration from v11.x to v12.0

**Breaking changes**:

1. **ThresholdError** raised for crypto symbols with threshold < 1000 dbps
2. Presets "micro", "tight", "standard", "medium" will error for crypto symbols
3. Checkpoints with low thresholds fail to restore

**Migration options**:

```python
# Option 1: Use valid threshold for crypto
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps=1000)

# Option 2: Override minimum via environment variable
import os
os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "250"
from rangebar import clear_threshold_cache
clear_threshold_cache()

# Option 3: Use "macro" preset (1000 dbps)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", threshold_decimal_bps="macro")
```

## Documentation

| Document                                                                                                | Description                  |
| ------------------------------------------------------------------------------------------------------- | ---------------------------- |
| [API Reference](https://github.com/terrylica/rangebar-py/blob/main/docs/api.md)                         | Full API documentation       |
| [Architecture](https://github.com/terrylica/rangebar-py/blob/main/docs/ARCHITECTURE.md)                 | System design                |
| [Performance Guide](https://github.com/terrylica/rangebar-py/blob/main/docs/development/PERFORMANCE.md) | Benchmark methodology        |
| [CLAUDE.md](https://github.com/terrylica/rangebar-py/blob/main/CLAUDE.md)                               | AI assistant project context |

## License

MIT License. See [LICENSE](https://github.com/terrylica/rangebar-py/blob/main/LICENSE).

## Citation

```bibtex
@software{rangebar-py,
  title = {rangebar-py: High-performance range bar construction for quantitative trading},
  author = {Terry Li},
  url = {https://github.com/terrylica/rangebar-py}
}
```
