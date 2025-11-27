# rangebar-py

Python bindings for the [rangebar](https://github.com/terrylica/rangebar) Rust crate via PyO3/maturin.

[![PyPI](https://img.shields.io/pypi/v/rangebar.svg)](https://pypi.org/project/rangebar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/terrylica/rangebar-py/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/rangebar.svg)](https://pypi.org/project/rangebar/)

## Links

| Resource | URL |
|----------|-----|
| PyPI | https://pypi.org/project/rangebar/ |
| Repository | https://github.com/terrylica/rangebar-py |
| Performance Dashboard | https://terrylica.github.io/rangebar-py/ |
| Upstream Rust Crate | https://github.com/terrylica/rangebar |
| Issues | https://github.com/terrylica/rangebar-py/issues |

## Installation

```bash
pip install rangebar
```

Pre-built wheels: Linux (x86_64), macOS (ARM64), Python 3.10+

Source build requires Rust toolchain and maturin.

## Overview

Converts trade tick data into range bars for backtesting. Range bars close when price moves a fixed percentage from the opening price, adapting to market volatility rather than fixed time intervals.

**Output format**: pandas DataFrame with DatetimeIndex and OHLCV columns, compatible with [backtesting.py](https://github.com/kernc/backtesting.py).

## Quick Start

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Load trade data (Binance aggTrades format)
trades = pd.read_csv("BTCUSDT-aggTrades.csv")

# Convert to range bars
# threshold_bps: price movement threshold in 0.1 basis point units
# 250 = 25 basis points = 0.25%
data = process_trades_to_dataframe(trades, threshold_bps=250)

# Output: DataFrame with Open, High, Low, Close, Volume columns
print(data.head())
```

## API Reference

### process_trades_to_dataframe

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict], pd.DataFrame],
    threshold_bps: int = 250,
) -> pd.DataFrame
```

**Parameters**:
- `trades`: Trade data with columns `timestamp` (ms), `price`, `quantity`
- `threshold_bps`: Price movement threshold in 0.1 basis point units (default: 250 = 0.25%)

**Returns**: pandas DataFrame with DatetimeIndex and columns `Open`, `High`, `Low`, `Close`, `Volume`

### RangeBarProcessor

```python
class RangeBarProcessor:
    def __init__(self, threshold_bps: int) -> None
    def process_trades(self, trades: List[Dict]) -> List[Dict]
    def to_dataframe(self, bars: List[Dict]) -> pd.DataFrame
    def reset(self) -> None
```

Lower-level API for custom workflows and stateful processing.

## Threshold Reference

| threshold_bps | Percentage | Description |
|---------------|------------|-------------|
| 100 | 0.10% | Higher sensitivity |
| 250 | 0.25% | Default |
| 500 | 0.50% | Lower sensitivity |
| 1000 | 1.00% | Lowest sensitivity |

## Requirements

**Runtime**: Python >= 3.10, pandas >= 2.0, numpy >= 1.24

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
maturin develop
pytest tests/
```

## Documentation

| Document | Description |
|----------|-------------|
| [Examples](https://github.com/terrylica/rangebar-py/tree/main/examples) | Usage examples |
| [CLAUDE.md](https://github.com/terrylica/rangebar-py/blob/main/CLAUDE.md) | Project context |
| [Architecture Decisions](https://github.com/terrylica/rangebar-py/tree/main/docs/decisions) | ADRs |
| [Performance Guide](https://github.com/terrylica/rangebar-py/blob/main/docs/PERFORMANCE.md) | Benchmark methodology |

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
