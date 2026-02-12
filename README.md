[//]: # SSoT-OK

# rangebar-py

High-performance range bar construction for quantitative trading, with Python bindings via PyO3/maturin.

[![PyPI](https://img.shields.io/pypi/v/rangebar.svg)](https://pypi.org/project/rangebar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/terrylica/rangebar-py/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/rangebar.svg)](https://pypi.org/project/rangebar/)

| Resource              | URL                                               |
| --------------------- | ------------------------------------------------- |
| PyPI                  | <https://pypi.org/project/rangebar/>              |
| Repository            | <https://github.com/terrylica/rangebar-py>        |
| Performance Dashboard | <https://terrylica.github.io/rangebar-py/>        |
| API Reference         | [docs/api/INDEX.md](docs/api/INDEX.md)            |
| Issues                | <https://github.com/terrylica/rangebar-py/issues> |

## Installation

```bash
pip install rangebar
```

Pre-built wheels: Linux (x86_64), macOS (ARM64), Python 3.13. Source build requires Rust toolchain and maturin.

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

Output: pandas DataFrame with DatetimeIndex and OHLCV columns, compatible with [backtesting.py](https://github.com/kernc/backtesting.py).

## API Overview

| Function                     | Use Case                        |
| ---------------------------- | ------------------------------- |
| `get_range_bars()`           | Date-bounded, auto-fetch        |
| `get_n_range_bars()`         | Exact N bars (ML training)      |
| `process_trades_polars()`    | Polars DataFrames (2-3x faster) |
| `process_trades_chunked()`   | Large datasets (>10M trades)    |
| `populate_cache_resumable()` | Long ranges (>30 days)          |
| `run_sidecar()`              | Real-time streaming sidecar     |

```python
# Count-bounded (ML training)
from rangebar import get_n_range_bars
df = get_n_range_bars("BTCUSDT", n_bars=10000)

# Polars (2-3x faster)
import polars as pl
from rangebar import process_trades_polars
bars = process_trades_polars(pl.scan_parquet("trades.parquet"), threshold_decimal_bps=250)

# With microstructure features (57 columns: OFI, Kyle lambda, Hurst, etc.)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", include_microstructure=True)

# Real-time streaming sidecar
from rangebar import run_sidecar, SidecarConfig
config = SidecarConfig(symbol="BTCUSDT", threshold_decimal_bps=250)
run_sidecar(config)
```

## Designed for Claude Code

This repository uses a [CLAUDE.md](CLAUDE.md) network that provides comprehensive project context for AI-assisted development via Anthropic's [Claude Code](https://code.claude.com/) CLI.

```bash
npm install -g @anthropic-ai/claude-code
cd rangebar-py
claude
```

Claude Code reads the CLAUDE.md files automatically and understands the full architecture, API, build system, and development workflow.

## Development

```bash
git clone https://github.com/terrylica/rangebar-py.git
cd rangebar-py
mise install          # Setup tools (Rust, Python, zig)
mise run build        # maturin develop
mise run test         # Rust tests
mise run test-py      # Python tests
```

## Requirements

**Runtime**: Python >= 3.13, pandas >= 2.0, numpy >= 1.24, polars >= 1.0

**Build**: Rust toolchain, maturin >= 1.7

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{rangebar-py,
  title = {rangebar-py: High-performance range bar construction for quantitative trading},
  author = {Terry Li},
  url = {https://github.com/terrylica/rangebar-py}
}
```
