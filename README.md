# rangebar-py

**Python bindings for [rangebar](https://github.com/terrylica/rangebar) - High-performance range bar construction for cryptocurrency trading**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Performance Dashboard](https://img.shields.io/badge/performance-dashboard-brightgreen.svg)](https://terrylica.github.io/rangebar-py/)

## Status: ✅ Production Ready (v0.3.0)

Core functionality implemented, tested (95%+ coverage), and released to PyPI.

**Latest**: v0.3.0 - Full test suite, CI/CD, automated releases, daily performance monitoring

---

## What is rangebar-py?

Python bindings (via PyO3/maturin) to the high-performance [rangebar](https://github.com/terrylica/rangebar) Rust crate, enabling Python users to leverage temporally-sound range bar construction for backtesting cryptocurrency trading strategies.

### What are Range Bars?

Unlike traditional time-based bars (5-minute, 1-hour), **range bars** close when price moves a fixed percentage from the opening price:

- **Fixed Price Movement**: Each bar represents equal volatility (e.g., 0.25% = 25 basis points)
- **Market-Adaptive**: Bars form faster during volatile periods, slower during consolidation
- **No Lookahead Bias**: Strict temporal integrity (breach tick included in closing bar)
- **Eliminates Time Artifacts**: No arbitrary time intervals (5-min, 15-min, 1-hour)

### Why Range Bars?

Traditional time-based bars create artificial patterns in cryptocurrency markets:

- **Inverse Timeframe Effect**: MA crossover win rates degrade from 40.3% (5-min) → 38.7% (15-min) → 37.2% (1-hour)
- **High Volatility Bias**: Time bars compress information during volatile periods
- **Information Loss**: Equal time ≠ equal market activity

Range bars address these issues by adapting to market activity rather than clock time.

---

## Quick Start

```python
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from rangebar import process_trades_to_dataframe

# Load Binance aggTrades CSV
trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")

# Convert to range bars (25 basis points = 0.25%)
data = process_trades_to_dataframe(trades, threshold_bps=250)

# Define strategy
class RangeBarMA(Strategy):
    fast = 20
    slow = 50

    def init(self):
        self.sma_fast = self.I(SMA, self.data.Close, self.fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()

# Run backtest
bt = Backtest(data, RangeBarMA, cash=10000, commission=0.0002)
stats = bt.run()
print(stats)
bt.plot()
```

---

## Installation

### From PyPI (Recommended)

```bash
pip install rangebar
```

Pre-built wheels available for:

- **Linux**: x86_64 (manylinux)
- **macOS**: ARM64 (Apple Silicon)
- **Python**: 3.10, 3.11, 3.12

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/terrylica/rangebar-py.git
cd rangebar-py

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin develop

# Or build release wheel
maturin build --release
pip install target/wheels/rangebar-*.whl
```

---

## Usage

### Basic Example

```python
from rangebar import process_trades_to_dataframe

# Synthetic trade data
trades = [
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    {"timestamp": 1704067220000, "price": 42000.0, "quantity": 1.8},
]

# Convert to range bars
df = process_trades_to_dataframe(trades, threshold_bps=250)

print(df.head())
#                          Open      High       Low     Close  Volume
# timestamp
# 2024-01-01 00:00:10  42000.0  42105.0  42000.0  42105.0    3.8
```

### Binance CSV Loading

```python
import pandas as pd
from rangebar import process_trades_to_dataframe

# Load Binance aggTrades CSV
trades_df = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")

# Convert to range bars
range_bars = process_trades_to_dataframe(
    trades_df[["timestamp", "price", "quantity"]],
    threshold_bps=250  # 0.25%
)

# Save for later use
range_bars.to_csv("BTCUSDT-range-bars-25bps.csv")
```

### Advanced: Custom Processor

```python
from rangebar import RangeBarProcessor

# Create processor with custom threshold
processor = RangeBarProcessor(threshold_bps=100)  # 0.1% (more sensitive)

# Process trades
bars = processor.process_trades(trades)

# Convert to DataFrame
df = processor.to_dataframe(bars)

# Reset processor state
processor.reset()
```

---

## Features

- ✅ **High Performance**: Rust-powered processing (>1M trades/sec)
- ✅ **Pandas Integration**: Returns DataFrames with DatetimeIndex
- ✅ **backtesting.py Compatible**: OHLCV format ready for backtesting
- ✅ **Type Hints**: Full IDE support with `.pyi` stubs
- ✅ **No Lookahead Bias**: Strict temporal integrity guaranteed
- ✅ **Fixed-Point Arithmetic**: Preserves 8 decimal precision
- ✅ **Zero Upstream Dependencies**: No changes needed to rangebar Rust crate

---

## API Overview

### `process_trades_to_dataframe()`

Convenience function for one-step conversion:

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict], pd.DataFrame],
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Convert trades to range bars DataFrame.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns: timestamp, price, quantity
    threshold_bps : int
        Threshold in 0.1 basis point units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex
    """
```

### `RangeBarProcessor`

Lower-level API for custom workflows:

```python
class RangeBarProcessor:
    def __init__(self, threshold_bps: int) -> None: ...
    def process_trades(self, trades: List[Dict]) -> List[Dict]: ...
    def to_dataframe(self, bars: List[Dict]) -> pd.DataFrame: ...
    def reset(self) -> None: ...
```

---

## Threshold Selection Guide

| threshold_bps | Percentage | Use Case                 | Bars/Day\* |
| ------------- | ---------- | ------------------------ | ---------- |
| 100           | 0.1%       | High-frequency, scalping | ~500       |
| 250           | 0.25%      | **Default**, general use | ~200       |
| 500           | 0.5%       | Swing trading            | ~100       |
| 1000          | 1.0%       | Position trading         | ~50        |

\*Approximate for BTC/USDT

**Recommendation**: Start with `threshold_bps=250` (0.25%) and adjust based on:

- Market volatility (higher volatility → higher threshold)
- Trading timeframe (shorter timeframe → lower threshold)
- Strategy requirements (mean reversion vs trend following)

---

## Examples

See [`examples/`](examples/) directory for complete examples:

- **`basic_usage.py`**: Simple demonstration with synthetic data
- **`binance_csv_example.py`**: Load and convert Binance aggTrades CSV
- **`backtesting_integration.py`**: Complete MA crossover strategy with backtesting.py
- **`validate_output.py`**: OHLCV validation and format checking

Run examples:

```bash
# Basic usage
python examples/basic_usage.py

# Binance CSV (creates sample data)
python examples/binance_csv_example.py

# Or with your own Binance CSV
python examples/binance_csv_example.py path/to/BTCUSDT-aggTrades.csv

# Backtesting integration (requires backtesting.py)
pip install backtesting.py
python examples/backtesting_integration.py

# Validation
python examples/validate_output.py
```

---

## Requirements

### Runtime

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24

### Optional

- backtesting.py ≥ 0.3 (for backtesting integration)

### Build (from source)

- Rust toolchain ≥ 1.70
- maturin ≥ 1.7

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/terrylica/rangebar-py.git
cd rangebar-py

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install maturin pytest pandas mypy black ruff

# Install in development mode
maturin develop
```

### Testing

```bash
# Run all tests (excluding slow tests)
make test

# Run all tests including slow tests
make test-all

# Run with coverage (95%+ Python, 90%+ Rust)
make coverage

# Run benchmarks
make benchmark

# Run specific test file
pytest tests/test_python_api.py -v
```

### Code Quality

```bash
# Run all linters (ruff, mypy, black, clippy)
make lint

# Auto-format code
make format

# Combined check (lint + test)
make check

# Individual tools
mypy python/rangebar/        # Type checking
ruff check python/           # Python linting
black python/ --check        # Format checking
cargo clippy -- -D warnings  # Rust linting
```

### Building

```bash
# Development build
maturin develop

# Release build
maturin build --release

# Build for multiple Python versions
maturin build --release --interpreter python3.10 python3.11 python3.12

# Check wheels
ls target/wheels/
```

---

## Project Structure

```
rangebar-py/
├── src/
│   └── lib.rs                      # PyO3 Rust bindings (5 Rust tests)
├── python/
│   └── rangebar/
│       ├── __init__.py             # Python API
│       └── __init__.pyi            # Type stubs
├── tests/
│   ├── test_rust_bindings.py       # Rust bindings tests (14 tests)
│   ├── test_python_api.py          # Python API tests (19 tests)
│   ├── test_edge_cases.py          # Edge case tests (12 tests)
│   ├── test_real_data.py           # Real Binance data tests (3 tests)
│   ├── test_examples.py            # Example validation tests (4 tests)
│   ├── test_performance.py         # Performance benchmarks (6 tests)
│   └── fixtures/
│       └── BTCUSDT-aggTrades-sample-10k.csv  # Sample real data
├── examples/
│   ├── basic_usage.py              # Simple example
│   ├── binance_csv_example.py      # CSV loading example
│   ├── backtesting_integration.py  # Full backtesting.py example
│   ├── validate_output.py          # OHLCV validation
│   └── README.md                   # Examples documentation
├── docs/
│   ├── decisions/                  # Architecture Decision Records
│   │   ├── 0003-testing-strategy-real-data.md
│   │   ├── 0004-cicd-multiplatform-builds.md
│   │   └── 0005-automated-release-management.md
│   └── plan/                       # Implementation plans
│       ├── 0003-testing-strategy/
│       ├── 0004-cicd-architecture/
│       └── 0005-release-management/
├── .github/
│   └── workflows/
│       ├── ci-test-build.yml       # CI workflow
│       └── release.yml             # Automated release workflow
├── Cargo.toml                      # Rust dependencies
├── pyproject.toml                  # Python packaging + semantic-release
├── Makefile                        # Quality tooling commands
├── .pre-commit-config.yaml         # Pre-commit hooks
├── CLAUDE.md                       # Project memory
├── IMPLEMENTATION_PLAN.md          # Development roadmap
└── README.md                       # This file
```

---

## Architecture

```
rangebar-core (Rust crate on crates.io) [maintained by terrylica]
    ↓ [Cargo dependency: v5.0.0]
rangebar-py (This project) [Python wrapper]
    ├── src/lib.rs (PyO3 bindings)
    ├── python/rangebar/ (Pythonic API)
    └── Type stubs (.pyi)
    ↓ [pip install]
Python users (pandas, backtesting.py)
```

**Key Principle**: The rangebar-core maintainer does **ZERO** work. We import their crate as a dependency and handle all Python integration ourselves.

---

## Documentation

- **User Guide**: This README
- **Examples**: [`examples/README.md`](examples/README.md)
- **API Reference**: [`docs/api.md`](docs/api.md) _(planned)_
- **Project Context**: [`CLAUDE.md`](CLAUDE.md) (for contributors)
- **Implementation Plan**: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## Troubleshooting

### Error: "KeyError: missing 'timestamp'"

**Cause**: Trade dict missing required fields

**Fix**:

```python
# Ensure trades have: timestamp (ms), price, quantity
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

### Warning: "No range bars generated"

**Cause**: Price movement below threshold

**Fix**:

- Lower `threshold_bps` (e.g., 250 → 100)
- Use more volatile data
- Check data covers sufficient time period

---

## Contributing

This project is in active development. Contributions welcome!

**Before Contributing**:

1. Read [`CLAUDE.md`](CLAUDE.md) for project context
2. Review [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for roadmap
3. Check existing issues and pull requests

**Development Workflow**:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes
4. Run tests (`pytest tests/ -v`)
5. Run type checking (`mypy python/rangebar/`)
6. Run linting (`ruff check python/`)
7. Commit with conventional commits (`feat:`, `fix:`, `docs:`, etc.)
8. Push to branch
9. Open Pull Request

---

## Roadmap

- [x] **Phase 1-5**: Core implementation (✅ Completed)
  - [x] Project scaffolding
  - [x] Rust PyO3 bindings
  - [x] Python API layer
  - [x] backtesting.py integration
  - [x] Examples and validation

- [x] **Phase 6**: Documentation (✅ Completed)
  - [x] README.md
  - [x] Project memory (CLAUDE.md)
  - [x] Implementation plan

- [x] **Phase 7**: Testing & Quality (✅ Completed - v0.1.0)
  - [x] 95%+ Python test coverage (21 new tests)
  - [x] 90%+ Rust test coverage
  - [x] Performance benchmarks (>1M trades/sec, <100MB)
  - [x] Linting and type checking (ruff, mypy, black, clippy)
  - [x] Quality tooling (Makefile)

- [x] **Phase 8**: Distribution & CI/CD (✅ Completed - v0.1.0)
  - [x] Multi-platform wheels (Linux x86_64, macOS ARM64)
  - [x] GitHub Actions CI/CD
  - [x] Automated semantic releases
  - [x] PyPI publishing with Trusted Publisher

- [ ] **Future Enhancements** (v0.2.0+)
  - [ ] Streaming API (incremental processing)
  - [ ] Multi-symbol batch processing
  - [ ] Parquet output support
  - [ ] Polars integration
  - [ ] Visualization tools

---

## Performance

Benchmarks (Apple M1, validated in Phase 7):

- **Processing**: >1M trades/sec (Rust backend)
- **Conversion**: 10k trades → DataFrame in <10ms
- **Memory**: <100MB for 1M trades

Run benchmarks yourself:

```bash
make benchmark
# or: pytest tests/test_performance.py -v --benchmark-only
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

This matches the license of the upstream [rangebar](https://github.com/terrylica/rangebar) Rust crate.

---

## Credits

- **Python bindings**: This project (rangebar-py)
- **Rust crate**: [terrylica/rangebar](https://github.com/terrylica/rangebar) (rangebar-core v5.0.0)
- **Target framework**: [backtesting.py](https://kernc.github.io/backtesting.py/)
- **Build system**: [PyO3](https://pyo3.rs/) + [maturin](https://www.maturin.rs/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/terrylica/rangebar-py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/terrylica/rangebar-py/discussions)
- **Upstream (Rust)**: [rangebar Issues](https://github.com/terrylica/rangebar/issues)

---

## Citation

If you use rangebar-py in your research, please cite:

```bibtex
@software{rangebar-py,
  title = {rangebar-py: Python bindings for high-performance range bar construction},
  author = {Terry Li},
  year = {2024},
  url = {https://github.com/terrylica/rangebar-py}
}
```

And also cite the upstream Rust crate:

```bibtex
@software{rangebar,
  title = {rangebar: Non-lookahead range bar construction},
  author = {Terry Li},
  year = {2024},
  url = {https://github.com/terrylica/rangebar}
}
```

---

## Acknowledgments

Special thanks to:

- The PyO3 team for excellent Rust-Python bindings
- The maturin maintainers for seamless packaging
- The backtesting.py project for inspiration and integration target
- The cryptocurrency trading community for feedback and use cases

---

**Ready to get started?** See [Quick Start](#quick-start) above or run the [examples](examples/).

**Questions?** Open an issue or start a discussion!
