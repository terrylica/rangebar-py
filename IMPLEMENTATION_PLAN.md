# Implementation Plan - rangebar-py

**Version**: 1.0.0
**Status**: Planning Phase
**Last Updated**: 2025-10-06

---

## Executive Summary

This document provides a step-by-step implementation roadmap for `rangebar-py`, a Python package providing PyO3 bindings to the rangebar Rust crate. The plan is structured in phases with clear deliverables and success criteria.

**Total Estimated Time**: 2-3 days for MVP, 1 week for production-ready release.

---

## Phase 1: Project Scaffolding (30 minutes)

### Objectives

- Create basic project structure
- Initialize Git repository
- Set up build configuration

### Tasks

#### 1.1: Create Directory Structure

```bash
cd ~/eon/rangebar-py

mkdir -p src
mkdir -p python/rangebar
mkdir -p tests
mkdir -p examples
mkdir -p docs
```

#### 1.2: Initialize Git Repository

```bash
git init
echo "target/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo "*.so" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".venv/" >> .gitignore
git add .
git commit -m "chore: initialize rangebar-py project"
```

#### 1.3: Create `Cargo.toml`

See detailed configuration in Phase 2.

#### 1.4: Create `pyproject.toml`

See detailed configuration in Phase 2.

### Success Criteria

- [x] Directory structure created
- [x] Git repository initialized
- [x] .gitignore configured

---

## Phase 2: Build Configuration (1 hour)

### Objectives

- Configure Rust build with PyO3
- Configure Python packaging with maturin
- Verify build system works

### Tasks

#### 2.1: Create `Cargo.toml`

```toml
[package]
name = "rangebar-py"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
description = "Python bindings for rangebar: Non-lookahead range bar construction"
repository = "https://github.com/your-username/rangebar-py"
readme = "README.md"

[lib]
name = "rangebar_core"
crate-type = ["cdylib"]

[dependencies]
# Import rangebar from crates.io (ZERO changes to upstream)
rangebar-core = "5.0"

# PyO3 for Python bindings
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }

# Utilities
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[build-dependencies]
pyo3-build-config = "0.22"
```

**Note**: Check current versions:

```bash
# Check rangebar-core version on crates.io
cargo search rangebar-core --limit 1

# Update version in Cargo.toml if needed
```

#### 2.2: Create `pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "rangebar"
version = "0.1.0"
description = "Python bindings for rangebar: Non-lookahead range bar construction for cryptocurrency trading"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
keywords = ["trading", "cryptocurrency", "range-bars", "backtesting", "technical-analysis"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Topic :: Office/Business :: Financial :: Investment",
]

dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
backtesting = ["backtesting.py>=0.3"]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "mypy>=1.0",
    "black>=23.0",
    "ruff>=0.1",
]

[project.urls]
Repository = "https://github.com/your-username/rangebar-py"
Documentation = "https://github.com/your-username/rangebar-py#readme"
"Upstream (Rust)" = "https://github.com/terrylica/rangebar"

[tool.maturin]
python-source = "python"
module-name = "rangebar._core"
features = ["pyo3/extension-module"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]

[tool.ruff]
line-length = 100
target-version = "py39"
```

#### 2.3: Create Minimal `src/lib.rs`

```rust
use pyo3::prelude::*;

/// A minimal Python module to verify build works
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

#### 2.4: Create Minimal `python/rangebar/__init__.py`

```python
"""rangebar: Python bindings for range bar construction."""

from ._core import __version__

__all__ = ["__version__"]
```

#### 2.5: Verify Build

```bash
# Install maturin
pip install maturin

# Build (development mode)
maturin develop

# Test import
python -c "import rangebar; print(rangebar.__version__)"
```

### Success Criteria

- [x] `maturin develop` succeeds
- [x] `import rangebar` works
- [x] `rangebar.__version__` returns "0.1.0"

### Troubleshooting

**Issue**: `rangebar-core` not found on crates.io
**Solution**: Verify crate name with `cargo search rangebar`

**Issue**: PyO3 compilation errors
**Solution**: Check Rust toolchain version: `rustc --version` (need >=1.70)

---

## Phase 3: Core Rust Bindings (4 hours)

### Objectives

- Implement PyO3 wrapper for `RangeBarProcessor`
- Convert Rust types to Python types
- Handle errors properly

### Tasks

#### 3.1: Implement `PyRangeBarProcessor` in `src/lib.rs`

**File**: `src/lib.rs`

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyKeyError};
use rangebar_core::{RangeBarProcessor, AggTrade, RangeBar};
use std::collections::HashMap;

#[pyclass]
struct PyRangeBarProcessor {
    processor: RangeBarProcessor,
    threshold_bps: u32,
}

#[pymethods]
impl PyRangeBarProcessor {
    #[new]
    fn new(threshold_bps: u32) -> PyResult<Self> {
        let processor = RangeBarProcessor::new(threshold_bps)
            .map_err(|e| PyValueError::new_err(
                format!("Failed to create processor: {}", e)
            ))?;

        Ok(Self { processor, threshold_bps })
    }

    fn process_trades(&mut self, py: Python, trades: Vec<HashMap<String, PyObject>>)
        -> PyResult<Vec<PyObject>>
    {
        // TODO: Implement conversion and processing
        todo!("Implement in Phase 3.2")
    }

    fn reset(&mut self) -> PyResult<()> {
        self.processor.reset()
            .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {}", e)))
    }

    #[getter]
    fn threshold_bps(&self) -> u32 {
        self.threshold_bps
    }
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRangeBarProcessor>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

#### 3.2: Implement Trade Conversion

**Challenge**: Convert Python dicts to Rust `AggTrade` structs.

**Steps**:

1. Extract fields from Python dict
2. Validate required fields exist
3. Convert to Rust types
4. Create `AggTrade` struct

**Code** (add to `process_trades` method):

```rust
let mut agg_trades = Vec::new();
for (idx, trade_dict) in trades.iter().enumerate() {
    // Extract timestamp
    let timestamp_ms: i64 = trade_dict.get("timestamp")
        .ok_or_else(|| PyKeyError::new_err(format!("Trade {}: missing 'timestamp'", idx)))?
        .extract(py)?;

    // Extract price
    let price: f64 = trade_dict.get("price")
        .ok_or_else(|| PyKeyError::new_err(format!("Trade {}: missing 'price'", idx)))?
        .extract(py)?;

    // Extract quantity
    let quantity: f64 = trade_dict.get("quantity")
        .ok_or_else(|| PyKeyError::new_err(format!("Trade {}: missing 'quantity'", idx)))?
        .extract(py)?;

    // Create AggTrade (adjust based on actual struct definition)
    agg_trades.push(AggTrade {
        timestamp_ms,
        price: rangebar_core::FixedPoint::from_f64(price),
        quantity: rangebar_core::FixedPoint::from_f64(quantity),
        // Add other fields based on rangebar_core::AggTrade definition
    });
}
```

**Note**: Check actual `AggTrade` struct definition:

```bash
# View rangebar-core source
cargo doc --open
# Or check GitHub: https://github.com/terrylica/rangebar
```

#### 3.3: Implement Bar Conversion

**Challenge**: Convert Rust `RangeBar` to Python dict.

**Helper Function**:

```rust
fn rangebar_to_dict(py: Python, bar: &RangeBar) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    // Convert timestamp (milliseconds → RFC3339 string)
    let timestamp = chrono::DateTime::from_timestamp_millis(bar.timestamp_ms)
        .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?
        .to_rfc3339();

    dict.set_item("timestamp", timestamp)?;
    dict.set_item("open", bar.open.to_f64())?;
    dict.set_item("high", bar.high.to_f64())?;
    dict.set_item("low", bar.low.to_f64())?;
    dict.set_item("close", bar.close.to_f64())?;
    dict.set_item("volume", bar.volume.to_f64())?;

    Ok(dict.into())
}
```

**Complete `process_trades`**:

```rust
fn process_trades(&mut self, py: Python, trades: Vec<HashMap<String, PyObject>>)
    -> PyResult<Vec<PyObject>>
{
    // Convert Python dicts to AggTrade (see 3.2)
    let agg_trades = /* ... */;

    // Process through rangebar
    let bars = self.processor.process_agg_trade_records(&agg_trades)
        .map_err(|e| PyRuntimeError::new_err(format!("Processing failed: {}", e)))?;

    // Convert to Python dicts
    bars.iter()
        .map(|bar| rangebar_to_dict(py, bar))
        .collect()
}
```

#### 3.4: Test Rust Bindings

**File**: `tests/test_rust_bindings.py`

```python
import pytest
from rangebar._core import PyRangeBarProcessor

def test_processor_creation():
    processor = PyRangeBarProcessor(threshold_bps=250)
    assert processor.threshold_bps == 250

def test_process_simple_trades():
    processor = PyRangeBarProcessor(threshold_bps=250)

    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
        {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    ]

    bars = processor.process_trades(trades)

    assert isinstance(bars, list)
    assert len(bars) > 0
    assert "timestamp" in bars[0]
    assert "open" in bars[0]
```

**Run**:

```bash
maturin develop
pytest tests/test_rust_bindings.py -v
```

### Success Criteria

- [x] `PyRangeBarProcessor` compiles
- [x] `process_trades()` converts Python dicts to Rust structs
- [x] `process_trades()` returns list of Python dicts
- [x] Tests pass

### Troubleshooting

**Issue**: `AggTrade` struct fields unknown
**Solution**: Check rangebar-core documentation or source code

**Issue**: `FixedPoint::from_f64()` not available
**Solution**: Check rangebar-core API (might be `new()` or different method)

---

## Phase 4: Python API Layer (3 hours)

### Objectives

- Create Pythonic wrapper around Rust bindings
- Implement pandas DataFrame conversion
- Add type hints and documentation

### Tasks

#### 4.1: Implement `RangeBarProcessor` Class

**File**: `python/rangebar/__init__.py`

```python
from typing import List, Dict, Union
import pandas as pd
from ._core import PyRangeBarProcessor as _RangeBarProcessor

class RangeBarProcessor:
    """
    Process tick-level trade data into range bars.

    Parameters
    ----------
    threshold_bps : int
        Threshold in 0.1 basis point units (250 = 25bps = 0.25%)

    Examples
    --------
    >>> processor = RangeBarProcessor(threshold_bps=250)
    >>> trades = [{"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5}]
    >>> bars = processor.process_trades(trades)
    >>> df = processor.to_dataframe(bars)
    """

    def __init__(self, threshold_bps: int):
        if threshold_bps <= 0:
            raise ValueError("threshold_bps must be positive")
        self._processor = _RangeBarProcessor(threshold_bps)
        self.threshold_bps = threshold_bps

    def process_trades(self, trades: List[Dict[str, Union[int, float]]]) -> List[Dict]:
        """Process trades into range bars."""
        if not trades:
            return []

        # Validate
        required = {"timestamp", "price", "quantity"}
        for i, trade in enumerate(trades):
            missing = required - set(trade.keys())
            if missing:
                raise ValueError(f"Trade {i} missing: {missing}")

        return self._processor.process_trades(trades)

    def to_dataframe(self, bars: List[Dict]) -> pd.DataFrame:
        """Convert bars to pandas DataFrame (backtesting.py compatible)."""
        if not bars:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Capitalize for backtesting.py
        return df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })[["Open", "High", "Low", "Close", "Volume"]]

    def reset(self):
        """Reset processor state."""
        self._processor.reset()
```

#### 4.2: Implement Convenience Function

**File**: `python/rangebar/__init__.py` (add to existing)

```python
def process_trades_to_dataframe(
    trades: Union[List[Dict], pd.DataFrame],
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """
    Convenience function to process trades directly to DataFrame.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns: timestamp, price, quantity
    threshold_bps : int
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py
    """
    processor = RangeBarProcessor(threshold_bps)

    # Convert DataFrame to list of dicts
    if isinstance(trades, pd.DataFrame):
        required = {"timestamp", "price", "quantity"}
        if not required.issubset(trades.columns):
            raise ValueError(f"DataFrame missing columns: {required - set(trades.columns)}")

        # Convert timestamp to milliseconds
        trades_copy = trades.copy()
        if pd.api.types.is_datetime64_any_dtype(trades_copy["timestamp"]):
            trades_copy["timestamp"] = (trades_copy["timestamp"].astype("int64") // 10**6)

        trades = trades_copy[["timestamp", "price", "quantity"]].to_dict("records")

    bars = processor.process_trades(trades)
    return processor.to_dataframe(bars)
```

#### 4.3: Add Type Stubs

**File**: `python/rangebar/__init__.pyi`

```python
from typing import List, Dict, Union
import pandas as pd

__version__: str

class RangeBarProcessor:
    threshold_bps: int
    def __init__(self, threshold_bps: int) -> None: ...
    def process_trades(self, trades: List[Dict[str, Union[int, float]]]) -> List[Dict]: ...
    def to_dataframe(self, bars: List[Dict]) -> pd.DataFrame: ...
    def reset(self) -> None: ...

def process_trades_to_dataframe(
    trades: Union[List[Dict], pd.DataFrame],
    threshold_bps: int = 250,
) -> pd.DataFrame: ...
```

#### 4.4: Test Python API

**File**: `tests/test_python_api.py`

```python
import pandas as pd
from rangebar import RangeBarProcessor, process_trades_to_dataframe

def test_processor_returns_dataframe():
    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
        {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    ]

    df = process_trades_to_dataframe(trades, threshold_bps=250)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

def test_dataframe_input():
    trades_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
        "price": [42000 + i*10 for i in range(10)],
        "quantity": [1.5] * 10,
    })

    df = process_trades_to_dataframe(trades_df, threshold_bps=250)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
```

### Success Criteria

- [x] `RangeBarProcessor` class implemented
- [x] `process_trades_to_dataframe()` convenience function works
- [x] Type stubs added
- [x] Tests pass

---

## Phase 5: Backtesting.py Integration (2 hours)

### Objectives

- Create backtesting.py integration utilities
- Test with actual backtesting.py
- Validate OHLCV compatibility

### Tasks

#### 5.1: Implement Integration Utilities

**File**: `python/rangebar/backtesting.py`

```python
"""Integration utilities for backtesting.py."""

from typing import Tuple
import pandas as pd
from . import process_trades_to_dataframe

def load_from_binance_csv(csv_path: str, threshold_bps: int = 250) -> pd.DataFrame:
    """
    Load Binance aggTrades CSV and convert to range bars.

    Binance CSV format:
    agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,is_buyer_maker
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"timestamp": "timestamp", "price": "price", "quantity": "quantity"})
    return process_trades_to_dataframe(df[["timestamp", "price", "quantity"]], threshold_bps)

def split_train_test(data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split range bar data into train/test sets."""
    split_idx = int(len(data) * train_ratio)
    return data.iloc[:split_idx], data.iloc[split_idx:]
```

#### 5.2: Create Backtesting.py Integration Test

**File**: `tests/test_backtesting_integration.py`

```python
import pytest
import pandas as pd
import numpy as np

def test_backtesting_py_compatibility():
    """Test actual backtesting.py integration."""
    try:
        from backtesting import Backtest, Strategy
    except ImportError:
        pytest.skip("backtesting.py not installed")

    from rangebar import process_trades_to_dataframe

    # Generate synthetic trade data
    trades = [
        {"timestamp": int(ts), "price": 42000 + i * 10, "quantity": 1.0}
        for i, ts in enumerate(
            pd.date_range("2024-01-01", periods=1000, freq="min").astype(int) / 10**6
        )
    ]

    df = process_trades_to_dataframe(trades, threshold_bps=250)

    # Simple buy-and-hold strategy
    class DummyStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            if not self.position:
                self.buy()

    # Run backtest
    bt = Backtest(df, DummyStrategy, cash=10000, commission=0.0002)
    stats = bt.run()

    # Verify stats structure
    assert "Return [%]" in stats
    assert "Sharpe Ratio" in stats
```

**Run**:

```bash
pip install backtesting.py
pytest tests/test_backtesting_integration.py -v
```

### Success Criteria

- [x] Integration utilities implemented
- [x] backtesting.py test passes
- [x] No OHLCV validation errors from backtesting.py

---

## Phase 6: Documentation & Examples (2 hours)

### Objectives

- Write comprehensive README
- Create usage examples
- Document API

### Tasks

#### 6.1: Create README.md

See example in CLAUDE.md → README section.

#### 6.2: Create Examples

**File**: `examples/basic_usage.py`

```python
#!/usr/bin/env python3
"""Basic rangebar-py usage example."""

import pandas as pd
from rangebar import process_trades_to_dataframe

# Synthetic trade data
trades = [
    {"timestamp": 1704067200000 + i*1000, "price": 42000 + i*10, "quantity": 1.0}
    for i in range(100)
]

# Convert to range bars
df = process_trades_to_dataframe(trades, threshold_bps=250)

print(f"Generated {len(df)} range bars from {len(trades)} trades")
print("\nFirst 5 bars:")
print(df.head())
```

**File**: `examples/backtesting_integration.py`

```python
#!/usr/bin/env python3
"""Complete backtesting.py integration example."""

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from rangebar import process_trades_to_dataframe

# Generate synthetic data
trades_df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10000, freq="min"),
    "price": (42000 + pd.Series(range(10000)).cumsum() * 0.01),
    "quantity": 1.0,
})

# Convert to range bars
data = process_trades_to_dataframe(trades_df, threshold_bps=250)

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

### Success Criteria

- [x] README.md written
- [x] Examples created and tested
- [x] Examples run successfully

---

## Phase 7: Testing & Quality (3 hours)

### Objectives

- Achieve 95%+ test coverage
- Add performance benchmarks
- Run linting and type checking

### Tasks

#### 7.1: Add Comprehensive Tests

**Coverage Goals**:

- `src/lib.rs`: 90%+ (PyO3 bindings)
- `python/rangebar/`: 95%+ (Python API)
- Edge cases: Empty input, invalid data, etc.

**Run coverage**:

```bash
# Python coverage
pip install pytest-cov
pytest tests/ --cov=python/rangebar --cov-report=html

# View report
open htmlcov/index.html
```

#### 7.2: Add Performance Benchmarks

**File**: `tests/test_performance.py`

```python
import pytest
import pandas as pd
from rangebar import process_trades_to_dataframe

@pytest.mark.benchmark
def test_process_1m_trades(benchmark):
    """Benchmark processing 1 million trades."""
    trades = [
        {"timestamp": 1704067200000 + i, "price": 42000 + i * 0.01, "quantity": 1.0}
        for i in range(1_000_000)
    ]

    result = benchmark(process_trades_to_dataframe, trades, 250)

    assert len(result) > 0
    print(f"\nProcessed {len(trades)} trades → {len(result)} bars")
```

**Run**:

```bash
pip install pytest-benchmark
pytest tests/test_performance.py --benchmark-only
```

#### 7.3: Run Linting and Type Checking

```bash
# Type checking
mypy python/rangebar/

# Linting
ruff check python/
black python/ --check

# Format
black python/
```

### Success Criteria

- [x] Test coverage ≥95%
- [x] Benchmark shows >1M trades/sec
- [x] No linting errors
- [x] Type checking passes

---

## Phase 8: Distribution (2 hours)

### Objectives

- Build wheels for multiple platforms
- Publish to PyPI (test first)
- Set up CI/CD

### Tasks

#### 8.1: Build Wheels

```bash
# Build for current platform
maturin build --release

# Build for multiple Python versions
maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12

# Check wheels
ls target/wheels/
```

#### 8.2: Test Wheel Installation

```bash
# Create fresh venv
python -m venv test-env
source test-env/bin/activate

# Install wheel
pip install target/wheels/rangebar-*.whl

# Test
python -c "import rangebar; print(rangebar.__version__)"

deactivate
```

#### 8.3: Publish to Test PyPI

```bash
# Build
maturin build --release --sdist

# Upload to test.pypi.org
maturin publish --repository testpypi
```

#### 8.4: Set Up GitHub Actions (Optional)

**File**: `.github/workflows/ci.yml`

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - run: pip install maturin pytest pandas
      - run: maturin develop
      - run: pytest tests/

  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: pip install maturin
      - run: maturin build --release
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: target/wheels/
```

### Success Criteria

- [x] Wheels built for Linux, macOS, Windows
- [x] Wheels installable in fresh environment
- [x] Published to test PyPI
- [x] CI/CD runs successfully

---

## Phase 9: Production Release (1 hour)

### Objectives

- Finalize documentation
- Publish to PyPI
- Tag release on GitHub

### Tasks

#### 9.1: Final Checklist

- [ ] README complete with examples
- [ ] LICENSE file added (MIT)
- [ ] CHANGELOG.md created
- [ ] All tests passing
- [ ] Wheels built for all platforms

#### 9.2: Publish to PyPI

```bash
# Build release
maturin build --release --sdist

# Publish
maturin publish
```

#### 9.3: Create GitHub Release

```bash
git tag v0.1.0
git push origin v0.1.0

# Create release on GitHub UI or via gh CLI
gh release create v0.1.0 --title "v0.1.0 - Initial Release" --notes "First public release of rangebar-py"
```

### Success Criteria

- [x] Published to PyPI
- [x] GitHub release created
- [x] Installation works: `pip install rangebar`

---

## Maintenance & Future Work

### Post-Release Tasks

- Monitor PyPI downloads
- Respond to GitHub issues
- Update for new rangebar-core versions

### Future Enhancements

- [ ] Streaming API (incremental processing)
- [ ] Multi-symbol batch processing
- [ ] Parquet output support
- [ ] Visualization utilities
- [ ] Advanced backtesting.py helpers

---

## Quick Reference Commands

```bash
# Development workflow
maturin develop          # Editable install
pytest tests/ -v         # Run tests
mypy python/rangebar/    # Type check
black python/            # Format

# Building
maturin build --release  # Build wheel
maturin publish          # Publish to PyPI

# Testing
pytest tests/ --cov=python/rangebar --cov-report=html
pytest tests/test_performance.py --benchmark-only
```

---

## Estimated Timeline

| Phase             | Tasks                                | Time    | Cumulative |
| ----------------- | ------------------------------------ | ------- | ---------- |
| 1. Scaffolding    | Directory structure, Git init        | 30 min  | 30 min     |
| 2. Build Config   | Cargo.toml, pyproject.toml           | 1 hour  | 1.5 hours  |
| 3. Rust Bindings  | PyO3 wrapper, type conversion        | 4 hours | 5.5 hours  |
| 4. Python API     | Pythonic wrapper, pandas integration | 3 hours | 8.5 hours  |
| 5. Backtesting.py | Integration utilities, testing       | 2 hours | 10.5 hours |
| 6. Documentation  | README, examples                     | 2 hours | 12.5 hours |
| 7. Testing        | Coverage, benchmarks, linting        | 3 hours | 15.5 hours |
| 8. Distribution   | Build wheels, CI/CD                  | 2 hours | 17.5 hours |
| 9. Release        | Finalize, publish                    | 1 hour  | 18.5 hours |

**Total**: ~19 hours (~2.5 days of focused work)

**MVP** (Phases 1-5): ~11 hours (~1.5 days)

---

## Support & Resources

- **Upstream rangebar**: https://github.com/terrylica/rangebar
- **PyO3 Guide**: https://pyo3.rs/
- **Maturin Docs**: https://www.maturin.rs/
- **backtesting.py**: https://kernc.github.io/backtesting.py/

---

## Next Steps

When ready to start implementation:

```bash
cd ~/eon/rangebar-py
# Start with Phase 1
```

Read `CLAUDE.md` for complete project context before beginning.
