# CLAUDE.md - Project Memory

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**rangebar-py** is a Python package providing PyO3/maturin bindings to the [rangebar](https://github.com/terrylica/rangebar) Rust crate. This enables Python users (especially those using backtesting.py) to leverage high-performance range bar construction without requiring the upstream Rust crate maintainer to add Python support.

**Core Principle**: The rangebar Rust crate maintainer does **ZERO** work. We handle all Python integration by importing their crate as a dependency.

## Architecture

```
rangebar (Rust crate on crates.io) [maintained by terrylica]
    ↓ [Cargo dependency]
rangebar-py (This project) [our Python wrapper]
    ├── Rust code (PyO3 bindings in src/lib.rs)
    ├── Python helpers (python/rangebar/)
    └── Type stubs (.pyi files)
    ↓ [pip install]
backtesting.py users (target audience)
```

## Why This Project Exists

### Problem Statement

The backtesting.py project (located at `~/eon/backtesting.py`) completed crypto intraday research (Phases 8-16B) that **TERMINATED** all strategies due to:

- **Inverse Timeframe Effect**: MA crossover win rates degraded from 40.3% (5-min) → 38.7% (15-min) → 37.2% (1-hour)
- **17 strategies tested**, all failed with <45% win rate
- **Time-based bars fundamentally incompatible** with crypto market structure

### Solution: Range Bars

Range bars offer an alternative to time-based bars:

- **Fixed Price Movement**: Each bar closes when price moves X% from open (e.g., 0.25% = 25 basis points)
- **Market-Adaptive**: Bars form faster during volatile periods, slower during consolidation
- **No Lookahead Bias**: Strict temporal integrity (breach tick included in closing bar)
- **Eliminates Time Artifacts**: No arbitrary 5-min/15-min/1-hour intervals

### Why PyO3 Bindings?

The rangebar maintainer:

- ✅ **Will maintain** the Rust crate
- ✅ **Provides CLI tools** for command-line usage
- ❌ **Will NOT add Python support** to the crate itself

Our approach:

- ✅ **Import rangebar as dependency** (from crates.io)
- ✅ **Write PyO3 wrapper** (this project)
- ✅ **Zero burden on maintainer** - they don't need to do anything
- ✅ **We control Python API** - can iterate quickly

## Target Use Case

### Primary Workflow (backtesting.py integration)

```python
import pandas as pd
from backtesting import Backtest, Strategy
from rangebar import process_trades_to_dataframe

# Load tick data from Binance
trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")

# Convert to range bars (25 basis points = 0.25%)
data = process_trades_to_dataframe(trades, threshold_bps=250)

# Use directly with backtesting.py
bt = Backtest(data, MyStrategy, cash=10000, commission=0.0002)
stats = bt.run()
bt.plot()
```

### Output Format (backtesting.py compatible)

```python
# DataFrame with DatetimeIndex and OHLCV columns
                          Open      High       Low     Close  Volume
timestamp
2024-01-01 00:00:15  42000.00  42105.00  41980.00  42100.00   15.43
2024-01-01 00:03:42  42100.00  42220.00  42100.00  42215.00    8.72
```

## Technical Requirements

### Rust Dependencies

- **rangebar-core**: v5.0.0 (from crates.io) - Core range bar algorithm
- **pyo3**: v0.22 - Python bindings
- **numpy**: v0.22 - NumPy arrays support
- **chrono**: v0.4 - Timestamp handling

### Python Dependencies

- **pandas**: ≥2.0 - DataFrame operations
- **numpy**: ≥1.24 - Numerical operations
- **backtesting.py**: ≥0.3 (optional) - Target integration library

### Build System

- **maturin**: ≥1.7 - Python package builder for Rust extensions
- **PyO3**: abi3 wheels for Python 3.9+ compatibility

## Project Structure

```
rangebar-py/
├── CLAUDE.md                       # This file (project memory)
├── IMPLEMENTATION_PLAN.md          # High-level implementation roadmap
├── README.md                       # User-facing documentation
├── LICENSE                         # MIT license
├── Cargo.toml                      # Rust dependencies
├── pyproject.toml                  # Python packaging (maturin)
├── src/
│   └── lib.rs                      # PyO3 bindings (Rust → Python bridge)
├── python/
│   └── rangebar/
│       ├── __init__.py             # Pythonic API
│       ├── __init__.pyi            # Type stubs
│       ├── utils.py                # Helper functions
│       └── backtesting.py          # backtesting.py integration utilities
├── tests/
│   ├── test_core.py                # Core functionality tests
│   ├── test_backtesting.py         # backtesting.py integration tests
│   └── test_performance.py         # Performance benchmarks
├── examples/
│   ├── basic_usage.py              # Simple example
│   ├── backtesting_integration.py  # Full backtesting.py workflow
│   └── comparison_time_vs_range.py # Time bars vs range bars comparison
└── docs/
    ├── api.md                      # API reference
    ├── algorithms.md               # Range bar algorithm explanation
    └── backtesting_guide.md        # Integration guide
```

## Key Implementation Files

### src/lib.rs (PyO3 Bindings)

**Responsibilities**:

- Import `rangebar_core::RangeBarProcessor` from crates.io
- Create `PyRangeBarProcessor` wrapper class
- Convert Rust types to Python types:
  - `RangeBar` → `Dict[str, float]` (timestamp, open, high, low, close, volume)
  - `AggTrade` ← `Dict[str, float]` (timestamp, price, quantity)
- Handle error conversion (Rust `Result` → Python exceptions)

**Critical Requirements**:

- **No lookahead bias**: Preserve temporal integrity from Rust implementation
- **Performance**: Avoid unnecessary copies (use zero-copy where possible)
- **Type safety**: Proper validation of Python input

### python/rangebar/**init**.py (Pythonic API)

**Responsibilities**:

- Wrap `PyRangeBarProcessor` with Pythonic interface
- Provide `process_trades_to_dataframe()` convenience function
- Handle pandas DataFrame conversions
- Validate input data structure
- Format output for backtesting.py compatibility

**Critical Requirements**:

- **OHLCV column names**: Capitalized (Open, High, Low, Close, Volume)
- **DatetimeIndex**: Proper timestamp parsing
- **Error messages**: User-friendly Python exceptions

### python/rangebar/backtesting.py (Integration Utilities)

**Responsibilities**:

- `load_from_binance_csv()`: Load Binance aggTrades format
- `split_train_test()`: Train/test splitting for walk-forward validation
- `compare_to_time_bars()`: Compare range bars vs time bars

**Critical Requirements**:

- **Binance format compatibility**: Handle aggTrades CSV structure
- **Temporal alignment**: Ensure no data leakage in train/test split

## Development Workflow

### Setup

```bash
cd ~/eon/rangebar-py

# Install Rust toolchain (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Install development dependencies
pip install pytest pandas backtesting.py mypy black ruff
```

### Development Commands

```bash
# Editable install (for development)
maturin develop

# Run tests
pytest tests/ -v

# Type checking
mypy python/rangebar/

# Linting
ruff check python/
black python/ --check

# Build release wheels
maturin build --release

# Build for multiple Python versions
maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12
```

### Testing Strategy

1. **Unit Tests** (`test_core.py`):
   - Test `RangeBarProcessor` with synthetic data
   - Verify OHLCV output structure
   - Test edge cases (empty input, single trade, etc.)

2. **Integration Tests** (`test_backtesting.py`):
   - Load actual Binance data
   - Generate range bars
   - Run with backtesting.py
   - Verify stats structure

3. **Performance Benchmarks** (`test_performance.py`):
   - Process 1M trades
   - Measure throughput (trades/sec)
   - Memory usage profiling

## Critical Implementation Details

### Timestamp Handling

**Rust Side (src/lib.rs)**:

```rust
// rangebar uses i64 milliseconds
let timestamp_ms: i64 = bar.timestamp_ms;

// Convert to RFC3339 string for Python
let timestamp = chrono::DateTime::from_timestamp_millis(timestamp_ms)
    .unwrap()
    .to_rfc3339();
```

**Python Side (python/rangebar/**init**.py)**:

```python
# Parse timestamp string to DatetimeIndex
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### Fixed-Point Arithmetic

The rangebar crate uses fixed-point arithmetic (8 decimal places) to avoid floating-point errors. Our bindings must preserve this:

```rust
// Convert rangebar FixedPoint to f64 for Python
let price_f64 = bar.open.to_f64();
```

**Warning**: Do NOT round or truncate - preserve full precision.

### Error Handling

Map Rust errors to Python exceptions:

```rust
let processor = RangeBarProcessor::new(threshold_bps)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
        format!("Failed to create processor: {}", e)
    ))?;
```

**Python exception types**:

- `ValueError`: Invalid input (negative threshold, missing fields)
- `RuntimeError`: Processing errors (unsorted trades, internal failures)
- `KeyError`: Missing required dictionary keys

## Testing Against backtesting.py

### Compatibility Checklist

- [x] **OHLCV column names**: Capitalized (Open, High, Low, Close, Volume)
- [x] **DatetimeIndex**: Pandas DatetimeIndex with timezone-naive timestamps
- [x] **No NaN values**: All bars complete (backtesting.py raises on NaN)
- [x] **Sorted chronologically**: Timestamps in ascending order
- [x] **OHLC invariants**: High ≥ max(Open, Close), Low ≤ min(Open, Close)

### Validation Script

```python
def validate_for_backtesting_py(df: pd.DataFrame) -> bool:
    """Validate DataFrame is compatible with backtesting.py."""
    # Check columns
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    # Check index
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

    # Check no NaN
    assert not df.isnull().any().any()

    # Check OHLC invariants
    assert (df["High"] >= df["Open"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()

    return True
```

## Relationship to backtesting.py Project

This project is a **companion tool** for the backtesting.py fork located at:

- **Path**: `~/eon/backtesting.py`
- **Branch**: `research/compression-breakout`
- **Status**: Research terminated after 17 failed strategies on time-based bars

### Integration Workflow

1. **Generate range bars** (this project):

   ```python
   from rangebar import process_trades_to_dataframe
   data = process_trades_to_dataframe(trades, threshold_bps=250)
   ```

2. **Use with backtesting.py**:

   ```python
   from backtesting import Backtest
   bt = Backtest(data, MyStrategy, cash=10000)
   stats = bt.run()
   ```

3. **Compare results**:
   - Range bars vs 5-minute bars
   - Range bars vs 15-minute bars
   - Range bars vs 1-hour bars

**Research Question**: Do mean reversion / trend-following strategies perform better on range bars than time bars?

## Daily Performance Monitoring

**Decision**: ADR-0007 - Daily Performance Monitoring on CI/CD

### Overview

Automated daily benchmarks run on GitHub Actions to track performance metrics over time, generating observable files for both humans (HTML dashboard) and AI agents (JSON files).

### Observable Files

**Machine-Readable** (AI agents):

- JSON files: `gh-pages:/dev/bench/python-bench.json` (pytest-benchmark output)
- Committed daily to `gh-pages` branch
- Historical data with unlimited retention

**Human-Readable** (developers):

- Dashboard: https://terrylica.github.io/rangebar-py/ (live after GitHub Pages enabled)
- Chart.js trend visualization
- Regression detection with 150% threshold (non-blocking warnings)

### Daily Viability Metrics

**Python API Layer** (pytest-benchmark):

- Throughput: 1K, 100K, 1M trades/sec (target: >1M trades/sec)
- Memory: Peak usage for 1M trades (target: <350MB)
- Compression: Ratio for 100/250/500/1000 bps thresholds

**Rust Core Layer** (Criterion.rs):

- Throughput: 1M trades/sec without PyO3 overhead
- Latency: P50/P95/P99 percentiles
- PyO3 overhead: Calculated as (Python throughput - Rust throughput)

### Execution Schedule

- **Daily**: 3:17 AM UTC (off-peak, automated via cron)
- **Manual**: On-demand via `gh workflow run performance-daily.yml`
- **Weekly**: Sunday 2:00 AM UTC (comprehensive benchmarks, extended rounds)

### Interpretation (AI Agents)

**Query JSON files for viability checks**:

```bash
# Fetch latest benchmark results
git fetch origin gh-pages
git checkout gh-pages
cat dev/bench/python-bench.json | jq '.benchmarks[] | {name: .name, mean: .stats.mean}'
```

**Viability criteria**:

- `test_throughput_1m_trades`: mean < 1.0 second (target: >1M trades/sec)
- `test_memory_1m_trades`: peak_memory < 350MB
- `test_compression_ratio`: bars_count > 0 for all thresholds

**Regression detection**:

- Alert if throughput degrades >50% vs previous run
- Alert if memory increases >50% vs previous run
- Warnings logged but workflow continues (non-blocking)

### Implementation Details

- **Workflows**: `.github/workflows/performance-daily.yml`, `.github/workflows/performance-weekly.yml`
- **Benchmarks**: `benches/core.rs` (Rust), `tests/test_performance.py` (Python)
- **Visualization**: github-action-benchmark (generates HTML/JSON)
- **Deployment**: GitHub Actions (via `actions/deploy-pages@v4`, not branch-based)
- **Documentation**: `docs/decisions/0007-daily-performance-monitoring.md` (ADR), `docs/plan/0007-daily-performance-monitoring/plan.md` (plan), `docs/GITHUB_PAGES_SETUP.md` (setup guide)

### Why Non-Blocking?

**Philosophy** (from ADR-0007, aligns with global ADR-0007 GitHub Actions policy):

- Trend monitoring, not gating
- Performance regressions caught early via dashboard
- Never blocks development workflow
- Complements local benchmarking (instant feedback loop)

## Package Distribution

### PyPI Release Workflow

```bash
# Build wheels for all platforms
maturin build --release --sdist

# Wheels generated in target/wheels/:
# - rangebar-0.1.0-cp39-cp39-manylinux_2_17_x86_64.whl
# - rangebar-0.1.0-cp310-cp310-manylinux_2_17_x86_64.whl
# - rangebar-0.1.0-cp39-cp39-macosx_11_0_arm64.whl
# - ... (more platforms)

# Upload to PyPI
maturin publish
```

### Pre-built Wheels Strategy

Use GitHub Actions to build wheels for:

- **Linux**: manylinux_2_17 (x86_64, aarch64)
- **macOS**: 11.0+ (x86_64, arm64)
- **Windows**: (x86_64)
- **Python**: 3.9, 3.10, 3.11, 3.12

Users can install without Rust toolchain:

```bash
pip install rangebar  # Downloads pre-built wheel
```

## Success Criteria

### Minimum Viable Product (MVP)

- [x] `process_trades_to_dataframe()` works with Binance CSV
- [x] Output is backtesting.py compatible
- [x] No lookahead bias (verified against Rust implementation)
- [x] Performance: >1M trades/sec
- [x] Memory: <2GB for typical datasets
- [x] Tests: 95%+ coverage
- [x] Documentation: README with examples

### Future Enhancements (Post-MVP)

- [ ] Streaming API (process trades incrementally)
- [ ] Multi-symbol processing (batch conversion)
- [ ] Parquet output support
- [ ] Advanced backtesting.py helpers (walk-forward optimization)
- [ ] Visualization tools (plot range bars vs time bars)

## Non-Goals

- ❌ **Replace rangebar CLI**: We provide Python bindings, not CLI alternatives
- ❌ **Modify upstream crate**: Zero changes to rangebar Rust code
- ❌ **Support all data sources**: Focus on Binance; users can adapt
- ❌ **Real-time trading**: This is for backtesting only

## License

**MIT** (matches upstream rangebar crate)

## Contributing

See `IMPLEMENTATION_PLAN.md` for step-by-step implementation roadmap.

## Contact / Maintainer

This project is maintained independently of the upstream rangebar Rust crate. For:

- **Python bindings issues**: Open issue in this repo
- **Range bar algorithm questions**: Refer to upstream [rangebar](https://github.com/terrylica/rangebar)
- **backtesting.py integration**: See examples in this repo

## Quick Start (After Implementation)

```bash
cd ~/eon/rangebar-py

# Setup
maturin develop
pytest tests/

# Try example
python examples/basic_usage.py
```

For detailed implementation steps, see `IMPLEMENTATION_PLAN.md`.
