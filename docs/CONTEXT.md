# Project Context

**Parent**: [/CLAUDE.md](/CLAUDE.md)

This document explains why rangebar-py exists and its relationship to other projects.

---

## Problem Statement

The backtesting.py project (located at `~/eon/backtesting.py`) completed crypto intraday research (Phases 8-16B) that **TERMINATED** all strategies due to:

- **Inverse Timeframe Effect**: MA crossover win rates degraded from 40.3% (5-min) → 38.7% (15-min) → 37.2% (1-hour)
- **17 strategies tested**, all failed with <45% win rate
- **Time-based bars fundamentally incompatible** with crypto market structure

---

## Solution: Range Bars

Range bars offer an alternative to time-based bars:

- **Fixed Price Movement**: Each bar closes when price moves X% from open (e.g., 0.25% = 25 basis points)
- **Market-Adaptive**: Bars form faster during volatile periods, slower during consolidation
- **No Lookahead Bias**: Strict temporal integrity (breach tick included in closing bar)
- **Eliminates Time Artifacts**: No arbitrary 5-min/15-min/1-hour intervals

---

## Why PyO3 Bindings?

The upstream rangebar Rust crate maintainer:

- ✅ **Will maintain** the Rust crate
- ✅ **Provides CLI tools** for command-line usage
- ❌ **Will NOT add Python support** to the crate itself

Our approach:

- ✅ **Local Rust crates** (in `crates/`) - we control the full source
- ✅ **Write PyO3 wrapper** (this project)
- ✅ **Zero external dependency** - all Rust code is local
- ✅ **We control Python API** - can iterate quickly

---

## Target Use Case

### Primary: backtesting.py Integration

```python
from backtesting import Backtest, Strategy
from rangebar import get_range_bars

# Fetch data and generate range bars in one call
data = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Use directly with backtesting.py
bt = Backtest(data, MyStrategy, cash=10000, commission=0.0002)
stats = bt.run()
bt.plot()
```

### Secondary: ML Training

```python
from rangebar import get_n_range_bars

# Get exactly N bars for walk-forward validation
train_data = get_n_range_bars("BTCUSDT", n_bars=10000)
```

---

## Relationship to backtesting.py

This project is a **companion tool** for the backtesting.py fork:

- **Path**: `~/eon/backtesting.py`
- **Branch**: `research/compression-breakout`
- **Status**: Research terminated after 17 failed strategies on time-based bars

### Research Question

Do mean reversion / trend-following strategies perform better on range bars than time bars?

### Integration Workflow

1. **Generate range bars** (this project):
   ```python
   from rangebar import get_range_bars
   data = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")
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

---

## Technical Requirements

### Rust Dependencies

All Rust crates are local (in `crates/`):
- **rangebar-core**: Core range bar algorithm
- **rangebar-providers**: Data sources (Binance, Exness)
- **pyo3**: Python bindings
- **chrono**: Timestamp handling

### Python Dependencies

- **pandas**: ≥2.0 - DataFrame operations
- **numpy**: ≥1.24 - Numerical operations
- **polars**: ≥0.20 - High-performance DataFrames
- **backtesting.py**: ≥0.3 (optional) - Target integration library

### Build System

- **maturin**: ≥1.7 - Python package builder for Rust extensions
- **PyO3**: abi3 wheels for Python 3.9+ compatibility

---

## Success Criteria

### Achieved (MVP Complete)

- [x] `process_trades_to_dataframe()` works with Binance CSV
- [x] Output is backtesting.py compatible
- [x] No lookahead bias (verified against Rust implementation)
- [x] Performance: >1M trades/sec
- [x] Memory: <2GB for typical datasets
- [x] Tests: 95%+ coverage
- [x] Documentation: README with examples

### Future Enhancements

- [ ] Streaming API (process trades incrementally)
- [ ] Multi-symbol processing (batch conversion)
- [ ] Advanced backtesting.py helpers (walk-forward optimization)
- [ ] Visualization tools (plot range bars vs time bars)

---

## Non-Goals

- ❌ **Replace rangebar CLI**: We provide Python bindings, not CLI alternatives
- ❌ **Support all data sources**: Focus on Binance; users can adapt
- ❌ **Real-time trading**: This is for backtesting only

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md) - System architecture
- [/docs/api.md](/docs/api.md) - Python API reference
