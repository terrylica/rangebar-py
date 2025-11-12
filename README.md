# rangebar-py

**Python bindings for [rangebar](https://github.com/terrylica/rangebar) - Non-lookahead range bar construction for cryptocurrency trading**

## Status: 🚧 Planning Phase

This project is currently in the planning stage. See `IMPLEMENTATION_PLAN.md` for the complete roadmap.

---

## What is rangebar-py?

Python bindings (via PyO3/maturin) to the high-performance rangebar Rust crate, enabling Python users to leverage temporally-sound range bar construction without requiring the upstream maintainer to add Python support.

### What are Range Bars?

Unlike traditional time-based bars (5-minute, 1-hour), **range bars** close when price moves a fixed percentage from the opening price:

- **Fixed Price Movement**: Each bar represents equal volatility (e.g., 0.25% = 25 basis points)
- **Market-Adaptive**: Bars form faster during trending, slower during consolidation
- **No Lookahead Bias**: Strict temporal integrity (breach tick included in closing bar)
- **Eliminates Time Artifacts**: No arbitrary time intervals

### Why This Project?

The backtesting.py project (`~/eon/backtesting.py`) tested **17 crypto strategies** on time-based bars with **100% failure rate**:

- MA crossover: 40.3% (5-min) → 38.7% (15-min) → 37.2% (1-hour) - **inverse timeframe effect**
- All strategies: <45% win rate, ≈-100% returns
- **Conclusion**: Time-based bars fundamentally incompatible with crypto markets

Range bars offer an alternative that may reveal hidden market structure.

---

## Quick Start (After Implementation)

```python
import pandas as pd
from backtesting import Backtest, Strategy
from rangebar import process_trades_to_dataframe

# Load tick data
trades = pd.read_csv("BTCUSDT-aggTrades.csv")

# Convert to range bars (25 basis points = 0.25%)
data = process_trades_to_dataframe(trades, threshold_bps=250)

# Use with backtesting.py
bt = Backtest(data, MyStrategy, cash=10000, commission=0.0002)
stats = bt.run()
bt.plot()
```

---

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

**Key Principle**: The rangebar maintainer does **ZERO** work. We import their crate as a dependency.

---

## Project Files

- **`CLAUDE.md`**: Complete project memory (read this first when working in this directory)
- **`IMPLEMENTATION_PLAN.md`**: Step-by-step implementation roadmap (9 phases, ~19 hours)
- **`TODO.md`**: Quick checklist of immediate next steps

---

## Development Workflow (Post-MVP)

```bash
# Setup
cd ~/eon/rangebar-py
pip install maturin pytest pandas backtesting.py

# Develop
maturin develop

# Test
pytest tests/ -v

# Build
maturin build --release

# Publish
maturin publish
```

---

## Features (Planned)

- ✅ **High Performance**: >1M trades/sec via Rust backend
- ✅ **Pandas Integration**: Returns DataFrames ready for analysis
- ✅ **Backtesting.py Compatible**: Drop-in replacement for time bars
- ✅ **Type Hints**: Full IDE support with type stubs
- ✅ **Zero Dependencies on Upstream**: No changes needed to rangebar crate

---

## Requirements

- Python ≥ 3.9
- Rust toolchain (for building)
- pandas ≥ 2.0
- numpy ≥ 1.24

---

## Contributing

This project is in the planning phase. See `IMPLEMENTATION_PLAN.md` for how to contribute.

---

## License

MIT (matches upstream rangebar crate)

---

## Credits

- **Python bindings**: This project
- **Rust crate**: [terrylica/rangebar](https://github.com/terrylica/rangebar)
- **Target framework**: [backtesting.py](https://kernc.github.io/backtesting.py/)

---

## Next Steps

1. Read `CLAUDE.md` for complete project context
2. Review `IMPLEMENTATION_PLAN.md` for implementation roadmap
3. Start with Phase 1 (Project Scaffolding)

```bash
cd ~/eon/rangebar-py
# Ready to implement!
```
