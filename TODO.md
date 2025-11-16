# TODO - Quick Checklist

**Status**: Planning → Implementation
**Last Updated**: 2025-10-06

---

## Immediate Next Steps (When Ready to Start)

### Phase 1: Project Scaffolding (30 minutes)

- [ ] Create directory structure (`src/`, `python/rangebar/`, `tests/`, `examples/`, `docs/`)
- [ ] Initialize Git repository
- [ ] Create `.gitignore` (Rust + Python patterns)
- [ ] Verify Rust toolchain installed (`rustc --version`)
- [ ] Verify Python 3.9+ installed (`python --version`)

**Commands**:

```bash
cd ~/eon/rangebar-py
mkdir -p src python/rangebar tests examples docs
git init
# Create .gitignore (see IMPLEMENTATION_PLAN.md)
```

---

### Phase 2: Build Configuration (1 hour)

- [ ] Create `Cargo.toml` with rangebar-core dependency
- [ ] Create `pyproject.toml` with maturin configuration
- [ ] Create minimal `src/lib.rs` (hello world module)
- [ ] Create minimal `python/rangebar/__init__.py`
- [ ] Verify build works: `maturin develop`
- [ ] Verify import works: `python -c "import rangebar; print(rangebar.__version__)"`

**Critical**: Check rangebar-core version on crates.io before adding to Cargo.toml:

```bash
cargo search rangebar-core --limit 1
```

---

### Phase 3: Core Rust Bindings (4 hours)

- [ ] Research rangebar-core API (check GitHub or `cargo doc --open`)
- [ ] Understand `AggTrade` struct definition
- [ ] Understand `RangeBar` struct definition
- [ ] Implement `PyRangeBarProcessor` wrapper class
- [ ] Implement Python dict → Rust `AggTrade` conversion
- [ ] Implement Rust `RangeBar` → Python dict conversion
- [ ] Add error handling (Rust `Result` → Python exceptions)
- [ ] Test with synthetic data

**Blockers to Resolve**:

- [ ] Verify exact fields in `rangebar_core::AggTrade`
- [ ] Verify exact fields in `rangebar_core::RangeBar`
- [ ] Verify `FixedPoint` API (how to convert to/from f64)

---

### Phase 4: Python API Layer (3 hours)

- [ ] Implement `RangeBarProcessor` class in `python/rangebar/__init__.py`
- [ ] Implement `process_trades_to_dataframe()` convenience function
- [ ] Add pandas DataFrame input support
- [ ] Create type stubs (`python/rangebar/__init__.pyi`)
- [ ] Add docstrings with examples
- [ ] Test DataFrame output format (verify OHLCV columns)

---

### Phase 5: Backtesting.py Integration (2 hours)

- [ ] Create `python/rangebar/backtesting.py` with integration utilities
- [ ] Implement `load_from_binance_csv()`
- [ ] Implement `split_train_test()`
- [ ] Install backtesting.py: `pip install backtesting.py`
- [ ] Create integration test (`tests/test_backtesting_integration.py`)
- [ ] Run test and verify no OHLCV errors

---

### Phase 6: Documentation & Examples (2 hours)

- [ ] Expand README.md with usage examples
- [ ] Create `examples/basic_usage.py`
- [ ] Create `examples/backtesting_integration.py`
- [ ] Create `examples/comparison_time_vs_range.py`
- [ ] Test all examples run successfully

---

### Phase 7: Testing & Quality (3 hours)

- [ ] Achieve 95%+ test coverage
- [ ] Add performance benchmarks (`tests/test_performance.py`)
- [ ] Run mypy type checking
- [ ] Run ruff linting
- [ ] Fix all linting errors
- [ ] Format with black

**Commands**:

```bash
pytest tests/ --cov=python/rangebar --cov-report=html
pytest tests/test_performance.py --benchmark-only
mypy python/rangebar/
ruff check python/
black python/
```

---

### Phase 8: Distribution (2 hours)

- [ ] Build wheels: `maturin build --release`
- [ ] Test wheel installation in fresh venv
- [ ] Publish to Test PyPI: `maturin publish --repository testpypi`
- [ ] Test install from Test PyPI
- [ ] (Optional) Set up GitHub Actions CI/CD

---

### Phase 9: Production Release (1 hour)

- [ ] Add LICENSE file (MIT)
- [ ] Create CHANGELOG.md
- [ ] Final README review
- [ ] Publish to PyPI: `maturin publish`
- [ ] Create GitHub release: `gh release create v0.1.0`
- [ ] Test install: `pip install rangebar`

---

## Research Tasks (Before Phase 3)

### Rangebar-Core API Investigation

**Goal**: Understand the exact API before writing bindings.

**Steps**:

1. Clone rangebar repository or check docs:

   ```bash
   git clone https://github.com/terrylica/rangebar /tmp/rangebar
   cd /tmp/rangebar
   cargo doc --open
   ```

2. Find answers to:
   - [ ] What are the fields in `AggTrade` struct?
   - [ ] What are the fields in `RangeBar` struct?
   - [ ] How do I create a `FixedPoint` from `f64`?
   - [ ] How do I convert a `FixedPoint` to `f64`?
   - [ ] What does `RangeBarProcessor::new()` expect?
   - [ ] What does `RangeBarProcessor::process_agg_trade_records()` return?

3. Document findings in `docs/rangebar_core_api.md`

---

## Dependency Installation Checklist

### System Dependencies

- [ ] Rust toolchain (≥1.70)

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source $HOME/.cargo/env
  rustc --version
  ```

- [ ] Python 3.9+ with venv
  ```bash
  python3 --version
  python3 -m venv venv
  source venv/bin/activate
  ```

### Python Dependencies

- [ ] maturin (build system)

  ```bash
  pip install maturin
  ```

- [ ] Core dependencies

  ```bash
  pip install pandas numpy
  ```

- [ ] Testing dependencies

  ```bash
  pip install pytest pytest-cov pytest-benchmark
  ```

- [ ] Quality tools

  ```bash
  pip install mypy black ruff
  ```

- [ ] Target framework (optional, for testing)
  ```bash
  pip install backtesting.py
  ```

---

## Quick Decision Log

**Questions to resolve during implementation**:

1. **Q**: Should we support streaming API (incremental processing)?
   **A**: Not in MVP. Add post-v1.0.0 if users request it.

2. **Q**: Should we provide CLI tools (like rangebar-cli)?
   **A**: No. Users can use upstream rangebar-cli. We focus on Python API.

3. **Q**: Should we include data fetching (Binance API)?
   **A**: No. Users provide data, we process it. Separation of concerns.

4. **Q**: Support Parquet output?
   **A**: Maybe post-MVP. Start with pandas DataFrame only.

5. **Q**: Multi-symbol batch processing?
   **A**: Post-MVP. Start with single-symbol workflows.

---

## Success Metrics

### MVP (Minimum Viable Product)

- [x] `pip install rangebar` works (from PyPI)
- [x] `process_trades_to_dataframe()` converts Binance CSV to range bars
- [x] Output works with backtesting.py (no OHLCV errors)
- [x] Performance: >1M trades/sec
- [x] Test coverage: ≥95%
- [x] Documentation: README with 3+ examples

### Post-MVP Enhancements

- [ ] Streaming API
- [ ] Multi-symbol processing
- [ ] Parquet output
- [ ] Visualization utilities
- [ ] Advanced backtesting.py helpers

---

## Current Blockers

**None** - Ready to start Phase 1 when you're ready!

---

## Timeline Estimate

| Phase                    | Time      | Cumulative |
| ------------------------ | --------- | ---------- |
| 1-2: Setup               | 1.5 hours | 1.5 hours  |
| 3-4: Core Implementation | 7 hours   | 8.5 hours  |
| 5-6: Integration & Docs  | 4 hours   | 12.5 hours |
| 7-9: Testing & Release   | 6 hours   | 18.5 hours |

**Total**: ~19 hours (~2.5 days)
**MVP (Phases 1-5)**: ~11 hours (~1.5 days)

---

## Useful Commands Reference

```bash
# Development
maturin develop          # Editable install
pytest tests/ -v         # Run tests
mypy python/rangebar/    # Type check
black python/            # Format code
ruff check python/       # Lint

# Building
maturin build --release  # Build wheel
maturin publish          # Publish to PyPI

# Testing
pytest tests/ --cov=python/rangebar --cov-report=html
pytest tests/test_performance.py --benchmark-only

# Debugging
python -c "import rangebar; help(rangebar.RangeBarProcessor)"
python -c "from rangebar import __version__; print(__version__)"
```

---

## Next Action

**When ready to start**:

```bash
cd ~/eon/rangebar-py
# Read CLAUDE.md for full context
# Follow IMPLEMENTATION_PLAN.md Phase 1
```

**First command to run**:

```bash
mkdir -p src python/rangebar tests examples docs
```
