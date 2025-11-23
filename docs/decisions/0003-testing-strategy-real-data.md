# ADR-003: Testing Strategy with Real Binance Data

**Status**: Accepted

**Date**: 2024-11-16

**Decision Makers**: Engineering Team

---

## Context

rangebar-py processes cryptocurrency trade data into range bars for backtesting. Current test suite (33 tests) uses synthetic data and achieves ~85% Python / ~88% Rust coverage. Production use requires validation against real market data to ensure correctness under diverse market conditions.

### Requirements

- Test coverage targets: 95% Python code, 90% Rust code
- Performance validation: >1M trades/sec throughput, <100MB memory for 1M trades
- Real-world edge case coverage: flash crashes, trading halts, timestamp format changes
- Example verification: All 4 examples must run without errors

### Constraints

- Repository size limit: Cannot commit large datasets (>10MB)
- CI runtime limit: Tests must complete within 10 minutes
- Platform compatibility: Tests must run on Linux, macOS (Intel + ARM)

---

## Decision

Use real Binance historical aggTrades data for testing while maintaining fast CI execution.

### Data Strategy

**Primary data source**: Binance Data Vision (https://data.binance.vision/)

**Downloaded files**:

- BTCUSDT-aggTrades-2024-01-01.csv.zip (~500MB) - Stored in `tests/fixtures/`, gitignored
- BTCUSDT-aggTrades-sample-10k.csv (~1MB) - 10,000 row sample, committed to repository

**Rationale**: Small sample enables fast CI tests; full dataset available for comprehensive local testing.

### Coverage Strategy

**Additional test files** (21 new tests):

1. `test_edge_cases.py` (12 tests) - Error handling, boundary conditions, type validation
2. `test_real_data.py` (3 tests) - Binance data format, OHLC invariants, threshold variations
3. `test_examples.py` (4 tests) - Automated verification all examples execute successfully
4. `test_performance.py` (6 tests) - Throughput benchmarks, memory profiling, compression ratios

**Gap analysis**: Current ~85% Python coverage missing edge cases (empty data, type mismatches, timezone handling). New tests target uncovered branches.

### Benchmark Strategy

**Tooling**: pytest-benchmark for Python, cargo-llvm-cov for Rust

**Targets** (Apple M1 baseline):

- Throughput: >1M trades/sec (batch processing)
- Memory: <100MB peak RSS for 1M trades
- Compression: 10-50x ratio (trades → bars) depending on threshold

**Baseline storage**: `.benchmarks/apple-m1-baseline.json` committed for regression detection

### Quality Tooling

**Linting**: ruff (strict mode), mypy (--strict), black, clippy (-D warnings)

**Automation**: Makefile with targets: test, coverage, lint, format, check, benchmark

---

## Consequences

### Positive

**Correctness**: Real market data validates production scenarios (2025 microsecond timestamps, flash crashes, gaps)

**Observability**: Coverage reports (`htmlcov/`, `target/llvm-cov/html/`) visualize untested code paths

**Maintainability**: Makefile provides single-command quality checks (`make check`)

**Availability**: Sample data in repository enables fast CI without downloads

### Negative

**Setup complexity**: First-time contributors must download ~500MB Binance data for comprehensive testing

**Storage cost**: Developers with full dataset consume ~500MB disk space

**Maintenance burden**: Binance format changes require test updates

### Mitigations

- Provide download script with checksum validation
- Tests gracefully skip if full dataset missing (sample-only mode)
- Document data source URLs in `tests/fixtures/README.md`

---

## Alternatives Considered

### Alternative 1: Synthetic Data Only

**Rejected**: Misses real-world edge cases (timestamp format changes, extreme volatility)

### Alternative 2: Mock/Stub Binance API

**Rejected**: Adds complexity, doesn't validate actual data format compatibility

### Alternative 3: Commit Full Dataset to Repository

**Rejected**: Violates GitHub file size limits, slows git operations

---

## Compliance

**SLO**:

- **Correctness**: 95%/90% coverage ensures code correctness
- **Observability**: Coverage reports + benchmark baselines provide visibility
- **Maintainability**: Real data tests document expected behavior

**Error Handling**: Tests fail loudly if coverage below target (no silent degradation)

**OSS Dependencies**: pytest-cov, pytest-benchmark, cargo-llvm-cov (no custom tooling)

---

## References

- Binance Data Vision: https://data.binance.vision/
- pytest-cov: https://pytest-cov.readthedocs.io/
- cargo-llvm-cov: https://github.com/taiki-e/cargo-llvm-cov

---

**Related**:

- ADR-004: CI/CD Multi-Platform Builds
- Plan: `docs/plan/0003-testing-strategy/plan.yaml`
