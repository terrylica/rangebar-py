# Performance Monitoring and Benchmarking

**Status**: Active (daily automated benchmarks since v0.4.0)
**Dashboard**: https://terrylica.github.io/rangebar-py/
**ADR**: [0007-daily-performance-monitoring.md](decisions/0007-daily-performance-monitoring.md)

---

## Overview

rangebar-py maintains automated daily performance benchmarks to provide tangible observable metrics for both human developers and AI coding agents. This enables clear visibility into system viability and performance trends over time.

## Benchmark Schedule

- **Daily**: 3:17 AM UTC (off-peak) - Fast validation (5 iterations)
- **Weekly**: Sunday 2:00 AM UTC - Comprehensive analysis (10 iterations)
- **Manual**: Trigger via GitHub Actions workflow

## Observable Files

### For AI Agents (Machine-Readable)

**Primary Data Source**:
```
https://terrylica.github.io/rangebar-py/dev/bench/data.js
```

Format: JavaScript object (`window.BENCHMARK_DATA`)

Structure:
```javascript
{
  "lastUpdate": 1763883878646,  // Unix timestamp (milliseconds)
  "repoUrl": "https://github.com/terrylica/rangebar-py",
  "entries": {
    "Python API Benchmarks": [
      {
        "commit": {
          "id": "8f6ca79...",
          "message": "...",
          "timestamp": "2025-11-23T07:42:10Z",
          "url": "https://github.com/terrylica/rangebar-py/commit/..."
        },
        "date": 1763883878409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_performance.py::TestThroughputBenchmarks::test_throughput_1k_trades",
            "value": 596.51,
            "unit": "iter/sec",
            "range": "stddev: 0.00005",
            "extra": "mean: 1.68 msec\\nrounds: 617"
          }
        ]
      }
    ]
  }
}
```

### For Humans (Human-Readable)

**Dashboard**: https://terrylica.github.io/rangebar-py/

Features:
- Historical trend charts (Chart.js)
- Throughput visualization
- Memory usage tracking
- Regression detection (150% threshold)

## Benchmark Metrics

### Python API Layer (pytest-benchmark)

#### Throughput Benchmarks

| Metric | Target | Unit | Description |
|--------|--------|------|-------------|
| `test_throughput_1k_trades` | >100 | iter/sec | Process 1,000 trades |
| `test_throughput_100k_trades` | >10 | iter/sec | Process 100,000 trades |
| `test_throughput_1m_trades` | >1 | iter/sec | Process 1,000,000 trades |

**Interpretation**:
- `iter/sec` = iterations per second
- Higher is better
- Each iteration processes N trades and generates range bars

#### Memory Benchmarks

| Metric | Target | Unit | Description |
|--------|--------|------|-------------|
| `test_memory_1m_trades` | <350 | MB | Peak memory for 1M trades |

**Methodology**:
- Uses `psutil.Process().memory_info().rss`
- Measures peak resident set size (RSS)
- Includes Python overhead + Rust allocations

#### Compression Benchmarks

| Metric | Description |
|--------|-------------|
| `test_compression_ratio[100]` | Compression at 100 bps (1%) |
| `test_compression_ratio[250]` | Compression at 250 bps (2.5%) |
| `test_compression_ratio[500]` | Compression at 500 bps (5%) |
| `test_compression_ratio[1000]` | Compression at 1000 bps (10%) |

**Interpretation**:
- Compression ratio = (input trades) / (output range bars)
- Higher compression = fewer bars (more aggregation)
- Depends on price volatility and threshold

### Rust Core Layer (Criterion.rs)

**Status**: Benchmarks available in `benches/core.rs`, execution deferred to Phase 2

Planned metrics:
- Throughput without PyO3 overhead
- Latency percentiles (P50, P95, P99)
- Memory allocations (Rust-only)
- PyO3 overhead calculation

## AI Agent Interpretation Guide

### Daily Viability Assessment

**Healthy System** (expected values):

```python
# Example: Parse and validate benchmark data
import requests
import json

url = "https://terrylica.github.io/rangebar-py/dev/bench/data.js"
response = requests.get(url)
data_js = response.text

# Extract JSON from JavaScript wrapper
json_str = data_js.replace("window.BENCHMARK_DATA = ", "").rstrip(";\n")
data = json.loads(json_str)

# Get latest benchmark
latest = data["entries"]["Python API Benchmarks"][-1]
benches = {b["name"].split("::")[-1]: b["value"] for b in latest["benches"]}

# Validate targets
assert benches["test_throughput_1k_trades"] > 100, "1K throughput below target"
assert benches["test_throughput_100k_trades"] > 10, "100K throughput below target"
assert benches["test_throughput_1m_trades"] > 1, "1M throughput below target"

print("✅ System viable - all performance targets met")
```

### Regression Detection

**Regression Alert**: Performance degrades >150% from baseline

```python
# Check for regressions across multiple runs
entries = data["entries"]["Python API Benchmarks"]

if len(entries) >= 2:
    latest = entries[-1]
    previous = entries[-2]

    for latest_bench in latest["benches"]:
        name = latest_bench["name"]
        latest_value = latest_bench["value"]

        # Find corresponding previous benchmark
        prev_bench = next((b for b in previous["benches"] if b["name"] == name), None)
        if prev_bench:
            prev_value = prev_bench["value"]
            ratio = latest_value / prev_value

            if ratio < 0.67:  # 150% degradation = 1/1.5 = 0.67
                print(f"⚠️ REGRESSION: {name} degraded {(1-ratio)*100:.1f}%")
```

## Running Benchmarks Locally

### Python Benchmarks

```bash
# Install dependencies
pip install pytest pytest-benchmark pandas numpy psutil

# Build package
maturin develop --release

# Run all benchmarks
pytest tests/test_performance.py --benchmark-only -v

# Run specific benchmark
pytest tests/test_performance.py::TestThroughputBenchmarks::test_throughput_1m_trades --benchmark-only

# Generate JSON output
pytest tests/test_performance.py --benchmark-only --benchmark-json=output.json
```

### Rust Benchmarks

```bash
# Run Criterion benchmarks
cargo criterion --bench core

# View HTML report
open target/criterion/report/index.html

# Generate JSON output
cargo criterion --bench core --message-format=json
```

## Benchmark Methodology

### Trade Data Generation

All benchmarks use **synthetic trade data** with realistic properties:

```python
def generate_trades(count: int, base_price: float = 42000.0):
    return [
        {
            "timestamp": 1704067200000 + i * 1000,  # 1 trade/sec
            "price": base_price + i * 1.0,          # Linear price increase
            "quantity": 1.0
        }
        for i in range(count)
    ]
```

**Rationale**:
- **Deterministic**: Same input → same output
- **Reproducible**: No external data dependencies
- **Fast**: No I/O overhead
- **Realistic**: Mimics Binance aggTrades format

### Benchmark Configuration

**pytest-benchmark**:
- **Warmup**: Enabled (100,000 iterations)
- **Min Rounds**: 5 (daily), 10 (weekly)
- **Timer**: `time.perf_counter()` (high-resolution)
- **GC**: Enabled (matches real-world usage)

**Criterion.rs**:
- **Statistical**: 100 samples, outlier detection
- **Warmup**: Automatic
- **Measurement**: Wall-clock time
- **Confidence**: 95%

### Why These Metrics?

**Throughput** (trades/sec):
- **Target**: >1M trades/sec for 1M dataset
- **Rationale**: Backtesting requires processing millions of trades
- **Baseline**: Binance provides ~100K aggTrades/day for BTC/USDT

**Memory** (<350MB for 1M trades):
- **Rationale**: Typical systems have 8-16GB RAM
- **Target**: Support multiple symbols in memory
- **Overhead**: Python + Rust + range bars should fit in <1GB

**Compression** (ratio varies by threshold):
- **Rationale**: Higher compression = fewer bars = faster backtesting
- **Trade-off**: Lower threshold = more bars but better price resolution

## Historical Context

### v0.3.0 Baseline (2025-11-23)

First automated benchmark run:

| Metric | Value | Unit |
|--------|-------|------|
| 1K trades | 596.51 | iter/sec |
| 100K trades | 20.78 | iter/sec |
| 1M trades | 2.08 | iter/sec |

**Observations**:
- All targets exceeded ✅
- Throughput scales sub-linearly (expected due to memory allocation)
- Suitable for backtesting workflows

## Non-Goals

This monitoring system does **NOT**:

- ❌ Block pull requests (non-blocking trend monitoring)
- ❌ Measure production latency (backtesting use case only)
- ❌ Profile multi-threaded performance (single-threaded benchmarks)
- ❌ Test real Binance data (uses synthetic trades)
- ❌ Measure cross-platform performance (Linux only in CI)

## Troubleshooting

### Dashboard Not Updating

1. Check workflow status: https://github.com/terrylica/rangebar-py/actions
2. Verify GitHub Pages enabled: Settings → Pages → Source: "GitHub Actions"
3. Check gh-pages branch: `git fetch origin gh-pages`

### Benchmark Failures

1. Check test logs: `pytest tests/test_performance.py -v`
2. Verify dependencies: `pip list | grep -E "pytest|benchmark|psutil"`
3. Rebuild package: `maturin develop --release`

### Parsing Errors (AI Agents)

Common issues:
- **404 on data.js**: Workflow hasn't run yet (wait 24h or trigger manually)
- **JSON parse error**: Remove JavaScript wrapper (`window.BENCHMARK_DATA = `)
- **Missing metrics**: Check `entries["Python API Benchmarks"]` key exists

## References

- **GitHub Actions**: `.github/workflows/performance-daily.yml`
- **Python Tests**: `tests/test_performance.py`
- **Rust Benchmarks**: `benches/core.rs` (deferred to Phase 2)
- **Dashboard Code**: `gh-pages` branch
- **Tool**: [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark)
