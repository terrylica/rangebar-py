# Performance Monitoring

**Dashboard**: https://terrylica.github.io/rangebar-py/
**ADR**: https://github.com/terrylica/rangebar-py/blob/main/docs/decisions/0007-daily-performance-monitoring.md

## Overview

Automated daily performance benchmarks provide observable metrics for system viability assessment.

## Schedule

| Frequency | Time (UTC) | Configuration |
|-----------|------------|---------------|
| Daily | 03:17 | 5 iterations |
| Weekly | Sunday 02:00 | 10 iterations |
| Manual | On-demand | Via GitHub Actions |

## Data Sources

### Machine-Readable (AI Agents)

**URL**: `https://terrylica.github.io/rangebar-py/dev/bench/data.js`

**Format**: JavaScript object (`window.BENCHMARK_DATA`)

```javascript
{
  "lastUpdate": <unix_timestamp_ms>,
  "repoUrl": "https://github.com/terrylica/rangebar-py",
  "entries": {
    "Python API Benchmarks": [
      {
        "commit": { "id": "...", "timestamp": "..." },
        "benches": [
          { "name": "...", "value": <number>, "unit": "iter/sec" }
        ]
      }
    ]
  }
}
```

### Human-Readable

**Dashboard**: https://terrylica.github.io/rangebar-py/

- Historical trend charts
- Throughput visualization
- Memory usage tracking

## Metrics

### Throughput (iter/sec)

| Benchmark | Dataset Size | Target | Description |
|-----------|--------------|--------|-------------|
| `test_throughput_1k_trades` | 1,000 | >100 | Small dataset |
| `test_throughput_100k_trades` | 100,000 | >10 | Medium dataset |
| `test_throughput_1m_trades` | 1,000,000 | >1 | Large dataset |

**Interpretation**: `iter/sec` = iterations per second. Each iteration processes N trades into range bars.

### Memory

| Benchmark | Target | Description |
|-----------|--------|-------------|
| `test_memory_1m_trades` | <350 MB | Peak RSS for 1M trades |

### Compression

| Benchmark | Threshold |
|-----------|-----------|
| `test_compression_ratio[100]` | 100 bps |
| `test_compression_ratio[250]` | 250 bps |
| `test_compression_ratio[500]` | 500 bps |
| `test_compression_ratio[1000]` | 1000 bps |

**Interpretation**: Compression ratio = input trades / output bars. Higher = more aggregation.

## AI Agent Usage

### Viability Check

```python
import requests
import json
import re

url = "https://terrylica.github.io/rangebar-py/dev/bench/data.js"
response = requests.get(url)

# Extract JSON from JavaScript wrapper
json_str = re.sub(r'^\s*window\.BENCHMARK_DATA\s*=\s*', '', response.text)
json_str = re.sub(r'\s*;?\s*$', '', json_str)
data = json.loads(json_str)

# Get latest benchmark
latest = data["entries"]["Python API Benchmarks"][-1]
benches = {b["name"].split("::")[-1]: b["value"] for b in latest["benches"]}

# Validate targets
TARGETS = {
    "test_throughput_1k_trades": 100,
    "test_throughput_100k_trades": 10,
    "test_throughput_1m_trades": 1,
}

for name, target in TARGETS.items():
    actual = benches.get(name, 0)
    status = "PASS" if actual > target else "FAIL"
    print(f"{name}: {actual:.2f} iter/sec (target: >{target}) [{status}]")
```

### Regression Detection

```python
entries = data["entries"]["Python API Benchmarks"]
REGRESSION_THRESHOLD = 0.67  # 150% degradation

if len(entries) >= 2:
    latest, previous = entries[-1], entries[-2]

    for bench in latest["benches"]:
        name = bench["name"]
        prev = next((b for b in previous["benches"] if b["name"] == name), None)
        if prev:
            ratio = bench["value"] / prev["value"]
            if ratio < REGRESSION_THRESHOLD:
                print(f"REGRESSION: {name} ({ratio:.2%})")
```

## Local Execution

### Python

```bash
pip install pytest pytest-benchmark pandas numpy psutil
maturin develop --release
pytest tests/test_performance.py --benchmark-only -v
```

### Rust

```bash
cargo criterion --bench core
```

## Methodology

### Synthetic Trade Data

```python
def generate_trades(count: int, base_price: float = 42000.0):
    return [
        {
            "timestamp": 1704067200000 + i * 1000,
            "price": base_price + i * 1.0,
            "quantity": 1.0
        }
        for i in range(count)
    ]
```

Properties: deterministic, reproducible, no I/O overhead.

### Configuration

**pytest-benchmark**: warmup enabled, min rounds configurable, `time.perf_counter()` timer

**Criterion.rs**: 100 samples, outlier detection, 95% confidence

## Scope

**Included**:
- Throughput measurement
- Memory profiling
- Compression ratio analysis
- Trend monitoring

**Excluded**:
- Pull request blocking (non-blocking monitoring)
- Production latency measurement
- Multi-threaded benchmarks
- Real exchange data (synthetic only)
- Cross-platform comparison (CI runs on Linux)

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| Dashboard not updating | Check workflow status at https://github.com/terrylica/rangebar-py/actions |
| 404 on data.js | Wait for daily run or trigger manually |
| JSON parse error | Remove `window.BENCHMARK_DATA = ` wrapper |

## References

| Resource | Location |
|----------|----------|
| Daily workflow | `.github/workflows/performance-daily.yml` |
| Python benchmarks | `tests/test_performance.py` |
| Rust benchmarks | `benches/core.rs` |
| Dashboard source | `gh-pages` branch |
| Tool | https://github.com/benchmark-action/github-action-benchmark |
