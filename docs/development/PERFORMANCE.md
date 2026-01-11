# Performance Monitoring

**Parent**: [/CLAUDE.md](/CLAUDE.md) | **Decision**: ADR-0007

Automated daily benchmarks track performance metrics over time.

---

## Overview

Performance monitoring runs on GitHub Actions, generating observable files for:
- **Humans**: HTML dashboard with Chart.js visualization
- **AI agents**: JSON files for automated analysis

**Philosophy** (ADR-0007): Trend monitoring, not gating. Never blocks development.

---

## Observable Files

### Machine-Readable (AI Agents)

- **Location**: `gh-pages:/dev/bench/python-bench.json`
- **Format**: pytest-benchmark output
- **Retention**: Unlimited historical data

```bash
# Query benchmark results
git fetch origin gh-pages
git checkout gh-pages
cat dev/bench/python-bench.json | jq '.benchmarks[] | {name: .name, mean: .stats.mean}'
```

### Human-Readable (Developers)

- **Dashboard**: https://terrylica.github.io/rangebar-py/
- **Visualization**: Chart.js trend charts
- **Regression**: 150% threshold (non-blocking warnings)

---

## Viability Metrics

### Python API Layer (pytest-benchmark)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput 1K trades | >1K/sec | `test_throughput_1k_trades` |
| Throughput 100K trades | >100K/sec | `test_throughput_100k_trades` |
| Throughput 1M trades | >1M/sec | `test_throughput_1m_trades` |
| Memory 1M trades | <350MB | `test_memory_1m_trades` |
| Compression ratio | >0 bars | `test_compression_ratio` |

### Rust Core Layer (Criterion.rs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | >1M trades/sec | `rangebar_bench` |
| Latency P50 | <1ms | Criterion output |
| Latency P99 | <10ms | Criterion output |
| PyO3 overhead | <20% | Python - Rust delta |

---

## Execution Schedule

| Schedule | Time | Scope |
|----------|------|-------|
| Daily | 3:17 AM UTC | Standard benchmarks |
| Weekly | Sunday 2:00 AM UTC | Extended rounds |
| Manual | On-demand | `gh workflow run performance-daily.yml` |

---

## Viability Criteria (AI Agents)

When checking performance viability:

```python
# Thresholds
THROUGHPUT_1M_MAX_SECONDS = 1.0  # >1M trades/sec
MEMORY_1M_MAX_MB = 350
REGRESSION_THRESHOLD = 1.5  # 50% degradation

# Check
if benchmark["test_throughput_1m_trades"]["mean"] > THROUGHPUT_1M_MAX_SECONDS:
    warn("Throughput regression detected")

if benchmark["test_memory_1m_trades"]["peak_memory"] > MEMORY_1M_MAX_MB * 1e6:
    warn("Memory regression detected")
```

---

## Implementation

### Workflows

- `.github/workflows/performance-daily.yml` - Daily benchmarks
- `.github/workflows/performance-weekly.yml` - Weekly comprehensive

### Benchmarks

- `benches/core.rs` - Rust Criterion benchmarks
- `tests/test_performance.py` - Python pytest-benchmark

### Visualization

- github-action-benchmark generates HTML/JSON
- Deployed via `actions/deploy-pages@v4`

---

## Local Benchmarking

```bash
# Run Rust benchmarks
mise run bench:run

# Quick validation
mise run bench:validate  # Verifies 1M ticks < 100ms

# Create baseline
mise run bench:baseline

# Compare against baseline
mise run bench:compare
```

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/.mise.toml](/.mise.toml) - Benchmark tasks
- ADR-0007 - Daily Performance Monitoring decision
