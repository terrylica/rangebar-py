# Daily Performance Monitoring - Implementation Plan

**ADR ID**: 0007

**Status**: In Progress

**Last Updated**: 2025-11-22 22:39:29 UTC

---

## (a) Plan

### Objective

Implement automated daily performance monitoring for rangebar-py to generate tangible observable files (machine-readable JSON + human-readable HTML dashboard) providing clear visibility into system viability for both AI agents and human developers.

### Goals

1. **Daily automated benchmarks**: Python API (pytest-benchmark) + Rust core (Criterion.rs)
2. **Observable artifacts**: JSON files for AI agents, HTML dashboard for humans
3. **Non-blocking execution**: Never block development workflow (aligns with ADR-0007)
4. **Historical tracking**: Unlimited retention via GitHub Pages
5. **Regression detection**: Automated 150% threshold alerts (warnings only)

### Non-Goals

1. ❌ **PR-blocking benchmarks**: No CI checks that fail builds
2. ❌ **Real-time benchmarks**: Not for production monitoring
3. ❌ **Advanced statistics**: No change point detection (Phase 1)
4. ❌ **Multi-platform benchmarks**: Linux only (ubuntu-latest)
5. ❌ **Flamegraph generation**: Deferred to Phase 2

### Success Criteria

- [x] ADR-0007 documented and accepted
- [ ] Daily workflow runs successfully at 3:17 AM UTC
- [ ] GitHub Pages dashboard accessible at https://terrylica.github.io/rangebar-py/
- [ ] JSON files committed to gh-pages branch (machine-readable)
- [ ] Rust benchmarks execute without errors
- [ ] Manual workflow trigger works
- [ ] Build validation passes (no known errors)

---

## (b) Context

### Background

**Problem Statement** (from user):

> "How many kinds of performance can we put on the CI/CD on GitHub to create some tangible observable file so that we can have a very clear view for the user as well as AI coding agents to understand what's actually viable on a daily basis."

**Current State**:

- **Existing**: pytest-benchmark configured, 8 metrics in `tests/test_performance.py`
- **Missing**: No CI/CD tracking, no historical data, no dashboard

**Research Findings** (from sub-agents):

- **21 performance metrics cataloged**: 8 implemented, 6 easy additions, 7 require new instrumentation
- **8 Rust benchmarking tools evaluated**: Criterion.rs recommended (de facto standard)
- **3 visualization platforms compared**: github-action-benchmark selected (OSS, zero-cost)
- **Real-world patterns analyzed**: pydantic-core, polars, orjson, tokenizers, cryptography

**User Decisions** (from clarification questions):

1. **Goal**: Trend monitoring (non-blocking)
2. **Scope**: Python API (pytest-benchmark) + Rust core (Criterion.rs)
3. **Platform**: github-action-benchmark
4. **Schedule**: Daily cron (off-peak) + manual trigger + weekly comprehensive

### Architecture

**Execution Flow**:

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions (Daily Cron: 3:17 AM UTC)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Checkout code                                            │
│ 2. Setup Python 3.12                                        │
│ 3. Install: maturin, pytest, pytest-benchmark               │
│ 4. Build: maturin develop --release                         │
│ 5. Run: pytest tests/test_performance.py                    │
│    └─> Output: python-bench.json                            │
│ 6. Run: cargo criterion --bench core                        │
│    └─> Output: target/criterion/.../estimates.json          │
│ 7. Parse JSON with github-action-benchmark                  │
│ 8. Update gh-pages branch (auto-commit)                     │
│ 9. GitHub Pages auto-deploys dashboard                      │
└─────────────────────────────────────────────────────────────┘
           │                               │
           ▼                               ▼
    python-bench.json            HTML Dashboard
    (AI-readable)                (Human-readable)
    gh-pages:/dev/bench/         https://terrylica.github.io/rangebar-py/
```

**Metrics Tracked** (Phase 1):

**Python API (8 existing)**:
- `test_throughput_1k_trades`: 1K trades/sec
- `test_throughput_100k_trades`: 100K trades/sec
- `test_throughput_1m_trades`: 1M trades/sec (target: >1M/sec)
- `test_memory_1m_trades`: Peak memory for 1M trades (target: <350MB)
- `test_compression_ratio[100]`: Compression at 100 bps
- `test_compression_ratio[250]`: Compression at 250 bps
- `test_compression_ratio[500]`: Compression at 500 bps
- `test_compression_ratio[1000]`: Compression at 1000 bps

**Rust Core (6 new)**:
- `bench_process_1m_trades`: Throughput without PyO3 overhead
- `bench_latency_p50`: P50 latency per bar
- `bench_latency_p95`: P95 latency per bar
- `bench_latency_p99`: P99 latency per bar
- `bench_memory_allocations`: Rust-only allocations
- `bench_pyo3_overhead`: Calculated as (Python throughput - Rust throughput)

### Dependencies

**OSS Libraries**:

- `benchmark-action/github-action-benchmark@v1`: Dashboard generation
- `pytest-benchmark`: Python benchmarking framework
- `criterion`: Rust statistical benchmarking

**GitHub Services**:

- GitHub Actions: Workflow execution (free tier: 2000 min/month)
- GitHub Pages: Dashboard hosting (free tier: 1GB storage, unlimited retention)

**Python Dependencies** (test-only):

- `pytest>=7.0`
- `pytest-benchmark>=4.0`
- `maturin>=1.7`

**Rust Dependencies**:

```toml
[dev-dependencies]
criterion = "0.5"
```

### Constraints

**GitHub Actions**:

- Free tier: 2000 minutes/month (daily benchmarks ~150 min/month = 7.5% usage)
- Artifact retention: 90 days (GitHub Pages has unlimited retention)
- Concurrent jobs: 20 (non-issue for scheduled workflows)

**GitHub Pages**:

- Storage limit: 1GB (estimated usage: <10MB/year)
- Bandwidth: 100GB/month soft limit
- Builds: 10 builds/hour (non-issue for daily updates)

**Performance**:

- Benchmark execution time: <5 minutes (Python + Rust)
- Dashboard update latency: <1 minute (gh-pages deployment)

---

## (c) Task List

### Phase 1: Daily Python Benchmarks

- [x] **1.1**: Create ADR-0007 (docs/decisions/0007-daily-performance-monitoring.md)
- [x] **1.2**: Create plan document (docs/plan/0007-daily-performance-monitoring/plan.md)
- [ ] **1.3**: Create workflow file (.github/workflows/performance-daily.yml)
  - Daily cron: `17 3 * * *`
  - Manual trigger: `workflow_dispatch`
  - Python 3.12 setup
  - pytest-benchmark execution
  - JSON output: `python-bench.json`
- [ ] **1.4**: Configure GitHub Pages
  - Enable gh-pages branch
  - Set source: Deploy from branch (gh-pages, root)
  - Verify dashboard URL: https://terrylica.github.io/rangebar-py/
- [ ] **1.5**: Validate workflow execution
  - Trigger manual run: `gh workflow run performance-daily.yml`
  - Check logs: `gh run list --workflow=performance-daily.yml`
  - Verify JSON output in gh-pages branch
  - Verify dashboard renders correctly

### Phase 2: Rust Core Benchmarks

- [ ] **2.1**: Create benches/core.rs
  - `bench_process_1m_trades`: Throughput benchmark
  - `bench_latency_percentiles`: P50/P95/P99 latency
  - `bench_memory_allocations`: Memory profiling
- [ ] **2.2**: Update Cargo.toml
  - Add `[[bench]]` section
  - Add `criterion = "0.5"` to dev-dependencies
- [ ] **2.3**: Test Rust benchmarks locally
  - Run: `cargo criterion --bench core`
  - Verify JSON output: `target/criterion/*/estimates.json`
- [ ] **2.4**: Integrate Rust benchmarks into workflow
  - Add step: `cargo criterion --bench core --message-format=json`
  - Parse Criterion JSON with github-action-benchmark
  - Merge with Python results in dashboard

### Phase 3: Weekly Comprehensive Benchmarks

- [ ] **3.1**: Create workflow file (.github/workflows/performance-weekly.yml)
  - Weekly cron: `0 2 * * 0` (Sunday 2:00 AM UTC)
  - Extended rounds: 10 iterations (vs 5 for daily)
  - Additional metrics: Flamegraphs, instruction counts
- [ ] **3.2**: Document weekly vs daily differences
  - Daily: Fast checks (5 min), 5 iterations
  - Weekly: Comprehensive (15 min), 10 iterations + extras

### Phase 4: Documentation & Validation

- [ ] **4.1**: Update CLAUDE.md
  - Add "Daily Performance Monitoring" section
  - Document observable file locations
  - Explain "daily viability" interpretation
- [ ] **4.2**: Update README.md
  - Add badge: Performance Dashboard link
  - Document benchmark execution instructions
- [ ] **4.3**: Create docs/PERFORMANCE.md
  - Explain benchmark methodology
  - Document metrics and targets
  - Provide AI agent parsing examples
- [ ] **4.4**: Validate build (no known errors)
  - Run: `maturin develop --release`
  - Run: `pytest tests/test_performance.py`
  - Run: `cargo test`
  - Fix any errors immediately

### Phase 5: Release & Publishing

- [ ] **5.1**: Commit all changes with conventional commit
  - `feat: add daily performance monitoring with github-action-benchmark`
- [ ] **5.2**: Use semantic-release skill to create release
  - Conventional commits → version tag → GitHub release → changelog
- [ ] **5.3**: Verify release workflow success
- [ ] **5.4**: (Optional) Use pypi-doppler skill to publish if version bumped

---

## Open Questions

1. ✅ **RESOLVED**: Should we use github-action-benchmark or CodSpeed? → github-action-benchmark (OSS, zero-cost)
2. ✅ **RESOLVED**: Should benchmarks block PRs? → No (non-blocking trend monitoring)
3. ✅ **RESOLVED**: What regression threshold? → 150% (non-blocking warnings)
4. ⏳ **PENDING**: Should we enable GitHub Pages immediately or wait until workflow validated? → Enable immediately (fail-fast if broken)
5. ⏳ **PENDING**: Should we add Iai-Callgrind deterministic benchmarks in Phase 1? → Defer to Phase 2 (not available on macOS)

---

## Validation Plan

### Build Validation

After each file change:

```bash
# Python build
maturin develop --release

# Python tests
pytest tests/test_performance.py -v

# Rust tests
cargo test

# Rust benchmarks (local)
cargo criterion --bench core
```

**Error Handling**: Raise and propagate immediately; do NOT leave known errors unresolved.

### Workflow Validation

After workflow creation:

```bash
# Validate YAML syntax
yamllint .github/workflows/performance-daily.yml

# Trigger manual run
gh workflow run performance-daily.yml

# Monitor execution
gh run watch

# Check logs
gh run view --log

# Verify outputs
git checkout gh-pages
cat dev/bench/python-bench.json | jq '.'
```

### Dashboard Validation

After GitHub Pages enabled:

```bash
# Open dashboard
open https://terrylica.github.io/rangebar-py/

# Verify charts render
# Verify data points present
# Verify trend lines visible
```

---

## Progress Log

**2025-11-22 22:39:29 UTC**: Plan created, starting implementation
**2025-11-22 22:40:00 UTC**: ADR-0007 documented and accepted
**2025-11-22 22:41:00 UTC**: Task 1.1 completed (ADR created)
**2025-11-22 22:42:00 UTC**: Task 1.2 completed (Plan created)
**2025-11-22 22:43:00 UTC**: Starting Task 1.3 (workflow creation)

---

## Notes

- All times in UTC
- Log files: `logs/0007-daily-performance-monitoring-YYYYMMDD_HHMMSS.log`
- Dashboard URL will be added after GitHub Pages configuration
- Rust benchmarks depend on rangebar-core v5.0.0 API stability
