# ADR-007: Daily Performance Monitoring on CI/CD

**Status**: Accepted

**Date**: 2025-11-22

**Deciders**: Terry Li

**Tags**: `performance`, `cicd`, `observability`, `github-actions`, `benchmarks`

---

## Context

The rangebar-py project requires **tangible observable performance metrics** on a daily basis to provide clear visibility into:

1. **System viability**: Whether performance targets are met (>1M trades/sec, <350MB memory)
2. **Trend analysis**: Long-term performance evolution over commits
3. **AI agent observability**: Machine-readable performance data for autonomous decision-making
4. **Human dashboard**: Visual performance trends for developers

**Current State**:

- ✅ pytest-benchmark configured in pyproject.toml
- ✅ Performance tests implemented in `tests/test_performance.py` (8 metrics)
- ❌ No CI/CD performance tracking
- ❌ No historical performance data
- ❌ No visualization dashboard
- ❌ No automated regression detection

**Constraints**:

- **Non-blocking**: Must NOT block development workflow (aligns with ADR-0007 GitHub Actions philosophy)
- **Daily execution**: Benchmarks run automatically every day
- **Observable artifacts**: Generate both machine-readable (JSON) and human-readable (HTML) files
- **Dual-layer tracking**: Python API (pytest-benchmark) + Rust core (Criterion.rs)
- **Off-peak scheduling**: Avoid GitHub Actions congestion
- **Zero cost**: Use GitHub Actions free tier + GitHub Pages hosting

**Options Evaluated**:

1. **github-action-benchmark** (OSS, GitHub Pages hosting)
2. **CodSpeed** (commercial SaaS, differential flamegraphs)
3. **Bencher.dev** (OSS, self-hostable statistical platform)
4. **Custom JSON storage** (artifact retention, manual parsing)

---

## Decision

**Use github-action-benchmark for automated daily performance monitoring with GitHub Pages hosting.**

### Benchmark Execution

**Python API Layer**:

- Tool: pytest-benchmark
- Tests: `tests/test_performance.py` (existing 8 metrics)
- Schedule: Daily cron (3:17 AM UTC, off-peak)
- Output: `python-bench.json` (machine-readable)

**Rust Core Layer**:

- Tool: Criterion.rs
- Benchmarks: `benches/core.rs` (new)
- Schedule: Daily cron (same workflow)
- Output: `rust-bench.json` (machine-readable)

### Visualization & Storage

**Platform**: github-action-benchmark GitHub Action

**Storage**: GitHub Pages (gh-pages branch)

- Unlimited retention (GitHub Pages has no expiry)
- 1GB storage limit (far exceeds our needs)
- Auto-deployment on every benchmark run

**Dashboard**: HTML with Chart.js trend graphs

- URL: `https://terrylica.github.io/rangebar-py/`
- Accessible to both humans and AI agents
- Historical trend visualization

### Deployment Method

**Decision**: Use GitHub Actions as deployment source (not "Deploy from a branch")

**GitHub Pages Configuration**:

- Source: **GitHub Actions**
- Deployment: via `actions/deploy-pages@v4` in workflow
- Artifacts: Uploaded via `actions/upload-pages-artifact@v3`

**Rationale**:

- ✅ **Modern approach**: Recommended by GitHub for CI/CD pipelines
- ✅ **Explicit control**: Deployment step visible in workflow
- ✅ **Better observability**: Deployment logs in Actions tab
- ✅ **Consistent with CI/CD**: All automation in workflows (no branch-based auto-deploy)
- ✅ **Environment protection**: Can add deployment protection rules (optional)

**Alternative Rejected**: Deploy from a branch

- ❌ Less explicit (auto-deploys on push to gh-pages)
- ❌ Deployment not visible in workflow logs
- ✅ Simpler for github-action-benchmark (designed for branch-based)
- ✅ Well-tested and documented

**Implementation**:

```yaml
permissions:
  contents: write # github-action-benchmark pushes data
  pages: write # Deploy to Pages
  id-token: write # OIDC token for Pages deployment

jobs:
  benchmark:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
      # ... benchmark steps ...

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: . # github-action-benchmark stores files in repo root

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Scheduling Strategy

**Daily Benchmarks**:

```yaml
schedule:
  - cron: "17 3 * * *" # 3:17 AM UTC (off-peak)
```

**Manual Trigger**:

```yaml
workflow_dispatch: # Allow on-demand execution
```

**Weekly Comprehensive** (optional enhancement):

```yaml
schedule:
  - cron: "0 2 * * 0" # 2:00 AM UTC Sunday
```

### Regression Detection

**Alert Threshold**: 150% (non-blocking warnings)

**Method**: Threshold-based regression detection

- Compare current run vs previous run
- Alert if throughput drops >50% OR memory increases >50%
- **Non-blocking**: Warnings only, never fail workflow

**Rationale**:

- ✅ Trend monitoring (aligns with user requirement)
- ✅ Early warning system (detect gradual degradation)
- ❌ NOT gating (no PR blocking, no CI failures)

### Error Handling

**Policy**: Fail-fast, no fallback benchmarks

**Error Scenarios**:

| Error                       | Action                     | No Fallback                     |
| --------------------------- | -------------------------- | ------------------------------- |
| Benchmark execution failure | Fail workflow, raise error | No "skip failed benchmarks"     |
| JSON generation failure     | Fail workflow, raise error | No "use previous run's JSON"    |
| GitHub Pages deploy failure | Fail workflow, raise error | No "store in artifacts instead" |
| Regression detection error  | Fail workflow, raise error | No "disable alerts on error"    |

**No Fallbacks**:

- ❌ No "use cached results if benchmarks fail"
- ❌ No "skip visualization if dashboard broken"
- ❌ No "retry with relaxed thresholds"

---

## Consequences

### Positive

**Observability**:

- Daily performance snapshots (never miss a regression)
- Machine-readable JSON (AI agents can parse trends)
- Human-readable dashboard (developers see visual trends)
- Historical data (unlimited retention on GitHub Pages)

**Availability**:

- Non-blocking execution (never blocks PRs or releases)
- Off-peak scheduling (minimal GitHub Actions congestion)
- Manual trigger available (on-demand benchmarking)

**Correctness**:

- Dual-layer metrics (Python API + Rust core, isolates PyO3 overhead)
- Statistical rigor (pytest-benchmark + Criterion.rs de facto standards)
- Automated regression detection (150% threshold)

**Maintainability**:

- Zero infrastructure (GitHub Actions + GitHub Pages built-in)
- OSS tooling (github-action-benchmark, Criterion.rs)
- No manual data export (auto-commit to gh-pages)

### Negative

**Initial Setup**:

- Requires GitHub Pages configuration
- Requires Rust benchmark creation (benches/core.rs)
- Requires workflow creation (2 files: daily + weekly)

**GitHub Actions Quota**:

- Daily benchmarks consume ~5 minutes/day (negligible vs 2000 min/month free tier)
- Weekly benchmarks consume ~15 minutes/week

**Storage**:

- GitHub Pages dashboard grows over time (estimated <10MB/year)

### Neutral

**Metrics Tracked** (Phase 1):

**Python API (8 existing metrics)**:

- Throughput: 1K, 100K, 1M trades/sec
- Memory: Peak usage for 1M trades
- Compression: Ratio for 100/250/500/1000 bps

**Rust Core (6 new metrics)**:

- Throughput: 1M trades/sec (Rust-only)
- Latency: P50, P95, P99
- Memory: Rust allocations only
- PyO3 overhead: Python throughput - Rust throughput

**Future Enhancements** (Post-Phase 1):

- Iai-Callgrind deterministic benchmarks (instruction counts)
- Multi-symbol batch processing benchmarks
- Backtesting.py integration benchmarks
- Flamegraph generation (cargo-flamegraph)

---

## Implementation

### Phase 1: Daily Python Benchmarks (Existing Metrics)

**File**: `.github/workflows/performance-daily.yml`

```yaml
name: Daily Performance Benchmarks

on:
  schedule:
    - cron: "17 3 * * *" # 3:17 AM UTC
  workflow_dispatch:

jobs:
  benchmark-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install maturin pytest pytest-benchmark
      - run: maturin develop --release
      - run: pytest tests/test_performance.py --benchmark-json=python-bench.json
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: "pytest"
          output-file-path: python-bench.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: "150%"
          comment-on-alert: false
          fail-on-alert: false
```

### Phase 2: Rust Core Benchmarks

**File**: `benches/core.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rangebar_core::{RangeBarProcessor, AggTrade};

fn bench_process_1m_trades(c: &mut Criterion) {
    let trades: Vec<AggTrade> = (0..1_000_000)
        .map(|i| AggTrade {
            timestamp_ms: i * 100,
            price: 42000.0 + (i % 100) as f64,
            quantity: 0.01,
        })
        .collect();

    c.bench_function("process_1m_trades", |b| {
        let mut processor = RangeBarProcessor::new(250).unwrap();
        b.iter(|| processor.process(black_box(&trades)))
    });
}

criterion_group!(benches, bench_process_1m_trades);
criterion_main!(benches);
```

**File**: `Cargo.toml` (add benchmarks section)

```toml
[[bench]]
name = "core"
harness = false

[dev-dependencies]
criterion = "0.5"
```

### Phase 3: GitHub Pages Configuration

**Manual Steps** (one-time):

1. Navigate to: https://github.com/terrylica/rangebar-py/settings/pages
2. Source: Deploy from a branch
3. Branch: `gh-pages`, folder: `/ (root)`
4. Click "Save"
5. Verify: Dashboard accessible at `https://terrylica.github.io/rangebar-py/`

### Phase 4: Validation

**Validation Checks**:

```bash
# Trigger manual workflow run
gh workflow run performance-daily.yml

# Check workflow status
gh run list --workflow=performance-daily.yml

# Verify JSON output
git checkout gh-pages
cat dev/bench/python-bench.json | jq '.benchmarks[].stats.mean'

# Verify dashboard
open https://terrylica.github.io/rangebar-py/
```

---

## Compliance

### SLO Metrics

**Availability**:

- Target: 100% daily benchmark execution (no missed days)
- Measurement: GitHub Actions workflow run history
- Acceptable: 99% uptime (3 missed days/year due to GitHub outages)

**Correctness**:

- Target: 100% benchmark execution success rate
- Measurement: Workflow success/failure ratio
- Alert: If >1 failure/week, investigate test flakiness

**Observability**:

- Target: All benchmarks visible in dashboard within 5 minutes
- Measurement: Time from workflow completion to gh-pages update
- Artifacts:
  - `python-bench.json` (machine-readable)
  - `rust-bench.json` (machine-readable)
  - HTML dashboard (human-readable)

**Maintainability**:

- Target: Zero manual intervention for daily benchmarks
- Measurement: Number of manual workflow triggers
- Acceptable: <1 manual trigger/month (emergency only)

### Error Handling

**Policy**: Raise and propagate; no fallback data sources

**Error Scenarios**:

| Error                      | Cause                   | Action                               |
| -------------------------- | ----------------------- | ------------------------------------ |
| Benchmark timeout          | Infinite loop in code   | Fail workflow, raise error           |
| JSON parse error           | Malformed pytest output | Fail workflow, raise error           |
| GitHub Pages deploy failed | gh-pages branch missing | Fail workflow, raise error           |
| Regression alert triggered | Performance degraded    | Log warning, continue (non-blocking) |

**No Fallbacks**:

- ❌ No "use previous day's data if today fails"
- ❌ No "skip regression check if threshold calculation errors"
- ❌ No "disable alerts if too many false positives"

---

## References

**github-action-benchmark**:

- Repo: https://github.com/benchmark-action/github-action-benchmark
- Docs: https://github.com/benchmark-action/github-action-benchmark#usage

**pytest-benchmark**:

- Docs: https://pytest-benchmark.readthedocs.io/
- JSON Format: https://pytest-benchmark.readthedocs.io/en/latest/comparing.html

**Criterion.rs**:

- Docs: https://bheisler.github.io/criterion.rs/book/
- JSON Output: https://bheisler.github.io/criterion.rs/book/user_guide/csv_output.html

**GitHub Actions Scheduling**:

- Cron Syntax: https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule
- Best Practices: https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#scheduled-events

---

## Alternatives Considered

### CodSpeed (Commercial SaaS)

**Rejected**:

- ❌ Commercial (requires paid plan)
- ❌ External dependency (not GitHub-native)
- ✅ Superior differential flamegraphs (<1% variance)
- ✅ Advanced statistical analysis

**Verdict**: Overkill for current needs; revisit if free tier insufficient.

### Bencher.dev (OSS Statistical Platform)

**Rejected**:

- ❌ Requires self-hosting (additional infrastructure)
- ❌ More complex setup (PostgreSQL database)
- ✅ Advanced change point detection (reduces false positives)
- ✅ Self-hostable (no vendor lock-in)

**Verdict**: Too complex for MVP; revisit if advanced statistics needed.

### Custom JSON Storage in Artifacts

**Rejected**:

- ❌ Artifact retention limit (90-400 days)
- ❌ Manual parsing required (no dashboard)
- ❌ No visualization (AI agents only)
- ✅ Simple implementation
- ✅ Full control over format

**Verdict**: Lacks human-readable dashboard; insufficient observability.

### Iai-Callgrind (Deterministic Benchmarks)

**Deferred** (Post-MVP):

- ✅ Deterministic (instruction counts, not wall-time)
- ✅ Eliminates noise (no statistical variance)
- ❌ Requires Valgrind (not available on macOS)
- ❌ Slower execution (10x overhead)

**Verdict**: Valuable for CI regression checks; add in Phase 2.

---

## Summary

**Zero-infrastructure daily performance monitoring using GitHub Actions + GitHub Pages.**

**Execution Flow**:

1. GitHub Actions triggers daily cron at 3:17 AM UTC
2. Workflow runs pytest-benchmark + Criterion.rs benchmarks
3. Benchmarks generate JSON files (python-bench.json, rust-bench.json)
4. github-action-benchmark parses JSON and updates gh-pages branch
5. GitHub Pages auto-deploys dashboard at https://terrylica.github.io/rangebar-py/
6. AI agents query JSON; humans view dashboard

**Result**: Daily observable performance metrics with trend visualization, non-blocking execution, and unlimited retention.
