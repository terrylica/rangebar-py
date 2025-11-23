# Next Steps: Complete Performance Dashboard Deployment

**Status**: Phase 1.5 code-complete, manual steps required

**Commits ready**: 3 unpushed commits (4dc2fc2, c467aa5, 6eaaa93)

---

## Quick Start (3 Steps, ~12 minutes)

### Step 1: Push Commits to GitHub (~1 minute)

```bash
# If you have SSH configured:
git remote set-url origin git@github.com:terrylica/rangebar-py.git
git push origin main

# Or using HTTPS (will prompt for credentials):
git push origin main
```

**What this does**: Pushes 3 commits with performance monitoring implementation.

---

### Step 2: Configure GitHub Pages (~30 seconds)

1. Navigate to: https://github.com/terrylica/rangebar-py/settings/pages
2. Under "Build and deployment":
   - **Source**: Select **"GitHub Actions"** (dropdown)
3. Click **"Save"**

**What this does**: Enables GitHub Actions to deploy the performance dashboard.

---

### Step 3: Run Validation Script (~10 minutes)

```bash
./scripts/validate-performance-deployment.sh
```

**What this does**:

- Triggers the performance-daily.yml workflow
- Monitors execution in real-time
- Verifies deployment success
- Displays dashboard URL

**Expected output**:

```
✓ Workflow completed successfully
✓ Deployment verification complete

Dashboard URL: https://terrylica.github.io/rangebar-py/
```

---

## What Was Implemented

### Automated Daily Benchmarks

**Python API Layer** (pytest-benchmark):

- Throughput: 1K, 100K, 1M trades/sec (target: >1M/sec)
- Memory: Peak usage for 1M trades (target: <350MB)
- Compression: Ratio for 100/250/500/1000 bps

**Rust Core Layer** (Criterion.rs):

- Benchmarks ready in `benches/core.rs`
- Throughput without PyO3 overhead
- Latency percentiles (P50/P95/P99)

**Schedule**:

- Daily: 3:17 AM UTC (automated)
- Weekly: Sunday 2:00 AM UTC (comprehensive)
- Manual: `gh workflow run performance-daily.yml`

### Observable Files Generated

**Machine-Readable** (AI agents):

- JSON: `gh-pages:/dev/bench/python-bench.json`
- Unlimited retention on GitHub Pages

**Human-Readable** (developers):

- Dashboard: https://terrylica.github.io/rangebar-py/
- Chart.js trend visualization
- 150% regression threshold (non-blocking warnings)

### GitHub Actions Deployment

**Why GitHub Actions source (vs. Deploy from branch)**:

- ✅ Explicit deployment control (visible in workflow)
- ✅ Better observability (deployment logs in Actions tab)
- ✅ Modern approach recommended by GitHub
- ✅ Consistent with CI/CD best practices

**Workflow changes**:

- Added permissions: `pages: write`, `id-token: write`
- Added environment: `github-pages`
- Added deployment steps:
  - Checkout gh-pages branch
  - Upload Pages artifact
  - Deploy to GitHub Pages

---

## Optional: Release & Publish (~8 minutes)

### Create v0.4.0 Release (semantic-release)

```bash
# Use the semantic-release skill
# This will:
# - Analyze commits (feat: = MINOR bump)
# - Create v0.4.0 tag
# - Generate changelog
# - Create GitHub release
```

**Commits included**:

- `feat: add daily performance monitoring` (4dc2fc2)
- `feat(perf-monitoring): switch to GitHub Actions deployment` (c467aa5)
- `chore(perf-monitoring): add validation script` (6eaaa93)

### Publish to PyPI (pypi-doppler)

```bash
# Use the pypi-doppler skill
# This will:
# - Build wheels with maturin
# - Publish to PyPI using Doppler credentials
```

---

## Verification Checklist

After completing Steps 1-3:

- [ ] Commits pushed to GitHub (check: https://github.com/terrylica/rangebar-py/commits/main)
- [ ] GitHub Pages source set to "GitHub Actions" (check: Settings → Pages)
- [ ] Workflow executed successfully (check: Actions tab)
- [ ] Dashboard accessible at https://terrylica.github.io/rangebar-py/
- [ ] Benchmark graphs render correctly
- [ ] JSON files accessible (e.g., `/dev/bench/python-bench.json`)
- [ ] Deployment appears in Actions tab under "pages build and deployment"

---

## Troubleshooting

### Push fails with authentication error

```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:terrylica/rangebar-py.git
git push origin main

# Or authenticate gh CLI
gh auth login
gh repo sync
```

### Workflow not found

**Cause**: Commits not pushed yet

**Fix**: Complete Step 1 (push commits)

### Workflow fails

**Check logs**:

```bash
gh run list --workflow=performance-daily.yml
gh run view --log
```

**Common issues**:

1. GitHub Pages not configured to "GitHub Actions" source
2. Missing permissions (check workflow has `pages: write`)
3. gh-pages branch doesn't exist (first run creates it)

**See**: `docs/GITHUB_PAGES_SETUP.md` for detailed troubleshooting

---

## Documentation

**ADR**: `docs/decisions/0007-daily-performance-monitoring.md`

- Decision: GitHub Actions deployment
- Rationale: Modern, explicit, observable

**Plan**: `docs/plan/0007-daily-performance-monitoring/plan.md`

- Phase 1: COMPLETED
- Phase 1.5: IN PROGRESS (4/6 tasks done, 2 manual steps pending)

**Setup Guide**: `docs/GITHUB_PAGES_SETUP.md`

- GitHub Pages configuration
- Troubleshooting guide
- Verification checklist

**CLAUDE.md**: Daily Performance Monitoring section

- Observable files locations
- AI agent interpretation guide
- Deployment method documented

---

## Questions?

**Deployment Issues**: See `docs/GITHUB_PAGES_SETUP.md`

**Workflow Issues**: See `scripts/validate-performance-deployment.sh` output

**Implementation Details**: See `logs/0007-*.log` files

**ADR Rationale**: See `docs/decisions/0007-daily-performance-monitoring.md`
