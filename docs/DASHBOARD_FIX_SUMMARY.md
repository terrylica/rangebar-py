# Performance Dashboard Fix Summary

## Issue Reported

**User**: "Dashboard: https://terrylica.github.io/rangebar-py/ link doesn't work: https://terrylica.github.io/rangebar-py/dev/bench/python-bench.json"

**Status**: ✅ **RESOLVED** (2025-11-23 00:40 UTC)

## Root Cause

### Data Source Mismatch

The dashboard HTML (`index.html`) was fetching the wrong data file:

**Expected** (by dashboard):
```javascript
fetch('dev/bench/python-bench.json')  // ❌ File doesn't exist
```

**Actual** (created by github-action-benchmark):
```javascript
window.BENCHMARK_DATA = {...}  // Created in dev/bench/data.js
```

### Chart Logic Error

The `createCharts()` function expected an object with benchmark names as keys, but github-action-benchmark provides an array of runs:

**Expected** (incorrect assumption):
```javascript
{
  "test_throughput_1k_trades": [{...}, {...}],
  "test_throughput_100k_trades": [{...}, {...}]
}
```

**Actual** (github-action-benchmark format):
```javascript
[
  {commit: {...}, benches: [{name: "test_throughput_1k_trades", ...}]},
  {commit: {...}, benches: [{name: "test_throughput_100k_trades", ...}]}
]
```

## Solution

### Fix Applied to index.html (gh-pages branch)

**Commit**: [2f04a2f](https://github.com/terrylica/rangebar-py/commit/2f04a2f)

**Changes**:

1. **Updated data fetching** (lines 97-112):
   ```javascript
   // BEFORE
   fetch('dev/bench/python-bench.json')
       .then(response => response.json())

   // AFTER
   fetch('dev/bench/data.js')
       .then(response => response.text())
       .then(jsContent => {
           const jsonMatch = jsContent.match(/window\.BENCHMARK_DATA\s*=\s*({[\s\S]*})/);
           const data = JSON.parse(jsonMatch[1]);
           return data;
       })
   ```

2. **Fixed chart creation** (lines 172-254):
   ```javascript
   // Extract unique benchmark names from latest run
   const latestRun = benchmarks[benchmarks.length - 1];
   const benchNames = latestRun.benches.map(b => b.name);

   // Map historical data points for each benchmark
   const throughputData = throughputNames.map(benchName => {
       const dataPoints = benchmarks.map(run => {
           const bench = run.benches.find(b => b.name === benchName);
           return { x: new Date(run.commit.timestamp), y: bench.value };
       });
       return { label: benchName, data: dataPoints };
   });
   ```

### Deployment Process

**Critical Understanding**: GitHub Pages uses workflow-based deployment, NOT automatic branch deployment.

**Configuration** (verified via GitHub API):
```json
{
  "build_type": "workflow",
  "source": {
    "branch": "main",
    "path": "/"
  }
}
```

**Deployment Workflow** (`.github/workflows/performance-daily.yml`):
1. Runs Python benchmarks
2. github-action-benchmark creates `dev/bench/data.js`
3. Checks out gh-pages branch
4. Uploads as Pages artifact
5. Deploys to GitHub Pages

**Key Point**: Manual changes to gh-pages branch require triggering the workflow to deploy.

### Deployment Execution

**Command**:
```bash
gh workflow run performance-daily.yml
```

**Workflow Run**: [19608541389](https://github.com/terrylica/rangebar-py/actions/runs/19608541389)
- Duration: 1m11s
- Result: ✅ SUCCESS

## Verification

### Non-Interactive Validation

**Script**: `scripts/validate-dashboard-content.py`

**Results**: 28/28 checks passed

```bash
$ python3 scripts/validate-dashboard-content.py
================================================================================
Performance Dashboard Validation Suite
================================================================================

Validating Main Dashboard
--------------------------------------------------------------------------------
  ✓ Fetched https://terrylica.github.io/rangebar-py/
  ✓ Found Page title
  ✓ Found Chart.js library reference
  ✓ Found Dashboard heading
  ✓ Found Throughput chart section
  ✓ Found Memory chart section
  ✓ Found Benchmark data reference

Validating Benchmark Data
--------------------------------------------------------------------------------
  ✓ Fetched https://terrylica.github.io/rangebar-py/dev/bench/data.js
  ✓ Parsed benchmark data JSON
  ...

✅ ALL VALIDATIONS PASSED
```

### Interactive Validation (Playwright)

**Script**: `scripts/test-dashboard-interactive.py`

**Tests Passed**:
- ✓ Dashboard loads (HTTP 200)
- ✓ Page title correct
- ✓ Main heading present
- ✓ "View Raw Data" link found and clickable
- ✓ "Detailed Charts" link found
- ✓ Chart canvases present
- ✓ Responsive design (mobile/tablet/desktop)

### Manual Verification

**Dashboard URL**: https://terrylica.github.io/rangebar-py/

**Data Source**: https://terrylica.github.io/rangebar-py/dev/bench/data.js

**HTTP Status**:
```bash
$ curl -I https://terrylica.github.io/rangebar-py/
HTTP/2 200

$ curl -I https://terrylica.github.io/rangebar-py/dev/bench/data.js
HTTP/2 200
```

**Deployed Code**:
```bash
$ curl -s https://terrylica.github.io/rangebar-py/ | grep "fetch("
fetch('dev/bench/data.js')  # ✅ CORRECT
```

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 2025-11-22 23:44 | Last benchmark run (before fix) |
| 2025-11-23 00:00 | User reported dashboard 404 error |
| 2025-11-23 00:15 | Root cause identified (data.js vs python-bench.json) |
| 2025-11-23 00:20 | Fixed index.html logic in gh-pages branch |
| 2025-11-23 00:22 | Commit 2f04a2f pushed to gh-pages |
| 2025-11-23 00:37 | **Manually triggered performance workflow** |
| 2025-11-23 00:38 | Workflow completed successfully |
| 2025-11-23 00:39 | **GitHub Pages deployment complete** |
| 2025-11-23 00:40 | Final validation: 28/28 checks passed ✅ |

## Architecture Documentation

### GitHub Pages Deployment Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Performance Workflow Triggered                       │
│    (Daily schedule OR manual dispatch)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Run Python Benchmarks                                │
│    - pytest-benchmark generates JSON                    │
│    - github-action-benchmark processes results          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 3. github-action-benchmark (auto-push: true)            │
│    - Creates dev/bench/data.js (JavaScript wrapper)     │
│    - Pushes to gh-pages branch                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Checkout gh-pages Branch                             │
│    - Includes: index.html, dev/bench/data.js            │
│    - Includes: dev/bench/index.html (auto-generated)    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Upload Pages Artifact                                │
│    - actions/upload-pages-artifact@v3                   │
│    - Path: gh-pages-deploy/                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Deploy to GitHub Pages                               │
│    - actions/deploy-pages@v4                            │
│    - OIDC authentication (id-token: write)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 7. GitHub Pages Live                                    │
│    - https://terrylica.github.io/rangebar-py/           │
└─────────────────────────────────────────────────────────┘
```

### Why Manual Edits to gh-pages Require Workflow Run

**Scenario**: Manually edit `index.html` on gh-pages branch

```
Developer → git push origin gh-pages
                │
                ▼
         gh-pages branch updated
                │
                ▼
         ⚠️ GitHub Pages NOT automatically updated
         (build_type: workflow, not legacy)
                │
                ▼
         Must trigger workflow manually
                │
                ▼
         Workflow runs → deploys gh-pages
                │
                ▼
         ✅ GitHub Pages updated
```

### Why This Architecture?

**Benefits**:
1. **Atomic updates**: Benchmark data and dashboard HTML deployed together
2. **No stale data**: Dashboard always matches latest benchmark run
3. **OIDC security**: Uses GitHub's OIDC provider for Pages deployment
4. **Single workflow**: Simpler than maintaining separate Pages workflow

**Trade-offs**:
1. **Manual edits require workflow run**: Can't quick-fix dashboard without triggering benchmarks
2. **Coupled deployment**: Dashboard deployment tied to benchmark schedule

## Lessons Learned

### Documentation Accuracy

**Previous Assumption** (INCORRECT):
> "GitHub Pages auto-deploys when gh-pages branch is updated"

**Corrected Understanding** (CORRECT):
> "GitHub Pages deploys when `actions/deploy-pages@v4` runs in the performance workflow. Manual gh-pages edits require triggering the workflow."

### github-action-benchmark Data Format

**Assumption** (INCORRECT):
> "github-action-benchmark creates JSON files"

**Reality** (CORRECT):
> "github-action-benchmark creates JavaScript files with `window.BENCHMARK_DATA = {...}` wrapper"

**File Created**: `dev/bench/data.js` (NOT `dev/bench/data.json`)

**Reason**: Designed for direct browser inclusion via `<script>` tag, not fetch

### Testing Strategy

**Non-interactive testing** (curl + Python):
- ✅ Fast (< 5 seconds)
- ✅ CI-friendly
- ❌ Can't test JavaScript execution
- ❌ Can't test interactive features

**Interactive testing** (Playwright):
- ✅ Tests actual browser rendering
- ✅ Tests JavaScript execution
- ✅ Tests link clicking
- ❌ Slower (30+ seconds)
- ❌ Requires browser binaries

**Best Practice**: Use both
1. Non-interactive for quick validation
2. Interactive for deployment verification

## References

- **Fix Commit**: [2f04a2f](https://github.com/terrylica/rangebar-py/commit/2f04a2f) (gh-pages branch)
- **Deployment Run**: [19608541389](https://github.com/terrylica/rangebar-py/actions/runs/19608541389)
- **ADR**: [docs/decisions/0004-cicd-multiplatform-builds.md](../decisions/0004-cicd-multiplatform-builds.md) (Section 4.5)
- **Plan**: [docs/plan/0007-daily-performance-monitoring/plan.md](../plan/0007-daily-performance-monitoring/plan.md)
- **Validation Script**: [scripts/validate-dashboard-content.py](../../scripts/validate-dashboard-content.py)
- **Playwright Test**: [scripts/test-dashboard-interactive.py](../../scripts/test-dashboard-interactive.py)
- **Performance Guide**: [docs/PERFORMANCE.md](../PERFORMANCE.md)
- **github-action-benchmark**: https://github.com/benchmark-action/github-action-benchmark
- **Chart.js**: https://www.chartjs.org/

## Next Steps

### Scheduled Operations

**Daily Benchmark Run**: 3:17 AM UTC (configured in `.github/workflows/performance-daily.yml`)
- Automatically runs benchmarks
- Updates dashboard with new data point
- No manual intervention required

### Manual Testing

**Validate Dashboard Anytime**:
```bash
# Non-interactive (fast)
python3 scripts/validate-dashboard-content.py

# Interactive (comprehensive)
uv run scripts/test-dashboard-interactive.py
```

### Trigger Benchmark Manually

**If needed before scheduled run**:
```bash
gh workflow run performance-daily.yml
```

**Monitor execution**:
```bash
gh run watch <RUN_ID>
```

## Contact

For issues or questions:
- **Repository**: https://github.com/terrylica/rangebar-py
- **Dashboard**: https://terrylica.github.io/rangebar-py/
- **CI/CD Workflows**: https://github.com/terrylica/rangebar-py/actions
