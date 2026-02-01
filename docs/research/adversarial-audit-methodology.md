# Adversarial Audit Methodology for Quantitative Pattern Research

**Status**: Production-Ready Framework
**Developed**: 2026-01-31 to 2026-02-01
**Application**: Pattern research across 9 approaches, resulting in definitive invalidation

---

## Executive Summary

This document codifies the adversarial audit framework developed during exhaustive pattern research on range bar data. The framework successfully identified and invalidated apparent ODD robust patterns that would have led to false-positive trading signals.

**Key Achievement**: 100% of "promising" patterns were invalidated through systematic audit, preventing deployment of spurious signals.

---

## Framework Overview

```
Initial Discovery → Adversarial Audit → Validation/Invalidation
     ↓                    ↓                     ↓
  ODD robust?         Multi-perspective      Genuine Alpha
  |t| >= 3.0?         forensic analysis      or Artifact?
  Same sign?                                      ↓
                                           DEPLOY or REJECT
```

---

## Phase 1: Temporal Safety

### 1.1 Feature/Target Alignment

**Rule**: Features use `shift(1)`, targets use `shift(-1)`

```python
# CORRECT: Temporal-safe pattern
df = df.with_columns([
    pl.col("direction").shift(1).alias("prev_direction"),  # Feature: look-back
    pl.col("close").shift(-1).alias("next_close"),         # Target: look-forward
])

# WRONG: Data leakage via same-bar target
df = df.with_columns([
    pl.col("direction").alias("current_direction"),  # LEAKAGE: uses current bar info
    pl.col("close").shift(-1).alias("next_close"),
])
```

### 1.2 Range Bar Deferred-Open Semantics

**Critical**: Range bars use "deferred open" where the breaching trade becomes both:

- Close of bar N
- Open of bar N+1

**Implication**: Consecutive bars share a tick, creating mechanical correlation.

**Audit Check**:

```sql
-- Check temporal overlap percentage
WITH bars AS (
    SELECT
        timestamp_ms AS close_ms,
        timestamp_ms - duration_us / 1000 AS open_ms,
        leadInFrame(timestamp_ms - duration_us / 1000, 1) OVER (ORDER BY timestamp_ms) AS next_open_ms
    FROM range_bars
)
SELECT
    count() AS total_pairs,
    countIf(next_open_ms < close_ms) AS overlapping_pairs,
    round(100.0 * countIf(next_open_ms < close_ms) / count(), 2) AS overlap_pct
FROM bars
WHERE next_open_ms IS NOT NULL
```

**Red Flag**: If `overlap_pct > 50%`, apparent persistence is mechanical artifact.

---

## Phase 2: Statistical Rigor

### 2.1 ODD Robustness Criteria

A pattern is ODD (Out-of-Distribution) robust if:

| Criterion                    | Threshold     | Rationale                 |
| ---------------------------- | ------------- | ------------------------- |
| Same sign across ALL periods | 100%          | No regime dependence      |
| t-statistic                  | \|t\| >= 3.0  | ~0.3% false positive rate |
| Minimum samples per period   | >= 100        | Statistical power         |
| Cross-symbol replication     | ALL 4 symbols | Not asset-specific        |

### 2.2 FDR Correction (Benjamini-Hochberg)

When testing N patterns, apply FDR correction:

```python
from scipy.stats import false_discovery_control

p_values = [pattern.p_value for pattern in patterns]
significant = false_discovery_control(p_values, method='bh') < 0.05
```

**Note**: If testing 100 patterns at α=0.05, expect ~5 false positives. FDR corrects for this.

### 2.3 Hurst Exponent Adjustment

If returns show long memory (H ≠ 0.5), adjust effective sample size:

```python
T_eff = T ** (2 * (1 - H))

# Example: H = 0.79 (strong long memory)
# T = 10000 bars → T_eff = 10000 ** (2 * 0.21) = 10000 ** 0.42 ≈ 30
# Effective sample is ~30, not 10000!
```

**Implication**: Pattern research with H ~ 0.79 has drastically reduced statistical power.

---

## Phase 3: Multi-Perspective Audit

### 3.1 Data Leakage Detection

**Lookback Contamination**:

```python
# Check if regime labels use future information
regime_window = 20  # bars
assert feature_timestamp < target_timestamp - regime_window
```

**Regime Label Leakage**:

- SMA crossovers computed on close prices include current bar
- RSI includes current bar
- Solution: Use `shift(1)` on all regime indicators

### 3.2 Parameter Sensitivity

Test robustness across parameter sweeps:

| Parameter          | Sweep Values         |
| ------------------ | -------------------- |
| Tercile boundaries | 25/75, 33/67, 40/60  |
| Lookback window    | 10, 20, 50, 100 bars |
| Threshold (dbps)   | 50, 100, 200         |
| Lag                | 1, 2, 3, 5 bars      |

**Red Flag**: If pattern only appears at specific parameters, it's likely overfitted.

### 3.3 Mechanical Artifact Detection

**Duration Autocorrelation Audit**:

1. Compute inter-bar gaps
2. If 100% negative gaps (N+1 opens before N closes), persistence is mechanical
3. Test lag decay: genuine persistence decays smoothly, mechanical drops sharply

```sql
-- Lag decay analysis
SELECT
    lag,
    tercile,
    round(100.0 * countIf(future_tercile = current_tercile) / count(), 2) AS persist_pct
FROM classified_bars
GROUP BY lag, tercile
-- Expected: smooth decay from lag=1 to lag=5
-- Artifact: sharp drop from lag=1 to lag=2
```

### 3.4 Cross-Symbol Validation

Pattern must replicate across uncorrelated assets:

| Symbol  | Correlation to BTC | Required |
| ------- | ------------------ | -------- |
| BTCUSDT | 1.00               | ✓        |
| ETHUSDT | 0.85               | ✓        |
| SOLUSDT | 0.70               | ✓        |
| BNBUSDT | 0.65               | ✓        |

**Requirement**: ALL 4 symbols must show ODD robust pattern independently.

---

## Phase 4: MRH Framework Integration

### 4.1 Minimum Reliable Horizon Inequality

```
Supply(ADWIN) >= Demand(MinTRL)
```

- **ADWIN**: Adaptive window length where data is statistically consistent
- **MinTRL**: Minimum track record length for significance given higher moments

### 4.2 ADWIN for Regime Detection

Use parameter-free ADWIN instead of fixed windows:

```python
from river import drift

adwin = drift.ADWIN()
for return_value in returns:
    adwin.update(return_value)
    if adwin.drift_detected:
        # Regime change detected
        handle_regime_shift()
```

### 4.3 Probabilistic Sharpe Ratio

Account for skewness and kurtosis:

```python
def psr(sharpe, T, skew, kurt):
    """Probabilistic Sharpe Ratio."""
    se = sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / T)
    return norm.cdf(sharpe / se)
```

### 4.4 Deflated Sharpe Ratio

Account for multiple testing (backtest overfitting):

```python
def dsr(sharpe, T, n_trials, skew, kurt):
    """Deflated Sharpe Ratio."""
    E_max_sharpe = expected_max_sharpe(n_trials)
    return psr(sharpe - E_max_sharpe, T, skew, kurt)
```

---

## Audit Checklist

Before deploying any pattern, verify:

- [ ] **Temporal Safety**: Features use shift(1), targets use shift(-1)
- [ ] **No Overlap**: Inter-bar gaps > 0 (no deferred-open contamination)
- [ ] **ODD Robust**: Same sign, |t| >= 3.0, ALL symbols, ALL periods
- [ ] **FDR Corrected**: Adjusted for number of patterns tested
- [ ] **Hurst Adjusted**: Effective sample size accounts for long memory
- [ ] **Parameter Robust**: Holds across parameter sweeps
- [ ] **Lag Decay Smooth**: Persistence decays gradually, not sharply
- [ ] **ADWIN >= MinTRL**: Stationarity window exceeds convergence requirement
- [ ] **DSR > 0**: Deflated Sharpe accounts for trial multiplicity

---

## Lessons Learned

### What Invalidated Patterns

| Approach                 | Invalidation Cause                           |
| ------------------------ | -------------------------------------------- |
| Direction (U/D)          | Boundary-locked returns (H ~ 0.79)           |
| 2-bar/3-bar              | Forward returns show mean reversion          |
| TDA regime               | Lookback leakage in regime labels            |
| Microstructure           | Feature noise exceeds signal                 |
| Cross-threshold          | Temporal overlap contaminates alignment      |
| Duration autocorrelation | 100% mechanical (deferred-open)              |
| TDA velocity             | t-stats -1.67 to +1.01 (no predictive power) |

### Key Insight

Range bars are valuable for:

- ✅ Volatility normalization
- ✅ Execution quality measurement
- ✅ Market making analytics

But NOT for:

- ❌ Directional prediction
- ❌ Pattern-based alpha generation

The structural properties (boundary-locked returns, temporal overlap, long memory) make range bars unsuitable for predictive pattern research.

---

## Scripts Reference

| Script                              | Audit Purpose                     |
| ----------------------------------- | --------------------------------- |
| `duration_autocorrelation_audit.py` | Temporal overlap, gap analysis    |
| `fdr_corrected_patterns.py`         | FDR correction implementation     |
| `tda_volatility_forecast.py`        | TDA velocity → RV prediction test |
| `temporal_safe_patterns_polars.py`  | Temporal safety validation        |
| `combined_pattern_audit_polars.py`  | Multi-perspective audit           |

---

## References

- `docs/research/external/time-to-convergence-stationarity-gap.md` - MRH framework theory
- `docs/research/pattern-research-summary.md` - Complete research findings
- Issue #57 - Research completion summary
