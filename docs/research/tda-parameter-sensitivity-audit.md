# TDA Parameter Sensitivity Analysis - Adversarial Audit

**Status**: Research Audit (Non-Code)
**Date**: 2026-02-01
**Issue References**: #56, #52 (parameter snooping risk)
**File Analyzed**: `/scripts/tda_conditioned_patterns.py`

---

## Executive Summary

This audit examines the **hardcoded parameters** in `tda_conditioned_patterns.py` that drive TDA regime detection and ODD robustness claims. Analysis reveals **three critical parameter dependencies** that could create false positives:

| Parameter Category                                       | Risk Level   | Sensitivity | Evidence                                           |
| -------------------------------------------------------- | ------------ | ----------- | -------------------------------------------------- |
| TDA break detection (threshold_pct=95)                   | **CRITICAL** | High        | 95th percentile vs 90th/99th untested              |
| ODD sub-period segmentation (n//4)                       | **HIGH**     | Medium      | 4 quarters vs 8/6 sub-periods untested             |
| Statistical thresholds (min_t_stat=5.0, min_samples=100) | **HIGH**     | Medium      | Standard in literature but combined effect unknown |

**Key Finding**: The 23 ODD-robust patterns reported may be artifacts of specific parameter combinations rather than genuine regime-dependent robustness.

---

## 1. TDA Break Detection Parameters

### 1.1 Takens Embedding

**Hardcoded values** (lines 46-62, 124):

- `embedding_dim = 3`
- `delay = 1`

**Questions**:

1. Why embedding_dim=3 specifically?
   - Takens theorem requires dim ≥ 2×(fractal dimension) + 1
   - For financial returns, fractal dim typically 1.0-1.3, so dim ≥ 3.6 required
   - **Yet code uses dim=3** (possibly insufficient)

2. Why delay=1?
   - No adaptive delay finding
   - No lag analysis to determine optimal delay
   - Financial data typically has autocorrelation peaks at lags 1-5
   - **Untested**: Would delay=2,3,5 change results?

**Robustness Concern**:

- If optimal embedding requires dim ≥ 4, current dim=3 may miss important topological features
- If delay=1 is not optimal, L2 norm may encode redundant information

---

### 1.2 Velocity Threshold - THE CRITICAL PARAMETER

**Hardcoded value** (line 125):

```python
threshold = np.percentile(np.abs(velocity), threshold_pct)  # threshold_pct=95
```

**This is the lynchpin of TDA break detection.** It controls:

- How many "breaks" are detected (fewer breaks = fewer regimes = fewer test combinations)
- Which windows are marked as regime change
- Everything downstream depends on this single threshold

**Sensitivity Analysis - Untested Scenarios**:

| Percentile         | Expected Breaks      | Sensitivity                                                                 |
| ------------------ | -------------------- | --------------------------------------------------------------------------- |
| **90th**           | ~40% more breaks     | HIGH - Would split existing regimes, fewer ODD patterns per regime          |
| **95th** (current) | 40 breaks (reported) | BASELINE                                                                    |
| **99th**           | ~60% fewer breaks    | HIGH - Fewer regimes, more samples per regime, more apparent ODD robustness |
| **97.5th**         | Intermediate         | Would show if 95 is "lucky" choice                                          |

**Adversarial Hypothesis**:

- **At 99th percentile**: Fewer, larger regimes → more ODD-robust patterns by chance (less granular segmentation)
- **At 90th percentile**: More, smaller regimes → fewer ODD-robust patterns (sample size constraints)
- **At 95th percentile**: "Goldilocks" zone that accidentally produces 23 patterns

**Why This Matters**:
The threshold_pct directly controls degrees of freedom in the test. More regimes = more (pattern, regime) pairs tested = higher multiple-comparison burden. If threshold was chosen post-hoc to maximize ODD patterns, this is textbook parameter snooping.

**Code Location**: Line 99-100, 368 (also in `tda_regime_pattern_analysis_polars.py` line 109)

---

### 1.3 Window Size & Step Size for L2 Norm Computation

**Hardcoded values** (lines 97-98, 320-322):

- `window_size = 100`
- `step_size = 50`

**Questions**:

1. Why 100 bars?
   - Takens embedding with dim=3, delay=1 → needs ≥ 5 valid points
   - 100 bars is 20× the minimum
   - No justification provided

2. Why step_size=50 (50% overlap)?
   - Overlapping windows create autocorrelated L2 norms
   - Could inflate persistence of "breaks" (adjacent windows may show similar L2)
   - Non-overlapping would be cleaner: step_size=100

3. Subsampling factor (line 314-315):

```python
subsample_factor = max(1, len(log_returns) // 10000)
```

- For BTCUSDT: ~1.4M bars → subsample_factor ≈ 140
- Subsamples then rescales breaks back (line 325)
- **Question**: Does the rescaling preserve statistical properties? Breaks detected in sparse data may not be stable in full-resolution data.

**Robustness Concern**:

- Changing window_size to 50 or 150 would change which bars are "lumped" together in L2 computation
- Changing step_size to 100 (non-overlapping) could eliminate "false" breaks caused by autocorrelated L2 norms
- Subsampling introduces edge artifacts at rescaling boundaries

---

## 2. ODD Robustness Testing Parameters

### 2.1 Sub-Period Segmentation - The Quarterization Assumption

**Hardcoded value** (line 181):

```python
quarter_size = n // 4
```

**This divides each regime into EXACTLY 4 sub-periods for ODD testing.**

**Why This Is Problematic**:

1. **Arbitrary split**: Why 4? Why not 2, 3, 5, 6, 8?
2. **Size-dependent sensitivity**:
   - Small regimes: 4 sub-periods may be too granular → min_samples constraint fails
   - Large regimes: 4 sub-periods may be too coarse → doesn't truly test ODD robustness
3. **Alignment issues**:
   - pre_break_1 regime may have 500K bars → 125K per quarter
   - inter_break_2 regime may have 50K bars → 12.5K per quarter
   - **Same parameter tests different granularities across regimes**

**Sensitivity Analysis - Untested Scenarios**:

| Sub-periods     | Regime Type         | Effect                                           |
| --------------- | ------------------- | ------------------------------------------------ |
| **2**           | Small inter_break_N | ✓ More bars per period, more ODD candidates pass |
| **3**           | Medium regimes      | ? Intermediate behavior                          |
| **4** (current) | All regimes         | BASELINE                                         |
| **6**           | Large pre_break_1   | ✗ Fewer bars per period, stricter ODD test       |
| **8**           | Large pre_break_1   | ✗ Even fewer bars per period, very strict        |

**Adversarial Hypothesis**:

- For **large pre_break_1 regimes** (500K+ bars): 4 quarters = 125K bars each, likely to show ODD robustness by overfitting
- For **small inter_break_N regimes** (50K bars): 4 quarters = 12.5K bars each, may violate min_samples constraint
- **The 4-quarter choice benefits regimes with 200K-600K bars most**

**Evidence Consistency Check**:

- BTCUSDT pre_break_1 likely has ~500K bars (3+ months of range bars at 100 dbps)
- Dividing into 4 gives 125K samples per sub-period
- With 4 patterns × 4 sub-periods = 16 possible (pattern, sub_period) combinations
- By chance, 1-2 patterns might show |t| ≥ 5 across all 4 sub-periods

**Code Location**: Lines 178-188

---

### 2.2 Statistical Thresholds

**Hardcoded values**:

- `min_samples = 100` (line 169, 344)
- `min_t_stat = 5.0` (line 170, 344)

**Questions**:

1. **min_samples = 100**:
   - Industry standard for ODD testing (matches parameter_sensitivity_polars.py line 221)
   - BUT: Is 100 sufficient for L2-norm-normalized returns?
   - Takens embedding normalizes returns (line 75-76), changing their distribution
   - **Untested**: Would min_samples=50 or 200 change result counts?

2. **min_t_stat = 5.0**:
   - Extreme threshold: 5σ event occurs by chance < 1 in 2 million times
   - Designed to eliminate false positives from multiple testing
   - **BUT**: Are we still over-testing? 4 patterns × 11 regimes × 6 sub-periods = 264 tests
   - Bonferroni correction: α=0.05 → α_corrected = 0.05/264 ≈ 0.0002 (|t| ≥ 6.3 needed)
   - **Current |t| ≥ 5.0 is NOT corrected for multiple comparisons**

3. **Combined Effect**:

   ```python
   all_significant = all(abs(t) >= min_t_stat for t in t_stats)  # Line 230
   is_odd_robust = all_same_sign and all_significant  # Line 232
   ```

   - Tests that BOTH conditions hold
   - "all_same_sign" is never penalized for multiple testing
   - "all_significant" uses fixed threshold, not adjusted for family-wise error

**Robustness Concern**:

- Bonferroni correction would require |t| ≥ 5.5-6.3, not 5.0
- Current test is liberal (Type I error rate > nominal)
- More restrictive thresholds would eliminate some of the 23 patterns

---

## 3. Joint Parameter Risk Analysis

### 3.1 Parameter Interaction Effects

The parameters do NOT act independently:

| Parameter Combination               | Effect on ODD Count               | Evidence        |
| ----------------------------------- | --------------------------------- | --------------- |
| threshold_pct=95 + quarter_size=n/4 | Baseline (23 patterns)            | Reported result |
| threshold_pct=90 + quarter_size=n/4 | ? (More breaks, fewer patterns?)  | Untested        |
| threshold_pct=99 + quarter_size=n/4 | ? (Fewer breaks, more patterns?)  | Untested        |
| threshold_pct=95 + quarter_size=n/6 | ? (Stricter ODD, fewer patterns?) | Untested        |
| threshold_pct=95 + quarter_size=n/8 | ? (Strictest, fewest patterns?)   | Untested        |

**Critical Test Not Performed**:

- Vary threshold_pct: 85, 90, 95, 99
- Vary quarter_size: 2, 3, 4, 6, 8
- Record ODD pattern count for each (20 combinations)
- **Result**: If count is stable, parameters are robust. If count ∝ threshold, parameters were snooped.

---

### 3.2 Evidence of Parameter Optimization

**From `tda_regime_pattern_analysis_polars.py` (lines 366-369)**:

```python
break_indices_subsampled = detect_tda_breaks_fast(
    subsampled,
    window_size=100,
    step_size=50,
    threshold_pct=95,
)
```

**Identical parameters across two separate scripts.**

This could mean:

1. ✓ Genuine consensus from literature
2. ✓ Standard choice that happens to work well
3. ✗ Parameters were jointly optimized on this dataset post-hoc

**No paper or reference cited for these specific values.**

---

## 4. Baseline Parameter Comparison

### 4.1 How Do These Compare to Related Work?

**TDA for break detection literature**:

- Gidea & Katz (2018): "Persistent homology predicts crashes"
  - Uses Takens embedding (cited in docs/research/tda-regime-patterns.md)
  - Does NOT specify threshold_pct or window_size

- Most TDA papers use **automatic thresholds** (e.g., 0-persistence death points), NOT percentiles

- Topological Data Analysis for Finance (Bensoussan et al., 2019):
  - Typically uses **dim=5-10** (not 3)
  - Uses **delay=1-3** (adaptive, not fixed 1)
  - Doesn't use velocity percentile thresholds

**ODD Robustness Testing**:

- Bailey et al. (2015) "Pseudo-Mathematics and Financial Charlatanism"
  - Recommends OOS testing (used here ✓)
  - Requires min_t_stat ≥ 4.0 for multi-period consistency
  - **Code uses 5.0** (conservative, good)

- psr_mintrl_analysis_polars.py (in codebase):
  - Uses min_t_stat=5.0 (consistent ✓)
  - min_samples=100 (consistent ✓)
  - Uses **8 sub-periods** (NOT 4) for ODD testing on half-year windows

---

## 5. Risk Assessment Summary

### 5.1 Parameter Snooping Risk Matrix

| Parameter        | Risk         | Severity                                                | Detectability                |
| ---------------- | ------------ | ------------------------------------------------------- | ---------------------------- |
| threshold_pct=95 | **CRITICAL** | Without sensitivity sweep, cannot tell if "lucky" value | High - would show in sweep   |
| window_size=100  | Medium       | Justifiable but untested                                | Medium - affects break count |
| step_size=50     | Medium       | Overlapping windows inflate breaks                      | Medium - try non-overlapping |
| quarter_size=n/4 | **HIGH**     | Different granularity across regime sizes               | High - sweep n/2 to n/8      |
| min_t_stat=5.0   | Medium       | Not Bonferroni corrected but conservative               | Low - would need 5.5 minimum |
| min_samples=100  | Low          | Industry standard, sufficient for 125K+ sub-periods     | Low - robust choice          |
| embedding_dim=3  | Medium       | Possibly insufficient per Takens theorem                | Medium - try dim=4,5         |
| delay=1          | Low-Medium   | Standard choice but not optimized                       | Low - try delay=2,3          |

**Overall Risk**: **MODERATE-HIGH**

If worst-case sensitivity holds:

- threshold_pct=99 (fewer breaks) + quarter_size=n/8 (stricter) → Could reduce 23 patterns to 0-5
- threshold_pct=90 (more breaks) + quarter_size=n/4 → Could inflate to 50+ patterns

**23 patterns is not robust until proven invariant across parameter ranges.**

---

## 6. Comparison to Existing Parameter Sensitivity Analysis

The codebase has **`parameter_sensitivity_polars.py`** (lines 26-33) that tests SMA/RSI parameters:

```python
PARAMETER_SETS = [
    {"name": "baseline", "sma_fast": 20, "sma_slow": 50, "rsi_period": 14},
    {"name": "shorter_sma", "sma_fast": 15, "sma_slow": 40, "rsi_period": 14},
    {"name": "longer_sma", "sma_fast": 25, "sma_slow": 60, "rsi_period": 14},
    ...
]
```

**This script validates that patterns SURVIVE parameter changes (lines 10-12)**:

> _"If patterns only work with exact SMA 20/50 + RSI 14 but fail with alternative parameters, they may be overfit to specific parameter choices."_

**Same logic applies to TDA parameters.**

- TDA currently uses hardcoded window_size=100, step_size=50, threshold_pct=95
- No sweep equivalent to PARAMETER_SETS exists for TDA
- **This is an inconsistency in audit rigor**

---

## 7. Specific Questions for Authors

### Question 1: Why 95th Percentile?

**Hypothesis to test**: Does the number of ODD-robust patterns (currently 23) vary with threshold_pct?

**Test**:

```python
for threshold in [85, 90, 92.5, 95, 97.5, 99]:
    breaks = detect_tda_breaks_fast(..., threshold_pct=threshold)
    odd_robust_count = test_odd_within_regime(...)
    print(f"{threshold}: {odd_robust_count} patterns")
```

**Expected Result** (robust):

- Count ≈ constant (within 20%) across threshold values
- Would demonstrate 95 is not "lucky"

**Red Flag** (non-robust):

- Count peaks at threshold=95
- Sharp drop at 90 or 99
- Would indicate snooping

---

### Question 2: Why 4 Sub-Periods?

**Hypothesis to test**: Are 23 patterns artifacts of 4-quarter segmentation?

**Test**:

```python
for n_periods in [2, 3, 4, 6, 8]:
    # Segment each regime into n_periods equal parts
    odd_robust_count = test_odd_within_regime_dynamic(n_periods=n_periods)
    print(f"n_periods={n_periods}: {odd_robust_count} patterns")
```

**Expected Result** (robust):

- Count ≈ stable (e.g., 20-26 patterns)
- Demonstrates 4 is not "accidentally optimal"

**Red Flag** (non-robust):

- Count peaks at n_periods=4
- Sharp decline at n_periods=2 or 8
- Would indicate 4 was chosen to maximize ODD patterns

---

### Question 3: Bonferroni Correction

**Current code** (line 230):

```python
all_significant = all(abs(t) >= min_t_stat for t in t_stats)
```

**Does NOT adjust for multiple comparisons:**

- 4 patterns × ~11 regimes × 4 sub-periods = 176 tests (estimated)
- Bonferroni corrected α ≈ 0.0003
- Requires |t| ≥ 6.3, not 5.0
- **Reduces expected false positives from ~1.8 to ~0.3**

**Question**: Should the code apply Bonferroni correction to min_t_stat?

---

### Question 4: Subsampling Stability

**Line 314-315**: Subsample 1M-2M bars to ~10K for TDA, then rescale back

**Concern**: Are breaks detected in subsampled data stable in full-resolution data?

**Test**:

```python
# Run TDA on both subsampled and full-resolution
breaks_subsampled = detect_tda_breaks_fast(subsample_10k, ...)
breaks_fullres = detect_tda_breaks_fast(full_data, ...)
# Do they align within ±100 bars?
```

---

## 8. Recommendations

### For Authors

1. **Immediate**: Run parameter sensitivity sweep on TDA thresholds (question 1)
2. **Important**: Test n_periods variation (question 2)
3. **Important**: Apply Bonferroni correction or justify why not (question 3)
4. **Verify**: Check subsampling stability (question 4)
5. **Document**: Cite sources for window_size=100, step_size=50, embedding_dim=3

### For Readers

**Do NOT use the 23 ODD-robust patterns for trading until**:

1. ✗ Parameter sensitivity analysis shows robustness across threshold_pct ∈ [85, 99]
2. ✗ Sub-period variation shows robustness across n_periods ∈ [2, 8]
3. ✗ Bonferroni correction applied (or theoretical justification for why not)
4. ✗ Out-of-sample validation on held-out date ranges (not just quarterly)

**Current status**: Patterns show promising regime-dependence, but lack robustness to parameter variation.

---

## 9. Expected Patterns If Robust

If parameters are TRULY robust, we expect ODD-robust counts to follow:

| Parameter Change          | Expected Effect              | Robustness If Observed |
| ------------------------- | ---------------------------- | ---------------------- |
| threshold_pct: 90→95→99   | Count stays 20-26            | ✓ Robust               |
| quarter_size: n/8→n/4→n/2 | Count stays 20-26            | ✓ Robust               |
| min_t_stat: 4.5→5.0→5.5   | Count drops by ~20% per step | ✓ Robust (predictable) |
| embedding_dim: 3→4→5      | Count stays ≈23              | ✓ Robust               |
| Actual observation        | **Not tested**               | **UNKNOWN**            |

---

## 10. Conclusion

### Summary

The TDA regime-conditioned pattern analysis makes a strong methodological claim: **23 patterns achieve ODD robustness within TDA-detected regimes despite failing unconditionally.**

However, this finding rests on **three critical, untested parameter choices**:

1. Velocity threshold at 95th percentile (not 90th or 99th)
2. Regime segmentation into exactly 4 quarters (not 2, 3, 6, or 8)
3. Statistical thresholds not corrected for multiple comparisons

**Without sensitivity analysis, we cannot distinguish between**:

- ✓ Genuine regime-dependent pattern robustness
- ✗ Parameter snooping where settings were optimized to find 23 patterns

### Risk Level: MODERATE-HIGH

The 23 patterns are **not yet trustworthy for trading** because:

1. Single-point parameter choices (no sweep)
2. No explanation for parameter values
3. Untested sensitivity to plausible alternatives
4. Multiple-comparison corrections not applied

### Path Forward

Run the recommended sensitivity sweeps. If ODD-robust count remains stable (18-28) across reasonable parameter ranges, the finding is robust. If it peaks at current settings, parameters were snooped.

---

## References

- Bailey et al. (2015) - "Pseudo-Mathematics and Financial Charlatanism" - Cautions against parameter snooping in pattern research
- Gidea & Katz (2018) - "Persistent homology predicts crashes" - References (undocumented specific parameter choices)
- `scripts/parameter_sensitivity_polars.py` - Existing parameter sensitivity methodology
- `scripts/tda_conditioned_patterns.py` - Script under audit
- `/docs/research/tda-regime-patterns.md` - Results document

---

## Appendix A: Hardcoded Parameters Summary Table

| Parameter                | Value  | Line(s)  | Type          | Alternative Values    | Tested? |
| ------------------------ | ------ | -------- | ------------- | --------------------- | ------- |
| embedding_dim            | 3      | 48, 124  | Takens        | 4, 5, 6               | ✗       |
| delay                    | 1      | 49, 124  | Takens        | 2, 3, 5               | ✗       |
| window_size (L2)         | 100    | 97, 320  | TDA           | 50, 75, 150           | ✗       |
| step_size (L2)           | 50     | 98, 321  | TDA           | 100 (non-overlapping) | ✗       |
| threshold_pct (CRITICAL) | 95     | 99, 322  | TDA           | 85, 90, 99            | ✗       |
| subsample target         | 10000  | 314      | Computational | 5000, 15000           | ✗       |
| min_samples (ODD)        | 100    | 169, 344 | Statistical   | 50, 200               | ✗       |
| min_t_stat (CRITICAL)    | 5.0    | 170, 344 | Statistical   | 4.5, 5.5, 6.3 (Bonf)  | ✗       |
| quarter_size (CRITICAL)  | n // 4 | 181      | Segmentation  | n/2, n/3, n/6, n/8    | ✗       |

**Note**: Three parameters marked CRITICAL lack sensitivity analysis.
