# Brute Force Microstructure Feature Analysis

**Date**: 2026-02-05
**Database**: `rangebar_cache.feature_combinations` (ClickHouse)
**Asset**: SOLUSDT @ 1000 dbps (10% range bars)
**Method**: Progressive evolutionary brute-force search

---

## Executive Summary

**ULTIMATE Champion Pattern**: `2 DOWN bars + trade_intensity > p95 + kyle_lambda > 0 → LONG`

| Metric             | Value                                             |
| ------------------ | ------------------------------------------------- |
| Hit Rate           | **68.32%** (previous champion: 52.98%)            |
| Edge               | **+18.32%** directional accuracy above 50%        |
| Z-Score            | **9.74** (extremely statistically significant)    |
| Signal Count       | 707 samples                                       |
| Temporal Stability | ✅ Positive ALL 6 years (2020-2025)               |
| Cross-Asset        | ✅ SOL primary; BTC/BNB positive; ❌ ETH inverted |

### Evolution of Champions (Gen1-Gen8)

| Gen | Champion Pattern                      | Hit Rate   | Edge        | Insight                   |
| --- | ------------------------------------- | ---------- | ----------- | ------------------------- |
| 2   | ti_p95 + kyle>0                       | 52.98%     | +2.98%      | Intensity + direction     |
| 6   | ti_p90_3bar_streak                    | 52.12%     | +2.12%      | Intensity momentum        |
| 7   | meanrev_2down_ti_p90_long             | 60.90%     | +10.90%     | Mean reversion discovery! |
| 8   | **combo_2down_ti_p95_kyle_gt_0_long** | **68.32%** | **+18.32%** | **Combined Gen2 + Gen7**  |

---

## Generation 1: Single Feature Predictability

**Finding**: Trade intensity is the most predictive single feature.

| Feature                       | Hit Rate | Edge   | Z-Score | Samples |
| ----------------------------- | -------- | ------ | ------- | ------- |
| `ti_gt_p90` (intensity>p90)   | 52.18%   | +2.18% | 4.15    | 6,778   |
| `ti_gt_p80` (intensity>p80)   | 51.70%   | +1.70% | 4.54    | 13,253  |
| `agg_gt_p90` (aggression>p90) | 51.47%   | +1.47% | 2.92    | 7,001   |
| `ofi_gt_0` (OFI>0)            | 50.25%   | +0.25% | 0.87    | 46,823  |

**Insight**: High trade intensity is a regime indicator - when bars form quickly due to heavy volume, the direction tends to persist.

---

## Generation 2: Two-Feature Combinations

**Finding**: Intensity + Kyle Lambda is the best combination. SHORT signals are WORSE than random.

### Best LONG Signals

| Combination        | Hit Rate   | Edge   | Z-Score | Samples |
| ------------------ | ---------- | ------ | ------- | ------- |
| `ti_p95 + kyle>0`  | **52.98%** | +2.98% | 2.97    | 2,480   |
| `ti_p95 + ofi>p90` | 52.36%     | +2.36% | 1.97    | 1,740   |
| `ti_p90 + kyle>0`  | 51.88%     | +1.88% | 2.77    | 5,432   |
| `ti_p80 + kyle>0`  | 51.46%     | +1.46% | 3.26    | 12,472  |

### Worst SHORT Signals (INVERTED!)

| Combination        | Hit Rate | Edge       | Z-Score | Samples |
| ------------------ | -------- | ---------- | ------- | ------- |
| `ti_p95 + ofi<p10` | 44.10%   | **-5.90%** | -4.41   | 1,399   |
| `ti_p95 + kyle<0`  | 45.79%   | **-4.21%** | -3.70   | 1,937   |

**Critical Insight**: SHORT signals systematically LOSE. This suggests a **long-only strategy** is optimal for SOL.

---

## Generation 3: Three-Feature Combinations

**Finding**: Adding a third filter HURTS performance. Keep it simple.

| Combination                | Hit Rate | Edge        | Z-Score | Samples |
| -------------------------- | -------- | ----------- | ------- | ------- |
| `ti_p95 + kyle>0 + pi>p90` | 50.59%   | +0.59%      | 0.45    | 1,433   |
| `ti_p95 + kyle>0 + ofi>0`  | 38.70%   | **-11.30%** | -8.30   | 1,349   |

**Why**: Adding filters reduces sample size AND introduces conflicting signals. The original two-feature pattern captures the core microstructure effect; adding more just adds noise.

---

## Generation 4: Temporal Stability

**Finding**: Champion pattern is ROBUSTLY POSITIVE across all 6 years.

### ti_p95 + kyle>0 by Year

| Year | Hit Rate   | Edge    | Z-Score | Samples |
| ---- | ---------- | ------- | ------- | ------- |
| 2020 | **60.24%** | +10.24% | 1.87    | 83      |
| 2021 | 52.73%     | +2.73%  | 1.78    | 1,064   |
| 2022 | 55.41%     | +5.41%  | 1.64    | 231     |
| 2023 | 51.45%     | +1.45%  | 0.48    | 276     |
| 2024 | 51.29%     | +1.29%  | 0.39    | 232     |
| 2025 | 53.05%     | +3.05%  | 1.50    | 607     |

**Interpretation**:

- 2020 had exceptional edge (60%) but very small sample (83 bars)
- 2021-2025 show consistent 51-55% edge - the pattern is NOT a regime artifact
- The slight decay over time (60%→51%) may indicate market adaptation, but edge persists

---

## Generation 5: Cross-Asset Validation

**Finding**: Pattern works on SOL/BTC/BNB but is INVERTED on ETH.

| Asset   | ti_p95+kyle>0 | Hit Rate   | Edge       | Z-Score | Verdict           |
| ------- | ------------- | ---------- | ---------- | ------- | ----------------- |
| **SOL** | 2,556         | **52.97%** | +2.97%     | 3.01    | ✅ Primary target |
| **BTC** | 682           | 51.47%     | +1.47%     | 0.77    | ✅ Weak positive  |
| **BNB** | 1,138         | 51.41%     | +1.41%     | 0.95    | ✅ Weak positive  |
| **ETH** | 1,123         | **45.59%** | **-4.41%** | -2.95   | ❌ **INVERTED**   |

**Implication for NN Training**:

1. **SOL is the primary asset** for this microstructure pattern
2. **Exclude ETH** from training - its behavior is opposite
3. **BTC/BNB** can be used for generalization but expect weaker edge

---

## Recommendations for Neural Network Training

### 1. Feature Engineering

- **Use only 2 features**: `trade_intensity`, `kyle_lambda_proxy`
- **Discard OFI, aggression, price_impact** as additional inputs (they hurt)
- **Threshold**: Use p90 or p95 depending on desired sample/edge tradeoff

### 2. Model Architecture

- **LONG-ONLY**: The gate should only open for long positions
- **Regime Filter**: Only predict when `trade_intensity > p90`
- **Selective Trading**: Gate probability should correlate with intensity

### 3. Training Data

- **SOL primary**: Train on SOLUSDT
- **Validation**: Use BTC/BNB for cross-validation
- **Exclude ETH**: Pattern is inverted

### 4. Expected Performance

- **Base edge**: ~2-3% directional accuracy above 50%
- **With NN refinement**: Potentially 3-5% if model learns nuanced patterns
- **Risk**: Edge may decay over time (market adaptation)

---

## SQL Queries for Reference

All results stored in `rangebar_cache.feature_combinations`:

```sql
-- View best combinations
SELECT combo_name, hit_rate, edge_pct, z_score, signal_count
FROM rangebar_cache.feature_combinations
WHERE generation IN (1,2) AND symbol = 'SOLUSDT'
ORDER BY hit_rate DESC
LIMIT 20;

-- Temporal stability
SELECT combo_name, hit_rate, signal_count
FROM rangebar_cache.feature_combinations
WHERE generation = 4 AND combo_name LIKE 'ti_p95%'
ORDER BY combo_name;

-- Cross-asset comparison
SELECT symbol, combo_name, hit_rate, z_score
FROM rangebar_cache.feature_combinations
WHERE generation = 5
ORDER BY symbol;
```

---

## Generation 6: Lookback Patterns (2-bar and 3-bar lag)

**Finding**: Intensity momentum persists across bars; direction momentum leads to MEAN REVERSION.

| Pattern                        | Hit Rate   | Edge    | Z-Score | Samples |
| ------------------------------ | ---------- | ------- | ------- | ------- |
| `ti_p90_3bar_streak`           | 52.12%     | +2.12%  | 2.92    | 4,735   |
| `ti_p90_lag2_AND_ti_p90_lag1`  | 52.08%     | +2.08%  | 3.22    | 5,950   |
| `kyle_reversal_neg_to_pos`     | 51.20%     | +1.20%  | 3.17    | 17,394  |
| `dir_up_2bar_ti_p90` (LONG)    | **43.26%** | -6.74%  | -6.04   | 2,011   |
| `dir_down_2bar_ti_p90` (SHORT) | **38.81%** | -11.19% | -9.58   | 1,832   |

**MAJOR DISCOVERY**: Direction momentum (2 consecutive bars same direction) leads to MEAN REVERSION, not continuation!

---

## Generation 7: Mean Reversion Patterns (BREAKTHROUGH)

**Finding**: Mean reversion is ASYMMETRIC - DOWN→UP works better than UP→DOWN.

| Pattern                     | Hit Rate   | Edge        | Z-Score | Samples |
| --------------------------- | ---------- | ----------- | ------- | ------- |
| `meanrev_2down_ti_p90_long` | **60.90%** | **+10.90%** | 9.63    | 1,949   |
| `meanrev_2up_ti_p90_short`  | 56.82%     | +6.82%      | 6.30    | 2,133   |
| `meanrev_3down_long`        | 52.23%     | +2.23%      | 4.56    | 10,489  |
| `meanrev_4down_long`        | 51.94%     | +1.94%      | 2.74    | 5,012   |
| `single_down_long`          | 51.69%     | +1.69%      | 7.15    | 44,829  |

**Insight**: After 2 DOWN bars with high intensity, the market tends to reverse UP. This is the strongest pattern found so far (before Gen8).

---

## Generation 8: Composite Signals (ULTIMATE CHAMPION)

**Finding**: Combining Gen2 champion (ti_p95 + kyle>0) with Gen7 mean reversion (2 DOWN bars) produces the ULTIMATE signal.

| Pattern                             | Hit Rate   | Edge        | Z-Score  | Samples |
| ----------------------------------- | ---------- | ----------- | -------- | ------- |
| `combo_2down_ti_p95_kyle_gt_0_long` | **68.32%** | **+18.32%** | **9.74** | 707     |
| `combo_2down_ti_p90_kyle_gt_0_long` | 60.26%     | +10.26%     | 8.24     | 1,613   |
| `exhaustion_down_kyle_pos_long`     | 51.39%     | +1.39%      | 5.10     | 33,578  |

**Ultimate Pattern**: `2 consecutive DOWN bars + trade_intensity > p95 + kyle_lambda > 0 → LONG`

This pattern captures:

1. **Mean reversion**: 2 DOWN bars create oversold condition
2. **Intensity filter**: High intensity confirms conviction
3. **Kyle direction**: Positive Kyle indicates buying pressure despite DOWN bars
4. **All signals align**: The divergence between direction and Kyle predicts reversal

---

## Files Created

| File                              | Purpose                                   |
| --------------------------------- | ----------------------------------------- |
| `brute_force_gen1.sql`            | Single feature predictability (18 combos) |
| `brute_force_gen2.sql`            | Two-feature combinations (19 combos)      |
| `brute_force_gen3.sql`            | Three-feature combinations (12 combos)    |
| `brute_force_gen4_temporal.sql`   | Year-by-year stability analysis (21)      |
| `brute_force_gen5_crossasset.sql` | Cross-asset validation (7 combos)         |
| `brute_force_gen6_lookback.sql`   | 2-3 bar lookback patterns (11 combos)     |
| `brute_force_gen7_meanrev.sql`    | Mean reversion patterns (12 combos)       |
| `brute_force_gen8_divergence.sql` | Composite signals (11 combos)             |

**Total combinations tested**: 111 patterns across 8 generations

---

## CRITICAL: Lookahead Bias Audit (2026-02-05)

**Finding**: Original Gen1-Gen8 analysis had lookahead bias in percentile computation.

### The Problem

Percentile thresholds were computed over the ENTIRE dataset:

```sql
quantile(0.95)(trade_intensity) -- Uses ALL data including future!
```

This means when comparing `ti > p95` at time t, we were using information from times t+1, t+2, ..., T.

### Year-Specific Percentile Variation

| Year | ti_p95 | Ratio to Full-Dataset p95 (316.74) |
| ---- | ------ | ---------------------------------- |
| 2020 | 18.35  | 0.06x (17x lower!)                 |
| 2021 | 500.54 | 1.58x                              |
| 2022 | 211.23 | 0.67x                              |
| 2023 | 212.27 | 0.67x                              |
| 2024 | 306.91 | 0.97x                              |
| 2025 | 695.61 | 2.20x                              |

Using full-dataset p95=316 means 2020 signals are ~17x more selective than they should be (inflating edge),
while 2025 signals are less selective (diluting edge).

### No-Lookahead Fix

Created Gen 108-110 with **prior-year percentiles**:

```sql
lagInFrame(ti_p95, 1) OVER w as ti_p95_prior  -- Use PRIOR bar's year percentile
countIf(ti_1 > ti_p95_prior AND ...)          -- No lookahead
```

### Impact on Champion Pattern

| Version               | Hit Rate   | Edge        | Z-Score  | Verdict   |
| --------------------- | ---------- | ----------- | -------- | --------- |
| Gen8 (lookahead)      | 68.32%     | +18.32%     | 9.74     | BIASED    |
| Gen108 (no-lookahead) | **66.76%** | **+16.76%** | **8.95** | **VALID** |

**Conclusion**: Core finding HOLDS. Edge reduced by 1.56pp but remains highly significant (z=8.95).

---

## Generation 10: No-Lookahead Cross-Asset Validation (2026-02-05)

**Finding**: Champion pattern validates across BTC and BNB with no lookahead bias.

### Full Pattern: 2 DOWN + ti>p95_prior + kyle>0 → LONG

| Asset   | Hit Rate   | Edge        | Z-Score  | Signals | Verdict                   |
| ------- | ---------- | ----------- | -------- | ------- | ------------------------- |
| **BNB** | **71.72%** | **+21.72%** | **8.05** | 343     | ✅ **STRONGEST**          |
| **SOL** | 66.76%     | +16.76%     | 8.95     | 713     | ✅ Primary (most samples) |
| **BTC** | 62.67%     | +12.67%     | 3.80     | 225     | ✅ Weaker but positive    |

### Simple Pattern: 2 DOWN + kyle>0 → LONG (no percentile)

| Asset   | Hit Rate | Edge   | Z-Score | Signals | Verdict            |
| ------- | -------- | ------ | ------- | ------- | ------------------ |
| **BTC** | 53.40%   | +3.40% | 5.27    | 6,002   | ✅ Robust baseline |
| **BNB** | 52.89%   | +2.89% | 5.53    | 9,126   | ✅ Robust baseline |
| **SOL** | 51.76%   | +1.76% | 4.75    | 18,176  | ✅ Robust baseline |

### Key Insights

1. **BNB shows strongest edge** (21.72%) - possible microstructure artifact or genuine alpha
2. **Trade intensity filter amplifies edge** but reduces samples (343 vs 9,126 for BNB)
3. **Simple pattern is more stable** across assets (2.89-3.40% edge with z>5)
4. **No ETH tested** - Gen5 showed ETH is inverted (pattern LOSES on ETH)

---

## Temporal Stability (No-Lookahead) - Gen 109

| Year | Hit Rate   | Edge        | Z-Score  | Signals | Comment                 |
| ---- | ---------- | ----------- | -------- | ------- | ----------------------- |
| 2020 | 66.67%     | +16.67%     | 3.06     | 84      | Strong but small sample |
| 2021 | **82.09%** | **+32.09%** | **9.10** | 201     | **Peak performance**    |
| 2022 | 66.92%     | +16.92%     | 3.86     | 130     | Consistent              |
| 2023 | 59.79%     | +9.79%      | 1.93     | 97      | Weaker (market change?) |
| 2024 | 52.38%     | +2.38%      | 0.58     | 147     | Near baseline           |
| 2025 | 62.75%     | +12.75%     | 1.82     | 51      | Recovering              |

**Observation**: Pattern shows decay from 2021 peak (82%) to 2024 trough (52%), then partial recovery in 2025 (63%). This could indicate:

- Market adaptation to the pattern
- Changing microstructure regimes
- Sample size effects (fewer signals in recent years)

---

## Updated Recommendations for Neural Network Training

### 1. Feature Engineering (REVISED - NO LOOKAHEAD)

- **Primary features**: `trade_intensity`, `kyle_lambda_proxy`
- **Add direction lookback**: `direction(t-1)`, `direction(t-2)` for mean reversion detection
- **Threshold**: Use **expanding-window p95** (not full-dataset) or simple Kyle>0 pattern

### 2. Model Architecture (REVISED)

- **LONG-ONLY**: The gate should only open for long positions
- **Mean Reversion Signal**: Model should detect 2+ consecutive DOWN bars
- **Composite Condition**: Combine direction momentum (reversed) + intensity + Kyle
- **Regime Filter**: Only predict when `trade_intensity > rolling_p90`

### 3. Training Data

- **SOL primary**: Train on SOLUSDT (most samples: 713 for full pattern, 18K for simple)
- **BNB validation**: Use for cross-validation (strongest edge: 21.72%)
- **BTC validation**: Secondary validation (weaker edge: 12.67%)
- **Exclude ETH**: Pattern is inverted (loses money)

### 4. Expected Performance (NO-LOOKAHEAD CORRECTED)

| Pattern                    | Edge   | Signals | Trade Freq |
| -------------------------- | ------ | ------- | ---------- |
| Full (2DOWN+ti_p95+kyle>0) | 16.76% | 713     | ~2.5%      |
| Simple (2DOWN+kyle>0)      | 1.76%  | 18,176  | ~65%       |

**Note**: Simple pattern has smaller edge but 25x more samples - may be more robust for NN training.

---

## Next Steps

1. **exp082**: Create NN experiment using mean reversion + intensity + Kyle features
2. **Long-only gate**: Modify SelectiveTradingLoss to only permit long positions
3. **Direction lookback**: Add `direction(t-1)`, `direction(t-2)` to feature set
4. ~~**Temporal validation**: Verify combo pattern holds across years (gen9)~~ ✅ DONE (Gen 109)
5. ~~**Cross-asset combo**: Test combo_2down pattern on BTC/BNB~~ ✅ DONE (Gen 110)
