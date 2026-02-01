# Multi-Factor Range Bar Pattern Research Summary

**Status**: Complete
**Last Updated**: 2026-02-01

---

## Executive Summary

**Combined two-factor filtering (RV regime + multi-threshold alignment) yields 2.4x more OOD robust patterns than single-factor approaches, with 96% out-of-sample retention.**

| Research Area                    | Issue    | Universal Patterns | Audit Status    |
| -------------------------------- | -------- | ------------------ | --------------- |
| SMA/RSI Regime                   | #52      | 11                 | VALIDATED       |
| RV Volatility Regime             | #54      | 12                 | VALIDATED       |
| Multi-Threshold Alignment        | #55      | 11                 | VALIDATED       |
| **Combined (RV + Alignment)**    | #54, #55 | **26-29**          | **VALIDATED**   |
| 3-Bar Patterns                   | #54, #55 | 8                  | VALIDATED       |
| **RV + 3-Bar Patterns**          | #54, #55 | **24**             | **VALIDATED**   |
| Alignment + 3-Bar Patterns       | #54, #55 | 20                 | PENDING         |
| **Three-Factor (RV+Align+3bar)** | #54, #55 | **49**             | **PENDING**     |
| TDA Regime-Conditioned           | #56      | 23 (within regime) | **INVALIDATED** |
| Multi-Granularity (50/100/200)   | #52      | 0                  | N/A             |
| Microstructure Features          | #52, #56 | 0                  | NO PATTERNS     |
| Cross-Threshold Alignment        | #52, #56 | 0                  | **INVALIDATED** |

### Key Finding

Two-factor combination outperforms single-factor by 2.4x:

- Single factor: 11-12 patterns each
- Combined: 26-29 patterns (2.4x improvement)
- All validated via adversarial audit

---

## Research Timeline

| Date       | Milestone                        | Commit           |
| ---------- | -------------------------------- | ---------------- |
| 2026-01-30 | SMA/RSI regime analysis complete | Issue #52 closed |
| 2026-01-31 | RV regime analysis + audit       | 7c30a46          |
| 2026-01-31 | Multi-threshold alignment        | b4fc08a          |
| 2026-01-31 | Combined analysis + audit        | c5178b6          |
| 2026-01-31 | 3-bar pattern analysis           | 4f984ad          |
| 2026-01-31 | 3-bar pattern audit              | f2c8ba9          |
| 2026-01-31 | 3-bar + alignment analysis       | ff9a538          |
| 2026-01-31 | Three-factor pattern analysis    | 26659a8          |

---

## Methodology

### OOD Robustness Criteria

All patterns must satisfy:

- |t-statistic| >= 5 in ALL quarterly periods
- Same sign (direction) across ALL periods
- Minimum 100 samples per period
- Cross-validated on ALL 4 symbols (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT)

### Transaction Cost Assumption

- Round-trip cost: 15 dbps (high VIP tier on Binance)
- Net returns = Gross returns - 1.5 bps

### Data Coverage

| Symbol  | Period                   | 50 dbps | 100 dbps | 200 dbps |
| ------- | ------------------------ | ------- | -------- | -------- |
| BTCUSDT | 2022-01-01 to 2026-01-31 | 4.2M    | 1.4M     | 303K     |
| ETHUSDT | 2022-01-01 to 2026-01-31 | 6.0M    | 2.0M     | 499K     |
| SOLUSDT | 2023-06-01 to 2026-01-31 | 6.3M    | 2.0M     | 540K     |
| BNBUSDT | 2022-01-01 to 2026-01-31 | 4.6M    | 1.4M     | 389K     |

---

## Research Areas

### 1. SMA/RSI Market Regime (Issue #52)

**Hypothesis**: Deterministic regime filters may reveal OOD robust pattern subsets.

**Regime Definition**:

- SMA 20/50 crossovers for trend direction
- RSI 14 for momentum (oversold/overbought)
- 7 regime states: chop, bull_neutral, bear_neutral, bull_hot, bear_cold, bull_cold, bear_hot

**Results**: 11 universal patterns

| Regime       | Patterns           |
| ------------ | ------------------ |
| chop         | DD, DU, UD, UU (4) |
| bull_neutral | DD, DU, UD, UU (4) |
| bear_neutral | DD, DU, UU (3)     |

**Status**: CLOSED, VALIDATED

### 2. RV Volatility Regime (Issue #54)

**Hypothesis**: Volatility-based regimes may reveal different or complementary patterns.

**Regime Definition**:

- Realized Volatility (20-bar rolling std of log returns)
- Percentile classification: quiet (<25th), active (25-75th), volatile (>75th)

**Results**: 12 universal patterns (ALL 4 patterns in ALL 3 regimes)

| Regime   | Patterns       |
| -------- | -------------- |
| quiet    | DD, DU, UD, UU |
| active   | DD, DU, UD, UU |
| volatile | DD, DU, UD, UU |

**Audit Results**:

- Parameter sensitivity: 12/12 patterns across 5 param sets
- OOS validation: 100% retention rate
- Bootstrap CI: 100% exclude zero

**Status**: COMPLETE, VALIDATED

### 3. Multi-Threshold Alignment (Issue #55)

**Hypothesis**: Patterns aligned across multiple thresholds (50, 100, 200 dbps) are more robust.

**Alignment Classification**:

- aligned_up: U at ALL thresholds
- aligned_down: D at ALL thresholds
- partial_up: U at 2/3 thresholds
- partial_down: D at 2/3 thresholds
- mixed: No clear alignment

**Results**: 11 universal aligned patterns

**Key Finding**: Alignment provides marginal improvement (~0.5-1 bps) but confirms validity.

**Status**: COMPLETE, VALIDATED

### 4. Combined RV + Alignment

**Hypothesis**: Two-factor filtering produces stronger signals than either alone.

**Results**: 26-29 universal patterns (2.4x improvement)

**Top Patterns by Net Return**:

| Pattern                   | N    | Net (bps) |
| ------------------------- | ---- | --------- |
| active_partial_up\|DU     | 5K   | +12.90    |
| quiet_partial_up\|DU      | 3K   | +12.52    |
| volatile_aligned_down\|DU | 235K | +11.77    |
| active_aligned_down\|DU   | 407K | +10.54    |
| active_aligned_up\|UU     | 481K | +7.31     |

**Audit Results**:

- Parameter sensitivity: 26 patterns across 5 param sets
- OOS validation: 96.0% retention rate (24/25)
- Bootstrap CI: 100% (36/36) exclude zero

**Status**: COMPLETE, VALIDATED

### 5. 3-Bar Pattern Analysis (Issue #54, #55)

**Hypothesis**: Longer patterns (3-bar) may provide better predictive power than 2-bar.

**Pattern Definition**:

- 8 possible patterns: DDD, DDU, DUD, DUU, UDD, UDU, UUD, UUU
- Direction sequence over 3 consecutive bars

**Results**: 8 universal 3-bar patterns (2x improvement over 2-bar)

| Pattern | N    | Net (bps) |
| ------- | ---- | --------- |
| DUD     | 761K | +10.93    |
| DUU     | 812K | +9.99     |
| UUD     | 812K | +7.29     |
| UUU     | 1.0M | +7.00     |
| DDU     | 755K | +4.60     |
| UDU     | 769K | +4.51     |
| UDD     | 802K | +4.10     |
| DDD     | 783K | +3.34     |

**Key Finding**: DU-ending patterns (DUD, DUU) dominate with +10-11 bps net returns.

**RV + 3-Bar Combination**: 24 universal patterns (3x improvement over 3-bar alone)

**Audit Results**:

- Parameter sensitivity: 8/8 3-bar, 24/24 RV+3-bar across 5 param sets
- OOS validation: 100% retention for both pattern types
- Bootstrap CI: All patterns exclude zero

**Status**: COMPLETE, VALIDATED

---

## Pattern Hierarchy

```
Level 1: Base Patterns (2-bar)
├── DD (continuation down)
├── DU (reversal up) ← MOST PROFITABLE
├── UD (reversal down)
└── UU (continuation up) ← SECOND MOST PROFITABLE

Level 2: Single-Factor Filters
├── SMA/RSI Regime: 11 universal patterns
├── RV Regime: 12 universal patterns
└── Multi-Threshold Alignment: 11 universal patterns

Level 3: Two-Factor Combination
├── RV Regime + Alignment: 26-29 universal patterns (2.4x)
└── RV Regime + 3-Bar: 24 universal patterns (3x over 3-bar)

Level 4: Extended Patterns (3-bar)
├── DUD (reversal-continuation) ← MOST PROFITABLE (+10.93 bps)
├── DUU (reversal-continuation) ← SECOND MOST (+9.99 bps)
├── UUD, UUU (continuation patterns) ← +7 bps
└── DDU, UDU, UDD, DDD ← +3-5 bps
```

---

## Trading Recommendations

### Signal Selection

| Priority | Pattern Type                   | Net Return | Recommendation   |
| -------- | ------------------------------ | ---------- | ---------------- |
| 1        | DU (reversal) + any filter     | +10-13 bps | Primary signal   |
| 2        | UU (continuation) + any filter | +7 bps     | Secondary signal |
| 3        | DD, UD                         | Negative   | Do NOT trade     |

### Filter Application

| Scenario        | Filter               | Expected Patterns |
| --------------- | -------------------- | ----------------- |
| Quick filter    | RV regime only       | 12 patterns       |
| High confidence | RV + alignment       | 26 patterns       |
| Conservative    | Full alignment (3/3) | Higher confidence |

### Position Sizing (Kelly)

| Pattern Type   | Kelly Fraction | Recommended Size |
| -------------- | -------------- | ---------------- |
| DU patterns    | 0.99+          | Full Kelly       |
| UU patterns    | 0.87-0.92      | ~90% Kelly       |
| DD/UD patterns | 0.0            | Do not trade     |

---

## Scripts Reference

### Pattern Analysis Scripts

| Script                                               | Purpose                   | Status      |
| ---------------------------------------------------- | ------------------------- | ----------- |
| `scripts/market_regime_patterns_polars.py`           | SMA/RSI regime analysis   | INVALIDATED |
| `scripts/volatility_regime_analysis_polars.py`       | RV regime analysis        | INVALIDATED |
| `scripts/volatility_regime_audit_polars.py`          | RV regime audit           | INVALIDATED |
| `scripts/multi_threshold_pattern_analysis_polars.py` | Multi-threshold alignment | INVALIDATED |
| `scripts/combined_regime_analysis_polars.py`         | Combined analysis         | INVALIDATED |
| `scripts/combined_pattern_audit_polars.py`           | Combined audit            | INVALIDATED |
| `scripts/cross_regime_correlation_polars.py`         | Regime correlation        | INVALIDATED |
| `scripts/three_bar_pattern_analysis_polars.py`       | 3-bar pattern analysis    | INVALIDATED |
| `scripts/three_bar_pattern_audit_polars.py`          | 3-bar pattern audit       | INVALIDATED |
| `scripts/three_factor_pattern_analysis_polars.py`    | Three-factor analysis     | INVALIDATED |
| `scripts/multifactor_patterns.py`                    | Multi-factor patterns     | INVALIDATED |
| `scripts/volume_conditioned_patterns.py`             | Volume conditioning       | INVALIDATED |
| `scripts/cross_asset_correlation_polars.py`          | Crypto-Forex correlation  | INVALIDATED |

### Forex Data Pipeline Scripts

| Script                                      | Purpose                    | Status    |
| ------------------------------------------- | -------------------------- | --------- |
| `scripts/download_exness_eurusd.py`         | Download Exness tick data  | COMPLETED |
| `scripts/process_exness_eurusd_to_cache.py` | Generate EURUSD range bars | COMPLETED |
| `scripts/upload_eurusd_to_clickhouse.py`    | Upload to ClickHouse       | COMPLETED |

### TDA Research Scripts

| Script                                            | Purpose                    | Status      |
| ------------------------------------------------- | -------------------------- | ----------- |
| `scripts/tda_structural_break_analysis_polars.py` | TDA structural breaks      | DESCRIPTIVE |
| `scripts/tda_regime_pattern_analysis_polars.py`   | TDA regime patterns        | INVALIDATED |
| `scripts/tda_hurst_by_regime_polars.py`           | TDA Hurst analysis         | DIAGNOSTIC  |
| `scripts/tda_conditioned_patterns.py`             | TDA-conditioned patterns   | INVALIDATED |
| `scripts/tda_volatility_forecast.py`              | TDA velocity → RV forecast | INVALIDATED |

### Audit Scripts (Adversarial Validation)

| Script                                      | Purpose                    | Finding     |
| ------------------------------------------- | -------------------------- | ----------- |
| `scripts/duration_autocorrelation_audit.py` | Duration persistence audit | MECHANICAL  |
| `scripts/duration_autocorrelation.py`       | Duration autocorrelation   | INVALIDATED |
| `scripts/duration_volatility_prediction.py` | Duration → RV prediction   | 0 ODD       |
| `scripts/fdr_corrected_patterns.py`         | FDR correction analysis    | 0 PASS      |
| `scripts/microstructure_patterns.py`        | Microstructure features    | 0 ODD       |
| `scripts/temporal_safe_patterns_polars.py`  | Temporal safety check      | INVALIDATED |

### Cache Population Scripts

| Script                             | Purpose                       |
| ---------------------------------- | ----------------------------- |
| `scripts/fill_gaps_littleblack.py` | Fill cache gaps (littleblack) |
| `scripts/fill_gaps_2025_2026.py`   | Fill 2025-2026 data gaps      |
| `scripts/fill_gaps_retry.py`       | Retry failed segments         |

---

## GitHub Issues

| Issue | Title                                            | Status |
| ----- | ------------------------------------------------ | ------ |
| #52   | Market Regime Filter for ODD Robust Patterns     | CLOSED |
| #54   | Volatility Regime Filter for ODD Robust Patterns | CLOSED |
| #55   | Multi-Threshold Pattern Confirmation Signals     | CLOSED |
| #56   | TDA Structural Break Detection for Range Bars    | CLOSED |
| #57   | Research Complete: ZERO ODD Robust Patterns      | CLOSED |

---

## MRH Framework Validation (Latest)

**Status**: COMPLETE (2026-01-31)

Applied Minimum Reliable Horizon framework from Gemini research:

| Metric | Description                                     | Result                      |
| ------ | ----------------------------------------------- | --------------------------- |
| PSR    | Probabilistic Sharpe Ratio (skew/kurt adjusted) | 36 patterns >= 0.95         |
| MinTRL | Minimum Track Record Length                     | All patterns exceed         |
| DSR    | Deflated Sharpe Ratio (multiple testing)        | 36 patterns >= 0.95         |
| Gap    | T_required - T_available                        | All 36 negative (CONVERGED) |

**Production-Ready Patterns**: 36 three-factor patterns pass all criteria.

**Script**: `scripts/psr_mintrl_analysis_polars.py` - Commit 0557b27

### ADWIN Regime Detection

**Status**: COMPLETE (2026-01-31)

Applied parameter-free ADWIN (Adaptive Windowing with Hoeffding bounds):

| Symbol  | Bars  | ADWIN Regime Changes | Fixed-Window Changes |
| ------- | ----- | -------------------- | -------------------- |
| BTCUSDT | 1.38M | 0                    | 10,795               |
| ETHUSDT | 2.00M | 0                    | 56,541               |
| SOLUSDT | 1.98M | 0                    | 58,256               |
| BNBUSDT | 1.43M | 0                    | 42,424               |

**Key Finding**: ADWIN detects ZERO distributional shifts across 4 years of data.
This confirms T_available (supply) is the full dataset, explaining massive negative gaps.

**Script**: `scripts/adwin_regime_detection_polars.py` - Commit de1f721

### Hurst Exponent Analysis (Critical)

**Status**: COMPLETE (2026-01-31)

Applied R/S and DFA methods to compute Hurst exponent:

| Context             | Hurst (H) | Interpretation  |
| ------------------- | --------- | --------------- |
| Raw returns         | 0.50-0.51 | Random walk     |
| Pattern-conditioned | 0.79      | Strong trending |

**Critical Finding**: Pattern-conditioned returns show H ~ 0.79 (long memory)

- T_eff = T^(2(1-H)) = T^0.42
- 761K samples → ~268 effective samples
- **ZERO raw 3-bar patterns survive Hurst adjustment**

This reclassifies patterns from CONVERGED to **BLIND SPOT** per MRH Framework.

**Scripts**:

- `scripts/hurst_exponent_analysis_polars.py` - Commit cc66023
- `scripts/hurst_adjusted_psr_analysis_polars.py` - Commit c2709da

### TDA Structural Break Detection

**Status**: COMPLETE (2026-01-31)

Applied Topological Data Analysis (Persistent Homology) for geometric structural break detection:

| Symbol  | N Bars  | Avg L2(H1) | Max L2(H1) | TDA Breaks | ADWIN Breaks |
| ------- | ------- | ---------- | ---------- | ---------- | ------------ |
| BTCUSDT | 323,112 | 3.54       | 8.78       | 3          | 0            |
| ETHUSDT | 646,417 | 4.33       | 11.91      | 3          | 0            |
| SOLUSDT | 675,738 | 5.35       | 6.44       | 3          | 0            |
| BNBUSDT | 313,184 | 5.13       | 6.44       | 3          | 0            |

**Key Finding**: TDA detects 12 structural breaks that ADWIN completely missed.

This suggests:

- H1 (loop) features capture cyclic market structure
- Topological features precede statistical moment shifts
- Geometric instabilities exist despite stable moments
- Patterns may be less reliable than ADWIN-only analysis suggests

**Methodology**:

- Takens delay embedding (dim=3) of log returns
- Vietoris-Rips filtration via ripser
- L2 norm of H1 persistence landscape
- Velocity threshold (95th percentile) for break detection

**Reference**: Gidea & Katz (2018) - Lp-norm of H1 features predicts crashes 250 trading days before statistical volatility models.

**Script**: `scripts/tda_structural_break_analysis_polars.py` - Commit 1480ee9

**Note**: TDA is O(n^2) complexity. For pattern-specific analysis, use GPU with giotto-tda CUDA backend.

### TDA Regime Pattern Analysis

**Status**: COMPLETE (2026-02-01)

Analyzed pattern performance across TDA-defined regimes (full multi-year data):

| Symbol    | N Bars    | TDA Breaks | Significant Pattern Differences |
| --------- | --------- | ---------- | ------------------------------- |
| BTCUSDT   | 1,382,518 | 10         | 1/4 (25%)                       |
| ETHUSDT   | 1,996,522 | 10         | 3/4 (75%)                       |
| SOLUSDT   | 1,977,403 | 10         | 4/4 (100%)                      |
| BNBUSDT   | 1,432,019 | 10         | 4/4 (100%)                      |
| **Total** | **6.8M**  | **40**     | **12/16 (75%)**                 |

**Key Findings**:

- 75% of pattern-regime comparisons show statistically significant differences (p < 0.05)
- Pattern returns can flip sign between TDA regimes (e.g., DU: -1.27 bps → +0.89 bps)
- Extreme regimes (likely major market events) show 10x higher pattern returns
- Validates TDA as early warning for pattern degradation

**Script**: `scripts/tda_regime_pattern_analysis_polars.py` - Commit 7055673

**Full details**: [docs/research/tda-regime-patterns.md](/docs/research/tda-regime-patterns.md)

### TDA Break Event Alignment

**Status**: COMPLETE (2026-02-01)

Aligned TDA-detected breaks with major market events:

| Event             | Date       | Symbols Detecting TDA Break |
| ----------------- | ---------- | --------------------------- |
| Luna/UST Collapse | 2022-05-09 | BTCUSDT (3d before)         |
| FTX Bankruptcy    | 2022-11-11 | ETHUSDT (4d after)          |
| Bitcoin Halving   | 2024-04-19 | BNBUSDT, ETHUSDT, SOLUSDT   |
| Yen Carry Unwind  | 2024-08-05 | BTCUSDT, ETHUSDT, SOLUSDT   |
| Trump Election    | 2024-11-05 | ETHUSDT, SOLUSDT            |

**Key Finding**: 27.5% (11/40) of TDA breaks correlate with known market events. Luna collapse detected 3 days BEFORE crash (early warning).

**Script**: `scripts/tda_break_event_alignment_polars.py` - Commit 3960eb6

### TDA Hurst Exponent by Regime

**Status**: COMPLETE (2026-02-01)

Computed Hurst exponent for each TDA-defined regime:

| Category                | Count | Percentage | Interpretation             |
| ----------------------- | ----- | ---------- | -------------------------- |
| Trending (H > 0.55)     | 7     | 16%        | Momentum strategies viable |
| Random Walk (0.45-0.55) | 37    | 84%        | No memory advantage        |
| Mean-Reverting (H<0.45) | 0     | 0%         | None detected              |

**Key Finding**: Hurst is stable across TDA regimes (range 0.097), confirming TDA breaks detect non-memory structural changes.

**Script**: `scripts/tda_hurst_by_regime_polars.py` - Commit 5a731fc

### Multi-Factor Multi-Granularity Patterns

**Status**: COMPLETE (2026-02-01)

Tested combinations of range bars at different thresholds (50, 100, 200 dbps):

| Factor Type   | Patterns Tested | ODD Robust | Universal (4 symbols) |
| ------------- | --------------- | ---------- | --------------------- |
| Single-factor | 16              | 0          | 0                     |
| Multi-factor  | 32              | 0          | 0                     |

**Key Finding**: Multi-factor patterns do NOT improve ODD robustness. Both fail due to sign reversals.

**Script**: `scripts/multifactor_multigranularity_patterns.py` - Commit 61847d4

**Full details**: [docs/research/multifactor-patterns.md](/docs/research/multifactor-patterns.md)

### TDA Regime-Conditioned ODD Robustness

**Status**: INVALIDATED BY ADVERSARIAL AUDIT (2026-02-01)

Tested whether patterns achieve ODD robustness WITHIN TDA-defined stable regimes:

| Metric                              | Value | Percentage |
| ----------------------------------- | ----- | ---------- |
| Pattern-regime combinations tested  | 176   | -          |
| Same sign across sub-periods        | 87    | 49.4%      |
| ODD robust (same sign + \|t\| >= 5) | 23    | 13.1%      |

**Adversarial Audit Findings (6 parallel agents)**:

| Audit Focus            | Status      | Critical Issue                                       |
| ---------------------- | ----------- | ---------------------------------------------------- |
| Data Leakage (Regime)  | **FAIL**    | Pattern uses `shift(-1)` = future bar direction      |
| Local Bias/Overfitting | **FAIL**    | 1,920+ tests, params appear optimized post-hoc       |
| Mechanical vs Alpha    | **PARTIAL** | 1-bar returns tautological; 3-10 bar IS genuine      |
| Data Leakage (TDA)     | **FAIL**    | 95th percentile computed on entire 4-year dataset    |
| Parameter Sensitivity  | **FAIL**    | threshold_pct=95, quarter_size=4 hardcoded, no sweep |
| Statistical Validity   | **FAIL**    | 8.4x FDR inflation, 2/23 cross-symbol replicate      |

**Key Issues**:

1. **Data Leakage**: TDA threshold computed globally (includes future volatility)
2. **Multiple Testing**: 176 tests with no Bonferroni/FDR correction
3. **Autocorrelation**: Effective samples ≈ 6.3 per sub-period (H ~ 0.79)
4. **Cross-Symbol Failure**: Only 8.7% replicate vs 25% expected by chance
5. **Parameter Snooping**: Critical params hardcoded with no sensitivity analysis

**Verdict**: The 23 patterns are statistical artifacts, NOT tradeable signals.

**Temporal-Safe Validation (2026-02-01)**:

Script `scripts/temporal_safe_patterns_polars.py` re-ran pattern analysis using ONLY past data:

- Pattern calculation: `direction[i-1] + direction[i]` (NOT `direction[i] + direction[i+1]`)
- Forward return: still `shift(-1)` (predicting bar i+1)

| Metric                | Leaky (original) | Temporal-Safe |
| --------------------- | ---------------- | ------------- |
| 2-bar patterns tested | 16               | 16            |
| 2-bar ODD robust      | 4                | **0**         |
| 3-bar patterns tested | 32               | 32            |
| 3-bar ODD robust      | 8                | **0**         |
| Universal 2-bar       | 4                | **0**         |
| Universal 3-bar       | 8                | **0**         |

**Conclusion**: When patterns use only past data, ZERO patterns achieve ODD robustness.
This confirms the original patterns were artifacts of data leakage.

**Audit Documents**:

- `docs/research/tda-parameter-sensitivity-audit.md` (530 lines)
- `STATISTICAL_VALIDITY_AUDIT.md` (591 lines)

**Scripts**:

- `scripts/tda_conditioned_patterns.py` (leaky) - Commit 89a0b19
- `scripts/temporal_safe_patterns_polars.py` (fixed pattern) - Commit bcf1c0d
- `scripts/tda_rolling_threshold.py` (fixed TDA threshold) - Commit pending

**Rolling TDA Threshold Validation (2026-02-01)**:

Script `scripts/tda_rolling_threshold.py` compares global vs rolling threshold for TDA break detection:

| Metric                      | Global (leaky) | Rolling (temporal-safe) |
| --------------------------- | -------------- | ----------------------- |
| Pattern-regime combinations | 176            | 200                     |
| ODD robust                  | 23 (13.1%)     | 35 (17.5%)              |

Note: Rolling threshold finds MORE ODD robust patterns because it creates more granular
regimes (more breaks detected with lower early thresholds). However, both scripts use
temporal-safe pattern calculation (shift(1) not shift(-1)).

**FDR Correction Validation (2026-02-01)**:

Script `scripts/fdr_corrected_patterns.py` implements Benjamini-Hochberg FDR control:

| Metric                        | Value |
| ----------------------------- | ----- |
| Total patterns tested         | 16    |
| Total period tests            | 252   |
| Bonferroni t-threshold        | 3.72  |
| Patterns with same sign       | 0     |
| Raw ODD robust (\|t\| >= 5)   | 0     |
| FDR ODD robust (BH corrected) | 0     |

**Conclusion**: With temporal-safe pattern calculation, ZERO patterns achieve ODD
robustness, regardless of statistical correction method.

### Cross-Asset Correlation Analysis (Issue #145)

**Status**: INVALIDATED (2026-02-01)

Tested whether crypto-forex correlation regimes reveal OOD robust patterns.

**Methodology**:

1. Aligned crypto (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT) with forex (EURUSD) by hourly buckets
2. Computed rolling 24-hour correlation between crypto and forex returns
3. Classified correlation regimes: positive (>0.3), negative (<-0.3), neutral
4. Tested if correlation regime + direction patterns predict forward returns
5. Validated OOD robustness via quarterly cross-validation

**Data Coverage**:

- EURUSD: 25,508 bars (100 dbps, 2022-01-03 to 2026-01-30)
- Crypto: 1.4M-4M bars each (100 dbps, 2022-01-01 to 2026-01-30)
- Aligned samples: ~11,800 hourly observations per pair

**Correlation Regime Distribution**:

| Regime        | BTCUSDT | ETHUSDT | SOLUSDT | BNBUSDT |
| ------------- | ------- | ------- | ------- | ------- |
| neutral_corr  | 66%     | 64%     | 67%     | 70%     |
| positive_corr | 32%     | 33%     | 30%     | 27%     |
| negative_corr | 2%      | 2%      | 3%      | 3%      |

**Results**:

| Symbol  | Patterns Tested | OOD Robust |
| ------- | --------------- | ---------- |
| BTCUSDT | 4               | 0          |
| ETHUSDT | 4               | 0          |
| SOLUSDT | 4               | 0          |
| BNBUSDT | 4               | 0          |

**Key Finding**: ZERO OOD robust patterns found across any crypto-forex correlation regime.

**Interpretation**:

1. Crypto and forex are largely uncorrelated (66-70% neutral correlation regime)
2. Even when correlation regimes are identified, they don't predict forward returns
3. Cross-asset validation confirms that range bar patterns lack predictive power
4. Adding a fifth asset class (forex) doesn't overcome the structural limitations

**Script**: `scripts/cross_asset_correlation_polars.py`

---

## Future Research Directions

- [x] 3-bar pattern analysis (COMPLETE - 8 universal, 24 with RV)
- [x] Adversarial audit of 3-bar patterns (VALIDATED - 100% retention)
- [x] MRH Framework validation (COMPLETE - 36 production-ready pre-Hurst)
- [x] ADWIN regime detection (COMPLETE - zero regime changes = stable distribution)
- [x] Hurst exponent analysis (COMPLETE - H ~ 0.79 for pattern returns)
- [x] Hurst-adjusted validation (COMPLETE - zero patterns survive)
- [x] Three-factor Hurst analysis (COMPLETE - H ~ 0.71, still insufficient)
- [x] TDA structural break detection (COMPLETE - 12 breaks vs 0 ADWIN)
- [x] TDA regime pattern analysis (COMPLETE - 40 breaks, 75% patterns regime-dependent)
- [x] TDA break event alignment (COMPLETE - 27.5% correlation with major events)
- [x] TDA Hurst by regime (COMPLETE - H stable, non-memory changes)
- [x] Multi-factor multi-granularity (COMPLETE - no improvement over single)
- [x] TDA regime-conditioned ODD (COMPLETE - 23 patterns ODD robust within regimes)
- [x] Temporal-safe pattern validation (COMPLETE - 0 patterns survive after fixing leakage)
- [x] FDR correction validation (COMPLETE - confirms 0 ODD robust patterns)
- [x] Forex symbol validation (COMPLETE - 0 ODD robust patterns, Issue #145)

---

## Conclusion

The multi-factor approach to range bar pattern research has yielded significant results:

### OOD Robustness (Traditional Metrics)

1. **Single-factor filters** each reveal 11-12 OOD robust patterns
2. **Two-factor combination** (RV + alignment) yields 26-29 patterns (2.4x improvement)
3. **3-bar patterns** provide 2x improvement over 2-bar (8 vs 4 universal)
4. **RV + 3-bar combination** yields 24 universal patterns (3x over 3-bar alone)
5. **Three-factor combination** yields 49 universal patterns (2x over best two-factor)
6. **DU-ending patterns** (DU, DUD, DUU) are consistently the most profitable (+10-13 bps net)

### MRH Framework Findings (Critical Update)

1. **Hurst exponent analysis reveals long memory** (H ~ 0.79 for pattern returns)
2. **Effective sample size drastically reduced** (T_eff = T^0.42)
3. **ZERO patterns survive** Hurst-adjusted PSR/MinTRL validation
4. **Three-factor filtering reduces H to ~0.71** but still insufficient
5. **All patterns in BLIND SPOT state** per MRH Framework

### Interpretation

The patterns show strong OOD robustness by traditional metrics (t-statistics, quarterly retention).
However, the MRH Framework reveals autocorrelation in pattern-conditioned returns that
invalidates the IID assumption underlying PSR/MinTRL calculations.

**Trading Recommendation**: Patterns may still be tradeable with appropriate position sizing
that accounts for reduced effective degrees of freedom. The edge is real but sample-size
claims must be discounted by the Hurst exponent factor.

### Adversarial Audit Findings (Critical Update - 2026-02-01)

**The pattern research was INVALIDATED by adversarial audit.** Six parallel audit agents
identified critical methodological flaws:

1. **Data Leakage in Pattern Calculation**: Patterns used `shift(-1)` (future bar direction)
   instead of `shift(1)` (past bar direction). With temporal-safe calculation, ZERO
   patterns achieve ODD robustness.

2. **Data Leakage in TDA Threshold**: Global 95th percentile computed on entire 4-year
   dataset leaks future volatility into regime assignments.

3. **Multiple Testing Inflation**: 176+ tests without Bonferroni/FDR correction.

4. **Parameter Snooping**: threshold_pct=95, quarter_size=4 hardcoded without sensitivity
   analysis.

5. **Autocorrelation Ignored**: Hurst H~0.79 reduces effective samples to ~6.3 per sub-period.

6. **Cross-Symbol Failure**: Only 8.7% of patterns replicate vs 25% expected by chance.

**Conclusion**: The 23-49 "ODD robust" patterns reported in earlier sections are statistical
artifacts caused by data leakage. With proper temporal-safe methodology:

| Metric                   | Original (leaky) | Temporal-safe |
| ------------------------ | ---------------- | ------------- |
| 2-bar ODD robust         | 4                | **0**         |
| 3-bar ODD robust         | 8                | **0**         |
| Same sign across periods | ~50%             | **0%**        |

**No tradeable patterns exist in range bar direction sequences.**

### Microstructure Feature Analysis (2026-02-01)

After direction patterns (U/D) were invalidated, analysis pivoted to microstructure features
computed during bar construction. These features capture genuine market dynamics without
lookahead bias:

| Feature              | Description                     | Range     |
| -------------------- | ------------------------------- | --------- |
| ofi                  | Order Flow Imbalance (buy-sell) | [-1, 1]   |
| vwap_close_deviation | Close vs VWAP deviation         | ~[-1, 1]  |
| trade_intensity      | Trades per second               | [0, +inf) |
| aggression_ratio     | Buy count vs sell count         | [0, 100]  |
| turnover_imbalance   | Buy vs sell turnover            | [-1, 1]   |

**Methodology**:

- Tercile discretization (LOW/MID/HIGH) based on feature quantiles
- Forward returns computed correctly via `shift(-1)` on close prices
- t-statistics computed per quarterly period
- ODD robust = same sign + |t| >= 3.0 across ALL periods

**Results** (via ClickHouse direct query):

| Metric                       | Value |
| ---------------------------- | ----- |
| Total feature-tercile combos | 60    |
| Same sign across periods     | 3     |
| ODD robust (t >= 3.0)        | 0     |
| ODD robust (t >= 2.0)        | 0     |

**Key Finding**: Even lowering threshold to |t| >= 2.0, ZERO microstructure features achieve
ODD robustness. The Min |t| column shows every feature-tercile has at least one period with
near-zero t-stat, meaning the predictive effect disappears or reverses in some quarters.

**Notable Non-Robust Patterns**:

- `vwap_close_deviation HIGH` for SOLUSDT: Mean t = -20.81, but varies across periods
- `aggression_ratio HIGH` for ETHUSDT: Mean t = +9.49, but Min |t| = 0.21
- `ofi HIGH` for ETHUSDT: Mean t = +6.23, but Min |t| = 0.43

These features show strong average predictive power but FAIL the ODD test due to period
variability. The effect is not consistent enough to be tradeable.

**Script**: `scripts/microstructure_clickhouse.py` - Queries ClickHouse directly to avoid
pandas memory overhead on large datasets.

### Cross-Threshold Signal Alignment (2026-02-01)

**STATUS: INVALIDATED BY FORENSIC AUDIT**

After microstructure features failed, tested whether range bars at multiple thresholds
(50/100/200 dbps) agreeing on direction provides ODD robust signals.

**Methodology**:

- Hourly bucketing: aggregate direction (majority U/D) per threshold per hour
- Alignment: all_up = all three thresholds show U, all_down = all three show D
- Forward return: next hour's close / current hour's close

**Initial Results (BUGGY)**:

| Alignment | Mean bps | Min t | ODD Robust? | Universal? |
| --------- | -------- | ----- | ----------- | ---------- |
| all_up    | +34-64   | 7.44  | YES         | 4/4        |
| all_down  | -42-74   | 12.32 | YES         | 4/4        |

**Forensic Audit Findings (CRITICAL BUG)**:

The original analysis used `argMax(close, hour_ts)` to select the "last" close price in each
hour bucket. However, since `hour_ts` is the SAME value for all bars in that hour (e.g.,
`2024-01-01 15:00:00`), ClickHouse's `argMax` selects an **arbitrary** row, not the last one.

| Audit Step            | Finding                                                              |
| --------------------- | -------------------------------------------------------------------- |
| Bar timing leakage    | PASS - all bars close within their hour bucket                       |
| Cross-hour bar span   | PASS - bars open and close in same hour                              |
| Momentum continuation | PASS - current hour return ~0, next hour is 35-44 bps (not momentum) |
| **argMax bug**        | **FAIL** - `argMax(close, hour_ts)` selects FIRST bar, not last bar  |
| **Values match**      | **FAIL** - 0% of hours have correct close price selection            |

**Corrected Results**:

When using `argMax(close, bar_ts_ms)` (the actual bar timestamp):

| Alignment | Mean bps | Std bps | ODD Robust? |
| --------- | -------- | ------- | ----------- |
| all_up    | +0.2     | 57.9    | NO          |
| all_down  | +1.1     | 61.3    | NO          |
| mixed     | -1.5     | 55.5    | NO          |

**Conclusion**: The original +35/-44 bps signal was an **artifact of the argMax bug**.
With correct close price selection, cross-threshold alignment provides ZERO predictive
power. All returns are within noise (< 2 bps mean, ~60 bps std).

**Root Cause**: The bug selected an arbitrary bar's close (often the first bar) rather
than the actual last bar in the hour. The difference between buggy and correct close
prices ranged from -210 to +169, introducing massive artificial noise that happened to
correlate with alignment direction.

**Script**: `scripts/cross_threshold_alignment.py` (contains bug, kept for reference)

**Forensic Audit Script**: Direct ClickHouse queries documented in conversation transcript.

### Return Persistence Analysis (2026-02-01)

**STATUS: INVALIDATED - Equivalent to Direction Pattern**

Tested if bar N's return tercile (LOW/MID/HIGH) predicts bar N+1's return.

**Initial Results**:

| Tercile | Periods | Same Sign | Min t | Avg t | Mean bps |
| ------- | ------- | --------- | ----- | ----- | -------- |
| HIGH    | 17      | 17/17 (+) | +40   | +132  | +6.25    |
| LOW     | 17      | 17/17 (-) | -39   | -132  | -6.28    |
| MID     | 17      | 10+/7-    | -3.5  | +0.3  | ~0       |

**Forensic Audit Findings**:

1. **Tercile = Direction**: HIGH tercile is 100% up bars, LOW is 100% down bars
2. **Return distribution**: 80%+ of bars have returns at exactly ±threshold (±10 bps)
3. **Equivalent to direction pattern**: Already invalidated by temporal-safe analysis
4. **Temporal overlap**: 38% of consecutive bars overlap, introducing mechanical correlation

**Conclusion**: Return persistence is just the direction pattern in disguise.

### Coarse-to-Fine Cascade Analysis (2026-02-01)

**STATUS: BLOCKED - ClickHouse Limitations**

Attempted to test if 200 dbps bar direction predicts 100 dbps bar returns.

**Approaches Tried**:

1. **Correlated subquery**: NOT_IMPLEMENTED in ClickHouse
2. **ASOF JOIN**: Requires equi-join column, only matches within same hour bucket
3. **Hourly aggregation**: NaN results due to hour boundary mismatches

**Hourly Cascade Results** (simplified approach):

| Direction | Mean bps | Min t | Max t | Same Sign |
| --------- | -------- | ----- | ----- | --------- |
| D         | +0.5     | -2.3  | +2.4  | Mixed     |
| U         | +0.6     | -1.5  | +2.1  | Mixed     |

**Conclusion**: No predictive power in coarse-to-fine cascade at hourly granularity.
ClickHouse limitations prevent bar-level cascade analysis.

### Research Status Summary (2026-02-01)

**All pattern research directions have been exhausted with ZERO ODD robust patterns found:**

| Approach                     | Status      | Issue                          |
| ---------------------------- | ----------- | ------------------------------ |
| Direction patterns (U/D)     | INVALIDATED | Data leakage (shift(-1))       |
| 2-bar/3-bar patterns         | INVALIDATED | Same leakage issue             |
| TDA regime conditioning      | INVALIDATED | Global threshold leakage       |
| Microstructure features      | 0 ODD       | No consistent quarterly signal |
| Cross-threshold alignment    | INVALIDATED | argMax bug in ClickHouse       |
| Return persistence (tercile) | INVALIDATED | Equivalent to direction        |
| Coarse-to-fine cascade       | BLOCKED     | ClickHouse query limitations   |

**Key Learnings**:

1. Range bar returns are concentrated at ±threshold (80%+ at boundary)
2. 38% of consecutive bars overlap temporally (mechanical correlation)
3. ClickHouse does not support correlated subqueries needed for cascade analysis
4. Pattern-conditioned returns show H ~ 0.79 (long memory), reducing effective samples

---

## Final Conclusion: Pattern Research Complete (2026-02-01)

### Definitive Negative Finding

**After exhaustive adversarial audit across 7 distinct research approaches, ZERO ODD robust
predictive patterns exist in range bar price data.**

This is a strong negative result. The research systematically tested:

- **Direction sequences** (U/D, 2-bar, 3-bar patterns)
- **Regime conditioning** (SMA/RSI, RV volatility, TDA structural breaks)
- **Multi-factor combinations** (RV + alignment, three-factor)
- **Microstructure features** (OFI, VWAP deviation, trade intensity, aggression)
- **Cross-threshold alignment** (50/100/200 dbps consensus)
- **Return persistence** (tercile momentum)
- **Coarse-to-fine cascade** (200→100 dbps direction flow)

Every approach that initially appeared promising was subsequently invalidated by forensic
audit revealing data leakage, ClickHouse query bugs, or equivalence to already-invalidated
patterns.

### Why Patterns Fail

Range bars have structural properties that make directional prediction fundamentally difficult:

1. **Boundary-Locked Returns**: 80%+ of range bar returns are at exactly ±threshold. The
   return distribution is bimodal, not continuous. This means "predicting" the return is
   equivalent to predicting direction, and direction shows no ODD robust autocorrelation.

2. **Temporal Overlap**: 38% of consecutive bars overlap in time. Bar N and Bar N+1 often
   contain the same trades, introducing mechanical correlation that inflates t-statistics
   but doesn't represent tradeable alpha.

3. **Long Memory Illusion**: Pattern-conditioned returns show Hurst H ~ 0.79, but this
   reflects the overlapping structure, not genuine predictability. The effective sample
   size is drastically reduced (T_eff = T^0.42).

4. **Microstructure Noise**: Features like OFI, VWAP deviation, and trade intensity show
   period-to-period variability that prevents ODD robustness, even when averaging strong.

### Implications

**Range bars are NOT suitable for directional prediction strategies.**

The structural properties (boundary-locked returns, temporal overlap, long memory from
overlap) mean that:

- Direction patterns will always have inflated apparent significance
- Any pattern that appears ODD robust is likely an artifact of data leakage or bugs
- Microstructure features don't provide consistent enough signals across market regimes

### Alternative Research Directions

Range bars may still be valuable for **non-directional** applications:

| Direction             | Rationale                                       | Status      |
| --------------------- | ----------------------------------------------- | ----------- |
| **Volatility Regime** | Bar duration predicts future volatility (known) | INVALIDATED |
| **Risk Management**   | TDA breaks as early warning for regime change   | INVALIDATED |
| **Execution Quality** | Bar timing for optimal order placement          | UNEXPLORED  |
| **Market Making**     | Microstructure for spread/inventory management  | UNEXPLORED  |

**Volatility prediction (Duration Autocorrelation)** - INVALIDATED (2026-02-01):

Duration autocorrelation appeared promising with 90%+ ODD robust patterns (P(persist) = 0.82-0.91
vs null 0.33). However, forensic audit revealed mechanical artifacts:

- 75-100% temporal overlap between consecutive bars
- 100% negative gaps (bar N+1 opens before bar N closes)
- SHORT tercile shows 100% persistence regardless of quantile boundary
- Lag decay erratic, not smooth (measuring construction, not volatility)

The apparent "volatility clustering" is due to range bar deferred-open semantics where the
same tick serves as both close of bar N and open of bar N+1.

Scripts: `duration_volatility_prediction.py`, `duration_autocorrelation.py`,
`duration_autocorrelation_audit.py`

**TDA early warning (Velocity → Forward RV)** - INVALIDATED (2026-02-01):

TDA H1 L2 velocity tercile does NOT predict forward realized volatility:

- All t-stats between -1.67 and +1.01 (threshold: ±3.0)
- Same sign across periods: 0%
- Min P(HIGH_RV) often 0.0, Max P up to 0.857

While TDA structural breaks correlate with market events (27.5% within 7 days of Luna collapse,
FTX bankruptcy, Bitcoin halving, etc.), TDA velocity does NOT provide actionable volatility
forecast signal.

Script: `tda_volatility_forecast.py`

### Research Integrity Note

This negative finding is itself valuable. The adversarial audit methodology developed during
this research provides a robust framework for evaluating future pattern research:

1. **Temporal-safe patterns**: Always use `shift(1)` for features, `shift(-1)` for targets
2. **FDR correction**: Apply Benjamini-Hochberg for multiple testing
3. **Cross-symbol validation**: Require replication across uncorrelated assets
4. **Parameter sensitivity**: Sweep all thresholds, don't hardcode
5. **Forensic audit**: Verify ClickHouse queries return expected values

The research investment was not wasted—it conclusively closed a research direction and
established validation standards for future work.
