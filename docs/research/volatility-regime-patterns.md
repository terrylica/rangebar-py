# Volatility Regime Patterns Research

**Issue**: #54 - Volatility Regime Filter for ODD Robust Patterns
**Status**: INVALIDATED (see pattern-research-summary.md)
**Last Updated**: 2026-02-01

---

## Executive Summary

**Realized Volatility (RV) regimes reveal 12 universal OOD robust patterns - MORE than SMA/RSI regimes (11).**

All 4 pattern types (DD, DU, UD, UU) are OOD robust across ALL 3 RV regimes (quiet, active, volatile). This is stronger than SMA/RSI where extreme regimes (bull_hot, bear_cold) have fewer patterns.

### Key Finding

| Regime Type         | Universal Patterns | Coverage                   |
| ------------------- | ------------------ | -------------------------- |
| SMA/RSI (Issue #52) | 11                 | 3 regimes × ~3.7 patterns  |
| ATR volatility      | 4                  | 1 regime × 4 patterns      |
| **RV volatility**   | **12**             | **3 regimes × 4 patterns** |
| Combined (ATR × RV) | 12                 | 3 regimes × 4 patterns     |

---

## Methodology

### Volatility Regime Definitions

#### ATR-based Volatility Regimes

| Regime     | Condition                    | Description |
| ---------- | ---------------------------- | ----------- |
| Low Vol    | ATR(14) < SMA(ATR, 50) × 0.7 | Compression |
| Normal Vol | 0.7 ≤ ratio ≤ 1.3            | Typical     |
| High Vol   | ATR(14) > SMA(ATR, 50) × 1.3 | Expansion   |

#### Realized Volatility (RV) Regimes

| Regime   | Condition                | Description  |
| -------- | ------------------------ | ------------ |
| Quiet    | RV(20) < 25th percentile | Very low vol |
| Active   | 25th-75th percentile     | Normal vol   |
| Volatile | RV(20) > 75th percentile | High vol     |

### OOD Robustness Criteria

Same as Issue #52:

- |t-stat| ≥ 5 in ALL quarterly periods
- Same sign across ALL periods
- ≥100 samples per period
- Cross-validated on ALL 4 symbols

---

## Results

### Universal OOD Robust Patterns

#### ATR-based Vol Regime (4 patterns)

Only `normal_vol` regime produces universal patterns:

| Regime         | Patterns       |
| -------------- | -------------- |
| low_vol        | 0              |
| **normal_vol** | DD, DU, UD, UU |
| high_vol       | 0              |

**Interpretation**: Extreme ATR regimes (low_vol, high_vol) are too transient or inconsistent for OOD robustness.

#### RV Regime (12 patterns) ⭐ BEST

ALL patterns are universal across ALL RV regimes:

| Regime    | Patterns       | Count  |
| --------- | -------------- | ------ |
| quiet     | DD, DU, UD, UU | 4      |
| active    | DD, DU, UD, UU | 4      |
| volatile  | DD, DU, UD, UU | 4      |
| **TOTAL** |                | **12** |

**Interpretation**: RV regime provides consistent filtering regardless of volatility level. This is a stronger result than SMA/RSI.

#### Combined Regime (12 patterns)

All patterns in `normal_vol` × any RV regime:

| Regime              | Patterns       | Count  |
| ------------------- | -------------- | ------ |
| normal_vol_quiet    | DD, DU, UD, UU | 4      |
| normal_vol_active   | DD, DU, UD, UU | 4      |
| normal_vol_volatile | DD, DU, UD, UU | 4      |
| **TOTAL**           |                | **12** |

### Per-Symbol Results

| Symbol  | Bars      | ATR Robust | RV Robust | Combined |
| ------- | --------- | ---------- | --------- | -------- |
| BTCUSDT | 1,382,518 | 6          | 12        | 14       |
| ETHUSDT | 1,996,522 | 5          | 12        | 16       |
| SOLUSDT | 3,760,244 | 5          | 12        | 14       |
| BNBUSDT | 1,432,019 | 5          | 12        | 13       |

---

## Comparison with SMA/RSI Patterns

### Pattern Coverage

| Metric                      | SMA/RSI | RV Regime    |
| --------------------------- | ------- | ------------ |
| Total universal patterns    | 11      | 12           |
| Regimes with all 4 patterns | 3/7     | **3/3**      |
| Extreme regime coverage     | Sparse  | **Complete** |

### Key Differences

1. **RV has complete coverage**: All 4 patterns work in all 3 RV regimes
2. **SMA/RSI has gaps**: bull_hot (1 pattern) and bear_cold (2 patterns)
3. **RV is simpler**: 3 regimes vs 7 for SMA/RSI
4. **RV is orthogonal**: Measures different market characteristic (volatility vs trend)

---

## Trading Implications

### Diversification Potential

The RV and SMA/RSI regimes measure different market characteristics:

- **SMA/RSI**: Trend direction + momentum
- **RV**: Volatility level

This suggests **combining both** for diversification:

```
                    RV Regime
              quiet | active | volatile
SMA/RSI  ┌─────────┼────────┼──────────┐
  chop   │ signal1 │ signal2│ signal3  │
bull_neut│ signal4 │ signal5│ signal6  │
bear_neut│ signal7 │ signal8│ signal9  │
         └─────────┴────────┴──────────┘
```

### Implementation Recommendation

1. **Primary**: Use SMA/RSI regime patterns (Issue #52) - established research
2. **Secondary**: Add RV regime as confirmation or diversification
3. **Future**: Test combined SMA/RSI × RV regimes for refined signals

---

## Scripts

| Script                                         | Purpose                            |
| ---------------------------------------------- | ---------------------------------- |
| `scripts/volatility_regime_analysis_polars.py` | Volatility regime pattern analysis |
| `scripts/combined_regime_analysis_polars.py`   | Combined SMA/RSI x RV analysis     |
| `scripts/volatility_regime_audit_polars.py`    | Adversarial audit (param/OOS/CI)   |
| `scripts/cross_regime_correlation_polars.py`   | Cross-regime correlation analysis  |
| `scripts/rv_return_profile_analysis_polars.py` | Return profile and Kelly analysis  |

---

## Combined SMA/RSI x RV Regime Analysis (2026-01-31)

**Combining trend and volatility regimes reveals 16 universal patterns.**

### Universal Combined Patterns (All 4 Symbols)

| Combined Regime     | Patterns        |
| ------------------- | --------------- |
| bear_neutral_active | DU, UD, UU      |
| bear_neutral_quiet  | DD, DU          |
| bull_neutral_active | DU, UD          |
| bull_neutral_quiet  | DU, UD          |
| chop_active         | DU, UD, UU      |
| chop_quiet          | DU              |
| chop_volatile       | DU, UD, UU      |
| **TOTAL**           | **16 patterns** |

### Breakdown by Trend Regime

| Trend Regime | Combined Patterns | Vol Regimes Covered     |
| ------------ | ----------------- | ----------------------- |
| chop         | 7                 | active, quiet, volatile |
| bear_neutral | 5                 | active, quiet           |
| bull_neutral | 4                 | active, quiet           |

### Breakdown by Vol Regime

| Vol Regime | Combined Patterns | Trend Regimes Covered |
| ---------- | ----------------- | --------------------- |
| active     | 8                 | all 3 trends          |
| quiet      | 5                 | all 3 trends          |
| volatile   | 3                 | chop only             |

### Pattern Count Comparison

| Filter Type     | Universal Patterns |
| --------------- | ------------------ |
| SMA/RSI only    | 11                 |
| RV only         | 12                 |
| **Combined**    | **16**             |
| Theoretical max | 33 (11 x 3)        |

### Key Insights

1. **Chop + volatile** is a unique combination - only works in consolidation with high vol
2. **Active vol regime** is the most fertile - works across all trend regimes
3. **Bull/bear + volatile** has no universal patterns - too unstable
4. **DU pattern (reversal)** is universal across MOST combinations - strongest signal

### Trading Implications

1. **Primary signals**: Chop regime patterns work across all vol conditions
2. **Best vol regime**: Active (normal vol) - most consistent signals
3. **Avoid**: Trend + volatile combinations (no universal patterns)
4. **Signal refinement**: Use vol regime to filter trend signals

---

## Adversarial Audit (2026-01-31)

**ALL 12 RV patterns PASS all three audit perspectives.**

### Audit 1: Parameter Sensitivity

Tested 5 parameter variations:

| Parameter Set | RV Window | Percentiles | Universal Patterns |
| ------------- | --------- | ----------- | ------------------ |
| baseline      | 20        | 25-75       | 12                 |
| shorter_rv    | 10        | 25-75       | 12                 |
| longer_rv     | 30        | 25-75       | 12                 |
| tighter_pct   | 20        | 33-67       | 12                 |
| wider_pct     | 20        | 20-80       | 12                 |

**Result**: ALL 12 patterns robust across ALL 5 parameter sets. **PASS**

### Audit 2: Out-of-Sample Validation

| Period  | Dates                    | Universal Patterns |
| ------- | ------------------------ | ------------------ |
| Train   | 2022-01-01 to 2024-12-31 | 12                 |
| Test    | 2025-01-01 to 2026-01-31 | 12                 |
| Overlap | -                        | **12 (100%)**      |

**Result**: 100% OOS retention rate. **PASS**

### Audit 3: Bootstrap Confidence Intervals

| Pattern      | n    | Mean (bps) | 95% CI           | Excludes Zero |
| ------------ | ---- | ---------- | ---------------- | ------------- |
| quiet\|DD    | 111K | -6.42      | [-6.55, -6.27]   | YES           |
| quiet\|DU    | 81K  | +10.02     | [+9.83, +10.19]  | YES           |
| quiet\|UD    | 81K  | -10.11     | [-10.27, -9.93]  | YES           |
| quiet\|UU    | 113K | +6.82      | [+6.67, +6.98]   | YES           |
| active\|DD   | 177K | -6.98      | [-7.12, -6.86]   | YES           |
| active\|DU   | 135K | +10.46     | [+10.30, +10.60] | YES           |
| active\|UD   | 135K | -10.34     | [-10.47, -10.19] | YES           |
| active\|UU   | 176K | +7.25      | [+7.12, +7.38]   | YES           |
| volatile\|DD | 102K | -6.95      | [-7.12, -6.79]   | YES           |
| volatile\|DU | 85K  | +10.61     | [+10.43, +10.80] | YES           |
| volatile\|UD | 85K  | -10.69     | [-10.86, -10.49] | YES           |
| volatile\|UU | 102K | +7.03      | [+6.86, +7.22]   | YES           |

**Result**: 100% of patterns have CI excluding zero (500 bootstrap iterations). **PASS**

### Audit Conclusion

| Audit                 | Status        | Detail                             |
| --------------------- | ------------- | ---------------------------------- |
| Parameter Sensitivity | **PASS**      | 12/12 patterns across 5 param sets |
| OOS Validation        | **PASS**      | 100% retention rate                |
| Bootstrap CI          | **PASS**      | 100% CI excludes zero              |
| **OVERALL**           | **VALIDATED** | RV regime patterns are genuine     |

**The RV regime patterns are MORE robust than SMA/RSI patterns:**

- RV: 12/12 patterns PASS all audits
- SMA/RSI (Issue #52): 10/11 patterns PASS parameter sensitivity

---

## Cross-Regime Correlation Analysis (2026-01-31)

**Combining SMA/RSI and RV patterns provides MODERATE diversification benefit.**

### Portfolio Metrics

| Metric                | SMA/RSI Alone | RV Alone | Combined |
| --------------------- | ------------- | -------- | -------- |
| Total Patterns        | 11            | 12       | 23       |
| Effective DoF         | 4.5           | 4.5      | **4.93** |
| Diversification Ratio | 0.41          | 0.38     | **0.21** |
| Diversification Gain  | -             | -        | +0.43    |

### Cross-Correlation Analysis

| Metric                | Value |
| --------------------- | ----- |
| Average Cross-Corr    | 0.335 |
| Max Cross-Corr        | 0.925 |
| Min Cross-Corr        | 0.0   |
| Pairs with corr > 0.7 | 8     |
| Pairs with corr < 0.3 | 45    |

### High Correlation Pairs (AVOID combining)

| SMA/RSI Pattern  | RV Pattern   | Correlation |
| ---------------- | ------------ | ----------- |
| chop\|UU         | active\|DD   | 0.925       |
| chop\|DD         | active\|UU   | 0.918       |
| bear_neutral\|UU | volatile\|DD | 0.892       |

### Low Correlation Pairs (GOOD for diversification)

| SMA/RSI Pattern  | RV Pattern | Correlation |
| ---------------- | ---------- | ----------- |
| chop\|UD         | active\|DU | 0.0         |
| chop\|DU         | active\|UD | 0.0         |
| bull_neutral\|UD | quiet\|DU  | 0.05        |

### Interpretation

1. **Moderate diversification**: Combined DoF of 4.93 vs 4.5 for either alone (+0.43 gain)
2. **Not fully orthogonal**: Average cross-correlation of 0.335 indicates partial overlap
3. **Selective combination**: 45 low-correlation pairs available for portfolio construction
4. **Avoid redundancy**: 8 highly correlated pairs (>0.7) should not be combined

### Recommendation

**SELECTIVE COMBINATION** - Use both SMA/RSI and RV patterns but:

- Avoid pairing continuation patterns (UU/DD) across regimes (highly correlated)
- Prioritize pairing reversal patterns (DU/UD) across regimes (low correlation)
- Focus on chop x active and bull/bear x quiet combinations

---

## Return Profile Analysis (2026-01-31)

**6 of 12 RV patterns are profitable after 15 dbps transaction costs.**

### Return Distribution by Pattern

| Pattern      | N       | Gross (bps) | Net (bps)  | Win Rate | Kelly  | Skew    |
| ------------ | ------- | ----------- | ---------- | -------- | ------ | ------- |
| active\|DD   | 848,362 | -8.75       | -10.25     | 1.1%     | 0.0000 | -4.309  |
| active\|DU   | 724,598 | +11.78      | **+10.28** | 99.8%    | 0.9973 | +8.441  |
| active\|UD   | 721,880 | -11.76      | -13.26     | 0.1%     | 0.0000 | -8.382  |
| active\|UU   | 842,674 | +8.74       | **+7.24**  | 93.7%    | 0.9141 | +20.944 |
| quiet\|DD    | 497,309 | -8.49       | -9.99      | 1.0%     | 0.0000 | -4.270  |
| quiet\|DU    | 425,054 | +11.56      | **+10.06** | 99.8%    | 0.9978 | +8.644  |
| quiet\|UD    | 425,036 | -11.56      | -13.06     | 0.0%     | 0.0000 | -6.922  |
| quiet\|UU    | 505,951 | +8.44       | **+6.94**  | 93.5%    | 0.9153 | +9.120  |
| volatile\|DD | 471,081 | -8.60       | -10.10     | 2.0%     | 0.0000 | -2.990  |
| volatile\|DU | 424,149 | +12.62      | **+11.12** | 99.7%    | 0.9948 | +8.039  |
| volatile\|UD | 426,883 | -12.58      | -14.08     | 0.1%     | 0.0000 | -11.499 |
| volatile\|UU | 475,481 | +8.63       | **+7.13**  | 91.3%    | 0.8726 | -2.013  |

### Profitability Summary

| Category       | Patterns                           | Count |
| -------------- | ---------------------------------- | ----- |
| **Profitable** | DU (all regimes), UU (all regimes) | 6     |
| Unprofitable   | DD (all regimes), UD (all regimes) | 6     |

### Key Insights

1. **Reversal patterns (DU) are most profitable**: ~10-11 bps net across all regimes
2. **Continuation patterns (UU) are second best**: ~7 bps net across all regimes
3. **Down patterns (DD, UD) are consistently unprofitable**: Negative expectancy
4. **Volatile regime has strongest reversal signal**: DU +11.12 bps net (highest)
5. **Win rates are extreme**: DU >99%, DD/UD <2% - patterns are highly predictive

### Kelly Fractions

Best patterns for position sizing (highest Kelly fractions):

| Pattern      | Kelly Fraction | Interpretation        |
| ------------ | -------------- | --------------------- |
| quiet\|DU    | 0.9978         | Full Kelly allocation |
| active\|DU   | 0.9973         | Full Kelly allocation |
| volatile\|DU | 0.9948         | Full Kelly allocation |
| quiet\|UU    | 0.9153         | ~90% Kelly allocation |
| active\|UU   | 0.9141         | ~90% Kelly allocation |

### Skew Analysis

| Category      | Patterns | Interpretation                            |
| ------------- | -------- | ----------------------------------------- |
| Positive skew | 5        | DU (all), UU (quiet, active) - right tail |
| Negative skew | 7        | DD, UD (all), UU (volatile) - left tail   |

**Trading implication**: DU patterns have favorable skew (unlimited upside, limited downside).

### Script

`scripts/rv_return_profile_analysis_polars.py`

---

## Next Steps

- [x] Compute correlation between RV patterns and SMA/RSI patterns - DONE (moderate benefit)
- [x] Test combined SMA/RSI x RV regime filters - DONE (16 universal patterns)
- [x] Adversarial audit - DONE (ALL PASS)
- [x] Analyze return profiles for RV regime patterns - DONE (6 profitable)
- [ ] Transaction cost sensitivity analysis for RV patterns

---

## References

- Issue #52: Market Regime Filter for ODD Robust Patterns (baseline)
- Issue #54: Volatility Regime Filter for ODD Robust Patterns (this research)
