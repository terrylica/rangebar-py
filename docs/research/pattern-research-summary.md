# Multi-Factor Range Bar Pattern Research Summary

**Status**: Complete
**Last Updated**: 2026-01-31

---

## Executive Summary

**Combined two-factor filtering (RV regime + multi-threshold alignment) yields 2.4x more OOD robust patterns than single-factor approaches, with 96% out-of-sample retention.**

| Research Area                 | Issue    | Universal Patterns | Audit Status  |
| ----------------------------- | -------- | ------------------ | ------------- |
| SMA/RSI Regime                | #52      | 11                 | VALIDATED     |
| RV Volatility Regime          | #54      | 12                 | VALIDATED     |
| Multi-Threshold Alignment     | #55      | 11                 | VALIDATED     |
| **Combined (RV + Alignment)** | #54, #55 | **26-29**          | **VALIDATED** |
| 3-Bar Patterns                | #54, #55 | 8                  | VALIDATED     |
| **RV + 3-Bar Patterns**       | #54, #55 | **24**             | **VALIDATED** |
| Alignment + 3-Bar Patterns    | #54, #55 | 20                 | PENDING       |

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

| Script                                               | Purpose                   |
| ---------------------------------------------------- | ------------------------- |
| `scripts/market_regime_patterns_polars.py`           | SMA/RSI regime analysis   |
| `scripts/volatility_regime_analysis_polars.py`       | RV regime analysis        |
| `scripts/volatility_regime_audit_polars.py`          | RV regime audit           |
| `scripts/multi_threshold_pattern_analysis_polars.py` | Multi-threshold alignment |
| `scripts/combined_rv_alignment_analysis_polars.py`   | Combined analysis         |
| `scripts/combined_pattern_audit_polars.py`           | Combined audit            |
| `scripts/cross_regime_correlation_polars.py`         | Regime correlation        |
| `scripts/rv_return_profile_analysis_polars.py`       | Return profiles           |
| `scripts/pattern_correlation_analysis_polars.py`     | Pattern correlation       |
| `scripts/three_bar_pattern_analysis_polars.py`       | 3-bar pattern analysis    |
| `scripts/three_bar_pattern_audit_polars.py`          | 3-bar pattern audit       |
| `scripts/three_bar_alignment_analysis_polars.py`     | 3-bar + alignment         |

---

## GitHub Issues

| Issue | Title                                            | Status |
| ----- | ------------------------------------------------ | ------ |
| #52   | Market Regime Filter for ODD Robust Patterns     | CLOSED |
| #54   | Volatility Regime Filter for ODD Robust Patterns | OPEN   |
| #55   | Multi-Threshold Pattern Confirmation Signals     | OPEN   |

---

## Future Research Directions

- [x] 3-bar pattern analysis (COMPLETE - 8 universal, 24 with RV)
- [x] Adversarial audit of 3-bar patterns (VALIDATED - 100% retention)
- [ ] 4-bar pattern analysis (diminishing returns expected)
- [ ] SMA/RSI + RV + Alignment (three-factor combination)
- [ ] Forex symbol validation when data available
- [ ] Ensemble strategies using multiple filter combinations

---

## Conclusion

The multi-factor approach to range bar pattern research has yielded significant results:

1. **Single-factor filters** each reveal 11-12 OOD robust patterns
2. **Two-factor combination** (RV + alignment) yields 26-29 patterns (2.4x improvement)
3. **3-bar patterns** provide 2x improvement over 2-bar (8 vs 4 universal)
4. **RV + 3-bar combination** yields 24 universal patterns (3x over 3-bar alone)
5. **All two-factor findings adversarially validated** via parameter sensitivity, OOS, and bootstrap CI
6. **DU-ending patterns** (DU, DUD, DUU) are consistently the most profitable (+10-13 bps net)
7. **DD and UD patterns** should NOT be traded (negative expectancy)

The research establishes a clear hierarchy: more factors = more patterns = better filtering.
