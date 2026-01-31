# Volatility Regime Patterns Research

**Issue**: #54 - Volatility Regime Filter for ODD Robust Patterns
**Status**: In Progress
**Last Updated**: 2026-01-31

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

---

## Next Steps

- [ ] Compute correlation between RV patterns and SMA/RSI patterns
- [ ] Test combined SMA/RSI × RV regime filters
- [ ] Analyze return profiles for RV regime patterns
- [ ] Transaction cost analysis for RV patterns

---

## References

- Issue #52: Market Regime Filter for ODD Robust Patterns (baseline)
- Issue #54: Volatility Regime Filter for ODD Robust Patterns (this research)
