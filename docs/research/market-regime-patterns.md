# Market Regime Patterns Research

**Issue**: #52 - Market Regime Filter for ODD Robust Multi-Factor Range Bar Patterns
**Status**: In Progress (data fill running)
**Last Updated**: 2026-01-31

---

## Hypothesis

Deterministic market regime filters (SMA crossovers, RSI levels) may reveal ODD robust subsets of multi-factor range bar patterns. Patterns that show no ODD robustness across all market conditions may exhibit strong robustness WITHIN specific regimes.

---

## Market Regime Definitions

### SMA-based Regimes

| Regime            | Condition                             | Description        |
| ----------------- | ------------------------------------- | ------------------ |
| **Uptrend**       | Price > SMA(20) > SMA(50)             | Bullish momentum   |
| **Downtrend**     | Price < SMA(20) < SMA(50)             | Bearish momentum   |
| **Consolidation** | SMA(20) crossed SMA(50) within N bars | No clear direction |

### RSI-based Regimes

| Regime         | Condition           | Description               |
| -------------- | ------------------- | ------------------------- |
| **Overbought** | RSI(14) > 70        | Extreme bullish sentiment |
| **Oversold**   | RSI(14) < 30        | Extreme bearish sentiment |
| **Neutral**    | 30 <= RSI(14) <= 70 | Normal conditions         |

### Combined Regime Matrix

| Regime       | SMA           | RSI        | Expected Pattern Behavior   |
| ------------ | ------------- | ---------- | --------------------------- |
| Bull-Hot     | Uptrend       | Overbought | Mean-reversion more likely  |
| Bull-Neutral | Uptrend       | Neutral    | Trend continuation expected |
| Bear-Cold    | Downtrend     | Oversold   | Bounce/reversal possible    |
| Bear-Neutral | Downtrend     | Neutral    | Trend continuation expected |
| Chop         | Consolidation | Any        | High noise, avoid trading   |

---

## Data Requirements

### Symbols

| Symbol  | Start Date | End Date   | Status      |
| ------- | ---------- | ---------- | ----------- |
| BTCUSDT | 2022-01-01 | 2026-01-31 | In Progress |
| ETHUSDT | 2022-01-01 | 2026-01-31 | Pending     |
| SOLUSDT | 2023-06-01 | 2026-01-31 | Pending     |
| BNBUSDT | 2022-01-01 | 2026-01-31 | Pending     |

### Thresholds (dbps)

- **50 dbps**: Fine granularity (primary signals)
- **100 dbps**: Standard granularity (primary signals)
- **200 dbps**: Coarse granularity (trend filter/confirmation)

---

## Methodology

### Phase 1: Data Preparation

1. Load range bars from ClickHouse cache
2. Compute SMA(20), SMA(50) on Close prices
3. Compute RSI(14) on Close prices
4. Label each bar with market regime

### Phase 2: Pattern Detection

1. Identify 2-bar and 3-bar patterns at 100 dbps
2. Filter patterns by market regime
3. Compute forward returns (1-bar, 3-bar, 5-bar)

### Phase 3: ODD Robustness Testing

1. Split data into rolling quarterly periods
2. For each regime subset, compute:
   - Pattern frequency
   - Win rate
   - Mean return
   - t-statistic
3. Pattern is ODD robust if |t-stat| >= 5 with same sign across ALL periods WITHIN the regime

### Phase 4: Multi-Factor Confirmation

1. Add 200 dbps trend filter (last N bars direction)
2. Test if confirmation improves robustness
3. Document which combinations pass ODD criteria

---

## Success Criteria

A pattern-regime combination is considered **ODD robust** if:

1. **Significance**: |t-stat| >= 5 in all rolling periods
2. **Consistency**: Same sign (positive or negative) across all periods
3. **Sample size**: >= 100 bars per period within the regime
4. **No look-ahead**: Regime determined from data BEFORE pattern

---

## Results

### Findings Summary

_To be populated after analysis completes_

### ODD Robust Patterns Found

_To be populated after analysis completes_

### Patterns That Failed ODD Criteria

_To be populated after analysis completes_

---

## Scripts

| Script                        | Purpose              |
| ----------------------------- | -------------------- |
| `scripts/regime_analysis.py`  | Main analysis script |
| `scripts/fill_all_symbols.py` | Data population      |

---

## References

- [Price Action Patterns](./price-action-patterns.md) - Prior research (no ODD robust patterns found unfiltered)
- GitHub Issue #52 - Research tracking
- GitHub Issue #53 - Memory guard design issue discovered during data fill
