# Multi-Factor Multi-Granularity Pattern Analysis

**Issue**: #52 - Market Regime Filter for ODD Robust Patterns
**Status**: Complete
**Last Updated**: 2026-02-01

---

## Executive Summary

**Multi-factor patterns (200 dbps trend + 100 dbps pattern) do NOT improve ODD robustness over single-factor patterns. Both fail due to sign reversals across quarterly periods.**

This validates the MRH Framework finding that pattern returns are regime-dependent and cannot achieve unconditional ODD robustness regardless of factor combinations.

### Key Finding

| Factor Type   | Patterns Tested | ODD Robust | Universal (4 symbols) |
| ------------- | --------------- | ---------- | --------------------- |
| Single-factor | 16              | 0          | 0                     |
| Multi-factor  | 32              | 0          | 0                     |

---

## Methodology

### Multi-Granularity Approach

1. **50 dbps**: Fine-grained bars (~6M per symbol) - not used in final signal
2. **100 dbps**: Primary pattern detection (~2M per symbol)
3. **200 dbps**: Higher timeframe trend filter (~400K per symbol)

### Multi-Factor Signal Construction

```
multifactor_signal = HTF_trend(200 dbps) + "|" + pattern(100 dbps)
```

Where:

- HTF trend = "U" if majority of last 3 bars at 200 dbps were Up, "D" otherwise
- Pattern = 2-bar pattern at 100 dbps (DD, DU, UD, UU)

### ODD Robustness Criteria

- Same sign across all quarterly periods (16-17 quarters)
- |t-stat| ≥ 5.0 in each period
- Minimum 100 samples per period

---

## Results

### Single-Factor Patterns (100 dbps)

| Symbol  | Pattern | Mean t-stat | Min  | t   |     | Same Sign | ODD Robust |
| ------- | ------- | ----------- | ---- | --- | --- | --------- | ---------- |
| BTCUSDT | DU      | -15.45      | 0.05 | no  | no  |
| BTCUSDT | UD      | +14.86      | 0.54 | no  | no  |
| ETHUSDT | DU      | -13.68      | 1.05 | no  | no  |
| ETHUSDT | UD      | +13.61      | 0.25 | no  | no  |
| SOLUSDT | DU      | -9.12       | 0.56 | no  | no  |
| SOLUSDT | UD      | +9.11       | 2.09 | no  | no  |
| BNBUSDT | DU      | -17.11      | 2.79 | no  | no  |
| BNBUSDT | UD      | +16.75      | 4.00 | no  | no  |

**Key observation**: Strong aggregate t-stats but sign reversals in specific periods invalidate ODD robustness.

### Multi-Factor Patterns (200 dbps trend + 100 dbps pattern)

| Symbol  | Signal | Mean t-stat | Min  | t   |     | Same Sign | ODD Robust |
| ------- | ------ | ----------- | ---- | --- | --- | --------- | ---------- |
| BTCUSDT | D\|UD  | +13.08      | 0.45 | no  | no  |
| BTCUSDT | U\|DU  | -13.78      | 0.17 | no  | no  |
| ETHUSDT | D\|UD  | +13.57      | 1.98 | no  | no  |
| ETHUSDT | U\|DU  | -13.17      | 0.69 | no  | no  |
| BNBUSDT | D\|UD  | +12.75      | 2.15 | no  | no  |
| BNBUSDT | U\|DU  | -13.84      | 2.42 | no  | no  |

**Key observation**: Adding HTF trend filter does not stabilize sign across periods.

---

## Key Insights

### 1. Multi-Factor Does NOT Improve ODD Robustness

Both single and multi-factor patterns fail the same way:

- Strong aggregate t-statistics (>10 in many cases)
- Sign reversals in specific quarterly periods
- Minimum |t| often < 1 in at least one period

### 2. Pattern Sign Instability is Regime-Driven

The sign reversals occur in specific periods (likely corresponding to:

- Major market events (Luna collapse, FTX, etc.)
- TDA-detected structural breaks
- Volatility regime shifts

### 3. HTF Trend Filter Doesn't Stabilize

Using 200 dbps trend as a filter splits the data into subsets but doesn't eliminate sign instability. Both "up trend + pattern" and "down trend + pattern" show reversals.

### 4. Mechanical Returns Remain Dominant

The patterns still show mechanical return characteristics:

- DU: consistently negative aggregate t-stat (price reversal artifact)
- UD: consistently positive aggregate t-stat (price reversal artifact)

---

## Connection to Previous Findings

| Prior Finding                                | This Analysis                                |
| -------------------------------------------- | -------------------------------------------- |
| Patterns are regime-dependent (TDA analysis) | ✓ Sign reversals confirm regime dependency   |
| 1-bar returns are mechanical (Audit)         | ✓ DU/UD still dominate signal                |
| Hurst stable across TDA regimes              | ✓ Memory structure doesn't predict stability |
| 75% pattern-regime comparisons significant   | ✓ Regime filtering needed, not HTF filtering |

---

## Trading Implications

### What Doesn't Work

- Using HTF trend filter (200 dbps) to stabilize LTF patterns (100 dbps)
- Assuming multi-factor combinations improve ODD robustness
- Deploying patterns without regime conditioning

### What Might Work

Based on prior research:

1. **TDA-based regime filtering** (pause trading during TDA breaks)
2. **SMA/RSI regime conditioning** (patterns within specific regimes)
3. **Multi-bar forward returns** (3-10 bars show genuine persistence)

---

## Scripts

| Script                                             | Purpose                             |
| -------------------------------------------------- | ----------------------------------- |
| `scripts/multifactor_multigranularity_patterns.py` | Multi-factor ODD robustness testing |

---

## Future Research

- [ ] Test regime-conditioned multi-factor patterns (TDA regime + HTF + pattern)
- [ ] Explore alternative HTF filters (volatility, momentum indicators)
- [ ] Test longer lookback periods for HTF trend (5, 10 bars)

---

## References

- Issue #52: Market Regime Filter for ODD Robust Patterns
- docs/research/tda-regime-patterns.md
- docs/research/market-regime-patterns.md
