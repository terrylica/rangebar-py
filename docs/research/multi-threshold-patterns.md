# Multi-Threshold Pattern Research

**Issue**: #55 - Multi-Threshold Pattern Confirmation Signals
**Status**: INVALIDATED (see pattern-research-summary.md)
**Last Updated**: 2026-02-01

---

## Executive Summary

**Multi-threshold alignment provides marginal improvement (~0.5-1 bps) but confirms pattern validity.**

11 aligned patterns are OOD robust across all 4 symbols. Full alignment (50/100/200 dbps agreeing) slightly outperforms partial alignment.

### Key Finding

| Pattern Type  | Aligned Return (bps) | Partial Return (bps) | Improvement |
| ------------- | -------------------- | -------------------- | ----------- |
| DU (reversal) | +12.22               | +11.52               | +0.70 bps   |
| UU (continue) | +8.70                | +8.55                | +0.15 bps   |

---

## Methodology

### Alignment Classification

For each 100 dbps bar, find the most recent pattern at 50 dbps and 200 dbps using asof joins.

| Alignment    | Definition          | Sample Size |
| ------------ | ------------------- | ----------- |
| aligned_up   | U at ALL thresholds | ~1M bars    |
| aligned_down | D at ALL thresholds | ~1M bars    |
| partial_up   | U at 2/3 thresholds | ~700K bars  |
| partial_down | D at 2/3 thresholds | ~700K bars  |
| mixed        | No clear alignment  | Rare        |

### OOD Robustness Criteria

Same as Issue #52:

- |t-stat| ≥ 5 in ALL quarterly periods
- Same sign across ALL periods
- ≥100 samples per period
- Cross-validated on ALL 4 symbols

---

## Results

### Universal Aligned Patterns (11)

| Base Pattern | Aligned Patterns                       |
| ------------ | -------------------------------------- |
| DD           | aligned_down, partial_down             |
| DU           | aligned_down, partial_down, partial_up |
| UD           | aligned_up, partial_down, partial_up   |
| UU           | aligned_up, partial_down, partial_up   |

### Return Comparison by Alignment

| Base | Alignment    | N         | Mean (bps) | Net (bps) |
| ---- | ------------ | --------- | ---------- | --------- |
| DD   | aligned_down | 1,045,002 | -8.71      | -10.21    |
| DD   | partial_down | 763,289   | -8.58      | -10.08    |
| DU   | aligned_down | 907,412   | **+12.22** | +10.72    |
| DU   | partial_down | 654,034   | +11.52     | +10.02    |
| DU   | partial_up   | 12,348    | +14.42     | +12.92    |
| UD   | aligned_up   | 924,368   | -12.18     | -13.68    |
| UD   | partial_up   | 637,205   | -11.52     | -13.02    |
| UU   | aligned_up   | 1,076,435 | **+8.70**  | +7.20     |
| UU   | partial_up   | 738,745   | +8.55      | +7.05     |

### Data Coverage

| Symbol  | 50 dbps   | 100 dbps  | 200 dbps | Aligned |
| ------- | --------- | --------- | -------- | ------- |
| BTCUSDT | 4,179,709 | 1,382,518 | 303,440  | 1.38M   |
| ETHUSDT | 5,967,593 | 1,996,522 | 498,640  | 2.00M   |
| SOLUSDT | 6,299,337 | 1,977,403 | 540,056  | 1.98M   |
| BNBUSDT | 4,553,434 | 1,432,019 | 388,670  | 1.43M   |

---

## Key Insights

### 1. Alignment Provides Small but Consistent Improvement

- DU_aligned_down: +12.22 bps vs DU_partial_down: +11.52 bps (+0.70 bps)
- UU_aligned_up: +8.70 bps vs UU_partial_up: +8.55 bps (+0.15 bps)

### 2. Counter-Trend Alignment is Rare but Strong

- DU_partial_up: +14.42 bps (strongest signal)
- Only 12K samples - too few for reliable trading
- May indicate strong reversal zones

### 3. Most Bars Show Alignment

- ~70% of 100 dbps bars have aligned or partial alignment
- Mixed alignment is rare
- Full alignment dominates when it occurs

### 4. Alignment Confirms Rather Than Predicts

- Aligned patterns don't outperform dramatically
- Better use: confirmation filter rather than primary signal
- Combine with regime filters (RV, SMA/RSI) for best results

---

## Trading Implications

### Recommended Usage

1. **Primary signal**: Use 100 dbps patterns (DU, UU)
2. **Confirmation**: Check 50/200 dbps alignment
3. **Extra confirmation**: Combine with RV regime filter
4. **Position sizing**: Aligned patterns → larger position

### Entry Rules

| Alignment Level | Position Size | Confidence |
| --------------- | ------------- | ---------- |
| Full (3/3)      | 100%          | High       |
| Partial (2/3)   | 75%           | Medium     |
| Mixed           | Skip          | Low        |

---

## Scripts

| Script                                               | Purpose                            |
| ---------------------------------------------------- | ---------------------------------- |
| `scripts/multi_threshold_pattern_analysis_polars.py` | Multi-threshold alignment analysis |

---

## Status: INVALIDATED

This document reflects initial findings BEFORE adversarial forensic audit.

**Invalidation Cause**: Cross-threshold alignment was invalidated by forensic audit that revealed:

1. argMax bug in ClickHouse queries selecting arbitrary bar closes instead of actual last bar
2. With correct close price selection, alignment provides ZERO predictive power
3. All returns within noise (< 2 bps mean, ~60 bps std)

See `docs/research/pattern-research-summary.md` for full invalidation details.

---

## References

- Issue #52: Market Regime Filter for ODD Robust Patterns
- Issue #54: Volatility Regime Filter for ODD Robust Patterns
- Issue #55: Multi-Threshold Pattern Confirmation Signals (this research)
