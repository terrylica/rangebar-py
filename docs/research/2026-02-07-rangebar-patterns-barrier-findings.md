# Range Bar Pattern Discovery: Barrier Framework Findings

**Source**: [terrylica/rangebar-patterns](https://github.com/terrylica/rangebar-patterns) repository
**Date**: 2026-02-07
**Status**: ACTIVE — contradicts "ZERO patterns" conclusion when microstructure filters + barriers are applied

---

## Executive Summary

The `rangebar-patterns` repository performed a 17-generation SQL brute-force sweep across range bar microstructure features on ClickHouse, discovering a statistically significant pattern (z=8.25, DSR=1.000) that becomes profitable (PF=1.268) when combined with triple barrier risk management. However, **Kelly fractions are negative across all 100 barrier configurations**, meaning the edge is too thin for standalone trading.

**Critical insight**: The pattern uses only 3 of the 12 populated features (trade_intensity, kyle_lambda_proxy, direction). The remaining 9 populated features — and especially the 16 unpopulated lookback/intra columns — have never been tested as additional entry filters. These additional filters could concentrate signals into higher-conviction trades with thicker edges (positive Kelly).

---

## The Champion Pattern

```
IF   2 consecutive DOWN bars (close < open on bar t-1 AND t-2)
AND  trade_intensity(t-1) > p95_expanding(all prior bars)
AND  kyle_lambda_proxy(t-1) > 0
THEN LONG at next bar's open
```

### Performance

| Metric                                   | Value  |
| ---------------------------------------- | ------ |
| Directional hit rate (TRUE no-lookahead) | 62.93% |
| Z-score                                  | 8.25   |
| DSR (Deflated Sharpe Ratio, N=111)       | 1.000  |
| Signals (SOL @500dbps)                   | 1,894  |

### Triple Barrier Results (Gen200, SOL @500dbps)

All 100 parameter combos (5 TP × 5 SL × 4 time):

| R:R Ratio        | Best Combo                 | PF        | Win Rate | Kelly      |
| ---------------- | -------------------------- | --------- | -------- | ---------- |
| TP < SL (0.33:1) | tp=0.5x, sl=1.5x, 50 bars  | **1.268** | 43.8%    | -0.173     |
| TP = SL (1:1)    | tp=0.5x, sl=0.5x, 50 bars  | 1.139     | 41.5%    | -0.147     |
| TP > SL (2:1)    | tp=0.5x, sl=0.25x, 50 bars | 1.036     | 30.6%    | **-0.068** |
| TP > SL (4:1)    | tp=1.0x, sl=0.25x, 50 bars | 1.018     | 11.6%    | -0.236     |
| TP >> SL (12:1)  | tp=3.0x, sl=0.25x, 50 bars | 0.964     | 0.0%     | -0.410     |

**Key finding**: ALL 100 combos have negative Kelly. The best Kelly (-0.068) is at 2:1 R:R. The pattern produces small, short-lived bounces — wide TP targets are never reached.

### Temporal Decay

| Year | Hit Rate | Z-Score | Significant? |
| ---- | -------- | ------- | ------------ |
| 2020 | 71.4%    | 3.05    | Yes          |
| 2021 | 68.0%    | 3.13    | Yes          |
| 2022 | 66.7%    | 2.89    | Yes          |
| 2023 | 62.5%    | 2.24    | Yes          |
| 2024 | 56.0%    | 1.24    | **No**       |
| 2025 | 54.0%    | 0.87    | **No**       |

### Cross-Asset

| Asset | Hit Rate | Direction                    |
| ----- | -------- | ---------------------------- |
| SOL   | 62.93%   | Long                         |
| BNB   | 71.72%   | Long                         |
| BTC   | 62.67%   | Long                         |
| ETH   | 45.00%   | **Inverted** (pattern fails) |

---

## Why This Matters for rangebar-py

### The "ZERO patterns" conclusion needs revision

The `rangebar-py` research (Issue #57) concluded ZERO ODD robust patterns. But that research:

1. Did NOT use triple barrier framework (only measured raw returns)
2. Did NOT test composite microstructure filters (only individual features)
3. Did NOT use expanding-window p95 normalization (used fixed percentiles)

The `rangebar-patterns` research shows that **composite features + barriers** transform a 62.93% directional edge into PF > 1.0. The edge is real but thin. The path to thicker edges requires **more features as entry filters**.

### Features that could push Kelly positive

The hypothesis: adding more microstructure conditions on top of the champion pattern filters out weak signals, leaving only high-conviction setups where the bounce is large enough for TP > SL with positive Kelly.

**Currently available but untested as filters**:

| Feature             | Column                      | Hypothesis                                            |
| ------------------- | --------------------------- | ----------------------------------------------------- |
| OFI                 | `ofi`                       | Extreme negative OFI = exhaustion → stronger reversal |
| Aggression ratio    | `aggression_ratio`          | Low aggression during down bars = selling exhaustion  |
| Turnover imbalance  | `turnover_imbalance`        | Extreme imbalance = capitulation                      |
| Price impact        | `price_impact`              | High impact = thin book → violent reversal            |
| VWAP deviation      | `vwap_close_deviation`      | Close far below VWAP = oversold intrabar              |
| Volume per trade    | `volume_per_trade`          | Large trades = institutional, small = retail herding  |
| Aggregation density | `aggregation_density`       | High density = fragmented retail flow                 |
| Buy/sell volume     | `buy_volume`, `sell_volume` | Volume asymmetry as conviction measure                |

**Schema exists but EMPTY — critical for next phase**:

| Feature             | Lookback Column                | Intra Column                | Research Basis                 |
| ------------------- | ------------------------------ | --------------------------- | ------------------------------ |
| Burstiness          | `lookback_burstiness`          | `intra_burstiness`          | B≈1 = cascade/stop hunt → fade |
| Hurst exponent      | `lookback_hurst`               | `intra_hurst`               | H<0.5 = mean reverting regime  |
| Permutation entropy | `lookback_permutation_entropy` | `intra_permutation_entropy` | Low PE = predictable regime    |
| Kaufman ER          | `lookback_kaufman_er`          | `intra_kaufman_er`          | Efficiency ratio for regime    |
| Garman-Klass vol    | `lookback_garman_klass_vol`    | `intra_garman_klass_vol`    | Volatility regime filter       |
| Volume skew         | `lookback_volume_skew`         | `intra_volume_skew`         | Distribution asymmetry         |
| Volume kurtosis     | `lookback_volume_kurt`         | `intra_volume_kurt`         | Tail heaviness (whales)        |
| Kyle lambda         | `lookback_kyle_lambda`         | `intra_kyle_lambda`         | Market depth from lookback     |
| OFI                 | `lookback_ofi`                 | `intra_ofi`                 | Directional pressure buildup   |

These features are **already implemented in Rust** (crates/rangebar-core/src/interbar.rs and intrabar/) but the ClickHouse data was never populated with them.

---

## Gemini Deep Research References

Three deep research documents provide academic backing for these features:

1. **Microstructure features** (`rangebar-patterns/findings/2026-02-05-microstructure-deep-research.md`)
   - Gemini 3 Pro: VPIN, inverted Kyle lambda, Shannon entropy, burstiness, HMM regime detection

2. **Intrabar microstructure** (`rangebar-py/docs/research/2026-02-02-intrabar-microstructure-gemini-3-pro.md`)
   - Gemini 3 Pro: Volume-weighted moments, POC dynamics, Hawkes processes, wavelet decomposition

3. **Feature selection for OOD robustness** (`trading-fitness/docs/research/external/2026-02-02-feature-selection-ood-robustness-gemini.md`)
   - Gemini 3 Pro: TSKI, PCMCI+, HSIC Lasso for handling autocorrelated features

---

## Request to rangebar-py

**Priority 1**: Backfill the 16 lookback + 22 intra-bar columns for SOL @250dbps and @500dbps. These columns have Rust implementations and ClickHouse schema but contain only zeros/nulls.

**Priority 2**: After backfill, the `rangebar-patterns` repo will run Gen300+ barrier sweeps using these features as additional entry filters to find positive-Kelly configurations.

**Reference**: GitHub Issue to be created in terrylica/rangebar-py with full technical details.
