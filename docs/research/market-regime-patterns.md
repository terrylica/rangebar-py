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

### Preliminary Findings (BTCUSDT @ 100 dbps)

**Date**: 2026-01-31
**Data**: 1,252,498 bars (2022-01-01 to 2026-01-31)

#### Regime Distribution

| Regime       | Bars    | Percentage |
| ------------ | ------- | ---------- |
| Chop         | 501,219 | 40.0%      |
| Bear Neutral | 346,544 | 27.7%      |
| Bull Neutral | 340,681 | 27.2%      |
| Bull Hot     | 33,713  | 2.7%       |
| Bear Cold    | 30,341  | 2.4%       |

#### ODD Robust 2-Bar Patterns (16 found)

| Regime       | Patterns       |
| ------------ | -------------- |
| Chop         | DD, DU, UU, UD |
| Bear Neutral | DD, UD, DU, UU |
| Bear Cold    | DU, DD         |
| Bull Neutral | UU, UD, DD, DU |
| Bull Hot     | UD, UU         |

#### ODD Robust 3-Bar Patterns (32 found)

| Regime       | Patterns                               |
| ------------ | -------------------------------------- |
| Chop         | DDU, DUU, UUU, UUD, UDU, UDD, DDD, DUD |
| Bear Neutral | DDU, UDU, DUU, UUU, UUD, DUD, DDD, UDD |
| Bear Cold    | DUD, DDD, DDU, DUU                     |
| Bull Neutral | UUD, UDD, DDU, DUU, UDU, DUD, DDD, UUU |
| Bull Hot     | UDD, UUD, UDU, UUU                     |

### Cross-Symbol Validation (ETHUSDT @ 100 dbps)

**Date**: 2026-01-31
**Data**: 1,582,770 bars (2022-01-01 to 2026-01-31)

#### Cross-Symbol Pattern Consistency

| Pattern Type | BTC Patterns | ETH Patterns | Common | Consistency |
| ------------ | ------------ | ------------ | ------ | ----------- |
| 2-bar        | 16           | 15           | 15     | 93.8%       |
| 3-bar        | 32           | 30           | 30     | 93.8%       |

#### Universal Cross-Symbol ODD Robust Patterns (2-bar)

| Regime       | Patterns       | Note                       |
| ------------ | -------------- | -------------------------- |
| Chop         | DD, DU, UD, UU | ALL 4 patterns robust      |
| Bull Neutral | DD, DU, UD, UU | ALL 4 patterns robust      |
| Bear Neutral | DD, DU, UD, UU | ALL 4 patterns robust      |
| Bull Hot     | UD             | Only 1 pattern (reversal?) |
| Bear Cold    | DD, DU         | 2 patterns (continuation?) |

### Complete Cross-Symbol Validation (All 4 Symbols)

**Date**: 2026-01-31

| Symbol  | Bars      | 2-bar Robust | 3-bar Robust |
| ------- | --------- | ------------ | ------------ |
| BTCUSDT | 1,252,498 | 16           | 32           |
| ETHUSDT | 1,582,770 | 15           | 30           |
| SOLUSDT | 3,712,525 | 19           | 31           |
| BNBUSDT | 1,372,614 | 16           | 32           |

#### Universal ODD Robust Patterns (ALL 4 Symbols)

**2-bar patterns**: 15 universal (75% of theoretical max)

| Regime       | Patterns       | Count |
| ------------ | -------------- | ----- |
| Chop         | DD, DU, UD, UU | 4     |
| Bull Neutral | DD, DU, UD, UU | 4     |
| Bear Neutral | DD, DU, UD, UU | 4     |
| Bull Hot     | UD             | 1     |
| Bear Cold    | DD, DU         | 2     |

**3-bar patterns**: 30 universal (75% of theoretical max)

| Regime       | Patterns                               | Count |
| ------------ | -------------------------------------- | ----- |
| Chop         | DDD, DDU, DUD, DUU, UDD, UDU, UUD, UUU | 8     |
| Bull Neutral | DDD, DDU, DUD, DUU, UDD, UDU, UUD, UUU | 8     |
| Bear Neutral | DDD, DDU, DUD, DUU, UDD, UDU, UUD, UUU | 8     |
| Bull Hot     | UDD, UUU                               | 2     |
| Bear Cold    | DDD, DDU, DUD, DUU                     | 4     |

### Key Observations

1. **Chop regime dominates** (40% of bars) - market spends most time in consolidation
2. **Extreme regimes are rare** - Bull Hot + Bear Cold < 5% combined
3. **All regimes have ODD robust patterns** - suggests regime filtering reveals predictability
4. **75% universal cross-symbol consistency** - 15/20 2-bar and 30/40 3-bar patterns are universal
5. **Full pattern coverage in neutral regimes** - ALL 8 possible patterns (4x 2-bar + 8x 3-bar) are universal
6. **Extreme regimes have fewer universal patterns** - Bull Hot (1-2) and Bear Cold (2-4)

### Research Conclusion

**The hypothesis is CONFIRMED**: Deterministic market regime filters (SMA crossovers + RSI levels) reveal ODD robust pattern subsets that were not found in unfiltered analysis.

**Key finding**: In neutral regimes (Chop, Bull Neutral, Bear Neutral), ALL possible 2-bar and 3-bar directional patterns are ODD robust across all 4 tested symbols. This suggests the regime filter itself is the primary source of predictability, not the specific pattern.

### Next Steps

- [x] Validate on all symbols (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT) - DONE
- [x] Check for cross-symbol pattern consistency - DONE (75% universal)
- [ ] Add 200 dbps trend filter confirmation
- [ ] Compute actual return statistics per pattern/regime
- [ ] Test statistical significance of regime transitions

### Patterns That Failed ODD Criteria

_Analysis pending - need to examine patterns with |t-stat| < 5 or inconsistent signs_

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
