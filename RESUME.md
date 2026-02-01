# RESUME.md - Session Context

**Last Updated**: 2026-02-01
**Purpose**: Quick context for continuing work in future sessions

---

## Current State Summary

### Pattern Research: COMPLETE

**Finding**: ZERO ODD robust predictive patterns exist in range bar data.

After exhaustive testing across **10 research approaches**, all were invalidated:

| Approach                  | Status      | Root Cause                            |
| ------------------------- | ----------- | ------------------------------------- |
| Direction patterns (U/D)  | INVALIDATED | Boundary-locked returns (H~0.79)      |
| 2-bar/3-bar patterns      | INVALIDATED | Forward returns show mean reversion   |
| TDA regime conditioning   | INVALIDATED | Lookback leakage in regime labels     |
| Microstructure features   | 0 ODD       | Feature noise exceeds signal          |
| Cross-threshold alignment | INVALIDATED | Temporal overlap contaminates signals |
| Return persistence        | INVALIDATED | Same sign but t < 3.0                 |
| Coarse-to-fine cascade    | BLOCKED     | Combinatorial explosion               |
| Duration autocorrelation  | INVALIDATED | 100% mechanical (deferred-open)       |
| TDA velocity forecast     | INVALIDATED | t-stats -1.67 to +1.01                |
| Cross-asset correlation   | INVALIDATED | 0 ODD (Issue #145, crypto-forex)      |

**Key Insight**: Range bars are unsuitable for directional prediction due to:

1. Boundary-locked returns (mechanically bounded by threshold)
2. Temporal overlap (75-100% consecutive bar overlap)
3. Long memory (Hurst H~0.79 reduces effective sample size)

---

## Available Resources

### Data

| Location            | Content                       | Size       |
| ------------------- | ----------------------------- | ---------- |
| bigblack ClickHouse | Range bars (ouroboros='year') | 260M+ bars |
| Local tick cache    | Binance crypto ticks          | Multi-year |

**Crypto**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
**Forex**: EURUSD (Exness Raw_Spread, Issue #143-#145)
**Thresholds**: 25, 50, 100, 200, 250 dbps (crypto); 50, 100, 200 dbps (forex)
**Period**: 2022-2026 (continuous, all gaps filled)

### Documentation

| Document                                         | Lines | Purpose                  |
| ------------------------------------------------ | ----- | ------------------------ |
| `docs/research/INDEX.md`                         | 124   | Navigation index         |
| `docs/research/pattern-research-summary.md`      | 944   | Master findings          |
| `docs/research/adversarial-audit-methodology.md` | 303   | Reusable audit framework |

### Scripts

42+ research scripts in `scripts/`:

- Pattern analysis: `*_patterns*.py` (INVALIDATED)
- TDA research: `tda_*.py` (INVALIDATED)
- Audit scripts: `*_audit*.py` (methodology reference)
- Cache population: `fill_gaps_*.py`

---

## GitHub Issues

| Issue | Title                                       | Status |
| ----- | ------------------------------------------- | ------ |
| #57   | Research Complete: ZERO ODD Robust Patterns | CLOSED |
| #56   | TDA Structural Break Detection              | CLOSED |
| #52   | Market Regime Filter                        | CLOSED |
| #54   | Volatility Regime Filter                    | CLOSED |
| #55   | Multi-Threshold Pattern Confirmation        | CLOSED |

---

## What NOT to Retry

These approaches have been definitively invalidated:

1. **Any direction-based pattern** - Returns are boundary-locked
2. **Duration-based volatility prediction** - Mechanical artifact
3. **TDA velocity for forward RV** - No predictive power
4. **Microstructure features for alpha** - Noise exceeds signal
5. **Cross-threshold alignment signals** - Temporal overlap contamination

---

## Unexplored Directions

These remain unexplored but are **operational** (not predictive):

| Direction         | Rationale                                      | Status     |
| ----------------- | ---------------------------------------------- | ---------- |
| Execution quality | Bar timing for optimal order placement         | UNEXPLORED |
| Market making     | Microstructure for spread/inventory management | UNEXPLORED |

**Note**: These are not "ODD robust MULTI-FACTOR patterns" as per the encouraged guidance. They are operational use cases that don't require directional prediction.

---

## Technical Stack

- **Python**: 3.13 (REQUIRED, never downgrade)
- **Package manager**: UV
- **Data processing**: Polars + Arrow (not Pandas)
- **Database**: ClickHouse (bigblack via SSH)
- **TDA**: ripser, persim
- **Regime detection**: river (ADWIN)

---

## Session Handoff Notes

If continuing pattern research:

1. **DO NOT** retry invalidated approaches
2. The methodology is sound - the finding is genuinely negative
3. 10 approaches exhausted = research is complete
4. Focus on operational use cases if any research continues

If working on other features:

1. Range bar generation is production-ready
2. Microstructure features work correctly
3. ClickHouse caching is fully operational

---

## References

- Root hub: [CLAUDE.md](/CLAUDE.md)
- Research index: [docs/research/INDEX.md](/docs/research/INDEX.md)
- Audit methodology: [docs/research/adversarial-audit-methodology.md](/docs/research/adversarial-audit-methodology.md)
