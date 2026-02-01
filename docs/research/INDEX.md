# Research Documentation Index

**Status**: Research Complete (2026-02-01)
**Finding**: ZERO ODD robust predictive patterns in range bar data

---

## Quick Reference

| Document                                                                 | Lines | Purpose                            | Status      |
| ------------------------------------------------------------------------ | ----- | ---------------------------------- | ----------- |
| [pattern-research-summary.md](pattern-research-summary.md)               | 944   | **Master document** - all findings | COMPLETE    |
| [adversarial-audit-methodology.md](adversarial-audit-methodology.md)     | 303   | Reusable audit framework           | PRODUCTION  |
| [market-regime-patterns.md](market-regime-patterns.md)                   | 1172  | SMA/RSI regime analysis            | INVALIDATED |
| [volatility-regime-patterns.md](volatility-regime-patterns.md)           | 421   | RV-based regime analysis           | INVALIDATED |
| [tda-regime-patterns.md](tda-regime-patterns.md)                         | 391   | TDA structural break patterns      | INVALIDATED |
| [tda-parameter-sensitivity-audit.md](tda-parameter-sensitivity-audit.md) | 530   | TDA parameter sweep audit          | AUDIT       |
| [multi-threshold-patterns.md](multi-threshold-patterns.md)               | 152   | Cross-threshold alignment          | INVALIDATED |
| [multifactor-patterns.md](multifactor-patterns.md)                       | 163   | Multi-factor combinations          | INVALIDATED |
| [price-action-patterns.md](price-action-patterns.md)                     | 313   | Direction patterns (U/D)           | INVALIDATED |
| [labeling-for-ml.md](labeling-for-ml.md)                                 | 773   | ML labeling methodology            | REFERENCE   |

### External References

| Document                                                                                             | Purpose              |
| ---------------------------------------------------------------------------------------------------- | -------------------- |
| [external/time-to-convergence-stationarity-gap.md](external/time-to-convergence-stationarity-gap.md) | MRH framework theory |

---

## Reading Order

### For New Researchers

1. **Start here**: [pattern-research-summary.md](pattern-research-summary.md)
   - Executive summary of all findings
   - Why patterns fail (boundary-locked returns, temporal overlap)
   - Final conclusion

2. **Methodology**: [adversarial-audit-methodology.md](adversarial-audit-methodology.md)
   - Reusable audit framework
   - Checklist for pattern validation
   - Lessons learned

3. **Deep dives** (optional, all INVALIDATED):
   - [market-regime-patterns.md](market-regime-patterns.md) - SMA/RSI conditioning
   - [tda-regime-patterns.md](tda-regime-patterns.md) - TDA structural breaks
   - [volatility-regime-patterns.md](volatility-regime-patterns.md) - RV-based regimes

### For ML Engineers

1. [labeling-for-ml.md](labeling-for-ml.md) - Triple barrier labeling
2. [pattern-research-summary.md](pattern-research-summary.md) - Why range bar patterns don't work

---

## Key Findings Summary

### Definitive Negative Result

After exhaustive testing across **9 research approaches**, we found **ZERO ODD robust predictive patterns** in range bar data.

### Why Patterns Fail

| Cause                       | Impact                                                    |
| --------------------------- | --------------------------------------------------------- |
| **Boundary-locked returns** | Returns mechanically bounded by threshold                 |
| **Temporal overlap**        | 75-100% consecutive bar overlap (deferred-open)           |
| **Long memory**             | Hurst H ~ 0.79 reduces effective sample to ~30 from 10000 |
| **Lookback leakage**        | Regime indicators reflect past, not future                |

### What Range Bars ARE Good For

| Application              | Status                      |
| ------------------------ | --------------------------- |
| Volatility normalization | ✅ Proven                   |
| Execution quality        | ✅ Unexplored but promising |
| Market making analytics  | ✅ Unexplored but promising |

### What Range Bars Are NOT Good For

| Application            | Status         |
| ---------------------- | -------------- |
| Directional prediction | ❌ INVALIDATED |
| Pattern-based alpha    | ❌ INVALIDATED |
| Signal generation      | ❌ INVALIDATED |

---

## GitHub Issues

| Issue | Title                                           | Status |
| ----- | ----------------------------------------------- | ------ |
| #52   | Market Regime Filter for ODD Robust Patterns    | CLOSED |
| #54   | Volatility Regime Filter                        | CLOSED |
| #55   | Multi-Threshold Pattern Confirmation            | CLOSED |
| #56   | TDA Structural Break Detection                  | CLOSED |
| #57   | **Research Complete: ZERO ODD Robust Patterns** | CLOSED |

---

## Scripts (42+ total)

All research scripts are in `scripts/`. Key categories:

- **Pattern Analysis**: `*_patterns*.py` - INVALIDATED
- **TDA Research**: `tda_*.py` - INVALIDATED
- **Audit Scripts**: `*_audit*.py` - Methodology
- **Cache Population**: `fill_gaps_*.py` - Infrastructure

Full listing in [pattern-research-summary.md](pattern-research-summary.md#scripts-reference).

---

## Data Coverage

| Symbol  | Period    | 50 dbps | 100 dbps | 200 dbps |
| ------- | --------- | ------- | -------- | -------- |
| BTCUSDT | 2022-2026 | 6.9M    | 2.8M     | 303K     |
| ETHUSDT | 2022-2026 | 10.1M   | 5.0M     | 499K     |
| SOLUSDT | 2022-2026 | 40.8M   | 12.5M    | 1.07M    |
| BNBUSDT | 2022-2026 | 13.2M   | 4.8M     | 389K     |

**Total**: 260M+ bars in ClickHouse cache (bigblack)
