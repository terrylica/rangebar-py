# RESUME.md - Session Context

**Last Updated**: 2026-02-02
**Purpose**: Quick context for continuing work in future sessions

---

## Current Work: Issue #59 Inter-Bar Features

### Status: Phase 1 COMPLETE (Rust Core)

**Objective**: Enrich 1000 dbps bars with microstructure features computed from raw aggTrades in a lookback window BEFORE each bar opens.

**Plan**: [docs/plans/issue-59-inter-bar-features.md](/docs/plans/issue-59-inter-bar-features.md)

### Completed (Phase 1)

- Created `crates/rangebar-core/src/interbar.rs` - 16 inter-bar features
- Added 16 `Option<T>` fields to `RangeBar` struct
- All 222 workspace tests pass
- Academic validation: Kyle (1985), Goh-Barabási (2008), Garman-Klass (1980), Bandt-Pompe (2002)

### Remaining Phases

| Phase | Task                                                   | Status |
| ----- | ------------------------------------------------------ | ------ |
| 2     | Integrate `TradeHistory` into `RangeBarProcessor`      | TODO   |
| 3     | Update PyO3 bindings to expose features to Python      | TODO   |
| 4     | Update Python type hints, constants, ClickHouse schema | TODO   |

### Key Files

| File                                    | Purpose                            |
| --------------------------------------- | ---------------------------------- |
| `crates/rangebar-core/src/interbar.rs`  | Core feature computation (NEW)     |
| `crates/rangebar-core/src/types.rs`     | RangeBar struct with 16 new fields |
| `crates/rangebar-core/src/processor.rs` | Needs TradeHistory integration     |

---

## Pattern Research: COMPLETE

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

**Data Storage Policy**: All data must be stored on remote ClickHouse (bigblack primary, littleblack secondary). Local storage is FORBIDDEN.

**Crypto**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
**Forex**: EURUSD (Exness Raw_Spread, Issue #143-#145)
**Thresholds**: 25, 50, 100, 200, 250 dbps (crypto); 50, 100, 200 dbps (forex)
**Period**: 2022-2026 (continuous, all gaps filled)

### Documentation

| Document                                         | Purpose                  |
| ------------------------------------------------ | ------------------------ |
| `docs/research/INDEX.md`                         | Navigation index         |
| `docs/research/pattern-research-summary.md`      | Master findings          |
| `docs/research/adversarial-audit-methodology.md` | Reusable audit framework |
| `docs/plans/issue-59-inter-bar-features.md`      | Inter-bar features plan  |

---

## What NOT to Retry

These approaches have been definitively invalidated:

1. **Any direction-based pattern** - Returns are boundary-locked
2. **Duration-based volatility prediction** - Mechanical artifact
3. **TDA velocity for forward RV** - No predictive power
4. **Microstructure features for alpha** - Noise exceeds signal
5. **Cross-threshold alignment signals** - Temporal overlap contamination

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

If continuing Issue #59:

1. Phase 1 (Rust core) is complete - 73 tests in interbar.rs pass
2. Next: Integrate `TradeHistory` into `RangeBarProcessor::process_trade()`
3. Plan file has detailed pseudocode for Phase 2-4

If continuing pattern research:

1. **DO NOT** retry invalidated approaches
2. The methodology is sound - the finding is genuinely negative
3. 10 approaches exhausted = research is complete

---

## References

- Root hub: [CLAUDE.md](/CLAUDE.md)
- Crates: [crates/CLAUDE.md](/crates/CLAUDE.md)
- Research index: [docs/research/INDEX.md](/docs/research/INDEX.md)
- Inter-bar plan: [docs/plans/issue-59-inter-bar-features.md](/docs/plans/issue-59-inter-bar-features.md)
