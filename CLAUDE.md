# CLAUDE.md - Project Memory

**rangebar-py**: Rust workspace with Python bindings via PyO3/maturin. Publishes to PyPI (`rangebar`) and crates.io (`rangebar-core`, `rangebar-providers`, `rangebar-streaming`).

---

## Quick Reference

| Task                     | Entry Point                                                                                             | Details                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Generate range bars      | `get_range_bars()`                                                                                      | [Python API](#python-api)                             |
| Real-time streaming      | `run_sidecar()` / `scripts/streaming_sidecar.py`                                                        | [ADR](/docs/adr/2026-01-31-realtime-streaming-api.md) |
| Understand architecture  | [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)                                                           | 8-crate workspace                                     |
| Work with Rust crates    | [crates/CLAUDE.md](/crates/CLAUDE.md)                                                                   | Core algorithm, microstructure, inter-bar features    |
| Work with Python layer   | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)                                                 | API, caching, validation, symbol registry             |
| ClickHouse cache ops     | [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md)                           | Schema, population, remote setup, dedup hardening     |
| Plugin system            | [python/rangebar/plugins/CLAUDE.md](/python/rangebar/plugins/CLAUDE.md)                                 | FeatureProvider protocol, entry-point discovery       |
| Operations & scripts     | [scripts/CLAUDE.md](/scripts/CLAUDE.md)                                                                 | Pueue, cache population, per-year parallelism         |
| Release workflow         | [docs/development/RELEASE.md](/docs/development/RELEASE.md)                                             | Zig cross-compile, mise tasks                         |
| Deploy to bigblack       | `mise run deploy:bigblack`                                                                              | Git pull + PyPI install + verify                      |
| Performance monitoring   | [docs/development/PERFORMANCE.md](/docs/development/PERFORMANCE.md)                                     | Benchmarks, metrics                                   |
| Performance optimization | [.claude/pgo-workflow.md](/.claude/pgo-workflow.md)                                                     | LTO, PGO, SIMD, GPU strategies (Issue #96)            |
| Streaming sidecar (P0)   | [docs/development/ISSUE-107-SIDECAR-RELIABILITY.md](/docs/development/ISSUE-107-SIDECAR-RELIABILITY.md) | Watchdog, error recovery, systemd auto-restart        |
| Project context          | [docs/CONTEXT.md](/docs/CONTEXT.md)                                                                     | Why this project exists                               |
| API reference            | [docs/api/INDEX.md](/docs/api/INDEX.md)                                                                 | Full Python API docs                                  |
| Research                 | [docs/research/INDEX.md](/docs/research/INDEX.md)                                                       | ML labeling, regime patterns, TDA                     |
| Oracle verification      | [docs/verification/](/docs/verification/)                                                               | Bit-exact cross-reference reports                     |
| Source validation        | [.claude/skills/source-validation/SKILL.md](/.claude/skills/source-validation/SKILL.md)                 | Vision→REST→WS junction proof, fromId pagination      |

---

## Critical Principles

### 1. Leverage Rust

**ALWAYS use local Rust crates for heavy lifting.** Python handles I/O only.

1. **Check Rust first**: Before writing Python code, check if `rangebar-core` already provides the capability
2. **Stream to Rust**: The `RangeBarProcessor` maintains state between `process_trades()` calls
3. **Checkpoint API**: Use `create_checkpoint()` and `from_checkpoint()` for cross-session continuity
4. **No reinventing**: Don't reimplement range bar logic in Python

```python
# CORRECT: Stream to Rust (maintains state automatically)
for chunk in data_stream:
    bars = processor.process_trades(chunk.to_dicts())

# WRONG: Buffer in Python (OOM risk)
all_data = []
for chunk in data_stream:
    all_data.extend(chunk)
```

### 2. Per-Year Parallelization (Default for Cache Population)

**Never queue a monolithic multi-year cache population job.** Ouroboros resets processor state at year boundaries, making each year an independent unit. Always split into per-year pueue jobs.

- A single DOGEUSDT@500 job: **22 days**. Per-year splits: **3-4 days**.
- Each year gets its own checkpoint file, ClickHouse dedup handles overlaps.
- **Full details**: [scripts/CLAUDE.md](/scripts/CLAUDE.md#per-year-parallelization-default-strategy)
- **Skills**: `Skill(devops-tools:pueue-job-orchestration)` | `Skill(devops-tools:distributed-job-safety)`

### 3. Symbol Registry Gate

**Every symbol must be registered before processing.** Unregistered symbols raise `SymbolNotRegisteredError`.

- **SSoT**: `symbols.toml` (repo root symlink → `python/rangebar/data/symbols.toml`)
- **Adding new symbols**: Edit `symbols.toml`, run `maturin develop`, then process
- **Override** (dev only): `export RANGEBAR_SYMBOL_GATE=off`
- **Full details**: [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md#symbol-registry-issue-79)

---

## Python API

```python
from rangebar import get_range_bars, get_n_range_bars, process_trades_polars

# Date-bounded (backtesting)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Count-bounded (ML training)
df = get_n_range_bars("BTCUSDT", n_bars=10000)

# Polars users (2-3x faster)
import polars as pl
trades = pl.scan_parquet("trades.parquet")
df = process_trades_polars(trades, threshold_decimal_bps=250)

# With microstructure features
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", include_microstructure=True)

# Long ranges (>30 days) — must populate cache first
from rangebar import populate_cache_resumable
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

| API                          | Use Case                   | Details                                                   |
| ---------------------------- | -------------------------- | --------------------------------------------------------- |
| `get_range_bars()`           | Date range, backtesting    | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `get_n_range_bars()`         | Exact N bars, ML           | [docs/api/primary-api.md](/docs/api/primary-api.md)       |
| `populate_cache_resumable()` | Long ranges (>30 days)     | [docs/api/cache-api.md](/docs/api/cache-api.md)           |
| `process_trades_polars()`    | Polars DataFrames          | [docs/api/processing-api.md](/docs/api/processing-api.md) |
| `process_trades_chunked()`   | Large datasets >10M trades | [docs/api/processing-api.md](/docs/api/processing-api.md) |

---

## Architecture

```
rangebar-py/
├── crates/                    8 Rust crates (3 on crates.io, 5 internal)
│   └── rangebar-core/         Core algorithm, microstructure features
├── src/                       PyO3 bindings (Issue #94: domain modules)
│   ├── lib.rs                 Thin orchestrator (~200 lines)
│   ├── helpers.rs             Conversion functions
│   ├── core_bindings.rs       PyRangeBarProcessor, PyPositionVerification
│   ├── arrow_bindings.rs      Arrow export functions
│   ├── binance_bindings.rs    Binance data fetching
│   ├── exness_bindings.rs     Exness data fetching
│   └── streaming_bindings.rs  LiveBarEngine, streaming classes
├── python/rangebar/           Python API layer
│   ├── clickhouse/            ClickHouse cache (bigblack)
│   ├── sidecar.py             Streaming sidecar orchestrator (v12.20+)
│   ├── validation/            Microstructure validation
│   └── storage/               Tier 1 cache (Parquet)
├── scripts/                   Pueue jobs, validation, cache population
├── .cargo/config.toml         Cross-compile friendly rustflags
└── pyproject.toml             Maturin config
```

**Key files**: `src/lib.rs` (PyO3 module registration), `python/rangebar/__init__.py` (public API), `crates/rangebar-core/src/processor.rs` (core algorithm), `python/rangebar/sidecar.py` (streaming sidecar)

**Full architecture**: [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)

---

## Development Commands

```bash
# Setup (mise manages all tools including zig)
mise install

# Build & test
mise run build              # maturin develop
mise run test               # Rust tests (cargo nextest)
mise run test-py            # Python tests (pytest)

# Quality
mise run check-full         # fmt + lint + test + deny

# Release (see docs/development/RELEASE.md)
mise run release:full       # Full 4-phase workflow
mise run release:linux      # Zig cross-compile (~55 sec)
mise run publish            # Upload to PyPI

# Benchmarks
mise run bench:run          # Full benchmarks
mise run bench:validate     # Verify 1M ticks < 100ms
```

---

## Build System

Zig cross-compilation from macOS (no remote SSH needed).

| Platform     | Strategy          | Time    | Command                        |
| ------------ | ----------------- | ------- | ------------------------------ |
| macOS ARM64  | Native maturin    | ~10 sec | `mise run release:macos-arm64` |
| Linux x86_64 | Zig cross-compile | ~55 sec | `mise run release:linux`       |

**Why rustls**: Eliminates OpenSSL dependency, enabling pure Rust cross-compilation.

**Full release workflow**: [docs/development/RELEASE.md](/docs/development/RELEASE.md)

---

## Microstructure Features

10 intra-bar features + 16 inter-bar lookback features computed in Rust during bar construction.

**Intra-bar**: `ofi`, `vwap_close_deviation`, `kyle_lambda_proxy`, `trade_intensity`, `volume_per_trade`, `aggression_ratio`, `aggregation_density`, `turnover_imbalance`, `duration_us`, `price_impact`

**Inter-bar** (lookback window): `lookback_ofi`, `lookback_intensity`, `lookback_kyle_lambda`, `lookback_burstiness`, `lookback_hurst`, `lookback_permutation_entropy`, + 10 more

**Formulas and ranges**: [crates/CLAUDE.md](/crates/CLAUDE.md#microstructure-features-v70)

**Validation**: [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md#validation-framework-v70)

---

## ClickHouse Infrastructure

All range bar data served from **bigblack** (remote GPU host). No local ClickHouse.

| Host        | Thresholds (dbps)   | ClickHouse | Total Bars |
| ----------- | ------------------- | ---------- | ---------- |
| bigblack    | 250, 500, 750, 1000 | Native     | 260M+      |
| littleblack | 100                 | Docker     | —          |

**Connection mode**: `RANGEBAR_MODE=remote` (set in `.mise.toml`). Preflight: `mise run db:ensure`.

**Cache operations**: [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) (schema, population, dedup hardening)

**Distributed jobs**: [scripts/CLAUDE.md](/scripts/CLAUDE.md) (pueue, per-year parallelization, autoscaler)

---

## Monitoring & Gap Prevention

Automated monitoring to prevent silent data loss in the ClickHouse cache. Born from the Feb 2026 incident where 5.5 days of BTCUSDT@500 bars were lost due to `try_cache_write()` silently swallowing connection errors.

### Root Cause (Resolved)

`try_cache_write()` in `orchestration/range_bars_cache.py` was fire-and-forget — `populate_cache_resumable()` reused it even though ClickHouse IS the destination, not an optional cache. When SSH tunnel dropped, writes failed silently for days while the checkpoint advanced.

### Three-Layer Defense

| Layer             | Mechanism                                      | Details                                           |
| ----------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Fatal writes**  | `_fatal_cache_write()` in populate path        | Raises on ClickHouse failure instead of warning   |
| **T-1 guard**     | `populate_cache_resumable()` clamps `end_date` | Prevents crash on unavailable Binance Vision data |
| **Gap detection** | `scripts/detect_gaps.py` + heartbeat           | Every 15 min via launchd, alerts via @rangebarbot |

### Telegram Notifications (@rangebarbot)

| Event                      | Behavior                                                   |
| -------------------------- | ---------------------------------------------------------- |
| All checks pass            | Ephemeral green heartbeat, auto-deletes after 2 min        |
| Gap or regression detected | **Persistent** red alert, LOUD notification, stays forever |
| Cache write failure        | Hook-driven alert via `rangebar.notify.telegram`           |

**Config**: `RANGEBAR_TELEGRAM_TOKEN` + `RANGEBAR_TELEGRAM_CHAT_ID` in `.mise.toml`

**Heartbeat binary**: `~/.claude/automation/rangebar-heartbeat/rangebar-heartbeat` (compiled Swift, launchd)

**Full details**: [scripts/CLAUDE.md](/scripts/CLAUDE.md#gap-detection--monitoring)

---

## Common Errors

| Error                                   | Cause                  | Fix                                        |
| --------------------------------------- | ---------------------- | ------------------------------------------ |
| `RangeBarProcessor has no attribute X`  | Outdated binding       | `maturin develop`                          |
| `Invalid threshold_decimal_bps`         | Wrong units            | Use 250 for 0.25%                          |
| `High < Low` assertion                  | Bad input data         | Check sorting                              |
| `target-cpu=native` cross-compile error | RUSTFLAGS pollution    | Use `RUSTFLAGS=""` or `.cargo/config.toml` |
| OOM with `include_microstructure=True`  | Large date range       | MEM-011 adaptive chunk size                |
| Duplicate ticks in Parquet cache        | Pre-v12.8 bug          | `scripts/deduplicate_parquet_cache.py`     |
| `SymbolNotRegisteredError`              | Symbol not in registry | Edit `symbols.toml`, `maturin develop`     |

**Memory guards**: [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) (MEM-001 through MEM-013)

---

## CLAUDE.md Network (Hub-and-Spoke)

This file is the **hub**. Each spoke CLAUDE.md is loaded automatically when working in that directory.

| Directory                      | CLAUDE.md                                                                     | SSoT For                                                           |
| ------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `/`                            | This file                                                                     | Project overview, API, architecture, principles                    |
| `/crates/`                     | [crates/CLAUDE.md](/crates/CLAUDE.md)                                         | Rust crates, microstructure formulas, algorithm details            |
| `/python/rangebar/`            | [python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)                       | Python API, caching, validation, symbol registry, memory guards    |
| `/python/rangebar/clickhouse/` | [python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) | ClickHouse schema, dedup hardening, cache population               |
| `/python/rangebar/plugins/`    | [python/rangebar/plugins/CLAUDE.md](/python/rangebar/plugins/CLAUDE.md)       | FeatureProvider protocol, entry-point discovery, plugin authoring  |
| `/scripts/`                    | [scripts/CLAUDE.md](/scripts/CLAUDE.md)                                       | Pueue ops, per-year parallelism, gap detection, monitoring scripts |

---

## Terminology

| Term                  | Definition                                                                                                                                                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **dbps**              | 1 dbps = 0.00001 = 0.001%. Example: 250 dbps = 0.25%. **All threshold values use dbps.**                                                                                                                                           |
| **Ouroboros**         | Cyclical reset boundary (year/month/week). Resets processor state for reproducibility and cache-friendly processing.                                                                                                               |
| **Orphan Bar**        | Incomplete bar emitted when processor resets at an ouroboros boundary. Marked `is_orphan=True`. Monthly mode: 12/year; yearly: 1/year. Filter for ML: `df[~df.get("is_orphan", False)]`. Synonyms: "orphaned bar", "boundary bar". |
| **Dynamic Ouroboros** | Forex-specific: reset at first tick after weekend gap, auto-handling DST shifts.                                                                                                                                                   |
| **Kintsugi**          | Self-healing gap reconciliation (Issue #115). Discovers gaps ("shards") and repairs using Ariadne+Ouroboros.                                                                                                                       |
| **Shard**             | A gap in the data timeline. Classified P0 (staleness), P1 (recent <48h), P2 (historical >=48h). Repaired by Kintsugi.                                                                                                              |

<!-- gitnexus:start -->

# GitNexus MCP

This project is indexed by GitNexus as **rangebar-py** (7737 symbols, 20106 relationships, 300 execution flows).

## Always Start Here

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task                                         | Read this skill file                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md`       |
| Blast radius / "What breaks if I change X?"  | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?"             | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md`       |
| Rename / extract / split / refactor          | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md`     |
| Tools, resources, schema reference           | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md`           |
| Index, status, clean, wiki CLI commands      | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`             |

<!-- gitnexus:end -->
