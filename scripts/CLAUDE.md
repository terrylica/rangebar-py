# Scripts & Operations

**Parent**: [/CLAUDE.md](/CLAUDE.md) | **Skills**: `Skill(devops-tools:pueue-job-orchestration)` | `Skill(rangebar-job-safety)`

Operational scripts for cache population, validation, and distributed job management on remote GPU hosts.

---

## Quick Reference

| Task                       | Command                                           | Details                                               |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| Queue all cache population | `mise run cache:populate-all`                     | [Pueue Population](#pueue-population)                 |
| Check pueue status         | `mise run cache:populate-status`                  | [Monitoring](#monitoring)                             |
| Per-year parallelization   | `--start-date` / `--end-date`                     | [Per-Year Parallelization](#per-year-parallelization) |
| Autoscale parallelism      | `mise run cache:autoscale-loop`                   | [Autoscaler](#autoscaler)                             |
| Check cache status         | `mise run cache:status`                           | ClickHouse bar counts                                 |
| Detect volume overflow     | `mise run cache:detect-overflow`                  | Issue #88                                             |
| Validate dedup hardening   | `mise run cache:validate-dedup`                   | Issue #90                                             |
| Validate microstructure    | `uv run python scripts/validate_microstructure_*` | [Validation Scripts](#validation-scripts)             |
| Run streaming sidecar      | `mise run streaming:sidecar`                      | v12.20+, real-time range bars                         |

---

## Pueue Population

**State-of-the-art approach** for long-running cache population on remote hosts. Pueue daemon survives SSH disconnects, crashes, and reboots.

### Architecture

```
mise run cache:populate-all
  └─→ pueue-populate.sh
        └─→ pueue add (per symbol × threshold)
              └─→ [systemd-run --scope -p MemoryMax=XG]  (Linux only)
                    └─→ uv run python populate_full_cache.py
                          └─→ populate_cache_resumable()   (day-by-day, checkpointed)
```

### Key Files

| Script                      | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| `pueue-populate.sh`         | Pueue job orchestrator (groups, phases, guards) |
| `populate_full_cache.py`    | Single-job entry point (Python)                 |
| `pueue-autoscaler.sh`       | Dynamic parallelism tuning (CPU/memory aware)   |
| `detect_volume_overflow.py` | Post-population integrity check (Issue #88)     |
| `streaming_sidecar.py`      | Streaming sidecar CLI entry point (v12.20+)     |

### Phases

| Phase | Threshold (dbps) | Pueue Group | Parallel | Memory/Job |
| ----- | ---------------- | ----------- | -------- | ---------- |
| 1     | 1000             | p1          | 4        | ~1 GB      |
| 2     | 250              | p2          | 2        | ~5 GB      |
| 3     | 500, 750         | p3          | 3        | ~1.5 GB    |

### Setup (One-Time on Remote Host)

```bash
# Install pueue
~/.local/bin/pueued -d                      # Start daemon
./scripts/pueue-populate.sh setup           # Configure groups

# Or via mise
mise run cache:populate-setup
```

### Resource Guards

On Linux with cgroups v2, each job runs inside a `systemd-run` scope with per-threshold memory caps and `MemorySwapMax=0` (prevents swap escape). Bypass with `RANGEBAR_NO_CGROUP=1`.

---

## Per-Year Parallelization (Default Strategy)

**This is the default approach for all multi-year cache population.** Never queue a monolithic multi-year job. Ouroboros resets processor state at year boundaries, making each year an independent processing unit. Per-year splits provide massive speedup on multi-core hosts (22 days → 3-4 days for DOGEUSDT@500).

### Why It's Safe (Three Isolation Layers)

| Layer                 | Why No Conflicts                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------- |
| **Checkpoint files**  | `_get_checkpoint_path()` uses `{symbol}_{threshold}_{start}_{end}.json` — unique per year |
| **ClickHouse writes** | INSERT is append-only; `OPTIMIZE TABLE FINAL` deduplicates afterward                      |
| **Tick data**         | Read-only Parquet files; no write contention                                              |

### Usage

```bash
# populate_full_cache.py supports --start-date and --end-date overrides
uv run python scripts/populate_full_cache.py \
    --symbol SHIBUSDT --threshold 250 --include-microstructure \
    --start-date 2021-05-10 --end-date 2021-12-31

# Queue per-year jobs via pueue
pueue group add yearly --parallel 5
for year in 2021 2022 2023 2024 2025; do
    pueue add --group yearly --label "SHIB@250:${year}" -- \
        uv run python scripts/populate_full_cache.py \
        --symbol SHIBUSDT --threshold 250 --include-microstructure \
        --start-date "${year}-01-01" --end-date "${year}-12-31"
done
```

### Critical Rules

1. **Working directory**: Always `cd ~/rangebar-py &&` before `pueue add` — SSH cwd defaults to `$HOME`, causing instant job failure with `No such file or directory`.
2. **No `--force-refresh` on per-year jobs** when other year-jobs are running — `force_refresh` deletes cached bars for the specified date range, but if the original full-range job already ran `force_refresh`, the cache is already empty.
3. **First year uses symbol's `effective_start`** from `symbols.toml`, not `01-01`.
4. **Last year uses `probe_latest_available_date()`** as end date.
5. **Chain `OPTIMIZE TABLE FINAL`** after all year-jobs complete via `pueue add --after`.
6. **Memory budget**: Each job peaks at ~2-8 GB depending on threshold. With 61 GB total, 4-5 concurrent year-jobs are safe.

### Anti-Patterns (Hard Lessons)

| Anti-Pattern                       | What Happened                                                                                  | Fix                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Monolithic multi-year job          | DOGEUSDT@500 estimated 22 days single-threaded                                                 | Split into per-year pueue jobs (22d → 3-4d)      |
| Wrong cwd in remote pueue          | `ssh host "pueue add ..."` used `$HOME` as cwd                                                 | `cd ~/rangebar-py && pueue add ...`              |
| OPTIMIZE TABLE timeout             | Synchronous OPTIMIZE timed out at 300s under load                                              | Made OPTIMIZE non-fatal (Issue #90)              |
| Cascade dependency failure         | OPTIMIZE TABLE chained `--after` populate jobs; when any populate failed, OPTIMIZE was skipped | Run OPTIMIZE manually after retrying failed jobs |
| Per-job env override bypasses SSoT | `env RANGEBAR_CRYPTO_MIN_THRESHOLD=250` in each pueue job                                      | Set in `.mise.toml [env]` instead (one place)    |

### When to Use Per-Year vs Sequential

| Scenario                                | Approach                 |
| --------------------------------------- | ------------------------ |
| High-volume symbol (SHIBUSDT, DOGEUSDT) | Per-year (5+ cores idle) |
| Low-volume symbol (NEARUSDT, ATOMUSDT)  | Sequential (fast enough) |
| Single threshold, long backfill         | Per-year                 |
| Multiple thresholds, same symbol        | Sequential per threshold |

### Performance Context

SHIBUSDT@250 on 2021-05-10 (listing pump day): **3.1M trades → 128K bars**. Single-threaded intra-bar feature computation (Hurst DFA, Permutation Entropy) takes 2-4 hours for this one day. Per-year parallelization moves later years forward while 2021 grinds through the hard days.

---

## Autoscaler

Pueue has no resource awareness. The autoscaler complements it with dynamic parallelism tuning.

```bash
mise run cache:autoscale         # Dry-run (shows what would change)
mise run cache:autoscale-apply   # Apply changes
mise run cache:autoscale-loop    # Continuous (60s interval)
```

**Scaling thresholds**:

```
CPU < 40% AND MEM < 60%  →  SCALE UP (+1 per group)
CPU > 80% OR  MEM > 80%  →  SCALE DOWN (-1 per group)
Otherwise                 →  HOLD
```

---

## Monitoring

```bash
# Pueue status
pueue status                          # All groups
pueue status --group p2               # Specific group
pueue follow <id>                     # Watch live output
pueue log <id> --lines 20            # Recent output

# Cache status
mise run cache:status                 # ClickHouse bar counts
python scripts/cache_status.py        # Direct

# System resources (bigblack)
ssh bigblack 'uptime && free -h'
```

---

## Gap Detection & Monitoring

Automated gap detection to prevent silent data loss in ClickHouse. Born from the Feb 2026 BTCUSDT@500 incident (5.5-day gap caused by `try_cache_write()` silently swallowing SSH tunnel failures).

### Scripts

| Script                           | Purpose                                            | Exit Code              |
| -------------------------------- | -------------------------------------------------- | ---------------------- |
| `detect_gaps.py`                 | Find temporal/price gaps + freshness in ClickHouse | 0=clean, 1=gaps, 2=err |
| `verify_checkpoint_integrity.py` | Cross-validate checkpoint files vs ClickHouse      | 0=clean, 1=mismatch    |
| `setup_daily_gap_check.sh`       | Install daily pueue monitoring job on bigblack     | —                      |

### Usage

```bash
# mise tasks
mise run cache:check-gaps            # All symbols, all time
mise run cache:check-gaps:recent     # Last 30 days, 6h threshold

# Direct script usage
python scripts/detect_gaps.py --help
python scripts/detect_gaps.py --symbol BTCUSDT --threshold 500
python scripts/detect_gaps.py --recent-days 7 --json

# Freshness check (flag stale data >72 hours since last bar)
python scripts/detect_gaps.py --recent-days 7 --max-stale-hours 72

# Checkpoint integrity
python scripts/verify_checkpoint_integrity.py --quiet
```

### Freshness Detection

The `--max-stale-hours N` flag computes `now - latest_bar` for each (symbol, threshold) pair. If any pair hasn't produced a bar in N hours, it's flagged as stale and triggers exit code 1. The heartbeat uses `--max-stale-hours 72` (3 days).

This catches scenarios where ALL thresholds for a symbol stop simultaneously (e.g., backfill_watcher not running) — something inter-bar gap detection alone misses because it only checks gaps between consecutive bars within existing data.

### Telegram Alerts (@rangebarbot)

Gap detection integrates with `rangebar.notify.telegram`:

- **Gaps found** → persistent alert (stays forever, LOUD notification)
- **All clear** → no message (or ephemeral heartbeat via launchd)
- **`--no-notify`** flag suppresses Telegram (for dry runs)

### Heartbeat (launchd)

Compiled Swift binary at `~/.claude/automation/rangebar-heartbeat/rangebar-heartbeat` runs every 15 min via launchd (`com.eonlabs.rangebar-heartbeat`). Executes `detect_gaps.py` + `verify_checkpoint_integrity.py`, sends ephemeral green heartbeat (auto-deletes 2 min) on clean, persistent red alert on regression.

### Gap Detection Query (SSoT)

Uses ClickHouse `neighbor()` window function to compute inter-bar time deltas within each `(symbol, threshold)` partition. Gaps exceeding `--min-gap-hours` (default 6h) are flagged.

---

## Validation Scripts

Portable scripts for GPU workstations without full dev environment:

| Script                                 | Purpose                             |
| -------------------------------------- | ----------------------------------- |
| `validate_n_range_bars.py`             | Count-bounded API validation        |
| `validate_microstructure_features.py`  | v7.0 feature validation             |
| `validate_microstructure_roundtrip.py` | Compute → store → read → verify     |
| `validate_backfill_preflight.py`       | Pre-population checks               |
| `validate_clickhouse.py`               | ClickHouse connectivity             |
| `validate_memory_efficiency.py`        | Memory usage during processing      |
| `deduplicate_parquet_cache.py`         | Fix pre-v12.8 duplicate ticks (#78) |
| `detect_volume_overflow.py`            | Find negative volumes (#88)         |
| `validate_dedup_hardening.py`          | Issue #90 dedup layer validation    |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/crates/CLAUDE.md](/crates/CLAUDE.md) - Rust crate details, microstructure features
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python API, caching, validation
- [/python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) - ClickHouse cache layer
- [/.mise/tasks/cache.toml](/.mise/tasks/cache.toml) - mise task definitions
