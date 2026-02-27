# FILE-SIZE-OK: Cohesive reconciliation engine (discover/repair/verify)
"""Kintsugi: Self-healing gap reconciliation for the ClickHouse cache (Issue #115).

Named after the Japanese art of repairing broken pottery with gold lacquer
(金継ぎ). Interior gaps ("shards") in the cache are discovered and repaired
autonomously, making the dataset stronger than before.

**Every repair uses Ariadne + Ouroboros by default.**  Ariadne's fromId cursor
pagination eliminates trade drops at millisecond boundaries.  Ouroboros
boundary-aware splits prevent processor state bleed across reset points.

Architecture
------------
::

    detect_gaps.py --json  →  discover_shards()  →  classify → repair → verify
                                                         │
                         ┌───────────────────────────────┘
                         ▼
              P0/P1: backfill_recent()     (now Ariadne-native, Phase 2)
              P2:    populate_cache_resumable()  (already Ariadne-native)

Usage
-----
Single pass::

    >>> from rangebar.kintsugi import kintsugi_pass
    >>> results = kintsugi_pass()

Daemon mode::

    >>> from rangebar.kintsugi import kintsugi_daemon
    >>> kintsugi_daemon()  # Long-running, adaptive polling
"""

from __future__ import annotations

import json
import logging
import os
import socket as _socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _sd_notify(state: str) -> None:
    """Send sd_notify message to systemd (if NOTIFY_SOCKET is set).

    This is a minimal implementation — no external dependency needed.
    Used for WatchdogSec integration (Issue #117).
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM)
        try:
            if addr.startswith("@"):
                addr = "\0" + addr[1:]
            sock.sendto(state.encode(), addr)
        finally:
            sock.close()
    except OSError:
        logger.debug("sd_notify(%s) failed", state, exc_info=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_P1_THRESHOLD_HOURS = 48.0  # Recent gap boundary
_MAX_ATTEMPTS_PER_SHARD = 3  # Per 24h, tracked in kintsugi_log
_DEFAULT_MAX_P2_JOBS = 1
_DETECT_GAPS_EXIT_ERROR = 2  # detect_gaps.py connection/query failure
_MIN_DEAD_LETTER_PARTS = 3  # symbol_threshold_timestamp
_MAX_FAILURES_IN_ALERT = 5  # Max failures shown in Telegram alert


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Shard:
    """A gap in the data timeline — a broken shard to be repaired with gold."""

    symbol: str
    threshold_decimal_bps: int
    gap_start_ms: int
    gap_end_ms: int
    gap_hours: float
    priority: str  # "P0", "P1", "P2"
    last_agg_trade_id_before: int | None = None  # Ariadne anchor
    ouroboros_boundaries: list[int] = field(default_factory=list)


@dataclass
class KintsugiResult:
    """Result of a single repair operation."""

    shard: Shard
    healed: bool
    bars_written: int
    duration_seconds: float
    ariadne_used: bool  # Always True unless first-ever backfill
    ouroboros_splits: int  # Number of boundary sub-repairs
    error: str | None = None


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


def discover_shards(
    *,
    min_gap_hours: float = 6.0,
    max_stale_hours: float = 72.0,
    symbol: str | None = None,
    threshold: int | None = None,
) -> list[Shard]:
    """Discover shards by running ``detect_gaps.py --json`` as subprocess.

    For each gap, queries ``last_agg_trade_id`` from the bar before the gap
    (Ariadne anchor) and computes Ouroboros boundaries within the gap.
    """
    cmd = [
        sys.executable, "-W", "ignore",
        "scripts/detect_gaps.py",
        "--json",
        "--min-gap-hours", str(min_gap_hours),
        "--max-stale-hours", str(max_stale_hours),
    ]
    if symbol:
        cmd.extend(["--symbol", symbol])
    if threshold:
        cmd.extend(["--threshold", str(threshold)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False,
        )
    except subprocess.TimeoutExpired:
        logger.exception("detect_gaps.py timed out after 120s")
        return []

    if result.returncode == _DETECT_GAPS_EXIT_ERROR:
        logger.error(
            "detect_gaps.py connection error: %s", result.stderr.strip(),
        )
        return []

    # Parse JSON output (exit code 0 = clean, 1 = gaps found)
    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.exception("failed to parse detect_gaps.py JSON output")
        return []

    raw_gaps = data.get("gaps", [])
    raw_stale = data.get("stale_pairs", [])

    shards: list[Shard] = []

    # Convert temporal gaps to shards
    for g in raw_gaps:
        if g.get("gap_type") not in ("temporal", "both"):
            continue
        shard = Shard(
            symbol=g["symbol"],
            threshold_decimal_bps=g["threshold_dbps"],
            gap_start_ms=g["gap_start_ms"],
            gap_end_ms=g["gap_end_ms"],
            gap_hours=g["gap_hours"],
            priority=classify_shard_priority(g["gap_hours"]),
        )
        shards.append(shard)

    # Convert stale pairs to P0 shards
    for s in raw_stale:
        shard = Shard(
            symbol=s["symbol"],
            threshold_decimal_bps=s["threshold_dbps"],
            gap_start_ms=0,  # Will be resolved by backfill_recent
            gap_end_ms=int(time.time() * 1000),
            gap_hours=s["hours_since_last_bar"],
            priority="P0",
        )
        shards.append(shard)

    # Enrich with Ariadne anchors and Ouroboros boundaries
    _enrich_shards(shards)

    # Issue #117-119: Emit GAP_DETECTED hook for each shard
    if shards:
        try:
            from rangebar.hooks import HookEvent, emit_hook
            for shard in shards:
                emit_hook(
                    HookEvent.GAP_DETECTED, symbol=shard.symbol,
                    threshold_dbps=shard.threshold_decimal_bps,
                    gap_hours=shard.gap_hours, priority=shard.priority,
                )
        except (ImportError, OSError, RuntimeError, ValueError):
            logger.debug("failed to emit GAP_DETECTED hooks", exc_info=True)

    return shards


def classify_shard_priority(gap_hours: float) -> str:
    """Classify gap into P0 (staleness), P1 (<48h), P2 (>=48h)."""
    if gap_hours < _P1_THRESHOLD_HOURS:
        return "P1"
    return "P2"


def _enrich_shards(shards: list[Shard]) -> None:
    """Add Ariadne anchors and Ouroboros boundaries to shards."""
    from rangebar.clickhouse import RangeBarCache
    from rangebar.ouroboros import get_ouroboros_boundaries

    if not shards:
        return

    try:
        with RangeBarCache() as cache:
            for shard in shards:
                # Ariadne anchor: last_agg_trade_id before the gap
                hwm = cache.get_ariadne_high_water_mark(
                    shard.symbol, shard.threshold_decimal_bps,
                )
                shard.last_agg_trade_id_before = hwm

                # Ouroboros boundaries within the gap
                if shard.gap_start_ms > 0 and shard.gap_end_ms > 0:
                    start_date = datetime.fromtimestamp(
                        shard.gap_start_ms / 1000, tz=UTC,
                    ).date()
                    end_date = datetime.fromtimestamp(
                        shard.gap_end_ms / 1000, tz=UTC,
                    ).date()
                    boundaries = get_ouroboros_boundaries(
                        start_date, end_date, "year",
                    )
                    shard.ouroboros_boundaries = [
                        b.timestamp_ms for b in boundaries
                    ]
    except (OSError, RuntimeError, ConnectionError) as e:
        logger.warning("failed to enrich shards: %s", e)


# ---------------------------------------------------------------------------
# Shard repair
# ---------------------------------------------------------------------------


def repair_shard(
    shard: Shard,
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> KintsugiResult:
    """Repair a single shard using Ariadne + Ouroboros.

    P0/P1 shards use ``backfill_recent()`` (now Ariadne-native).
    P2 shards use ``populate_cache_resumable()`` with per-boundary splits.
    """
    # Validate threshold before attempting repair — reject orphaned invalid data
    from rangebar.threshold import resolve_and_validate_threshold

    try:
        resolve_and_validate_threshold(
            shard.symbol, shard.threshold_decimal_bps,
        )
    except ValueError:
        return KintsugiResult(
            shard=shard,
            healed=False,
            bars_written=0,
            duration_seconds=0.0,
            ariadne_used=False,
            ouroboros_splits=0,
            error=(
                f"threshold {shard.threshold_decimal_bps} dbps "
                f"below minimum for {shard.symbol}"
            ),
        )

    t0 = time.monotonic()
    total_bars = 0
    n_splits = max(len(shard.ouroboros_boundaries), 1)
    ariadne_used = shard.last_agg_trade_id_before is not None

    try:
        if shard.priority in ("P0", "P1"):
            total_bars = _repair_p0_p1(
                shard,
                include_microstructure=include_microstructure,
                verbose=verbose,
            )
        else:
            total_bars = _repair_p2(
                shard,
                include_microstructure=include_microstructure,
                verbose=verbose,
            )
    except (OSError, RuntimeError, ConnectionError, ValueError) as e:
        elapsed = time.monotonic() - t0
        logger.exception(
            "repair failed for %s@%d (%s)",
            shard.symbol, shard.threshold_decimal_bps, shard.priority,
        )
        return KintsugiResult(
            shard=shard,
            healed=False,
            bars_written=0,
            duration_seconds=elapsed,
            ariadne_used=ariadne_used,
            ouroboros_splits=n_splits,
            error=str(e),
        )

    elapsed = time.monotonic() - t0
    healed = total_bars > 0

    logger.info(
        "repair %s for %s@%d: %d bars in %.1fs (ariadne=%s, splits=%d)",
        "HEALED" if healed else "NO_BARS",
        shard.symbol, shard.threshold_decimal_bps,
        total_bars, elapsed, ariadne_used, n_splits,
    )

    return KintsugiResult(
        shard=shard,
        healed=healed,
        bars_written=total_bars,
        duration_seconds=elapsed,
        ariadne_used=ariadne_used,
        ouroboros_splits=n_splits,
    )


def _repair_p0_p1(
    shard: Shard,
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> int:
    """Repair P0/P1 shard via backfill_recent() (Ariadne-native)."""
    from rangebar.recency import backfill_recent

    result = backfill_recent(
        shard.symbol,
        shard.threshold_decimal_bps,
        include_microstructure=include_microstructure,
        verbose=verbose,
    )
    if result.error:
        msg = f"backfill_recent error: {result.error}"
        raise RuntimeError(msg)
    return result.bars_written


def _repair_p2(
    shard: Shard,
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> int:
    """Repair P2 shard via populate_cache_resumable() with Ouroboros splits."""
    from rangebar.checkpoint import populate_cache_resumable

    total_bars = 0

    # Split at Ouroboros boundaries
    segments = _split_at_boundaries(shard)

    for seg_start, seg_end in segments:
        start_str = datetime.fromtimestamp(
            seg_start / 1000, tz=UTC,
        ).strftime("%Y-%m-%d")
        end_str = datetime.fromtimestamp(
            seg_end / 1000, tz=UTC,
        ).strftime("%Y-%m-%d")

        bars = populate_cache_resumable(
            shard.symbol,
            start_str,
            end_str,
            threshold_decimal_bps=shard.threshold_decimal_bps,
            include_microstructure=include_microstructure,
            verbose=verbose,
            notify=False,
        )
        total_bars += bars

    return total_bars


def _split_at_boundaries(
    shard: Shard,
) -> list[tuple[int, int]]:
    """Split a shard's time range at Ouroboros boundaries.

    Returns list of (start_ms, end_ms) segments.
    """
    if not shard.ouroboros_boundaries:
        return [(shard.gap_start_ms, shard.gap_end_ms)]

    segments: list[tuple[int, int]] = []
    current_start = shard.gap_start_ms

    for boundary_ms in sorted(shard.ouroboros_boundaries):
        if current_start < boundary_ms < shard.gap_end_ms:
            segments.append((current_start, boundary_ms - 1))
            current_start = boundary_ms

    segments.append((current_start, shard.gap_end_ms))
    return segments


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_repair(shard: Shard) -> bool:
    """Re-run gap detection for the specific pair to confirm gap is healed."""
    cmd = [
        sys.executable, "-W", "ignore",
        "scripts/detect_gaps.py",
        "--json",
        "--symbol", shard.symbol,
        "--threshold", str(shard.threshold_decimal_bps),
        "--min-gap-hours", str(max(shard.gap_hours * 0.5, 1.0)),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("verify_repair timed out for %s@%d",
                        shard.symbol, shard.threshold_decimal_bps)
        return False

    if result.returncode == 0:
        return True  # Clean — no gaps

    # Check if our specific gap still exists
    try:
        data = json.loads(result.stdout)
        for g in data.get("gaps", []):
            if (g.get("gap_start_ms") == shard.gap_start_ms
                    and g.get("gap_end_ms") == shard.gap_end_ms):
                return False  # Same gap still present
    except (json.JSONDecodeError, ValueError):
        return False

    return True  # Gap not found in results — healed


# ---------------------------------------------------------------------------
# Kintsugi log (ClickHouse)
# ---------------------------------------------------------------------------


def _log_repair(result: KintsugiResult) -> None:
    """Write repair result to kintsugi_log ClickHouse table."""
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            cache.client.command(
                "INSERT INTO rangebar_cache.kintsugi_log "
                "(symbol, threshold_decimal_bps, priority, "
                "gap_start_ms, gap_end_ms, gap_hours, "
                "action, result, ariadne_from_id, ouroboros_splits, "
                "bars_written, duration_seconds, error_message) "
                "VALUES "
                "({symbol:String}, {threshold:UInt32}, {priority:String}, "
                "{gap_start:Int64}, {gap_end:Int64}, {gap_hours:Float64}, "
                "{action:String}, {result:String}, {ariadne:Int64}, "
                "{splits:UInt8}, {bars:UInt32}, {duration:Float64}, "
                "{error:String})",
                parameters={
                    "symbol": result.shard.symbol,
                    "threshold": result.shard.threshold_decimal_bps,
                    "priority": result.shard.priority,
                    "gap_start": result.shard.gap_start_ms,
                    "gap_end": result.shard.gap_end_ms,
                    "gap_hours": result.shard.gap_hours,
                    "action": (
                        "backfill_recent"
                        if result.shard.priority in ("P0", "P1")
                        else "populate_cache"
                    ),
                    "result": "healed" if result.healed else "failed",
                    "ariadne": (
                        result.shard.last_agg_trade_id_before or 0
                    ),
                    "splits": result.ouroboros_splits,
                    "bars": result.bars_written,
                    "duration": result.duration_seconds,
                    "error": result.error or "",
                },
            )
    except (OSError, RuntimeError, ConnectionError) as e:
        logger.warning("failed to log kintsugi result: %s", e)


# ---------------------------------------------------------------------------
# Pass + daemon
# ---------------------------------------------------------------------------


def kintsugi_pass(
    *,
    dry_run: bool = False,
    symbol: str | None = None,
    max_p2_jobs: int = _DEFAULT_MAX_P2_JOBS,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> list[KintsugiResult]:
    """Single pass: discover shards, classify, repair, verify, log.

    Parameters
    ----------
    dry_run
        If True, discover and classify only — no repairs.
    symbol
        Filter to a specific symbol.
    max_p2_jobs
        Maximum concurrent P2 (historical) repairs per pass.
    include_microstructure
        Include microstructure features in repaired bars.
    verbose
        Enable detailed logging.

    Returns
    -------
    list[KintsugiResult]
        Results for each shard repaired (or discovered in dry_run).
    """
    pass_t0 = time.monotonic()
    logger.info("kintsugi pass starting (dry_run=%s)", dry_run)
    shards = discover_shards(symbol=symbol)

    if not shards:
        logger.info("kintsugi: no shards found — cache is clean")
        return []

    # Sort by priority: P0 first, then P1, then P2
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    shards.sort(key=lambda s: priority_order.get(s.priority, 9))

    logger.info(
        "kintsugi: discovered %d shards (P0=%d, P1=%d, P2=%d)",
        len(shards),
        sum(1 for s in shards if s.priority == "P0"),
        sum(1 for s in shards if s.priority == "P1"),
        sum(1 for s in shards if s.priority == "P2"),
    )

    if dry_run:
        for s in shards:
            logger.info(
                "  [DRY-RUN] %s %s@%d gap=%.1fh ariadne=%s ouro_splits=%d",
                s.priority, s.symbol, s.threshold_decimal_bps,
                s.gap_hours,
                s.last_agg_trade_id_before is not None,
                len(s.ouroboros_boundaries),
            )
        return []

    results: list[KintsugiResult] = []
    p2_count = 0

    for shard in shards:
        # Rate-limit P2 jobs
        if shard.priority == "P2":
            if p2_count >= max_p2_jobs:
                logger.info(
                    "skipping P2 shard %s@%d (max_p2_jobs=%d reached)",
                    shard.symbol, shard.threshold_decimal_bps, max_p2_jobs,
                )
                continue
            p2_count += 1

        # Check rate limit from kintsugi_log (max 3 attempts/24h)
        if _is_rate_limited(shard):
            logger.info(
                "skipping %s@%d — rate limited (%d attempts in 24h)",
                shard.symbol, shard.threshold_decimal_bps,
                _MAX_ATTEMPTS_PER_SHARD,
            )
            continue

        result = repair_shard(
            shard,
            include_microstructure=include_microstructure,
            verbose=verbose,
        )

        # Verify repair
        if result.healed:
            verified = verify_repair(shard)
            if not verified:
                logger.warning(
                    "repair for %s@%d reported healed but gap persists",
                    shard.symbol, shard.threshold_decimal_bps,
                )
                result.healed = False

        # Log to ClickHouse
        _log_repair(result)
        results.append(result)

    # Summary
    pass_duration = time.monotonic() - pass_t0
    healed = sum(1 for r in results if r.healed)
    failed = sum(1 for r in results if not r.healed)
    total_bars = sum(r.bars_written for r in results)
    logger.info(
        "kintsugi pass complete: %d healed, %d failed, %d bars written (%.1fs)",
        healed, failed, total_bars, pass_duration,
    )

    # Issue #117-119: Emit KINTSUGI_PASS_COMPLETE hook + quiet Telegram
    try:
        from rangebar.hooks import HookEvent, emit_hook
        emit_hook(
            HookEvent.KINTSUGI_PASS_COMPLETE, symbol="*",
            n_discovered=len(shards), n_healed=healed, n_failed=failed,
            total_bars=total_bars, pass_duration=pass_duration,
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.debug("failed to emit KINTSUGI_PASS_COMPLETE hook", exc_info=True)
    try:
        from rangebar.notify.telegram import _build_context_block, send_telegram
        pass_msg = (
            "<b>KINTSUGI PASS COMPLETE</b>\n"
            f"<b>Duration:</b> {pass_duration:.0f}s\n"
            f"<b>Shards discovered:</b> {len(shards)}\n"
            f"<b>Repaired:</b> {healed} | <b>Failed:</b> {failed}\n"
            f"<b>Bars written:</b> {total_bars:,}"
            + _build_context_block("kintsugi")
        )
        send_telegram(pass_msg, disable_notification=True)
    except (ImportError, OSError, RuntimeError):
        logger.debug("failed to send kintsugi pass Telegram", exc_info=True)

    # Notify on failures
    if failed > 0:
        _notify_kintsugi_failures(results, pass_duration=pass_duration)

    return results


def _is_rate_limited(shard: Shard) -> bool:
    """Check if shard has exceeded max attempts in the last 24h."""
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            result = cache.client.query(
                "SELECT count() FROM rangebar_cache.kintsugi_log "
                "WHERE symbol = {sym:String} "
                "  AND threshold_decimal_bps = {thr:UInt32} "
                "  AND gap_start_ms = {gs:Int64} "
                "  AND timestamp > now() - INTERVAL 24 HOUR",
                parameters={
                    "sym": shard.symbol,
                    "thr": shard.threshold_decimal_bps,
                    "gs": shard.gap_start_ms,
                },
            )
            count = result.result_rows[0][0] if result.result_rows else 0
    except (OSError, RuntimeError, ConnectionError):
        return False  # On error, allow repair attempt
    else:
        return count >= _MAX_ATTEMPTS_PER_SHARD


def _notify_kintsugi_failures(
    results: list[KintsugiResult],
    pass_duration: float = 0.0,
) -> None:
    """Send Telegram alert for failed repairs."""
    try:
        from rangebar.notify.telegram import _build_context_block, send_telegram
    except ImportError:
        return

    failed = [r for r in results if not r.healed]
    if not failed:
        return

    healed_count = sum(1 for r in results if r.healed)
    total_shards = len(results)

    lines = [
        f"<b>KINTSUGI</b>: {len(failed)} repair(s) failed\n",
        f"<b>Pass duration:</b> {pass_duration:.1f}s",
        f"<b>Total shards:</b> {total_shards} "
        f"({healed_count} healed, {len(failed)} failed)",
    ]
    for r in failed[:_MAX_FAILURES_IN_ALERT]:
        lines.append(
            f"  {r.shard.symbol}@{r.shard.threshold_decimal_bps} "
            f"({r.shard.priority}, {r.shard.gap_hours:.1f}h): "
            f"<code>{r.error or 'unknown'}</code>"
        )
    if len(failed) > _MAX_FAILURES_IN_ALERT:
        lines.append(f"  ... and {len(failed) - _MAX_FAILURES_IN_ALERT} more")
    lines.append(_build_context_block("kintsugi"))

    try:
        send_telegram("\n".join(lines), disable_notification=False)
    except (OSError, RuntimeError, ConnectionError):
        logger.exception("failed to send kintsugi failure alert")


def kintsugi_daemon(
    *,
    interval_clean_s: int = 1800,
    interval_active_s: int = 900,
    max_p2_jobs: int = _DEFAULT_MAX_P2_JOBS,
    symbol: str | None = None,
    include_microstructure: bool = True,
) -> None:
    """Long-running daemon with adaptive polling.

    Runs ``kintsugi_pass()`` repeatedly:
    - Every ``interval_clean_s`` (30 min) when cache is clean
    - Every ``interval_active_s`` (15 min) when shards are found

    Stop with Ctrl+C (SIGINT) or SIGTERM.
    """
    logger.info(
        "kintsugi daemon starting (clean=%ds, active=%ds, max_p2=%d)",
        interval_clean_s, interval_active_s, max_p2_jobs,
    )

    iteration = 0
    try:
        while True:
            iteration += 1
            logger.info("=== kintsugi iteration %d ===", iteration)

            results = kintsugi_pass(
                symbol=symbol,
                max_p2_jobs=max_p2_jobs,
                include_microstructure=include_microstructure,
            )

            # Issue #117: Ping systemd watchdog after each pass
            _sd_notify("WATCHDOG=1")

            has_activity = any(r.healed or r.error for r in results)
            sleep_s = interval_active_s if has_activity else interval_clean_s

            logger.info(
                "kintsugi sleeping %ds (%s)",
                sleep_s, "active" if has_activity else "clean",
            )
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("kintsugi daemon stopped after %d iterations", iteration)


# ---------------------------------------------------------------------------
# Dead-letter replay
# ---------------------------------------------------------------------------


def replay_dead_letters() -> int:
    """Replay any dead-letter Parquet files from sidecar flush failures.

    Returns number of bars successfully replayed.
    """
    from rangebar.sidecar import _DEAD_LETTER_DIR

    if not _DEAD_LETTER_DIR.exists():
        return 0

    import polars as pl

    from rangebar.clickhouse import RangeBarCache

    total_replayed = 0
    for path in sorted(_DEAD_LETTER_DIR.glob("*.parquet")):
        parts = path.stem.split("_")
        if len(parts) < _MIN_DEAD_LETTER_PARTS:
            logger.warning("skipping malformed dead-letter: %s", path)
            continue

        symbol = parts[0]
        threshold = int(parts[1])

        try:
            bars_df = pl.read_parquet(path)
            with RangeBarCache() as cache:
                written = cache.store_bars_batch(
                    symbol=symbol,
                    threshold_decimal_bps=threshold,
                    bars=bars_df,
                )
            total_replayed += written
            path.unlink()
            logger.info(
                "replayed dead-letter %s: %d bars", path.name, written,
            )
        except (OSError, RuntimeError, ConnectionError, ValueError) as e:
            logger.warning("failed to replay dead-letter %s: %s", path, e)
            break  # Stop on first failure (ClickHouse likely down)

    if total_replayed > 0:
        logger.info("dead-letter replay: %d bars total", total_replayed)

    return total_replayed


__all__ = [
    "KintsugiResult",
    "Shard",
    "discover_shards",
    "kintsugi_daemon",
    "kintsugi_pass",
    "repair_shard",
    "replay_dead_letters",
    "verify_repair",
]
