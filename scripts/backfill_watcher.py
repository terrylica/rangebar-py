#!/usr/bin/env python3
"""On-demand backfill watcher for flowsurface requests (Issue #102).

Polls `rangebar_cache.backfill_requests` for pending requests inserted by
flowsurface, processes them via `backfill_recent()`, and updates status.

Status lifecycle: pending -> running -> completed | failed

Features (Wave 3):
- Fan-out: threshold=0 resolves to ALL cached thresholds (not just lowest)
- Dedup-aware: GROUP BY (symbol, threshold) collapses duplicate requests
- Cooldown: skip recently-completed (symbol, threshold) pairs (default 300s)
- Gap tracking: gap_seconds written at _mark_running() time for UI feedback
- Exception capture: actual exception messages preserved in failed status

USAGE:
======
# Long-running daemon (30s poll interval)
uv run python scripts/backfill_watcher.py

# Poll once and exit (testing)
uv run python scripts/backfill_watcher.py --once

# Verbose logging
uv run python scripts/backfill_watcher.py --verbose

# Via mise
mise run cache:watcher
mise run cache:watcher-once

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/102
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
DEFAULT_THRESHOLD_DBPS = 250

# In-memory cooldown tracker: (symbol, threshold) -> monotonic timestamp of last completion
_cooldown_tracker: dict[tuple[str, int], float] = {}

_COOLDOWN_SECONDS = int(
    os.environ.get("RANGEBAR_BACKFILL_COOLDOWN_SECONDS", "300")
)


def _is_on_cooldown(symbol: str, threshold: int) -> bool:
    """Check if (symbol, threshold) was completed within the cooldown window."""
    key = (symbol, threshold)
    last_completed = _cooldown_tracker.get(key)
    if last_completed is None:
        return False
    return (time.monotonic() - last_completed) < _COOLDOWN_SECONDS


def _record_cooldown(symbol: str, threshold: int) -> None:
    """Record completion time for cooldown tracking."""
    _cooldown_tracker[(symbol, threshold)] = time.monotonic()


def _fetch_pending_requests(cache: object) -> list[dict]:
    """Query backfill_requests for pending rows, grouped by symbol only.

    Issue #103/#105: Process all thresholds for a symbol together to prevent
    fast-completing thresholds (BPR250) from monopolizing the queue while
    slower thresholds (BPR50, BPR75, BPR100) wait indefinitely.

    Returns one logical request per symbol with all thresholds and
    request_ids collected for batch status updates.
    """
    result = cache.client.query(
        "SELECT symbol, "
        "groupArray(DISTINCT threshold_decimal_bps) AS thresholds, "
        "groupArray(request_id) AS all_ids, "
        "min(requested_at) AS earliest "
        "FROM rangebar_cache.backfill_requests FINAL "
        "WHERE status = 'pending' "
        "GROUP BY symbol "
        "ORDER BY earliest ASC"
    )
    return [
        {
            "symbol": row[0],
            "thresholds": [int(t) for t in row[1]],
            "all_ids": [str(rid) for rid in row[2]],
            "earliest": row[3],
        }
        for row in result.result_rows
    ]


def _mark_running(
    cache: object, request_ids: list[str], *, gap_seconds: float = 0.0
) -> None:
    """Transition request(s) from pending -> running with gap_seconds."""
    for request_id in request_ids:
        cache.client.command(
            "ALTER TABLE rangebar_cache.backfill_requests UPDATE "
            "status = 'running', started_at = now64(3), "
            "gap_seconds = {gap_seconds:Float64} "
            "WHERE request_id = {request_id:String}",
            parameters={
                "request_id": request_id,
                "gap_seconds": gap_seconds,
            },
        )


def _mark_completed(
    cache: object,
    request_ids: list[str],
    bars_written: int,
    gap_seconds: float,
) -> None:
    """Transition request(s) from running -> completed."""
    for request_id in request_ids:
        cache.client.command(
            "ALTER TABLE rangebar_cache.backfill_requests UPDATE "
            "status = 'completed', completed_at = now64(3), "
            "bars_written = {bars_written:UInt32}, gap_seconds = {gap_seconds:Float64} "
            "WHERE request_id = {request_id:String}",
            parameters={
                "request_id": request_id,
                "bars_written": bars_written,
                "gap_seconds": gap_seconds,
            },
        )


def _mark_failed(cache: object, request_ids: list[str], error: str) -> None:
    """Transition request(s) from running -> failed."""
    error_truncated = error[:500] if len(error) > 500 else error
    for request_id in request_ids:
        cache.client.command(
            "ALTER TABLE rangebar_cache.backfill_requests UPDATE "
            "status = 'failed', completed_at = now64(3), "
            "error = {error:String} "
            "WHERE request_id = {request_id:String}",
            parameters={
                "request_id": request_id,
                "error": error_truncated,
            },
        )


def _update_progress(
    cache: object,
    request_ids: list[str],
    bars_written: int,
    completed_thresholds: int,
    total_thresholds: int,
) -> None:
    """Issue #104: Update in-flight progress for flowsurface to poll.

    Updates bars_written while request is still 'running', so flowsurface
    can display incremental progress without waiting for full completion.
    """
    for request_id in request_ids:
        cache.client.command(
            "ALTER TABLE rangebar_cache.backfill_requests UPDATE "
            "bars_written = {bars_written:UInt32} "
            "WHERE request_id = {request_id:String} AND status = 'running'",
            parameters={
                "request_id": request_id,
                "bars_written": bars_written,
            },
        )
    logger.debug(
        "Progress: %d/%d thresholds, %d bars written",
        completed_thresholds,
        total_thresholds,
        bars_written,
    )


def _resolve_thresholds(cache: object, symbol: str, threshold: int) -> list[int]:
    """Resolve threshold_decimal_bps, handling the DEFAULT 0 case.

    When threshold=0 (flowsurface default): return ALL cached thresholds for
    the symbol. This prevents starvation where only the lowest threshold gets
    backfilled.

    When threshold > 0: return [threshold] as a single-element list.
    """
    if threshold > 0:
        return [threshold]

    result = cache.client.query(
        "SELECT DISTINCT threshold_decimal_bps "
        "FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = {symbol:String} "
        "ORDER BY threshold_decimal_bps ASC",
        parameters={"symbol": symbol},
    )
    if result.result_rows:
        thresholds = [int(row[0]) for row in result.result_rows]
        logger.info(
            "%s: threshold=0 resolved to %s dbps (all cached)",
            symbol,
            thresholds,
        )
        return thresholds

    logger.info(
        "%s: threshold=0, no cache — using %d dbps", symbol, DEFAULT_THRESHOLD_DBPS
    )
    return [DEFAULT_THRESHOLD_DBPS]


def _compute_gap_seconds(cache: object, symbol: str, threshold: int) -> float:
    """Compute gap_seconds between latest cached bar and now."""
    from datetime import UTC, datetime

    latest_ts = cache.get_latest_bar_timestamp(symbol, threshold)
    if latest_ts is None:
        return 0.0
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return (now_ms - latest_ts) / 1000.0


def _process_request_group(
    cache: object, symbol: str, thresholds: list[int], all_ids: list[str],
    *, verbose: bool = False
) -> bool:
    """Validate and process all thresholds for a symbol together.

    Issue #103/#105: Process all thresholds for a symbol in one cycle to ensure
    fair scheduling. Fast thresholds (BPR250) cannot monopolize the queue by
    re-entering before slow thresholds (BPR50/75/100) complete.

    A request group contains one symbol with potentially multiple thresholds
    and multiple request_ids that all get the same status updates.

    Returns True if backfill succeeded, False on failure.
    """
    from rangebar.recency import backfill_recent
    from rangebar.symbol_registry import get_symbol_entries

    display_id = all_ids[0][:8] if all_ids else "unknown"

    logger.info(
        "Processing request group %s (%d request(s)): %s @ %d threshold(s)",
        display_id,
        len(all_ids),
        symbol,
        len(thresholds),
    )

    # Validate symbol is registered and enabled (fail fast, before marking running)
    entries = get_symbol_entries()
    if symbol not in entries or not entries[symbol].enabled:
        error_msg = f"symbol '{symbol}' not registered or not enabled"
        logger.warning("Request %s: %s", display_id, error_msg)
        _mark_failed(cache, all_ids, error_msg)
        return False

    # If any threshold is 0 (fan-out case), resolve to all cached thresholds
    # Issue #103: Only resolve 0 if explicitly in the list; otherwise use as-is
    if 0 in thresholds:
        resolved = _resolve_thresholds(cache, symbol, 0)
        thresholds = resolved
        logger.info(
            "Request %s: %s — threshold=0 resolved to %s dbps",
            display_id,
            symbol,
            thresholds,
        )

    # Filter by cooldown
    actionable = [t for t in thresholds if not _is_on_cooldown(symbol, t)]
    if not actionable:
        logger.info(
            "Request %s: %s — all thresholds %s on cooldown, skipping",
            display_id,
            symbol,
            thresholds,
        )
        _mark_completed(cache, all_ids, bars_written=0, gap_seconds=0.0)
        return True

    skipped = [t for t in thresholds if t not in actionable]
    if skipped:
        logger.info(
            "Request %s: %s — skipping thresholds %s (cooldown), processing %s",
            display_id,
            symbol,
            skipped,
            actionable,
        )

    # Compute gap_seconds for the first actionable threshold (for UI display)
    gap_seconds = _compute_gap_seconds(cache, symbol, actionable[0])

    # Mark all request_ids as running with gap_seconds
    _mark_running(cache, all_ids, gap_seconds=gap_seconds)

    # Run backfill for each actionable threshold
    total_bars = 0
    max_gap = gap_seconds
    any_failed = False
    last_error = ""

    for i, t in enumerate(actionable):
        try:
            result = backfill_recent(
                symbol,
                t,
                include_microstructure=True,
                verbose=verbose,
            )
        except (RuntimeError, OSError, ConnectionError, ValueError) as e:
            logger.exception(
                "Request %s failed: %s @ %d",
                display_id,
                symbol,
                t,
            )
            last_error = str(e)[:500]
            any_failed = True
            continue

        if result.error:
            logger.warning(
                "Request %s backfill error: %s @ %d: %s",
                display_id,
                symbol,
                t,
                result.error,
            )
            last_error = result.error
            any_failed = True
            continue

        total_bars += result.bars_written
        max_gap = max(max_gap, result.gap_seconds)
        _record_cooldown(symbol, t)

        logger.info(
            "Request %s: %s @ %d — %d bars, gap %.1f min",
            display_id,
            symbol,
            t,
            result.bars_written,
            result.gap_seconds / 60,
        )

        # Issue #104: Incremental progress update after each threshold completes
        _update_progress(cache, all_ids, total_bars, i + 1, len(actionable))

    if any_failed and total_bars == 0:
        _mark_failed(cache, all_ids, last_error)
        return False

    logger.info(
        "Request group %s done: %s — %d bars total across %d threshold(s)",
        display_id,
        symbol,
        total_bars,
        len(actionable),
    )
    _mark_completed(cache, all_ids, total_bars, max_gap)
    return True


def poll_once(*, verbose: bool = False) -> int:
    """Poll for pending requests and process all of them.

    Flow:
    1. Fetch pending requests (GROUP BY symbol only — not per-threshold)
    2. For each symbol:
       a. Get all pending thresholds for the symbol
       b. Filter by cooldown (skip recently completed thresholds)
       c. Validate symbol is registered & enabled
       d. Mark all request_ids as running (with gap_seconds)
       e. For each actionable threshold: call backfill_recent()
       f. Record cooldown on success for each threshold
       g. Mark all request_ids as completed/failed with aggregate bars_written

    Issue #103/#105: By grouping by symbol only, all thresholds (BPR50, BPR75,
    BPR100, BPR250) process fairly without fast thresholds monopolizing the queue.

    Returns the number of request groups processed.
    """
    from rangebar.clickhouse import RangeBarCache

    with RangeBarCache() as cache:
        pending = _fetch_pending_requests(cache)

        if not pending:
            logger.debug("No pending backfill requests")
            return 0

        logger.info("Found %d pending symbol group(s)", len(pending))

        processed = 0
        for request in pending:
            symbol = request["symbol"]
            thresholds = request["thresholds"]
            all_ids = request["all_ids"]
            _process_request_group(cache, symbol, thresholds, all_ids, verbose=verbose)
            processed += 1

    return processed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill request watcher for flowsurface (Issue #102)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.once:
        try:
            n = poll_once(verbose=args.verbose)
            logger.info("Done: %d request(s) processed", n)
            return 0
        except (OSError, ConnectionError, RuntimeError):
            logger.exception("Watcher error")
            return 1

    logger.info(
        "Starting backfill watcher (poll interval: %ds)", POLL_INTERVAL_SECONDS
    )

    try:
        while True:
            try:
                poll_once(verbose=args.verbose)
            except (OSError, ConnectionError, RuntimeError):
                logger.exception("Poll error")

            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user")

    return 0


if __name__ == "__main__":
    sys.exit(main())
