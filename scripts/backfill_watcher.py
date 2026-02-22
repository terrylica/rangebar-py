#!/usr/bin/env python3
"""On-demand backfill watcher for flowsurface requests (Issue #102).

Polls `rangebar_cache.backfill_requests` for pending requests inserted by
flowsurface, processes them via `backfill_recent()`, and updates status.

Status lifecycle: pending -> running -> completed | failed

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


def _fetch_pending_requests(cache: object) -> list[dict]:
    """Query backfill_requests for all pending rows, oldest first."""
    result = cache.client.query(
        "SELECT request_id, symbol, threshold_decimal_bps "
        "FROM rangebar_cache.backfill_requests FINAL "
        "WHERE status = 'pending' "
        "ORDER BY requested_at ASC"
    )
    return [
        {
            "request_id": str(row[0]),
            "symbol": row[1],
            "threshold_decimal_bps": int(row[2]),
        }
        for row in result.result_rows
    ]


def _mark_running(cache: object, request_id: str) -> None:
    """Transition a request from pending -> running."""
    cache.client.command(
        "ALTER TABLE rangebar_cache.backfill_requests UPDATE "
        "status = 'running', started_at = now64(3) "
        "WHERE request_id = {request_id:String}",
        parameters={"request_id": request_id},
    )


def _mark_completed(
    cache: object,
    request_id: str,
    bars_written: int,
    gap_seconds: float,
) -> None:
    """Transition a request from running -> completed."""
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


def _mark_failed(cache: object, request_id: str, error: str) -> None:
    """Transition a request from running -> failed."""
    error_truncated = error[:500] if len(error) > 500 else error
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


def _resolve_threshold(cache: object, symbol: str, threshold: int) -> int:
    """Resolve threshold_decimal_bps, handling the DEFAULT 0 case.

    flowsurface may insert requests with threshold=0 (ClickHouse DEFAULT).
    Query the first cached threshold for the symbol, fall back to 250 dbps.
    """
    if threshold > 0:
        return threshold

    result = cache.client.query(
        "SELECT DISTINCT threshold_decimal_bps "
        "FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = {symbol:String} "
        "ORDER BY threshold_decimal_bps ASC LIMIT 1",
        parameters={"symbol": symbol},
    )
    if result.result_rows:
        resolved = int(result.result_rows[0][0])
        logger.info(
            "%s: threshold=0 resolved to %d dbps (from cache)", symbol, resolved
        )
        return resolved

    logger.info("%s: threshold=0, no cache — using %d dbps", symbol, DEFAULT_THRESHOLD_DBPS)
    return DEFAULT_THRESHOLD_DBPS


def _process_request(cache: object, request: dict, *, verbose: bool = False) -> bool:
    """Validate and process a single backfill request.

    Uses the shared cache connection for status mutations to avoid
    opening multiple SSH tunnels per request.

    Returns True if backfill succeeded, False on failure.
    """
    from rangebar.recency import backfill_recent
    from rangebar.symbol_registry import get_symbol_entries

    request_id = request["request_id"]
    symbol = request["symbol"]
    threshold = request["threshold_decimal_bps"]

    logger.info(
        "Processing request %s: %s @ %d dbps",
        request_id[:8],
        symbol,
        threshold,
    )

    # Validate symbol is registered and enabled (fail fast, before marking running)
    entries = get_symbol_entries()
    if symbol not in entries or not entries[symbol].enabled:
        error_msg = f"symbol '{symbol}' not registered or not enabled"
        logger.warning("Request %s: %s", request_id[:8], error_msg)
        _mark_failed(cache, request_id, error_msg)
        return False

    # Resolve threshold (handle DEFAULT 0 from flowsurface)
    threshold = _resolve_threshold(cache, symbol, threshold)

    # Mark as running
    _mark_running(cache, request_id)

    # Run backfill via existing recency.py API
    try:
        result = backfill_recent(
            symbol,
            threshold,
            include_microstructure=True,
            verbose=verbose,
        )
    except (RuntimeError, OSError, ConnectionError, ValueError):
        logger.exception(
            "Request %s failed: %s @ %d",
            request_id[:8],
            symbol,
            threshold,
        )
        _mark_failed(cache, request_id, "Backfill failed with exception")
        return False

    if result.error:
        logger.warning(
            "Request %s backfill error: %s @ %d: %s",
            request_id[:8],
            symbol,
            threshold,
            result.error,
        )
        _mark_failed(cache, request_id, result.error)
        return False

    logger.info(
        "Request %s done: %s @ %d — %d bars, gap %.1f min",
        request_id[:8],
        symbol,
        threshold,
        result.bars_written,
        result.gap_seconds / 60,
    )
    _mark_completed(cache, request_id, result.bars_written, result.gap_seconds)
    return True


def poll_once(*, verbose: bool = False) -> int:
    """Poll for pending requests and process all of them.

    Returns the number of requests processed.
    """
    from rangebar.clickhouse import RangeBarCache

    with RangeBarCache() as cache:
        pending = _fetch_pending_requests(cache)

        if not pending:
            logger.debug("No pending backfill requests")
            return 0

        logger.info("Found %d pending request(s)", len(pending))

        processed = 0
        for request in pending:
            _process_request(cache, request, verbose=verbose)
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
