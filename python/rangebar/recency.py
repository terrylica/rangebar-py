"""Layer 2 recency backfill for range bar cache (Issue #92).

Bridges the gap between the latest Binance Vision archive data and the current
time by fetching recent trades via REST API and computing range bars.

Architecture:
    Binance Vision (Tier 1) → [gap] → REST API (Layer 2) → ClickHouse

The adaptive polling loop self-tunes its frequency based on observed gap size,
configured via RANGEBAR_RECENCY_* env vars in .mise.toml (SSoT).

Usage
-----
Single-shot backfill:

>>> from rangebar.recency import backfill_recent
>>> result = backfill_recent("BTCUSDT", threshold_decimal_bps=250)
>>> print(f"Filled {result.bars_written} bars, gap was {result.gap_seconds}s")

Backfill all cached symbols:

>>> from rangebar.recency import backfill_all_recent
>>> results = backfill_all_recent()

Adaptive loop (long-running sidecar):

>>> from rangebar.recency import run_adaptive_loop
>>> run_adaptive_loop()  # Runs until interrupted
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Adaptive Thresholds (from .mise.toml [env] SSoT)
# =============================================================================

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _fresh_threshold_s() -> int:
    """Gap below which data is considered fresh (sleep longer)."""
    return _env_int("RANGEBAR_RECENCY_FRESH_THRESHOLD_MIN", 30) * 60


def _stale_threshold_s() -> int:
    """Gap above which data is stale (poll more aggressively)."""
    return _env_int("RANGEBAR_RECENCY_STALE_THRESHOLD_MIN", 120) * 60


def _critical_threshold_s() -> int:
    """Gap above which data is critically stale (fast backfill)."""
    return _env_int("RANGEBAR_RECENCY_CRITICAL_THRESHOLD_MIN", 1440) * 60


def _adaptive_sleep(gap_seconds: float) -> float:
    """Compute sleep duration based on observed gap size.

    Returns sleep time in seconds. Larger gaps = shorter sleep = faster polling.
    """
    fresh = _fresh_threshold_s()
    stale = _stale_threshold_s()
    critical = _critical_threshold_s()

    if gap_seconds < fresh:
        return 15 * 60  # 15 min — data is fresh
    if gap_seconds < stale:
        return 5 * 60  # 5 min — getting stale
    if gap_seconds < critical:
        return 2 * 60  # 2 min — large gap
    return 30  # 30 sec — major gap


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class BackfillResult:
    """Result of a single backfill operation."""

    symbol: str
    threshold_decimal_bps: int
    bars_written: int
    gap_seconds: float
    latest_ts_before: int | None
    latest_ts_after: int | None
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class LoopState:
    """State tracked across adaptive loop iterations."""

    iteration: int = 0
    total_bars_written: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    last_gaps: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Core Backfill Logic
# =============================================================================

def _get_cached_symbol_threshold_pairs() -> list[tuple[str, int]]:
    """Query ClickHouse for all distinct (symbol, threshold) pairs in cache.

    Returns
    -------
    list[tuple[str, int]]
        List of (symbol, threshold_decimal_bps) pairs.
    """
    from rangebar.clickhouse import RangeBarCache

    with RangeBarCache() as cache:
        result = cache.client.query(
            "SELECT DISTINCT symbol, threshold_decimal_bps "
            "FROM rangebar_cache.range_bars FINAL "
            "ORDER BY symbol, threshold_decimal_bps"
        )
        return [(row[0], row[1]) for row in result.result_rows]


def backfill_recent(
    symbol: str,
    threshold_decimal_bps: int = 250,
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> BackfillResult:
    """Backfill the gap between latest cached bar and now via REST API.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    include_microstructure : bool
        Include all 26 microstructure features (default True).
    verbose : bool
        Enable detailed logging.

    Returns
    -------
    BackfillResult
        Summary of the backfill operation.
    """
    from rangebar.clickhouse import RangeBarCache
    from rangebar.symbol_registry import validate_symbol_registered
    from rangebar.threshold import resolve_and_validate_threshold

    validate_symbol_registered(symbol, operation="backfill_recent")
    threshold_decimal_bps = resolve_and_validate_threshold(
        symbol, threshold_decimal_bps
    )

    t0 = time.monotonic()
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # Step 1: Query latest bar timestamp
    with RangeBarCache() as cache:
        latest_ts = cache.get_latest_bar_timestamp(
            symbol, threshold_decimal_bps
        )

    if latest_ts is None:
        logger.warning(
            "No cached bars for %s @ %d dbps — cannot backfill "
            "(run populate_cache_resumable first)",
            symbol,
            threshold_decimal_bps,
        )
        return BackfillResult(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            bars_written=0,
            gap_seconds=0,
            latest_ts_before=None,
            latest_ts_after=None,
            duration_seconds=time.monotonic() - t0,
            error="no_cached_bars",
        )

    gap_ms = now_ms - latest_ts
    gap_seconds = gap_ms / 1000.0

    if gap_seconds < 60:
        # Less than 1 minute gap — nothing meaningful to backfill
        logger.debug(
            "%s @ %d dbps: gap %.0fs — fresh, skipping",
            symbol,
            threshold_decimal_bps,
            gap_seconds,
        )
        return BackfillResult(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            bars_written=0,
            gap_seconds=gap_seconds,
            latest_ts_before=latest_ts,
            latest_ts_after=latest_ts,
            duration_seconds=time.monotonic() - t0,
        )

    logger.info(
        "%s @ %d dbps: gap %.1f min — backfilling from REST API",
        symbol,
        threshold_decimal_bps,
        gap_seconds / 60,
    )

    # Step 2: Fetch recent trades via REST API
    try:
        bars_written = _fetch_and_process_gap(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            start_ms=latest_ts + 1,
            end_ms=now_ms,
            include_microstructure=include_microstructure,
            verbose=verbose,
        )
    except (RuntimeError, OSError, ConnectionError) as e:
        logger.warning(
            "%s @ %d dbps: backfill failed: %s",
            symbol,
            threshold_decimal_bps,
            e,
        )
        return BackfillResult(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            bars_written=0,
            gap_seconds=gap_seconds,
            latest_ts_before=latest_ts,
            latest_ts_after=latest_ts,
            duration_seconds=time.monotonic() - t0,
            error=str(e),
        )

    # Step 3: Query new latest timestamp
    with RangeBarCache() as cache:
        new_latest_ts = cache.get_latest_bar_timestamp(
            symbol, threshold_decimal_bps
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "%s @ %d dbps: backfilled %d bars in %.1fs (gap was %.1f min)",
        symbol,
        threshold_decimal_bps,
        bars_written,
        elapsed,
        gap_seconds / 60,
    )

    return BackfillResult(
        symbol=symbol,
        threshold_decimal_bps=threshold_decimal_bps,
        bars_written=bars_written,
        gap_seconds=gap_seconds,
        latest_ts_before=latest_ts,
        latest_ts_after=new_latest_ts,
        duration_seconds=elapsed,
    )


def _fetch_aggtrades_rest(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch aggregated trades from Binance REST API via Rust (native performance).

    Delegates to Rust `fetch_aggtrades_rest()` which uses reqwest to paginate
    through /api/v3/aggTrades (max 1000 per request) with async I/O.

    Returns list of trade dicts compatible with RangeBarProcessor.process_trades().
    """
    from rangebar._core import fetch_aggtrades_rest

    try:
        return fetch_aggtrades_rest(symbol, start_ms, end_ms)
    except Exception as e:
        msg = (
            f"REST API fetch failed for {symbol} "
            f"[{start_ms}-{end_ms}]: {e}"
        )
        raise RuntimeError(msg) from e


def _fetch_and_process_gap(
    *,
    symbol: str,
    threshold_decimal_bps: int,
    start_ms: int,
    end_ms: int,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> int:
    """Fetch trades from Binance REST API and process into range bars.

    Paginates through /api/v3/aggTrades, processes through Rust
    RangeBarProcessor, and writes completed bars to ClickHouse.

    Returns number of bars written.
    """
    import polars as pl

    from rangebar.clickhouse import RangeBarCache
    from rangebar.orchestration.helpers import (
        _create_processor,
    )

    # Fetch trades from Binance REST API (paginated)
    trades = _fetch_aggtrades_rest(symbol, start_ms, end_ms)

    if not trades:
        logger.debug(
            "%s: no trades in gap %d-%d",
            symbol,
            start_ms,
            end_ms,
        )
        return 0

    logger.debug(
        "%s: fetched %d trades from REST API (%d ms range)",
        symbol,
        len(trades),
        end_ms - start_ms,
    )

    # Process trades through Rust processor
    processor = _create_processor(
        threshold_decimal_bps,
        include_microstructure,
        symbol=symbol,
    )
    bars = processor.process_trades(trades)

    if not bars:
        logger.debug("%s: no completed bars from %d trades", symbol, len(trades))
        return 0

    # Convert bars to Polars DataFrame for batch storage
    bars_df = pl.DataFrame(bars)

    # Write to ClickHouse via store_bars_batch (with dedup token)
    with RangeBarCache() as cache:
        written = cache.store_bars_batch(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            bars=bars_df,
        )

    return written


# =============================================================================
# Multi-Symbol Backfill
# =============================================================================

def backfill_all_recent(
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> list[BackfillResult]:
    """Backfill all cached symbol x threshold pairs.

    Queries ClickHouse for all distinct (symbol, threshold) pairs and runs
    backfill_recent for each.

    Returns
    -------
    list[BackfillResult]
        Results for each pair, including any errors.
    """
    pairs = _get_cached_symbol_threshold_pairs()
    logger.info("Backfilling %d symbol x threshold pairs", len(pairs))

    results = []
    for symbol, threshold in pairs:
        try:
            result = backfill_recent(
                symbol,
                threshold,
                include_microstructure=include_microstructure,
                verbose=verbose,
            )
            results.append(result)
        except (ValueError, RuntimeError, OSError, ConnectionError) as e:
            logger.warning(
                "Backfill failed for %s @ %d: %s",
                symbol,
                threshold,
                e,
            )
            results.append(
                BackfillResult(
                    symbol=symbol,
                    threshold_decimal_bps=threshold,
                    bars_written=0,
                    gap_seconds=0,
                    latest_ts_before=None,
                    latest_ts_after=None,
                    error=str(e),
                )
            )

    total_bars = sum(r.bars_written for r in results)
    errors = sum(1 for r in results if r.error)
    logger.info(
        "Backfill complete: %d pairs, %d bars written, %d errors",
        len(results),
        total_bars,
        errors,
    )
    return results


# =============================================================================
# Adaptive Loop
# =============================================================================

def run_adaptive_loop(
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> None:
    """Run adaptive recency backfill loop until interrupted.

    Self-tunes polling frequency based on observed gap sizes. Designed to run
    as a long-lived sidecar process (systemd service or tmux session).

    The loop:
    1. Queries all cached symbol x threshold pairs
    2. Backfills each pair
    3. Computes max gap across all pairs
    4. Sleeps for adaptive duration based on max gap
    5. Repeats

    Stop with Ctrl+C (SIGINT).
    """
    state = LoopState()
    logger.info("Starting adaptive recency backfill loop")

    try:
        while True:
            state.iteration += 1
            logger.info("=== Iteration %d ===", state.iteration)

            results = backfill_all_recent(
                include_microstructure=include_microstructure,
                verbose=verbose,
            )

            # Track gaps per symbol for adaptive sleep
            max_gap = 0.0
            for r in results:
                key = f"{r.symbol}@{r.threshold_decimal_bps}"
                state.last_gaps[key] = r.gap_seconds
                state.total_bars_written += r.bars_written
                max_gap = max(max_gap, r.gap_seconds)

            sleep_seconds = _adaptive_sleep(max_gap)
            logger.info(
                "Max gap: %.1f min | Sleep: %.0fs | Total bars: %d",
                max_gap / 60,
                sleep_seconds,
                state.total_bars_written,
            )

            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logger.info(
            "Loop stopped after %d iterations, %d total bars written",
            state.iteration,
            state.total_bars_written,
        )


__all__ = [
    "BackfillResult",
    "LoopState",
    "backfill_all_recent",
    "backfill_recent",
    "run_adaptive_loop",
]
