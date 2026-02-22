# Issue #46: Modularization - Extract cache helpers from range_bars.py
"""ClickHouse cache read/write helpers for get_range_bars().

Provides standalone functions extracted from get_range_bars() to reduce
module size and improve testability:
- try_cache_read(): Fast-path cache lookup for precomputed bars
- try_cache_write(): Non-fatal cache write after bar computation
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def try_cache_read(
    symbol: str,
    threshold_decimal_bps: int,
    start_ts: int,
    end_ts: int,
    include_microstructure: bool,
    ouroboros: str,
    trace_id: str,
) -> pd.DataFrame | None:
    """Attempt to read bars from ClickHouse cache.

    Fast-path for precomputed bars. Returns cached DataFrame on hit,
    None on miss, import error, or connection error. Includes staleness
    detection for microstructure-enabled queries.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    start_ts : int
        Start timestamp in milliseconds.
    end_ts : int
        End timestamp in milliseconds.
    include_microstructure : bool
        Whether microstructure columns were requested.
    ouroboros : str
        Ouroboros mode ("year", "month", "week").
    trace_id : str
        Trace ID for pipeline correlation logging.

    Returns
    -------
    pd.DataFrame or None
        Cached bars DataFrame on hit, None on miss or error.
    """
    try:
        from rangebar.clickhouse import RangeBarCache
        from rangebar.logging import get_logger
        from rangebar.validation.cache_staleness import detect_staleness

        with RangeBarCache() as cache:
            # Ouroboros mode filter ensures cache isolation
            _t0 = time.perf_counter()
            cached_bars = cache.get_bars_by_timestamp_range(
                symbol=symbol,
                threshold_decimal_bps=threshold_decimal_bps,
                start_ts=start_ts,
                end_ts=end_ts,
                include_microstructure=include_microstructure,
                ouroboros_mode=ouroboros,
            )
            _query_ms = (time.perf_counter() - _t0) * 1000
            _hit = cached_bars is not None and len(cached_bars) > 0
            get_logger().bind(
                component="cache_query",
                trace_id=trace_id,
                symbol=symbol,
                threshold_dbps=threshold_decimal_bps,
                hit=_hit,
                bar_count=len(cached_bars) if _hit else 0,
                query_ms=round(_query_ms, 2),
            ).info(
                f"cache_query: {symbol}@{threshold_decimal_bps} "
                f"{'HIT' if _hit else 'MISS'} ({_query_ms:.1f}ms)"
            )
            if cached_bars is not None and len(cached_bars) > 0:
                # Tier 0 validation: Content-based staleness detection (Issue #39)
                # This catches stale cached data from pre-v7.0 (e.g., VWAP=0)
                if include_microstructure:
                    staleness = detect_staleness(
                        cached_bars, require_microstructure=True
                    )
                    if staleness.is_stale:
                        logger.warning(
                            "Stale cache data detected for %s: %s. "
                            "Falling through to recompute.",
                            symbol,
                            staleness.reason,
                        )
                        # Fall through to tick processing path
                        return None
                    # Fast path: return validated bars from ClickHouse (~50ms)
                    return cached_bars
                # Fast path: return precomputed bars from ClickHouse (~50ms)
                return cached_bars
    except ImportError:
        # ClickHouse not available, fall through to tick processing
        pass
    except ConnectionError:
        # ClickHouse connection failed, fall through to tick processing
        pass

    return None


def try_cache_write(
    bars_df: pd.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
    ouroboros: str,
) -> None:
    """Write bars to ClickHouse cache. Non-fatal on error.

    Cache write is non-blocking: failures don't affect the caller.
    The computation succeeded, so bars are returned even if caching fails.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Range bar DataFrame to cache.
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    ouroboros : str
        Ouroboros mode ("year", "month", "week").
    """
    if bars_df is None or bars_df.empty:
        return

    try:
        from rangebar.clickhouse import RangeBarCache
        from rangebar.exceptions import CacheError

        with RangeBarCache() as cache:
            # Phase 3+4: Route through store_bars_batch (Arrow path, 2-3 copies)
            # instead of store_bars_bulk (pandas path, 5-7 copies)
            import polars as pl

            bars_pl = pl.from_pandas(bars_df.reset_index())
            written = cache.store_bars_batch(
                symbol=symbol,
                threshold_decimal_bps=threshold_decimal_bps,
                bars=bars_pl,
                version="",  # Version tracked elsewhere
                ouroboros_mode=ouroboros,
            )
            logger.info(
                "Cached %d bars for %s @ %d dbps",
                written,
                symbol,
                threshold_decimal_bps,
            )
    except ImportError:
        # ClickHouse not available - skip caching
        pass
    except ConnectionError:
        # ClickHouse connection failed - skip caching
        pass
    except (CacheError, OSError, RuntimeError) as e:
        # Log but don't fail - cache is optimization layer
        # CacheError: All cache-specific errors
        # OSError: Network/disk errors
        # RuntimeError: ClickHouse driver errors
        logger.warning("Cache write failed (non-fatal): %s", e)


def fatal_cache_write(
    bars_df: pd.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
    ouroboros: str,
) -> int:
    """Write bars to ClickHouse cache. Raises on failure.

    Unlike try_cache_write(), this function is for populate_cache_resumable()
    where ClickHouse IS the destination, not an optional cache layer.
    Failures must be visible to the caller so that:
    - bars_written reflects reality (actual persisted count, not computed count)
    - checkpoints are only saved for days that succeeded
    - force_refresh + write failure does not silently lose data

    Parameters
    ----------
    bars_df : pd.DataFrame
        Range bar DataFrame to write.
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    ouroboros : str
        Ouroboros mode ("year", "month", "week").

    Returns
    -------
    int
        Number of rows actually written to ClickHouse.

    Raises
    ------
    CacheWriteError
        If the write fails for any reason (connection, schema, disk, etc.).
    """
    from rangebar.exceptions import CacheWriteError

    if bars_df is None or bars_df.empty:
        return 0

    try:
        import polars as pl

        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            bars_pl = pl.from_pandas(bars_df.reset_index())
            written = cache.store_bars_batch(
                symbol=symbol,
                threshold_decimal_bps=threshold_decimal_bps,
                bars=bars_pl,
                version="",
                ouroboros_mode=ouroboros,
            )
            logger.info(
                "Persisted %d bars for %s @ %d dbps",
                written,
                symbol,
                threshold_decimal_bps,
            )
            return written
    except ImportError as e:
        msg = f"ClickHouse not available: {e}"
        raise CacheWriteError(
            msg,
            symbol=symbol,
            operation="fatal_cache_write",
        ) from e
    except ConnectionError as e:
        msg = f"ClickHouse connection failed: {e}"
        raise CacheWriteError(
            msg,
            symbol=symbol,
            operation="fatal_cache_write",
        ) from e
    except (OSError, RuntimeError) as e:
        msg = f"ClickHouse write failed: {e}"
        raise CacheWriteError(
            msg,
            symbol=symbol,
            operation="fatal_cache_write",
        ) from e
