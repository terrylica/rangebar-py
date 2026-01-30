# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M4 - Extract get_range_bars from __init__.py
"""Date-bounded range bar generation.

Provides get_range_bars() - the single entry point for all range bar generation
with automatic data fetching, caching, and ouroboros boundary handling.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from rangebar.constants import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
)
from rangebar.processors.core import RangeBarProcessor
from rangebar.validation.cache_staleness import detect_staleness

from .helpers import (
    _fetch_binance,
    _fetch_exness,
    _process_binance_trades,
    _process_exness_ticks,
    _stream_range_bars_binance,
)

if TYPE_CHECKING:
    import polars as pl


def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    # Ouroboros: Cyclical reset boundaries (v11.0+)
    ouroboros: Literal["year", "month", "week"] = "year",
    include_orphaned_bars: bool = False,
    # Streaming options (v8.0+)
    materialize: bool = True,
    batch_size: int = 10_000,
    # Data source configuration
    source: str = "binance",
    market: str = "spot",
    # Exness-specific options
    validation: str = "strict",
    # Processing options
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,  # Issue #8: Exchange session flags
    # Timestamp gating (Issue #36)
    prevent_same_timestamp_close: bool = True,
    # Data integrity (Issue #43)
    verify_checksum: bool = True,
    # Caching options
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    cache_dir: str | None = None,
    # Memory guards (Issue #49)
    max_memory_mb: int | None = None,
) -> pd.DataFrame | Iterator[pl.DataFrame]:
    """Get range bars for a symbol with automatic data fetching and caching.

    This is the single entry point for all range bar generation. It supports
    multiple data sources (Binance crypto, Exness forex), all market types,
    and exposes the full configurability of the underlying Rust engine.

    Parameters
    ----------
    symbol : str
        Trading symbol (uppercase).
        - Binance: "BTCUSDT", "ETHUSDT", etc.
        - Exness: "EURUSD", "GBPUSD", "XAUUSD", etc.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points. Can be:
        - Integer: Direct value (250 dbps = 0.25%)
        - String preset: "micro" (10 dbps), "tight" (50 dbps), "standard" (100 dbps),
          "medium" (250 dbps), "wide" (500 dbps), "macro" (1000 dbps)
        Valid range: 1-100,000 dbps (0.001% to 100%)
    ouroboros : {"year", "month", "week"}, default="year"
        Cyclical reset boundary for reproducible bar construction (v11.0+).
        Processor state resets at each boundary for deterministic results.
        - "year" (default): Reset at January 1st 00:00:00 UTC (cryptocurrency)
        - "month": Reset at 1st of each month 00:00:00 UTC
        - "week": Reset at Sunday 00:00:00 UTC (required for Forex)
        Named after the Greek serpent eating its tail (οὐροβόρος).
    include_orphaned_bars : bool, default=False
        Include incomplete bars from ouroboros boundaries.
        If True, orphaned bars are included with ``is_orphan=True`` column.
        Useful for analysis; filter with ``df[~df.get('is_orphan', False)]``.
    materialize : bool, default=True
        If True, return a single pd.DataFrame (legacy behavior).
        If False, return an Iterator[pl.DataFrame] that yields batches
        of bars for memory-efficient streaming (v8.0+).
    batch_size : int, default=10_000
        Number of bars per batch when materialize=False.
        Each batch is ~500 KB. Only used in streaming mode.

    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only):
        - "spot": Spot market
        - "futures-um" or "um": USD-M perpetual futures
        - "futures-cm" or "cm": COIN-M perpetual futures
    validation : str, default="strict"
        Validation strictness (Exness only):
        - "permissive": Basic checks (bid > 0, ask > 0, bid < ask)
        - "strict": + Spread < 10% (catches obvious errors)
        - "paranoid": + Spread < 1% (flags suspicious data)
    include_incomplete : bool, default=False
        Include the final incomplete bar (useful for analysis).
        If False (default), only completed bars are returned.
    include_microstructure : bool, default=False
        Include market microstructure columns:
        - buy_volume, sell_volume: Volume by aggressor side
        - vwap: Volume-weighted average price
        - trade_count: Number of trades in bar
        - (Exness) spread_min, spread_max, spread_avg: Spread statistics
    include_exchange_sessions : bool, default=False
        Include traditional exchange market session flags (Issue #8).
        When True, adds boolean columns indicating active sessions at bar close:
        - exchange_session_sydney: ASX (10:00-16:00 Sydney time)
        - exchange_session_tokyo: TSE (09:00-15:00 Tokyo time)
        - exchange_session_london: LSE (08:00-17:00 London time)
        - exchange_session_newyork: NYSE (10:00-16:00 New York time)
        Useful for analyzing crypto/forex behavior during traditional market hours.
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention (Issue #36).
        If True (default): A bar cannot close on the same timestamp it opened.
        This prevents flash crash scenarios from creating thousands of bars
        at identical timestamps. If False: Legacy v8 behavior where bars can
        close immediately on breach regardless of timestamp. Use False for
        comparative analysis between old and new behavior.
    verify_checksum : bool, default=True
        Verify SHA-256 checksum of downloaded data (Issue #43).
        If True (default): Verify downloaded ZIP files against Binance-provided
        checksums to detect data corruption early. If verification fails,
        raises RuntimeError. If False: Skip checksum verification for faster
        downloads (use when data integrity is verified elsewhere).
    use_cache : bool, default=True
        Cache tick data locally in Parquet format.
    fetch_if_missing : bool, default=True
        If True (default), fetch tick data from source when not available
        in cache. If False, return only cached data (may return empty
        DataFrame if no cached data exists for the date range).
    cache_dir : str or None, default=None
        Custom cache directory. If None, uses platform default:
        - macOS: ~/Library/Caches/rangebar/
        - Linux: ~/.cache/rangebar/
        - Windows: %LOCALAPPDATA%/terrylica/rangebar/Cache/
    max_memory_mb : int or None, default=None
        Memory budget in MB for tick data loading. If the estimated
        in-memory size exceeds this limit, raises MemoryError. If None,
        uses automatic detection (80% of available RAM). Set to 0 to
        disable all memory guards.

    Returns
    -------
    pd.DataFrame or Iterator[pl.DataFrame]
        If materialize=True (default): Single pd.DataFrame ready for
        backtesting.py, with DatetimeIndex and OHLCV columns.

        If materialize=False: Iterator yielding pl.DataFrame batches
        (batch_size bars each) for memory-efficient streaming. Convert
        to pandas with: ``pl.concat(list(iterator)).to_pandas()``

        Columns: Open, High, Low, Close, Volume
        (if include_microstructure) Additional columns

    Raises
    ------
    ValueError
        - Invalid threshold (outside 1-100,000 range)
        - Invalid dates or date format
        - Unknown source, market, or validation level
        - Unknown threshold preset name
    RuntimeError
        - Data fetching failed
        - No data available for date range
        - Feature not enabled (e.g., Exness without exness feature)

    Examples
    --------
    Basic usage - Binance spot:

    >>> from rangebar import get_range_bars
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

    Using threshold presets:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", "tight")

    Binance USD-M Futures:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

    Exness forex with spread monitoring:

    >>> df = get_range_bars(
    ...     "EURUSD", "2024-01-01", "2024-01-31",
    ...     source="exness",
    ...     threshold_decimal_bps="standard",
    ...     include_microstructure=True,  # includes spread stats
    ... )

    Include incomplete bar for analysis:

    >>> df = get_range_bars(
    ...     "ETHUSDT", "2024-01-01", "2024-01-07",
    ...     include_incomplete=True,
    ... )

    Use with backtesting.py:

    >>> from backtesting import Backtest, Strategy
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31")
    >>> bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
    >>> stats = bt.run()

    Streaming mode for large datasets (v8.0+):

    >>> import polars as pl
    >>> # Memory-efficient: yields ~500 KB batches
    >>> for batch in get_range_bars(
    ...     "BTCUSDT", "2024-01-01", "2024-06-30",
    ...     materialize=False,
    ...     batch_size=10_000,
    ... ):
    ...     process_batch(batch)  # batch is pl.DataFrame
    ...
    >>> # Or collect to single DataFrame:
    >>> batches = list(get_range_bars(
    ...     "BTCUSDT", "2024-01-01", "2024-03-31",
    ...     materialize=False,
    ... ))
    >>> df = pl.concat(batches).to_pandas()

    Notes
    -----
    Threshold units (decimal basis points):
        The threshold is specified in decimal basis points (0.1bps) for precision.
        Common conversions:
        - 10 = 1bps = 0.01%
        - 100 = 10bps = 0.1%
        - 250 = 25bps = 0.25%
        - 1000 = 100bps = 1%

    Tier-1 symbols:
        18 high-liquidity symbols available on ALL Binance markets:
        AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE, ETH, FIL,
        LINK, LTC, NEAR, SOL, SUI, UNI, WIF, WLD, XRP

    Non-lookahead guarantee:
        - Threshold computed from bar OPEN price only
        - Breaching trade included in closing bar
        - No future information used in bar construction

    See Also
    --------
    TIER1_SYMBOLS : Tuple of high-liquidity symbols
    THRESHOLD_PRESETS : Dictionary of named threshold values
    """
    from datetime import datetime
    from pathlib import Path

    from rangebar.storage.parquet import TickStorage

    # -------------------------------------------------------------------------
    # Resolve threshold (support presets)
    # -------------------------------------------------------------------------
    if isinstance(threshold_decimal_bps, str):
        if threshold_decimal_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_decimal_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold_decimal_bps = THRESHOLD_PRESETS[threshold_decimal_bps]

    if not THRESHOLD_DECIMAL_MIN <= threshold_decimal_bps <= THRESHOLD_DECIMAL_MAX:
        msg = (
            f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and {THRESHOLD_DECIMAL_MAX}, "
            f"got {threshold_decimal_bps}"
        )
        raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Validate ouroboros mode (v11.0+)
    # -------------------------------------------------------------------------
    from rangebar.ouroboros import validate_ouroboros_mode

    ouroboros = validate_ouroboros_mode(ouroboros)

    # -------------------------------------------------------------------------
    # Validate source and market
    # -------------------------------------------------------------------------
    source = source.lower()
    if source not in ("binance", "exness"):
        msg = f"Unknown source: {source!r}. Must be 'binance' or 'exness'"
        raise ValueError(msg)

    # Normalize market type
    market_map = {
        "spot": "spot",
        "futures-um": "um",
        "futures-cm": "cm",
        "um": "um",
        "cm": "cm",
    }
    market = market.lower()
    if source == "binance" and market not in market_map:
        msg = (
            f"Unknown market: {market!r}. "
            "Must be 'spot', 'futures-um'/'um', or 'futures-cm'/'cm'"
        )
        raise ValueError(msg)
    market_normalized = market_map.get(market, market)

    # Validate Exness validation strictness
    validation = validation.lower()
    if source == "exness" and validation not in ("permissive", "strict", "paranoid"):
        msg = (
            f"Unknown validation: {validation!r}. "
            "Must be 'permissive', 'strict', or 'paranoid'"
        )
        raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Parse and validate dates
    # -------------------------------------------------------------------------
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
        raise ValueError(msg) from e

    if start_dt > end_dt:
        msg = "start_date must be <= end_date"
        raise ValueError(msg)

    # Convert to milliseconds for cache lookup
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int((end_dt.timestamp() + 86399) * 1000)  # End of day

    # -------------------------------------------------------------------------
    # Streaming mode (v8.0+): Return generator instead of materializing
    # -------------------------------------------------------------------------
    if not materialize:
        if source == "exness":
            msg = (
                "Streaming mode (materialize=False) is not yet supported for Exness. "
                "Use materialize=True or use Binance source."
            )
            raise ValueError(msg)

        # Binance streaming: yields batches directly from network
        return _stream_range_bars_binance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=threshold_decimal_bps,
            market=market_normalized,
            batch_size=batch_size,
            include_microstructure=include_microstructure,
            include_incomplete=include_incomplete,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
            verify_checksum=verify_checksum,
        )

    # -------------------------------------------------------------------------
    # Check ClickHouse bar cache first (Issue #21: fast path for precomputed bars)
    # -------------------------------------------------------------------------
    if use_cache:
        try:
            from rangebar.clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                # Ouroboros mode filter ensures cache isolation (Plan: sparkling-coalescing-dijkstra.md)
                cached_bars = cache.get_bars_by_timestamp_range(
                    symbol=symbol,
                    threshold_decimal_bps=threshold_decimal_bps,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    include_microstructure=include_microstructure,
                    ouroboros_mode=ouroboros,
                )
                if cached_bars is not None and len(cached_bars) > 0:
                    # Tier 0 validation: Content-based staleness detection (Issue #39)
                    # This catches stale cached data from pre-v7.0 (e.g., VWAP=0)
                    if include_microstructure:
                        staleness = detect_staleness(
                            cached_bars, require_microstructure=True
                        )
                        if staleness.is_stale:
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.warning(
                                "Stale cache data detected for %s: %s. "
                                "Falling through to recompute.",
                                symbol,
                                staleness.reason,
                            )
                            # Fall through to tick processing path
                        else:
                            # Fast path: return validated bars from ClickHouse (~50ms)
                            return cached_bars
                    else:
                        # Fast path: return precomputed bars from ClickHouse (~50ms)
                        return cached_bars
        except ImportError:
            # ClickHouse not available, fall through to tick processing
            pass
        except ConnectionError:
            # ClickHouse connection failed, fall through to tick processing
            pass

    # -------------------------------------------------------------------------
    # Initialize storage (Tier 1: local Parquet ticks)
    # -------------------------------------------------------------------------
    storage = TickStorage(cache_dir=Path(cache_dir) if cache_dir else None)

    # Cache key includes source and market to avoid collisions
    cache_symbol = f"{source}_{market_normalized}_{symbol}".upper()

    # -------------------------------------------------------------------------
    # Determine tick data source (cache or network)
    # -------------------------------------------------------------------------
    has_cached_ticks = use_cache and storage.has_ticks(cache_symbol, start_ts, end_ts)

    if not has_cached_ticks and not fetch_if_missing:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # For Exness, load all ticks upfront (smaller datasets)
    if source == "exness":
        if has_cached_ticks:
            tick_data = storage.read_ticks(cache_symbol, start_ts, end_ts)
        else:
            tick_data = _fetch_exness(symbol, start_date, end_date, validation)
            if use_cache and not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)
        if tick_data.is_empty():
            msg = f"No data available for {symbol} from {start_date} to {end_date}"
            raise RuntimeError(msg)
        return _process_exness_ticks(
            tick_data,
            symbol,
            threshold_decimal_bps,
            validation,
            include_incomplete,
            include_microstructure,
        )

    # -------------------------------------------------------------------------
    # MEM-010: Pre-flight memory estimation (Issue #49)
    # Check if cached tick data would fit in memory before loading.
    # -------------------------------------------------------------------------
    if has_cached_ticks and max_memory_mb != 0:
        import warnings

        from rangebar.resource_guard import estimate_tick_memory

        estimate = estimate_tick_memory(
            storage, cache_symbol, start_ts, end_ts
        )
        if estimate.recommendation == "will_oom":
            msg = (
                f"Loading {symbol} ({start_date} -> {end_date}) would "
                f"require ~{estimate.estimated_memory_mb} MB "
                f"(available: {estimate.system_available_mb} MB). "
                f"Use precompute_range_bars() for streaming processing."
            )
            if max_memory_mb is not None:
                estimate.check_or_raise(max_mb=max_memory_mb)
            else:
                raise MemoryError(msg)
        elif estimate.recommendation == "streaming_recommended":
            warnings.warn(
                f"Large tick dataset for {symbol} "
                f"(~{estimate.estimated_memory_mb} MB). "
                f"Consider precompute_range_bars() for memory-safe "
                f"processing.",
                ResourceWarning,
                stacklevel=2,
            )

    # -------------------------------------------------------------------------
    # Binance: Process with ouroboros segment iteration (Issue #51)
    # Load ticks per-segment to avoid OOM on large date ranges.
    # Each segment loads only the ticks within its boundaries (~1 year max).
    # -------------------------------------------------------------------------
    from rangebar.ouroboros import iter_ouroboros_segments

    all_bars: list[pd.DataFrame] = []
    processor: RangeBarProcessor | None = None
    any_data_found = False

    for segment_start, segment_end, boundary in iter_ouroboros_segments(
        start_dt.date(), end_dt.date(), ouroboros
    ):
        # Reset processor at ouroboros boundary
        if boundary is not None and processor is not None:
            orphaned_bar = processor.reset_at_ouroboros()
            if include_orphaned_bars and orphaned_bar is not None:
                # Add orphan metadata
                orphaned_bar["is_orphan"] = True
                orphaned_bar["ouroboros_boundary"] = boundary.timestamp
                orphaned_bar["ouroboros_reason"] = boundary.reason
                orphan_df = pd.DataFrame([orphaned_bar])
                # Convert timestamp to datetime index
                if "timestamp" in orphan_df.columns:
                    orphan_df["timestamp"] = pd.to_datetime(
                        orphan_df["timestamp"], unit="us", utc=True
                    )
                    orphan_df = orphan_df.set_index("timestamp")
                all_bars.append(orphan_df)

        # Load tick data scoped to this segment (not the full range)
        segment_start_ms = int(segment_start.timestamp() * 1_000)
        segment_end_ms = int(segment_end.timestamp() * 1_000)

        if has_cached_ticks:
            segment_ticks = storage.read_ticks(
                cache_symbol, segment_start_ms, segment_end_ms
            )
        else:
            # Fetch from network for this segment only
            seg_start_str = segment_start.strftime("%Y-%m-%d")
            seg_end_str = segment_end.strftime("%Y-%m-%d")
            segment_ticks = _fetch_binance(
                symbol, seg_start_str, seg_end_str, market_normalized
            )
            # Cache segment ticks
            if use_cache and not segment_ticks.is_empty():
                storage.write_ticks(cache_symbol, segment_ticks)

        if segment_ticks.is_empty():
            continue

        any_data_found = True

        # Process segment (reuse processor for state continuity within segment)
        segment_bars, processor = _process_binance_trades(
            segment_ticks,
            threshold_decimal_bps,
            include_incomplete,
            include_microstructure,
            processor=processor,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
        )

        if segment_bars is not None and not segment_bars.empty:
            all_bars.append(segment_bars)

    if not any_data_found:
        msg = f"No data available for {symbol} from {start_date} to {end_date}"
        raise RuntimeError(msg)

    # Concatenate all segments
    if not all_bars:
        bars_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    elif len(all_bars) == 1:
        bars_df = all_bars[0]
    else:
        bars_df = pd.concat(all_bars, axis=0)
        bars_df = bars_df.sort_index()

    # -------------------------------------------------------------------------
    # Add exchange session flags (Issue #8)
    # -------------------------------------------------------------------------
    # Session flags indicate which traditional market sessions were active
    # at bar close time. Useful for analyzing crypto/forex behavior.
    if include_exchange_sessions and not bars_df.empty:
        import warnings

        from rangebar.ouroboros import get_active_exchange_sessions

        # Compute session flags for each bar based on close timestamp (index)
        session_data = {
            "exchange_session_sydney": [],
            "exchange_session_tokyo": [],
            "exchange_session_london": [],
            "exchange_session_newyork": [],
        }
        for ts in bars_df.index:
            # Ensure timezone-aware UTC timestamp
            if ts.tzinfo is None:
                ts_utc = ts.tz_localize("UTC")
            else:
                ts_utc = ts.tz_convert("UTC")
            # Suppress nanosecond warning - session detection is hour-granularity
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Discarding nonzero nanoseconds")
                flags = get_active_exchange_sessions(ts_utc.to_pydatetime())
            session_data["exchange_session_sydney"].append(flags.sydney)
            session_data["exchange_session_tokyo"].append(flags.tokyo)
            session_data["exchange_session_london"].append(flags.london)
            session_data["exchange_session_newyork"].append(flags.newyork)

        # Add columns to DataFrame
        for col, values in session_data.items():
            bars_df[col] = values

    # -------------------------------------------------------------------------
    # Write computed bars to ClickHouse cache (Issue #37)
    # -------------------------------------------------------------------------
    # Cache write is non-blocking: failures don't affect the return value.
    # The computation succeeded, so we return bars even if caching fails.
    if use_cache and bars_df is not None and not bars_df.empty:
        try:
            from rangebar.clickhouse import RangeBarCache
            from rangebar.exceptions import CacheError

            with RangeBarCache() as cache:
                # Use store_bars_bulk for bars computed without exact CacheKey
                # Ouroboros mode determines cache key (Plan: sparkling-coalescing-dijkstra.md)
                written = cache.store_bars_bulk(
                    symbol=symbol,
                    threshold_decimal_bps=threshold_decimal_bps,
                    bars=bars_df,
                    version="",  # Version tracked elsewhere
                    ouroboros_mode=ouroboros,
                )
                import logging

                logger = logging.getLogger(__name__)
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
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Cache write failed (non-fatal): %s", e)

    return bars_df


def get_range_bars_pandas(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    **kwargs: Any,
) -> pd.DataFrame:
    """Get range bars as pandas DataFrame (deprecated compatibility shim).

    .. deprecated:: 8.0
        Use ``get_range_bars(materialize=True)`` directly instead.
        This function will be removed in v9.0.

    This function exists for backward compatibility with code written before
    the streaming API was introduced. It simply calls ``get_range_bars()``
    with ``materialize=True`` and returns the result.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points
    **kwargs
        Additional arguments passed to ``get_range_bars()``

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    Examples
    --------
    Instead of:

    >>> df = get_range_bars_pandas("BTCUSDT", "2024-01-01", "2024-06-30")

    Use:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", materialize=True)
    """
    import warnings

    warnings.warn(
        "get_range_bars_pandas() is deprecated. "
        "Use get_range_bars(materialize=True) instead. "
        "This function will be removed in v9.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_range_bars(
        symbol,
        start_date,
        end_date,
        threshold_decimal_bps,
        materialize=True,
        **kwargs,
    )
