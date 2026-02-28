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

from rangebar.processors.core import RangeBarProcessor

from .helpers import (
    _datetime_to_end_ms,
    _datetime_to_start_ms,
    _empty_ohlcv_dataframe,
    _fetch_binance,
    _fetch_exness,
    _no_data_error_message,
    _parse_and_validate_dates,
    _parse_microstructure_env_vars,
    _process_binance_trades,
    _process_exness_ticks,
    _stream_range_bars_binance,
    _validate_and_normalize_source_market,
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
    ouroboros_mode: Literal["year", "month", "week"] | None = None,
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
    # Inter-bar features (Issue #59)
    inter_bar_lookback_count: int | None = None,
    # Bar-relative lookback (Issue #81)
    inter_bar_lookback_bars: int | None = None,
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
    ouroboros_mode : {"year", "month", "week"}, default=None (resolved from config)
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
    inter_bar_lookback_count : int or None, default=None
        Number of trades to keep in lookback buffer for inter-bar feature
        computation (Issue #59). If set, enables 16 inter-bar features
        computed from trades BEFORE each bar opens. Recommended: 100-500.
        If None (default), inter-bar features are disabled.

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
    from pathlib import Path

    from rangebar.storage.parquet import TickStorage

    # Issue #96 Task #79: Normalize symbol ONCE to eliminate 3-4 redundant .upper() calls
    # downstream (validate_symbol_registered, resolve_and_validate_threshold, cache lookups)
    symbol = symbol.upper()

    # -------------------------------------------------------------------------
    # Symbol registry gate + start date clamping (Issue #79)
    # -------------------------------------------------------------------------
    from rangebar.symbol_registry import (
        validate_and_clamp_start_date,
        validate_symbol_registered,
    )

    validate_symbol_registered(symbol, operation="get_range_bars")
    start_date = validate_and_clamp_start_date(symbol, start_date)

    # Telemetry: generate trace_id for pipeline correlation
    from rangebar.logging import generate_trace_id

    trace_id = generate_trace_id("grb")  # grb = get_range_bars

    # -------------------------------------------------------------------------
    # Resolve threshold with asset-class validation (Issue #62)
    # -------------------------------------------------------------------------
    from rangebar.threshold import resolve_and_validate_threshold

    threshold_decimal_bps = resolve_and_validate_threshold(symbol, threshold_decimal_bps)

    # -------------------------------------------------------------------------
    # Validate ouroboros mode (v11.0+)
    # Issue #126: Resolve from config if not specified
    # -------------------------------------------------------------------------
    if ouroboros_mode is None:
        from rangebar.ouroboros import get_operational_ouroboros_mode

        ouroboros_mode = get_operational_ouroboros_mode()
    else:
        from rangebar.ouroboros import validate_ouroboros_mode

        ouroboros_mode = validate_ouroboros_mode(ouroboros_mode)

    # -------------------------------------------------------------------------
    # Validate source and market
    # -------------------------------------------------------------------------
    source, market_normalized = _validate_and_normalize_source_market(source, market)

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
    start_dt, end_dt = _parse_and_validate_dates(start_date, end_date)

    # Convert to milliseconds for cache lookup
    start_ts = _datetime_to_start_ms(start_dt)
    end_ts = _datetime_to_end_ms(end_dt)

    # -------------------------------------------------------------------------
    # MEM-013: Force ClickHouse-first for long date ranges (>30 days)
    # -------------------------------------------------------------------------
    # For long ranges, require cache population first to prevent OOM.
    # This forces users to use the memory-safe incremental workflow.
    from rangebar.constants import LONG_RANGE_DAYS

    days = (end_dt - start_dt).days
    if days > LONG_RANGE_DAYS:
        # For long ranges, data MUST come from cache
        if not use_cache:
            msg = (
                f"Date range of {days} days requires use_cache=True.\n"
                f"Long ranges must use ClickHouse cache for memory safety."
            )
            raise ValueError(msg)

        # Check if cache has data for long range
        try:
            from rangebar.clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                cached = cache.get_bars_by_timestamp_range(
                    symbol=symbol,
                    threshold_decimal_bps=threshold_decimal_bps,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    include_microstructure=include_microstructure,
                    ouroboros_mode=ouroboros_mode,
                )
                if cached is not None and len(cached) > 0:
                    # Success: return cached data
                    cached.index.name = "timestamp"
                    return cached

        except ImportError:
            msg = (
                f"Date range of {days} days requires ClickHouse cache.\n"
                f"Install: pip install clickhouse-connect"
            )
            raise ValueError(msg) from None
        except ConnectionError as e:
            msg = (
                f"Date range of {days} days requires ClickHouse cache.\n"
                f"Connection failed: {e}\n"
                f"Configure ClickHouse or reduce date range to <= {LONG_RANGE_DAYS} days."
            )
            raise ValueError(msg) from None

        # Cache empty: direct user to populate it
        msg = (
            f"Date range {start_date} to {end_date} ({days} days) exceeds "
            f"{LONG_RANGE_DAYS}-day limit.\n\n"
            f"Cache is empty. Populate it first:\n"
            f"  populate_cache_resumable('{symbol}', '{start_date}', '{end_date}', "
            f"threshold_decimal_bps={threshold_decimal_bps})\n\n"
            f"Then call get_range_bars() again to read from cache."
        )
        raise ValueError(msg)

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
            inter_bar_lookback_count=inter_bar_lookback_count,
        )

    # -------------------------------------------------------------------------
    # Check ClickHouse bar cache first (Issue #21: fast path for precomputed bars)
    # -------------------------------------------------------------------------
    if use_cache:
        from .range_bars_cache import try_cache_read

        cached_bars = try_cache_read(
            symbol=symbol,
            threshold_decimal_bps=threshold_decimal_bps,
            start_ts=start_ts,
            end_ts=end_ts,
            include_microstructure=include_microstructure,
            ouroboros_mode=ouroboros_mode,
            trace_id=trace_id,
        )
        if cached_bars is not None:
            return cached_bars

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
        return _empty_ohlcv_dataframe()

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
            inter_bar_lookback_count=inter_bar_lookback_count,
        )

    # -------------------------------------------------------------------------
    # MEM-010: Pre-flight memory estimation (Issue #49, #53)
    # Estimate memory for the LARGEST ouroboros segment, not the full range.
    # Data is loaded per-segment in the loop below, so the peak memory usage
    # is bounded by the largest single segment.
    # -------------------------------------------------------------------------
    if has_cached_ticks and max_memory_mb != 0:
        import warnings

        from rangebar.ouroboros import iter_ouroboros_segments
        from rangebar.resource_guard import estimate_tick_memory

        # Find the largest segment by time span
        segments = list(
            iter_ouroboros_segments(start_dt.date(), end_dt.date(), ouroboros_mode)
        )
        if segments:
            largest = max(
                segments, key=lambda s: (s[1] - s[0]).total_seconds()
            )
            est_start = _datetime_to_start_ms(largest[0])
            est_end = _datetime_to_end_ms(largest[1])
        else:
            est_start, est_end = start_ts, end_ts

        estimate = estimate_tick_memory(
            storage, cache_symbol, est_start, est_end
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
        start_dt.date(), end_dt.date(), ouroboros_mode
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
                # Convert close_time_ms to datetime index
                if "close_time_ms" in orphan_df.columns:
                    orphan_df["timestamp"] = pd.to_datetime(
                        orphan_df["close_time_ms"], unit="ms", utc=True
                    )
                    orphan_df = orphan_df.set_index("timestamp")
                    orphan_df = orphan_df.drop(columns=["close_time_ms", "open_time_ms"], errors="ignore")
                all_bars.append(orphan_df)

        # Load tick data scoped to this segment (not the full range)
        segment_start_ms = int(segment_start.timestamp() * 1_000)
        segment_end_ms = int(segment_end.timestamp() * 1_000)

        # Issue #48: Emit DOWNLOAD_START hook for tick fetch tracing
        from rangebar.hooks import HookEvent, emit_hook
        from rangebar.logging import log_download_event

        _tick_source = "cache" if has_cached_ticks else "network"
        emit_hook(
            HookEvent.DOWNLOAD_START, symbol=symbol,
            segment_start=str(segment_start), segment_end=str(segment_end),
            source=_tick_source,
        )
        log_download_event(
            "download_start", symbol=symbol,
            date=str(segment_start.date()), trace_id=trace_id,
            source=_tick_source,
        )

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

        # Issue #48: Emit DOWNLOAD_COMPLETE hook with tick count
        emit_hook(
            HookEvent.DOWNLOAD_COMPLETE, symbol=symbol,
            segment_start=str(segment_start), segment_end=str(segment_end),
            tick_count=len(segment_ticks),
        )
        log_download_event(
            "download_complete", symbol=symbol,
            date=str(segment_start.date()), trace_id=trace_id,
            tick_count=len(segment_ticks),
        )

        if segment_ticks.is_empty():
            continue

        any_data_found = True

        # Process segment (reuse processor for state continuity within segment)
        # Issue #68: Auto-enable v12 features when include_microstructure=True
        # Issue #81: Bar-relative lookback threading
        effective_lookback, effective_bars, enable_intra = _parse_microstructure_env_vars(
            include_microstructure, inter_bar_lookback_count, inter_bar_lookback_bars,
        )

        segment_bars, processor = _process_binance_trades(
            segment_ticks,
            threshold_decimal_bps,
            include_incomplete,
            include_microstructure,
            processor=processor,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
            inter_bar_lookback_count=effective_lookback,
            include_intra_bar_features=enable_intra,
            inter_bar_lookback_bars=effective_bars,
        )

        if segment_bars is not None and not segment_bars.empty:
            all_bars.append(segment_bars)

    if not any_data_found:
        # Issue #76: Auto-detect earliest available date for helpful error message
        raise RuntimeError(_no_data_error_message(symbol, start_date, end_date, market))

    # Concatenate all segments — use Polars-backed concat for memory efficiency (MEM-006)
    if not all_bars:
        bars_df = _empty_ohlcv_dataframe()
    elif len(all_bars) == 1:
        bars_df = all_bars[0]
    else:
        from rangebar.conversion import _concat_pandas_via_polars

        bars_df = _concat_pandas_via_polars(all_bars)

    # Issue #76: Explicit memory cleanup after concatenation
    # pd.concat creates a new DataFrame but source frames remain in list
    # Without this, memory grows unbounded during long-running cache population
    del all_bars

    # -------------------------------------------------------------------------
    # Add exchange session flags (Issue #8)
    # -------------------------------------------------------------------------
    if include_exchange_sessions:
        from .range_bars_enrich import enrich_exchange_sessions

        bars_df = enrich_exchange_sessions(bars_df)

    # -------------------------------------------------------------------------
    # Plugin feature enrichment (Issue #98: post-Rust, pre-cache)
    # -------------------------------------------------------------------------
    from rangebar.plugins.loader import enrich_bars

    bars_df = enrich_bars(bars_df, symbol, threshold_decimal_bps)

    # -------------------------------------------------------------------------
    # Write computed bars to ClickHouse cache (Issue #37)
    # -------------------------------------------------------------------------
    if use_cache:
        from .range_bars_cache import try_cache_write

        try_cache_write(bars_df, symbol, threshold_decimal_bps, ouroboros_mode)

    # -------------------------------------------------------------------------
    # Filter output columns based on include_microstructure (Issue #75)
    # -------------------------------------------------------------------------
    from .range_bars_enrich import filter_output_columns

    return filter_output_columns(bars_df, include_microstructure)


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
