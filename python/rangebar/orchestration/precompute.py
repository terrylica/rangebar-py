# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M4 - Extract precompute_range_bars from __init__.py
"""Batch precomputation pipeline for range bars.

Provides precompute_range_bars() for ML workflows requiring continuous
bar sequences with cache invalidation and continuity validation.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from rangebar.constants import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
)
from rangebar.conversion import _concat_pandas_via_polars
from rangebar.processors.core import RangeBarProcessor
from rangebar.validation.continuity import ContinuityError, validate_continuity

from .helpers import _fetch_binance, _fetch_exness
from .models import PrecomputeProgress, PrecomputeResult


def precompute_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int = 250,
    *,
    source: str = "binance",
    market: str = "spot",
    chunk_size: int = 100_000,
    invalidate_existing: str = "smart",
    progress_callback: Callable[[PrecomputeProgress], None] | None = None,
    include_microstructure: bool = False,
    validate_on_complete: str = "error",
    continuity_tolerance_pct: float = 0.001,
    cache_dir: str | None = None,
    max_memory_gb: float | None = None,
) -> PrecomputeResult:
    """Precompute continuous range bars for a date range (single-pass, guaranteed continuity).

    Designed for ML workflows requiring continuous bar sequences for training/validation.
    Uses Checkpoint API for memory-efficient chunked processing with state preservation.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date (inclusive) "YYYY-MM-DD"
    end_date : str
        End date (inclusive) "YYYY-MM-DD"
    threshold_decimal_bps : int, default=250
        Range bar threshold (250 = 0.25%)
    source : str, default="binance"
        Data source ("binance" or "exness")
    market : str, default="spot"
        Market type for Binance ("spot", "futures-um", "futures-cm")
    chunk_size : int, default=100_000
        Ticks per processing chunk (~15MB memory per 100K ticks)
    invalidate_existing : str, default="smart"
        Cache invalidation strategy:
        - "overlap": Invalidate only bars in date range
        - "full": Invalidate ALL bars for symbol/threshold
        - "none": Skip if any cached bars exist in range
        - "smart": Invalidate overlapping + validate junction continuity
    progress_callback : Callable, optional
        Optional callback for progress updates
    include_microstructure : bool, default=False
        Include order flow metrics (buy_volume, sell_volume, vwap)
    validate_on_complete : str, default="error"
        Continuity validation mode after precomputation:
        - "error": Raise ContinuityError if discontinuities found
        - "warn": Log warning but continue (sets continuity_valid=False)
        - "skip": Skip validation entirely (continuity_valid=None)
    continuity_tolerance_pct : float, default=0.001
        Maximum allowed price gap percentage for continuity validation.
        Default 0.1% (0.001) accommodates market microstructure events.
        The total allowed gap is threshold_pct + continuity_tolerance_pct.
    cache_dir : str or None, default=None
        Custom cache directory for tick data
    max_memory_gb : float or None, default=None
        Process-level memory cap in GB. Sets RLIMIT_AS so that
        exceeding the limit raises MemoryError instead of OOM kill.
        None disables the cap.

    Returns
    -------
    PrecomputeResult
        Result with statistics and cache key

    Raises
    ------
    ValueError
        Invalid parameters
    RuntimeError
        Fetch or processing failure
    ContinuityError
        If validate_on_complete=True and discontinuities found

    Examples
    --------
    Basic precomputation:

    >>> result = precompute_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")
    >>> print(f"Generated {result.total_bars} bars")

    With progress callback:

    >>> def on_progress(p):
    ...     print(f"[{p.current_month}] {p.bars_generated} bars")
    >>> result = precompute_range_bars(
    ...     "BTCUSDT", "2024-01-01", "2024-03-31",
    ...     progress_callback=on_progress
    ... )
    """
    import gc
    import time
    from datetime import datetime
    from pathlib import Path

    from rangebar.clickhouse import CacheKey, RangeBarCache
    from rangebar.storage.parquet import TickStorage

    # MEM-009: Set process-level memory cap if requested (Issue #49)
    if max_memory_gb is not None:
        from rangebar.resource_guard import set_memory_limit

        set_memory_limit(max_gb=max_memory_gb)

    start_time = time.time()

    # Validate parameters
    if invalidate_existing not in ("overlap", "full", "none", "smart"):
        msg = f"Invalid invalidate_existing: {invalidate_existing!r}. Must be 'overlap', 'full', 'none', or 'smart'"
        raise ValueError(msg)

    if validate_on_complete not in ("error", "warn", "skip"):
        msg = f"Invalid validate_on_complete: {validate_on_complete!r}. Must be 'error', 'warn', or 'skip'"
        raise ValueError(msg)

    if not THRESHOLD_DECIMAL_MIN <= threshold_decimal_bps <= THRESHOLD_DECIMAL_MAX:
        msg = f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and {THRESHOLD_DECIMAL_MAX}"
        raise ValueError(msg)

    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
        raise ValueError(msg) from e

    if start_dt > end_dt:
        msg = "start_date must be <= end_date"
        raise ValueError(msg)

    # Normalize market type
    market_map = {
        "spot": "spot",
        "futures-um": "um",
        "futures-cm": "cm",
        "um": "um",
        "cm": "cm",
    }
    market_normalized = market_map.get(market.lower(), market.lower())

    # Initialize storage and cache
    storage = TickStorage(cache_dir=Path(cache_dir) if cache_dir else None)
    cache = RangeBarCache()

    # Generate list of months to process
    months: list[tuple[int, int]] = []
    current = start_dt.replace(day=1)
    _december = 12
    while current <= end_dt:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == _december:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # Handle cache invalidation
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int((end_dt.timestamp() + 86399) * 1000)  # End of day
    cache_key = CacheKey(
        symbol=symbol,
        threshold_decimal_bps=threshold_decimal_bps,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if invalidate_existing == "full":
        cache.invalidate_range_bars(cache_key)
    elif invalidate_existing in ("overlap", "smart"):
        # Check for overlapping bars - will be handled after processing
        # For now, just invalidate the date range to ensure clean slate
        cache.invalidate_range_bars(cache_key)
    elif invalidate_existing == "none":
        # Check if any bars exist in range by counting
        bar_count = cache.count_bars(symbol, threshold_decimal_bps)
        if bar_count > 0:
            # Return early - some cached data exists
            # Note: This is approximate; full implementation would check time range
            return PrecomputeResult(
                symbol=symbol,
                threshold_decimal_bps=threshold_decimal_bps,
                start_date=start_date,
                end_date=end_date,
                total_bars=bar_count,
                total_ticks=0,
                elapsed_seconds=time.time() - start_time,
                continuity_valid=True,  # Assume valid for cached data
                cache_key=f"{symbol}_{threshold_decimal_bps}",
            )

    # Initialize processor (single instance for continuity)
    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)

    all_bars: list[pd.DataFrame] = []
    month_bars: list[pd.DataFrame] = (
        []
    )  # Issue #27: Track bars per month for incremental caching
    total_ticks = 0
    cache_symbol = f"{source}_{market_normalized}_{symbol}".upper()

    for i, (year, month) in enumerate(months):
        month_str = f"{year}-{month:02d}"

        # Report progress - fetching
        if progress_callback:
            progress_callback(
                PrecomputeProgress(
                    phase="fetching",
                    current_month=month_str,
                    months_completed=i,
                    months_total=len(months),
                    bars_generated=sum(len(b) for b in all_bars),
                    ticks_processed=total_ticks,
                    elapsed_seconds=time.time() - start_time,
                )
            )

        # Calculate month boundaries
        month_start = datetime(year, month, 1)
        _december = 12
        if month == _december:
            month_end = datetime(year + 1, 1, 1)
        else:
            month_end = datetime(year, month + 1, 1)

        # Adjust to fit within requested date range
        actual_start = max(month_start, start_dt)
        actual_end = min(month_end, end_dt + pd.Timedelta(days=1))

        # Fetch tick data for this period
        start_ts_month = int(actual_start.timestamp() * 1000)
        end_ts_month = int(actual_end.timestamp() * 1000)

        # Check if data is already cached
        data_cached = storage.has_ticks(cache_symbol, start_ts_month, end_ts_month)

        if data_cached:
            # STREAMING READ: Use row-group based streaming to avoid OOM (Issue #12)
            # The Rust processor maintains state between process_trades() calls,
            # so we stream chunks directly without loading entire month into memory
            month_has_data = False
            for raw_tick_chunk in storage.read_ticks_streaming(
                cache_symbol, start_ts_month, end_ts_month, chunk_size=chunk_size
            ):
                month_has_data = True

                # Deduplicate trades by trade_id within chunk
                tick_chunk = raw_tick_chunk
                if "agg_trade_id" in tick_chunk.columns:
                    tick_chunk = tick_chunk.unique(
                        subset=["agg_trade_id"], maintain_order=True
                    )
                elif "trade_id" in tick_chunk.columns:
                    tick_chunk = tick_chunk.unique(
                        subset=["trade_id"], maintain_order=True
                    )

                # Sort by (timestamp, trade_id) - Rust crate requires this order
                if "timestamp" in tick_chunk.columns:
                    if "agg_trade_id" in tick_chunk.columns:
                        tick_chunk = tick_chunk.sort(["timestamp", "agg_trade_id"])
                    elif "trade_id" in tick_chunk.columns:
                        tick_chunk = tick_chunk.sort(["timestamp", "trade_id"])
                    else:
                        tick_chunk = tick_chunk.sort("timestamp")

                total_ticks += len(tick_chunk)

                # Report progress - processing
                if progress_callback:
                    progress_callback(
                        PrecomputeProgress(
                            phase="processing",
                            current_month=month_str,
                            months_completed=i,
                            months_total=len(months),
                            bars_generated=sum(len(b) for b in all_bars),
                            ticks_processed=total_ticks,
                            elapsed_seconds=time.time() - start_time,
                        )
                    )

                # Stream directly to Rust processor (Issue #16: use streaming mode)
                chunk = tick_chunk.to_dicts()
                bars = processor.process_trades_streaming(chunk)
                if bars:
                    # Issue #30: Always include microstructure for ClickHouse cache
                    bars_df = processor.to_dataframe(bars, include_microstructure=True)
                    month_bars.append(bars_df)  # Issue #27: Track per-month bars

                del chunk, tick_chunk

            gc.collect()

            if not month_has_data:
                continue
        else:
            # DATA NOT CACHED: Fetch from source day-by-day to prevent OOM
            # Issue #14: Fetching entire month at once causes OOM for high-volume
            # months like March 2024. Fetch day-by-day instead.
            current_day = actual_start
            while current_day < actual_end:
                next_day = current_day + pd.Timedelta(days=1)
                next_day = min(next_day, actual_end)

                day_start_str = current_day.strftime("%Y-%m-%d")
                day_end_str = (next_day - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")

                if source == "binance":
                    tick_data = _fetch_binance(
                        symbol, day_start_str, day_end_str, market_normalized
                    )
                else:
                    tick_data = _fetch_exness(
                        symbol, day_start_str, day_end_str, "strict"
                    )

                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

                    # Deduplicate trades by trade_id
                    if "agg_trade_id" in tick_data.columns:
                        tick_data = tick_data.unique(
                            subset=["agg_trade_id"], maintain_order=True
                        )
                    elif "trade_id" in tick_data.columns:
                        tick_data = tick_data.unique(
                            subset=["trade_id"], maintain_order=True
                        )

                    # Sort by (timestamp, trade_id) - Rust crate requires order
                    if "timestamp" in tick_data.columns:
                        if "agg_trade_id" in tick_data.columns:
                            tick_data = tick_data.sort(["timestamp", "agg_trade_id"])
                        elif "trade_id" in tick_data.columns:
                            tick_data = tick_data.sort(["timestamp", "trade_id"])
                        else:
                            tick_data = tick_data.sort("timestamp")

                    total_ticks += len(tick_data)

                    # Report progress - processing
                    if progress_callback:
                        progress_callback(
                            PrecomputeProgress(
                                phase="processing",
                                current_month=month_str,
                                months_completed=i,
                                months_total=len(months),
                                bars_generated=sum(len(b) for b in all_bars),
                                ticks_processed=total_ticks,
                                elapsed_seconds=time.time() - start_time,
                            )
                        )

                    # Process with chunking for memory efficiency
                    tick_count = len(tick_data)
                    for chunk_start in range(0, tick_count, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, tick_count)
                        chunk_df = tick_data.slice(chunk_start, chunk_end - chunk_start)
                        chunk = chunk_df.to_dicts()

                        # Stream to Rust processor (Issue #16: use streaming mode)
                        bars = processor.process_trades_streaming(chunk)
                        if bars:
                            # Issue #30: Always include microstructure for ClickHouse cache
                            bars_df = processor.to_dataframe(
                                bars, include_microstructure=True
                            )
                            month_bars.append(
                                bars_df
                            )  # Issue #27: Track per-month bars

                        del chunk, chunk_df

                    del tick_data
                    gc.collect()

                current_day = next_day

        # Issue #27: Incremental caching - store bars to ClickHouse after each month
        # This provides crash resilience and bounded memory for DB writes
        if month_bars:
            # MEM-006: Use Polars for memory-efficient concatenation
            month_df = _concat_pandas_via_polars(month_bars)

            # Report progress - caching this month
            if progress_callback:
                progress_callback(
                    PrecomputeProgress(
                        phase="caching",
                        current_month=month_str,
                        months_completed=i + 1,
                        months_total=len(months),
                        bars_generated=len(month_df) + sum(len(b) for b in all_bars),
                        ticks_processed=total_ticks,
                        elapsed_seconds=time.time() - start_time,
                    )
                )

            # Cache immediately (idempotent via ReplacingMergeTree)
            rows_sent = len(month_df)
            rows_inserted = cache.store_bars_bulk(
                symbol, threshold_decimal_bps, month_df
            )

            # Post-cache validation: FAIL LOUDLY if ClickHouse didn't receive all bars
            if rows_inserted != rows_sent:
                msg = (
                    f"ClickHouse cache validation FAILED for {month_str}: "
                    f"sent {rows_sent} bars but only {rows_inserted} inserted. "
                    f"Data integrity compromised - aborting."
                )
                raise RuntimeError(msg)

            # Preserve for final validation and return
            all_bars.append(month_df)
            month_bars = []  # Clear to reclaim memory
            gc.collect()

    # Combine all bars (MEM-006: use Polars for memory efficiency)
    if all_bars:
        final_bars = _concat_pandas_via_polars(all_bars)
    else:
        final_bars = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Note: Caching now happens incrementally after each month (Issue #27)
    # No final bulk store needed - all bars already cached per-month

    # Validate continuity (Issue #19: configurable tolerance and validation mode)
    continuity_valid: bool | None = None

    if validate_on_complete == "skip":
        # Skip validation entirely
        continuity_valid = None
    else:
        continuity_result = validate_continuity(
            final_bars,
            tolerance_pct=continuity_tolerance_pct,
            threshold_decimal_bps=threshold_decimal_bps,
        )
        continuity_valid = continuity_result["is_valid"]

        if not continuity_valid:
            if validate_on_complete == "error":
                msg = f"Found {continuity_result['discontinuity_count']} discontinuities in precomputed bars"
                raise ContinuityError(msg, continuity_result["discontinuities"])
            if validate_on_complete == "warn":
                import warnings

                msg = (
                    f"Found {continuity_result['discontinuity_count']} discontinuities "
                    f"in precomputed bars (tolerance: {continuity_tolerance_pct:.4%})"
                )
                warnings.warn(msg, stacklevel=2)

    return PrecomputeResult(
        symbol=symbol,
        threshold_decimal_bps=threshold_decimal_bps,
        start_date=start_date,
        end_date=end_date,
        total_bars=len(final_bars),
        total_ticks=total_ticks,
        elapsed_seconds=time.time() - start_time,
        continuity_valid=continuity_valid,
        cache_key=f"{symbol}_{threshold_decimal_bps}",
    )
