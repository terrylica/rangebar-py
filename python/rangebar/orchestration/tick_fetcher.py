# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization - Extract tick fetching loop from count_bounded.py
"""Tick fetching orchestration with storage caching and deduplication.

This module provides the unified tick fetching loop used by both
_fill_gap_and_cache() and _fetch_and_compute_bars() in count_bounded.py.

The key insight: For 24/7 crypto markets, ALL ticks must be processed with
a SINGLE processor to maintain the bar[i+1].open == bar[i].close invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from rangebar.storage.parquet import TickStorage

logger = logging.getLogger("rangebar")


@dataclass
class FetchResult:
    """Result of a tick fetching operation."""

    ticks: pl.DataFrame | None
    """Merged and deduplicated tick data, sorted chronologically."""

    oldest_timestamp_ms: int | None
    """Oldest timestamp in the fetched data (milliseconds)."""

    total_ticks: int
    """Total number of ticks fetched."""


def fetch_ticks_with_backoff(
    *,
    symbol: str,
    source: str,
    market: str,
    target_ticks: int,
    end_dt: datetime,
    oldest_ts: int | None,
    max_lookback_days: int,
    storage: TickStorage,
    cache_dir: Path | None = None,
    max_attempts: int = 5,
    initial_multiplier: float = 2.0,
) -> FetchResult:
    """Fetch tick data with adaptive exponential backoff.

    This function implements the common tick fetching loop used by both
    cache-aware and compute-only code paths. It handles:
    - Adaptive backoff to estimate required tick volume
    - Local storage caching (read existing, write new)
    - Chronological merging and deduplication
    - Lookback safety limits

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT", "EURUSD")
    source : str
        Data source: "binance" or "exness"
    market : str
        Normalized market type: "spot", "um", or "cm"
    target_ticks : int
        Target number of ticks to fetch (with buffer)
    end_dt : datetime
        End datetime for fetching (timezone-aware UTC)
    oldest_ts : int | None
        Oldest known timestamp (milliseconds) to fetch before, or None
    max_lookback_days : int
        Safety limit: maximum days to look back
    storage : TickStorage
        Tick storage instance for caching
    cache_dir : Path | None, default=None
        Custom cache directory (passed to storage)
    max_attempts : int, default=5
        Maximum number of fetch attempts with backoff
    initial_multiplier : float, default=2.0
        Initial backoff multiplier

    Returns
    -------
    FetchResult
        Contains merged tick data, oldest timestamp, and total tick count
    """
    import polars as pl

    from .helpers import _fetch_binance, _fetch_exness

    cache_symbol = f"{source}_{market}_{symbol}".upper()
    all_tick_data: list[pl.DataFrame] = []
    total_ticks = 0
    multiplier = initial_multiplier
    current_oldest_ts = oldest_ts

    for _attempt in range(max_attempts):
        # Calculate fetch range
        if current_oldest_ts is not None:
            fetch_end_dt = datetime.fromtimestamp(current_oldest_ts / 1000, tz=UTC)
        else:
            fetch_end_dt = end_dt

        # Estimate days to fetch based on remaining ticks needed
        remaining_ticks = target_ticks - total_ticks
        days_to_fetch = max(1, remaining_ticks // 1_000_000)
        days_to_fetch = min(days_to_fetch, max_lookback_days)

        fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

        # Check lookback limit
        if (end_dt - fetch_start_dt).days > max_lookback_days:
            break

        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
        start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
        end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

        # Fetch tick data (from storage or source)
        tick_data: pl.DataFrame
        if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
            tick_data = storage.read_ticks(cache_symbol, start_ts_fetch, end_ts_fetch)
        else:
            if source == "binance":
                tick_data = _fetch_binance(symbol, start_date, end_date_str, market)
            else:  # exness
                tick_data = _fetch_exness(symbol, start_date, end_date_str, "strict")

            if not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)

        if tick_data.is_empty():
            break

        # Prepend (older data first)
        all_tick_data.insert(0, tick_data)
        total_ticks += len(tick_data)

        # Update oldest timestamp for next iteration
        if "timestamp" in tick_data.columns:
            current_oldest_ts = int(tick_data["timestamp"].min())

        # Check if we have enough ticks
        if total_ticks >= target_ticks:
            break

        multiplier *= 2

    if not all_tick_data:
        return FetchResult(ticks=None, oldest_timestamp_ms=None, total_ticks=0)

    # Merge all ticks chronologically
    merged_ticks = pl.concat(all_tick_data)
    merged_ticks = _sort_and_deduplicate(merged_ticks)

    # Get oldest timestamp from merged data
    final_oldest_ts: int | None = None
    if "timestamp" in merged_ticks.columns:
        final_oldest_ts = int(merged_ticks["timestamp"].min())

    return FetchResult(
        ticks=merged_ticks,
        oldest_timestamp_ms=final_oldest_ts,
        total_ticks=len(merged_ticks),
    )


def _sort_and_deduplicate(ticks: pl.DataFrame) -> pl.DataFrame:
    """Sort tick data chronologically and remove duplicates.

    Sorting order follows Rust crate requirements: (timestamp, trade_id).
    Deduplication uses trade_id or agg_trade_id to handle boundary overlaps.

    Parameters
    ----------
    ticks : pl.DataFrame
        Raw tick data (potentially with duplicates and unsorted)

    Returns
    -------
    pl.DataFrame
        Sorted and deduplicated tick data
    """
    # Sort by (timestamp, trade_id) - Rust crate requires this order
    if "agg_trade_id" in ticks.columns:
        ticks = ticks.sort(["timestamp", "agg_trade_id"])
        # Deduplicate by agg_trade_id (Binance data may have duplicates at boundaries)
        ticks = ticks.unique(subset=["agg_trade_id"], maintain_order=True)
    elif "trade_id" in ticks.columns:
        ticks = ticks.sort(["timestamp", "trade_id"])
        ticks = ticks.unique(subset=["trade_id"], maintain_order=True)
    else:
        ticks = ticks.sort("timestamp")

    return ticks


def estimate_ticks_per_bar(threshold_decimal_bps: int, base_ticks: int = 2500) -> int:
    """Estimate ticks needed per bar based on threshold.

    Uses inverse relationship: smaller threshold = more bars = fewer ticks per bar.
    Calibrated for medium threshold (250 dbps) = 2500 ticks per bar.

    Parameters
    ----------
    threshold_decimal_bps : int
        Threshold in decimal basis points
    base_ticks : int, default=2500
        Base ticks per bar at 250 dbps

    Returns
    -------
    int
        Estimated ticks per bar for the given threshold
    """
    threshold_ratio = 250 / max(threshold_decimal_bps, 1)
    return int(base_ticks * threshold_ratio)
