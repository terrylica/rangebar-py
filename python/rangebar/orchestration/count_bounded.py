# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M4 - Extract count-bounded orchestration from __init__.py
"""Count-bounded range bar retrieval (get_n_range_bars).

This module provides the count-bounded API for retrieving exactly N range bars,
useful for ML training and walk-forward optimization. Includes adaptive
gap-filling with exponential backoff for cache misses.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from rangebar.constants import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
)
from rangebar.conversion import _concat_pandas_via_polars
from rangebar.validation.cache_staleness import detect_staleness
from rangebar.validation.continuity import (
    ContinuityError,
    ContinuityWarning,
    validate_junction_continuity,
)

from .helpers import (
    _fetch_binance,
    _fetch_exness,
    _process_binance_trades,
    _process_exness_ticks,
)

if TYPE_CHECKING:
    import polars as pl

    from rangebar.clickhouse import RangeBarCache

# Module-level logger (matches __init__.py pattern)
import logging

logger = logging.getLogger("rangebar")


def get_n_range_bars(
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: int | str = 250,
    *,
    end_date: str | None = None,
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    prevent_same_timestamp_close: bool = True,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    validate_on_return: bool = False,
    continuity_action: str = "warn",
    chunk_size: int = 100_000,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Get exactly N range bars ending at or before a given date.

    Unlike `get_range_bars()` which uses date bounds (producing variable bar counts),
    this function returns a deterministic number of bars. This is useful for:
    - ML training (exactly 10,000 samples)
    - Walk-forward optimization (fixed window sizes)
    - Consistent backtest comparisons

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    n_bars : int
        Number of bars to retrieve. Must be > 0.
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points. Can be:
        - Integer: Direct value (250 = 25bps = 0.25%)
        - String preset: "micro", "tight", "standard", "medium", "wide", "macro"
    end_date : str or None, default=None
        End date in YYYY-MM-DD format. If None, uses most recent available data.
    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only): "spot", "futures-um", or "futures-cm"
    include_microstructure : bool, default=False
        Include microstructure columns (vwap, buy_volume, sell_volume)
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention (Issue #36).
        If True (default): A bar cannot close on the same timestamp it opened.
        If False: Legacy v8 behavior for comparative analysis.
    use_cache : bool, default=True
        Use ClickHouse cache for bar retrieval/storage
    fetch_if_missing : bool, default=True
        Fetch and process new data if cache doesn't have enough bars
    max_lookback_days : int, default=90
        Safety limit: maximum days to look back when fetching missing data.
        Prevents runaway fetches on empty caches.
    warn_if_fewer : bool, default=True
        Emit UserWarning if returning fewer bars than requested.
    validate_on_return : bool, default=False
        If True, validate bar continuity before returning.
        Uses continuity_action to determine behavior on failure.
    continuity_action : str, default="warn"
        Action when discontinuity found during validation:
        - "warn": Log warning but return data
        - "raise": Raise ContinuityError
        - "log": Silent logging only
    chunk_size : int, default=100_000
        Number of ticks per processing chunk for memory efficiency.
        Larger values = faster processing, more memory.
        Default 100K = ~15MB memory overhead.
    cache_dir : str or None, default=None
        Custom cache directory for tick data (Tier 1).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with exactly n_bars rows (or fewer if not enough data),
        sorted chronologically (oldest first). Columns:
        - Open, High, Low, Close, Volume
        - (if include_microstructure) vwap, buy_volume, sell_volume

    Raises
    ------
    ValueError
        - n_bars <= 0
        - Invalid threshold
        - Invalid date format
    RuntimeError
        - ClickHouse not available when use_cache=True
        - Data fetching failed

    Examples
    --------
    Get last 10,000 bars for ML training:

    >>> from rangebar import get_n_range_bars
    >>> df = get_n_range_bars("BTCUSDT", n_bars=10000)
    >>> assert len(df) == 10000

    Get 5,000 bars ending at specific date for walk-forward:

    >>> df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

    With safety limit (won't fetch more than 30 days of data):

    >>> df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)

    Notes
    -----
    Cache behavior:
        - Fast path: If cache has >= n_bars, returns immediately (~50ms)
        - Slow path: If cache has < n_bars and fetch_if_missing=True,
          fetches additional data, computes bars, stores in cache, returns

    Gap-filling algorithm:
        Uses adaptive exponential backoff to estimate how many ticks to fetch.
        Learns compression ratio (ticks/bar) for each (symbol, threshold) pair.

    See Also
    --------
    get_range_bars : Date-bounded bar retrieval (variable bar count)
    THRESHOLD_PRESETS : Named threshold values
    """
    import warnings
    from datetime import datetime

    import numpy as np

    # -------------------------------------------------------------------------
    # Validation helper (closure over validate_on_return, continuity_action)
    # -------------------------------------------------------------------------
    def _apply_validation(df: pd.DataFrame) -> pd.DataFrame:
        """Apply continuity validation if enabled, then return DataFrame."""
        if not validate_on_return or df.empty or len(df) <= 1:
            return df

        # Check continuity: Close[i] should equal Open[i+1]
        close_prices = df["Close"].to_numpy()[:-1]
        open_prices = df["Open"].to_numpy()[1:]

        # Calculate relative differences
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.abs(open_prices - close_prices) / np.abs(close_prices)

        # 0.01% tolerance for floating-point errors
        tolerance = 0.0001
        discontinuities_mask = rel_diff > tolerance

        if not np.any(discontinuities_mask):
            return df

        # Found discontinuities
        discontinuity_count = int(np.sum(discontinuities_mask))
        msg = f"Found {discontinuity_count} discontinuities in {len(df)} bars"

        if continuity_action == "raise":
            # Build details for ContinuityError
            indices = np.where(discontinuities_mask)[0]
            details = []
            for idx in indices[:10]:  # Limit to first 10
                details.append(
                    {
                        "bar_index": int(idx),
                        "prev_close": float(close_prices[idx]),
                        "next_open": float(open_prices[idx]),
                        "gap_pct": float(rel_diff[idx] * 100),
                    }
                )
            raise ContinuityError(msg, details)
        if continuity_action == "warn":
            warnings.warn(msg, ContinuityWarning, stacklevel=3)
        else:  # "log"
            logging.getLogger("rangebar").warning(msg)

        return df

    # -------------------------------------------------------------------------
    # Validate parameters
    # -------------------------------------------------------------------------
    if n_bars <= 0:
        msg = f"n_bars must be > 0, got {n_bars}"
        raise ValueError(msg)

    # Resolve threshold (support presets)
    threshold: int
    if isinstance(threshold_decimal_bps, str):
        if threshold_decimal_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_decimal_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold = THRESHOLD_PRESETS[threshold_decimal_bps]
    else:
        threshold = threshold_decimal_bps

    if not THRESHOLD_DECIMAL_MIN <= threshold <= THRESHOLD_DECIMAL_MAX:
        msg = (
            f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and "
            f"{THRESHOLD_DECIMAL_MAX}, got {threshold}"
        )
        raise ValueError(msg)

    # Normalize source and market
    source = source.lower()
    if source not in ("binance", "exness"):
        msg = f"Unknown source: {source!r}. Must be 'binance' or 'exness'"
        raise ValueError(msg)

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

    # Parse end_date if provided
    end_ts: int | None = None
    if end_date is not None:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            # End of day in milliseconds
            end_ts = int((end_dt.timestamp() + 86399) * 1000)
        except ValueError as e:
            msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
            raise ValueError(msg) from e

    # -------------------------------------------------------------------------
    # Try cache first (if enabled)
    # -------------------------------------------------------------------------
    if use_cache:
        try:
            from rangebar.clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                # Fast path: check if cache has enough bars
                bars_df, available_count = cache.get_n_bars(
                    symbol=symbol,
                    threshold_decimal_bps=threshold,
                    n_bars=n_bars,
                    before_ts=end_ts,
                    include_microstructure=include_microstructure,
                )

                if bars_df is not None and len(bars_df) >= n_bars:
                    # Tier 0 validation: Content-based staleness detection (Issue #39)
                    if include_microstructure:
                        staleness = detect_staleness(
                            bars_df, require_microstructure=True
                        )
                        if staleness.is_stale:
                            logger.warning(
                                "Stale cache data detected for %s: %s. "
                                "Falling through to recompute.",
                                symbol,
                                staleness.reason,
                            )
                            # Fall through to fetch_if_missing path
                        else:
                            # Cache hit - return exactly n_bars
                            return _apply_validation(bars_df.tail(n_bars))
                    else:
                        # Cache hit - return exactly n_bars
                        return _apply_validation(bars_df.tail(n_bars))

                # Slow path: need to fetch more data
                if fetch_if_missing:
                    bars_df = _fill_gap_and_cache(
                        symbol=symbol,
                        threshold=threshold,
                        n_bars=n_bars,
                        end_ts=end_ts,
                        source=source,
                        market=market_normalized,
                        include_microstructure=include_microstructure,
                        max_lookback_days=max_lookback_days,
                        cache=cache,
                        cache_dir=Path(cache_dir) if cache_dir else None,
                        current_bars=bars_df,
                        current_count=available_count,
                        chunk_size=chunk_size,
                        prevent_same_timestamp_close=prevent_same_timestamp_close,
                    )

                    if bars_df is not None and len(bars_df) >= n_bars:
                        return _apply_validation(bars_df.tail(n_bars))

                # Return what we have (or None)
                if bars_df is not None and len(bars_df) > 0:
                    if warn_if_fewer and len(bars_df) < n_bars:
                        warnings.warn(
                            f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                            f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return _apply_validation(bars_df)

                # Empty result
                if warn_if_fewer:
                    warnings.warn(
                        f"Returning 0 bars instead of requested {n_bars}. "
                        "No data available in cache or from source.",
                        UserWarning,
                        stacklevel=2,
                    )
                return pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"]
                ).set_index(pd.DatetimeIndex([]))

        except Exception as e:
            # ClickHouse not available - fall through to compute-only mode
            if "ClickHouseNotConfigured" in type(e).__name__:
                pass  # Fall through to compute-only mode
            else:
                raise

    # -------------------------------------------------------------------------
    # Compute-only mode (no cache)
    # -------------------------------------------------------------------------
    if not fetch_if_missing:
        if warn_if_fewer:
            warnings.warn(
                f"Returning 0 bars instead of requested {n_bars}. "
                "Cache disabled and fetch_if_missing=False.",
                UserWarning,
                stacklevel=2,
            )
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))

    # Fetch and compute without caching
    bars_df = _fetch_and_compute_bars(
        symbol=symbol,
        threshold=threshold,
        n_bars=n_bars,
        end_ts=end_ts,
        source=source,
        market=market_normalized,
        include_microstructure=include_microstructure,
        max_lookback_days=max_lookback_days,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    if bars_df is not None and len(bars_df) >= n_bars:
        return _apply_validation(bars_df.tail(n_bars))

    if bars_df is not None and len(bars_df) > 0:
        if warn_if_fewer:
            warnings.warn(
                f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                UserWarning,
                stacklevel=2,
            )
        return _apply_validation(bars_df)

    if warn_if_fewer:
        warnings.warn(
            f"Returning 0 bars instead of requested {n_bars}. "
            "No data available from source.",
            UserWarning,
            stacklevel=2,
        )
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).set_index(
        pd.DatetimeIndex([])
    )


def _fill_gap_and_cache(
    symbol: str,
    threshold: int,
    n_bars: int,
    end_ts: int | None,
    source: str,
    market: str,
    include_microstructure: bool,
    max_lookback_days: int,
    cache: RangeBarCache,
    cache_dir: Path | None,
    current_bars: pd.DataFrame | None,
    current_count: int,
    chunk_size: int = 100_000,
    prevent_same_timestamp_close: bool = True,
) -> pd.DataFrame | None:
    """Fill gap in cache by fetching and processing additional data.

    Uses checkpoint-based cross-file continuity for Binance (24/7 crypto markets).
    The key insight: ALL ticks must be processed with a SINGLE processor to
    maintain the bar[i+1].open == bar[i].close invariant.

    For Binance (24/7):
    1. Collect ALL tick data first (no intermediate processing)
    2. Merge all ticks chronologically
    3. Process with SINGLE processor (guarantees continuity)
    4. Store with unified cache key

    For Exness (forex):
    Session-bounded processing is acceptable since weekend gaps are natural.

    Parameters
    ----------
    chunk_size : int, default=100_000
        Number of ticks per processing chunk for memory efficiency when using
        chunked processing with checkpoint continuation.
    """
    from datetime import datetime, timedelta

    import polars as pl

    from rangebar.storage.parquet import TickStorage

    # Determine how many more bars we need
    bars_needed = n_bars - (len(current_bars) if current_bars is not None else 0)

    if bars_needed <= 0:
        return current_bars

    # Determine end date for fetching
    if end_ts is not None:
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
    else:
        end_dt = datetime.now(tz=UTC)

    # Get oldest bar timestamp to know where to start fetching
    oldest_ts = cache.get_oldest_bar_timestamp(symbol, threshold)

    # Estimate ticks needed using adaptive heuristic
    # Default: ~2500 ticks per bar for medium threshold (250)
    # Adjusted by threshold ratio
    base_ticks_per_bar = 2500
    threshold_ratio = 250 / max(threshold, 1)
    estimated_ticks_per_bar = int(base_ticks_per_bar * threshold_ratio)

    # Adaptive exponential backoff
    multiplier = 2.0  # Start with 2x buffer
    max_attempts = 5

    storage = TickStorage(cache_dir=cache_dir)
    cache_symbol = f"{source}_{market}_{symbol}".upper()

    # =========================================================================
    # BINANCE (24/7 CRYPTO): Single-pass processing with checkpoint continuity
    # =========================================================================
    if source == "binance":
        # Phase 1: Collect ALL tick data first
        all_tick_data: list[pl.DataFrame] = []
        total_ticks = 0
        target_ticks = bars_needed * estimated_ticks_per_bar * 2  # 2x buffer

        for _attempt in range(max_attempts):
            # Calculate fetch range
            if oldest_ts is not None:
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
            else:
                fetch_end_dt = end_dt

            # Estimate days to fetch
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

            # Fetch tick data
            tick_data: pl.DataFrame
            if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
                tick_data = storage.read_ticks(
                    cache_symbol, start_ts_fetch, end_ts_fetch
                )
            else:
                tick_data = _fetch_binance(symbol, start_date, end_date_str, market)
                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

            if tick_data.is_empty():
                break

            all_tick_data.insert(0, tick_data)  # Prepend (older data first)
            total_ticks += len(tick_data)

            # Update oldest_ts for next iteration
            if "timestamp" in tick_data.columns:
                oldest_ts = int(tick_data["timestamp"].min())

            # Check if we have enough estimated ticks
            if total_ticks >= target_ticks:
                break

            multiplier *= 2

        if not all_tick_data:
            return current_bars

        # Phase 2: Merge ALL ticks chronologically and deduplicate
        # Sort by (timestamp, trade_id) - Rust crate requires this order
        merged_ticks = pl.concat(all_tick_data)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "agg_trade_id"])
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "trade_id"])
        else:
            merged_ticks = merged_ticks.sort("timestamp")

        # Deduplicate by trade_id (Binance data may have duplicates at boundaries)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(
                subset=["agg_trade_id"], maintain_order=True
            )
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(subset=["trade_id"], maintain_order=True)

        # Phase 3: Process with SINGLE processor (guarantees continuity)
        new_bars, _ = _process_binance_trades(
            merged_ticks,
            threshold,
            False,
            include_microstructure,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
        )

        # Phase 4: Store with unified cache key
        if not new_bars.empty:
            cache.store_bars_bulk(symbol, threshold, new_bars)

        # Combine with existing bars
        if current_bars is not None and len(current_bars) > 0:
            # Validate continuity at junction (new_bars older, current_bars newer)
            is_continuous, gap_pct = validate_junction_continuity(
                new_bars, current_bars
            )
            if not is_continuous:
                import warnings

                warnings.warn(
                    f"Discontinuity detected at junction: {symbol} @ {threshold} dbps. "
                    f"Gap: {gap_pct:.4%}. This occurs because range bars from different "
                    f"processing sessions cannot guarantee bar[n].close == bar[n+1].open. "
                    f"Consider invalidating cache and re-fetching all data for continuous "
                    f"bars. See: https://github.com/terrylica/rangebar-py/issues/5",
                    stacklevel=3,
                )

            # MEM-006: Use Polars for memory-efficient concatenation
            combined = _concat_pandas_via_polars([new_bars, current_bars])
            return combined[~combined.index.duplicated(keep="last")]

        return new_bars

    # =========================================================================
    # EXNESS (FOREX): Session-bounded processing (weekend gaps are natural)
    # =========================================================================
    all_bars: list[pd.DataFrame] = []
    if current_bars is not None and len(current_bars) > 0:
        all_bars.append(current_bars)

    for _attempt in range(max_attempts):
        # Estimate days to fetch
        ticks_to_fetch = int(bars_needed * estimated_ticks_per_bar * multiplier)
        days_to_fetch = max(1, ticks_to_fetch // 1_000_000)
        days_to_fetch = min(days_to_fetch, max_lookback_days)

        # Calculate fetch range
        if oldest_ts is not None:
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
        else:
            fetch_end_dt = end_dt

        fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

        if (end_dt - fetch_start_dt).days > max_lookback_days:
            break

        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
        start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
        end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

        tick_data: pl.DataFrame
        if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
            tick_data = storage.read_ticks(cache_symbol, start_ts_fetch, end_ts_fetch)
        else:
            tick_data = _fetch_exness(symbol, start_date, end_date_str, "strict")
            if not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)

        if tick_data.is_empty():
            break

        # Process to bars (forex: session-bounded is OK)
        new_bars = _process_exness_ticks(
            tick_data, symbol, threshold, "strict", False, include_microstructure
        )

        if not new_bars.empty:
            cache.store_bars_bulk(symbol, threshold, new_bars)
            all_bars.insert(0, new_bars)
            oldest_ts = int(new_bars.index.min().timestamp() * 1000)

            total_bars = sum(len(df) for df in all_bars)
            if total_bars >= n_bars:
                break

        multiplier *= 2

    if not all_bars:
        return None

    # MEM-006: Use Polars for memory-efficient concatenation
    combined = _concat_pandas_via_polars(all_bars)
    return combined[~combined.index.duplicated(keep="last")]


def _fetch_and_compute_bars(
    symbol: str,
    threshold: int,
    n_bars: int,
    end_ts: int | None,
    source: str,
    market: str,
    include_microstructure: bool,
    max_lookback_days: int,
    cache_dir: Path | None,
) -> pd.DataFrame | None:
    """Fetch and compute bars without caching (compute-only mode).

    Uses single-pass processing for Binance (24/7 crypto) to guarantee continuity.
    """
    from datetime import datetime, timedelta

    import polars as pl

    from rangebar.storage.parquet import TickStorage

    # Determine end date
    if end_ts is not None:
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
    else:
        end_dt = datetime.now(tz=UTC)

    # Estimate ticks needed using heuristic
    base_ticks_per_bar = 2500
    threshold_ratio = 250 / max(threshold, 1)
    estimated_ticks_per_bar = int(base_ticks_per_bar * threshold_ratio)

    # Adaptive exponential backoff
    multiplier = 2.0
    max_attempts = 5

    storage = TickStorage(cache_dir=cache_dir)
    cache_symbol = f"{source}_{market}_{symbol}".upper()
    oldest_ts: int | None = None

    # =========================================================================
    # BINANCE (24/7 CRYPTO): Single-pass processing for continuity
    # =========================================================================
    if source == "binance":
        all_tick_data: list[pl.DataFrame] = []
        total_ticks = 0
        target_ticks = n_bars * estimated_ticks_per_bar * 2

        for _attempt in range(max_attempts):
            if oldest_ts is not None:
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
            else:
                fetch_end_dt = end_dt

            remaining_ticks = target_ticks - total_ticks
            days_to_fetch = max(1, remaining_ticks // 1_000_000)
            days_to_fetch = min(days_to_fetch, max_lookback_days)

            fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

            if (end_dt - fetch_start_dt).days > max_lookback_days:
                break

            start_date = fetch_start_dt.strftime("%Y-%m-%d")
            end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
            start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
            end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

            tick_data: pl.DataFrame
            if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
                tick_data = storage.read_ticks(
                    cache_symbol, start_ts_fetch, end_ts_fetch
                )
            else:
                tick_data = _fetch_binance(symbol, start_date, end_date_str, market)
                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

            if tick_data.is_empty():
                break

            all_tick_data.insert(0, tick_data)
            total_ticks += len(tick_data)

            if "timestamp" in tick_data.columns:
                oldest_ts = int(tick_data["timestamp"].min())

            if total_ticks >= target_ticks:
                break

            multiplier *= 2

        if not all_tick_data:
            return None

        # Sort by (timestamp, trade_id) - Rust crate requires this order
        merged_ticks = pl.concat(all_tick_data)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "agg_trade_id"])
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "trade_id"])
        else:
            merged_ticks = merged_ticks.sort("timestamp")

        # Deduplicate by trade_id (Binance data may have duplicates at boundaries)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(
                subset=["agg_trade_id"], maintain_order=True
            )
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(subset=["trade_id"], maintain_order=True)

        bars_df, _ = _process_binance_trades(
            merged_ticks, threshold, False, include_microstructure, symbol=symbol
        )
        return bars_df if not bars_df.empty else None

    # =========================================================================
    # EXNESS (FOREX): Session-bounded processing
    # =========================================================================
    all_bars: list[pd.DataFrame] = []

    for _attempt in range(max_attempts):
        bars_still_needed = n_bars - sum(len(df) for df in all_bars)
        ticks_to_fetch = int(bars_still_needed * estimated_ticks_per_bar * multiplier)
        days_to_fetch = max(1, ticks_to_fetch // 1_000_000)
        days_to_fetch = min(days_to_fetch, max_lookback_days)

        if oldest_ts is not None:
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
        else:
            fetch_end_dt = end_dt

        fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

        if (end_dt - fetch_start_dt).days > max_lookback_days:
            break

        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
        start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
        end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

        tick_data: pl.DataFrame
        if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
            tick_data = storage.read_ticks(cache_symbol, start_ts_fetch, end_ts_fetch)
        else:
            tick_data = _fetch_exness(symbol, start_date, end_date_str, "strict")
            if not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)

        if tick_data.is_empty():
            break

        new_bars = _process_exness_ticks(
            tick_data, symbol, threshold, "strict", False, include_microstructure
        )

        if not new_bars.empty:
            all_bars.insert(0, new_bars)
            oldest_ts = int(new_bars.index.min().timestamp() * 1000)

            total_bars = sum(len(df) for df in all_bars)
            if total_bars >= n_bars:
                break

        multiplier *= 2

    if not all_bars:
        return None

    # MEM-006: Use Polars for memory-efficient concatenation
    combined = _concat_pandas_via_polars(all_bars)
    # Remove duplicates (by index) and return
    return combined[~combined.index.duplicated(keep="last")]
