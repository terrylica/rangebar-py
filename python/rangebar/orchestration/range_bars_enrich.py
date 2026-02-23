# Issue #46: Modularization - Extract enrichment helpers from range_bars.py
"""Post-processing enrichment for range bar DataFrames.

Provides standalone functions extracted from get_range_bars() to reduce
module size and improve testability:
- enrich_exchange_sessions(): Add exchange session flags
- filter_output_columns(): Filter columns for backtesting.py compatibility
"""

from __future__ import annotations

import warnings

import pandas as pd


def enrich_exchange_sessions(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Add exchange session flags (sydney/tokyo/london/newyork) to bars.

    Session flags indicate which traditional market sessions were active
    at bar close time (the DataFrame index). Useful for analyzing crypto/forex
    behavior during traditional market hours.

    Columns added:
    - exchange_session_sydney: ASX (10:00-16:00 Sydney time)
    - exchange_session_tokyo: TSE (09:00-15:00 Tokyo time)
    - exchange_session_london: LSE (08:00-17:00 London time)
    - exchange_session_newyork: NYSE (10:00-16:00 New York time)

    Parameters
    ----------
    bars_df : pd.DataFrame
        Range bar DataFrame with DatetimeIndex (bar close timestamps).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 4 boolean session columns added.
    """
    if bars_df.empty:
        return bars_df

    from rangebar.ouroboros import get_active_exchange_sessions

    # Issue #96 Task #30: Vectorize timezone conversion and batch session lookups
    # Instead of 500+ function calls, use unique hourly timestamps (~24 calls)

    # 1. Vectorize timezone conversion (single operation instead of per-row)
    index = bars_df.index
    if index.tzinfo is None:
        index_utc = index.tz_localize("UTC")
    else:
        index_utc = index.tz_convert("UTC")

    # 2. Get unique hourly timestamps (session boundaries don't change minute-to-minute)
    hourly_index = index_utc.floor("1H")
    unique_hours = hourly_index.unique()

    # 3. Batch compute sessions for unique hours
    session_map = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Discarding nonzero nanoseconds")
        for hour_ts in unique_hours:
            flags = get_active_exchange_sessions(hour_ts.to_pydatetime())
            session_map[hour_ts] = flags

    # 4. Issue #96 Task #36: Batch consolidate map/apply chains into single pass
    # Extract all 4 flags from session_map in one map operation (not 4 separate)
    def extract_all_sessions(flags_obj: object) -> tuple[bool, bool, bool, bool]:
        if flags_obj is None:
            return False, False, False, False
        return (flags_obj.sydney, flags_obj.tokyo,
                flags_obj.london, flags_obj.newyork)

    # Single map pass returns Series of tuples
    sessions_series = hourly_index.map(session_map).map(extract_all_sessions)

    # Convert tuples to individual columns in one operation
    sessions_array = pd.DataFrame(
        sessions_series.tolist(),
        columns=["exchange_session_sydney", "exchange_session_tokyo",
                 "exchange_session_london", "exchange_session_newyork"],
        index=bars_df.index,
    )

    # Assign all 4 columns at once (1 operation instead of 4)
    bars_df[sessions_array.columns] = sessions_array

    return bars_df


def filter_output_columns(
    bars_df: pd.DataFrame, include_microstructure: bool
) -> pd.DataFrame:
    """Filter columns based on include_microstructure flag.

    Cache storage uses the full DataFrame (with trade IDs for data integrity).
    User-facing output respects include_microstructure for backtesting.py
    compatibility. When microstructure is not requested, only OHLCV columns
    are returned.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Range bar DataFrame (may include microstructure columns).
    include_microstructure : bool
        If True, return all columns. If False, return only OHLCV columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the requested columns.
    """
    if include_microstructure or bars_df is None or bars_df.empty:
        return bars_df

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in ohlcv_cols if c in bars_df.columns]
    return bars_df[available_cols]
