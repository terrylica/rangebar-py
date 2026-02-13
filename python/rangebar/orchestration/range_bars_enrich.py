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
