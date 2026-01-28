"""Conversion utilities for rangebar-py.

This module provides dtype conversion, normalization, and DataFrame manipulation
utilities used throughout the codebase. These functions handle:
- Converting bar dictionaries to Polars/pandas DataFrames
- Concatenating DataFrames with consistent dtypes
- Normalizing datetime precision (Issue #44 fix)
- Converting PyArrow dtypes to numpy for compatibility

SSoT (Single Source of Truth) for:
- _bars_list_to_polars: Convert bar dicts to Polars DataFrame
- _concat_pandas_via_polars: Memory-efficient DataFrame concatenation
- normalize_temporal_precision: Fix mixed datetime precision
- normalize_arrow_dtypes: Convert PyArrow to numpy dtypes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import polars as pl


def _bars_list_to_polars(
    bars: list[dict],
    include_microstructure: bool = False,
) -> pl.DataFrame:
    """Convert list of bar dicts to Polars DataFrame in backtesting.py format.

    Parameters
    ----------
    bars : list[dict]
        List of bar dictionaries from processor
    include_microstructure : bool
        Include microstructure columns

    Returns
    -------
    pl.DataFrame
        DataFrame with OHLCV columns (capitalized), DatetimeIndex
    """
    import polars as pl

    if not bars:
        return pl.DataFrame()

    bars_df = pl.DataFrame(bars)

    # Convert timestamp to datetime
    if "timestamp" in bars_df.columns:
        bars_df = bars_df.with_columns(
            pl.col("timestamp")
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z")
            .alias("timestamp")
        )

    # Rename to backtesting.py format
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    bars_df = bars_df.rename(
        {k: v for k, v in rename_map.items() if k in bars_df.columns}
    )

    # Select columns
    base_cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    if include_microstructure:
        # Include all columns
        return bars_df
    # Only OHLCV columns
    available = [c for c in base_cols if c in bars_df.columns]
    return bars_df.select(available)


def normalize_temporal_precision(pldf: pl.DataFrame) -> pl.DataFrame:
    """Normalize datetime columns to microsecond precision.

    This prevents SchemaError when concatenating DataFrames with mixed
    datetime precision (e.g., μs vs ns). See Issue #44.

    Parameters
    ----------
    pldf : pl.DataFrame
        Polars DataFrame to normalize

    Returns
    -------
    pl.DataFrame
        DataFrame with all datetime columns cast to microsecond precision
    """
    import polars as pl

    for col in pldf.columns:
        if pldf[col].dtype.is_temporal():
            pldf = pldf.with_columns(pl.col(col).dt.cast_time_unit("us"))
    return pldf


def _concat_pandas_via_polars(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate pandas DataFrames using Polars for memory efficiency (MEM-006).

    This function uses Polars' more efficient concatenation instead of pd.concat,
    reducing memory fragmentation and improving performance for large datasets.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of pandas DataFrames to concatenate

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with sorted DatetimeIndex
    """
    import polars as pl

    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    # Convert to Polars
    pl_dfs = [pl.from_pandas(df.reset_index()) for df in dfs]

    # Normalize datetime columns to consistent precision (μs) before concat
    # This prevents SchemaError when months have mixed precision (Issue #44)
    normalized = [normalize_temporal_precision(pldf) for pldf in pl_dfs]
    combined = pl.concat(normalized)

    # Sort by timestamp/index column
    index_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined = combined.sort(index_col)

    # Convert back to pandas with proper index
    result = combined.to_pandas()
    if index_col in result.columns:
        result = result.set_index(index_col)

    return result


def normalize_arrow_dtypes(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Convert PyArrow dtypes to numpy for compatibility.

    ClickHouse query_df_arrow returns double[pyarrow], but process_trades
    returns float64. This function normalizes the dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing PyArrow dtypes
    columns : list[str] | None
        Columns to normalize. If None, normalizes OHLCV columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with numpy dtypes
    """
    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    return df
