"""Parquet query functions: streaming reads, existence checks, date-range iteration.

Extracted from TickStorage methods in parquet.py. All functions take an
explicit ``ticks_dir: Path`` parameter instead of accessing ``self._ticks_dir``,
making them usable both as standalone utilities and as delegation targets
for TickStorage methods.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .parquet_io import _validate_and_recover_parquet

logger = logging.getLogger(__name__)


# =============================================================================
# Path helpers (shared by TickStorage and standalone query functions)
# =============================================================================


def _get_symbol_dir(ticks_dir: Path, symbol: str) -> Path:
    """Get directory for a symbol's tick files.

    Parameters
    ----------
    ticks_dir : Path
        Root ticks directory (e.g., ``cache_dir / "ticks"``).
    symbol : str
        Trading symbol (e.g., "BTCUSDT").

    Returns
    -------
    Path
        Symbol-specific subdirectory.
    """
    return ticks_dir / symbol


def _get_parquet_path(ticks_dir: Path, symbol: str, year_month: str) -> Path:
    """Get path for a specific month's parquet file.

    Parameters
    ----------
    ticks_dir : Path
        Root ticks directory.
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    year_month : str
        Year-month string (e.g., "2024-01").

    Returns
    -------
    Path
        Path to the parquet file.
    """
    return _get_symbol_dir(ticks_dir, symbol) / f"{year_month}.parquet"


def _timestamp_to_year_month(timestamp_ms: int) -> str:
    """Convert millisecond timestamp to year-month string.

    Parameters
    ----------
    timestamp_ms : int
        Timestamp in milliseconds since epoch.

    Returns
    -------
    str
        Year-month string (e.g., "2024-01").
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m")


# =============================================================================
# Query functions (extracted from TickStorage methods)
# =============================================================================


def read_ticks_streaming(
    ticks_dir: Path,
    symbol: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    *,
    chunk_size: int = 100_000,
    timestamp_col: str = "timestamp",
) -> Iterator[pl.DataFrame]:
    """Read tick data in streaming chunks to avoid OOM on large months.

    This function yields chunks of tick data instead of loading everything
    into memory at once. Essential for high-volume months like March 2024.

    Parameters
    ----------
    ticks_dir : Path
        Root ticks directory (e.g., ``cache_dir / "ticks"``).
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_ts : int | None
        Start timestamp in milliseconds (inclusive)
    end_ts : int | None
        End timestamp in milliseconds (inclusive)
    chunk_size : int
        Number of rows per chunk (default: 100,000)
    timestamp_col : str
        Name of the timestamp column

    Yields
    ------
    pl.DataFrame
        Chunks of tick data, sorted by timestamp within each chunk

    Notes
    -----
    Memory usage is O(chunk_size) instead of O(total_ticks).
    Each chunk is sorted independently; overall order is maintained
    because parquet files are read in month order.

    Examples
    --------
    >>> for chunk in read_ticks_streaming(ticks_dir, "BTCUSDT", start_ts, end_ts):
    ...     process_chunk(chunk)
    """
    symbol_dir = _get_symbol_dir(ticks_dir, symbol)

    if not symbol_dir.exists():
        return

    # Find relevant parquet files
    parquet_files = sorted(symbol_dir.glob("*.parquet"))

    if not parquet_files:
        return

    # Filter files by month if time range specified
    if start_ts is not None and end_ts is not None:
        start_month = _timestamp_to_year_month(start_ts)
        end_month = _timestamp_to_year_month(end_ts)

        parquet_files = [
            f for f in parquet_files if start_month <= f.stem <= end_month
        ]

    if not parquet_files:
        return

    # Issue #73: Validate files before streaming, auto-delete corrupted ones
    valid_files = []
    for f in parquet_files:
        if _validate_and_recover_parquet(f, auto_delete=True):
            valid_files.append(f)
        # else: corrupted file deleted, will be re-fetched on next call

    if not valid_files:
        return

    parquet_files = valid_files

    # Process each parquet file using PyArrow's row group-based reading
    # Row groups are Parquet's native chunking mechanism (typically 64K-1M rows)
    # This is the key to avoiding OOM - we never load the entire file into memory
    import pyarrow.parquet as pq

    for parquet_file in parquet_files:
        # Read parquet file in row groups
        parquet_reader = pq.ParquetFile(parquet_file)
        num_row_groups = parquet_reader.metadata.num_row_groups

        accumulated_rows: list[pl.DataFrame] = []
        accumulated_count = 0

        for rg_idx in range(num_row_groups):
            # Read single row group into PyArrow table (memory efficient)
            row_group = parquet_reader.read_row_group(rg_idx)
            chunk_df = pl.from_arrow(row_group)

            # Apply time range filter using Polars expressions
            if start_ts is not None:
                chunk_df = chunk_df.filter(
                    pl.col(timestamp_col) >= pl.lit(start_ts)
                )
            if end_ts is not None:
                chunk_df = chunk_df.filter(pl.col(timestamp_col) <= pl.lit(end_ts))

            if chunk_df.is_empty():
                continue

            accumulated_rows.append(chunk_df)
            accumulated_count += len(chunk_df)

            # Yield when accumulated enough rows
            while accumulated_count >= chunk_size:
                # Concatenate and slice
                combined = pl.concat(accumulated_rows)
                combined = combined.sort(timestamp_col)

                # Yield chunk_size rows
                yield combined.slice(0, chunk_size)

                # Keep remainder for next iteration
                remainder_count = accumulated_count - chunk_size
                if remainder_count > 0:
                    remainder = combined.slice(chunk_size, remainder_count)
                    accumulated_rows = [remainder]
                    accumulated_count = len(remainder)
                else:
                    accumulated_rows = []
                    accumulated_count = 0

                del combined

        # Yield any remaining rows
        if accumulated_rows:
            combined = pl.concat(accumulated_rows)
            combined = combined.sort(timestamp_col)
            if not combined.is_empty():
                yield combined
            del combined


def has_ticks(
    ticks_dir: Path,
    symbol: str,
    start_ts: int,
    end_ts: int,
    *,
    min_coverage: float = 0.95,
    timestamp_col: str = "timestamp",
) -> bool:
    """Check if tick data exists for the specified time range.

    Uses lazy Parquet scan (reads only metadata + timestamp column)
    instead of materializing all tick data into memory.

    Parameters
    ----------
    ticks_dir : Path
        Root ticks directory (e.g., ``cache_dir / "ticks"``).
    symbol : str
        Trading symbol
    start_ts : int
        Start timestamp in milliseconds
    end_ts : int
        End timestamp in milliseconds
    min_coverage : float
        Minimum coverage ratio (0.0 to 1.0)
    timestamp_col : str
        Name of the timestamp column

    Returns
    -------
    bool
        True if sufficient data exists
    """
    symbol_dir = _get_symbol_dir(ticks_dir, symbol)
    if not symbol_dir.exists():
        return False

    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return False

    # Filter files by month range
    start_month = _timestamp_to_year_month(start_ts)
    end_month = _timestamp_to_year_month(end_ts)
    parquet_files = [
        f for f in parquet_files if start_month <= f.stem <= end_month
    ]

    if not parquet_files:
        return False

    # Issue #73: Validate files before reading
    valid_files = [
        f
        for f in parquet_files
        if _validate_and_recover_parquet(f, auto_delete=True)
    ]
    if not valid_files:
        return False

    # Lazy scan: reads only timestamp column metadata, never materializes full data
    lazy = pl.scan_parquet(valid_files)

    # Apply time range filter (predicate pushdown)
    lazy = lazy.filter(
        (pl.col(timestamp_col) >= start_ts)
        & (pl.col(timestamp_col) <= end_ts)
    )

    # Aggregate: only reads min/max of timestamp column
    # Issue #96 Task #33: Add limit(1) to help Polars optimize query execution
    stats = lazy.select(
        pl.col(timestamp_col).min().alias("min_ts"),
        pl.col(timestamp_col).max().alias("max_ts"),
        pl.col(timestamp_col).count().alias("count"),
    ).limit(1).collect()

    if stats["count"][0] == 0:
        return False

    actual_start = stats["min_ts"][0]
    actual_end = stats["max_ts"][0]
    if actual_start is None or actual_end is None:
        return False

    actual_range = actual_end - actual_start
    requested_range = end_ts - start_ts

    if requested_range == 0:
        return stats["count"][0] > 0

    coverage = actual_range / requested_range
    return coverage >= min_coverage


def fetch_date_range(
    ticks_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    timestamp_col: str = "timestamp",
) -> Iterator[pl.LazyFrame]:
    """Iterate over tick data for date range, one month at a time.

    Yields LazyFrames for each month in range. This is the recommended
    approach for processing large date ranges without loading all data
    into memory at once.

    Parameters
    ----------
    ticks_dir : Path
        Root ticks directory (e.g., ``cache_dir / "ticks"``).
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date "YYYY-MM-DD"
    end_date : str
        End date "YYYY-MM-DD"
    timestamp_col : str
        Name of the timestamp column

    Yields
    ------
    pl.LazyFrame
        Lazy frame for each month with available data

    Examples
    --------
    >>> for lf in fetch_date_range(ticks_dir, "BTCUSDT", "2024-01-01", "2024-03-31"):
    ...     df = lf.collect()
    ...     print(f"Processing {len(df)} ticks")

    Notes
    -----
    - Only yields LazyFrames for months with cached data
    - Data is NOT automatically downloaded - use get_range_bars() first
    - Each LazyFrame can be collected independently for O(month) memory
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    current = start_dt.replace(day=1)
    while current <= end_dt:
        year = current.year
        month = current.month
        year_month = f"{year}-{month:02d}"

        parquet_path = _get_parquet_path(ticks_dir, symbol, year_month)
        if parquet_path.exists():
            lf = pl.scan_parquet(parquet_path)
            yield lf

        # Move to next month
        _december = 12
        if current.month == _december:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
