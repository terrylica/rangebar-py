"""Parquet-based tick storage with ZSTD-3 compression.

This module replaces ClickHouse Tier 1 (raw trades cache) with local
Parquet files for better portability and no server requirement.

Compression choice (based on empirical benchmark 2026-01-07):
- ZSTD-3: 6.50 MB for 761K trades (5.37x compression)
- Write: 0.019s, Read: 0.006s
- Beats Brotli on BOTH size AND speed for tick data
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from platformdirs import user_cache_dir

if TYPE_CHECKING:
    import pandas as pd

# Constants
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
APP_NAME = "rangebar"
APP_AUTHOR = "terrylica"


def get_cache_dir() -> Path:
    """Get the cross-platform cache directory for rangebar.

    Returns
    -------
    Path
        Platform-specific cache directory:
        - macOS:   ~/Library/Caches/rangebar/
        - Linux:   ~/.cache/rangebar/ (respects XDG_CACHE_HOME)
        - Windows: %USERPROFILE%\\AppData\\Local\\terrylica\\rangebar\\Cache\\

    Examples
    --------
    >>> from rangebar.storage import get_cache_dir
    >>> cache_dir = get_cache_dir()
    >>> print(cache_dir)
    /Users/username/Library/Caches/rangebar
    """
    # Allow override via environment variable
    env_override = os.getenv("RANGEBAR_CACHE_DIR")
    if env_override:
        return Path(env_override)

    return Path(user_cache_dir(APP_NAME, APP_AUTHOR))


class TickStorage:
    """Parquet-based tick data storage with ZSTD-3 compression.

    Stores raw tick data in Parquet files partitioned by symbol and month.
    Uses polars for fast I/O with ZSTD-3 compression.

    Parameters
    ----------
    cache_dir : Path | str | None
        Custom cache directory. If None, uses platformdirs default.

    Examples
    --------
    >>> storage = TickStorage()
    >>> storage.write_ticks("BTCUSDT", trades_df)
    >>> df = storage.read_ticks("BTCUSDT", start_ts, end_ts)

    Directory Structure
    -------------------
    ~/.cache/rangebar/ticks/
    ├── BTCUSDT/
    │   ├── 2024-01.parquet
    │   ├── 2024-02.parquet
    │   └── ...
    └── EURUSD/
        └── ...
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize tick storage.

        Parameters
        ----------
        cache_dir : Path | str | None
            Custom cache directory. If None, uses platformdirs default.
        """
        if cache_dir is None:
            self._cache_dir = get_cache_dir()
        else:
            self._cache_dir = Path(cache_dir)

        self._ticks_dir = self._cache_dir / "ticks"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self._cache_dir

    @property
    def ticks_dir(self) -> Path:
        """Get the ticks storage directory path."""
        return self._ticks_dir

    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get directory for a symbol's tick files."""
        return self._ticks_dir / symbol

    def _get_parquet_path(self, symbol: str, year_month: str) -> Path:
        """Get path for a specific month's parquet file.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        year_month : str
            Year-month string (e.g., "2024-01")

        Returns
        -------
        Path
            Path to the parquet file
        """
        return self._get_symbol_dir(symbol) / f"{year_month}.parquet"

    def _timestamp_to_year_month(self, timestamp_ms: int) -> str:
        """Convert millisecond timestamp to year-month string."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        return dt.strftime("%Y-%m")

    def write_ticks(
        self,
        symbol: str,
        ticks: pl.DataFrame | pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
    ) -> int:
        """Write tick data to Parquet files, partitioned by month.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        ticks : pl.DataFrame | pd.DataFrame
            Tick data with timestamp column
        timestamp_col : str
            Name of the timestamp column (milliseconds since epoch or datetime)

        Returns
        -------
        int
            Number of rows written

        Notes
        -----
        Tick data is partitioned by month and appended to existing files.
        Duplicates are not automatically removed - use ClickHouse for deduplication.
        """
        # Convert pandas to polars if needed
        if not isinstance(ticks, pl.DataFrame):
            ticks = pl.from_pandas(ticks)

        if ticks.is_empty():
            return 0

        # Ensure symbol directory exists
        symbol_dir = self._get_symbol_dir(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Convert timestamp to milliseconds if datetime
        if ticks.schema[timestamp_col] in (pl.Datetime, pl.Date):
            ticks = ticks.with_columns(
                pl.col(timestamp_col).dt.epoch(time_unit="ms").alias(timestamp_col)
            )

        # Add year_month column for partitioning (vectorized, no Python per-row calls)
        # MEM-001: Replaced map_elements() with native Polars dt operations
        # Impact: 13.4 GB → ~100 MB (99% reduction)
        ticks = ticks.with_columns(
            pl.col(timestamp_col)
            .cast(pl.Datetime(time_unit="ms"))
            .dt.strftime("%Y-%m")
            .alias("_year_month")
        )

        # Group by month and write
        total_rows = 0
        for (year_month,), group_df in ticks.group_by("_year_month"):
            parquet_path = self._get_parquet_path(symbol, year_month)

            # Drop the partition column before writing
            write_df = group_df.drop("_year_month")

            if parquet_path.exists():
                # Append to existing file
                existing_df = pl.read_parquet(parquet_path)
                combined_df = pl.concat([existing_df, write_df])
                combined_df.write_parquet(
                    parquet_path,
                    compression=COMPRESSION,
                    compression_level=COMPRESSION_LEVEL,
                )
            else:
                # Write new file
                write_df.write_parquet(
                    parquet_path,
                    compression=COMPRESSION,
                    compression_level=COMPRESSION_LEVEL,
                )

            total_rows += len(write_df)

        return total_rows

    # Default Parquet compression ratio (compressed → in-memory expansion)
    # Empirically measured: Binance aggTrades Parquet files expand ~4x
    _COMPRESSION_RATIO: float = 4.0

    def read_ticks(
        self,
        symbol: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        *,
        timestamp_col: str = "timestamp",
        max_memory_mb: int | None = None,
    ) -> pl.DataFrame:
        """Read tick data from Parquet files.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        start_ts : int | None
            Start timestamp in milliseconds (inclusive)
        end_ts : int | None
            End timestamp in milliseconds (inclusive)
        timestamp_col : str
            Name of the timestamp column
        max_memory_mb : int | None
            Memory budget in MB. If the estimated in-memory size exceeds
            this limit, raises MemoryError with a suggestion to use
            read_ticks_streaming(). None disables the guard.

        Returns
        -------
        pl.DataFrame
            Tick data filtered by time range

        Raises
        ------
        MemoryError
            If estimated memory exceeds max_memory_mb budget.

        Notes
        -----
        Reads all relevant monthly files and concatenates them.
        Uses lazy evaluation for efficient memory usage.
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return pl.DataFrame()

        # Find relevant parquet files
        parquet_files = sorted(symbol_dir.glob("*.parquet"))

        if not parquet_files:
            return pl.DataFrame()

        # Filter files by month if time range specified
        if start_ts is not None and end_ts is not None:
            start_month = self._timestamp_to_year_month(start_ts)
            end_month = self._timestamp_to_year_month(end_ts)

            parquet_files = [
                f for f in parquet_files if start_month <= f.stem <= end_month
            ]

        if not parquet_files:
            return pl.DataFrame()

        # MEM-004: Estimate size before materializing (Issue #49)
        if max_memory_mb is not None:
            total_bytes = sum(f.stat().st_size for f in parquet_files)
            estimated_mb = int(
                total_bytes * self._COMPRESSION_RATIO / (1024 * 1024)
            )
            if estimated_mb > max_memory_mb:
                msg = (
                    f"Estimated {estimated_mb} MB for {symbol} "
                    f"({len(parquet_files)} files), exceeds budget "
                    f"{max_memory_mb} MB. Use read_ticks_streaming() "
                    f"for chunked loading."
                )
                raise MemoryError(msg)

        # LAZY LOADING with predicate pushdown
        # Uses pl.scan_parquet() instead of pl.read_parquet() to enable:
        # 1. Predicate pushdown: filters applied at Parquet row-group level
        # 2. Lazy evaluation: only filtered rows loaded into memory
        # 3. 2x I/O speedup, 50% memory reduction for filtered queries
        lazy_dfs = [pl.scan_parquet(f) for f in parquet_files]
        result = pl.concat(lazy_dfs)

        # Apply time range filter (pushed down to Parquet)
        if start_ts is not None:
            result = result.filter(pl.col(timestamp_col) >= start_ts)
        if end_ts is not None:
            result = result.filter(pl.col(timestamp_col) <= end_ts)

        # Sort and materialize
        return result.sort(timestamp_col).collect()

    def read_ticks_streaming(  # noqa: PLR0912
        self,
        symbol: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        *,
        chunk_size: int = 100_000,
        timestamp_col: str = "timestamp",
    ) -> Iterator[pl.DataFrame]:
        """Read tick data in streaming chunks to avoid OOM on large months.

        This method yields chunks of tick data instead of loading everything
        into memory at once. Essential for high-volume months like March 2024.

        Parameters
        ----------
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
        >>> storage = TickStorage()
        >>> for chunk in storage.read_ticks_streaming("BTCUSDT", start_ts, end_ts):
        ...     process_chunk(chunk)
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return

        # Find relevant parquet files
        parquet_files = sorted(symbol_dir.glob("*.parquet"))

        if not parquet_files:
            return

        # Filter files by month if time range specified
        if start_ts is not None and end_ts is not None:
            start_month = self._timestamp_to_year_month(start_ts)
            end_month = self._timestamp_to_year_month(end_ts)

            parquet_files = [
                f for f in parquet_files if start_month <= f.stem <= end_month
            ]

        if not parquet_files:
            return

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
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        min_coverage: float = 0.95,
        timestamp_col: str = "timestamp",
    ) -> bool:
        """Check if tick data exists for the specified time range.

        Parameters
        ----------
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
        tick_data = self.read_ticks(
            symbol, start_ts, end_ts, timestamp_col=timestamp_col
        )

        if tick_data.is_empty():
            return False

        actual_start = tick_data[timestamp_col].min()
        actual_end = tick_data[timestamp_col].max()

        if actual_start is None or actual_end is None:
            return False

        actual_range = actual_end - actual_start
        requested_range = end_ts - start_ts

        if requested_range == 0:
            return len(tick_data) > 0

        coverage = actual_range / requested_range
        return coverage >= min_coverage

    def list_symbols(self) -> list[str]:
        """List all symbols with stored tick data.

        Returns
        -------
        list[str]
            List of symbol names
        """
        if not self._ticks_dir.exists():
            return []

        return sorted(
            d.name for d in self._ticks_dir.iterdir() if d.is_dir() and d.name != ""
        )

    def list_months(self, symbol: str) -> list[str]:
        """List all months with stored tick data for a symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        list[str]
            List of year-month strings (e.g., ["2024-01", "2024-02"])
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return []

        return sorted(f.stem for f in symbol_dir.glob("*.parquet"))

    def delete_ticks(self, symbol: str, year_month: str | None = None) -> bool:
        """Delete tick data for a symbol or specific month.

        Parameters
        ----------
        symbol : str
            Trading symbol
        year_month : str | None
            Specific month to delete (e.g., "2024-01"), or None for all

        Returns
        -------
        bool
            True if files were deleted
        """
        if year_month is not None:
            # Delete specific month
            parquet_path = self._get_parquet_path(symbol, year_month)
            if parquet_path.exists():
                parquet_path.unlink()
                return True
            return False

        # Delete all data for symbol
        symbol_dir = self._get_symbol_dir(symbol)
        if symbol_dir.exists():
            shutil.rmtree(symbol_dir)
            return True
        return False

    def get_stats(self, symbol: str) -> dict:
        """Get storage statistics for a symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        dict
            Statistics including file count, total size, row count, date range
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return {
                "symbol": symbol,
                "exists": False,
                "file_count": 0,
                "total_size_bytes": 0,
                "total_rows": 0,
            }

        parquet_files = list(symbol_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        total_rows = 0
        months = []

        for f in parquet_files:
            file_data = pl.read_parquet(f)
            total_rows += len(file_data)
            months.append(f.stem)

        return {
            "symbol": symbol,
            "exists": True,
            "file_count": len(parquet_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "total_rows": total_rows,
            "months": sorted(months),
            "compression": f"{COMPRESSION}-{COMPRESSION_LEVEL}",
        }

    def fetch_month(
        self,
        symbol: str,
        year: int,
        month: int,
        *,
        timestamp_col: str = "timestamp",  # noqa: ARG002 - reserved for filtering
        force_refresh: bool = False,
    ) -> pl.LazyFrame:
        """Fetch tick data for a specific month (lazy loading).

        Returns a LazyFrame for memory efficiency. If data is not cached,
        this method does NOT automatically download from source - use
        `get_range_bars()` or manual fetching first.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT" or "BINANCE_SPOT_BTCUSDT")
        year : int
            Year (e.g., 2024)
        month : int
            Month (1-12)
        timestamp_col : str
            Name of the timestamp column
        force_refresh : bool
            If True, skip cache and return empty LazyFrame (caller must fetch)

        Returns
        -------
        pl.LazyFrame
            Lazy frame for the month's tick data, or empty LazyFrame if not cached

        Examples
        --------
        >>> storage = TickStorage()
        >>> lf = storage.fetch_month("BTCUSDT", 2024, 1)
        >>> df = lf.collect()  # Materialize when needed
        """
        year_month = f"{year}-{month:02d}"

        if force_refresh:
            return pl.LazyFrame()

        parquet_path = self._get_parquet_path(symbol, year_month)

        if not parquet_path.exists():
            return pl.LazyFrame()

        return pl.scan_parquet(parquet_path)

    def fetch_date_range(
        self,
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
        >>> storage = TickStorage()
        >>> for lf in storage.fetch_date_range("BTCUSDT", "2024-01-01", "2024-03-31"):
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

            lf = self.fetch_month(symbol, year, month, timestamp_col=timestamp_col)

            # Only yield non-empty LazyFrames
            # Note: We can't easily check if LazyFrame is empty without collecting
            # So we yield all and let caller handle empty results
            parquet_path = self._get_parquet_path(symbol, f"{year}-{month:02d}")
            if parquet_path.exists():
                yield lf

            # Move to next month
            _december = 12
            if current.month == _december:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
