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
from datetime import datetime, timezone
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
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
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

        # Add year_month column for partitioning
        ticks = ticks.with_columns(
            pl.col(timestamp_col)
            .map_elements(self._timestamp_to_year_month, return_dtype=pl.Utf8)
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

    def read_ticks(
        self,
        symbol: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        *,
        timestamp_col: str = "timestamp",
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

        Returns
        -------
        pl.DataFrame
            Tick data filtered by time range

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
