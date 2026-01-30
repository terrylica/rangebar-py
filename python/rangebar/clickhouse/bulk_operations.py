# Issue #46: Modularization M5 - Extract bulk operations from cache.py
"""Bulk store operations for ClickHouse range bar cache.

Provides mixin methods for storing range bars in bulk (pandas) and batch
(Polars/Arrow) modes. Used by RangeBarCache via mixin inheritance.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import pandas as pd

from .._core import __version__
from ..constants import EXCHANGE_SESSION_COLUMNS, MICROSTRUCTURE_COLUMNS
from ..exceptions import CacheWriteError

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class BulkStoreMixin:
    """Mixin providing bulk store operations for RangeBarCache.

    Requires `self.client` from ClickHouseClientMixin.
    """

    def store_bars_bulk(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        bars: pd.DataFrame,
        version: str | None = None,
        ouroboros_mode: str = "year",
    ) -> int:
        """Store bars without requiring CacheKey (for bar-count API).

        This method is for storing bars computed during gap-filling
        where we don't have exact date bounds.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        bars : pd.DataFrame
            DataFrame with OHLCV columns (from rangebar processing)
        version : str | None
            rangebar-core version for cache invalidation. If None (default),
            uses current package version for schema evolution tracking.
        ouroboros_mode : str
            Ouroboros reset mode: "year", "month", or "week" (default: "year")

        Returns
        -------
        int
            Number of rows inserted

        Raises
        ------
        CacheWriteError
            If the insert operation fails.
        """
        if bars.empty:
            logger.debug("Skipping bulk cache write for %s: empty DataFrame", symbol)
            return 0

        logger.debug(
            "Bulk writing %d bars to cache for %s @ %d dbps",
            len(bars),
            symbol,
            threshold_decimal_bps,
        )

        df = bars.copy()

        # Handle DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "timestamp" in df.columns:
                df["timestamp_ms"] = df["timestamp"].astype("int64") // 10**6
                df = df.drop(columns=["timestamp"])
            elif "index" in df.columns:
                df["timestamp_ms"] = df["index"].astype("int64") // 10**6
                df = df.drop(columns=["index"])

        # Normalize column names (lowercase)
        df.columns = df.columns.str.lower()

        # Add cache metadata (Ouroboros: Plan sparkling-coalescing-dijkstra.md)
        df["symbol"] = symbol
        df["threshold_decimal_bps"] = threshold_decimal_bps
        df["ouroboros_mode"] = ouroboros_mode
        df["rangebar_version"] = version if version is not None else __version__

        # For bulk storage without CacheKey, use timestamp range as source bounds
        if "timestamp_ms" in df.columns and len(df) > 0:
            df["source_start_ts"] = df["timestamp_ms"].min()
            df["source_end_ts"] = df["timestamp_ms"].max()
            # Generate cache_key from symbol, threshold, ouroboros, and timestamp range
            start_ts = df["source_start_ts"].iloc[0]
            end_ts = df["source_end_ts"].iloc[0]
            key_str = (
                f"{symbol}_{threshold_decimal_bps}_{start_ts}_{end_ts}_{ouroboros_mode}"
            )
            df["cache_key"] = hashlib.md5(key_str.encode()).hexdigest()
        else:
            df["source_start_ts"] = 0
            df["source_end_ts"] = 0
            df["cache_key"] = ""

        # Select columns for insertion
        columns = [
            "symbol",
            "threshold_decimal_bps",
            "ouroboros_mode",
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cache_key",
            "rangebar_version",
            "source_start_ts",
            "source_end_ts",
        ]

        # Add optional microstructure columns if present (from constants.py SSoT)
        for col in MICROSTRUCTURE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Add optional exchange session columns if present (Issue #8)
        # Cast numpy.bool_ to int for ClickHouse Nullable(UInt8) (Issue #50)
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(int)
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        try:
            summary = self.client.insert_df(
                "rangebar_cache.range_bars",
                df[columns],
            )
            written = summary.written_rows
            logger.info(
                "Bulk cached %d bars for %s @ %d dbps",
                written,
                symbol,
                threshold_decimal_bps,
            )
            return written
        except (OSError, RuntimeError) as e:
            logger.exception(
                "Bulk cache write failed for %s @ %d dbps",
                symbol,
                threshold_decimal_bps,
            )
            msg = f"Failed to bulk write bars for {symbol}: {e}"
            raise CacheWriteError(
                msg,
                symbol=symbol,
                operation="bulk_write",
            ) from e

    def store_bars_batch(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        bars: pl.DataFrame,
        version: str | None = None,
    ) -> int:
        """Store a batch of bars using Arrow for efficient streaming writes.

        This method is optimized for incremental streaming cache writes
        (Phase 4.3). It uses Arrow for zero-copy data transfer to ClickHouse.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        bars : pl.DataFrame
            Polars DataFrame with OHLCV columns (from streaming processing)
        version : str | None
            rangebar-core version for cache invalidation. If None (default),
            uses current package version for schema evolution tracking.

        Returns
        -------
        int
            Number of rows inserted

        Examples
        --------
        >>> from rangebar.clickhouse import RangeBarCache
        >>> with RangeBarCache() as cache:
        ...     # Stream bars and write incrementally
        ...     for batch in stream_range_bars("BTCUSDT", "2024-01-01", "2024-01-07"):
        ...         written = cache.store_bars_batch(
        ...             "BTCUSDT", 250, batch, version="7.1.3"
        ...         )
        ...         print(f"Wrote {written} bars")
        """
        import polars as pl

        if bars.is_empty():
            return 0

        # Normalize column names (lowercase)
        df = bars.rename({c: c.lower() for c in bars.columns if c != c.lower()})

        # Handle timestamp conversion from datetime to milliseconds
        if "timestamp" in df.columns:
            # Check if it's already datetime or string
            if df["timestamp"].dtype == pl.Datetime:
                df = df.with_columns(
                    (pl.col("timestamp").dt.epoch(time_unit="ms"))
                    .cast(pl.Int64)
                    .alias("timestamp_ms")
                ).drop("timestamp")
            elif df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("timestamp")
                    .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z")
                    .dt.epoch(time_unit="ms")
                    .cast(pl.Int64)
                    .alias("timestamp_ms")
                ).drop("timestamp")

        # Add cache metadata (ouroboros_mode defaults to "year" for batch storage)
        # Schema evolution: use __version__ if version not specified
        effective_version = version if version is not None else __version__
        df = df.with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(threshold_decimal_bps).alias("threshold_decimal_bps"),
            pl.lit("year").alias("ouroboros_mode"),  # Default for batch storage
            pl.lit(effective_version).alias("rangebar_version"),
        )

        # Add source bounds and cache_key
        if "timestamp_ms" in df.columns and len(df) > 0:
            start_ts = df["timestamp_ms"].min()
            end_ts = df["timestamp_ms"].max()
            key_str = f"{symbol}_{threshold_decimal_bps}_{start_ts}_{end_ts}_year"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            df = df.with_columns(
                pl.lit(start_ts).alias("source_start_ts"),
                pl.lit(end_ts).alias("source_end_ts"),
                pl.lit(cache_key).alias("cache_key"),
            )
        else:
            df = df.with_columns(
                pl.lit(0).alias("source_start_ts"),
                pl.lit(0).alias("source_end_ts"),
                pl.lit("").alias("cache_key"),
            )

        # Define columns for insertion
        columns = [
            "symbol",
            "threshold_decimal_bps",
            "ouroboros_mode",
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cache_key",
            "rangebar_version",
            "source_start_ts",
            "source_end_ts",
        ]

        # Add optional microstructure columns if present (from constants.py SSoT)
        for col in MICROSTRUCTURE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Add optional exchange session columns if present (Issue #8)
        # Cast bool to UInt8 for ClickHouse Nullable(UInt8) (Issue #50)
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.UInt8))
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        # Use Arrow for efficient insert (zero-copy)
        arrow_table = df.select(columns).to_arrow()
        summary = self.client.insert_arrow(
            "rangebar_cache.range_bars",
            arrow_table,
        )

        return summary.written_rows
