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
from ..constants import (
    _PLUGIN_FEATURE_COLUMNS,  # Issue #98: FeatureProvider plugin columns
    EXCHANGE_SESSION_COLUMNS,
    INTER_BAR_FEATURE_COLUMNS,  # Issue #78: Was missing, causing NULL lookback columns
    INTRA_BAR_FEATURE_COLUMNS,  # Issue #78: Also add intra-bar features
    MICROSTRUCTURE_COLUMNS,
    TRADE_ID_RANGE_COLUMNS,  # Issue #72
)
from ..exceptions import CacheWriteError
from ..hooks import HookEvent, emit_hook

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def _guard_timestamp_ms_scale(df: object, column: str = "timestamp_ms") -> None:
    """Raise if timestamp_ms contains seconds instead of milliseconds (#85)."""
    if column in (df.columns if hasattr(df, "columns") else []):
        min_ts = df[column].min()
        if min_ts < 1_000_000_000_000:
            msg = (
                f"timestamp_ms in seconds, not ms (min={min_ts}). "
                f"See Issue #85."
            )
            raise ValueError(msg)


_VOLUME_COLUMNS = ("volume", "buy_volume", "sell_volume")


def _guard_volume_non_negative(
    df: object,
    columns: tuple[str, ...] = _VOLUME_COLUMNS,
) -> None:
    """Reject writes with negative volumes (overflow indicator). Issue #88."""
    cols = df.columns if hasattr(df, "columns") else []
    for col in columns:
        if col in cols and df[col].min() < 0:
            msg = (
                f"Issue #88: {col} has negative values "
                f"(min={df[col].min()}). Overflow detected."
            )
            raise ValueError(msg)


def _run_write_guards(df: object) -> None:
    """Run all pre-write data integrity guards."""
    _guard_timestamp_ms_scale(df)
    _guard_volume_non_negative(df)


def _build_insert_settings(
    skip_dedup: bool,
    cache_key_accessor: object = None,
) -> dict[str, object]:
    """Build INSERT dedup settings from cache_key (Issue #90).

    Returns settings dict with insert_deduplication_token if dedup is enabled
    and a valid cache_key exists. Returns empty dict otherwise.
    """
    if skip_dedup or cache_key_accessor is None:
        return {}
    token = cache_key_accessor()
    if token:
        return {
            "insert_deduplicate": 1,
            "insert_deduplication_token": token,
        }
    return {}


class BulkStoreMixin:
    """Mixin providing bulk store operations for RangeBarCache.

    Requires `self.client` from ClickHouseClientMixin.
    """

    def store_bars_bulk(  # noqa: PLR0915
        self,
        symbol: str,
        threshold_decimal_bps: int,
        bars: pd.DataFrame,
        version: str | None = None,
        ouroboros_mode: str = "year",
        *,
        skip_dedup: bool = False,
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
        skip_dedup : bool
            If True, skip INSERT deduplication token (for force_refresh).
            Default False enables idempotent INSERTs (Issue #90).

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
            return 0

        # Issue #48: Emit CACHE_WRITE_START hook
        emit_hook(
            HookEvent.CACHE_WRITE_START, symbol=symbol,
            bars_count=len(bars), threshold_decimal_bps=threshold_decimal_bps,
        )
        logger.debug(
            "Bulk writing %d bars to cache for %s @ %d dbps",
            len(bars), symbol, threshold_decimal_bps,
        )

        # R1 (Issue #98): reset_index() already creates a new DataFrame,
        # so the prior .copy() was a redundant full-DataFrame copy.
        if isinstance(bars.index, pd.DatetimeIndex):
            df = bars.reset_index()  # creates new DF (no prior .copy() needed)
            if "timestamp" in df.columns:
                df["timestamp_ms"] = df["timestamp"].dt.as_unit("ms").astype("int64")
                df = df.drop(columns=["timestamp"])
            elif "index" in df.columns:
                df["timestamp_ms"] = df["index"].dt.as_unit("ms").astype("int64")
                df = df.drop(columns=["index"])
        else:
            df = bars.copy()

        _run_write_guards(df)

        # Normalize column names (lowercase)
        df.columns = df.columns.str.lower()

        # Add cache metadata (Ouroboros: Plan sparkling-coalescing-dijkstra.md)
        df["symbol"] = symbol
        df["threshold_decimal_bps"] = threshold_decimal_bps
        df["ouroboros_mode"] = ouroboros_mode
        df["rangebar_version"] = version if version is not None else __version__

        # For bulk storage without CacheKey, use timestamp range as source bounds
        if "timestamp_ms" in df.columns and len(df) > 0:
            start_ts = df["timestamp_ms"].min()
            end_ts = df["timestamp_ms"].max()
            df["source_start_ts"] = start_ts
            df["source_end_ts"] = end_ts
            # Generate cache_key from symbol, threshold, ouroboros, and timestamp range
            key_str = (
                f"{symbol}_{threshold_decimal_bps}"
                f"_{start_ts}_{end_ts}_{ouroboros_mode}"
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

        # Add optional columns if present (from constants.py SSoT)
        for col in (*MICROSTRUCTURE_COLUMNS, *TRADE_ID_RANGE_COLUMNS):
            if col in df.columns:
                columns.append(col)

        # Add exchange session columns; cast bool_ to int (Issue #8, #50)
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(int)
                columns.append(col)

        # Add inter-bar and intra-bar feature columns if present (Issue #78)
        # These are Nullable in ClickHouse; clickhouse-connect's insert_df
        # requires NaN (not Python None) for Nullable numeric columns.
        # R2 (Issue #98): Skip pd.to_numeric when dtype is already float64.
        for col in (*INTER_BAR_FEATURE_COLUMNS, *INTRA_BAR_FEATURE_COLUMNS):
            if col in df.columns:
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                columns.append(col)

        # Add plugin feature columns if present (Issue #98)
        for col in _PLUGIN_FEATURE_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        # Issue #90: INSERT dedup token for idempotent writes.
        insert_settings = _build_insert_settings(
            skip_dedup,
            (lambda: df["cache_key"].iloc[0])
            if df.get("cache_key") is not None and len(df) > 0
            else None,
        )

        try:
            written = self.client.insert_df(
                "rangebar_cache.range_bars",
                df[columns],
                settings=insert_settings,
            ).written_rows
            logger.info(
                "Bulk cached %d bars for %s @ %d dbps",
                written, symbol, threshold_decimal_bps,
            )
            emit_hook(
                HookEvent.CACHE_WRITE_COMPLETE, symbol=symbol,
                bars_written=written, threshold_decimal_bps=threshold_decimal_bps,
            )
            return written
        except (OSError, RuntimeError) as e:
            logger.exception(
                "Bulk cache write failed for %s @ %d dbps",
                symbol, threshold_decimal_bps,
            )
            emit_hook(
                HookEvent.CACHE_WRITE_FAILED, symbol=symbol,
                error=str(e), threshold_decimal_bps=threshold_decimal_bps,
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
        ouroboros_mode: str = "year",
        *,
        skip_dedup: bool = False,
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
        ouroboros_mode : str
            Ouroboros reset mode: "year", "month", or "week" (default: "year")
        skip_dedup : bool
            If True, skip INSERT deduplication token (for force_refresh).
            Default False enables idempotent INSERTs (Issue #90).

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
        # Issue #96 Task #34: Fast-path skip rename when all columns lowercase
        if any(c != c.lower() for c in bars.columns):
            df = bars.rename({c: c.lower() for c in bars.columns if c != c.lower()})
        else:
            df = bars

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

        _run_write_guards(df)

        # Add cache metadata
        # Schema evolution: use __version__ if version not specified
        effective_version = version if version is not None else __version__
        df = df.with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(threshold_decimal_bps).alias("threshold_decimal_bps"),
            pl.lit(ouroboros_mode).alias("ouroboros_mode"),
            pl.lit(effective_version).alias("rangebar_version"),
        )

        # Add source bounds and cache_key
        if "timestamp_ms" in df.columns and len(df) > 0:
            start_ts = df["timestamp_ms"].min()
            end_ts = df["timestamp_ms"].max()
            key_str = (
                f"{symbol}_{threshold_decimal_bps}"
                f"_{start_ts}_{end_ts}_{ouroboros_mode}"
            )
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

        # Add trade ID range columns if present (Issue #72)
        for col in TRADE_ID_RANGE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Add inter-bar feature columns if present (Issue #78: was missing)
        for col in INTER_BAR_FEATURE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Add intra-bar feature columns if present (Issue #78)
        for col in INTRA_BAR_FEATURE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Add plugin feature columns if present (Issue #98)
        for col in _PLUGIN_FEATURE_COLUMNS:
            if col in df.columns:
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        # Issue #90: INSERT dedup token for idempotent writes.
        insert_settings = _build_insert_settings(
            skip_dedup,
            (lambda: df["cache_key"][0])
            if "cache_key" in df.columns and len(df) > 0
            else None,
        )

        # Use Arrow for efficient insert (zero-copy)
        arrow_table = df.select(columns).to_arrow()
        summary = self.client.insert_arrow(
            "rangebar_cache.range_bars",
            arrow_table,
            settings=insert_settings,
        )

        return summary.written_rows
