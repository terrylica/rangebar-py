# FILE-SIZE-OK: bulk store + batch store + mode guard are cohesive
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
    TIMESTAMP_COLUMNS,
    TRADE_ID_RANGE_COLUMNS,  # Issue #72
)
from ..exceptions import CacheWriteError
from ..hooks import HookEvent, emit_hook

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def _guard_close_time_ms_scale(df: object, column: str = "close_time_ms") -> None:
    """Raise if close_time_ms contains seconds instead of milliseconds (#85)."""
    if column in (df.columns if hasattr(df, "columns") else []):
        min_ts = df[column].min()
        if min_ts < 1_000_000_000_000:
            msg = (
                f"close_time_ms in seconds, not ms (min={min_ts}). "
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


def _guard_bar_range_invariant(
    df: object,
    threshold_decimal_bps: int | None = None,
    multiplier: int = 3,
) -> None:
    """Issue #112: Reject writes with bars exceeding N * threshold range.

    Defense-in-depth: Rust should catch oversized bars first, but if a bug
    slips through, Python rejects at the ClickHouse write boundary.

    Args:
        df: DataFrame with Open, High, Low columns (or open, high, low).
        threshold_decimal_bps: Threshold in dbps. If None, skip check.
        multiplier: Safety multiplier (default 3 — generous to avoid false
            positives from rounding; Rust uses 2 in debug_assert).
    """
    if threshold_decimal_bps is None:
        return

    # Determine column names (capitalized or lowercase)
    cols = set(df.columns) if hasattr(df, "columns") else set()
    if {"Open", "High", "Low"}.issubset(cols):
        open_col, high_col, low_col = "Open", "High", "Low"
    elif {"open", "high", "low"}.issubset(cols):
        open_col, high_col, low_col = "open", "high", "low"
    else:
        return  # Cannot validate without OHLC columns

    threshold_ratio = threshold_decimal_bps / 100_000  # dbps → ratio
    max_range_ratio = threshold_ratio * multiplier

    # Check each bar's range
    highs = df[high_col]
    lows = df[low_col]
    opens = df[open_col]
    ranges = (highs - lows) / opens

    if hasattr(ranges, "max"):  # pandas/polars
        max_observed = ranges.max()
        if max_observed > max_range_ratio:
            idx = (
                ranges.idxmax()
                if hasattr(ranges, "idxmax")
                else "unknown"
            )
            msg = (
                f"Issue #112: Oversized bar detected at index {idx}. "
                f"Range ratio {max_observed:.6f} exceeds "
                f"{multiplier}x threshold ({max_range_ratio:.6f}). "
                f"threshold_decimal_bps={threshold_decimal_bps}"
            )
            raise ValueError(msg)


def _run_write_guards(
    df: object,
    threshold_decimal_bps: int | None = None,
) -> None:
    """Run all pre-write data integrity guards."""
    _guard_close_time_ms_scale(df)
    _guard_volume_non_negative(df)
    _guard_bar_range_invariant(df, threshold_decimal_bps)


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

    def _guard_ouroboros_mode_consistency(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        ouroboros_mode: str,
    ) -> None:
        """Guard against mixing ouroboros modes for the same (symbol, threshold) pair.

        Issue #126: ouroboros_mode is now in ORDER BY (Phase 5A migration),
        so cross-mode rows won't merge. This guard adds application-level
        safety to prevent accidental mode mixing during migration.

        Behavior controlled by RANGEBAR_OUROBOROS_GUARD:
        - "strict": raise CacheWriteError on mismatch or connection failure
        - "warn": log warning, allow write
        - "off": skip check entirely (used during migration)
        """
        from rangebar.config import Settings

        guard = Settings.get().population.ouroboros_guard
        if guard == "off":
            logger.debug("Mode consistency check skipped (guard=off)")
            return

        try:
            result = self.client.query(
                """
                SELECT DISTINCT ouroboros_mode
                FROM rangebar_cache.range_bars FINAL
                WHERE symbol = {symbol:String}
                  AND threshold_decimal_bps = {threshold:UInt32}
                LIMIT 2
                """,
                parameters={"symbol": symbol, "threshold": threshold_decimal_bps},
            )
            if result.result_rows:
                existing_mode = result.result_rows[0][0]
                if existing_mode and existing_mode != ouroboros_mode:
                    msg = (
                        f"Cannot write {ouroboros_mode} bars for "
                        f"{symbol}@{threshold_decimal_bps}: "
                        f"existing data uses {existing_mode}. "
                        f"Use force_refresh=True to clear existing data first."
                    )
                    if guard == "warn":
                        logger.warning(msg)
                        return
                    raise CacheWriteError(  # noqa: TRY301
                        msg,
                        symbol=symbol,
                        operation="mode_guard",
                    )
        except CacheWriteError:
            raise
        except (OSError, RuntimeError) as e:
            if guard == "strict":
                msg = f"Cannot verify mode consistency (ClickHouse unreachable): {e}"
                raise CacheWriteError(
                    msg,
                    symbol=symbol,
                    operation="mode_guard",
                ) from e
            # guard == "warn" (only remaining option since "off" returned early)
            logger.warning("Mode consistency check failed: %s", e)

    def store_bars_bulk(  # noqa: PLR0915
        self,
        symbol: str,
        threshold_decimal_bps: int,
        bars: pd.DataFrame,
        version: str | None = None,
        ouroboros_mode: str | None = None,
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
            If the insert operation fails or mode mismatch detected.
        """
        if bars.empty:
            return 0

        # Issue #126: Resolve ouroboros_mode from config if not specified
        if ouroboros_mode is None:
            from rangebar.ouroboros import get_operational_ouroboros_mode

            ouroboros_mode = get_operational_ouroboros_mode()
        logger.info(
            "cache_write: ouroboros_mode=%s symbol=%s threshold=%d bars=%d",
            ouroboros_mode, symbol, threshold_decimal_bps, len(bars),
        )

        # Issue #97: Guard against mixed ouroboros modes
        self._guard_ouroboros_mode_consistency(
            symbol, threshold_decimal_bps, ouroboros_mode,
        )

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
            idx_col = bars.index.name or "index"
            df["close_time_ms"] = df[idx_col].dt.as_unit("ms").astype("int64")
            df = df.drop(columns=[idx_col], errors="ignore")
        elif "close_time_ms" not in bars.columns:
            msg = "DataFrame must have DatetimeIndex or 'close_time_ms' column"
            raise ValueError(msg)
        else:
            df = bars.copy()

        _run_write_guards(df, threshold_decimal_bps)

        # Normalize column names (lowercase)
        df.columns = df.columns.str.lower()

        # Add cache metadata (Ouroboros: Plan sparkling-coalescing-dijkstra.md)
        df["symbol"] = symbol
        df["threshold_decimal_bps"] = threshold_decimal_bps
        df["ouroboros_mode"] = ouroboros_mode
        df["rangebar_version"] = version if version is not None else __version__

        # For bulk storage without CacheKey, use timestamp range as source bounds
        if "close_time_ms" in df.columns and len(df) > 0:
            start_ts = (
                df["open_time_ms"].min()
                if "open_time_ms" in df.columns
                else df["close_time_ms"].min()
            )
            end_ts = df["close_time_ms"].max()
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
            "close_time_ms",
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

        # Issue #96 Task #37: Cache available column set for O(1) membership testing
        # Instead of repeated df.columns lookups (O(n)), use frozenset (O(1))
        available_cols = frozenset(df.columns)

        # Add optional columns if present (from constants.py SSoT)
        # close_time_ms already in base list; open_time_ms via TIMESTAMP_COLUMNS
        all_optional = (
            *MICROSTRUCTURE_COLUMNS,
            *TRADE_ID_RANGE_COLUMNS,
            *TIMESTAMP_COLUMNS,
        )
        for col in all_optional:
            if col in available_cols and col not in columns:
                columns.append(col)

        # Add exchange session columns; cast bool_ to int (Issue #8, #50)
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in available_cols:
                df[col] = df[col].astype(int)
                columns.append(col)

        # Add inter-bar and intra-bar feature columns if present (Issue #78)
        # These are Nullable in ClickHouse; clickhouse-connect's insert_df
        # requires NaN (not Python None) for Nullable numeric columns.
        # R2 (Issue #98): Skip pd.to_numeric when dtype is already float64.
        for col in (*INTER_BAR_FEATURE_COLUMNS, *INTRA_BAR_FEATURE_COLUMNS):
            if col in available_cols:
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                columns.append(col)

        # Add plugin feature columns if present (Issue #98)
        for col in _PLUGIN_FEATURE_COLUMNS:
            if col in available_cols:
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                columns.append(col)

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
        ouroboros_mode: str | None = None,
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

        # Issue #126: Resolve ouroboros_mode from config if not specified
        if ouroboros_mode is None:
            from rangebar.ouroboros import get_operational_ouroboros_mode

            ouroboros_mode = get_operational_ouroboros_mode()
        logger.info(
            "cache_write: ouroboros_mode=%s symbol=%s threshold=%d bars=%d",
            ouroboros_mode, symbol, threshold_decimal_bps, len(bars),
        )

        # Issue #97: Guard against mixed ouroboros modes
        self._guard_ouroboros_mode_consistency(
            symbol, threshold_decimal_bps, ouroboros_mode,
        )

        # Normalize column names (lowercase)
        # Issue #96 Task #34: Fast-path skip rename when all columns lowercase
        if any(c != c.lower() for c in bars.columns):
            df = bars.rename({c: c.lower() for c in bars.columns if c != c.lower()})
        else:
            df = bars

        # Handle timestamp conversion from datetime to milliseconds
        if "timestamp" in df.columns and df["timestamp"].dtype == pl.Datetime:
            df = df.with_columns(
                (pl.col("timestamp").dt.epoch(time_unit="ms"))
                .cast(pl.Int64)
                .alias("close_time_ms")
            ).drop("timestamp")
        elif "close_time_ms" not in df.columns:
            msg = "DataFrame must have 'close_time_ms' column"
            raise ValueError(msg)

        _run_write_guards(df, threshold_decimal_bps)

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
        if "close_time_ms" in df.columns and len(df) > 0:
            start_ts = (
                df["open_time_ms"].min()
                if "open_time_ms" in df.columns
                else df["close_time_ms"].min()
            )
            end_ts = df["close_time_ms"].max()
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
            "close_time_ms",
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
        # close_time_ms already in base list; open_time_ms via TIMESTAMP_COLUMNS
        all_optional = (
            *MICROSTRUCTURE_COLUMNS,
            *TRADE_ID_RANGE_COLUMNS,
            *TIMESTAMP_COLUMNS,
        )
        for col in all_optional:
            if col in df.columns and col not in columns:
                columns.append(col)

        # Add optional exchange session columns if present (Issue #8)
        # Cast bool to UInt8 for ClickHouse Nullable(UInt8) (Issue #50)
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.UInt8))
                columns.append(col)

        # Add inter-bar and intra-bar feature columns if present (Issue #78)
        for col in (*INTER_BAR_FEATURE_COLUMNS, *INTRA_BAR_FEATURE_COLUMNS):
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
