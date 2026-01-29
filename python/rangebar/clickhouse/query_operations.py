# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M6 - Extract query operations from cache.py
"""Query operations for ClickHouse range bar cache.

Provides mixin methods for retrieving range bars by count (get_n_bars)
and by timestamp range (get_bars_by_timestamp_range). Used by RangeBarCache
via mixin inheritance.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..constants import (
    MICROSTRUCTURE_COLUMNS,
    MIN_VERSION_FOR_MICROSTRUCTURE,
)
from ..conversion import normalize_arrow_dtypes
from ..exceptions import CacheReadError

logger = logging.getLogger(__name__)


class QueryOperationsMixin:
    """Mixin providing query operations for RangeBarCache.

    Requires `self.client` and `self.count_bars()` from parent class.
    """

    def get_n_bars(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        n_bars: int,
        before_ts: int | None = None,
        include_microstructure: bool = False,
        min_schema_version: str | None = None,
    ) -> tuple[pd.DataFrame | None, int]:
        """Get N bars from cache, ordered chronologically (oldest first).

        Uses ORDER BY timestamp_ms DESC LIMIT N for efficient retrieval,
        then reverses in Python for chronological order.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        n_bars : int
            Maximum number of bars to retrieve
        before_ts : int | None
            Only get bars with timestamp_ms < before_ts.
            If None, gets most recent bars.
        include_microstructure : bool
            If True, includes vwap, buy_volume, sell_volume columns
        min_schema_version : str | None
            Minimum schema version required for cache hit. If specified,
            only returns data with rangebar_version >= min_schema_version.
            When include_microstructure=True and min_schema_version=None,
            automatically requires version >= 7.0.0.

        Returns
        -------
        tuple[pd.DataFrame | None, int]
            (bars_df, available_count) where:
            - bars_df is OHLCV DataFrame (or None if no bars)
            - available_count is total bars available (may be > len(bars_df))
        """
        # First get the count (for reporting)
        available_count = self.count_bars(symbol, threshold_decimal_bps, before_ts)

        if available_count == 0:
            return None, 0

        # Select columns
        base_cols = """
            timestamp_ms,
            open as Open,
            high as High,
            low as Low,
            close as Close,
            volume as Volume
        """
        if include_microstructure:
            base_cols += """,
            vwap,
            buy_volume,
            sell_volume,
            duration_us,
            ofi,
            vwap_close_deviation,
            price_impact,
            kyle_lambda_proxy,
            trade_intensity,
            volume_per_trade,
            aggression_ratio,
            aggregation_density,
            turnover_imbalance
        """

        # Determine effective min version for schema evolution filtering
        effective_min_version = min_schema_version
        if include_microstructure and effective_min_version is None:
            effective_min_version = MIN_VERSION_FOR_MICROSTRUCTURE

        # Build version filter if specified
        version_filter = ""
        if effective_min_version:
            version_filter = """
              AND rangebar_version != ''
              AND rangebar_version >= {min_version:String}"""

        if before_ts is not None:
            # Split path: with end_ts filter
            query = f"""
                SELECT {base_cols}
                FROM rangebar_cache.range_bars FINAL
                WHERE symbol = {{symbol:String}}
                  AND threshold_decimal_bps = {{threshold:UInt32}}
                  AND timestamp_ms < {{end_ts:Int64}}
                  {version_filter}
                ORDER BY timestamp_ms DESC
                LIMIT {{n_bars:UInt64}}
            """
            params: dict[str, str | int] = {
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
                "end_ts": before_ts,
                "n_bars": n_bars,
            }
            if effective_min_version:
                params["min_version"] = effective_min_version
            df = self.client.query_df_arrow(query, parameters=params)
        else:
            # Split path: no end_ts filter (most recent)
            query = f"""
                SELECT {base_cols}
                FROM rangebar_cache.range_bars FINAL
                WHERE symbol = {{symbol:String}}
                  AND threshold_decimal_bps = {{threshold:UInt32}}
                  {version_filter}
                ORDER BY timestamp_ms DESC
                LIMIT {{n_bars:UInt64}}
            """
            params = {
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
                "n_bars": n_bars,
            }
            if effective_min_version:
                params["min_version"] = effective_min_version
            df = self.client.query_df_arrow(query, parameters=params)

        if df.empty:
            return None, available_count

        # Reverse to chronological order (oldest first)
        df = df.iloc[::-1].reset_index(drop=True)

        # Convert to TZ-aware UTC DatetimeIndex (Issue #20: match get_range_bars output)
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])

        # Convert PyArrow dtypes to numpy for compatibility
        df = normalize_arrow_dtypes(df)

        # Convert microstructure columns if present
        if include_microstructure:
            df = normalize_arrow_dtypes(df, columns=list(MICROSTRUCTURE_COLUMNS))

        return df, available_count

    def get_bars_by_timestamp_range(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_ts: int,
        end_ts: int,
        include_microstructure: bool = False,
        include_exchange_sessions: bool = False,
        ouroboros_mode: str = "year",
        min_schema_version: str | None = None,
    ) -> pd.DataFrame | None:
        """Get bars within a timestamp range (for get_range_bars cache lookup).

        Unlike get_range_bars() which requires exact CacheKey match,
        this method queries by timestamp range, returning any cached bars
        that fall within [start_ts, end_ts].

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        start_ts : int
            Start timestamp in milliseconds (inclusive)
        end_ts : int
            End timestamp in milliseconds (inclusive)
        include_microstructure : bool
            If True, includes vwap, buy_volume, sell_volume columns
        include_exchange_sessions : bool
            If True, includes exchange_session_* columns (Issue #8)
        ouroboros_mode : str
            Ouroboros reset mode: "year", "month", or "week" (default: "year")
            Plan: sparkling-coalescing-dijkstra.md
        min_schema_version : str | None
            Minimum schema version required for cache hit. If specified,
            only returns data with rangebar_version >= min_schema_version.
            When include_microstructure=True and min_schema_version=None,
            automatically requires version >= 7.0.0.

        Returns
        -------
        pd.DataFrame | None
            OHLCV DataFrame with TZ-aware UTC timestamps if found, None otherwise.
            Returns None if no bars exist in the range or version mismatch.

        Raises
        ------
        CacheReadError
            If the query fails due to database errors.
        """
        # Build column list
        base_cols = """
            timestamp_ms,
            open as Open,
            high as High,
            low as Low,
            close as Close,
            volume as Volume
        """
        if include_microstructure:
            base_cols += """,
            vwap,
            buy_volume,
            sell_volume,
            duration_us,
            ofi,
            vwap_close_deviation,
            price_impact,
            kyle_lambda_proxy,
            trade_intensity,
            volume_per_trade,
            aggression_ratio,
            aggregation_density,
            turnover_imbalance
        """

        # Issue #8: Exchange session flags
        if include_exchange_sessions:
            base_cols += """,
            exchange_session_sydney,
            exchange_session_tokyo,
            exchange_session_london,
            exchange_session_newyork
        """

        # Ouroboros mode filter ensures cache isolation between modes
        # Plan: sparkling-coalescing-dijkstra.md

        # Determine effective min version for schema evolution filtering
        effective_min_version = min_schema_version
        if include_microstructure and effective_min_version is None:
            # Auto-require v7.0.0+ for microstructure features
            effective_min_version = MIN_VERSION_FOR_MICROSTRUCTURE

        # Build version filter if specified
        version_filter = ""
        if effective_min_version:
            version_filter = """
              AND rangebar_version != ''
              AND rangebar_version >= {min_version:String}"""

        query = f"""
            SELECT {base_cols}
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {{symbol:String}}
              AND threshold_decimal_bps = {{threshold:UInt32}}
              AND ouroboros_mode = {{ouroboros_mode:String}}
              AND timestamp_ms >= {{start_ts:Int64}}
              AND timestamp_ms <= {{end_ts:Int64}}
              {version_filter}
            ORDER BY timestamp_ms
        """

        # Build parameters
        params: dict[str, str | int] = {
            "symbol": symbol,
            "threshold": threshold_decimal_bps,
            "ouroboros_mode": ouroboros_mode,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }
        if effective_min_version:
            params["min_version"] = effective_min_version

        try:
            df = self.client.query_df_arrow(query, parameters=params)
        except (OSError, RuntimeError) as e:
            logger.exception(
                "Cache read failed for %s @ %d dbps (range query)",
                symbol,
                threshold_decimal_bps,
            )
            msg = f"Failed to read bars for {symbol}: {e}"
            raise CacheReadError(
                msg,
                symbol=symbol,
                operation="read_range",
            ) from e

        if df.empty:
            logger.debug(
                "Cache miss for %s @ %d dbps (range: %d-%d)",
                symbol,
                threshold_decimal_bps,
                start_ts,
                end_ts,
            )
            return None

        logger.debug(
            "Cache hit: %d bars for %s @ %d dbps (range query)",
            len(df),
            symbol,
            threshold_decimal_bps,
        )

        # Convert to TZ-aware UTC DatetimeIndex (matches get_range_bars output)
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])

        # Convert PyArrow dtypes to numpy float64 for compatibility
        df = normalize_arrow_dtypes(df)

        if include_microstructure:
            df = normalize_arrow_dtypes(df, columns=list(MICROSTRUCTURE_COLUMNS))

        return df
