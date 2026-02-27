# Issue #46: Modularization - Extract bar count operations from cache.py
"""Bar count and timestamp query operations for ClickHouse range bar cache.

Provides mixin methods for counting bars and querying oldest/newest timestamps.
Used by RangeBarCache via mixin inheritance.
"""

from __future__ import annotations

import logging

from .constants import FINAL_READ_SETTINGS

logger = logging.getLogger(__name__)


class BarCountMixin:
    """Mixin for bar count and timestamp query operations.

    Requires `self.client` from ClickHouseClientMixin.
    """

    def count_bars(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        before_ts: int | None = None,
        ouroboros_mode: str = "year",
    ) -> int:
        """Count available bars in cache.

        Uses split query paths to avoid OR conditions for ClickHouse optimization.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        before_ts : int | None
            Only count bars with close_time_ms < before_ts.
            If None, counts all bars for symbol/threshold.
        ouroboros_mode : str
            Ouroboros reset mode filter (default: "year").

        Returns
        -------
        int
            Number of bars in cache
        """
        if before_ts is not None:
            # Split path: with end_ts filter
            query = """
                SELECT count()
                FROM rangebar_cache.range_bars FINAL
                WHERE symbol = {symbol:String}
                  AND threshold_decimal_bps = {threshold:UInt32}
                  AND ouroboros_mode = {ouroboros_mode:String}
                  AND close_time_ms < {end_ts:Int64}
            """
            result = self.client.command(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "ouroboros_mode": ouroboros_mode,
                    "end_ts": before_ts,
                },
                settings=FINAL_READ_SETTINGS,
            )
        else:
            # Split path: no end_ts filter (most recent)
            query = """
                SELECT count()
                FROM rangebar_cache.range_bars FINAL
                WHERE symbol = {symbol:String}
                  AND threshold_decimal_bps = {threshold:UInt32}
                  AND ouroboros_mode = {ouroboros_mode:String}
            """
            result = self.client.command(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "ouroboros_mode": ouroboros_mode,
                },
                settings=FINAL_READ_SETTINGS,
            )
        return int(result) if result else 0

    def get_oldest_bar_timestamp(
        self,
        symbol: str,
        threshold_decimal_bps: int,
    ) -> int | None:
        """Get timestamp of oldest bar in cache.

        Useful for determining how far back we can query.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points

        Returns
        -------
        int | None
            Oldest bar timestamp in milliseconds, or None if no bars
        """
        query = """
            SELECT min(close_time_ms)
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
        """
        result = self.client.command(
            query,
            parameters={
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
            },
            settings=FINAL_READ_SETTINGS,
        )
        # ClickHouse returns 0 for min() on empty result
        return int(result) if result and result > 0 else None

    def get_newest_bar_timestamp(
        self,
        symbol: str,
        threshold_decimal_bps: int,
    ) -> int | None:
        """Get timestamp of newest bar in cache.

        Useful for determining the end of available data.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points

        Returns
        -------
        int | None
            Newest bar timestamp in milliseconds, or None if no bars
        """
        query = """
            SELECT max(close_time_ms)
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
        """
        result = self.client.command(
            query,
            parameters={
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
            },
            settings=FINAL_READ_SETTINGS,
        )
        # ClickHouse returns 0 for max() on empty result
        return int(result) if result and result > 0 else None
