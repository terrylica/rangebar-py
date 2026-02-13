# Issue #46: Modularization - Extract maintenance operations from cache.py
"""Table maintenance operations for ClickHouse range bar cache.

Provides mixin methods for deduplication and merge monitoring.
Used by RangeBarCache via mixin inheritance.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MaintenanceMixin:
    """Mixin for table maintenance operations (dedup, merge monitoring).

    Requires `self.client` from ClickHouseClientMixin.
    """

    def count_duplicates(
        self,
        symbol: str | None = None,
        threshold_decimal_bps: int | None = None,
    ) -> list[dict]:
        """Count duplicate rows by timestamp (same ORDER BY key).

        ReplacingMergeTree deduplicates during background merges, not at insert
        time. This method identifies rows that will be deduplicated on next merge.

        Parameters
        ----------
        symbol : str | None
            Filter to specific symbol, or None for all symbols
        threshold_decimal_bps : int | None
            Filter to specific threshold, or None for all thresholds

        Returns
        -------
        list[dict]
            List of dicts with keys: symbol, threshold_decimal_bps, timestamp_ms,
            duplicate_count, opens, closes. Empty list if no duplicates found.

        Examples
        --------
        >>> with RangeBarCache() as cache:
        ...     dupes = cache.count_duplicates("BTCUSDT", 1000)
        ...     print(f"Found {len(dupes)} timestamp(s) with duplicates")
        """
        where_clauses = []
        params = {}

        if symbol is not None:
            where_clauses.append("symbol = {symbol:String}")
            params["symbol"] = symbol

        if threshold_decimal_bps is not None:
            where_clauses.append("threshold_decimal_bps = {threshold:UInt32}")
            params["threshold"] = threshold_decimal_bps

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT
                symbol,
                threshold_decimal_bps,
                timestamp_ms,
                count(*) as duplicate_count,
                groupArray(open) as opens,
                groupArray(close) as closes
            FROM rangebar_cache.range_bars
            {where_sql}
            GROUP BY symbol, threshold_decimal_bps, timestamp_ms
            HAVING count(*) > 1
            ORDER BY symbol, threshold_decimal_bps, timestamp_ms
        """

        result = self.client.query(query, parameters=params)
        return [
            {
                "symbol": row[0],
                "threshold_decimal_bps": row[1],
                "timestamp_ms": row[2],
                "duplicate_count": row[3],
                "opens": row[4],
                "closes": row[5],
            }
            for row in result.result_rows
        ]

    def deduplicate_bars(
        self,
        symbol: str | None = None,
        threshold_decimal_bps: int | None = None,
        timeout: int = 600,
    ) -> None:
        """Force deduplication via OPTIMIZE TABLE FINAL with timeout resilience.

        Issue #90: Rewritten as fire-and-forget pattern. OPTIMIZE commands are
        submitted with short client-side timeouts. If the client times out,
        the merge continues server-side. Progress is monitored via system.merges.

        ReplacingMergeTree only deduplicates during background merges. This
        method forces an immediate merge to remove duplicate rows (same
        symbol + threshold + timestamp_ms), keeping the row with the latest
        computed_at timestamp.

        Parameters
        ----------
        symbol : str | None
            Optimize only partitions for this symbol, or None for all
        threshold_decimal_bps : int | None
            Optimize only partitions for this threshold, or None for all
        timeout : int
            Maximum seconds to wait for merges to complete (default: 600).
            After timeout, merges continue in background on the server.

        Examples
        --------
        >>> with RangeBarCache() as cache:
        ...     # Check for duplicates first
        ...     dupes = cache.count_duplicates("BTCUSDT", 1000)
        ...     if dupes:
        ...         print(f"Found {len(dupes)} duplicates, deduplicating...")
        ...         cache.deduplicate_bars("BTCUSDT", 1000)
        """
        if symbol is not None and threshold_decimal_bps is not None:
            logger.info(
                "Deduplicating bars for %s @ %d dbps (OPTIMIZE FINAL)",
                symbol,
                threshold_decimal_bps,
            )
            # Get list of partitions for this symbol/threshold
            partition_query = """
                SELECT DISTINCT partition
                FROM system.parts
                WHERE database = 'rangebar_cache'
                  AND table = 'range_bars'
                  AND active = 1
                  AND partition LIKE {pattern:String}
            """
            pattern = f"('{symbol}',{threshold_decimal_bps},%"
            result = self.client.query(partition_query, parameters={"pattern": pattern})

            for row in result.result_rows:
                partition_id = row[0]
                optimize_sql = (
                    f"OPTIMIZE TABLE rangebar_cache.range_bars "
                    f"PARTITION {partition_id} FINAL"
                )
                try:
                    self.client.command(
                        optimize_sql,
                        settings={
                            "send_timeout": 60,
                            "receive_timeout": 60,
                        },
                    )
                    logger.debug("Optimized partition %s", partition_id)
                except (OSError, RuntimeError) as e:
                    # Client timeout — OPTIMIZE continues server-side
                    logger.info(
                        "OPTIMIZE timed out client-side for partition %s, "
                        "merge continues on server: %s",
                        partition_id,
                        e,
                    )
        else:
            # Optimize entire table
            logger.info("Deduplicating all bars (OPTIMIZE TABLE FINAL)")
            try:
                self.client.command(
                    "OPTIMIZE TABLE rangebar_cache.range_bars FINAL",
                    settings={
                        "send_timeout": 60,
                        "receive_timeout": 60,
                    },
                )
            except (OSError, RuntimeError) as e:
                logger.info(
                    "Full-table OPTIMIZE timed out client-side, "
                    "merge continues on server: %s",
                    e,
                )

        # Poll system.merges until our table's merges complete
        self._wait_for_merges(timeout=timeout)
        logger.info("Deduplication complete")

    def _wait_for_merges(self, timeout: int = 600, poll_interval: int = 10) -> None:
        """Poll system.merges until no active merges for our table.

        Issue #90: After submitting OPTIMIZE, poll ClickHouse to wait for
        background merges to finish. If timeout is reached, log a warning
        and return — merges continue server-side.

        Parameters
        ----------
        timeout : int
            Maximum seconds to wait (default: 600).
        poll_interval : int
            Seconds between polls (default: 10).
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.client.query(
                "SELECT count() FROM system.merges "
                "WHERE database = 'rangebar_cache' AND table = 'range_bars'"
            )
            active = result.result_rows[0][0] if result.result_rows else 0
            if active == 0:
                logger.info("All merges complete")
                return
            logger.debug("Waiting for %d active merge(s)...", active)
            time.sleep(poll_interval)
        logger.warning(
            "Merge wait timed out after %ds, merges continue in background",
            timeout,
        )
