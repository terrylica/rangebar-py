# Issue #46: Modularization - Extract checkpoint operations from cache.py
"""Population checkpoint operations for ClickHouse range bar cache.

Provides mixin methods for saving, loading, and deleting population
checkpoints. Used by RangeBarCache via mixin inheritance.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CheckpointOperationsMixin:
    """Mixin for population checkpoint and bar deletion operations.

    Requires `self.client` from ClickHouseClientMixin.
    """

    def delete_bars(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_ts: int,
        end_ts: int,
        *,
        ouroboros_mode: str,  # Issue #126: MANDATORY
    ) -> int:
        """Delete bars in timestamp range (for force_refresh).

        This is an alias for invalidate_range_bars_by_range() with clearer
        naming for the force_refresh use case.

        Issue #126: ouroboros_mode is mandatory to prevent cross-mode deletion.

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
        ouroboros_mode : str
            Ouroboros mode filter — only delete bars with this mode.

        Returns
        -------
        int
            Always returns 0 (ClickHouse DELETE is async)
        """
        return self.invalidate_range_bars_by_range(
            symbol, threshold_decimal_bps, start_ts, end_ts,
            ouroboros_mode=ouroboros_mode,
        )

    def save_checkpoint(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_date: str,
        end_date: str,
        last_completed_date: str,
        last_trade_timestamp_ms: int | None,
        processor_checkpoint: str,
        bars_written: int,
        include_microstructure: bool = False,
        ouroboros_mode: str = "year",
        # Issue #72: Full Audit Trail - agg_trade_id range in incomplete bar
        first_agg_trade_id_in_bar: int | None = None,
        last_agg_trade_id_in_bar: int | None = None,
        # Issue #111 (Ariadne): High-water mark for deterministic resume
        last_processed_agg_trade_id: int | None = None,
    ) -> None:
        """Save population checkpoint to ClickHouse.

        Used for cross-machine resume support. Local checkpoints are
        saved separately via filesystem for faster same-machine resume.

        Parameters
        ----------
        symbol : str
            Trading symbol
        threshold_decimal_bps : int
            Threshold in decimal basis points
        start_date : str
            Population start date (YYYY-MM-DD)
        end_date : str
            Population end date (YYYY-MM-DD)
        last_completed_date : str
            Last fully processed date (YYYY-MM-DD)
        last_trade_timestamp_ms : int | None
            Timestamp of last processed trade (for mid-day resume)
        processor_checkpoint : str
            JSON-serialized processor state (incomplete bar, defer_open, etc.)
        bars_written : int
            Total bars written so far
        include_microstructure : bool
            Whether microstructure features are enabled
        ouroboros_mode : str
            Ouroboros reset mode
        """
        # Ensure table exists (in case schema wasn't updated)
        self._ensure_population_checkpoints_table()

        query = """
            INSERT INTO rangebar_cache.population_checkpoints
            (symbol, threshold_decimal_bps, start_date, end_date,
             last_completed_date, last_trade_timestamp_ms, bars_written,
             processor_checkpoint, include_microstructure, ouroboros_mode,
             first_agg_trade_id_in_bar, last_agg_trade_id_in_bar,
             last_processed_agg_trade_id,
             updated_at)
            VALUES
            ({symbol:String}, {threshold:UInt32}, {start_date:String},
             {end_date:String}, {last_date:String}, {last_ts:Int64},
             {bars:UInt64}, {checkpoint:String}, {micro:UInt8},
             {ouroboros:String}, {first_agg:Int64}, {last_agg:Int64},
             {hwm_agg:Int64},
             now64(3))
        """
        self.client.command(
            query,
            parameters={
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
                "start_date": start_date,
                "end_date": end_date,
                "last_date": last_completed_date,
                "last_ts": last_trade_timestamp_ms or 0,
                "bars": bars_written,
                "checkpoint": processor_checkpoint,
                "micro": 1 if include_microstructure else 0,
                "ouroboros": ouroboros_mode,
                "first_agg": first_agg_trade_id_in_bar or 0,  # Issue #72
                "last_agg": last_agg_trade_id_in_bar or 0,    # Issue #72
                "hwm_agg": last_processed_agg_trade_id or 0,  # Issue #111
            },
        )
        logger.debug(
            "Saved checkpoint for %s @ %d dbps: %s (%d bars)",
            symbol,
            threshold_decimal_bps,
            last_completed_date,
            bars_written,
        )

    def load_checkpoint(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_date: str,
        end_date: str,
        ouroboros_mode: str = "year",
    ) -> dict | None:
        """Load population checkpoint from ClickHouse.

        Used for cross-machine resume when local checkpoint is not available.

        Parameters
        ----------
        symbol : str
            Trading symbol
        threshold_decimal_bps : int
            Threshold in decimal basis points
        start_date : str
            Population start date (YYYY-MM-DD)
        end_date : str
            Population end date (YYYY-MM-DD)
        ouroboros_mode : str
            Ouroboros reset mode filter (default: "year").
            Prevents cross-mode checkpoint pollution.

        Returns
        -------
        dict | None
            Checkpoint data if found, None otherwise.
            Keys: last_completed_date, last_trade_timestamp_ms,
                  processor_checkpoint, bars_written, include_microstructure,
                  ouroboros_mode
        """
        query = """
            SELECT
                last_completed_date,
                last_trade_timestamp_ms,
                processor_checkpoint,
                bars_written,
                include_microstructure,
                ouroboros_mode,
                first_agg_trade_id_in_bar,
                last_agg_trade_id_in_bar,
                last_processed_agg_trade_id
            FROM rangebar_cache.population_checkpoints FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
              AND start_date = {start_date:String}
              AND end_date = {end_date:String}
              AND ouroboros_mode = {ouroboros:String}
            LIMIT 1
        """
        try:
            result = self.client.query(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "start_date": start_date,
                    "end_date": end_date,
                    "ouroboros": ouroboros_mode,
                },
            )
        except (OSError, RuntimeError) as e:
            # Table might not exist yet
            logger.debug("Checkpoint load failed: %s", e)
            return None

        rows = result.result_rows
        if not rows:
            return None

        row = rows[0]
        return {
            "last_completed_date": row[0],
            "last_trade_timestamp_ms": row[1] if row[1] > 0 else None,
            "processor_checkpoint": row[2] if row[2] else None,
            "bars_written": row[3],
            "include_microstructure": bool(row[4]),
            "ouroboros_mode": row[5],
            # Issue #72 / #111: Trade ID tracking
            "first_agg_trade_id_in_bar": row[6] if row[6] > 0 else None,
            "last_agg_trade_id_in_bar": row[7] if row[7] > 0 else None,
            "last_processed_agg_trade_id": row[8] if row[8] > 0 else None,
        }

    def delete_checkpoint(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_date: str,
        end_date: str,
        ouroboros_mode: str = "year",
    ) -> None:
        """Delete population checkpoint from ClickHouse.

        Used by force_refresh to clear checkpoint before repopulating.

        Parameters
        ----------
        symbol : str
            Trading symbol
        threshold_decimal_bps : int
            Threshold in decimal basis points
        start_date : str
            Population start date (YYYY-MM-DD)
        end_date : str
            Population end date (YYYY-MM-DD)
        ouroboros_mode : str
            Ouroboros reset mode filter (default: "year").
        """
        query = """
            ALTER TABLE rangebar_cache.population_checkpoints
            DELETE WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
              AND start_date = {start_date:String}
              AND end_date = {end_date:String}
              AND ouroboros_mode = {ouroboros:String}
        """
        try:
            self.client.command(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "start_date": start_date,
                    "end_date": end_date,
                    "ouroboros": ouroboros_mode,
                },
            )
            logger.debug(
                "Deleted checkpoint for %s @ %d dbps (%s to %s)",
                symbol,
                threshold_decimal_bps,
                start_date,
                end_date,
            )
        except (OSError, RuntimeError) as e:
            # Table might not exist yet - that's fine
            logger.debug("Checkpoint delete failed (table may not exist): %s", e)

    def _ensure_population_checkpoints_table(self) -> None:
        """Ensure population_checkpoints table exists.

        Called before checkpoint operations to handle upgrades from
        older installations without the table.
        """
        create_sql = """
            CREATE TABLE IF NOT EXISTS rangebar_cache.population_checkpoints (
                symbol LowCardinality(String),
                threshold_decimal_bps UInt32,
                start_date String,
                end_date String,
                last_completed_date String,
                last_trade_timestamp_ms Int64,
                bars_written UInt64,
                processor_checkpoint String DEFAULT '',
                first_agg_trade_id_in_bar Int64 DEFAULT 0,
                last_agg_trade_id_in_bar Int64 DEFAULT 0,
                last_processed_agg_trade_id Int64 DEFAULT 0,
                include_microstructure UInt8 DEFAULT 0,
                ouroboros_mode LowCardinality(String) DEFAULT 'year',
                created_at DateTime64(3) DEFAULT now64(3),
                updated_at DateTime64(3) DEFAULT now64(3)
            )
            ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY (symbol, threshold_decimal_bps,
                     start_date, end_date, ouroboros_mode)
        """
        try:
            self.client.command(create_sql)
        except (OSError, RuntimeError) as e:
            # Table already exists or other error - log and continue
            logger.debug("Checkpoints table creation: %s", e)
