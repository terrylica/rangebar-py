"""ClickHouse cache for computed range bars.

This module provides the RangeBarCache class that caches computed range bars
(Tier 2) in ClickHouse. Raw tick data is stored locally using Parquet files
via the `rangebar.storage` module.

Architecture:
- Tier 1 (raw ticks): Local Parquet files via `rangebar.storage.TickStorage`
- Tier 2 (computed bars): ClickHouse via this module

The cache uses mise environment variables for configuration and
supports multiple GPU workstations via SSH aliases.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

import pandas as pd

from .._core import __version__
from ..constants import (
    EXCHANGE_SESSION_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    MIN_VERSION_FOR_MICROSTRUCTURE,
)
from ..conversion import normalize_arrow_dtypes
from ..exceptions import (
    CacheReadError,
    CacheWriteError,
)
from .client import get_client
from .mixin import ClickHouseClientMixin
from .preflight import (
    HostConnection,
    get_available_clickhouse_host,
)
from .tunnel import SSHTunnel

if TYPE_CHECKING:
    import clickhouse_connect
    import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """Cache key for range bar lookups.

    Uniquely identifies a set of computed range bars based on:
    - Symbol (e.g., "BTCUSDT")
    - Threshold in decimal basis points (dbps)
    - Time range of source data
    - Ouroboros mode for reproducibility

    Attributes
    ----------
    symbol : str
        Trading symbol
    threshold_decimal_bps : int
        Threshold in decimal basis points (1 dbps = 0.001%)
    start_ts : int
        Start timestamp in milliseconds
    end_ts : int
        End timestamp in milliseconds
    ouroboros_mode : str
        Ouroboros reset mode: "year", "month", or "week" (default: "year")
    """

    symbol: str
    threshold_decimal_bps: int
    start_ts: int
    end_ts: int
    ouroboros_mode: str = "year"

    @property
    def hash_key(self) -> str:
        """Get hash key for cache lookups.

        Returns
        -------
        str
            MD5 hash of cache key components
        """
        key_str = (
            f"{self.symbol}_{self.threshold_decimal_bps}_"
            f"{self.start_ts}_{self.end_ts}_{self.ouroboros_mode}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()


class RangeBarCache(ClickHouseClientMixin):
    """ClickHouse cache for computed range bars.

    Caches computed range bars (Tier 2) in ClickHouse. For raw tick data
    storage (Tier 1), use `rangebar.storage.TickStorage` instead.

    PREFLIGHT runs in __init__ - fails loudly if no ClickHouse available.

    Parameters
    ----------
    client : Client | None
        External ClickHouse client. If None, creates connection based on
        environment configuration (mise env vars).

    Raises
    ------
    ClickHouseNotConfiguredError
        If no ClickHouse hosts available (with guidance for setup)

    Examples
    --------
    >>> with RangeBarCache() as cache:
    ...     # Check cache
    ...     if cache.has_range_bars(key):
    ...         bars = cache.get_range_bars(key)
    ...     else:
    ...         # Compute bars and store
    ...         cache.store_range_bars(key, bars)

    See Also
    --------
    rangebar.storage.TickStorage : Local Parquet storage for raw tick data
    """

    def __init__(self, client: clickhouse_connect.driver.Client | None = None) -> None:
        """Initialize cache with ClickHouse connection.

        Runs preflight check to find available ClickHouse host.
        Creates schema if it doesn't exist.
        """
        self._tunnel: SSHTunnel | None = None
        self._host_connection: HostConnection | None = None

        if client is None:
            # PREFLIGHT CHECK - runs before anything else
            # This function fails loudly if no CH available
            host_conn = get_available_clickhouse_host()
            self._host_connection = host_conn

            # Connect based on method (local, direct, or SSH tunnel)
            if host_conn.method == "local":
                self._init_client(get_client("localhost", host_conn.port))
            elif host_conn.method == "direct":
                from .preflight import _resolve_ssh_alias_to_ip

                ip = _resolve_ssh_alias_to_ip(host_conn.host)
                if ip is None:
                    msg = f"Could not resolve SSH alias: {host_conn.host}"
                    raise RuntimeError(msg)
                self._init_client(get_client(ip, host_conn.port))
            elif host_conn.method == "ssh_tunnel":
                self._tunnel = SSHTunnel(host_conn.host, host_conn.port)
                local_port = self._tunnel.start()
                self._init_client(get_client("localhost", local_port))
        else:
            self._init_client(client)

        self._ensure_schema()

    def close(self) -> None:
        """Close client and tunnel if owned."""
        super().close()
        if self._tunnel is not None:
            self._tunnel.stop()
            self._tunnel = None

    def _ensure_schema(self) -> None:
        """Create database and tables if they don't exist."""
        # Create database
        self.client.command("CREATE DATABASE IF NOT EXISTS rangebar_cache")

        # Load and execute schema
        schema_sql = resources.files(__package__).joinpath("schema.sql").read_text()

        # Split by semicolon and execute each statement
        for statement in schema_sql.split(";"):
            # Check if statement contains a CREATE TABLE/MATERIALIZED VIEW
            if "CREATE TABLE" in statement or "CREATE MATERIALIZED" in statement:
                # Strip single-line comments from each line, keep the SQL
                lines = []
                for line in statement.split("\n"):
                    line = line.strip()
                    # Skip pure comment lines
                    if line.startswith("--"):
                        continue
                    # Remove trailing comments
                    if "--" in line:
                        line = line[: line.index("--")].strip()
                    if line:
                        lines.append(line)
                clean_sql = " ".join(lines)
                if clean_sql:
                    self.client.command(clean_sql)

    # =========================================================================
    # Range Bars Cache (Tier 2)
    # =========================================================================
    # Note: Raw tick data (Tier 1) is now stored locally using Parquet files.
    # See rangebar.storage.TickStorage for tick data caching.

    def store_range_bars(
        self,
        key: CacheKey,
        bars: pd.DataFrame,
        version: str | None = None,
    ) -> int:
        """Store computed range bars in cache.

        Parameters
        ----------
        key : CacheKey
            Cache key identifying these bars
        bars : pd.DataFrame
            DataFrame with OHLCV columns (from rangebar processing)
        version : str | None
            rangebar-core version for cache invalidation. If None (default),
            uses current package version for schema evolution tracking.

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
            logger.debug("Skipping cache write for %s: empty DataFrame", key.symbol)
            return 0

        logger.debug(
            "Writing %d bars to cache for %s @ %d dbps",
            len(bars),
            key.symbol,
            key.threshold_decimal_bps,
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

        # Add cache metadata
        df["symbol"] = key.symbol
        df["threshold_decimal_bps"] = key.threshold_decimal_bps
        df["ouroboros_mode"] = key.ouroboros_mode
        df["cache_key"] = key.hash_key
        df["rangebar_version"] = version if version is not None else __version__
        df["source_start_ts"] = key.start_ts
        df["source_end_ts"] = key.end_ts

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

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        try:
            summary = self.client.insert_df(
                "rangebar_cache.range_bars",
                df[columns],
            )
            written = summary.written_rows
            logger.info(
                "Cached %d bars for %s @ %d dbps",
                written,
                key.symbol,
                key.threshold_decimal_bps,
            )
            return written
        except (OSError, RuntimeError) as e:
            logger.exception(
                "Cache write failed for %s @ %d dbps",
                key.symbol,
                key.threshold_decimal_bps,
            )
            msg = f"Failed to write bars for {key.symbol}: {e}"
            raise CacheWriteError(
                msg,
                symbol=key.symbol,
                operation="write",
            ) from e

    def get_range_bars(self, key: CacheKey) -> pd.DataFrame | None:
        """Get cached range bars.

        Parameters
        ----------
        key : CacheKey
            Cache key to lookup

        Returns
        -------
        pd.DataFrame | None
            OHLCV DataFrame if found, None otherwise

        Raises
        ------
        CacheReadError
            If the query fails due to database errors.
        """
        query = """
            SELECT
                timestamp_ms,
                open as Open,
                high as High,
                low as Low,
                close as Close,
                volume as Volume
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold_decimal_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
            ORDER BY timestamp_ms
        """
        try:
            # Use Arrow-optimized query for 3x faster DataFrame creation
            df = self.client.query_df_arrow(
                query,
                parameters={
                    "symbol": key.symbol,
                    "threshold_decimal_bps": key.threshold_decimal_bps,
                    "start_ts": key.start_ts,
                    "end_ts": key.end_ts,
                },
            )
        except (OSError, RuntimeError) as e:
            logger.exception(
                "Cache read failed for %s @ %d dbps",
                key.symbol,
                key.threshold_decimal_bps,
            )
            msg = f"Failed to read bars for {key.symbol}: {e}"
            raise CacheReadError(
                msg,
                symbol=key.symbol,
                operation="read",
            ) from e

        if df.empty:
            logger.debug(
                "Cache miss for %s @ %d dbps (key: %s)",
                key.symbol,
                key.threshold_decimal_bps,
                key.hash_key[:8],
            )
            return None

        logger.debug(
            "Cache hit: %d bars for %s @ %d dbps",
            len(df),
            key.symbol,
            key.threshold_decimal_bps,
        )

        # Convert to TZ-aware UTC DatetimeIndex (Issue #20: consistent timestamps)
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])

        # Convert PyArrow dtypes to numpy for compatibility with fresh computation
        # query_df_arrow returns double[pyarrow], but process_trades returns float64
        df = normalize_arrow_dtypes(df)

        return df

    def has_range_bars(self, key: CacheKey) -> bool:
        """Check if range bars exist in cache.

        Parameters
        ----------
        key : CacheKey
            Cache key to check

        Returns
        -------
        bool
            True if bars exist
        """
        query = """
            SELECT count() > 0
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold_decimal_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
            LIMIT 1
        """
        result = self.client.command(
            query,
            parameters={
                "symbol": key.symbol,
                "threshold_decimal_bps": key.threshold_decimal_bps,
                "start_ts": key.start_ts,
                "end_ts": key.end_ts,
            },
        )
        return bool(result)

    def invalidate_range_bars(self, key: CacheKey) -> int:
        """Invalidate (delete) cached range bars.

        Parameters
        ----------
        key : CacheKey
            Cache key to invalidate

        Returns
        -------
        int
            Number of rows deleted
        """
        # Note: ClickHouse DELETE is async via mutations
        query = """
            ALTER TABLE rangebar_cache.range_bars
            DELETE WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold_decimal_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
        """
        self.client.command(
            query,
            parameters={
                "symbol": key.symbol,
                "threshold_decimal_bps": key.threshold_decimal_bps,
                "start_ts": key.start_ts,
                "end_ts": key.end_ts,
            },
        )
        return 0  # ClickHouse DELETE is async, can't return count

    def invalidate_range_bars_by_range(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
    ) -> int:
        """Invalidate (delete) cached bars within a timestamp range.

        Unlike `invalidate_range_bars()` which requires an exact CacheKey match,
        this method deletes all bars for a symbol/threshold within a time range.
        Useful for overlap detection during precomputation.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        start_timestamp_ms : int
            Start timestamp in milliseconds (inclusive)
        end_timestamp_ms : int
            End timestamp in milliseconds (inclusive)

        Returns
        -------
        int
            Always returns 0 (ClickHouse DELETE is async via mutations)

        Examples
        --------
        >>> cache.invalidate_range_bars_by_range(
        ...     "BTCUSDT", 250,
        ...     start_timestamp_ms=1704067200000,  # 2024-01-01
        ...     end_timestamp_ms=1706745600000,    # 2024-01-31
        ... )
        """
        query = """
            ALTER TABLE rangebar_cache.range_bars
            DELETE WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
              AND timestamp_ms >= {start_ts:Int64}
              AND timestamp_ms <= {end_ts:Int64}
        """
        self.client.command(
            query,
            parameters={
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
                "start_ts": start_timestamp_ms,
                "end_ts": end_timestamp_ms,
            },
        )
        return 0  # ClickHouse DELETE is async, can't return count

    def get_last_bar_before(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        before_timestamp_ms: int,
    ) -> dict | None:
        """Get the last bar before a given timestamp.

        Useful for validating continuity at junction points during
        incremental precomputation.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        threshold_decimal_bps : int
            Threshold in decimal basis points
        before_timestamp_ms : int
            Timestamp boundary in milliseconds

        Returns
        -------
        dict | None
            Last bar as dict with OHLCV keys, or None if no bars found

        Examples
        --------
        >>> bar = cache.get_last_bar_before("BTCUSDT", 250, 1704067200000)
        >>> if bar:
        ...     print(f"Last close: {bar['Close']}")
        """
        query = """
            SELECT timestamp_ms, open, high, low, close, volume
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
              AND timestamp_ms < {before_ts:Int64}
            ORDER BY timestamp_ms DESC
            LIMIT 1
        """
        result = self.client.query(
            query,
            parameters={
                "symbol": symbol,
                "threshold": threshold_decimal_bps,
                "before_ts": before_timestamp_ms,
            },
        )

        rows = result.result_rows
        if not rows:
            return None

        row = rows[0]
        return {
            "timestamp_ms": row[0],
            "Open": row[1],
            "High": row[2],
            "Low": row[3],
            "Close": row[4],
            "Volume": row[5],
        }

    # =========================================================================
    # Bar-Count-Based API (get_n_range_bars support)
    # =========================================================================
    # These methods support retrieving a deterministic number of bars
    # regardless of time bounds. Uses split query paths to avoid OR conditions.

    def count_bars(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        before_ts: int | None = None,
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
            Only count bars with timestamp_ms < before_ts.
            If None, counts all bars for symbol/threshold.

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
                  AND timestamp_ms < {end_ts:Int64}
            """
            result = self.client.command(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "end_ts": before_ts,
                },
            )
        else:
            # Split path: no end_ts filter (most recent)
            query = """
                SELECT count()
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
            )
        return int(result) if result else 0

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
            SELECT min(timestamp_ms)
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
            SELECT max(timestamp_ms)
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
        )
        # ClickHouse returns 0 for max() on empty result
        return int(result) if result and result > 0 else None

    def get_bars_by_timestamp_range(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        start_ts: int,
        end_ts: int,
        include_microstructure: bool = False,
        include_exchange_sessions: bool = False,  # Issue #8
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
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
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
        for col in EXCHANGE_SESSION_COLUMNS:
            if col in df.columns:
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
