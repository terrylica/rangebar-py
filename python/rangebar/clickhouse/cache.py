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
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

import pandas as pd

from .client import get_client
from .mixin import ClickHouseClientMixin
from .preflight import (
    HostConnection,
    get_available_clickhouse_host,
)
from .tunnel import SSHTunnel

if TYPE_CHECKING:
    import clickhouse_connect


@dataclass(frozen=True)
class CacheKey:
    """Cache key for range bar lookups.

    Uniquely identifies a set of computed range bars based on:
    - Symbol (e.g., "BTCUSDT")
    - Threshold in basis points
    - Time range of source data

    Attributes
    ----------
    symbol : str
        Trading symbol
    threshold_decimal_bps : int
        Threshold in decimal basis points (0.1bps = 0.001%)
    start_ts : int
        Start timestamp in milliseconds
    end_ts : int
        End timestamp in milliseconds
    """

    symbol: str
    threshold_decimal_bps: int
    start_ts: int
    end_ts: int

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
            f"{self.start_ts}_{self.end_ts}"
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
        version: str = "",
    ) -> int:
        """Store computed range bars in cache.

        Parameters
        ----------
        key : CacheKey
            Cache key identifying these bars
        bars : pd.DataFrame
            DataFrame with OHLCV columns (from rangebar processing)
        version : str
            rangebar-core version for cache invalidation

        Returns
        -------
        int
            Number of rows inserted
        """
        if bars.empty:
            return 0

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
        df["cache_key"] = key.hash_key
        df["rangebar_version"] = version
        df["source_start_ts"] = key.start_ts
        df["source_end_ts"] = key.end_ts

        # Select columns for insertion
        columns = [
            "symbol",
            "threshold_decimal_bps",
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

        # Add optional microstructure columns if present
        optional = [
            "vwap",
            "buy_volume",
            "sell_volume",
            "individual_trade_count",
            "agg_record_count",
        ]
        for col in optional:
            if col in df.columns:
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        self.client.insert_df(
            "rangebar_cache.range_bars",
            df[columns],
        )

        return len(df)

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
        """
        query = """
            SELECT
                timestamp_ms,
                open as Open,
                high as High,
                low as Low,
                close as Close,
                volume as Volume
            FROM rangebar_cache.range_bars
            WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold_decimal_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
            ORDER BY timestamp_ms
        """
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

        if df.empty:
            return None

        # Convert to DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])

        # Convert PyArrow dtypes to numpy for compatibility with fresh computation
        # query_df_arrow returns double[pyarrow], but process_trades returns float64
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = df[col].astype("float64")

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
            FROM rangebar_cache.range_bars
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
                FROM rangebar_cache.range_bars
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
                FROM rangebar_cache.range_bars
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
            sell_volume
        """

        if before_ts is not None:
            # Split path: with end_ts filter
            query = f"""
                SELECT {base_cols}
                FROM rangebar_cache.range_bars
                WHERE symbol = {{symbol:String}}
                  AND threshold_decimal_bps = {{threshold:UInt32}}
                  AND timestamp_ms < {{end_ts:Int64}}
                ORDER BY timestamp_ms DESC
                LIMIT {{n_bars:UInt64}}
            """
            df = self.client.query_df_arrow(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "end_ts": before_ts,
                    "n_bars": n_bars,
                },
            )
        else:
            # Split path: no end_ts filter (most recent)
            query = f"""
                SELECT {base_cols}
                FROM rangebar_cache.range_bars
                WHERE symbol = {{symbol:String}}
                  AND threshold_decimal_bps = {{threshold:UInt32}}
                ORDER BY timestamp_ms DESC
                LIMIT {{n_bars:UInt64}}
            """
            df = self.client.query_df_arrow(
                query,
                parameters={
                    "symbol": symbol,
                    "threshold": threshold_decimal_bps,
                    "n_bars": n_bars,
                },
            )

        if df.empty:
            return None, available_count

        # Reverse to chronological order (oldest first)
        df = df.iloc[::-1].reset_index(drop=True)

        # Convert to DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])

        # Convert PyArrow dtypes to numpy for compatibility
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = df[col].astype("float64")

        # Convert microstructure columns if present
        if include_microstructure:
            for col in ["vwap", "buy_volume", "sell_volume"]:
                if col in df.columns:
                    df[col] = df[col].astype("float64")

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
            FROM rangebar_cache.range_bars
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
            FROM rangebar_cache.range_bars
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

    def store_bars_bulk(
        self,
        symbol: str,
        threshold_decimal_bps: int,
        bars: pd.DataFrame,
        version: str = "",
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
        version : str
            rangebar-core version for cache invalidation

        Returns
        -------
        int
            Number of rows inserted
        """
        if bars.empty:
            return 0

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
        df["symbol"] = symbol
        df["threshold_decimal_bps"] = threshold_decimal_bps
        df["rangebar_version"] = version

        # For bulk storage without CacheKey, use timestamp range as source bounds
        if "timestamp_ms" in df.columns and len(df) > 0:
            df["source_start_ts"] = df["timestamp_ms"].min()
            df["source_end_ts"] = df["timestamp_ms"].max()
            # Generate cache_key from symbol, threshold, and timestamp range
            start_ts = df["source_start_ts"].iloc[0]
            end_ts = df["source_end_ts"].iloc[0]
            key_str = f"{symbol}_{threshold_decimal_bps}_{start_ts}_{end_ts}"
            df["cache_key"] = hashlib.md5(key_str.encode()).hexdigest()
        else:
            df["source_start_ts"] = 0
            df["source_end_ts"] = 0
            df["cache_key"] = ""

        # Select columns for insertion
        columns = [
            "symbol",
            "threshold_decimal_bps",
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

        # Add optional microstructure columns if present
        optional = [
            "vwap",
            "buy_volume",
            "sell_volume",
            "individual_trade_count",
            "agg_record_count",
        ]
        for col in optional:
            if col in df.columns:
                columns.append(col)

        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]

        self.client.insert_df(
            "rangebar_cache.range_bars",
            df[columns],
        )

        return len(df)
