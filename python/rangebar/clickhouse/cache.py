"""Two-tier ClickHouse cache for range bar computation.

This module provides the main RangeBarCache class that implements:
- Tier 1: Raw trades cache (avoid re-downloading)
- Tier 2: Computed range bars cache (avoid re-computing)

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
    threshold_bps : int
        Threshold in 0.1 basis point units
    start_ts : int
        Start timestamp in milliseconds
    end_ts : int
        End timestamp in milliseconds
    """

    symbol: str
    threshold_bps: int
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
        key_str = f"{self.symbol}_{self.threshold_bps}_{self.start_ts}_{self.end_ts}"
        return hashlib.md5(key_str.encode()).hexdigest()


class RangeBarCache(ClickHouseClientMixin):
    """Two-tier ClickHouse cache for range bars.

    Provides caching for:
    - Tier 1: Raw trades (avoid re-downloading from exchange)
    - Tier 2: Computed range bars (avoid re-computing)

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
    ...     # Store trades
    ...     cache.store_raw_trades("BTCUSDT", trades_df)
    ...     # Get trades
    ...     df = cache.get_raw_trades("BTCUSDT", start_ts, end_ts)
    ...     # Check cache
    ...     if cache.has_range_bars(key):
    ...         bars = cache.get_range_bars(key)
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
    # Tier 1: Raw Trades Cache
    # =========================================================================

    def store_raw_trades(self, symbol: str, trades: pd.DataFrame) -> int:
        """Store raw trades in cache.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        trades : pd.DataFrame
            DataFrame with columns: timestamp, price, quantity
            Optional: agg_trade_id, first_trade_id, last_trade_id, is_buyer_maker

        Returns
        -------
        int
            Number of rows inserted
        """
        if trades.empty:
            return 0

        # Normalize column names
        df = trades.copy()

        # Ensure required columns
        if "timestamp" not in df.columns and "timestamp_ms" in df.columns:
            df = df.rename(columns={"timestamp_ms": "timestamp"})

        if "quantity" not in df.columns and "volume" in df.columns:
            df = df.rename(columns={"volume": "quantity"})

        # Convert timestamp to milliseconds if needed
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].astype("int64") // 10**6

        # Prepare data for insertion
        df["symbol"] = symbol
        df = df.rename(columns={"timestamp": "timestamp_ms"})

        # Select columns that exist
        columns = ["symbol", "timestamp_ms", "price", "quantity"]
        optional = [
            "agg_trade_id",
            "first_trade_id",
            "last_trade_id",
            "is_buyer_maker",
        ]
        for col in optional:
            if col in df.columns:
                columns.append(col)

        # Insert using Arrow for performance
        self.client.insert_df(
            "rangebar_cache.raw_trades",
            df[columns],
        )

        return len(df)

    def get_raw_trades(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> pd.DataFrame:
        """Get raw trades from cache.

        Parameters
        ----------
        symbol : str
            Trading symbol
        start_ts : int
            Start timestamp in milliseconds
        end_ts : int
            End timestamp in milliseconds

        Returns
        -------
        pd.DataFrame
            Trades DataFrame with columns matching store_raw_trades() input
        """
        query = """
            SELECT
                timestamp_ms as timestamp,
                price,
                quantity,
                agg_trade_id,
                first_trade_id,
                last_trade_id,
                is_buyer_maker
            FROM rangebar_cache.raw_trades
            WHERE symbol = {symbol:String}
              AND timestamp_ms >= {start_ts:Int64}
              AND timestamp_ms <= {end_ts:Int64}
            ORDER BY timestamp_ms, agg_trade_id
        """
        return self.client.query_df(
            query,
            parameters={"symbol": symbol, "start_ts": start_ts, "end_ts": end_ts},
        )

    def has_raw_trades(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        min_coverage: float = 0.95,
    ) -> bool:
        """Check if raw trades exist for time range.

        Parameters
        ----------
        symbol : str
            Trading symbol
        start_ts : int
            Start timestamp in milliseconds
        end_ts : int
            End timestamp in milliseconds
        min_coverage : float
            Minimum data coverage ratio (default: 95%)

        Returns
        -------
        bool
            True if sufficient data exists
        """
        query = """
            SELECT
                min(timestamp_ms) as min_ts,
                max(timestamp_ms) as max_ts,
                count() as count
            FROM rangebar_cache.raw_trades
            WHERE symbol = {symbol:String}
              AND timestamp_ms >= {start_ts:Int64}
              AND timestamp_ms <= {end_ts:Int64}
        """
        result = self.client.query(
            query,
            parameters={"symbol": symbol, "start_ts": start_ts, "end_ts": end_ts},
        )

        if not result.result_rows:
            return False

        min_ts, max_ts, count = result.result_rows[0]

        if count == 0:
            return False

        # Check coverage: actual range / requested range
        actual_range = max_ts - min_ts if max_ts and min_ts else 0
        requested_range = end_ts - start_ts

        if requested_range == 0:
            return count > 0

        coverage = actual_range / requested_range
        return coverage >= min_coverage

    # =========================================================================
    # Tier 2: Range Bars Cache
    # =========================================================================

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
        df["threshold_bps"] = key.threshold_bps
        df["cache_key"] = key.hash_key
        df["rangebar_version"] = version
        df["source_start_ts"] = key.start_ts
        df["source_end_ts"] = key.end_ts

        # Select columns for insertion
        columns = [
            "symbol",
            "threshold_bps",
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
              AND threshold_bps = {threshold_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
            ORDER BY timestamp_ms
        """
        df = self.client.query_df(
            query,
            parameters={
                "symbol": key.symbol,
                "threshold_bps": key.threshold_bps,
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
              AND threshold_bps = {threshold_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
            LIMIT 1
        """
        result = self.client.command(
            query,
            parameters={
                "symbol": key.symbol,
                "threshold_bps": key.threshold_bps,
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
              AND threshold_bps = {threshold_bps:UInt32}
              AND source_start_ts = {start_ts:Int64}
              AND source_end_ts = {end_ts:Int64}
        """
        self.client.command(
            query,
            parameters={
                "symbol": key.symbol,
                "threshold_bps": key.threshold_bps,
                "start_ts": key.start_ts,
                "end_ts": key.end_ts,
            },
        )
        return 0  # ClickHouse DELETE is async, can't return count
