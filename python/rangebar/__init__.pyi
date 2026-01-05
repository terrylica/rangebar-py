"""Type stubs for rangebar package."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import pandas as pd

__version__: str

# ============================================================================
# ClickHouse Cache Types
# ============================================================================

class InstallationLevel(IntEnum):
    """ClickHouse installation state levels."""

    ENV_NOT_CONFIGURED = -1
    NOT_INSTALLED = 0
    INSTALLED_NOT_RUNNING = 1
    RUNNING_NO_SCHEMA = 2
    FULLY_CONFIGURED = 3

@dataclass
class PreflightResult:
    """Result of preflight detection."""

    level: InstallationLevel
    version: str | None = None
    binary_path: str | None = None
    message: str = ""
    action_required: str | None = None

@dataclass
class HostConnection:
    """Connection details for a ClickHouse host."""

    host: str
    method: Literal["local", "direct", "ssh_tunnel"]
    port: int = 8123

@dataclass(frozen=True)
class CacheKey:
    """Cache key for range bar lookups."""

    symbol: str
    threshold_bps: int
    start_ts: int
    end_ts: int

    @property
    def hash_key(self) -> str: ...

class ClickHouseNotConfiguredError(RuntimeError):
    """Raised when no ClickHouse hosts configured and localhost unavailable."""

    def __init__(self, message: str | None = None) -> None: ...

class RangeBarCache:
    """Two-tier ClickHouse cache for range bars."""

    def __init__(self, client: object | None = None) -> None:
        """Initialize cache with ClickHouse connection."""

    def close(self) -> None:
        """Close client and tunnel if owned."""

    def store_raw_trades(self, symbol: str, trades: pd.DataFrame) -> int:
        """Store raw trades in cache."""

    def get_raw_trades(self, symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Get raw trades from cache."""

    def has_raw_trades(
        self, symbol: str, start_ts: int, end_ts: int, min_coverage: float = 0.95
    ) -> bool:
        """Check if raw trades exist for time range."""

    def store_range_bars(
        self, key: CacheKey, bars: pd.DataFrame, version: str = ""
    ) -> int:
        """Store computed range bars in cache."""

    def get_range_bars(self, key: CacheKey) -> pd.DataFrame | None:
        """Get cached range bars."""

    def has_range_bars(self, key: CacheKey) -> bool:
        """Check if range bars exist in cache."""

    def invalidate_range_bars(self, key: CacheKey) -> int:
        """Invalidate (delete) cached range bars."""

    def __enter__(self) -> RangeBarCache: ...
    def __exit__(self, *args: object) -> None: ...

def detect_clickhouse_state() -> PreflightResult:
    """Detect ClickHouse installation state on localhost."""

def get_available_clickhouse_host() -> HostConnection:
    """Find best available ClickHouse host."""

class RangeBarProcessor:
    """Process tick-level trade data into range bars."""

    threshold_bps: int

    def __init__(self, threshold_bps: int) -> None:
        """Initialize processor with given threshold.

        Parameters
        ----------
        threshold_bps : int
            Threshold in 0.1 basis point units (250 = 25bps = 0.25%)

        Raises
        ------
        ValueError
            If threshold_bps is out of valid range [1, 100_000]
        """

    def process_trades(
        self, trades: list[dict[str, int | float]]
    ) -> list[dict[str, str | float | int]]:
        """Process trades into range bars.

        Parameters
        ----------
        trades : List[Dict]
            List of trade dictionaries with required keys:
            timestamp, price, quantity (or volume)

        Returns
        -------
        List[Dict]
            List of range bar dictionaries

        Raises
        ------
        KeyError
            If required trade fields are missing
        RuntimeError
            If trades are not sorted chronologically
        """

    def to_dataframe(self, bars: list[dict[str, str | float | int]]) -> pd.DataFrame:
        """Convert range bars to pandas DataFrame.

        Parameters
        ----------
        bars : List[Dict]
            List of range bar dictionaries from process_trades()

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and OHLCV columns
        """

def process_trades_to_dataframe(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Convenience function to process trades directly to DataFrame.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns/keys: timestamp, price, quantity
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    Raises
    ------
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If trades are not sorted chronologically
    """

def process_trades_to_dataframe_cached(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    symbol: str,
    threshold_bps: int = 250,
    cache: RangeBarCache | None = None,
) -> pd.DataFrame:
    """Process trades to DataFrame with two-tier ClickHouse caching.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns/keys: timestamp, price, quantity
    symbol : str
        Trading symbol (e.g., "BTCUSDT"). Used as cache key.
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)
    cache : RangeBarCache | None
        External cache instance. If None, creates one (preflight runs).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    Raises
    ------
    ClickHouseNotConfiguredError
        If no ClickHouse hosts available
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If trades are not sorted chronologically
    """
