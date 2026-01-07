"""Type stubs for rangebar package."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal, TypedDict

import pandas as pd
import polars as pl

__version__: str

# ============================================================================
# Exness Types (feature-gated)
# ============================================================================

class ExnessInstrument(IntEnum):
    """Exness forex instruments (Raw_Spread variant).

    10 instruments validated against ~/eon/exness-data-preprocess (453M+ ticks).
    """

    EURUSD = 0
    GBPUSD = 1
    USDJPY = 2
    AUDUSD = 3
    USDCAD = 4
    NZDUSD = 5
    EURGBP = 6
    EURJPY = 7
    GBPJPY = 8
    XAUUSD = 9

    @property
    def symbol(self) -> str: ...
    @property
    def raw_spread_symbol(self) -> str: ...
    @property
    def is_jpy_pair(self) -> bool: ...
    @staticmethod
    def all() -> list[ExnessInstrument]: ...

class ValidationStrictness(IntEnum):
    """Validation strictness level for Exness tick processing."""

    Permissive = 0  # Basic checks only (bid > 0, ask > 0, bid < ask)
    Strict = 1  # + Spread < 10% [DEFAULT]
    Paranoid = 2  # + Spread < 1%

class SpreadStatsDict(TypedDict):
    """Spread statistics dictionary."""

    min_spread: float
    max_spread: float
    avg_spread: float
    tick_count: int

class ExnessRangeBarBuilder:
    """Builder for processing Exness tick data into range bars."""

    threshold_bps: int
    instrument: ExnessInstrument

    def __init__(
        self,
        instrument: ExnessInstrument,
        threshold_bps: int,
        strictness: ValidationStrictness = ...,
    ) -> None:
        """Initialize builder for instrument.

        Parameters
        ----------
        instrument : ExnessInstrument
            Exness instrument enum value
        threshold_bps : int
            Threshold in 0.1 basis point units (250 = 25bps = 0.25%)
        strictness : ValidationStrictness
            Validation strictness level (default: Strict)

        Raises
        ------
        ValueError
            If threshold_bps is out of valid range [1, 100_000]
        """

    def process_tick(
        self, tick: dict[str, float | int]
    ) -> dict[str, str | float | int | SpreadStatsDict] | None:
        """Process a single tick.

        Parameters
        ----------
        tick : Dict
            Tick dictionary with keys: bid, ask, timestamp_ms

        Returns
        -------
        Dict or None
            Range bar dictionary if bar completed, None if still accumulating

        Raises
        ------
        KeyError
            If required tick fields are missing
        RuntimeError
            If tick validation fails (crossed market, excessive spread)
        """

    def process_ticks(
        self, ticks: list[dict[str, float | int]]
    ) -> list[dict[str, str | float | int | SpreadStatsDict]]:
        """Process multiple ticks at once.

        Parameters
        ----------
        ticks : List[Dict]
            List of tick dictionaries with keys: bid, ask, timestamp_ms

        Returns
        -------
        List[Dict]
            List of completed range bar dictionaries
        """

    def get_incomplete_bar(
        self,
    ) -> dict[str, str | float | int | SpreadStatsDict] | None:
        """Get incomplete bar if exists.

        Returns
        -------
        Dict or None
            Partial bar dictionary if bar in progress, None otherwise
        """

def process_exness_ticks_to_dataframe(
    ticks: pd.DataFrame | Sequence[dict[str, float | int]],
    instrument: ExnessInstrument,
    threshold_bps: int = 250,
    strictness: ValidationStrictness | None = None,
) -> pd.DataFrame:
    """Process Exness tick data to range bars DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame or Sequence[Dict]
        Tick data with columns/keys: bid, ask, timestamp_ms
    instrument : ExnessInstrument
        Exness instrument enum value
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)
    strictness : ValidationStrictness, optional
        Validation strictness level (default: Strict)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex, compatible with backtesting.py.
        Additional column: spread_stats (dict with min/max/avg spread)

    Raises
    ------
    ImportError
        If Exness feature is not enabled
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If tick validation fails (crossed market, excessive spread)
    """

def is_exness_available() -> bool:
    """Check if Exness feature is available.

    Returns
    -------
    bool
        True if Exness bindings are available, False otherwise
    """

# ============================================================================
# ClickHouse Cache Types
# ============================================================================

class ConnectionMode(str, Enum):
    """Connection mode for ClickHouse cache."""

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"

def get_connection_mode() -> ConnectionMode:
    """Get the connection mode from RANGEBAR_MODE environment variable."""

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
    """ClickHouse cache for computed range bars.

    For raw tick data storage, use `rangebar.storage.TickStorage` instead.
    """

    def __init__(self, client: object | None = None) -> None:
        """Initialize cache with ClickHouse connection."""

    def close(self) -> None:
        """Close client and tunnel if owned."""

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

# ============================================================================
# Optimized Processing APIs
# ============================================================================

def process_trades_chunked(
    trades_iterator: Iterator[dict[str, int | float]],
    threshold_bps: int = 250,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Process trades in chunks to avoid memory spikes.

    Parameters
    ----------
    trades_iterator : Iterator[Dict]
        Iterator yielding trade dicts with keys: timestamp, price, quantity
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)
    chunk_size : int, default=100_000
        Number of trades per chunk

    Yields
    ------
    pd.DataFrame
        OHLCV bars for each chunk (partial bars may occur at boundaries)
    """

def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Process trades from Polars DataFrame (optimized pipeline).

    Parameters
    ----------
    trades : polars.DataFrame or polars.LazyFrame
        Trade data with columns: timestamp (int64 ms), price, quantity
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py
    """
