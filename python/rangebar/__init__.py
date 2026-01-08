"""rangebar: Python bindings for range bar construction.

This package provides high-performance range bar construction for cryptocurrency
trading backtesting, with non-lookahead bias guarantees and temporal integrity.

Examples
--------
Basic usage:

>>> from rangebar import process_trades_to_dataframe
>>> import pandas as pd
>>>
>>> # Load Binance aggTrades data
>>> trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
>>>
>>> # Convert to range bars (25 basis points = 0.25%)
>>> df = process_trades_to_dataframe(trades, threshold_bps=250)
>>>
>>> # Use with backtesting.py
>>> from backtesting import Backtest, Strategy
>>> bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
>>> stats = bt.run()
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import polars as pl

from ._core import PyRangeBarProcessor as _PyRangeBarProcessor
from ._core import __version__

# Lazy imports for ClickHouse cache (avoid import-time side effects)
# These are imported when first accessed
_clickhouse_imports_done = False


def _ensure_clickhouse_imports() -> None:
    """Ensure ClickHouse-related imports are done."""
    global _clickhouse_imports_done  # noqa: PLW0603
    if not _clickhouse_imports_done:
        from .clickhouse import (
            CacheKey,
            ClickHouseNotConfiguredError,
            RangeBarCache,
            detect_clickhouse_state,
            get_available_clickhouse_host,
        )

        _clickhouse_imports_done = True


__all__ = [
    "THRESHOLD_MAX",
    "THRESHOLD_MIN",
    "THRESHOLD_PRESETS",
    "TIER1_SYMBOLS",
    "__version__",
    "get_range_bars",
]


class RangeBarProcessor:
    """Process tick-level trade data into range bars.

    Range bars close when price moves ±threshold from the bar's opening price,
    providing market-adaptive time intervals that eliminate arbitrary time-based
    artifacts.

    Parameters
    ----------
    threshold_bps : int
        Threshold in 0.1 basis point units.
        Examples: 250 = 25bps = 0.25%, 100 = 10bps = 0.1%
        Valid range: [1, 100_000] (0.001% to 100%)

    Raises
    ------
    ValueError
        If threshold_bps is out of valid range [1, 100_000]

    Examples
    --------
    Create processor and convert trades to DataFrame:

    >>> processor = RangeBarProcessor(threshold_bps=250)  # 0.25%
    >>> trades = [
    ...     {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    ...     {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    ... ]
    >>> bars = processor.process_trades(trades)
    >>> df = processor.to_dataframe(bars)
    >>> print(df.columns.tolist())
    ['Open', 'High', 'Low', 'Close', 'Volume']

    Notes
    -----
    Non-lookahead bias guarantee:
    - Thresholds computed ONLY from bar open price (never recalculated)
    - Breaching trade INCLUDED in closing bar
    - Breaching trade also OPENS next bar

    Temporal integrity:
    - All trades processed in strict chronological order
    - Unsorted trades raise RuntimeError
    """

    def __init__(self, threshold_bps: int) -> None:
        """Initialize processor with given threshold.

        Parameters
        ----------
        threshold_bps : int
            Threshold in 0.1 basis point units (250 = 25bps = 0.25%)
        """
        # Validation happens in Rust layer, which raises PyValueError
        self._processor = _PyRangeBarProcessor(threshold_bps)
        self.threshold_bps = threshold_bps

    def process_trades(
        self, trades: list[dict[str, int | float]]
    ) -> list[dict[str, str | float | int]]:
        """Process trades into range bars.

        Parameters
        ----------
        trades : List[Dict]
            List of trade dictionaries with keys:
            - timestamp: int (milliseconds since epoch)
            - price: float
            - quantity: float (or 'volume')

            Optional keys:
            - agg_trade_id: int
            - first_trade_id: int
            - last_trade_id: int
            - is_buyer_maker: bool

        Returns
        -------
        List[Dict]
            List of range bar dictionaries with keys:
            - timestamp: str (RFC3339 format)
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: float
            - vwap: float (volume-weighted average price)
            - buy_volume: float
            - sell_volume: float
            - individual_trade_count: int
            - agg_record_count: int

        Raises
        ------
        KeyError
            If required trade fields are missing
        RuntimeError
            If trades are not sorted chronologically

        Examples
        --------
        >>> processor = RangeBarProcessor(250)
        >>> trades = [
        ...     {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
        ...     {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ... ]
        >>> bars = processor.process_trades(trades)
        >>> len(bars)
        1
        >>> bars[0]["open"]
        42000.0
        """
        if not trades:
            return []

        return self._processor.process_trades(trades)

    def to_dataframe(self, bars: list[dict[str, str | float | int]]) -> pd.DataFrame:
        """Convert range bars to pandas DataFrame (backtesting.py compatible).

        Parameters
        ----------
        bars : List[Dict]
            List of range bar dictionaries from process_trades()

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and OHLCV columns:
            - Index: timestamp (DatetimeIndex)
            - Columns: Open, High, Low, Close, Volume

        Notes
        -----
        Output format is compatible with backtesting.py:
        - Column names are capitalized (Open, High, Low, Close, Volume)
        - Index is DatetimeIndex
        - No NaN values (all bars complete)
        - Sorted chronologically

        Examples
        --------
        >>> processor = RangeBarProcessor(250)
        >>> trades = [
        ...     {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
        ...     {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ... ]
        >>> bars = processor.process_trades(trades)
        >>> df = processor.to_dataframe(bars)
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
        >>> list(df.columns)
        ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        if not bars:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            ).set_index(pd.DatetimeIndex([]))

        result = pd.DataFrame(bars)

        # Convert timestamp from RFC3339 string to DatetimeIndex
        # Use format='ISO8601' to handle variable-precision fractional seconds
        result["timestamp"] = pd.to_datetime(result["timestamp"], format="ISO8601")
        result = result.set_index("timestamp")

        # Rename columns to backtesting.py format (capitalized)
        result = result.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        # Return only OHLCV columns (drop microstructure fields for backtesting)
        return result[["Open", "High", "Low", "Close", "Volume"]]


def process_trades_to_dataframe(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Convenience function to process trades directly to DataFrame.

    This is the recommended high-level API for most users. Handles both
    list-of-dicts and pandas DataFrame inputs.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns/keys:
        - timestamp: int (milliseconds) or datetime
        - price: float
        - quantity: float (or 'volume')
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Capitalized columns: Open, High, Low, Close, Volume

    Raises
    ------
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If trades are not sorted chronologically

    Examples
    --------
    With list of dicts:

    >>> from rangebar import process_trades_to_dataframe
    >>> trades = [
    ...     {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    ...     {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    ... ]
    >>> df = process_trades_to_dataframe(trades, threshold_bps=250)

    With pandas DataFrame:

    >>> import pandas as pd
    >>> trades_df = pd.DataFrame({
    ...     "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
    ...     "price": [42000.0 + i for i in range(100)],
    ...     "quantity": [1.5] * 100,
    ... })
    >>> df = process_trades_to_dataframe(trades_df, threshold_bps=250)

    With Binance CSV:

    >>> trades_csv = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
    >>> df = process_trades_to_dataframe(trades_csv, threshold_bps=250)
    >>> # Use with backtesting.py
    >>> from backtesting import Backtest
    >>> bt = Backtest(df, MyStrategy, cash=10000)
    >>> stats = bt.run()
    """
    processor = RangeBarProcessor(threshold_bps)

    # Convert DataFrame to list of dicts if needed
    if isinstance(trades, pd.DataFrame):
        # Support both 'quantity' and 'volume' column names
        volume_col = "quantity" if "quantity" in trades.columns else "volume"

        required = {"timestamp", "price", volume_col}
        missing = required - set(trades.columns)
        if missing:
            msg = (
                f"DataFrame missing required columns: {missing}. "
                "Required: timestamp, price, quantity (or volume)"
            )
            raise ValueError(msg)

        # Convert timestamp to milliseconds if it's datetime
        trades_copy = trades.copy()
        if pd.api.types.is_datetime64_any_dtype(trades_copy["timestamp"]):
            # Convert datetime to milliseconds since epoch
            trades_copy["timestamp"] = trades_copy["timestamp"].astype("int64") // 10**6

        # Normalize column name to 'quantity'
        if volume_col == "volume":
            trades_copy = trades_copy.rename(columns={"volume": "quantity"})

        # Convert to list of dicts
        trades_list = trades_copy[["timestamp", "price", "quantity"]].to_dict("records")
    else:
        trades_list = trades

    # Process through Rust layer
    bars = processor.process_trades(trades_list)

    # Convert to DataFrame
    return processor.to_dataframe(bars)


def process_trades_to_dataframe_cached(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    symbol: str,
    threshold_bps: int = 250,
    cache: RangeBarCache | None = None,
) -> pd.DataFrame:
    """Process trades to DataFrame with two-tier ClickHouse caching.

    This function provides cached processing of trades into range bars.
    It uses a two-tier cache:
    - Tier 1: Raw trades (avoid re-downloading)
    - Tier 2: Computed range bars (avoid re-computing)

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns/keys:
        - timestamp: int (milliseconds) or datetime
        - price: float
        - quantity: float (or 'volume')
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
        If no ClickHouse hosts available (with setup guidance)
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If trades are not sorted chronologically

    Examples
    --------
    >>> from rangebar import process_trades_to_dataframe_cached
    >>> import pandas as pd
    >>>
    >>> trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
    >>> df = process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")
    >>>
    >>> # Second call uses cache (fast)
    >>> df2 = process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")
    """
    # Import cache components (lazy import)
    from .clickhouse import CacheKey
    from .clickhouse import RangeBarCache as _RangeBarCache

    # Convert trades to DataFrame if needed for timestamp extraction
    if isinstance(trades, list):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades

    # Get timestamp range
    if "timestamp" in trades_df.columns:
        ts_col = trades_df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            start_ts = int(ts_col.min().timestamp() * 1000)
            end_ts = int(ts_col.max().timestamp() * 1000)
        else:
            start_ts = int(ts_col.min())
            end_ts = int(ts_col.max())
    else:
        msg = "DataFrame missing 'timestamp' column"
        raise ValueError(msg)

    # Create cache key
    key = CacheKey(
        symbol=symbol,
        threshold_bps=threshold_bps,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    # Use provided cache or create new one
    _cache = cache if cache is not None else _RangeBarCache()
    owns_cache = cache is None

    try:
        # Check Tier 2 cache (computed range bars)
        if _cache.has_range_bars(key):
            cached_bars = _cache.get_range_bars(key)
            if cached_bars is not None:
                return cached_bars

        # Compute using core API
        result = process_trades_to_dataframe(trades, threshold_bps)

        # Store in Tier 2 cache
        if not result.empty:
            _cache.store_range_bars(key, result)

        return result

    finally:
        if owns_cache:
            _cache.close()


# ============================================================================
# Optimized Processing APIs (Phase 2-4 of Python pipeline optimization)
# ============================================================================


def process_trades_chunked(
    trades_iterator: Iterator[dict[str, int | float]],
    threshold_bps: int = 250,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Process trades in chunks to avoid memory spikes.

    This function enables streaming processing of large datasets without
    loading all trades into memory at once.

    Parameters
    ----------
    trades_iterator : Iterator[Dict]
        Iterator yielding trade dictionaries with keys:
        timestamp, price, quantity (or volume)
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)
    chunk_size : int, default=100_000
        Number of trades per chunk

    Yields
    ------
    pd.DataFrame
        OHLCV bars for each chunk. Note: partial bars may occur at
        chunk boundaries.

    Examples
    --------
    Process large Parquet file without OOM:

    >>> import polars as pl
    >>> from rangebar import process_trades_chunked
    >>> lazy_df = pl.scan_parquet("large_trades.parquet")
    >>> for chunk_df in lazy_df.collect().iter_slices(100_000):
    ...     trades = chunk_df.to_dicts()
    ...     for bars_df in process_trades_chunked(iter(trades)):
    ...         print(f"Got {len(bars_df)} bars")

    Notes
    -----
    Memory usage: O(chunk_size) instead of O(total_trades)
    For datasets >10M trades, use chunk_size=50_000 for safety.
    """
    from itertools import islice

    processor = RangeBarProcessor(threshold_bps)

    while True:
        chunk = list(islice(trades_iterator, chunk_size))
        if not chunk:
            break

        bars = processor.process_trades(chunk)
        if bars:
            yield processor.to_dataframe(bars)


def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Process trades from Polars DataFrame (optimized pipeline).

    This is the recommended API for Polars users. Uses lazy evaluation
    and minimal dict conversion for best performance.

    Parameters
    ----------
    trades : polars.DataFrame or polars.LazyFrame
        Trade data with columns:
        - timestamp: int64 (milliseconds since epoch)
        - price: float
        - quantity (or volume): float
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Capitalized columns: Open, High, Low, Close, Volume

    Examples
    --------
    With LazyFrame (predicate pushdown):

    >>> import polars as pl
    >>> from rangebar import process_trades_polars
    >>> lazy_df = pl.scan_parquet("trades.parquet")
    >>> lazy_filtered = lazy_df.filter(pl.col("timestamp") >= 1704067200000)
    >>> df = process_trades_polars(lazy_filtered, threshold_bps=250)

    With DataFrame:

    >>> df = pl.read_parquet("trades.parquet")
    >>> bars = process_trades_polars(df)

    Notes
    -----
    Performance optimization:
    - Only required columns are extracted (timestamp, price, quantity)
    - Lazy evaluation: predicates pushed to I/O layer
    - 2-3x faster than process_trades_to_dataframe() for Polars inputs
    """
    import polars as pl

    # Collect if lazy
    if isinstance(trades, pl.LazyFrame):
        trades = trades.collect()

    # Determine volume column name
    volume_col = "quantity" if "quantity" in trades.columns else "volume"

    # Select only required columns (minimal dict conversion)
    # This avoids converting unused columns to Python objects
    trades_minimal = trades.select(
        [
            pl.col("timestamp"),
            pl.col("price"),
            pl.col(volume_col).alias("quantity"),
        ]
    )

    # Convert to trades list
    trades_list = trades_minimal.to_dicts()

    # Process through Rust layer
    processor = RangeBarProcessor(threshold_bps)
    bars = processor.process_trades(trades_list)

    return processor.to_dataframe(bars)


# Re-export ClickHouse components for convenience
# ============================================================================
# Tier-1 Symbols (high-liquidity, available on all Binance markets)
# ============================================================================

TIER1_SYMBOLS: tuple[str, ...] = (
    "AAVE",
    "ADA",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "ETH",
    "FIL",
    "LINK",
    "LTC",
    "NEAR",
    "SOL",
    "SUI",
    "UNI",
    "WIF",
    "WLD",
    "XRP",
)

# Valid threshold range (from rangebar-core)
THRESHOLD_MIN = 1  # 0.1bps = 0.001%
THRESHOLD_MAX = 100_000  # 10,000bps = 100%

# Common threshold presets (in 0.1bps units)
THRESHOLD_PRESETS: dict[str, int] = {
    "micro": 10,  # 1bps = 0.01% (scalping)
    "tight": 50,  # 5bps = 0.05% (day trading)
    "standard": 100,  # 10bps = 0.1% (swing trading)
    "medium": 250,  # 25bps = 0.25% (default)
    "wide": 500,  # 50bps = 0.5% (position trading)
    "macro": 1000,  # 100bps = 1% (long-term)
}


def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_bps: int | str = 250,
    *,
    # Data source configuration
    source: str = "binance",
    market: str = "spot",
    # Exness-specific options
    validation: str = "strict",
    # Processing options
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    # Caching options
    use_cache: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Get range bars for a symbol with automatic data fetching and caching.

    This is the single entry point for all range bar generation. It supports
    multiple data sources (Binance crypto, Exness forex), all market types,
    and exposes the full configurability of the underlying Rust engine.

    Parameters
    ----------
    symbol : str
        Trading symbol (uppercase).
        - Binance: "BTCUSDT", "ETHUSDT", etc.
        - Exness: "EURUSD", "GBPUSD", "XAUUSD", etc.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    threshold_bps : int or str, default=250
        Threshold in 0.1bps units. Can be:
        - Integer: Direct value (250 = 25bps = 0.25%)
        - String preset: "micro" (1bps), "tight" (5bps), "standard" (10bps),
          "medium" (25bps), "wide" (50bps), "macro" (100bps)
        Valid range: 1-100,000 (0.001% to 100%)

    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only):
        - "spot": Spot market
        - "futures-um" or "um": USD-M perpetual futures
        - "futures-cm" or "cm": COIN-M perpetual futures
    validation : str, default="strict"
        Validation strictness (Exness only):
        - "permissive": Basic checks (bid > 0, ask > 0, bid < ask)
        - "strict": + Spread < 10% (catches obvious errors)
        - "paranoid": + Spread < 1% (flags suspicious data)
    include_incomplete : bool, default=False
        Include the final incomplete bar (useful for analysis).
        If False (default), only completed bars are returned.
    include_microstructure : bool, default=False
        Include market microstructure columns:
        - buy_volume, sell_volume: Volume by aggressor side
        - vwap: Volume-weighted average price
        - trade_count: Number of trades in bar
        - (Exness) spread_min, spread_max, spread_avg: Spread statistics
    use_cache : bool, default=True
        Cache tick data locally in Parquet format.
    cache_dir : str or None, default=None
        Custom cache directory. If None, uses platform default:
        - macOS: ~/Library/Caches/rangebar/
        - Linux: ~/.cache/rangebar/
        - Windows: %LOCALAPPDATA%/terrylica/rangebar/Cache/

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Columns: Open, High, Low, Close, Volume
        - (if include_microstructure) Additional columns

    Raises
    ------
    ValueError
        - Invalid threshold (outside 1-100,000 range)
        - Invalid dates or date format
        - Unknown source, market, or validation level
        - Unknown threshold preset name
    RuntimeError
        - Data fetching failed
        - No data available for date range
        - Feature not enabled (e.g., Exness without exness feature)

    Examples
    --------
    Basic usage - Binance spot:

    >>> from rangebar import get_range_bars
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

    Using threshold presets:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", "tight")

    Binance USD-M Futures:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

    Exness forex with spread monitoring:

    >>> df = get_range_bars(
    ...     "EURUSD", "2024-01-01", "2024-01-31",
    ...     source="exness",
    ...     threshold_bps="standard",
    ...     include_microstructure=True,  # includes spread stats
    ... )

    Include incomplete bar for analysis:

    >>> df = get_range_bars(
    ...     "ETHUSDT", "2024-01-01", "2024-01-07",
    ...     include_incomplete=True,
    ... )

    Use with backtesting.py:

    >>> from backtesting import Backtest, Strategy
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31")
    >>> bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
    >>> stats = bt.run()

    Notes
    -----
    Threshold units (0.1bps):
        The threshold is specified in tenths of basis points for precision.
        Common conversions:
        - 10 = 1bps = 0.01%
        - 100 = 10bps = 0.1%
        - 250 = 25bps = 0.25%
        - 1000 = 100bps = 1%

    Tier-1 symbols:
        18 high-liquidity symbols available on ALL Binance markets:
        AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE, ETH, FIL,
        LINK, LTC, NEAR, SOL, SUI, UNI, WIF, WLD, XRP

    Non-lookahead guarantee:
        - Threshold computed from bar OPEN price only
        - Breaching trade included in closing bar
        - No future information used in bar construction

    See Also
    --------
    TIER1_SYMBOLS : Tuple of high-liquidity symbols
    THRESHOLD_PRESETS : Dictionary of named threshold values
    """
    from datetime import datetime
    from pathlib import Path

    import polars as pl

    from .storage.parquet import TickStorage

    # -------------------------------------------------------------------------
    # Resolve threshold (support presets)
    # -------------------------------------------------------------------------
    if isinstance(threshold_bps, str):
        if threshold_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold_bps = THRESHOLD_PRESETS[threshold_bps]

    if not THRESHOLD_MIN <= threshold_bps <= THRESHOLD_MAX:
        msg = (
            f"threshold_bps must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}, "
            f"got {threshold_bps}"
        )
        raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Validate source and market
    # -------------------------------------------------------------------------
    source = source.lower()
    if source not in ("binance", "exness"):
        msg = f"Unknown source: {source!r}. Must be 'binance' or 'exness'"
        raise ValueError(msg)

    # Normalize market type
    market_map = {
        "spot": "spot",
        "futures-um": "um",
        "futures-cm": "cm",
        "um": "um",
        "cm": "cm",
    }
    market = market.lower()
    if source == "binance" and market not in market_map:
        msg = (
            f"Unknown market: {market!r}. "
            "Must be 'spot', 'futures-um'/'um', or 'futures-cm'/'cm'"
        )
        raise ValueError(msg)
    market_normalized = market_map.get(market, market)

    # Validate Exness validation strictness
    validation = validation.lower()
    if source == "exness" and validation not in ("permissive", "strict", "paranoid"):
        msg = (
            f"Unknown validation: {validation!r}. "
            "Must be 'permissive', 'strict', or 'paranoid'"
        )
        raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Parse and validate dates
    # -------------------------------------------------------------------------
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
        raise ValueError(msg) from e

    if start_dt > end_dt:
        msg = "start_date must be <= end_date"
        raise ValueError(msg)

    # Convert to milliseconds for cache lookup
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int((end_dt.timestamp() + 86399) * 1000)  # End of day

    # -------------------------------------------------------------------------
    # Initialize storage
    # -------------------------------------------------------------------------
    storage = TickStorage(cache_dir=Path(cache_dir) if cache_dir else None)

    # Cache key includes source and market to avoid collisions
    cache_symbol = f"{source}_{market_normalized}_{symbol}".upper()

    # -------------------------------------------------------------------------
    # Fetch tick data (cache or network)
    # -------------------------------------------------------------------------
    tick_data: pl.DataFrame

    if use_cache and storage.has_ticks(cache_symbol, start_ts, end_ts):
        tick_data = storage.read_ticks(cache_symbol, start_ts, end_ts)
    else:
        if source == "binance":
            tick_data = _fetch_binance(symbol, start_date, end_date, market_normalized)
        else:  # exness
            tick_data = _fetch_exness(symbol, start_date, end_date, validation)

        # Cache the tick data
        if use_cache and not tick_data.is_empty():
            storage.write_ticks(cache_symbol, tick_data)

    if tick_data.is_empty():
        msg = f"No data available for {symbol} from {start_date} to {end_date}"
        raise RuntimeError(msg)

    # -------------------------------------------------------------------------
    # Process to range bars
    # -------------------------------------------------------------------------
    if source == "exness":
        return _process_exness_ticks(
            tick_data,
            symbol,
            threshold_bps,
            validation,
            include_incomplete,
            include_microstructure,
        )

    return _process_binance_trades(
        tick_data,
        threshold_bps,
        include_incomplete,
        include_microstructure,
    )


def _fetch_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    market: str,
) -> pl.DataFrame:
    """Fetch Binance aggTrades data (internal)."""
    import polars as pl

    try:
        from ._core import MarketType, fetch_binance_aggtrades

        market_enum = {
            "spot": MarketType.Spot,
            "um": MarketType.FuturesUM,
            "cm": MarketType.FuturesCM,
        }[market]

        trades_list = fetch_binance_aggtrades(symbol, start_date, end_date, market_enum)
        return pl.DataFrame(trades_list)

    except ImportError as e:
        msg = (
            "Binance data fetching requires the 'data-providers' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e


def _fetch_exness(
    symbol: str,
    start_date: str,
    end_date: str,
    validation: str,  # noqa: ARG001 - reserved for future use
) -> pl.DataFrame:
    """Fetch Exness tick data (internal)."""
    try:
        from .exness import fetch_exness_ticks

        return fetch_exness_ticks(symbol, start_date, end_date)

    except ImportError as e:
        msg = (
            "Exness data fetching requires the 'exness' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e


def _process_binance_trades(
    trades: pl.DataFrame,
    threshold_bps: int,
    include_incomplete: bool,  # noqa: ARG001 - TODO: implement
    include_microstructure: bool,
) -> pd.DataFrame:
    """Process Binance trades to range bars (internal)."""
    import polars as pl

    # Collect if lazy
    if isinstance(trades, pl.LazyFrame):
        trades = trades.collect()

    # Determine volume column name
    volume_col = "quantity" if "quantity" in trades.columns else "volume"

    # Select only required columns
    trades_minimal = trades.select(
        [
            pl.col("timestamp"),
            pl.col("price"),
            pl.col(volume_col).alias("quantity"),
        ]
    )

    # Convert to trades list and process
    trades_list = trades_minimal.to_dicts()
    processor = RangeBarProcessor(threshold_bps)
    bars = processor.process_trades(trades_list)

    if not bars:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))

    # Build DataFrame with all fields
    result = pd.DataFrame(bars)
    result["timestamp"] = pd.to_datetime(result["timestamp"], format="ISO8601")
    result = result.set_index("timestamp")

    # Rename OHLCV columns to backtesting.py format
    result = result.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    if include_microstructure:
        # Return all columns including microstructure
        return result

    # Return only OHLCV columns (backtesting.py compatible)
    return result[["Open", "High", "Low", "Close", "Volume"]]


def _process_exness_ticks(
    ticks: pl.DataFrame,
    symbol: str,
    threshold_bps: int,
    validation: str,
    include_incomplete: bool,  # noqa: ARG001 - TODO: implement
    include_microstructure: bool,
) -> pd.DataFrame:
    """Process Exness ticks to range bars (internal)."""
    try:
        # Map validation string to enum
        from ._core import ValidationStrictness
        from .exness import process_exness_ticks_to_dataframe

        validation_enum = {
            "permissive": ValidationStrictness.Permissive,
            "strict": ValidationStrictness.Strict,
            "paranoid": ValidationStrictness.Paranoid,
        }[validation]

        # Get instrument enum
        from ._core import ExnessInstrument

        instrument_map = {
            "EURUSD": ExnessInstrument.EURUSD,
            "GBPUSD": ExnessInstrument.GBPUSD,
            "USDJPY": ExnessInstrument.USDJPY,
            "AUDUSD": ExnessInstrument.AUDUSD,
            "USDCAD": ExnessInstrument.USDCAD,
            "NZDUSD": ExnessInstrument.NZDUSD,
            "EURGBP": ExnessInstrument.EURGBP,
            "EURJPY": ExnessInstrument.EURJPY,
            "GBPJPY": ExnessInstrument.GBPJPY,
            "XAUUSD": ExnessInstrument.XAUUSD,
        }

        if symbol.upper() not in instrument_map:
            msg = (
                f"Unknown Exness instrument: {symbol}. "
                f"Valid instruments: {list(instrument_map.keys())}"
            )
            raise ValueError(msg)

        instrument = instrument_map[symbol.upper()]

        df = process_exness_ticks_to_dataframe(
            ticks.to_pandas(),
            instrument,
            threshold_bps,
            validation_enum,
        )

        if not include_microstructure:
            return df[["Open", "High", "Low", "Close", "Volume"]]

        return df

    except ImportError as e:
        msg = (
            "Exness processing requires the 'exness' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e


def __getattr__(name: str) -> object:
    """Lazy attribute access for ClickHouse and Exness components."""
    if name in {
        "RangeBarCache",
        "CacheKey",
        "get_available_clickhouse_host",
        "detect_clickhouse_state",
        "ClickHouseNotConfiguredError",
    }:
        from . import clickhouse

        return getattr(clickhouse, name)

    if name == "is_exness_available":
        from .exness import is_exness_available

        return is_exness_available

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
