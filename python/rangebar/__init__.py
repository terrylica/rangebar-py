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

import pandas as pd

from ._core import PyRangeBarProcessor as _PyRangeBarProcessor
from ._core import __version__

# Lazy imports for ClickHouse cache (avoid import-time side effects)
# These are imported when first accessed
_clickhouse_imports_done = False


def _ensure_clickhouse_imports() -> None:
    """Ensure ClickHouse-related imports are done."""
    global _clickhouse_imports_done  # noqa: PLW0603
    if not _clickhouse_imports_done:
        from .clickhouse import (  # noqa: F401
            CacheKey,
            ClickHouseNotConfiguredError,
            RangeBarCache,
            detect_clickhouse_state,
            get_available_clickhouse_host,
        )

        _clickhouse_imports_done = True


__all__ = [
    # Sorted for ruff RUF022
    "CacheKey",
    "ClickHouseNotConfiguredError",
    "RangeBarCache",
    "RangeBarProcessor",
    "__version__",
    "detect_clickhouse_state",
    "get_available_clickhouse_host",
    "process_trades_to_dataframe",
    "process_trades_to_dataframe_cached",
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
    cache: "RangeBarCache | None" = None,
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


# Re-export ClickHouse components for convenience
# These use lazy imports to avoid import-time side effects
def __getattr__(name: str) -> object:
    """Lazy attribute access for ClickHouse components."""
    if name in {
        "RangeBarCache",
        "CacheKey",
        "get_available_clickhouse_host",
        "detect_clickhouse_state",
        "ClickHouseNotConfiguredError",
    }:
        from . import clickhouse

        return getattr(clickhouse, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
