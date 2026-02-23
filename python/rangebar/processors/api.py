# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M3 - Extract process_trades_* functions from __init__.py
"""Convenience functions for processing trades into range bars.

Provides multiple entry points for different input formats (pandas, Polars,
iterators) with automatic DataFrame conversion.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd

from .core import RangeBarProcessor

if TYPE_CHECKING:
    import polars as pl

    from rangebar.clickhouse import RangeBarCache


def _arrow_bars_to_pandas(
    bars_pl: pl.DataFrame,
    include_microstructure: bool = False,
) -> pd.DataFrame:
    """Convert Arrow-format Polars bars to backtesting.py-compatible pandas.

    Arrow output has open_time/close_time as i64 microseconds.
    to_dataframe() expects dict bars with timestamp as RFC3339 string.
    This function converts directly without the dict roundtrip.
    """
    if bars_pl.is_empty():
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))

    result = bars_pl.to_pandas()

    # Arrow schema uses open_time (microseconds) as the timestamp
    result["timestamp"] = pd.to_datetime(result["open_time"], unit="us")
    result = result.set_index("timestamp")

    # Drop time columns (not needed for backtesting.py)
    result = result.drop(columns=["open_time", "close_time"], errors="ignore")

    # Rename to backtesting.py format
    # Issue #96 Task #32: Use Polars batch rename (more efficient than Pandas)
    import polars as pl
    result_pl = pl.from_pandas(result)
    result_pl = result_pl.rename({
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    result = result_pl.to_pandas()

    if include_microstructure:
        return result

    return result[["Open", "High", "Low", "Close", "Volume"]]


def process_trades_to_dataframe(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    threshold_decimal_bps: int = 250,
    *,
    symbol: str | None = None,
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
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
    symbol : str, optional
        Trading symbol (e.g., "BTCUSDT"). When provided, enables
        asset-class minimum threshold validation (Issue #62).

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
    >>> df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)

    With pandas DataFrame:

    >>> import pandas as pd
    >>> trades_df = pd.DataFrame({
    ...     "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
    ...     "price": [42000.0 + i for i in range(100)],
    ...     "quantity": [1.5] * 100,
    ... })
    >>> df = process_trades_to_dataframe(trades_df, threshold_decimal_bps=250)

    With Binance CSV:

    >>> trades_csv = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
    >>> df = process_trades_to_dataframe(trades_csv, threshold_decimal_bps=250)
    >>> # Use with backtesting.py
    >>> from backtesting import Backtest
    >>> bt = Backtest(df, MyStrategy, cash=10000)
    >>> stats = bt.run()
    """
    # Symbol registry gate (Issue #79) -- optional: only if symbol provided
    if symbol is not None:
        from rangebar.symbol_registry import validate_symbol_registered

        validate_symbol_registered(symbol, operation="process_trades_to_dataframe")

    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)

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
            trades_copy["timestamp"] = (
                trades_copy["timestamp"].dt.as_unit("ms").astype("int64")
            )

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
    threshold_decimal_bps: int = 250,
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
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
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
    # Symbol registry gate (Issue #79) -- required: symbol is cache key
    from rangebar.symbol_registry import validate_symbol_registered

    validate_symbol_registered(symbol, operation="process_trades_to_dataframe_cached")

    # Import cache components (lazy import)
    from rangebar.clickhouse import CacheKey
    from rangebar.clickhouse import RangeBarCache as _RangeBarCache

    # Convert trades to DataFrame if needed for timestamp extraction
    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades

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
        threshold_decimal_bps=threshold_decimal_bps,
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
        result = process_trades_to_dataframe(trades, threshold_decimal_bps)

        # Store in Tier 2 cache
        if not result.empty:
            _cache.store_range_bars(key, result)

        return result

    finally:
        if owns_cache:
            _cache.close()


def process_trades_chunked(
    trades_iterator: Iterator[dict[str, int | float]],
    threshold_decimal_bps: int = 250,
    chunk_size: int = 100_000,
    *,
    symbol: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Process trades in chunks to avoid memory spikes.

    This function enables streaming processing of large datasets without
    loading all trades into memory at once.

    Parameters
    ----------
    trades_iterator : Iterator[Dict]
        Iterator yielding trade dictionaries with keys:
        timestamp, price, quantity (or volume)
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
    chunk_size : int, default=100_000
        Number of trades per chunk
    symbol : str, optional
        Trading symbol (e.g., "BTCUSDT"). When provided, enables
        asset-class minimum threshold validation (Issue #62).

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

    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)

    while True:
        chunk = list(islice(trades_iterator, chunk_size))
        if not chunk:
            break

        bars = processor.process_trades(chunk)
        if bars:
            yield processor.to_dataframe(bars)


def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
    *,
    symbol: str | None = None,
    include_microstructure: bool = False,
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
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
    symbol : str, optional
        Trading symbol (e.g., "BTCUSDT"). When provided, enables
        asset-class minimum threshold validation (Issue #62).
    include_microstructure : bool, default=False
        If True, include all microstructure feature columns in output.
        If False (default), return only OHLCV columns for backtesting.py.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Capitalized columns: Open, High, Low, Close, Volume
        - If include_microstructure=True: all microstructure columns

    Examples
    --------
    With LazyFrame (predicate pushdown):

    >>> import polars as pl
    >>> from rangebar import process_trades_polars
    >>> lazy_df = pl.scan_parquet("trades.parquet")
    >>> lazy_filtered = lazy_df.filter(
    ...     pl.col("timestamp") >= 1704067200000
    ... )
    >>> df = process_trades_polars(
    ...     lazy_filtered, threshold_decimal_bps=250
    ... )

    With DataFrame:

    >>> df = pl.read_parquet("trades.parquet")
    >>> bars = process_trades_polars(df)

    With microstructure features:

    >>> bars = process_trades_polars(df, include_microstructure=True)

    Notes
    -----
    Performance optimization:
    - Only required columns are extracted (timestamp, price, quantity)
    - Lazy evaluation: predicates pushed to I/O layer
    - 2-3x faster than process_trades_to_dataframe() for Polars inputs
    """
    import polars as pl

    from rangebar.orchestration.helpers import _select_trade_columns

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    trades_selected = _select_trade_columns(trades)

    # Collect AFTER selection (for LazyFrame)
    if isinstance(trades_selected, pl.LazyFrame):
        trades_minimal = trades_selected.collect()
    else:
        trades_minimal = trades_selected

    # Issue #88: Arrow-native input — .to_arrow() is zero-copy
    # MEM-002: Process in chunks to bound memory
    chunk_size = 100_000
    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)
    bar_frames: list[pl.DataFrame] = []

    n_rows = len(trades_minimal)
    for start in range(0, n_rows, chunk_size):
        chunk_arrow = trades_minimal.slice(start, chunk_size).to_arrow()
        bars_arrow = processor.process_trades_arrow(chunk_arrow)
        bars_df = pl.from_arrow(bars_arrow)
        if not bars_df.is_empty():
            bar_frames.append(bars_df)

    if not bar_frames:
        return processor.to_dataframe([])

    # Convert Arrow-format Polars bars to backtesting.py-compatible pandas
    result_pl = pl.concat(bar_frames)
    return _arrow_bars_to_pandas(
        result_pl, include_microstructure=include_microstructure
    )
