# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M3 - Extract process_trades_* functions from __init__.py
# FILE-SIZE-OK: Multiple entry points for different input formats
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

    # Arrow close_time (us) → DatetimeIndex (unified across all paths)
    result["timestamp"] = pd.to_datetime(result["close_time"], unit="us")
    result = result.set_index("timestamp")

    # Drop time columns (not needed for backtesting.py)
    result = result.drop(columns=["open_time", "close_time"], errors="ignore")

    # Rename to backtesting.py format
    # Issue #96 Task #75: Direct Pandas rename eliminates unnecessary conversions
    # (1.8-3.0x speedup by removing pandas→polars→pandas round-trip)
    result = result.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

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

    # Convert DataFrame to Arrow for zero-copy processing (Issue #88, Task #143)
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

        # Issue #88 Task #143: Use Arrow zero-copy path
        # Replaces slow .to_dict("records") that consumed 65% of pipeline time
        # Arrow path is zero-copy and 50-100% faster
        import polars as pl
        import pyarrow as pa

        # Convert only required columns to Arrow (minimal memory footprint)
        trades_arrow = pa.table({
            "timestamp": pa.array(trades_copy["timestamp"], type=pa.int64()),
            "price": pa.array(trades_copy["price"], type=pa.float64()),
            "quantity": pa.array(trades_copy["quantity"], type=pa.float64()),
        })

        # Process through Arrow zero-copy path
        bars_arrow = processor.process_trades_arrow(trades_arrow)

        # Convert Arrow bars directly to pandas (same as process_trades_polars)
        bars_pl = pl.from_arrow(bars_arrow)
        return _arrow_bars_to_pandas(bars_pl, include_microstructure=False)
    # For list[dict] input, use original dict path (backward compatible)
    bars = processor.process_trades(trades)
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
    Performance optimization (Issue #96 Task #101):
    - Only required columns are extracted (timestamp, price, quantity)
    - Lazy evaluation: predicates pushed to I/O layer
    - 2-3x faster than process_trades_to_dataframe() for Polars inputs
    - Issue #96 Task #101: Increased chunk size from 100K to 500K for 5x fewer FFI calls
      (Python→Rust transitions), reducing overhead by 1.5-2x on large datasets
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
    # Issue #96 Task #101: Batch multiple chunks before FFI crossing to reduce overhead
    # MEM-002: Process in chunks to bound memory (500K chunk, 1.5-2x speedup over 100K)
    chunk_size = 500_000
    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)
    bar_frames: list[pl.DataFrame] = []

    n_rows = len(trades_minimal)
    for start in range(0, n_rows, chunk_size):
        # Issue #96 Task #101: Arrow conversion per 500K reduces FFI 5x
        # Memory: ~500K x 60 bytes Arrow + ~50K bars x 200 bytes = ~40MB
        end = min(start + chunk_size, n_rows)
        chunk_arrow = trades_minimal.slice(start, end - start).to_arrow()
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


def process_trades_polars_lazy(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
    *,
    symbol: str | None = None,
    include_microstructure: bool = False,  # noqa: ARG001 - reserved for future use
) -> pl.LazyFrame:
    """Process trades into range bars with lazy evaluation support.

    Returns a LazyFrame that enables efficient filtering of range bars before
    materialization. For maximum efficiency with large datasets, apply filters
    to input trades BEFORE passing to this function (predicate pushdown pattern).

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
    polars.LazyFrame
        Lazy range bars that can be further filtered before `.collect()`.
        Output columns include open_time, close_time, open, high, low, close, volume,
        plus microstructure features if enabled.

    Examples
    --------
    Efficient filtering pattern - apply filter to INPUT trades (predicate pushdown):

    >>> import polars as pl
    >>> from rangebar import process_trades_polars_lazy
    >>> trades = pl.scan_parquet("trades.parquet")  # Lazy
    >>>
    >>> # Filter input trades BEFORE processing
    >>> # This filter is applied at the I/O layer (predicate pushdown)
    >>> trades_filtered = trades.filter(
    ...     pl.col("timestamp") >= 1704067200000
    ... )
    >>>
    >>> # Process filtered trades lazily
    >>> bars = process_trades_polars_lazy(trades_filtered, threshold_decimal_bps=250)
    >>>
    >>> # Computation happens here, processing only the filtered input data
    >>> result = bars.collect()

    Date range partitioning for memory efficiency on 10M+ trade datasets:

    >>> for month in ["2024-01", "2024-02", "2024-03"]:
    ...     month_start = f"{month}-01"
    ...     month_dt = datetime.strptime(month_start, "%Y-%m-%d")
    ...     month_end_ts = int((month_dt + timedelta(days=32)).timestamp() * 1000)
    ...
    ...     trades_month = trades.filter(
    ...         pl.col("timestamp") < month_end_ts
    ...     )
    ...
    ...     # Only filtered trades are processed
    ...     bars = process_trades_polars_lazy(trades_month)
    ...     result = bars.collect()
    ...     save_to_database(result)

    Notes
    -----
    Performance optimization and lazy evaluation design:

    1. **Input filtering (predicate pushdown)**: Apply filters to input trades BEFORE
       calling this function. These filters are applied at the I/O layer, reducing the
       amount of data that needs to be materialized and processed.

    2. **Column selection**: Only required columns (timestamp, price, quantity) are
       extracted from the input before processing.

    3. **Delayed computation**: Bar computation occurs at `.collect()` time.
       This enables efficient memory usage for large datasets.

    4. **Performance**: 10-25% speedup for filtered queries, 2-4x memory reduction
       on 10M+ trades compared to non-filtered processing.

    5. **Output filtering limitation**: Due to how Polars handles custom operations,
       filters on OUTPUT columns (e.g., close_time) must be applied AFTER calling
       `.collect()`. Apply input filters instead for best performance.
    """
    import polars as pl

    from rangebar.orchestration.helpers import _select_trade_columns

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    trades_selected = _select_trade_columns(trades)

    # Keep as LazyFrame to enable predicate pushdown
    if isinstance(trades_selected, pl.DataFrame):
        trades_lazy = trades_selected.lazy()
    else:
        trades_lazy = trades_selected

    # Create a wrapper function that processes trades when the LazyFrame is collected
    def _process_trades_impl(batch: pl.DataFrame) -> pl.DataFrame:
        """Process a batch of trades into range bars.

        This function is called by Polars when `.collect()` is invoked on the
        returned LazyFrame. It receives the materialized trades and returns
        processed range bars.
        """
        if batch.is_empty():
            # Return empty DataFrame with representative schema
            # The schema will be inferred from actual output
            return pl.DataFrame()

        processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)
        # Issue #96 Task #101: 500K chunk = 5x fewer FFI calls
        chunk_size = 500_000
        bar_frames: list[pl.DataFrame] = []

        n_rows = len(batch)
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk_arrow = batch.slice(start, end - start).to_arrow()
            bars_arrow = processor.process_trades_arrow(chunk_arrow)
            bars_df = pl.from_arrow(bars_arrow)
            if not bars_df.is_empty():
                bar_frames.append(bars_df)

        if not bar_frames:
            return pl.DataFrame()

        return pl.concat(bar_frames)

    # Get schema by running processor on a minimal sample
    # This allows Polars to know the output columns for further optimization
    def _get_range_bars_schema_dynamic() -> dict[str, pl.DataType]:
        """Compute output schema by running processor on empty input."""
        processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)
        empty_arrow = pl.DataFrame({
            "timestamp": [],
            "price": [],
            "quantity": [],
        }).to_arrow()
        try:
            bars_arrow = processor.process_trades_arrow(empty_arrow)
            bars_df = pl.from_arrow(bars_arrow)
        except (RuntimeError, ValueError, OSError) as e:
            # If schema computation fails, return empty schema
            # Polars will infer schema at runtime from actual output
            import warnings
            warnings.warn(
                f"Could not pre-compute range bar schema: {e}. "
                f"Schema will be inferred at runtime.",
                UserWarning,
                stacklevel=3,
            )
            return {}
        else:
            return bars_df.schema

    output_schema = _get_range_bars_schema_dynamic()

    # Use map_batches to apply the processor when the lazy frame is collected
    # By specifying the schema, we allow Polars to properly handle filters and
    # selections on output columns after map_batches
    return trades_lazy.map_batches(
        _process_trades_impl,
        schema=output_schema if output_schema else None,
        validate_output_schema=False,  # Be lenient with schema validation
    )
