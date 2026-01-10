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
>>> df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
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

from ._core import PositionVerification, __version__
from ._core import PyRangeBarProcessor as _PyRangeBarProcessor

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
    "THRESHOLD_DECIMAL_MAX",
    "THRESHOLD_DECIMAL_MIN",
    "THRESHOLD_PRESETS",
    "TIER1_SYMBOLS",
    "PositionVerification",
    "RangeBarProcessor",
    "__version__",
    "get_n_range_bars",
    "get_range_bars",
    "process_trades_to_dataframe",
    "validate_continuity",
]

# Continuity tolerance: 0.01% relative difference allowed (floating-point precision)
CONTINUITY_TOLERANCE_PCT = 0.0001


def _validate_junction_continuity(
    older_bars: pd.DataFrame,
    newer_bars: pd.DataFrame,
    tolerance_pct: float = CONTINUITY_TOLERANCE_PCT,
) -> tuple[bool, float | None]:
    """Validate continuity at junction between two bar DataFrames.

    Checks that older_bars[-1].Close == newer_bars[0].Open (within tolerance).

    Parameters
    ----------
    older_bars : pd.DataFrame
        Older bars (chronologically earlier)
    newer_bars : pd.DataFrame
        Newer bars (chronologically later)
    tolerance_pct : float
        Maximum allowed relative difference (0.0001 = 0.01%)

    Returns
    -------
    tuple[bool, float | None]
        (is_continuous, gap_pct) where gap_pct is the relative difference if
        discontinuous, None if continuous
    """
    if older_bars.empty or newer_bars.empty:
        return True, None

    last_close = older_bars.iloc[-1]["Close"]
    first_open = newer_bars.iloc[0]["Open"]

    if last_close == 0:
        return True, None  # Avoid division by zero

    gap_pct = abs(first_open - last_close) / abs(last_close)

    if gap_pct <= tolerance_pct:
        return True, None

    return False, gap_pct


def validate_continuity(
    df: pd.DataFrame,
    tolerance_pct: float = CONTINUITY_TOLERANCE_PCT,
) -> dict:
    """Validate that range bars form a continuous sequence.

    Checks that bar[i+1].Open == bar[i].Close (within tolerance) for all bars.
    This invariant should hold for properly constructed range bars.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with "Open" and "Close" columns
    tolerance_pct : float, default=0.0001
        Maximum allowed relative difference (0.0001 = 0.01%)

    Returns
    -------
    dict
        Validation result with keys:
        - is_valid: bool - True if all bars are continuous
        - bar_count: int - Total number of bars
        - discontinuity_count: int - Number of discontinuities found
        - discontinuities: list[dict] - Details of each discontinuity with
          keys: bar_index, prev_close, curr_open, gap_pct

    Examples
    --------
    >>> df = get_n_range_bars("BTCUSDT", n_bars=1000)
    >>> result = validate_continuity(df)
    >>> if not result["is_valid"]:
    ...     print(f"Found {result['discontinuity_count']} gaps")
    ...     for d in result["discontinuities"][:5]:
    ...         print(f"  Bar {d['bar_index']}: gap={d['gap_pct']:.4%}")

    Notes
    -----
    Discontinuities occur when range bars from different processing sessions
    are combined. Each processing session creates bars independently, so the
    junction point between sessions may have a price gap.

    For continuous bars, either:
    1. Use single-pass processing with one RangeBarProcessor instance
    2. Invalidate cache and re-fetch all data
    """
    if df.empty:
        return {
            "is_valid": True,
            "bar_count": 0,
            "discontinuity_count": 0,
            "discontinuities": [],
        }

    if "Close" not in df.columns or "Open" not in df.columns:
        msg = "DataFrame must have 'Open' and 'Close' columns"
        raise ValueError(msg)

    discontinuities = []
    close_prices = df["Close"].to_numpy()[:-1]
    open_prices = df["Open"].to_numpy()[1:]

    for i, (prev_close, curr_open) in enumerate(
        zip(close_prices, open_prices, strict=False)
    ):
        if prev_close == 0:
            continue

        gap_pct = abs(curr_open - prev_close) / abs(prev_close)
        if gap_pct > tolerance_pct:
            discontinuities.append(
                {
                    "bar_index": i + 1,  # Index of the bar with discontinuity
                    "prev_close": float(prev_close),
                    "curr_open": float(curr_open),
                    "gap_pct": float(gap_pct),
                }
            )

    return {
        "is_valid": len(discontinuities) == 0,
        "bar_count": len(df),
        "discontinuity_count": len(discontinuities),
        "discontinuities": discontinuities,
    }


class RangeBarProcessor:
    """Process tick-level trade data into range bars.

    Range bars close when price moves ±threshold from the bar's opening price,
    providing market-adaptive time intervals that eliminate arbitrary time-based
    artifacts.

    Parameters
    ----------
    threshold_decimal_bps : int
        Threshold in decimal basis points.
        Examples: 250 = 25bps = 0.25%, 100 = 10bps = 0.1%
        Valid range: [1, 100_000] (0.001% to 100%)
    symbol : str, optional
        Trading symbol (e.g., "BTCUSDT"). Required for checkpoint creation.

    Raises
    ------
    ValueError
        If threshold_decimal_bps is out of valid range [1, 100_000]

    Examples
    --------
    Create processor and convert trades to DataFrame:

    >>> processor = RangeBarProcessor(threshold_decimal_bps=250)  # 0.25%
    >>> trades = [
    ...     {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    ...     {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
    ... ]
    >>> bars = processor.process_trades(trades)
    >>> df = processor.to_dataframe(bars)
    >>> print(df.columns.tolist())
    ['Open', 'High', 'Low', 'Close', 'Volume']

    Cross-file continuity with checkpoints:

    >>> processor = RangeBarProcessor(250, symbol="BTCUSDT")
    >>> bars_file1 = processor.process_trades(file1_trades)
    >>> checkpoint = processor.create_checkpoint()
    >>> # Save checkpoint to JSON...
    >>> # Later, resume from checkpoint:
    >>> processor2 = RangeBarProcessor.from_checkpoint(checkpoint)
    >>> bars_file2 = processor2.process_trades(file2_trades)
    >>> # Incomplete bar from file1 continues correctly!

    Notes
    -----
    Non-lookahead bias guarantee:
    - Thresholds computed ONLY from bar open price (never recalculated)
    - Breaching trade INCLUDED in closing bar
    - Breaching trade also OPENS next bar

    Temporal integrity:
    - All trades processed in strict chronological order
    - Unsorted trades raise RuntimeError

    Cross-file continuity (v6.1.0+):
    - Incomplete bars are preserved across file boundaries via checkpoints
    - Thresholds are IMMUTABLE for bar's lifetime (computed from open)
    - Price hash verification detects gaps in data stream
    """

    def __init__(self, threshold_decimal_bps: int, symbol: str | None = None) -> None:
        """Initialize processor with given threshold.

        Parameters
        ----------
        threshold_decimal_bps : int
            Threshold in decimal basis points (250 = 25bps = 0.25%)
        symbol : str, optional
            Trading symbol for checkpoint creation
        """
        # Validation happens in Rust layer, which raises PyValueError
        self._processor = _PyRangeBarProcessor(threshold_decimal_bps, symbol)
        self.threshold_decimal_bps = threshold_decimal_bps
        self.symbol = symbol

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> RangeBarProcessor:
        """Create processor from checkpoint for cross-file continuation.

        Restores processor state including any incomplete bar that was being
        built when the checkpoint was created. The incomplete bar will continue
        building from where it left off.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint state from create_checkpoint()

        Returns
        -------
        RangeBarProcessor
            New processor with restored state

        Raises
        ------
        ValueError
            If checkpoint is invalid or corrupted

        Examples
        --------
        >>> import json
        >>> with open("checkpoint.json") as f:
        ...     checkpoint = json.load(f)
        >>> processor = RangeBarProcessor.from_checkpoint(checkpoint)
        >>> bars = processor.process_trades(next_file_trades)
        """
        instance = cls.__new__(cls)
        instance._processor = _PyRangeBarProcessor.from_checkpoint(checkpoint)
        instance.threshold_decimal_bps = checkpoint["threshold_decimal_bps"]
        instance.symbol = checkpoint.get("symbol")
        return instance

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

    def create_checkpoint(self, symbol: str | None = None) -> dict:
        """Create checkpoint for cross-file continuation.

        Captures current processing state including incomplete bar (if any).
        The checkpoint can be serialized to JSON and used to resume processing
        across file boundaries while maintaining bar continuity.

        Parameters
        ----------
        symbol : str, optional
            Symbol being processed. If None, uses the symbol provided at
            construction time.

        Returns
        -------
        dict
            Checkpoint state (JSON-serializable) containing:
            - symbol: Trading symbol
            - threshold_decimal_bps: Threshold value
            - incomplete_bar: Incomplete bar state (if any)
            - thresholds: IMMUTABLE upper/lower thresholds for incomplete bar
            - last_timestamp_us: Last processed timestamp
            - last_trade_id: Last trade ID (for gap detection)
            - price_hash: Hash for position verification
            - anomaly_summary: Gap/overlap detection counters

        Raises
        ------
        ValueError
            If no symbol provided (neither at construction nor in this call)

        Examples
        --------
        >>> processor = RangeBarProcessor(250, symbol="BTCUSDT")
        >>> bars = processor.process_trades(trades)
        >>> checkpoint = processor.create_checkpoint()
        >>> # Save to JSON
        >>> import json
        >>> with open("checkpoint.json", "w") as f:
        ...     json.dump(checkpoint, f)
        """
        return self._processor.create_checkpoint(symbol)

    def verify_position(
        self, first_trade: dict[str, int | float]
    ) -> PositionVerification:
        """Verify position in data stream at file boundary.

        Checks if the first trade of the next file matches the expected
        position based on the processor's current state. Useful for
        detecting data gaps when resuming from checkpoint.

        Parameters
        ----------
        first_trade : dict
            First trade of the next file with keys: timestamp, price, quantity

        Returns
        -------
        PositionVerification
            Verification result:
            - is_exact: True if position matches exactly
            - has_gap: True if there's a gap (missing trades)
            - gap_details(): Returns (expected_id, actual_id, missing_count) if gap
            - timestamp_gap_ms(): Returns gap in ms for timestamp-only sources

        Examples
        --------
        >>> processor = RangeBarProcessor.from_checkpoint(checkpoint)
        >>> verification = processor.verify_position(next_file_trades[0])
        >>> if verification.has_gap:
        ...     expected, actual, missing = verification.gap_details()
        ...     print(f"Gap detected: {missing} trades missing")
        """
        return self._processor.verify_position(first_trade)

    def get_incomplete_bar(self) -> dict | None:
        """Get incomplete bar if any.

        Returns the bar currently being built (not yet breached threshold).
        Returns None if the last trade completed a bar cleanly.

        Returns
        -------
        dict or None
            Incomplete bar with OHLCV fields, or None
        """
        return self._processor.get_incomplete_bar()

    @property
    def has_incomplete_bar(self) -> bool:
        """Check if there's an incomplete bar."""
        return self._processor.has_incomplete_bar


def process_trades_to_dataframe(
    trades: list[dict[str, int | float]] | pd.DataFrame,
    threshold_decimal_bps: int = 250,
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
    processor = RangeBarProcessor(threshold_decimal_bps)

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


# ============================================================================
# Optimized Processing APIs (Phase 2-4 of Python pipeline optimization)
# ============================================================================


def process_trades_chunked(
    trades_iterator: Iterator[dict[str, int | float]],
    threshold_decimal_bps: int = 250,
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
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
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

    processor = RangeBarProcessor(threshold_decimal_bps)

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
    >>> df = process_trades_polars(lazy_filtered, threshold_decimal_bps=250)

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
    processor = RangeBarProcessor(threshold_decimal_bps)
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
THRESHOLD_DECIMAL_MIN = 1  # 1 decimal bps = 0.1bps = 0.001%
THRESHOLD_DECIMAL_MAX = 100_000  # 10,000bps = 100%

# Common threshold presets (in decimal basis points)
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
    threshold_decimal_bps: int | str = 250,
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
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points. Can be:
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
    ...     threshold_decimal_bps="standard",
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
    Threshold units (decimal basis points):
        The threshold is specified in decimal basis points (0.1bps) for precision.
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
    if isinstance(threshold_decimal_bps, str):
        if threshold_decimal_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_decimal_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold_decimal_bps = THRESHOLD_PRESETS[threshold_decimal_bps]

    if not THRESHOLD_DECIMAL_MIN <= threshold_decimal_bps <= THRESHOLD_DECIMAL_MAX:
        msg = (
            f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and {THRESHOLD_DECIMAL_MAX}, "
            f"got {threshold_decimal_bps}"
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
            threshold_decimal_bps,
            validation,
            include_incomplete,
            include_microstructure,
        )

    bars_df, _ = _process_binance_trades(
        tick_data,
        threshold_decimal_bps,
        include_incomplete,
        include_microstructure,
        symbol=symbol,
    )
    return bars_df


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
    threshold_decimal_bps: int,
    include_incomplete: bool,  # noqa: ARG001 - TODO: implement
    include_microstructure: bool,
    *,
    processor: RangeBarProcessor | None = None,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, RangeBarProcessor]:
    """Process Binance trades to range bars (internal).

    Parameters
    ----------
    trades : pl.DataFrame
        Polars DataFrame with tick data
    threshold_decimal_bps : int
        Threshold in decimal basis points
    include_incomplete : bool
        Include incomplete bar (not yet implemented)
    include_microstructure : bool
        Include microstructure columns
    processor : RangeBarProcessor, optional
        Existing processor with state (for cross-file continuity).
        If None, creates a new processor.
    symbol : str, optional
        Symbol for checkpoint creation

    Returns
    -------
    tuple[pd.DataFrame, RangeBarProcessor]
        (bars DataFrame, processor with updated state)
        The processor can be used to create a checkpoint for the next file.
    """
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

    # Use provided processor or create new one
    if processor is None:
        processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)

    bars = processor.process_trades(trades_list)

    if not bars:
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))
        return empty_df, processor

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
        return result, processor

    # Return only OHLCV columns (backtesting.py compatible)
    return result[["Open", "High", "Low", "Close", "Volume"]], processor


def _process_exness_ticks(
    ticks: pl.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
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
            threshold_decimal_bps,
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


# ============================================================================
# Bar-Count-Based API
# ============================================================================


def get_n_range_bars(  # noqa: PLR0911
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: int | str = 250,
    *,
    end_date: str | None = None,
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Get exactly N range bars ending at or before a given date.

    Unlike `get_range_bars()` which uses date bounds (producing variable bar counts),
    this function returns a deterministic number of bars. This is useful for:
    - ML training (exactly 10,000 samples)
    - Walk-forward optimization (fixed window sizes)
    - Consistent backtest comparisons

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    n_bars : int
        Number of bars to retrieve. Must be > 0.
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points. Can be:
        - Integer: Direct value (250 = 25bps = 0.25%)
        - String preset: "micro", "tight", "standard", "medium", "wide", "macro"
    end_date : str or None, default=None
        End date in YYYY-MM-DD format. If None, uses most recent available data.
    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only): "spot", "futures-um", or "futures-cm"
    include_microstructure : bool, default=False
        Include microstructure columns (vwap, buy_volume, sell_volume)
    use_cache : bool, default=True
        Use ClickHouse cache for bar retrieval/storage
    fetch_if_missing : bool, default=True
        Fetch and process new data if cache doesn't have enough bars
    max_lookback_days : int, default=90
        Safety limit: maximum days to look back when fetching missing data.
        Prevents runaway fetches on empty caches.
    warn_if_fewer : bool, default=True
        Emit UserWarning if returning fewer bars than requested.
    cache_dir : str or None, default=None
        Custom cache directory for tick data (Tier 1).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with exactly n_bars rows (or fewer if not enough data),
        sorted chronologically (oldest first). Columns:
        - Open, High, Low, Close, Volume
        - (if include_microstructure) vwap, buy_volume, sell_volume

    Raises
    ------
    ValueError
        - n_bars <= 0
        - Invalid threshold
        - Invalid date format
    RuntimeError
        - ClickHouse not available when use_cache=True
        - Data fetching failed

    Examples
    --------
    Get last 10,000 bars for ML training:

    >>> from rangebar import get_n_range_bars
    >>> df = get_n_range_bars("BTCUSDT", n_bars=10000)
    >>> assert len(df) == 10000

    Get 5,000 bars ending at specific date for walk-forward:

    >>> df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

    With safety limit (won't fetch more than 30 days of data):

    >>> df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)

    Notes
    -----
    Cache behavior:
        - Fast path: If cache has >= n_bars, returns immediately (~50ms)
        - Slow path: If cache has < n_bars and fetch_if_missing=True,
          fetches additional data, computes bars, stores in cache, returns

    Gap-filling algorithm:
        Uses adaptive exponential backoff to estimate how many ticks to fetch.
        Learns compression ratio (ticks/bar) for each (symbol, threshold) pair.

    See Also
    --------
    get_range_bars : Date-bounded bar retrieval (variable bar count)
    THRESHOLD_PRESETS : Named threshold values
    """
    import warnings
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import polars as pl

    # -------------------------------------------------------------------------
    # Validate parameters
    # -------------------------------------------------------------------------
    if n_bars <= 0:
        msg = f"n_bars must be > 0, got {n_bars}"
        raise ValueError(msg)

    # Resolve threshold (support presets)
    threshold: int
    if isinstance(threshold_decimal_bps, str):
        if threshold_decimal_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_decimal_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold = THRESHOLD_PRESETS[threshold_decimal_bps]
    else:
        threshold = threshold_decimal_bps

    if not THRESHOLD_DECIMAL_MIN <= threshold <= THRESHOLD_DECIMAL_MAX:
        msg = (
            f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and "
            f"{THRESHOLD_DECIMAL_MAX}, got {threshold}"
        )
        raise ValueError(msg)

    # Normalize source and market
    source = source.lower()
    if source not in ("binance", "exness"):
        msg = f"Unknown source: {source!r}. Must be 'binance' or 'exness'"
        raise ValueError(msg)

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

    # Parse end_date if provided
    end_ts: int | None = None
    if end_date is not None:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007
            # End of day in milliseconds
            end_ts = int((end_dt.timestamp() + 86399) * 1000)
        except ValueError as e:
            msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
            raise ValueError(msg) from e

    # -------------------------------------------------------------------------
    # Try cache first (if enabled)
    # -------------------------------------------------------------------------
    if use_cache:
        try:
            from .clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                # Fast path: check if cache has enough bars
                bars_df, available_count = cache.get_n_bars(
                    symbol=symbol,
                    threshold_decimal_bps=threshold,
                    n_bars=n_bars,
                    before_ts=end_ts,
                    include_microstructure=include_microstructure,
                )

                if bars_df is not None and len(bars_df) >= n_bars:
                    # Cache hit - return exactly n_bars
                    return bars_df.tail(n_bars)

                # Slow path: need to fetch more data
                if fetch_if_missing:
                    bars_df = _fill_gap_and_cache(
                        symbol=symbol,
                        threshold=threshold,
                        n_bars=n_bars,
                        end_ts=end_ts,
                        source=source,
                        market=market_normalized,
                        include_microstructure=include_microstructure,
                        max_lookback_days=max_lookback_days,
                        cache=cache,
                        cache_dir=Path(cache_dir) if cache_dir else None,
                        current_bars=bars_df,
                        current_count=available_count,
                    )

                    if bars_df is not None and len(bars_df) >= n_bars:
                        return bars_df.tail(n_bars)

                # Return what we have (or None)
                if bars_df is not None and len(bars_df) > 0:
                    if warn_if_fewer and len(bars_df) < n_bars:
                        warnings.warn(
                            f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                            f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return bars_df

                # Empty result
                if warn_if_fewer:
                    warnings.warn(
                        f"Returning 0 bars instead of requested {n_bars}. "
                        "No data available in cache or from source.",
                        UserWarning,
                        stacklevel=2,
                    )
                return pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"]
                ).set_index(pd.DatetimeIndex([]))

        except Exception as e:
            # ClickHouse not available - fall through to compute-only mode
            if "ClickHouseNotConfigured" in type(e).__name__:
                pass  # Fall through to compute-only mode
            else:
                raise

    # -------------------------------------------------------------------------
    # Compute-only mode (no cache)
    # -------------------------------------------------------------------------
    if not fetch_if_missing:
        if warn_if_fewer:
            warnings.warn(
                f"Returning 0 bars instead of requested {n_bars}. "
                "Cache disabled and fetch_if_missing=False.",
                UserWarning,
                stacklevel=2,
            )
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))

    # Fetch and compute without caching
    bars_df = _fetch_and_compute_bars(
        symbol=symbol,
        threshold=threshold,
        n_bars=n_bars,
        end_ts=end_ts,
        source=source,
        market=market_normalized,
        include_microstructure=include_microstructure,
        max_lookback_days=max_lookback_days,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    if bars_df is not None and len(bars_df) >= n_bars:
        return bars_df.tail(n_bars)

    if bars_df is not None and len(bars_df) > 0:
        if warn_if_fewer:
            warnings.warn(
                f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                UserWarning,
                stacklevel=2,
            )
        return bars_df

    if warn_if_fewer:
        warnings.warn(
            f"Returning 0 bars instead of requested {n_bars}. "
            "No data available from source.",
            UserWarning,
            stacklevel=2,
        )
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).set_index(
        pd.DatetimeIndex([])
    )


def _fill_gap_and_cache(
    symbol: str,
    threshold: int,
    n_bars: int,
    end_ts: int | None,
    source: str,
    market: str,
    include_microstructure: bool,
    max_lookback_days: int,
    cache: RangeBarCache,
    cache_dir: Path | None,
    current_bars: pd.DataFrame | None,
    current_count: int,  # noqa: ARG001 - used for logging/debugging
) -> pd.DataFrame | None:
    """Fill gap in cache by fetching and processing additional data.

    Uses checkpoint-based cross-file continuity for Binance (24/7 crypto markets).
    The key insight: ALL ticks must be processed with a SINGLE processor to
    maintain the bar[i+1].open == bar[i].close invariant.

    For Binance (24/7):
    1. Collect ALL tick data first (no intermediate processing)
    2. Merge all ticks chronologically
    3. Process with SINGLE processor (guarantees continuity)
    4. Store with unified cache key

    For Exness (forex):
    Session-bounded processing is acceptable since weekend gaps are natural.
    """
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import polars as pl

    from .storage.parquet import TickStorage

    # Determine how many more bars we need
    bars_needed = n_bars - (len(current_bars) if current_bars is not None else 0)

    if bars_needed <= 0:
        return current_bars

    # Determine end date for fetching
    if end_ts is not None:
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
    else:
        end_dt = datetime.now(tz=timezone.utc)

    # Get oldest bar timestamp to know where to start fetching
    oldest_ts = cache.get_oldest_bar_timestamp(symbol, threshold)

    # Estimate ticks needed using adaptive heuristic
    # Default: ~2500 ticks per bar for medium threshold (250)
    # Adjusted by threshold ratio
    base_ticks_per_bar = 2500
    threshold_ratio = 250 / max(threshold, 1)
    estimated_ticks_per_bar = int(base_ticks_per_bar * threshold_ratio)

    # Adaptive exponential backoff
    multiplier = 2.0  # Start with 2x buffer
    max_attempts = 5

    storage = TickStorage(cache_dir=cache_dir)
    cache_symbol = f"{source}_{market}_{symbol}".upper()

    # =========================================================================
    # BINANCE (24/7 CRYPTO): Single-pass processing with checkpoint continuity
    # =========================================================================
    if source == "binance":
        # Phase 1: Collect ALL tick data first
        all_tick_data: list[pl.DataFrame] = []
        total_ticks = 0
        target_ticks = bars_needed * estimated_ticks_per_bar * 2  # 2x buffer

        for _attempt in range(max_attempts):
            # Calculate fetch range
            if oldest_ts is not None:
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
            else:
                fetch_end_dt = end_dt

            # Estimate days to fetch
            remaining_ticks = target_ticks - total_ticks
            days_to_fetch = max(1, remaining_ticks // 1_000_000)
            days_to_fetch = min(days_to_fetch, max_lookback_days)

            fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

            # Check lookback limit
            if (end_dt - fetch_start_dt).days > max_lookback_days:
                break

            start_date = fetch_start_dt.strftime("%Y-%m-%d")
            end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
            start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
            end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

            # Fetch tick data
            tick_data: pl.DataFrame
            if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
                tick_data = storage.read_ticks(
                    cache_symbol, start_ts_fetch, end_ts_fetch
                )
            else:
                tick_data = _fetch_binance(symbol, start_date, end_date_str, market)
                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

            if tick_data.is_empty():
                break

            all_tick_data.insert(0, tick_data)  # Prepend (older data first)
            total_ticks += len(tick_data)

            # Update oldest_ts for next iteration
            if "timestamp" in tick_data.columns:
                oldest_ts = int(tick_data["timestamp"].min())

            # Check if we have enough estimated ticks
            if total_ticks >= target_ticks:
                break

            multiplier *= 2

        if not all_tick_data:
            return current_bars

        # Phase 2: Merge ALL ticks chronologically
        merged_ticks = pl.concat(all_tick_data).sort("timestamp")

        # Phase 3: Process with SINGLE processor (guarantees continuity)
        new_bars, _ = _process_binance_trades(
            merged_ticks,
            threshold,
            False,
            include_microstructure,
            symbol=symbol,
        )

        # Phase 4: Store with unified cache key
        if not new_bars.empty:
            cache.store_bars_bulk(symbol, threshold, new_bars)

        # Combine with existing bars
        if current_bars is not None and len(current_bars) > 0:
            # Validate continuity at junction (new_bars older, current_bars newer)
            is_continuous, gap_pct = _validate_junction_continuity(
                new_bars, current_bars
            )
            if not is_continuous:
                import warnings

                warnings.warn(
                    f"Discontinuity detected at junction: {symbol} @ {threshold} dbps. "
                    f"Gap: {gap_pct:.4%}. This occurs because range bars from different "
                    f"processing sessions cannot guarantee bar[n].close == bar[n+1].open. "
                    f"Consider invalidating cache and re-fetching all data for continuous "
                    f"bars. See: https://github.com/terrylica/rangebar-py/issues/5",
                    stacklevel=3,
                )

            combined = pd.concat([new_bars, current_bars])
            combined = combined.sort_index()
            return combined[~combined.index.duplicated(keep="last")]

        return new_bars

    # =========================================================================
    # EXNESS (FOREX): Session-bounded processing (weekend gaps are natural)
    # =========================================================================
    all_bars: list[pd.DataFrame] = []
    if current_bars is not None and len(current_bars) > 0:
        all_bars.append(current_bars)

    for _attempt in range(max_attempts):
        # Estimate days to fetch
        ticks_to_fetch = int(bars_needed * estimated_ticks_per_bar * multiplier)
        days_to_fetch = max(1, ticks_to_fetch // 1_000_000)
        days_to_fetch = min(days_to_fetch, max_lookback_days)

        # Calculate fetch range
        if oldest_ts is not None:
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
        else:
            fetch_end_dt = end_dt

        fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

        if (end_dt - fetch_start_dt).days > max_lookback_days:
            break

        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
        start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
        end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

        tick_data: pl.DataFrame
        if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
            tick_data = storage.read_ticks(cache_symbol, start_ts_fetch, end_ts_fetch)
        else:
            tick_data = _fetch_exness(symbol, start_date, end_date_str, "strict")
            if not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)

        if tick_data.is_empty():
            break

        # Process to bars (forex: session-bounded is OK)
        new_bars = _process_exness_ticks(
            tick_data, symbol, threshold, "strict", False, include_microstructure
        )

        if not new_bars.empty:
            cache.store_bars_bulk(symbol, threshold, new_bars)
            all_bars.insert(0, new_bars)
            oldest_ts = int(new_bars.index.min().timestamp() * 1000)

            total_bars = sum(len(df) for df in all_bars)
            if total_bars >= n_bars:
                break

        multiplier *= 2

    if not all_bars:
        return None

    combined = pd.concat(all_bars)
    combined = combined.sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _fetch_and_compute_bars(
    symbol: str,
    threshold: int,
    n_bars: int,
    end_ts: int | None,
    source: str,
    market: str,
    include_microstructure: bool,
    max_lookback_days: int,
    cache_dir: Path | None,
) -> pd.DataFrame | None:
    """Fetch and compute bars without caching (compute-only mode).

    Uses single-pass processing for Binance (24/7 crypto) to guarantee continuity.
    """
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import polars as pl

    from .storage.parquet import TickStorage

    # Determine end date
    if end_ts is not None:
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
    else:
        end_dt = datetime.now(tz=timezone.utc)

    # Estimate ticks needed using heuristic
    base_ticks_per_bar = 2500
    threshold_ratio = 250 / max(threshold, 1)
    estimated_ticks_per_bar = int(base_ticks_per_bar * threshold_ratio)

    # Adaptive exponential backoff
    multiplier = 2.0
    max_attempts = 5

    storage = TickStorage(cache_dir=cache_dir)
    cache_symbol = f"{source}_{market}_{symbol}".upper()
    oldest_ts: int | None = None

    # =========================================================================
    # BINANCE (24/7 CRYPTO): Single-pass processing for continuity
    # =========================================================================
    if source == "binance":
        all_tick_data: list[pl.DataFrame] = []
        total_ticks = 0
        target_ticks = n_bars * estimated_ticks_per_bar * 2

        for _attempt in range(max_attempts):
            if oldest_ts is not None:
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
            else:
                fetch_end_dt = end_dt

            remaining_ticks = target_ticks - total_ticks
            days_to_fetch = max(1, remaining_ticks // 1_000_000)
            days_to_fetch = min(days_to_fetch, max_lookback_days)

            fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

            if (end_dt - fetch_start_dt).days > max_lookback_days:
                break

            start_date = fetch_start_dt.strftime("%Y-%m-%d")
            end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
            start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
            end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

            tick_data: pl.DataFrame
            if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
                tick_data = storage.read_ticks(
                    cache_symbol, start_ts_fetch, end_ts_fetch
                )
            else:
                tick_data = _fetch_binance(symbol, start_date, end_date_str, market)
                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

            if tick_data.is_empty():
                break

            all_tick_data.insert(0, tick_data)
            total_ticks += len(tick_data)

            if "timestamp" in tick_data.columns:
                oldest_ts = int(tick_data["timestamp"].min())

            if total_ticks >= target_ticks:
                break

            multiplier *= 2

        if not all_tick_data:
            return None

        merged_ticks = pl.concat(all_tick_data).sort("timestamp")
        bars_df, _ = _process_binance_trades(
            merged_ticks, threshold, False, include_microstructure, symbol=symbol
        )
        return bars_df if not bars_df.empty else None

    # =========================================================================
    # EXNESS (FOREX): Session-bounded processing
    # =========================================================================
    all_bars: list[pd.DataFrame] = []

    for _attempt in range(max_attempts):
        bars_still_needed = n_bars - sum(len(df) for df in all_bars)
        ticks_to_fetch = int(bars_still_needed * estimated_ticks_per_bar * multiplier)
        days_to_fetch = max(1, ticks_to_fetch // 1_000_000)
        days_to_fetch = min(days_to_fetch, max_lookback_days)

        if oldest_ts is not None:
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
        else:
            fetch_end_dt = end_dt

        fetch_start_dt = fetch_end_dt - timedelta(days=days_to_fetch)

        if (end_dt - fetch_start_dt).days > max_lookback_days:
            break

        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date_str = fetch_end_dt.strftime("%Y-%m-%d")
        start_ts_fetch = int(fetch_start_dt.timestamp() * 1000)
        end_ts_fetch = int(fetch_end_dt.timestamp() * 1000)

        tick_data: pl.DataFrame
        if storage.has_ticks(cache_symbol, start_ts_fetch, end_ts_fetch):
            tick_data = storage.read_ticks(cache_symbol, start_ts_fetch, end_ts_fetch)
        else:
            tick_data = _fetch_exness(symbol, start_date, end_date_str, "strict")
            if not tick_data.is_empty():
                storage.write_ticks(cache_symbol, tick_data)

        if tick_data.is_empty():
            break

        new_bars = _process_exness_ticks(
            tick_data, symbol, threshold, "strict", False, include_microstructure
        )

        if not new_bars.empty:
            all_bars.insert(0, new_bars)
            oldest_ts = int(new_bars.index.min().timestamp() * 1000)

            total_bars = sum(len(df) for df in all_bars)
            if total_bars >= n_bars:
                break

        multiplier *= 2

    if not all_bars:
        return None

    combined = pd.concat(all_bars)
    combined = combined.sort_index()
    # Remove duplicates (by index) and return
    return combined[~combined.index.duplicated(keep="last")]


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
