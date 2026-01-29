# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M2 - Extract RangeBarProcessor from __init__.py
"""RangeBarProcessor: Core processor for converting tick data to range bars.

This module contains the RangeBarProcessor class, which wraps the Rust-based
PyRangeBarProcessor to provide a Pythonic interface for range bar construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from rangebar._core import PositionVerification
from rangebar._core import PyRangeBarProcessor as _PyRangeBarProcessor

if TYPE_CHECKING:
    from arro3.core import RecordBatch as PyRecordBatch


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
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention (Issue #36).
        If True (default): A bar cannot close on the same timestamp it opened.
        This prevents flash crash scenarios from creating thousands of bars
        at identical timestamps. If False: Legacy v8 behavior where bars can
        close immediately on breach regardless of timestamp.

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

    def __init__(
        self,
        threshold_decimal_bps: int,
        symbol: str | None = None,
        *,
        prevent_same_timestamp_close: bool = True,
    ) -> None:
        """Initialize processor with given threshold.

        Parameters
        ----------
        threshold_decimal_bps : int
            Threshold in decimal basis points (250 = 25bps = 0.25%)
        symbol : str, optional
            Trading symbol for checkpoint creation
        prevent_same_timestamp_close : bool, default=True
            Timestamp gating for flash crash prevention (Issue #36)
        """
        # Validation happens in Rust layer, which raises PyValueError
        self._processor = _PyRangeBarProcessor(
            threshold_decimal_bps, symbol, prevent_same_timestamp_close
        )
        self.threshold_decimal_bps = threshold_decimal_bps
        self.symbol = symbol
        self.prevent_same_timestamp_close = prevent_same_timestamp_close

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
        # Default to True for old checkpoints without this field
        instance.prevent_same_timestamp_close = checkpoint.get(
            "prevent_same_timestamp_close", True
        )
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

    def process_trades_streaming(
        self, trades: list[dict[str, int | float]]
    ) -> list[dict[str, str | float | int]]:
        """Process trades into range bars (streaming mode - preserves state).

        Unlike `process_trades()`, this method maintains processor state across
        calls, enabling continuous processing across multiple batches (e.g.,
        month-by-month or chunk-by-chunk processing).

        Use this method for:
        - Multi-month precomputation (Issue #16)
        - Chunked processing of large datasets
        - Any scenario requiring bar continuity across batches

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
            List of range bar dictionaries (only completed bars).
            Same structure as process_trades().

        Notes
        -----
        State persistence: The processor remembers the incomplete bar from
        the previous call. When new trades arrive, they continue building
        that bar until threshold breach, ensuring continuity.

        Examples
        --------
        >>> processor = RangeBarProcessor(250)
        >>> # First batch (month 1)
        >>> bars1 = processor.process_trades_streaming(month1_trades)
        >>> # Second batch (month 2) - continues from month 1's state
        >>> bars2 = processor.process_trades_streaming(month2_trades)
        >>> # No discontinuity at month boundary
        """
        if not trades:
            return []

        return self._processor.process_trades_streaming(trades)

    def to_dataframe(
        self,
        bars: list[dict[str, str | float | int]],
        include_microstructure: bool = False,
    ) -> pd.DataFrame:
        """Convert range bars to pandas DataFrame (backtesting.py compatible).

        Parameters
        ----------
        bars : List[Dict]
            List of range bar dictionaries from process_trades()
        include_microstructure : bool, default=False
            If True, include all microstructure columns (vwap, buy_volume,
            sell_volume, ofi, kyle_lambda_proxy, etc.)

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and OHLCV columns:
            - Index: timestamp (DatetimeIndex)
            - Columns: Open, High, Low, Close, Volume
            - (if include_microstructure) Additional microstructure columns

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

        if include_microstructure:
            # Return all columns including microstructure
            return result

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

    def process_trades_streaming_arrow(
        self, trades: list[dict[str, int | float]]
    ) -> PyRecordBatch:
        """Process trades into range bars, returning Arrow RecordBatch.

        This is the most memory-efficient streaming API. Returns Arrow
        RecordBatch for zero-copy transfer to Polars or other Arrow-compatible
        systems.

        Parameters
        ----------
        trades : List[Dict]
            List of trade dictionaries with keys:
            - timestamp: int (milliseconds since epoch)
            - price: float
            - quantity: float (or 'volume')

        Returns
        -------
        PyRecordBatch
            Arrow RecordBatch with 30 columns (OHLCV + microstructure).
            Use `polars.from_arrow()` for zero-copy conversion.

        Examples
        --------
        >>> import polars as pl
        >>> processor = RangeBarProcessor(250)
        >>> for trade_batch in stream_binance_trades(  # doctest: +SKIP
        ...     "BTCUSDT", "2024-01-01", "2024-01-01"
        ... ):
        ...     arrow_batch = processor.process_trades_streaming_arrow(trade_batch)
        ...     df = pl.from_arrow(arrow_batch)  # Zero-copy!
        ...     process_batch(df)

        Notes
        -----
        Requires the `arrow-export` feature to be enabled (default in v8.0+).
        """
        if not trades:
            # Return empty batch with correct schema
            from rangebar._core import bars_to_arrow

            return bars_to_arrow([])

        return self._processor.process_trades_streaming_arrow(trades)

    def reset_at_ouroboros(self) -> dict | None:
        """Reset processor state at an ouroboros boundary.

        Clears the incomplete bar and position tracking while preserving
        the threshold configuration. Use this when starting fresh at a
        known boundary (year/month/week) for reproducibility.

        Returns
        -------
        dict or None
            The orphaned incomplete bar (if any), or None.
            Mark returned bars with `is_orphan=True` for ML filtering.

        Examples
        --------
        >>> # At year boundary (Jan 1 00:00:00 UTC)
        >>> orphaned = processor.reset_at_ouroboros()
        >>> if orphaned:
        ...     orphaned["is_orphan"] = True
        ...     orphaned["ouroboros_boundary"] = "2024-01-01T00:00:00Z"
        ...     orphaned["reason"] = "year_boundary"
        >>> # Continue processing with clean state
        """
        return self._processor.reset_at_ouroboros()
