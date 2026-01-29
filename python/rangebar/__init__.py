# polars-exception: backtesting.py requires Pandas DataFrames for OHLCV data
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

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal

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
    # Constants (from rangebar.constants - SSoT)
    "ALL_OPTIONAL_COLUMNS",
    "ASSET_CLASS_MULTIPLIERS",
    "EXCHANGE_SESSION_COLUMNS",
    "MICROSTRUCTURE_COLUMNS",
    "MIN_VERSION_FOR_MICROSTRUCTURE",
    "MIN_VERSION_FOR_OUROBOROS",
    "SCHEMA_VERSION_MICROSTRUCTURE",
    "SCHEMA_VERSION_OHLCV_ONLY",
    "SCHEMA_VERSION_OUROBOROS",
    "THRESHOLD_DECIMAL_MAX",
    "THRESHOLD_DECIMAL_MIN",
    "THRESHOLD_PRESETS",
    "TIER1_SYMBOLS",
    "VALIDATION_PRESETS",
    # Core classes
    "AssetClass",
    "ContinuityError",
    "ContinuityWarning",
    "GapInfo",
    "GapTier",
    "OrphanedBarMetadata",
    "OuroborosBoundary",
    "OuroborosMode",
    "PositionVerification",
    "PrecomputeProgress",
    "PrecomputeResult",
    "RangeBarProcessor",
    "StalenessResult",
    "TierSummary",
    "TierThresholds",
    "TieredValidationResult",
    "ValidationPreset",
    "__version__",
    # Functions
    "detect_asset_class",
    "detect_staleness",
    "get_n_range_bars",
    "get_ouroboros_boundaries",
    "get_range_bars",
    "get_range_bars_pandas",
    # Conversion utilities (from rangebar.conversion - SSoT)
    "normalize_arrow_dtypes",
    "normalize_temporal_precision",
    "populate_cache_resumable",
    "precompute_range_bars",
    "process_trades_polars",
    "process_trades_to_dataframe",
    "validate_continuity",
    "validate_continuity_tiered",
]

# Re-export checkpoint API per plan (#40)
from datetime import UTC

from .checkpoint import populate_cache_resumable

# Import constants from centralized module (SSoT)
from .constants import (
    _CRYPTO_BASES,
    _FOREX_CURRENCIES,
    ALL_OPTIONAL_COLUMNS,
    EXCHANGE_SESSION_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    MIN_VERSION_FOR_MICROSTRUCTURE,
    MIN_VERSION_FOR_OUROBOROS,
    SCHEMA_VERSION_MICROSTRUCTURE,
    SCHEMA_VERSION_OHLCV_ONLY,
    SCHEMA_VERSION_OUROBOROS,
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
    TIER1_SYMBOLS,
)

# Import conversion utilities from centralized module (SSoT)
from .conversion import (
    _bars_list_to_polars,
    _concat_pandas_via_polars,
    normalize_arrow_dtypes,
    normalize_temporal_precision,
)

# Re-export ouroboros API (cyclical reset boundaries for reproducibility)
from .ouroboros import (
    OrphanedBarMetadata,
    OuroborosBoundary,
    OuroborosMode,
    get_ouroboros_boundaries,
)

# Import staleness detection for cache validation (Issue #39: Schema Evolution)
from .validation.cache_staleness import StalenessResult, detect_staleness

# Continuity tolerance: 0.01% relative difference allowed (floating-point precision)
CONTINUITY_TOLERANCE_PCT = 0.0001


class ContinuityError(Exception):
    """Raised when range bar continuity is violated.

    Range bars must satisfy bar[i+1].open == bar[i].close for 24/7 crypto markets.
    This error indicates discontinuities were detected in the bar sequence.

    Attributes
    ----------
    message : str
        Human-readable error description
    discontinuities : list[dict]
        List of discontinuity details with keys: bar_index, prev_close, curr_open, gap_pct
    """

    def __init__(self, message: str, discontinuities: list[dict] | None = None) -> None:
        super().__init__(message)
        self.discontinuities = discontinuities or []


class ContinuityWarning(UserWarning):
    """Warning issued when range bar discontinuities are detected but not fatal."""


@dataclass
class PrecomputeProgress:
    """Progress update for precomputation.

    Attributes
    ----------
    phase : Literal["fetching", "processing", "caching"]
        Current phase of precomputation
    current_month : str
        Current month being processed ("YYYY-MM" format)
    months_completed : int
        Number of months completed
    months_total : int
        Total number of months to process
    bars_generated : int
        Total bars generated so far
    ticks_processed : int
        Total ticks processed so far
    elapsed_seconds : float
        Elapsed time since precomputation started
    """

    phase: Literal["fetching", "processing", "caching"]
    current_month: str
    months_completed: int
    months_total: int
    bars_generated: int
    ticks_processed: int
    elapsed_seconds: float


@dataclass
class PrecomputeResult:
    """Result of precomputation.

    Attributes
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    threshold_decimal_bps : int
        Threshold used for bar construction
    start_date : str
        Start date of precomputation ("YYYY-MM-DD")
    end_date : str
        End date of precomputation ("YYYY-MM-DD")
    total_bars : int
        Total number of bars generated
    total_ticks : int
        Total number of ticks processed
    elapsed_seconds : float
        Total elapsed time for precomputation
    continuity_valid : bool | None
        True if all bars pass continuity validation, False if not,
        None if validation was skipped
    cache_key : str
        Cache key for the generated bars
    """

    symbol: str
    threshold_decimal_bps: int
    start_date: str
    end_date: str
    total_bars: int
    total_ticks: int
    elapsed_seconds: float
    continuity_valid: bool | None
    cache_key: str


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
    tolerance_pct: float | None = None,
    threshold_decimal_bps: int = 250,
) -> dict:
    """Validate that range bars come from continuous single-session processing.

    Range bars do NOT guarantee bar[i].close == bar[i+1].open. The next bar
    opens at the first tick AFTER the previous bar closes, not at the close
    price. This is by design - range bars capture actual market movements.

    What this function validates:
    1. OHLC invariants hold (High >= max(Open, Close), Low <= min(Open, Close))
    2. Price gaps between bars don't exceed threshold + tolerance
       (gaps larger than threshold indicate bars from different sessions)
    3. Timestamps are monotonically increasing

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with OHLC columns
    tolerance_pct : float, optional
        Additional tolerance beyond threshold for gap detection.
        Default is 0.5% (0.005) to account for floating-point precision.
    threshold_decimal_bps : int, default=250
        Range bar threshold used to generate these bars (250 = 0.25%).
        Gaps larger than this indicate session boundaries.

    Returns
    -------
    dict
        Validation result with keys:
        - is_valid: bool - True if bars appear from single session
        - bar_count: int - Total number of bars
        - discontinuity_count: int - Number of session boundaries found
        - discontinuities: list[dict] - Details of each discontinuity

    Notes
    -----
    A "discontinuity" here means bars from different processing sessions
    were combined. Within a single session, the gap between bar[i].close
    and bar[i+1].open should never exceed the threshold (since a bar only
    closes when price moves by threshold from open).

    The tolerance parameter accounts for:
    - Floating-point precision in price calculations
    - Minor price movements between close tick and next tick
    """
    if tolerance_pct is None:
        tolerance_pct = 0.005  # 0.5% default tolerance

    if df.empty:
        return {
            "is_valid": True,
            "bar_count": 0,
            "discontinuity_count": 0,
            "discontinuities": [],
        }

    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        msg = f"DataFrame must have columns: {required_cols}"
        raise ValueError(msg)

    discontinuities = []

    # Convert threshold to percentage (250 dbps = 0.25% = 0.0025)
    threshold_pct = threshold_decimal_bps / 100000.0

    # Maximum allowed gap = threshold + tolerance
    # Within single session, gap should never exceed this
    max_gap_pct = threshold_pct + tolerance_pct

    close_prices = df["Close"].to_numpy()[:-1]
    open_prices = df["Open"].to_numpy()[1:]

    for i, (prev_close, curr_open) in enumerate(
        zip(close_prices, open_prices, strict=False)
    ):
        if prev_close == 0:
            continue

        gap_pct = abs(curr_open - prev_close) / abs(prev_close)
        if gap_pct > max_gap_pct:
            discontinuities.append(
                {
                    "bar_index": i + 1,
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
        >>> for trade_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-01"):
        ...     arrow_batch = processor.process_trades_streaming_arrow(trade_batch)
        ...     df = pl.from_arrow(arrow_batch)  # Zero-copy!
        ...     process_batch(df)

        Notes
        -----
        Requires the `arrow-export` feature to be enabled (default in v8.0+).
        """
        if not trades:
            # Return empty batch with correct schema
            from ._core import bars_to_arrow

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

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    # This enables predicate pushdown and avoids materializing unused columns
    # Memory impact: 10-100x reduction depending on filter selectivity

    # Determine volume column name (works for both DataFrame and LazyFrame)
    if isinstance(trades, pl.LazyFrame):
        available_cols = trades.collect_schema().names()
    else:
        available_cols = trades.columns

    volume_col = "quantity" if "quantity" in available_cols else "volume"

    # Build column list - include is_buyer_maker for microstructure features (Issue #30)
    columns = [
        pl.col("timestamp"),
        pl.col("price"),
        pl.col(volume_col).alias("quantity"),
    ]
    if "is_buyer_maker" in available_cols:
        columns.append(pl.col("is_buyer_maker"))

    # Apply selection (predicates pushed down for LazyFrame)
    trades_selected = trades.select(columns)

    # Collect AFTER selection (for LazyFrame)
    if isinstance(trades_selected, pl.LazyFrame):
        trades_minimal = trades_selected.collect()
    else:
        trades_minimal = trades_selected

    # MEM-002: Process in chunks to bound memory (2.5 GB → ~50 MB per chunk)
    # Chunked .to_dicts() avoids materializing 1M+ trade dicts at once
    chunk_size = 100_000
    processor = RangeBarProcessor(threshold_decimal_bps)
    all_bars: list[dict] = []

    n_rows = len(trades_minimal)
    for start in range(0, n_rows, chunk_size):
        chunk = trades_minimal.slice(start, chunk_size).to_dicts()
        bars = processor.process_trades_streaming(chunk)
        all_bars.extend(bars)

    return processor.to_dataframe(all_bars)


# Re-export ClickHouse components for convenience
# NOTE: TIER1_SYMBOLS, THRESHOLD_PRESETS, THRESHOLD_DECIMAL_MIN/MAX
# are imported from constants.py (SSoT) at the top of this file


# =============================================================================
# Tiered Validation System (Issue #19 - v6.2.0+)
# =============================================================================


class GapTier(IntEnum):
    """Gap severity classification for range bar continuity validation.

    Tiers are based on empirical analysis of 30-month BTC data (Issue #19)
    which identified 49 legitimate market microstructure events.

    Examples
    --------
    >>> gap_pct = 0.05  # 0.05% gap
    >>> if gap_pct < 0.00001:
    ...     tier = GapTier.PRECISION
    >>> elif gap_pct < 0.0001:
    ...     tier = GapTier.NOISE
    >>> elif gap_pct < 0.001:
    ...     tier = GapTier.MARKET_MOVE
    >>> elif gap_pct < threshold * 2:
    ...     tier = GapTier.MICROSTRUCTURE
    >>> else:
    ...     tier = GapTier.SESSION_BOUNDARY
    """

    PRECISION = 1  # < 0.001% - Floating-point artifacts (always ignored)
    NOISE = 2  # 0.001% - 0.01% - Tick-level noise (logged, not flagged)
    MARKET_MOVE = 3  # 0.01% - 0.1% - Normal market movement (configurable)
    MICROSTRUCTURE = 4  # > 0.1% - Flash crashes, liquidations (warning/error)
    SESSION_BOUNDARY = 5  # > threshold*2 - Definite session break (always error)


class AssetClass(Enum):
    """Asset class for tolerance calibration.

    Different asset classes have different typical gap magnitudes:
    - Crypto: 24/7 markets, flash crashes possible, baseline tolerance
    - Forex: Session-based, weekend gaps, tighter tolerance
    - Equities: Overnight gaps, circuit breakers, looser tolerance

    Examples
    --------
    >>> from rangebar import detect_asset_class, AssetClass
    >>> detect_asset_class("BTCUSDT")
    <AssetClass.CRYPTO: 'crypto'>
    >>> detect_asset_class("EURUSD")
    <AssetClass.FOREX: 'forex'>
    """

    CRYPTO = "crypto"  # 24/7 markets, flash crashes possible
    FOREX = "forex"  # Session-based, weekend gaps
    EQUITIES = "equities"  # Overnight gaps, circuit breakers
    UNKNOWN = "unknown"  # Fallback to crypto defaults


# Tolerance multipliers by asset class (relative to baseline)
ASSET_CLASS_MULTIPLIERS: dict[AssetClass, float] = {
    AssetClass.CRYPTO: 1.0,  # Baseline
    AssetClass.FOREX: 0.5,  # Tighter (more stable)
    AssetClass.EQUITIES: 1.5,  # Looser (overnight gaps)
    AssetClass.UNKNOWN: 1.0,  # Default to crypto
}

# NOTE: _CRYPTO_BASES and _FOREX_CURRENCIES are imported from constants.py (SSoT)


def detect_asset_class(symbol: str) -> AssetClass:
    """Auto-detect asset class from symbol pattern.

    Detection Rules:
    - Crypto: Contains common crypto bases (BTC, ETH, BNB, SOL, etc.)
             or ends with USDT/BUSD/USDC
    - Forex: Standard 6-char pairs (EURUSD, GBPJPY, etc.)
             or commodity symbols (XAU, XAG, BRENT, WTI)
    - Unknown: Fallback for unrecognized patterns

    Parameters
    ----------
    symbol : str
        Trading symbol (case-insensitive)

    Returns
    -------
    AssetClass
        Detected asset class

    Examples
    --------
    >>> detect_asset_class("BTCUSDT")
    <AssetClass.CRYPTO: 'crypto'>
    >>> detect_asset_class("EURUSD")
    <AssetClass.FOREX: 'forex'>
    >>> detect_asset_class("AAPL")
    <AssetClass.UNKNOWN: 'unknown'>
    """
    symbol_upper = symbol.upper()

    # Crypto patterns: contains known base or ends with stablecoin
    if any(base in symbol_upper for base in _CRYPTO_BASES):
        return AssetClass.CRYPTO
    if symbol_upper.endswith(("USDT", "BUSD", "USDC", "TUSD", "FDUSD")):
        return AssetClass.CRYPTO

    # Forex patterns: 6-char standard pairs (e.g., EURUSD)
    forex_pair_length = 6
    if len(symbol_upper) == forex_pair_length:
        base, quote = symbol_upper[:3], symbol_upper[3:]
        if base in _FOREX_CURRENCIES and quote in _FOREX_CURRENCIES:
            return AssetClass.FOREX

    # Commodities via forex brokers
    if any(x in symbol_upper for x in ("XAU", "XAG", "BRENT", "WTI")):
        return AssetClass.FOREX

    return AssetClass.UNKNOWN


@dataclass(frozen=True)
class TierThresholds:
    """Configurable boundaries between gap tiers (in percentage).

    These thresholds define the boundaries for classifying gaps into tiers.
    Values are percentages (e.g., 0.00001 = 0.001%).

    Attributes
    ----------
    precision : float
        Tier 1/2 boundary - gaps below this are floating-point artifacts
    noise : float
        Tier 2/3 boundary - gaps below this are tick-level noise
    market_move : float
        Tier 3/4 boundary - gaps below this are normal market movement
    session_factor : float
        Tier 5 multiplier - gaps > (threshold * factor) are session breaks

    Examples
    --------
    >>> thresholds = TierThresholds()
    >>> thresholds.precision
    1e-05
    >>> thresholds.noise
    0.0001
    """

    precision: float = 0.00001  # 0.001% - Tier 1/2 boundary
    noise: float = 0.0001  # 0.01% - Tier 2/3 boundary
    market_move: float = 0.001  # 0.1% - Tier 3/4 boundary
    session_factor: float = 2.0  # Tier 5 = threshold * factor


@dataclass(frozen=True)
class ValidationPreset:
    """Immutable validation configuration preset.

    Presets bundle tolerance, behavior mode, and tier thresholds into
    named configurations for common use cases.

    Attributes
    ----------
    tolerance_pct : float
        Maximum gap percentage before flagging (e.g., 0.01 = 1%)
    mode : Literal["error", "warn", "skip"]
        Behavior on validation failure
    tier_thresholds : TierThresholds
        Boundaries for gap tier classification
    asset_class : AssetClass | None
        Override auto-detection if set
    description : str
        Human-readable description of the preset

    Examples
    --------
    >>> preset = VALIDATION_PRESETS["research"]
    >>> preset.tolerance_pct
    0.02
    >>> preset.mode
    'warn'
    """

    tolerance_pct: float  # Max gap before flagging
    mode: Literal["error", "warn", "skip"]  # Behavior on failure
    tier_thresholds: TierThresholds = field(default_factory=TierThresholds)
    asset_class: AssetClass | None = None  # Override auto-detection
    description: str = ""


# Named validation presets for common scenarios
VALIDATION_PRESETS: dict[str, ValidationPreset] = {
    # =========================================================================
    # GENERAL-PURPOSE PRESETS
    # =========================================================================
    "permissive": ValidationPreset(
        tolerance_pct=0.05,  # 5%
        mode="warn",
        description="Accept most microstructure events, warn on extreme gaps",
    ),
    "research": ValidationPreset(
        tolerance_pct=0.02,  # 2%
        mode="warn",
        description="Standard exploratory analysis with monitoring",
    ),
    "standard": ValidationPreset(
        tolerance_pct=0.01,  # 1%
        mode="warn",
        description="Balanced tolerance for production backtesting",
    ),
    "strict": ValidationPreset(
        tolerance_pct=0.005,  # 0.5%
        mode="error",
        description="Strict validation for ML training data",
    ),
    "paranoid": ValidationPreset(
        tolerance_pct=0.001,  # 0.1%
        mode="error",
        description="Maximum strictness (original v6.1.0 behavior)",
    ),
    # =========================================================================
    # ASSET-CLASS SPECIFIC PRESETS
    # =========================================================================
    "crypto": ValidationPreset(
        tolerance_pct=0.02,  # 2%
        mode="warn",
        asset_class=AssetClass.CRYPTO,
        description="Crypto: Tuned for 24/7 markets with flash crashes",
    ),
    "forex": ValidationPreset(
        tolerance_pct=0.01,  # 1%
        mode="warn",
        asset_class=AssetClass.FOREX,
        description="Forex: Accounts for session boundaries",
    ),
    "equities": ValidationPreset(
        tolerance_pct=0.03,  # 3%
        mode="warn",
        asset_class=AssetClass.EQUITIES,
        description="Equities: Accounts for overnight gaps",
    ),
    # =========================================================================
    # SPECIAL PRESETS
    # =========================================================================
    "skip": ValidationPreset(
        tolerance_pct=0.0,
        mode="skip",
        description="Disable validation entirely",
    ),
    "audit": ValidationPreset(
        tolerance_pct=0.002,  # 0.2%
        mode="error",
        description="Data quality audit mode",
    ),
}


@dataclass
class GapInfo:
    """Details of a single gap between consecutive bars.

    Attributes
    ----------
    bar_index : int
        Index of the bar with the gap (0-based)
    prev_close : float
        Close price of the previous bar
    curr_open : float
        Open price of the current bar
    gap_pct : float
        Gap magnitude as percentage (e.g., 0.01 = 1%)
    tier : GapTier
        Severity classification of this gap
    timestamp : pd.Timestamp | None
        Timestamp of the bar (if available from DataFrame index)
    """

    bar_index: int
    prev_close: float
    curr_open: float
    gap_pct: float
    tier: GapTier
    timestamp: pd.Timestamp | None = None


@dataclass
class TierSummary:
    """Per-tier statistics for gap analysis.

    Attributes
    ----------
    count : int
        Number of gaps in this tier
    max_gap_pct : float
        Maximum gap percentage in this tier
    avg_gap_pct : float
        Average gap percentage in this tier (0 if count == 0)
    """

    count: int = 0
    max_gap_pct: float = 0.0
    avg_gap_pct: float = 0.0


@dataclass
class TieredValidationResult:
    """Comprehensive validation result with tier breakdown.

    This result provides detailed gap analysis categorized by severity tier,
    enabling nuanced handling of different gap magnitudes.

    Attributes
    ----------
    is_valid : bool
        True if no SESSION_BOUNDARY gaps (tier 5) detected
    bar_count : int
        Total number of bars validated
    gaps_by_tier : dict[GapTier, TierSummary]
        Per-tier statistics
    all_gaps : list[GapInfo]
        All gaps above PRECISION tier (detailed list)
    threshold_used_pct : float
        Range bar threshold used for validation (as percentage)
    asset_class_detected : AssetClass
        Auto-detected or overridden asset class
    preset_used : str | None
        Name of preset used, or None for custom config

    Examples
    --------
    >>> result = validate_continuity_tiered(df, validation="research")
    >>> result.is_valid
    True
    >>> result.gaps_by_tier[GapTier.MICROSTRUCTURE].count
    3
    >>> result.has_microstructure_events
    True
    """

    is_valid: bool
    bar_count: int
    gaps_by_tier: dict[GapTier, TierSummary]
    all_gaps: list[GapInfo]
    threshold_used_pct: float
    asset_class_detected: AssetClass
    preset_used: str | None

    @property
    def has_session_breaks(self) -> bool:
        """True if any SESSION_BOUNDARY gaps detected."""
        return self.gaps_by_tier[GapTier.SESSION_BOUNDARY].count > 0

    @property
    def has_microstructure_events(self) -> bool:
        """True if any MICROSTRUCTURE gaps detected."""
        return self.gaps_by_tier[GapTier.MICROSTRUCTURE].count > 0

    def summary_dict(self) -> dict[str, int]:
        """Return gap counts by tier name for logging.

        Returns
        -------
        dict[str, int]
            Mapping of tier name to gap count

        Examples
        --------
        >>> result.summary_dict()
        {'PRECISION': 0, 'NOISE': 5, 'MARKET_MOVE': 10, 'MICROSTRUCTURE': 3, 'SESSION_BOUNDARY': 0}
        """
        return {tier.name: summary.count for tier, summary in self.gaps_by_tier.items()}


def _resolve_validation(
    validation: str | dict | ValidationPreset,
    symbol: str | None = None,
) -> tuple[ValidationPreset, AssetClass, str | None]:
    """Resolve validation parameter to preset and asset class.

    Parameters
    ----------
    validation : str, dict, or ValidationPreset
        Validation configuration:
        - "auto": Auto-detect asset class from symbol
        - str: Preset name ("research", "strict", "crypto", etc.)
        - dict: Custom config {"tolerance_pct": 0.01, "mode": "warn"}
        - ValidationPreset: Direct preset instance
    symbol : str, optional
        Symbol for asset class auto-detection

    Returns
    -------
    tuple[ValidationPreset, AssetClass, str | None]
        (resolved preset, detected asset class, preset name or None)

    Raises
    ------
    ValueError
        If unknown preset name provided
    TypeError
        If validation is not a supported type
    """
    # Handle "auto" - detect from symbol
    if validation == "auto":
        asset_class = detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        preset_name = (
            asset_class.value if asset_class != AssetClass.UNKNOWN else "standard"
        )
        return VALIDATION_PRESETS[preset_name], asset_class, preset_name

    # Handle preset string
    if isinstance(validation, str):
        if validation not in VALIDATION_PRESETS:
            valid_presets = ", ".join(sorted(VALIDATION_PRESETS.keys()))
            msg = f"Unknown validation preset: {validation!r}. Valid presets: {valid_presets}"
            raise ValueError(msg)
        preset = VALIDATION_PRESETS[validation]
        asset_class = preset.asset_class or (
            detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        )
        return preset, asset_class, validation

    # Handle dict
    if isinstance(validation, dict):
        tier_thresholds = validation.get("tier_thresholds", TierThresholds())
        if isinstance(tier_thresholds, dict):
            tier_thresholds = TierThresholds(**tier_thresholds)
        preset = ValidationPreset(
            tolerance_pct=validation["tolerance_pct"],
            mode=validation["mode"],
            tier_thresholds=tier_thresholds,
            description="Custom",
        )
        asset_class = detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        return preset, asset_class, None

    # Handle ValidationPreset directly
    if isinstance(validation, ValidationPreset):
        asset_class = validation.asset_class or (
            detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        )
        return validation, asset_class, None

    msg = f"Invalid validation type: {type(validation).__name__}. Expected str, dict, or ValidationPreset"
    raise TypeError(msg)


def _classify_gap(
    gap_pct: float,
    tier_thresholds: TierThresholds,
    session_threshold_pct: float,
) -> GapTier:
    """Classify a gap into a severity tier.

    Parameters
    ----------
    gap_pct : float
        Gap magnitude as percentage (absolute value)
    tier_thresholds : TierThresholds
        Boundaries between tiers
    session_threshold_pct : float
        Session boundary threshold (threshold * session_factor)

    Returns
    -------
    GapTier
        Severity classification
    """
    if gap_pct < tier_thresholds.precision:
        return GapTier.PRECISION
    if gap_pct < tier_thresholds.noise:
        return GapTier.NOISE
    if gap_pct < tier_thresholds.market_move:
        return GapTier.MARKET_MOVE
    if gap_pct < session_threshold_pct:
        return GapTier.MICROSTRUCTURE
    return GapTier.SESSION_BOUNDARY


def validate_continuity_tiered(
    df: pd.DataFrame,
    threshold_decimal_bps: int = 250,
    *,
    validation: str | dict | ValidationPreset = "standard",
    symbol: str | None = None,
) -> TieredValidationResult:
    """Validate range bar continuity with tiered gap classification.

    This function categorizes gaps by severity tier, enabling nuanced
    handling of different gap magnitudes. It's the opt-in v6.2.0 API
    that will become the default in v7.0.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with OHLCV columns
    threshold_decimal_bps : int, default=250
        Range bar threshold (250 = 0.25% = 25 basis points)
    validation : str, dict, or ValidationPreset, default="standard"
        Validation configuration:
        - "auto": Auto-detect asset class from symbol
        - str: Preset name ("research", "strict", "crypto", etc.)
        - dict: Custom config {"tolerance_pct": 0.01, "mode": "warn"}
        - ValidationPreset: Direct preset instance
    symbol : str, optional
        Symbol for asset class auto-detection. If None and validation is
        "auto", uses "standard" preset.

    Returns
    -------
    TieredValidationResult
        Comprehensive result with per-tier statistics

    Raises
    ------
    ContinuityError
        If validation mode is "error" and tolerance exceeded
    ContinuityWarning
        If validation mode is "warn" and tolerance exceeded (via warnings module)

    Examples
    --------
    >>> result = validate_continuity_tiered(df, validation="research")
    >>> print(f"Valid: {result.is_valid}, Microstructure events: {result.has_microstructure_events}")
    Valid: True, Microstructure events: True

    >>> # Custom configuration
    >>> result = validate_continuity_tiered(
    ...     df,
    ...     validation={"tolerance_pct": 0.015, "mode": "warn"},
    ...     symbol="BTCUSDT",
    ... )

    >>> # Auto-detect asset class
    >>> result = validate_continuity_tiered(df, validation="auto", symbol="EURUSD")
    >>> result.asset_class_detected
    <AssetClass.FOREX: 'forex'>
    """
    import warnings

    min_bars_for_gap = 2  # Need at least 2 bars to have a gap
    if len(df) < min_bars_for_gap:
        # No gaps possible with fewer than 2 bars
        return TieredValidationResult(
            is_valid=True,
            bar_count=len(df),
            gaps_by_tier={tier: TierSummary() for tier in GapTier},
            all_gaps=[],
            threshold_used_pct=threshold_decimal_bps / 10000.0,
            asset_class_detected=(
                detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
            ),
            preset_used=validation if isinstance(validation, str) else None,
        )

    # Resolve validation configuration
    preset, asset_class, preset_name = _resolve_validation(validation, symbol)

    # Skip validation if mode is "skip"
    if preset.mode == "skip":
        return TieredValidationResult(
            is_valid=True,
            bar_count=len(df),
            gaps_by_tier={tier: TierSummary() for tier in GapTier},
            all_gaps=[],
            threshold_used_pct=threshold_decimal_bps / 10000.0,
            asset_class_detected=asset_class,
            preset_used=preset_name,
        )

    # Calculate session boundary threshold
    threshold_pct = threshold_decimal_bps / 10000.0  # Convert to percentage
    session_threshold_pct = threshold_pct * preset.tier_thresholds.session_factor

    # Analyze gaps
    all_gaps: list[GapInfo] = []
    tier_gaps: dict[GapTier, list[float]] = {tier: [] for tier in GapTier}

    # Get Close and Open columns
    close_col = "Close" if "Close" in df.columns else "close"
    open_col = "Open" if "Open" in df.columns else "open"

    closes = df[close_col].to_numpy()
    opens = df[open_col].to_numpy()
    index = df.index

    for i in range(1, len(df)):
        prev_close = float(closes[i - 1])
        curr_open = float(opens[i])

        if prev_close == 0:
            continue  # Skip division by zero

        gap_pct = abs(curr_open - prev_close) / prev_close
        tier = _classify_gap(gap_pct, preset.tier_thresholds, session_threshold_pct)

        tier_gaps[tier].append(gap_pct)

        # Record non-PRECISION gaps for detailed list
        if tier != GapTier.PRECISION:
            timestamp = index[i] if isinstance(index, pd.DatetimeIndex) else None
            all_gaps.append(
                GapInfo(
                    bar_index=i,
                    prev_close=prev_close,
                    curr_open=curr_open,
                    gap_pct=gap_pct,
                    tier=tier,
                    timestamp=timestamp,
                )
            )

    # Build tier summaries
    gaps_by_tier: dict[GapTier, TierSummary] = {}
    for tier in GapTier:
        gaps = tier_gaps[tier]
        if gaps:
            gaps_by_tier[tier] = TierSummary(
                count=len(gaps),
                max_gap_pct=max(gaps),
                avg_gap_pct=sum(gaps) / len(gaps),
            )
        else:
            gaps_by_tier[tier] = TierSummary()

    # Determine validity: no SESSION_BOUNDARY gaps
    is_valid = gaps_by_tier[GapTier.SESSION_BOUNDARY].count == 0

    # Check tolerance threshold
    tolerance_exceeded = any(
        gap.gap_pct > preset.tolerance_pct
        for gap in all_gaps
        if gap.tier >= GapTier.MARKET_MOVE  # Only check MARKET_MOVE and above
    )

    result = TieredValidationResult(
        is_valid=is_valid,
        bar_count=len(df),
        gaps_by_tier=gaps_by_tier,
        all_gaps=all_gaps,
        threshold_used_pct=threshold_pct,
        asset_class_detected=asset_class,
        preset_used=preset_name,
    )

    # Handle tolerance violations based on mode
    if tolerance_exceeded:
        violating_gaps = [g for g in all_gaps if g.gap_pct > preset.tolerance_pct]
        max_gap = max(violating_gaps, key=lambda g: g.gap_pct)

        msg = (
            f"Continuity tolerance exceeded: {len(violating_gaps)} gap(s) > {preset.tolerance_pct:.4%}. "
            f"Max gap: {max_gap.gap_pct:.4%} at bar {max_gap.bar_index}. "
            f"Tier breakdown: {result.summary_dict()}"
        )

        if preset.mode == "error":
            discontinuities = [
                {
                    "bar_index": g.bar_index,
                    "prev_close": g.prev_close,
                    "curr_open": g.curr_open,
                    "gap_pct": g.gap_pct,
                    "tier": g.tier.name,
                }
                for g in violating_gaps
            ]
            raise ContinuityError(msg, discontinuities)

        if preset.mode == "warn":
            warnings.warn(msg, ContinuityWarning, stacklevel=2)

    return result


def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    # Ouroboros: Cyclical reset boundaries (v11.0+)
    ouroboros: Literal["year", "month", "week"] = "year",
    include_orphaned_bars: bool = False,
    # Streaming options (v8.0+)
    materialize: bool = True,
    batch_size: int = 10_000,
    # Data source configuration
    source: str = "binance",
    market: str = "spot",
    # Exness-specific options
    validation: str = "strict",
    # Processing options
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,  # Issue #8: Exchange session flags
    # Timestamp gating (Issue #36)
    prevent_same_timestamp_close: bool = True,
    # Data integrity (Issue #43)
    verify_checksum: bool = True,
    # Caching options
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame | Iterator[pl.DataFrame]:
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
        - Integer: Direct value (250 dbps = 0.25%)
        - String preset: "micro" (10 dbps), "tight" (50 dbps), "standard" (100 dbps),
          "medium" (250 dbps), "wide" (500 dbps), "macro" (1000 dbps)
        Valid range: 1-100,000 dbps (0.001% to 100%)
    ouroboros : {"year", "month", "week"}, default="year"
        Cyclical reset boundary for reproducible bar construction (v11.0+).
        Processor state resets at each boundary for deterministic results.
        - "year" (default): Reset at January 1st 00:00:00 UTC (cryptocurrency)
        - "month": Reset at 1st of each month 00:00:00 UTC
        - "week": Reset at Sunday 00:00:00 UTC (required for Forex)
        Named after the Greek serpent eating its tail (οὐροβόρος).
    include_orphaned_bars : bool, default=False
        Include incomplete bars from ouroboros boundaries.
        If True, orphaned bars are included with ``is_orphan=True`` column.
        Useful for analysis; filter with ``df[~df.get('is_orphan', False)]``.
    materialize : bool, default=True
        If True, return a single pd.DataFrame (legacy behavior).
        If False, return an Iterator[pl.DataFrame] that yields batches
        of bars for memory-efficient streaming (v8.0+).
    batch_size : int, default=10_000
        Number of bars per batch when materialize=False.
        Each batch is ~500 KB. Only used in streaming mode.

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
    include_exchange_sessions : bool, default=False
        Include traditional exchange market session flags (Issue #8).
        When True, adds boolean columns indicating active sessions at bar close:
        - exchange_session_sydney: ASX (10:00-16:00 Sydney time)
        - exchange_session_tokyo: TSE (09:00-15:00 Tokyo time)
        - exchange_session_london: LSE (08:00-17:00 London time)
        - exchange_session_newyork: NYSE (10:00-16:00 New York time)
        Useful for analyzing crypto/forex behavior during traditional market hours.
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention (Issue #36).
        If True (default): A bar cannot close on the same timestamp it opened.
        This prevents flash crash scenarios from creating thousands of bars
        at identical timestamps. If False: Legacy v8 behavior where bars can
        close immediately on breach regardless of timestamp. Use False for
        comparative analysis between old and new behavior.
    verify_checksum : bool, default=True
        Verify SHA-256 checksum of downloaded data (Issue #43).
        If True (default): Verify downloaded ZIP files against Binance-provided
        checksums to detect data corruption early. If verification fails,
        raises RuntimeError. If False: Skip checksum verification for faster
        downloads (use when data integrity is verified elsewhere).
    use_cache : bool, default=True
        Cache tick data locally in Parquet format.
    fetch_if_missing : bool, default=True
        If True (default), fetch tick data from source when not available
        in cache. If False, return only cached data (may return empty
        DataFrame if no cached data exists for the date range).
    cache_dir : str or None, default=None
        Custom cache directory. If None, uses platform default:
        - macOS: ~/Library/Caches/rangebar/
        - Linux: ~/.cache/rangebar/
        - Windows: %LOCALAPPDATA%/terrylica/rangebar/Cache/

    Returns
    -------
    pd.DataFrame or Iterator[pl.DataFrame]
        If materialize=True (default): Single pd.DataFrame ready for
        backtesting.py, with DatetimeIndex and OHLCV columns.

        If materialize=False: Iterator yielding pl.DataFrame batches
        (batch_size bars each) for memory-efficient streaming. Convert
        to pandas with: ``pl.concat(list(iterator)).to_pandas()``

        Columns: Open, High, Low, Close, Volume
        (if include_microstructure) Additional columns

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

    Streaming mode for large datasets (v8.0+):

    >>> import polars as pl
    >>> # Memory-efficient: yields ~500 KB batches
    >>> for batch in get_range_bars(
    ...     "BTCUSDT", "2024-01-01", "2024-06-30",
    ...     materialize=False,
    ...     batch_size=10_000,
    ... ):
    ...     process_batch(batch)  # batch is pl.DataFrame
    ...
    >>> # Or collect to single DataFrame:
    >>> batches = list(get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", materialize=False))
    >>> df = pl.concat(batches).to_pandas()

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
    # Validate ouroboros mode (v11.0+)
    # -------------------------------------------------------------------------
    from .ouroboros import validate_ouroboros_mode

    ouroboros = validate_ouroboros_mode(ouroboros)

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
    # Streaming mode (v8.0+): Return generator instead of materializing
    # -------------------------------------------------------------------------
    if not materialize:
        if source == "exness":
            msg = (
                "Streaming mode (materialize=False) is not yet supported for Exness. "
                "Use materialize=True or use Binance source."
            )
            raise ValueError(msg)

        # Binance streaming: yields batches directly from network
        return _stream_range_bars_binance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=threshold_decimal_bps,
            market=market_normalized,
            batch_size=batch_size,
            include_microstructure=include_microstructure,
            include_incomplete=include_incomplete,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
            verify_checksum=verify_checksum,
        )

    # -------------------------------------------------------------------------
    # Check ClickHouse bar cache first (Issue #21: fast path for precomputed bars)
    # -------------------------------------------------------------------------
    if use_cache:
        try:
            from .clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                # Ouroboros mode filter ensures cache isolation (Plan: sparkling-coalescing-dijkstra.md)
                cached_bars = cache.get_bars_by_timestamp_range(
                    symbol=symbol,
                    threshold_decimal_bps=threshold_decimal_bps,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    include_microstructure=include_microstructure,
                    ouroboros_mode=ouroboros,
                )
                if cached_bars is not None and len(cached_bars) > 0:
                    # Tier 0 validation: Content-based staleness detection (Issue #39)
                    # This catches stale cached data from pre-v7.0 (e.g., VWAP=0)
                    if include_microstructure:
                        staleness = detect_staleness(
                            cached_bars, require_microstructure=True
                        )
                        if staleness.is_stale:
                            logger.warning(
                                "Stale cache data detected for %s: %s. "
                                "Falling through to recompute.",
                                symbol,
                                staleness.reason,
                            )
                            # Fall through to tick processing path
                        else:
                            # Fast path: return validated bars from ClickHouse (~50ms)
                            return cached_bars
                    else:
                        # Fast path: return precomputed bars from ClickHouse (~50ms)
                        return cached_bars
        except ImportError:
            # ClickHouse not available, fall through to tick processing
            pass
        except ConnectionError:
            # ClickHouse connection failed, fall through to tick processing
            pass

    # -------------------------------------------------------------------------
    # Initialize storage (Tier 1: local Parquet ticks)
    # -------------------------------------------------------------------------
    storage = TickStorage(cache_dir=Path(cache_dir) if cache_dir else None)

    # Cache key includes source and market to avoid collisions
    cache_symbol = f"{source}_{market_normalized}_{symbol}".upper()

    # -------------------------------------------------------------------------
    # Fetch tick data (cache or network) - slow path
    # -------------------------------------------------------------------------
    tick_data: pl.DataFrame

    if use_cache and storage.has_ticks(cache_symbol, start_ts, end_ts):
        tick_data = storage.read_ticks(cache_symbol, start_ts, end_ts)
    elif fetch_if_missing:
        if source == "binance":
            tick_data = _fetch_binance(symbol, start_date, end_date, market_normalized)
        else:  # exness
            tick_data = _fetch_exness(symbol, start_date, end_date, validation)

        # Cache the tick data
        if use_cache and not tick_data.is_empty():
            storage.write_ticks(cache_symbol, tick_data)
    else:
        # fetch_if_missing=False: Return empty DataFrame when no cached data
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if tick_data.is_empty():
        msg = f"No data available for {symbol} from {start_date} to {end_date}"
        raise RuntimeError(msg)

    # -------------------------------------------------------------------------
    # Process to range bars with Ouroboros boundaries (v11.0+)
    # -------------------------------------------------------------------------
    if source == "exness":
        # Exness uses week ouroboros implicitly (aligns with Sunday market open)
        return _process_exness_ticks(
            tick_data,
            symbol,
            threshold_decimal_bps,
            validation,
            include_incomplete,
            include_microstructure,
        )

    # Binance: Process with ouroboros segment iteration
    from .ouroboros import iter_ouroboros_segments

    all_bars: list[pd.DataFrame] = []
    processor: RangeBarProcessor | None = None

    for segment_start, segment_end, boundary in iter_ouroboros_segments(
        start_dt.date(), end_dt.date(), ouroboros
    ):
        # Reset processor at ouroboros boundary
        if boundary is not None and processor is not None:
            orphaned_bar = processor.reset_at_ouroboros()
            if include_orphaned_bars and orphaned_bar is not None:
                # Add orphan metadata
                orphaned_bar["is_orphan"] = True
                orphaned_bar["ouroboros_boundary"] = boundary.timestamp
                orphaned_bar["ouroboros_reason"] = boundary.reason
                orphan_df = pd.DataFrame([orphaned_bar])
                # Convert timestamp to datetime index
                if "timestamp" in orphan_df.columns:
                    orphan_df["timestamp"] = pd.to_datetime(
                        orphan_df["timestamp"], unit="us", utc=True
                    )
                    orphan_df = orphan_df.set_index("timestamp")
                all_bars.append(orphan_df)

        # Filter tick data for this segment
        # Note: Tick timestamps are in MILLISECONDS (not microseconds)
        segment_start_ms = int(segment_start.timestamp() * 1_000)
        segment_end_ms = int(segment_end.timestamp() * 1_000)

        segment_ticks = tick_data.filter(
            (pl.col("timestamp") >= segment_start_ms)
            & (pl.col("timestamp") <= segment_end_ms)
        )

        if segment_ticks.is_empty():
            continue

        # Process segment (reuse processor for state continuity within segment)
        segment_bars, processor = _process_binance_trades(
            segment_ticks,
            threshold_decimal_bps,
            include_incomplete,
            include_microstructure,
            processor=processor,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
        )

        if segment_bars is not None and not segment_bars.empty:
            all_bars.append(segment_bars)

    # Concatenate all segments
    if not all_bars:
        bars_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    elif len(all_bars) == 1:
        bars_df = all_bars[0]
    else:
        bars_df = pd.concat(all_bars, axis=0)
        bars_df = bars_df.sort_index()

    # -------------------------------------------------------------------------
    # Add exchange session flags (Issue #8)
    # -------------------------------------------------------------------------
    # Session flags indicate which traditional market sessions were active
    # at bar close time. Useful for analyzing crypto/forex behavior.
    if include_exchange_sessions and not bars_df.empty:
        import warnings

        from .ouroboros import get_active_exchange_sessions

        # Compute session flags for each bar based on close timestamp (index)
        session_data = {
            "exchange_session_sydney": [],
            "exchange_session_tokyo": [],
            "exchange_session_london": [],
            "exchange_session_newyork": [],
        }
        for ts in bars_df.index:
            # Ensure timezone-aware UTC timestamp
            if ts.tzinfo is None:
                ts_utc = ts.tz_localize("UTC")
            else:
                ts_utc = ts.tz_convert("UTC")
            # Suppress nanosecond warning - session detection is hour-granularity
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Discarding nonzero nanoseconds")
                flags = get_active_exchange_sessions(ts_utc.to_pydatetime())
            session_data["exchange_session_sydney"].append(flags.sydney)
            session_data["exchange_session_tokyo"].append(flags.tokyo)
            session_data["exchange_session_london"].append(flags.london)
            session_data["exchange_session_newyork"].append(flags.newyork)

        # Add columns to DataFrame
        for col, values in session_data.items():
            bars_df[col] = values

    # -------------------------------------------------------------------------
    # Write computed bars to ClickHouse cache (Issue #37)
    # -------------------------------------------------------------------------
    # Cache write is non-blocking: failures don't affect the return value.
    # The computation succeeded, so we return bars even if caching fails.
    if use_cache and bars_df is not None and not bars_df.empty:
        try:
            from .clickhouse import RangeBarCache
            from .exceptions import CacheError, CacheWriteError

            with RangeBarCache() as cache:
                # Use store_bars_bulk for bars computed without exact CacheKey
                # Ouroboros mode determines cache key (Plan: sparkling-coalescing-dijkstra.md)
                written = cache.store_bars_bulk(
                    symbol=symbol,
                    threshold_decimal_bps=threshold_decimal_bps,
                    bars=bars_df,
                    version="",  # Version tracked elsewhere
                    ouroboros_mode=ouroboros,
                )
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    "Cached %d bars for %s @ %d dbps",
                    written,
                    symbol,
                    threshold_decimal_bps,
                )
        except ImportError:
            # ClickHouse not available - skip caching
            pass
        except ConnectionError:
            # ClickHouse connection failed - skip caching
            pass
        except (CacheError, OSError, RuntimeError) as e:
            # Log but don't fail - cache is optimization layer
            # CacheError: All cache-specific errors
            # OSError: Network/disk errors
            # RuntimeError: ClickHouse driver errors
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Cache write failed (non-fatal): %s", e)

    return bars_df


def get_range_bars_pandas(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    **kwargs: Any,
) -> pd.DataFrame:
    """Get range bars as pandas DataFrame (deprecated compatibility shim).

    .. deprecated:: 8.0
        Use ``get_range_bars(materialize=True)`` directly instead.
        This function will be removed in v9.0.

    This function exists for backward compatibility with code written before
    the streaming API was introduced. It simply calls ``get_range_bars()``
    with ``materialize=True`` and returns the result.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points
    **kwargs
        Additional arguments passed to ``get_range_bars()``

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    Examples
    --------
    Instead of:

    >>> df = get_range_bars_pandas("BTCUSDT", "2024-01-01", "2024-06-30")

    Use:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30", materialize=True)
    """
    import warnings

    warnings.warn(
        "get_range_bars_pandas() is deprecated. "
        "Use get_range_bars(materialize=True) instead. "
        "This function will be removed in v9.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_range_bars(
        symbol,
        start_date,
        end_date,
        threshold_decimal_bps,
        materialize=True,
        **kwargs,
    )


def precompute_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int = 250,
    *,
    source: str = "binance",
    market: str = "spot",
    chunk_size: int = 100_000,
    invalidate_existing: str = "smart",
    progress_callback: Callable[[PrecomputeProgress], None] | None = None,
    include_microstructure: bool = False,  # noqa: ARG001 - reserved for future use
    validate_on_complete: str = "error",
    continuity_tolerance_pct: float = 0.001,
    cache_dir: str | None = None,
) -> PrecomputeResult:
    """Precompute continuous range bars for a date range (single-pass, guaranteed continuity).

    Designed for ML workflows requiring continuous bar sequences for training/validation.
    Uses Checkpoint API for memory-efficient chunked processing with state preservation.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date (inclusive) "YYYY-MM-DD"
    end_date : str
        End date (inclusive) "YYYY-MM-DD"
    threshold_decimal_bps : int, default=250
        Range bar threshold (250 = 0.25%)
    source : str, default="binance"
        Data source ("binance" or "exness")
    market : str, default="spot"
        Market type for Binance ("spot", "futures-um", "futures-cm")
    chunk_size : int, default=100_000
        Ticks per processing chunk (~15MB memory per 100K ticks)
    invalidate_existing : str, default="smart"
        Cache invalidation strategy:
        - "overlap": Invalidate only bars in date range
        - "full": Invalidate ALL bars for symbol/threshold
        - "none": Skip if any cached bars exist in range
        - "smart": Invalidate overlapping + validate junction continuity
    progress_callback : Callable, optional
        Optional callback for progress updates
    include_microstructure : bool, default=False
        Include order flow metrics (buy_volume, sell_volume, vwap)
    validate_on_complete : str, default="error"
        Continuity validation mode after precomputation:
        - "error": Raise ContinuityError if discontinuities found
        - "warn": Log warning but continue (sets continuity_valid=False)
        - "skip": Skip validation entirely (continuity_valid=None)
    continuity_tolerance_pct : float, default=0.001
        Maximum allowed price gap percentage for continuity validation.
        Default 0.1% (0.001) accommodates market microstructure events.
        The total allowed gap is threshold_pct + continuity_tolerance_pct.
    cache_dir : str or None, default=None
        Custom cache directory for tick data

    Returns
    -------
    PrecomputeResult
        Result with statistics and cache key

    Raises
    ------
    ValueError
        Invalid parameters
    RuntimeError
        Fetch or processing failure
    ContinuityError
        If validate_on_complete=True and discontinuities found

    Examples
    --------
    Basic precomputation:

    >>> result = precompute_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")
    >>> print(f"Generated {result.total_bars} bars")

    With progress callback:

    >>> def on_progress(p):
    ...     print(f"[{p.current_month}] {p.bars_generated} bars")
    >>> result = precompute_range_bars(
    ...     "BTCUSDT", "2024-01-01", "2024-03-31",
    ...     progress_callback=on_progress
    ... )
    """
    import gc
    import time
    from datetime import datetime
    from pathlib import Path

    import polars as pl

    from .clickhouse import CacheKey, RangeBarCache
    from .storage.parquet import TickStorage

    start_time = time.time()

    # Validate parameters
    if invalidate_existing not in ("overlap", "full", "none", "smart"):
        msg = f"Invalid invalidate_existing: {invalidate_existing!r}. Must be 'overlap', 'full', 'none', or 'smart'"
        raise ValueError(msg)

    if validate_on_complete not in ("error", "warn", "skip"):
        msg = f"Invalid validate_on_complete: {validate_on_complete!r}. Must be 'error', 'warn', or 'skip'"
        raise ValueError(msg)

    if not THRESHOLD_DECIMAL_MIN <= threshold_decimal_bps <= THRESHOLD_DECIMAL_MAX:
        msg = f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} and {THRESHOLD_DECIMAL_MAX}"
        raise ValueError(msg)

    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
        raise ValueError(msg) from e

    if start_dt > end_dt:
        msg = "start_date must be <= end_date"
        raise ValueError(msg)

    # Normalize market type
    market_map = {
        "spot": "spot",
        "futures-um": "um",
        "futures-cm": "cm",
        "um": "um",
        "cm": "cm",
    }
    market_normalized = market_map.get(market.lower(), market.lower())

    # Initialize storage and cache
    storage = TickStorage(cache_dir=Path(cache_dir) if cache_dir else None)
    cache = RangeBarCache()

    # Generate list of months to process
    months: list[tuple[int, int]] = []
    current = start_dt.replace(day=1)
    _december = 12
    while current <= end_dt:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == _december:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # Handle cache invalidation
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int((end_dt.timestamp() + 86399) * 1000)  # End of day
    cache_key = CacheKey(
        symbol=symbol,
        threshold_decimal_bps=threshold_decimal_bps,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if invalidate_existing == "full":
        cache.invalidate_range_bars(cache_key)
    elif invalidate_existing in ("overlap", "smart"):
        # Check for overlapping bars - will be handled after processing
        # For now, just invalidate the date range to ensure clean slate
        cache.invalidate_range_bars(cache_key)
    elif invalidate_existing == "none":
        # Check if any bars exist in range by counting
        bar_count = cache.count_bars(symbol, threshold_decimal_bps)
        if bar_count > 0:
            # Return early - some cached data exists
            # Note: This is approximate; full implementation would check time range
            return PrecomputeResult(
                symbol=symbol,
                threshold_decimal_bps=threshold_decimal_bps,
                start_date=start_date,
                end_date=end_date,
                total_bars=bar_count,
                total_ticks=0,
                elapsed_seconds=time.time() - start_time,
                continuity_valid=True,  # Assume valid for cached data
                cache_key=f"{symbol}_{threshold_decimal_bps}",
            )

    # Initialize processor (single instance for continuity)
    processor = RangeBarProcessor(threshold_decimal_bps, symbol=symbol)

    all_bars: list[pd.DataFrame] = []
    month_bars: list[pd.DataFrame] = (
        []
    )  # Issue #27: Track bars per month for incremental caching
    total_ticks = 0
    cache_symbol = f"{source}_{market_normalized}_{symbol}".upper()

    for i, (year, month) in enumerate(months):
        month_str = f"{year}-{month:02d}"

        # Report progress - fetching
        if progress_callback:
            progress_callback(
                PrecomputeProgress(
                    phase="fetching",
                    current_month=month_str,
                    months_completed=i,
                    months_total=len(months),
                    bars_generated=sum(len(b) for b in all_bars),
                    ticks_processed=total_ticks,
                    elapsed_seconds=time.time() - start_time,
                )
            )

        # Calculate month boundaries
        month_start = datetime(year, month, 1)  # noqa: DTZ001
        _december = 12
        if month == _december:
            month_end = datetime(year + 1, 1, 1)  # noqa: DTZ001
        else:
            month_end = datetime(year, month + 1, 1)  # noqa: DTZ001

        # Adjust to fit within requested date range
        actual_start = max(month_start, start_dt)
        actual_end = min(month_end, end_dt + pd.Timedelta(days=1))

        # Fetch tick data for this period
        start_ts_month = int(actual_start.timestamp() * 1000)
        end_ts_month = int(actual_end.timestamp() * 1000)

        # Check if data is already cached
        data_cached = storage.has_ticks(cache_symbol, start_ts_month, end_ts_month)

        if data_cached:
            # STREAMING READ: Use row-group based streaming to avoid OOM (Issue #12)
            # The Rust processor maintains state between process_trades() calls,
            # so we stream chunks directly without loading entire month into memory
            month_has_data = False
            for raw_tick_chunk in storage.read_ticks_streaming(
                cache_symbol, start_ts_month, end_ts_month, chunk_size=chunk_size
            ):
                month_has_data = True

                # Deduplicate trades by trade_id within chunk
                tick_chunk = raw_tick_chunk
                if "agg_trade_id" in tick_chunk.columns:
                    tick_chunk = tick_chunk.unique(
                        subset=["agg_trade_id"], maintain_order=True
                    )
                elif "trade_id" in tick_chunk.columns:
                    tick_chunk = tick_chunk.unique(
                        subset=["trade_id"], maintain_order=True
                    )

                # Sort by (timestamp, trade_id) - Rust crate requires this order
                if "timestamp" in tick_chunk.columns:
                    if "agg_trade_id" in tick_chunk.columns:
                        tick_chunk = tick_chunk.sort(["timestamp", "agg_trade_id"])
                    elif "trade_id" in tick_chunk.columns:
                        tick_chunk = tick_chunk.sort(["timestamp", "trade_id"])
                    else:
                        tick_chunk = tick_chunk.sort("timestamp")

                total_ticks += len(tick_chunk)

                # Report progress - processing
                if progress_callback:
                    progress_callback(
                        PrecomputeProgress(
                            phase="processing",
                            current_month=month_str,
                            months_completed=i,
                            months_total=len(months),
                            bars_generated=sum(len(b) for b in all_bars),
                            ticks_processed=total_ticks,
                            elapsed_seconds=time.time() - start_time,
                        )
                    )

                # Stream directly to Rust processor (Issue #16: use streaming mode)
                chunk = tick_chunk.to_dicts()
                bars = processor.process_trades_streaming(chunk)
                if bars:
                    # Issue #30: Always include microstructure for ClickHouse cache
                    bars_df = processor.to_dataframe(bars, include_microstructure=True)
                    month_bars.append(bars_df)  # Issue #27: Track per-month bars

                del chunk, tick_chunk

            gc.collect()

            if not month_has_data:
                continue
        else:
            # DATA NOT CACHED: Fetch from source day-by-day to prevent OOM
            # Issue #14: Fetching entire month at once causes OOM for high-volume
            # months like March 2024. Fetch day-by-day instead.
            current_day = actual_start
            while current_day < actual_end:
                next_day = current_day + pd.Timedelta(days=1)
                next_day = min(next_day, actual_end)

                day_start_str = current_day.strftime("%Y-%m-%d")
                day_end_str = (next_day - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")

                if source == "binance":
                    tick_data = _fetch_binance(
                        symbol, day_start_str, day_end_str, market_normalized
                    )
                else:
                    tick_data = _fetch_exness(
                        symbol, day_start_str, day_end_str, "strict"
                    )

                if not tick_data.is_empty():
                    storage.write_ticks(cache_symbol, tick_data)

                    # Deduplicate trades by trade_id
                    if "agg_trade_id" in tick_data.columns:
                        tick_data = tick_data.unique(
                            subset=["agg_trade_id"], maintain_order=True
                        )
                    elif "trade_id" in tick_data.columns:
                        tick_data = tick_data.unique(
                            subset=["trade_id"], maintain_order=True
                        )

                    # Sort by (timestamp, trade_id) - Rust crate requires order
                    if "timestamp" in tick_data.columns:
                        if "agg_trade_id" in tick_data.columns:
                            tick_data = tick_data.sort(["timestamp", "agg_trade_id"])
                        elif "trade_id" in tick_data.columns:
                            tick_data = tick_data.sort(["timestamp", "trade_id"])
                        else:
                            tick_data = tick_data.sort("timestamp")

                    total_ticks += len(tick_data)

                    # Report progress - processing
                    if progress_callback:
                        progress_callback(
                            PrecomputeProgress(
                                phase="processing",
                                current_month=month_str,
                                months_completed=i,
                                months_total=len(months),
                                bars_generated=sum(len(b) for b in all_bars),
                                ticks_processed=total_ticks,
                                elapsed_seconds=time.time() - start_time,
                            )
                        )

                    # Process with chunking for memory efficiency
                    tick_count = len(tick_data)
                    for chunk_start in range(0, tick_count, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, tick_count)
                        chunk_df = tick_data.slice(chunk_start, chunk_end - chunk_start)
                        chunk = chunk_df.to_dicts()

                        # Stream to Rust processor (Issue #16: use streaming mode)
                        bars = processor.process_trades_streaming(chunk)
                        if bars:
                            # Issue #30: Always include microstructure for ClickHouse cache
                            bars_df = processor.to_dataframe(
                                bars, include_microstructure=True
                            )
                            month_bars.append(
                                bars_df
                            )  # Issue #27: Track per-month bars

                        del chunk, chunk_df

                    del tick_data
                    gc.collect()

                current_day = next_day

        # Issue #27: Incremental caching - store bars to ClickHouse after each month
        # This provides crash resilience and bounded memory for DB writes
        if month_bars:
            # MEM-006: Use Polars for memory-efficient concatenation
            month_df = _concat_pandas_via_polars(month_bars)

            # Report progress - caching this month
            if progress_callback:
                progress_callback(
                    PrecomputeProgress(
                        phase="caching",
                        current_month=month_str,
                        months_completed=i + 1,
                        months_total=len(months),
                        bars_generated=len(month_df) + sum(len(b) for b in all_bars),
                        ticks_processed=total_ticks,
                        elapsed_seconds=time.time() - start_time,
                    )
                )

            # Cache immediately (idempotent via ReplacingMergeTree)
            rows_sent = len(month_df)
            rows_inserted = cache.store_bars_bulk(
                symbol, threshold_decimal_bps, month_df
            )

            # Post-cache validation: FAIL LOUDLY if ClickHouse didn't receive all bars
            if rows_inserted != rows_sent:
                msg = (
                    f"ClickHouse cache validation FAILED for {month_str}: "
                    f"sent {rows_sent} bars but only {rows_inserted} inserted. "
                    f"Data integrity compromised - aborting."
                )
                raise RuntimeError(msg)

            # Preserve for final validation and return
            all_bars.append(month_df)
            month_bars = []  # Clear to reclaim memory
            gc.collect()

    # Combine all bars (MEM-006: use Polars for memory efficiency)
    if all_bars:
        final_bars = _concat_pandas_via_polars(all_bars)
    else:
        final_bars = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Note: Caching now happens incrementally after each month (Issue #27)
    # No final bulk store needed - all bars already cached per-month

    # Validate continuity (Issue #19: configurable tolerance and validation mode)
    continuity_valid: bool | None = None

    if validate_on_complete == "skip":
        # Skip validation entirely
        continuity_valid = None
    else:
        continuity_result = validate_continuity(
            final_bars,
            tolerance_pct=continuity_tolerance_pct,
            threshold_decimal_bps=threshold_decimal_bps,
        )
        continuity_valid = continuity_result["is_valid"]

        if not continuity_valid:
            if validate_on_complete == "error":
                msg = f"Found {continuity_result['discontinuity_count']} discontinuities in precomputed bars"
                raise ContinuityError(msg, continuity_result["discontinuities"])
            if validate_on_complete == "warn":
                import warnings

                msg = (
                    f"Found {continuity_result['discontinuity_count']} discontinuities "
                    f"in precomputed bars (tolerance: {continuity_tolerance_pct:.4%})"
                )
                warnings.warn(msg, stacklevel=2)

    return PrecomputeResult(
        symbol=symbol,
        threshold_decimal_bps=threshold_decimal_bps,
        start_date=start_date,
        end_date=end_date,
        total_bars=len(final_bars),
        total_ticks=total_ticks,
        elapsed_seconds=time.time() - start_time,
        continuity_valid=continuity_valid,
        cache_key=f"{symbol}_{threshold_decimal_bps}",
    )


def _stream_range_bars_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int,
    market: str,
    batch_size: int = 10_000,
    include_microstructure: bool = False,
    include_incomplete: bool = False,
    prevent_same_timestamp_close: bool = True,
    verify_checksum: bool = True,
) -> Iterator[pl.DataFrame]:
    """Stream range bars in batches using memory-efficient chunked processing.

    This is the internal generator for Phase 4 streaming API. It:
    1. Streams trades in 6-hour chunks from Binance via stream_binance_trades()
    2. Processes each chunk to bars via process_trades_streaming_arrow()
    3. Yields batches of bars as Polars DataFrames

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date "YYYY-MM-DD"
    end_date : str
        End date "YYYY-MM-DD"
    threshold_decimal_bps : int
        Range bar threshold (250 = 0.25%)
    market : str
        Normalized market type: "spot", "um", or "cm"
    batch_size : int, default=10_000
        Number of bars per yielded DataFrame (~500 KB each)
    include_microstructure : bool, default=False
        Include microstructure columns in output
    include_incomplete : bool, default=False
        Include the final incomplete bar

    Yields
    ------
    pl.DataFrame
        Batches of range bars (OHLCV format, backtesting.py compatible)

    Memory Usage
    ------------
    Peak: ~50 MB (6-hour trade chunk + bar buffer)
    Per yield: ~500 KB (10,000 bars)
    """
    import polars as pl

    try:
        from ._core import MarketType, stream_binance_trades
    except ImportError as e:
        msg = (
            "Streaming requires the 'data-providers' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e

    # Map market string to enum
    market_enum = {
        "spot": MarketType.Spot,
        "um": MarketType.FuturesUM,
        "cm": MarketType.FuturesCM,
    }[market]

    # Create processor with symbol for checkpoint support
    processor = RangeBarProcessor(
        threshold_decimal_bps,
        symbol=symbol,
        prevent_same_timestamp_close=prevent_same_timestamp_close,
    )
    bar_buffer: list[dict] = []

    # Stream trades in 6-hour chunks
    for trade_batch in stream_binance_trades(
        symbol,
        start_date,
        end_date,
        chunk_hours=6,
        market_type=market_enum,
        verify_checksum=verify_checksum,
    ):
        # Process to bars via Arrow (zero-copy to Polars)
        arrow_batch = processor.process_trades_streaming_arrow(trade_batch)
        bars_df = pl.from_arrow(arrow_batch)

        if not bars_df.is_empty():
            # Add to buffer
            bar_buffer.extend(bars_df.to_dicts())

        # Yield when buffer reaches batch_size
        while len(bar_buffer) >= batch_size:
            batch = bar_buffer[:batch_size]
            bar_buffer = bar_buffer[batch_size:]
            yield _bars_list_to_polars(batch, include_microstructure)

    # Handle incomplete bar at end
    if include_incomplete:
        incomplete = processor.get_incomplete_bar()
        if incomplete:
            bar_buffer.append(incomplete)

    # Yield remaining bars
    if bar_buffer:
        yield _bars_list_to_polars(bar_buffer, include_microstructure)


def _fetch_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    market: str,
) -> pl.DataFrame:
    """Fetch Binance aggTrades data (internal).

    DEPRECATED: Use stream_binance_trades() for memory-efficient streaming.
    This function loads all trades into memory at once.
    """
    import warnings

    import polars as pl

    warnings.warn(
        "_fetch_binance() is deprecated. Use stream_binance_trades() for "
        "memory-efficient streaming. This function will be removed in v9.0.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    prevent_same_timestamp_close: bool = True,
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

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    # This enables predicate pushdown and avoids materializing unused columns
    # Memory impact: 10-100x reduction depending on filter selectivity

    # Determine volume column name (works for both DataFrame and LazyFrame)
    if isinstance(trades, pl.LazyFrame):
        available_cols = trades.collect_schema().names()
    else:
        available_cols = trades.columns

    volume_col = "quantity" if "quantity" in available_cols else "volume"

    # Build column list - include is_buyer_maker for microstructure features (Issue #30)
    columns = [
        pl.col("timestamp"),
        pl.col("price"),
        pl.col(volume_col).alias("quantity"),
    ]
    if "is_buyer_maker" in available_cols:
        columns.append(pl.col("is_buyer_maker"))

    # Apply selection (predicates pushed down for LazyFrame)
    trades_selected = trades.select(columns)

    # Collect AFTER selection (for LazyFrame)
    if isinstance(trades_selected, pl.LazyFrame):
        trades_minimal = trades_selected.collect()
    else:
        trades_minimal = trades_selected

    # Use provided processor or create new one
    if processor is None:
        processor = RangeBarProcessor(
            threshold_decimal_bps,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
        )

    # MEM-002: Process in chunks to bound memory (2.5 GB → ~50 MB per chunk)
    # Chunked .to_dicts() avoids materializing 1M+ trade dicts at once
    chunk_size = 100_000
    all_bars: list[dict] = []

    n_rows = len(trades_minimal)
    for start in range(0, n_rows, chunk_size):
        chunk = trades_minimal.slice(start, chunk_size).to_dicts()
        bars = processor.process_trades_streaming(chunk)
        all_bars.extend(bars)

    bars = all_bars

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
    prevent_same_timestamp_close: bool = True,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    validate_on_return: bool = False,
    continuity_action: str = "warn",
    chunk_size: int = 100_000,
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
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention (Issue #36).
        If True (default): A bar cannot close on the same timestamp it opened.
        If False: Legacy v8 behavior for comparative analysis.
    use_cache : bool, default=True
        Use ClickHouse cache for bar retrieval/storage
    fetch_if_missing : bool, default=True
        Fetch and process new data if cache doesn't have enough bars
    max_lookback_days : int, default=90
        Safety limit: maximum days to look back when fetching missing data.
        Prevents runaway fetches on empty caches.
    warn_if_fewer : bool, default=True
        Emit UserWarning if returning fewer bars than requested.
    validate_on_return : bool, default=False
        If True, validate bar continuity before returning.
        Uses continuity_action to determine behavior on failure.
    continuity_action : str, default="warn"
        Action when discontinuity found during validation:
        - "warn": Log warning but return data
        - "raise": Raise ContinuityError
        - "log": Silent logging only
    chunk_size : int, default=100_000
        Number of ticks per processing chunk for memory efficiency.
        Larger values = faster processing, more memory.
        Default 100K = ~15MB memory overhead.
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

    import numpy as np
    import polars as pl

    # -------------------------------------------------------------------------
    # Validation helper (closure over validate_on_return, continuity_action)
    # -------------------------------------------------------------------------
    def _apply_validation(df: pd.DataFrame) -> pd.DataFrame:
        """Apply continuity validation if enabled, then return DataFrame."""
        if not validate_on_return or df.empty or len(df) <= 1:
            return df

        # Check continuity: Close[i] should equal Open[i+1]
        close_prices = df["Close"].to_numpy()[:-1]
        open_prices = df["Open"].to_numpy()[1:]

        # Calculate relative differences
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.abs(open_prices - close_prices) / np.abs(close_prices)

        # 0.01% tolerance for floating-point errors
        tolerance = 0.0001
        discontinuities_mask = rel_diff > tolerance

        if not np.any(discontinuities_mask):
            return df

        # Found discontinuities
        discontinuity_count = int(np.sum(discontinuities_mask))
        msg = f"Found {discontinuity_count} discontinuities in {len(df)} bars"

        if continuity_action == "raise":
            # Build details for ContinuityError
            indices = np.where(discontinuities_mask)[0]
            details = []
            for idx in indices[:10]:  # Limit to first 10
                details.append(
                    {
                        "bar_index": int(idx),
                        "prev_close": float(close_prices[idx]),
                        "next_open": float(open_prices[idx]),
                        "gap_pct": float(rel_diff[idx] * 100),
                    }
                )
            raise ContinuityError(msg, details)
        if continuity_action == "warn":
            warnings.warn(msg, ContinuityWarning, stacklevel=3)
        else:  # "log"
            import logging

            logging.getLogger("rangebar").warning(msg)

        return df

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
                    # Tier 0 validation: Content-based staleness detection (Issue #39)
                    if include_microstructure:
                        staleness = detect_staleness(
                            bars_df, require_microstructure=True
                        )
                        if staleness.is_stale:
                            logger.warning(
                                "Stale cache data detected for %s: %s. "
                                "Falling through to recompute.",
                                symbol,
                                staleness.reason,
                            )
                            # Fall through to fetch_if_missing path
                        else:
                            # Cache hit - return exactly n_bars
                            return _apply_validation(bars_df.tail(n_bars))
                    else:
                        # Cache hit - return exactly n_bars
                        return _apply_validation(bars_df.tail(n_bars))

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
                        chunk_size=chunk_size,
                        prevent_same_timestamp_close=prevent_same_timestamp_close,
                    )

                    if bars_df is not None and len(bars_df) >= n_bars:
                        return _apply_validation(bars_df.tail(n_bars))

                # Return what we have (or None)
                if bars_df is not None and len(bars_df) > 0:
                    if warn_if_fewer and len(bars_df) < n_bars:
                        warnings.warn(
                            f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                            f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return _apply_validation(bars_df)

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
        return _apply_validation(bars_df.tail(n_bars))

    if bars_df is not None and len(bars_df) > 0:
        if warn_if_fewer:
            warnings.warn(
                f"Returning {len(bars_df)} bars instead of requested {n_bars}. "
                f"Insufficient data available within max_lookback_days={max_lookback_days}.",
                UserWarning,
                stacklevel=2,
            )
        return _apply_validation(bars_df)

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
    chunk_size: int = 100_000,  # noqa: ARG001 - reserved for future chunked processing
    prevent_same_timestamp_close: bool = True,
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

    Parameters
    ----------
    chunk_size : int, default=100_000
        Number of ticks per processing chunk for memory efficiency when using
        chunked processing with checkpoint continuation.
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
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
    else:
        end_dt = datetime.now(tz=UTC)

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
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
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

        # Phase 2: Merge ALL ticks chronologically and deduplicate
        # Sort by (timestamp, trade_id) - Rust crate requires this order
        merged_ticks = pl.concat(all_tick_data)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "agg_trade_id"])
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "trade_id"])
        else:
            merged_ticks = merged_ticks.sort("timestamp")

        # Deduplicate by trade_id (Binance data may have duplicates at boundaries)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(
                subset=["agg_trade_id"], maintain_order=True
            )
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(subset=["trade_id"], maintain_order=True)

        # Phase 3: Process with SINGLE processor (guarantees continuity)
        new_bars, _ = _process_binance_trades(
            merged_ticks,
            threshold,
            False,
            include_microstructure,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
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

            # MEM-006: Use Polars for memory-efficient concatenation
            combined = _concat_pandas_via_polars([new_bars, current_bars])
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
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
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

    # MEM-006: Use Polars for memory-efficient concatenation
    combined = _concat_pandas_via_polars(all_bars)
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
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
    else:
        end_dt = datetime.now(tz=UTC)

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
                fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
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

        # Sort by (timestamp, trade_id) - Rust crate requires this order
        merged_ticks = pl.concat(all_tick_data)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "agg_trade_id"])
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.sort(["timestamp", "trade_id"])
        else:
            merged_ticks = merged_ticks.sort("timestamp")

        # Deduplicate by trade_id (Binance data may have duplicates at boundaries)
        if "agg_trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(
                subset=["agg_trade_id"], maintain_order=True
            )
        elif "trade_id" in merged_ticks.columns:
            merged_ticks = merged_ticks.unique(subset=["trade_id"], maintain_order=True)

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
            fetch_end_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=UTC)
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

    # MEM-006: Use Polars for memory-efficient concatenation
    combined = _concat_pandas_via_polars(all_bars)
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
