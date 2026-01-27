# polars-exception: backtesting.py requires Pandas DataFrames for OHLCV data
"""Exness forex data provider integration.

This module provides Python bindings for processing Exness Raw_Spread tick data
into range bars. It is only available when the 'exness' feature is enabled.

Usage:
    from rangebar.exness import (
        ExnessInstrument,
        ExnessRangeBarBuilder,
        ValidationStrictness,
        process_exness_ticks_to_dataframe,
    )

    # Create builder for EURUSD with 25bps threshold
    builder = ExnessRangeBarBuilder(
        ExnessInstrument.EURUSD,
        threshold_decimal_bps=250,
        strictness=ValidationStrictness.Strict,
    )

    # Process tick data
    for tick in ticks:
        bar = builder.process_tick(tick)
        if bar:
            print(f"Bar closed: {bar}")

Note:
    - Volume is always 0 (Exness Raw_Spread has no volume data)
    - SpreadStats capture market stress signals via spread dynamics
    - JPY pairs (USDJPY, EURJPY, GBPJPY) use different pip values
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


def _check_exness_available() -> None:
    """Check if Exness feature is available."""
    try:
        from rangebar._core import ExnessInstrument as _  # noqa: F401
    except ImportError as e:
        msg = (
            "Exness support not available. "
            "Install with: pip install rangebar[exness] "
            "or build with: maturin develop --features exness"
        )
        raise ImportError(msg) from e


# Import Exness types from Rust bindings (feature-gated)
try:
    from rangebar._core import (
        ExnessInstrument,
        ExnessRangeBarBuilder,
        ValidationStrictness,
    )

    _EXNESS_AVAILABLE = True
except ImportError:
    _EXNESS_AVAILABLE = False
    # Define placeholder types for type checking
    ExnessInstrument = None  # type: ignore[misc,assignment]
    ExnessRangeBarBuilder = None  # type: ignore[misc,assignment]
    ValidationStrictness = None  # type: ignore[misc,assignment]


def process_exness_ticks_to_dataframe(
    ticks: pd.DataFrame | Sequence[dict[str, float | int]],
    instrument: ExnessInstrument,  # type: ignore[valid-type]
    threshold_decimal_bps: int = 250,
    strictness: ValidationStrictness = None,  # type: ignore[assignment]
) -> pd.DataFrame:
    """Process Exness tick data to range bars DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame or Sequence[Dict]
        Tick data with columns/keys: bid, ask, timestamp_ms
    instrument : ExnessInstrument
        Exness instrument enum value (e.g., ExnessInstrument.EURUSD)
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
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

    Examples
    --------
    >>> from rangebar.exness import (
    ...     ExnessInstrument,
    ...     process_exness_ticks_to_dataframe,
    ... )
    >>> import pandas as pd
    >>> ticks = pd.DataFrame({
    ...     "bid": [1.0800, 1.0810, 1.0830],
    ...     "ask": [1.0805, 1.0815, 1.0835],
    ...     "timestamp_ms": [1600000000000, 1600001000000, 1600002000000],
    ... })
    >>> bars = process_exness_ticks_to_dataframe(
    ...     ticks, ExnessInstrument.EURUSD, threshold_decimal_bps=250
    ... )
    """
    _check_exness_available()

    # Set default strictness
    if strictness is None:
        strictness = ValidationStrictness.Strict  # type: ignore[attr-defined]

    # Create builder
    builder = ExnessRangeBarBuilder(instrument, threshold_decimal_bps, strictness)  # type: ignore[misc]

    # Convert DataFrame to list of dicts if needed
    if isinstance(ticks, pd.DataFrame):
        required_cols = {"bid", "ask", "timestamp_ms"}
        missing = required_cols - set(ticks.columns)
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)
        tick_dicts = ticks[["bid", "ask", "timestamp_ms"]].to_dict("records")
    else:
        tick_dicts = list(ticks)

    # Process all ticks
    bars = builder.process_ticks(tick_dicts)

    # Include incomplete bar if exists
    incomplete = builder.get_incomplete_bar()
    if incomplete:
        bars.append(incomplete)

    if not bars:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "spread_stats"]
        ).set_index(pd.DatetimeIndex([], name="timestamp"))

    # Convert to DataFrame
    df = pd.DataFrame(bars)

    # Parse timestamps and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Rename columns for backtesting.py compatibility
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Select columns in standard order
    result = df[["Open", "High", "Low", "Close", "Volume", "spread_stats"]]

    return result


def is_exness_available() -> bool:
    """Check if Exness feature is available.

    Returns
    -------
    bool
        True if Exness bindings are available, False otherwise
    """
    return _EXNESS_AVAILABLE


def get_range_bars_exness(
    instrument: str,
    start_date: str,
    end_date: str,
    _threshold_decimal_bps: int = 250,  # Reserved for implementation
    *,
    _include_orphaned_bars: bool = False,  # Reserved for implementation
    _include_incomplete: bool = False,  # Reserved for implementation
    strictness: ValidationStrictness = None,  # type: ignore[assignment]
) -> pd.DataFrame:
    """Get range bars for Forex instruments with dynamic ouroboros.

    This is the Forex-specific API that uses **dynamic ouroboros** - the first
    tick after each weekend gap becomes the ouroboros point. This handles DST
    automatically since we use actual data gaps, not calendar calculations.

    For Forex, ouroboros is implicitly "week" aligned with market structure:
    - Markets close Friday ~21:00 UTC
    - Markets reopen Sunday ~17:00 UTC (shifts with DST)
    - First tick after weekend = ouroboros point

    Parameters
    ----------
    instrument : str
        Forex instrument (e.g., "EURUSD", "GBPUSD", "XAUUSD").
        Must match ExnessInstrument enum values.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 dbps = 0.25%).
    include_orphaned_bars : bool, default=False
        Include incomplete bars from weekend boundaries.
        If True, orphaned bars are included with ``is_orphan=True`` column.
    include_incomplete : bool, default=False
        Include the final incomplete bar (if any).
    strictness : ValidationStrictness, optional
        Tick validation strictness level (default: Strict).
        - Permissive: Accept all ticks
        - Strict: Reject crossed markets (bid > ask)
        - Paranoid: Reject excessive spreads

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex, compatible with backtesting.py.
        Additional columns:
        - spread_stats: dict with min/max/avg spread
        - is_orphan: bool (if include_orphaned_bars=True)
        - ouroboros_boundary: datetime (if include_orphaned_bars=True)

    Raises
    ------
    ImportError
        If Exness feature is not enabled.
    ValueError
        If instrument is invalid or dates are malformed.
    FileNotFoundError
        If tick data is not available for the date range.

    Notes
    -----
    - Volume is always 0 (Exness Raw_Spread has no volume data)
    - Ouroboros is always "week" for Forex (dynamic, based on actual gaps)
    - Weekend boundaries are detected from gaps > 40 hours in tick data

    Examples
    --------
    >>> from rangebar.exness import get_range_bars_exness
    >>> df = get_range_bars_exness("EURUSD", "2024-01-15", "2024-01-19")
    >>> print(f"Generated {len(df)} bars")
    Generated 150 bars

    See Also
    --------
    rangebar.get_range_bars : Generic API for all data sources
    process_exness_ticks_to_dataframe : Low-level tick processing
    """
    _check_exness_available()

    from datetime import datetime as dt

    # These imports are used in the implementation below (after NotImplementedError)
    from .ouroboros import (  # noqa: F401
        detect_forex_weekend_boundaries,
        iter_forex_ouroboros_segments,
    )

    # Set default strictness
    if strictness is None:
        strictness = ValidationStrictness.Strict  # type: ignore[attr-defined]

    # Parse instrument (validation happens before NotImplementedError)
    try:
        exness_instrument = ExnessInstrument[instrument.upper()]  # type: ignore[index]
        _ = exness_instrument  # Used in implementation below
    except (KeyError, TypeError) as e:
        valid = [i.name for i in ExnessInstrument] if ExnessInstrument else []  # type: ignore[attr-defined]
        msg = f"Invalid instrument: {instrument!r}. Valid: {valid}"
        raise ValueError(msg) from e

    # Parse dates (validation happens before NotImplementedError)
    # Note: Timezone not needed - we extract .date() which is timezone-naive by design
    try:
        start_dt = dt.strptime(start_date, "%Y-%m-%d").date()  # noqa: DTZ007
        end_dt = dt.strptime(end_date, "%Y-%m-%d").date()  # noqa: DTZ007
        _ = (start_dt, end_dt)  # Used in implementation below
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD. Error: {e}"
        raise ValueError(msg) from e

    # TODO: Fetch tick data from Exness API or local cache
    # For now, this requires pre-downloaded tick data
    # Future: Integrate with ExnessFetcher from Rust
    msg = (
        "get_range_bars_exness() requires tick data to be pre-loaded. "
        "Use process_exness_ticks_to_dataframe() with your tick data, "
        "or wait for ExnessFetcher integration (coming soon)."
    )
    raise NotImplementedError(msg)

    # The implementation below shows the intended flow once tick fetching is available:
    #
    # # Fetch ticks for date range
    # ticks = fetch_exness_ticks(instrument, start_dt, end_dt)
    #
    # # Extract timestamps for weekend detection
    # timestamps_ms = [t["timestamp_ms"] for t in ticks]
    #
    # # Detect weekend boundaries (dynamic ouroboros)
    # boundaries = detect_forex_weekend_boundaries(timestamps_ms)
    #
    # # Create builder
    # builder = ExnessRangeBarBuilder(
    #     exness_instrument, threshold_decimal_bps, strictness
    # )
    #
    # all_bars = []
    # for start_idx, end_idx, boundary in iter_forex_ouroboros_segments(
    #     timestamps_ms, start_dt, end_dt
    # ):
    #     # Reset at weekend boundary
    #     if boundary is not None:
    #         orphaned = builder.reset()  # Need to add reset method
    #         if include_orphaned_bars and orphaned:
    #             orphaned["is_orphan"] = True
    #             orphaned["ouroboros_boundary"] = boundary.timestamp
    #             all_bars.append(orphaned)
    #
    #     # Process segment ticks
    #     segment_ticks = ticks[start_idx:end_idx + 1]
    #     bars = builder.process_ticks(segment_ticks)
    #     all_bars.extend(bars)
    #
    # # Handle incomplete bar
    # if include_incomplete:
    #     incomplete = builder.get_incomplete_bar()
    #     if incomplete:
    #         all_bars.append(incomplete)
    #
    # return pd.DataFrame(all_bars)


__all__ = [
    "ExnessInstrument",
    "ExnessRangeBarBuilder",
    "ValidationStrictness",
    "get_range_bars_exness",
    "is_exness_available",
    "process_exness_ticks_to_dataframe",
]
