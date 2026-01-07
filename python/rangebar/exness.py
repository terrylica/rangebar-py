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
        threshold_bps=250,
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
    threshold_bps: int = 250,
    strictness: ValidationStrictness = None,  # type: ignore[assignment]
) -> pd.DataFrame:
    """Process Exness tick data to range bars DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame or Sequence[Dict]
        Tick data with columns/keys: bid, ask, timestamp_ms
    instrument : ExnessInstrument
        Exness instrument enum value (e.g., ExnessInstrument.EURUSD)
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
    ...     ticks, ExnessInstrument.EURUSD, threshold_bps=250
    ... )
    """
    _check_exness_available()

    # Set default strictness
    if strictness is None:
        strictness = ValidationStrictness.Strict  # type: ignore[attr-defined]

    # Create builder
    builder = ExnessRangeBarBuilder(instrument, threshold_bps, strictness)  # type: ignore[misc]

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


__all__ = [
    "ExnessInstrument",
    "ExnessRangeBarBuilder",
    "ValidationStrictness",
    "is_exness_available",
    "process_exness_ticks_to_dataframe",
]
