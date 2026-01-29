# Issue #46: Modularization M4 - Extract data classes from __init__.py
"""Data classes for orchestration results and progress tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
