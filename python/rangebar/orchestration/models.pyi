from dataclasses import dataclass
from typing import Literal

@dataclass
class PrecomputeProgress:
    phase: Literal["fetching", "processing", "caching"]
    """Current processing phase."""
    current_month: str
    """Current month being processed (YYYY-MM format)."""
    months_completed: int
    """Number of months already processed."""
    months_total: int
    """Total number of months to process."""
    bars_generated: int
    """Cumulative bars generated so far."""
    ticks_processed: int
    """Cumulative ticks processed so far."""
    elapsed_seconds: float
    """Seconds elapsed since precomputation started."""

@dataclass
class PrecomputeResult:
    symbol: str
    """Trading symbol that was precomputed."""
    threshold_decimal_bps: int
    """Threshold used for bar construction."""
    start_date: str
    """Start date of precomputed range (YYYY-MM-DD)."""
    end_date: str
    """End date of precomputed range (YYYY-MM-DD)."""
    total_bars: int
    """Total number of bars generated."""
    total_ticks: int
    """Total number of ticks processed."""
    elapsed_seconds: float
    """Total time taken for precomputation."""
    continuity_valid: bool | None
    """Whether all bars pass continuity validation. None if validation was skipped."""
    cache_key: str
    """Cache key for the stored bars."""
