from dataclasses import dataclass

@dataclass
class BackfillResult:
    symbol: str
    """Trading symbol that was backfilled."""
    threshold_decimal_bps: int
    """Threshold used for bar construction."""
    bars_written: int
    """Number of new bars written to ClickHouse."""
    gap_seconds: float
    """Gap in seconds between latest cached bar and now (before backfill)."""
    latest_ts_before: int | None
    """Latest bar timestamp (ms) before backfill, None if no cached bars."""
    latest_ts_after: int | None
    """Latest bar timestamp (ms) after backfill."""
    duration_seconds: float = ...
    """Time taken for the backfill operation."""
    error: str | None = ...
    """Error message if backfill failed, None on success."""

@dataclass
class LoopState:
    iteration: int
    """Number of completed loop iterations."""
    total_bars_written: int
    """Cumulative bars written across all iterations."""
    started_at: str
    """ISO 8601 timestamp of loop start."""
    last_gaps: dict[str, float]
    """Last observed gap (seconds) per 'SYMBOL@threshold' key."""

def backfill_recent(
    symbol: str,
    threshold_decimal_bps: int = 250,
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> BackfillResult:
    ...

def backfill_all_recent(
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> list[BackfillResult]:
    ...

def run_adaptive_loop(
    *,
    include_microstructure: bool = True,
    verbose: bool = False,
) -> None:
    ...
