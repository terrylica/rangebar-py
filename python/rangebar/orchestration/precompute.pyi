from collections.abc import Callable
from typing import Literal

from .models import PrecomputeProgress, PrecomputeResult

def precompute_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    chunk_size: int = 100_000,
    invalidate_existing: Literal["overlap", "full", "none", "smart"] = "smart",
    progress_callback: Callable[[PrecomputeProgress], None] | None = None,
    include_microstructure: bool = False,
    validate_on_complete: Literal["error", "warn", "skip"] = "error",
    continuity_tolerance_pct: float = 0.001,
    cache_dir: str | None = None,
    inter_bar_lookback_bars: int | None = None,
) -> PrecomputeResult:
    ...
