# Issue #46: Modularization M4 - Orchestration subpackage
"""Orchestration subpackage for range bar retrieval and precomputation.

Re-exports public symbols for backward compatibility:
    from rangebar.orchestration import get_range_bars, precompute_range_bars
"""

from .count_bounded import get_n_range_bars
from .models import PrecomputeProgress, PrecomputeResult
from .precompute import precompute_range_bars
from .range_bars import get_range_bars, get_range_bars_pandas

__all__ = [
    "PrecomputeProgress",
    "PrecomputeResult",
    "get_n_range_bars",
    "get_range_bars",
    "get_range_bars_pandas",
    "precompute_range_bars",
]
