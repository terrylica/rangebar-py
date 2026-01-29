# Modularization M2/M3: Extract RangeBarProcessor and process_trades_* from __init__.py
# Issue #46: Reduce __init__.py from 4,276 to ~500 lines
"""Processor subpackage for range bar construction.

Provides the RangeBarProcessor class and related processing functions.
"""

from .api import (
    process_trades_chunked,
    process_trades_polars,
    process_trades_to_dataframe,
    process_trades_to_dataframe_cached,
)
from .core import RangeBarProcessor

__all__ = [
    "RangeBarProcessor",
    "process_trades_chunked",
    "process_trades_polars",
    "process_trades_to_dataframe",
    "process_trades_to_dataframe_cached",
]
