# Modularization M2: Extract RangeBarProcessor from __init__.py
# Issue #46: Reduce __init__.py from 4,276 to ~500 lines
"""Processor subpackage for range bar construction.

Provides the RangeBarProcessor class and related processing functions.
"""

from .core import RangeBarProcessor

__all__ = [
    "RangeBarProcessor",
]
