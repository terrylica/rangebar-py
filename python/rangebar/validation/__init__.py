"""Validation framework for microstructure features (Issue #25).

Provides tiered validation for market microstructure features:
- Tier 1: Auto-validation on every precompute (<30 sec)
- Tier 2: Statistical validation before production ML (~10 min)
- Tier 3: Feature importance and drift analysis (30+ min, on-demand)
"""

from .tier1 import FEATURE_COLS, validate_tier1
from .tier2 import validate_tier2

__all__ = [
    "FEATURE_COLS",
    "validate_tier1",
    "validate_tier2",
]
