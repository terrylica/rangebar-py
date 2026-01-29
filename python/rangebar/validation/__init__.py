"""Validation framework for microstructure features (Issue #25) and cache integrity (Issue #39).

Provides tiered validation for market microstructure features and cache operations:
- Tier 0: Cache staleness detection (<5ms) - schema evolution support
- Tier 0: Post-storage validation after cache operations (<1 sec) - Issue #39
- Tier 1: Auto-validation on every precompute (<30 sec)
- Tier 2: Statistical validation before production ML (~10 min)
- Tier 3: Feature importance and drift analysis (30+ min, on-demand)
"""

from .cache_staleness import (
    StalenessResult,
    detect_staleness,
    validate_schema_version,
)
from .post_storage import (
    ValidationResult,
    compute_dataframe_checksum,
    validate_ohlc_invariants,
    validate_post_storage,
)
from .tier1 import FEATURE_COLS, validate_tier1
from .tier2 import validate_tier2

__all__ = [
    "FEATURE_COLS",
    "StalenessResult",
    "ValidationResult",
    "compute_dataframe_checksum",
    "detect_staleness",
    "validate_ohlc_invariants",
    "validate_post_storage",
    "validate_schema_version",
    "validate_tier1",
    "validate_tier2",
]
