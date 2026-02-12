from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal

import pandas as pd

class ContinuityError(Exception):
    discontinuities: list[dict]
    """List of discontinuity details (bar_index, prev_close, next_open, gap_pct)."""

    def __init__(
        self, message: str, discontinuities: list[dict] | None = None
    ) -> None: ...

class ContinuityWarning(UserWarning):
    ...

class GapTier(IntEnum):
    PRECISION = 1
    """< 0.001% - Floating-point artifacts (always ignored)"""
    NOISE = 2
    """0.001% - 0.01% - Tick-level noise (logged, not flagged)"""
    MARKET_MOVE = 3
    """0.01% - 0.1% - Normal market movement (configurable)"""
    MICROSTRUCTURE = 4
    """> 0.1% - Flash crashes, liquidations (warning/error)"""
    SESSION_BOUNDARY = 5
    """> threshold*2 - Definite session break (always error)"""

class AssetClass(Enum):
    CRYPTO = "crypto"
    """24/7 markets, flash crashes possible"""
    FOREX = "forex"
    """Session-based, weekend gaps"""
    EQUITIES = "equities"
    """Overnight gaps, circuit breakers"""
    UNKNOWN = "unknown"
    """Fallback to crypto defaults"""

ASSET_CLASS_MULTIPLIERS: dict[AssetClass, float]
"""Tolerance multipliers by asset class (relative to baseline)."""

def detect_asset_class(symbol: str) -> AssetClass:
    ...

@dataclass(frozen=True)
class TierThresholds:
    precision: float = ...
    """Tier 1/2 boundary (default: 0.00001 = 0.001%)"""
    noise: float = ...
    """Tier 2/3 boundary (default: 0.0001 = 0.01%)"""
    market_move: float = ...
    """Tier 3/4 boundary (default: 0.001 = 0.1%)"""
    session_factor: float = ...
    """Tier 5 multiplier (default: 2.0)"""

@dataclass(frozen=True)
class ValidationPreset:
    tolerance_pct: float
    """Maximum gap percentage before flagging (e.g., 0.01 = 1%)"""
    mode: Literal["error", "warn", "skip"]
    """Behavior on validation failure"""
    tier_thresholds: TierThresholds = ...
    """Boundaries for gap tier classification"""
    asset_class: AssetClass | None = ...
    """Override auto-detection if set"""
    description: str = ...
    """Human-readable description of the preset"""

VALIDATION_PRESETS: dict[str, ValidationPreset]
"""Named validation presets for common scenarios.

General-purpose:
- "permissive": 5% tolerance, warn mode
- "research": 2% tolerance, warn mode (exploratory analysis)
- "standard": 1% tolerance, warn mode (production backtesting)
- "strict": 0.5% tolerance, error mode (ML training data)
- "paranoid": 0.1% tolerance, error mode (original v6.1.0 behavior)

Asset-class specific:
- "crypto": 2% tolerance, crypto asset class
- "forex": 1% tolerance, forex asset class
- "equities": 3% tolerance, equities asset class

Special:
- "skip": Disable validation entirely
- "audit": 0.2% tolerance, error mode (data quality audit)
"""

@dataclass
class GapInfo:
    bar_index: int
    """Index of the bar with the gap (0-based)"""
    prev_close: float
    """Close price of the previous bar"""
    curr_open: float
    """Open price of the current bar"""
    gap_pct: float
    """Gap magnitude as percentage (e.g., 0.01 = 1%)"""
    tier: GapTier
    """Severity classification of this gap"""
    timestamp: pd.Timestamp | None = ...
    """Timestamp of the bar (if available from DataFrame index)"""

@dataclass
class TierSummary:
    count: int = ...
    """Number of gaps in this tier"""
    max_gap_pct: float = ...
    """Maximum gap percentage in this tier"""
    avg_gap_pct: float = ...
    """Average gap percentage in this tier (0 if count == 0)"""

@dataclass
class TieredValidationResult:
    is_valid: bool
    """True if no SESSION_BOUNDARY gaps (tier 5) detected"""
    bar_count: int
    """Total number of bars validated"""
    gaps_by_tier: dict[GapTier, TierSummary]
    """Per-tier statistics"""
    all_gaps: list[GapInfo]
    """All gaps above PRECISION tier (detailed list)"""
    threshold_used_pct: float
    """Range bar threshold used for validation (as percentage)"""
    asset_class_detected: AssetClass
    """Auto-detected or overridden asset class"""
    preset_used: str | None
    """Name of preset used, or None for custom config"""

    @property
    def has_session_breaks(self) -> bool:
        ...

    @property
    def has_microstructure_events(self) -> bool:
        ...

    def summary_dict(self) -> dict[str, int]:
        ...

def validate_continuity_tiered(
    df: pd.DataFrame,
    threshold_decimal_bps: int = 250,
    *,
    validation: str | dict | ValidationPreset = "standard",
    symbol: str | None = None,
) -> TieredValidationResult:
    ...
