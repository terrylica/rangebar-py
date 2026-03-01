"""Population configuration for cache population jobs.

Issue #110: Unified configuration via pydantic-settings.
Issue #128: Per-feature computation toggles.

All values load from environment variables (RANGEBAR_ prefix) with automatic
overlay: CLI > env > TOML > defaults via pydantic-settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PopulationConfig(BaseSettings):
    """Configuration for cache population jobs.

    Environment Variables
    ---------------------
    RANGEBAR_CRYPTO_MIN_THRESHOLD : int
        Default threshold in dbps (default: 250 = 0.25%)
    RANGEBAR_OUROBOROS_MODE : str
        Reset boundary: "year", "month", or "week" (default: "month")
    RANGEBAR_OUROBOROS_GUARD : str
        Guard behavior: "strict", "warn", "off" (default: "warn")
    RANGEBAR_INTER_BAR_LOOKBACK_COUNT : int
        Inter-bar lookback trade count (default: 200)
    RANGEBAR_INCLUDE_INTRA_BAR_FEATURES : bool
        Include intra-bar microstructure features (default: True)
    RANGEBAR_SYMBOL_GATE : str
        Symbol registry gate: "strict", "warn", "off" (default: "strict")
    RANGEBAR_CONTINUITY_TOLERANCE : float
        Continuity tolerance (default: 0.001)
    RANGEBAR_COMPUTE_TIER2 : bool
        Compute Tier 2 inter-bar features (default: True)
    RANGEBAR_COMPUTE_TIER3 : bool
        Compute Tier 3 inter-bar features (default: False)
    RANGEBAR_COMPUTE_HURST : bool
        Compute Hurst DFA (default: False)
    RANGEBAR_COMPUTE_PERMUTATION_ENTROPY : bool
        Compute permutation entropy (default: False)
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_",
        case_sensitive=False,
    )

    # Core — field names match env var suffixes (after RANGEBAR_ prefix)
    crypto_min_threshold: int = 250
    ouroboros_mode: Literal["year", "month", "week"] = "month"
    ouroboros_guard: Literal["strict", "warn", "off"] = "warn"
    inter_bar_lookback_count: int = 200
    include_intra_bar_features: bool = True
    symbol_gate: Literal["strict", "warn", "off"] = "strict"
    continuity_tolerance: float = 0.001

    # Issue #128: Per-feature computation toggles
    compute_tier2: bool = True
    compute_tier3: bool = False
    compute_hurst: bool = False
    compute_permutation_entropy: bool = False

    @field_validator("ouroboros_mode")
    @classmethod
    def _validate_ouroboros(cls, v: str) -> str:
        from rangebar.ouroboros import validate_ouroboros_mode

        validate_ouroboros_mode(v)
        return v

    # Backwards compatibility: existing code uses .default_threshold
    @property
    def default_threshold(self) -> int:
        """Alias for crypto_min_threshold (backwards compatibility)."""
        return self.crypto_min_threshold
