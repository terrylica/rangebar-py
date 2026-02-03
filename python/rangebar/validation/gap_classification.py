# Issue #19: Gap classification extracted from continuity.py for modularization
"""Gap classification types and presets for range bar validation.

This module provides the tiered gap classification system based on empirical
analysis of 30-month BTC data (Issue #19). It identifies 49 legitimate market
microstructure events and classifies gaps into severity tiers.

Gap Tiers:
- PRECISION: < 0.001% - Floating-point artifacts (always ignored)
- NOISE: 0.001% - 0.01% - Tick-level noise (logged, not flagged)
- MARKET_MOVE: 0.01% - 0.1% - Normal market movement (configurable)
- MICROSTRUCTURE: > 0.1% - Flash crashes, liquidations (warning/error)
- SESSION_BOUNDARY: > threshold*2 - Definite session break (always error)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Literal

from rangebar.constants import _CRYPTO_BASES, _FOREX_CURRENCIES

__all__ = [
    "ASSET_CLASS_MULTIPLIERS",
    "VALIDATION_PRESETS",
    "AssetClass",
    "GapTier",
    "TierThresholds",
    "ValidationPreset",
    "detect_asset_class",
]


# ============================================================================
# Gap Tier Enum
# ============================================================================


class GapTier(IntEnum):
    """Gap severity classification for range bar continuity validation.

    Tiers are based on empirical analysis of 30-month BTC data (Issue #19)
    which identified 49 legitimate market microstructure events.

    Examples
    --------
    >>> gap_pct = 0.05  # 0.05% gap
    >>> if gap_pct < 0.00001:
    ...     tier = GapTier.PRECISION
    >>> elif gap_pct < 0.0001:
    ...     tier = GapTier.NOISE
    >>> elif gap_pct < 0.001:
    ...     tier = GapTier.MARKET_MOVE
    >>> elif gap_pct < threshold * 2:
    ...     tier = GapTier.MICROSTRUCTURE
    >>> else:
    ...     tier = GapTier.SESSION_BOUNDARY
    """

    PRECISION = 1  # < 0.001% - Floating-point artifacts (always ignored)
    NOISE = 2  # 0.001% - 0.01% - Tick-level noise (logged, not flagged)
    MARKET_MOVE = 3  # 0.01% - 0.1% - Normal market movement (configurable)
    MICROSTRUCTURE = 4  # > 0.1% - Flash crashes, liquidations (warning/error)
    SESSION_BOUNDARY = 5  # > threshold*2 - Definite session break (always error)


# ============================================================================
# Asset Class Enum
# ============================================================================


class AssetClass(Enum):
    """Asset class for tolerance calibration.

    Different asset classes have different typical gap magnitudes:
    - Crypto: 24/7 markets, flash crashes possible, baseline tolerance
    - Forex: Session-based, weekend gaps, tighter tolerance
    - Equities: Overnight gaps, circuit breakers, looser tolerance

    Examples
    --------
    >>> from rangebar import detect_asset_class, AssetClass
    >>> detect_asset_class("BTCUSDT")
    <AssetClass.CRYPTO: 'crypto'>
    >>> detect_asset_class("EURUSD")
    <AssetClass.FOREX: 'forex'>
    """

    CRYPTO = "crypto"  # 24/7 markets, flash crashes possible
    FOREX = "forex"  # Session-based, weekend gaps
    EQUITIES = "equities"  # Overnight gaps, circuit breakers
    UNKNOWN = "unknown"  # Fallback to crypto defaults


# Tolerance multipliers by asset class (relative to baseline)
ASSET_CLASS_MULTIPLIERS: dict[AssetClass, float] = {
    AssetClass.CRYPTO: 1.0,  # Baseline
    AssetClass.FOREX: 0.5,  # Tighter (more stable)
    AssetClass.EQUITIES: 1.5,  # Looser (overnight gaps)
    AssetClass.UNKNOWN: 1.0,  # Default to crypto
}


def detect_asset_class(symbol: str) -> AssetClass:
    """Auto-detect asset class from symbol pattern.

    Detection Rules:
    - Crypto: Contains common crypto bases (BTC, ETH, BNB, SOL, etc.)
             or ends with USDT/BUSD/USDC
    - Forex: Standard 6-char pairs (EURUSD, GBPJPY, etc.)
             or commodity symbols (XAU, XAG, BRENT, WTI)
    - Unknown: Fallback for unrecognized patterns

    Parameters
    ----------
    symbol : str
        Trading symbol (case-insensitive)

    Returns
    -------
    AssetClass
        Detected asset class

    Examples
    --------
    >>> detect_asset_class("BTCUSDT")
    <AssetClass.CRYPTO: 'crypto'>
    >>> detect_asset_class("EURUSD")
    <AssetClass.FOREX: 'forex'>
    >>> detect_asset_class("AAPL")
    <AssetClass.UNKNOWN: 'unknown'>
    """
    symbol_upper = symbol.upper()

    # Crypto patterns: contains known base or ends with stablecoin
    if any(base in symbol_upper for base in _CRYPTO_BASES):
        return AssetClass.CRYPTO
    # Stablecoins: USDE (Ethena), USDM (Mountain Protocol) added for Issue #62
    if symbol_upper.endswith(("USDT", "BUSD", "USDC", "TUSD", "FDUSD", "USDE", "USDM")):
        return AssetClass.CRYPTO

    # Forex patterns: 6-char standard pairs (e.g., EURUSD)
    forex_pair_length = 6
    if len(symbol_upper) == forex_pair_length:
        base, quote = symbol_upper[:3], symbol_upper[3:]
        if base in _FOREX_CURRENCIES and quote in _FOREX_CURRENCIES:
            return AssetClass.FOREX

    # Commodities via forex brokers
    if any(x in symbol_upper for x in ("XAU", "XAG", "BRENT", "WTI")):
        return AssetClass.FOREX

    return AssetClass.UNKNOWN


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass(frozen=True)
class TierThresholds:
    """Configurable boundaries between gap tiers (in percentage).

    These thresholds define the boundaries for classifying gaps into tiers.
    Values are percentages (e.g., 0.00001 = 0.001%).

    Attributes
    ----------
    precision : float
        Tier 1/2 boundary - gaps below this are floating-point artifacts
    noise : float
        Tier 2/3 boundary - gaps below this are tick-level noise
    market_move : float
        Tier 3/4 boundary - gaps below this are normal market movement
    session_factor : float
        Tier 5 multiplier - gaps > (threshold * factor) are session breaks

    Examples
    --------
    >>> thresholds = TierThresholds()
    >>> thresholds.precision
    1e-05
    >>> thresholds.noise
    0.0001
    """

    precision: float = 0.00001  # 0.001% - Tier 1/2 boundary
    noise: float = 0.0001  # 0.01% - Tier 2/3 boundary
    market_move: float = 0.001  # 0.1% - Tier 3/4 boundary
    session_factor: float = 2.0  # Tier 5 = threshold * factor


@dataclass(frozen=True)
class ValidationPreset:
    """Immutable validation configuration preset.

    Presets bundle tolerance, behavior mode, and tier thresholds into
    named configurations for common use cases.

    Attributes
    ----------
    tolerance_pct : float
        Maximum gap percentage before flagging (e.g., 0.01 = 1%)
    mode : Literal["error", "warn", "skip"]
        Behavior on validation failure
    tier_thresholds : TierThresholds
        Boundaries for gap tier classification
    asset_class : AssetClass | None
        Override auto-detection if set
    description : str
        Human-readable description of the preset

    Examples
    --------
    >>> preset = VALIDATION_PRESETS["research"]
    >>> preset.tolerance_pct
    0.02
    >>> preset.mode
    'warn'
    """

    tolerance_pct: float  # Max gap before flagging
    mode: Literal["error", "warn", "skip"]  # Behavior on failure
    tier_thresholds: TierThresholds = field(default_factory=TierThresholds)
    asset_class: AssetClass | None = None  # Override auto-detection
    description: str = ""


# Named validation presets for common scenarios
VALIDATION_PRESETS: dict[str, ValidationPreset] = {
    # =========================================================================
    # GENERAL-PURPOSE PRESETS
    # =========================================================================
    "permissive": ValidationPreset(
        tolerance_pct=0.05,  # 5%
        mode="warn",
        description="Accept most microstructure events, warn on extreme gaps",
    ),
    "research": ValidationPreset(
        tolerance_pct=0.02,  # 2%
        mode="warn",
        description="Standard exploratory analysis with monitoring",
    ),
    "standard": ValidationPreset(
        tolerance_pct=0.01,  # 1%
        mode="warn",
        description="Balanced tolerance for production backtesting",
    ),
    "strict": ValidationPreset(
        tolerance_pct=0.005,  # 0.5%
        mode="error",
        description="Strict validation for ML training data",
    ),
    "paranoid": ValidationPreset(
        tolerance_pct=0.001,  # 0.1%
        mode="error",
        description="Maximum strictness (original v6.1.0 behavior)",
    ),
    # =========================================================================
    # ASSET-CLASS SPECIFIC PRESETS
    # =========================================================================
    "crypto": ValidationPreset(
        tolerance_pct=0.02,  # 2%
        mode="warn",
        asset_class=AssetClass.CRYPTO,
        description="Crypto: Tuned for 24/7 markets with flash crashes",
    ),
    "forex": ValidationPreset(
        tolerance_pct=0.01,  # 1%
        mode="warn",
        asset_class=AssetClass.FOREX,
        description="Forex: Accounts for session boundaries",
    ),
    "equities": ValidationPreset(
        tolerance_pct=0.03,  # 3%
        mode="warn",
        asset_class=AssetClass.EQUITIES,
        description="Equities: Accounts for overnight gaps",
    ),
    # =========================================================================
    # SPECIAL PRESETS
    # =========================================================================
    "skip": ValidationPreset(
        tolerance_pct=0.0,
        mode="skip",
        description="Disable validation entirely",
    ),
    "audit": ValidationPreset(
        tolerance_pct=0.002,  # 0.2%
        mode="error",
        description="Data quality audit mode",
    ),
}
