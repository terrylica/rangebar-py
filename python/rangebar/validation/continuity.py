# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
"""Continuity validation for range bar data (Issue #19, Issue #5).

This module provides tiered gap classification and validation for range bars,
enabling nuanced handling of different gap magnitudes from floating-point
precision artifacts to session boundary breaks.

Gap Tiers (from Issue #19 empirical analysis of 30-month BTC data):
- PRECISION: < 0.001% - Floating-point artifacts (always ignored)
- NOISE: 0.001% - 0.01% - Tick-level noise (logged, not flagged)
- MARKET_MOVE: 0.01% - 0.1% - Normal market movement (configurable)
- MICROSTRUCTURE: > 0.1% - Flash crashes, liquidations (warning/error)
- SESSION_BOUNDARY: > threshold*2 - Definite session break (always error)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal

import pandas as pd

from rangebar.constants import (
    _CRYPTO_BASES,
    _FOREX_CURRENCIES,
    CONTINUITY_TOLERANCE_PCT,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "ASSET_CLASS_MULTIPLIERS",
    "VALIDATION_PRESETS",
    "AssetClass",
    "GapInfo",
    "GapTier",
    "TierSummary",
    "TierThresholds",
    "TieredValidationResult",
    "ValidationPreset",
    "detect_asset_class",
    "validate_continuity",
    "validate_continuity_tiered",
    "validate_junction_continuity",
]


# ============================================================================
# Exceptions (kept in __init__.py for backward compatibility, imported here)
# ============================================================================


class ContinuityError(Exception):
    """Raised when range bar continuity validation fails.

    This indicates bars from different processing sessions were combined,
    which can happen when:
    1. Data was fetched in multiple chunks without processor state continuity
    2. Cached data from different runs was concatenated
    3. There are actual gaps in the source tick data

    Attributes
    ----------
    message : str
        Human-readable error message
    discontinuities : list[dict]
        List of discontinuity details with bar_index, prev_close, curr_open, gap_pct
    """

    def __init__(self, message: str, discontinuities: list[dict] | None = None) -> None:
        super().__init__(message)
        self.discontinuities = discontinuities or []


class ContinuityWarning(UserWarning):
    """Warning for non-fatal continuity issues."""



# ============================================================================
# Gap Classification (Issue #19)
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
    if symbol_upper.endswith(("USDT", "BUSD", "USDC", "TUSD", "FDUSD")):
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


# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class GapInfo:
    """Details of a single gap between consecutive bars.

    Attributes
    ----------
    bar_index : int
        Index of the bar with the gap (0-based)
    prev_close : float
        Close price of the previous bar
    curr_open : float
        Open price of the current bar
    gap_pct : float
        Gap magnitude as percentage (e.g., 0.01 = 1%)
    tier : GapTier
        Severity classification of this gap
    timestamp : pd.Timestamp | None
        Timestamp of the bar (if available from DataFrame index)
    """

    bar_index: int
    prev_close: float
    curr_open: float
    gap_pct: float
    tier: GapTier
    timestamp: pd.Timestamp | None = None


@dataclass
class TierSummary:
    """Per-tier statistics for gap analysis.

    Attributes
    ----------
    count : int
        Number of gaps in this tier
    max_gap_pct : float
        Maximum gap percentage in this tier
    avg_gap_pct : float
        Average gap percentage in this tier (0 if count == 0)
    """

    count: int = 0
    max_gap_pct: float = 0.0
    avg_gap_pct: float = 0.0


@dataclass
class TieredValidationResult:
    """Comprehensive validation result with tier breakdown.

    This result provides detailed gap analysis categorized by severity tier,
    enabling nuanced handling of different gap magnitudes.

    Attributes
    ----------
    is_valid : bool
        True if no SESSION_BOUNDARY gaps (tier 5) detected
    bar_count : int
        Total number of bars validated
    gaps_by_tier : dict[GapTier, TierSummary]
        Per-tier statistics
    all_gaps : list[GapInfo]
        All gaps above PRECISION tier (detailed list)
    threshold_used_pct : float
        Range bar threshold used for validation (as percentage)
    asset_class_detected : AssetClass
        Auto-detected or overridden asset class
    preset_used : str | None
        Name of preset used, or None for custom config

    Examples
    --------
    >>> result = validate_continuity_tiered(df, validation="research")
    >>> result.is_valid
    True
    >>> result.gaps_by_tier[GapTier.MICROSTRUCTURE].count
    3
    >>> result.has_microstructure_events
    True
    """

    is_valid: bool
    bar_count: int
    gaps_by_tier: dict[GapTier, TierSummary]
    all_gaps: list[GapInfo]
    threshold_used_pct: float
    asset_class_detected: AssetClass
    preset_used: str | None

    @property
    def has_session_breaks(self) -> bool:
        """True if any SESSION_BOUNDARY gaps detected."""
        return self.gaps_by_tier[GapTier.SESSION_BOUNDARY].count > 0

    @property
    def has_microstructure_events(self) -> bool:
        """True if any MICROSTRUCTURE gaps detected."""
        return self.gaps_by_tier[GapTier.MICROSTRUCTURE].count > 0

    def summary_dict(self) -> dict[str, int]:
        """Return gap counts by tier name for logging.

        Returns
        -------
        dict[str, int]
            Mapping of tier name to gap count

        Examples
        --------
        >>> result.summary_dict()
        {'PRECISION': 0, 'NOISE': 5, 'MARKET_MOVE': 10, ...}
        """
        return {
            tier.name: summary.count for tier, summary in self.gaps_by_tier.items()
        }


# ============================================================================
# Validation Functions
# ============================================================================


def validate_junction_continuity(
    older_bars: pd.DataFrame,
    newer_bars: pd.DataFrame,
    tolerance_pct: float = CONTINUITY_TOLERANCE_PCT,
) -> tuple[bool, float | None]:
    """Validate continuity at junction between two bar DataFrames.

    Checks that older_bars[-1].Close == newer_bars[0].Open (within tolerance).

    Parameters
    ----------
    older_bars : pd.DataFrame
        Older bars (chronologically earlier)
    newer_bars : pd.DataFrame
        Newer bars (chronologically later)
    tolerance_pct : float
        Maximum allowed relative difference (0.0001 = 0.01%)

    Returns
    -------
    tuple[bool, float | None]
        (is_continuous, gap_pct) where gap_pct is the relative difference if
        discontinuous, None if continuous
    """
    if older_bars.empty or newer_bars.empty:
        return True, None

    last_close = older_bars.iloc[-1]["Close"]
    first_open = newer_bars.iloc[0]["Open"]

    if last_close == 0:
        return True, None  # Avoid division by zero

    gap_pct = abs(first_open - last_close) / abs(last_close)

    if gap_pct <= tolerance_pct:
        return True, None

    return False, gap_pct


def validate_continuity(
    df: pd.DataFrame,
    tolerance_pct: float | None = None,
    threshold_decimal_bps: int = 250,
) -> dict:
    """Validate that range bars come from continuous single-session processing.

    Range bars do NOT guarantee bar[i].close == bar[i+1].open. The next bar
    opens at the first tick AFTER the previous bar closes, not at the close
    price. This is by design - range bars capture actual market movements.

    What this function validates:
    1. OHLC invariants hold (High >= max(Open, Close), Low <= min(Open, Close))
    2. Price gaps between bars don't exceed threshold + tolerance
       (gaps larger than threshold indicate bars from different sessions)
    3. Timestamps are monotonically increasing

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with OHLC columns
    tolerance_pct : float, optional
        Additional tolerance beyond threshold for gap detection.
        Default is 0.5% (0.005) to account for floating-point precision.
    threshold_decimal_bps : int, default=250
        Range bar threshold used to generate these bars (250 = 0.25%).
        Gaps larger than this indicate session boundaries.

    Returns
    -------
    dict
        Validation result with keys:
        - is_valid: bool - True if bars appear from single session
        - bar_count: int - Total number of bars
        - discontinuity_count: int - Number of session boundaries found
        - discontinuities: list[dict] - Details of each discontinuity

    Notes
    -----
    A "discontinuity" here means bars from different processing sessions
    were combined. Within a single session, the gap between bar[i].close
    and bar[i+1].open should never exceed the threshold (since a bar only
    closes when price moves by threshold from open).

    The tolerance parameter accounts for:
    - Floating-point precision in price calculations
    - Minor price movements between close tick and next tick
    """
    if tolerance_pct is None:
        tolerance_pct = 0.005  # 0.5% default tolerance

    if df.empty:
        return {
            "is_valid": True,
            "bar_count": 0,
            "discontinuity_count": 0,
            "discontinuities": [],
        }

    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        msg = f"DataFrame must have columns: {required_cols}"
        raise ValueError(msg)

    discontinuities = []

    # Convert threshold to percentage (250 dbps = 0.25% = 0.0025)
    threshold_pct = threshold_decimal_bps / 100000.0

    # Maximum allowed gap = threshold + tolerance
    # Within single session, gap should never exceed this
    max_gap_pct = threshold_pct + tolerance_pct

    close_prices = df["Close"].to_numpy()[:-1]
    open_prices = df["Open"].to_numpy()[1:]

    for i, (prev_close, curr_open) in enumerate(
        zip(close_prices, open_prices, strict=False)
    ):
        if prev_close == 0:
            continue

        gap_pct = abs(curr_open - prev_close) / abs(prev_close)
        if gap_pct > max_gap_pct:
            discontinuities.append(
                {
                    "bar_index": i + 1,
                    "prev_close": float(prev_close),
                    "curr_open": float(curr_open),
                    "gap_pct": float(gap_pct),
                }
            )

    return {
        "is_valid": len(discontinuities) == 0,
        "bar_count": len(df),
        "discontinuity_count": len(discontinuities),
        "discontinuities": discontinuities,
    }


def _resolve_validation(
    validation: str | dict | ValidationPreset,
    symbol: str | None = None,
) -> tuple[ValidationPreset, AssetClass, str | None]:
    """Resolve validation parameter to preset and asset class.

    Parameters
    ----------
    validation : str, dict, or ValidationPreset
        Validation configuration:
        - "auto": Auto-detect asset class from symbol
        - str: Preset name ("research", "strict", "crypto", etc.)
        - dict: Custom config {"tolerance_pct": 0.01, "mode": "warn"}
        - ValidationPreset: Direct preset instance
    symbol : str, optional
        Symbol for asset class auto-detection

    Returns
    -------
    tuple[ValidationPreset, AssetClass, str | None]
        (resolved preset, detected asset class, preset name or None)

    Raises
    ------
    ValueError
        If unknown preset name provided
    TypeError
        If validation is not a supported type
    """
    # Handle "auto" - detect from symbol
    if validation == "auto":
        asset_class = detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        preset_name = (
            asset_class.value if asset_class != AssetClass.UNKNOWN else "standard"
        )
        return VALIDATION_PRESETS[preset_name], asset_class, preset_name

    # Handle preset string
    if isinstance(validation, str):
        if validation not in VALIDATION_PRESETS:
            valid_presets = ", ".join(sorted(VALIDATION_PRESETS.keys()))
            msg = (
                f"Unknown validation preset: {validation!r}. "
                f"Valid presets: {valid_presets}"
            )
            raise ValueError(msg)
        preset = VALIDATION_PRESETS[validation]
        asset_class = preset.asset_class or (
            detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        )
        return preset, asset_class, validation

    # Handle dict
    if isinstance(validation, dict):
        tier_thresholds = validation.get("tier_thresholds", TierThresholds())
        if isinstance(tier_thresholds, dict):
            tier_thresholds = TierThresholds(**tier_thresholds)
        preset = ValidationPreset(
            tolerance_pct=validation["tolerance_pct"],
            mode=validation["mode"],
            tier_thresholds=tier_thresholds,
            description="Custom",
        )
        asset_class = detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        return preset, asset_class, None

    # Handle ValidationPreset directly
    if isinstance(validation, ValidationPreset):
        asset_class = validation.asset_class or (
            detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
        )
        return validation, asset_class, None

    msg = (
        f"Invalid validation type: {type(validation).__name__}. "
        "Expected str, dict, or ValidationPreset"
    )
    raise TypeError(msg)


def _classify_gap(
    gap_pct: float,
    tier_thresholds: TierThresholds,
    session_threshold_pct: float,
) -> GapTier:
    """Classify a gap into a severity tier.

    Parameters
    ----------
    gap_pct : float
        Gap magnitude as percentage (absolute value)
    tier_thresholds : TierThresholds
        Boundaries between tiers
    session_threshold_pct : float
        Session boundary threshold (threshold * session_factor)

    Returns
    -------
    GapTier
        Severity classification
    """
    if gap_pct < tier_thresholds.precision:
        return GapTier.PRECISION
    if gap_pct < tier_thresholds.noise:
        return GapTier.NOISE
    if gap_pct < tier_thresholds.market_move:
        return GapTier.MARKET_MOVE
    if gap_pct < session_threshold_pct:
        return GapTier.MICROSTRUCTURE
    return GapTier.SESSION_BOUNDARY


def validate_continuity_tiered(
    df: pd.DataFrame,
    threshold_decimal_bps: int = 250,
    *,
    validation: str | dict | ValidationPreset = "standard",
    symbol: str | None = None,
) -> TieredValidationResult:
    """Validate range bar continuity with tiered gap classification.

    This function categorizes gaps by severity tier, enabling nuanced
    handling of different gap magnitudes. It's the opt-in v6.2.0 API
    that will become the default in v7.0.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with OHLCV columns
    threshold_decimal_bps : int, default=250
        Range bar threshold (250 = 0.25% = 25 basis points)
    validation : str, dict, or ValidationPreset, default="standard"
        Validation configuration:
        - "auto": Auto-detect asset class from symbol
        - str: Preset name ("research", "strict", "crypto", etc.)
        - dict: Custom config {"tolerance_pct": 0.01, "mode": "warn"}
        - ValidationPreset: Direct preset instance
    symbol : str, optional
        Symbol for asset class auto-detection. If None and validation is
        "auto", uses "standard" preset.

    Returns
    -------
    TieredValidationResult
        Comprehensive result with per-tier statistics

    Raises
    ------
    ContinuityError
        If validation mode is "error" and tolerance exceeded
    ContinuityWarning
        If validation mode is "warn" and tolerance exceeded (via warnings)

    Examples
    --------
    >>> result = validate_continuity_tiered(df, validation="research")
    >>> print(f"Valid: {result.is_valid}")
    Valid: True

    >>> # Custom configuration
    >>> result = validate_continuity_tiered(
    ...     df,
    ...     validation={"tolerance_pct": 0.015, "mode": "warn"},
    ...     symbol="BTCUSDT",
    ... )

    >>> # Auto-detect asset class
    >>> result = validate_continuity_tiered(df, validation="auto", symbol="EURUSD")
    >>> result.asset_class_detected
    <AssetClass.FOREX: 'forex'>
    """
    min_bars_for_gap = 2  # Need at least 2 bars to have a gap
    if len(df) < min_bars_for_gap:
        # No gaps possible with fewer than 2 bars
        return TieredValidationResult(
            is_valid=True,
            bar_count=len(df),
            gaps_by_tier={tier: TierSummary() for tier in GapTier},
            all_gaps=[],
            threshold_used_pct=threshold_decimal_bps / 10000.0,
            asset_class_detected=(
                detect_asset_class(symbol) if symbol else AssetClass.UNKNOWN
            ),
            preset_used=validation if isinstance(validation, str) else None,
        )

    # Resolve validation configuration
    preset, asset_class, preset_name = _resolve_validation(validation, symbol)

    # Skip validation if mode is "skip"
    if preset.mode == "skip":
        return TieredValidationResult(
            is_valid=True,
            bar_count=len(df),
            gaps_by_tier={tier: TierSummary() for tier in GapTier},
            all_gaps=[],
            threshold_used_pct=threshold_decimal_bps / 10000.0,
            asset_class_detected=asset_class,
            preset_used=preset_name,
        )

    # Calculate session boundary threshold
    threshold_pct = threshold_decimal_bps / 10000.0  # Convert to percentage
    session_threshold_pct = threshold_pct * preset.tier_thresholds.session_factor

    # Analyze gaps
    all_gaps: list[GapInfo] = []
    tier_gaps: dict[GapTier, list[float]] = {tier: [] for tier in GapTier}

    # Get Close and Open columns
    close_col = "Close" if "Close" in df.columns else "close"
    open_col = "Open" if "Open" in df.columns else "open"

    closes = df[close_col].to_numpy()
    opens = df[open_col].to_numpy()
    index = df.index

    for i in range(1, len(df)):
        prev_close = float(closes[i - 1])
        curr_open = float(opens[i])

        if prev_close == 0:
            continue  # Skip division by zero

        gap_pct = abs(curr_open - prev_close) / prev_close
        tier = _classify_gap(gap_pct, preset.tier_thresholds, session_threshold_pct)

        tier_gaps[tier].append(gap_pct)

        # Record non-PRECISION gaps for detailed list
        if tier != GapTier.PRECISION:
            timestamp = index[i] if isinstance(index, pd.DatetimeIndex) else None
            all_gaps.append(
                GapInfo(
                    bar_index=i,
                    prev_close=prev_close,
                    curr_open=curr_open,
                    gap_pct=gap_pct,
                    tier=tier,
                    timestamp=timestamp,
                )
            )

    # Build tier summaries
    gaps_by_tier: dict[GapTier, TierSummary] = {}
    for tier in GapTier:
        gaps = tier_gaps[tier]
        if gaps:
            gaps_by_tier[tier] = TierSummary(
                count=len(gaps),
                max_gap_pct=max(gaps),
                avg_gap_pct=sum(gaps) / len(gaps),
            )
        else:
            gaps_by_tier[tier] = TierSummary()

    # Determine validity: no SESSION_BOUNDARY gaps
    is_valid = gaps_by_tier[GapTier.SESSION_BOUNDARY].count == 0

    # Check tolerance threshold
    tolerance_exceeded = any(
        gap.gap_pct > preset.tolerance_pct
        for gap in all_gaps
        if gap.tier >= GapTier.MARKET_MOVE  # Only check MARKET_MOVE and above
    )

    result = TieredValidationResult(
        is_valid=is_valid,
        bar_count=len(df),
        gaps_by_tier=gaps_by_tier,
        all_gaps=all_gaps,
        threshold_used_pct=threshold_pct,
        asset_class_detected=asset_class,
        preset_used=preset_name,
    )

    # Handle tolerance violations based on mode
    if tolerance_exceeded:
        violating_gaps = [g for g in all_gaps if g.gap_pct > preset.tolerance_pct]
        max_gap = max(violating_gaps, key=lambda g: g.gap_pct)

        msg = (
            f"Continuity tolerance exceeded: {len(violating_gaps)} gap(s) > "
            f"{preset.tolerance_pct:.4%}. Max gap: {max_gap.gap_pct:.4%} at bar "
            f"{max_gap.bar_index}. Tier breakdown: {result.summary_dict()}"
        )

        if preset.mode == "error":
            discontinuities = [
                {
                    "bar_index": g.bar_index,
                    "prev_close": g.prev_close,
                    "curr_open": g.curr_open,
                    "gap_pct": g.gap_pct,
                    "tier": g.tier.name,
                }
                for g in violating_gaps
            ]
            raise ContinuityError(msg, discontinuities)

        if preset.mode == "warn":
            warnings.warn(msg, ContinuityWarning, stacklevel=2)

    return result
