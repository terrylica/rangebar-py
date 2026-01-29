# polars-exception: ClickHouse cache returns Pandas for backtesting.py
"""Cache staleness detection for schema evolution.

This module provides content-based validation to detect stale cached data
that was computed with older versions lacking microstructure features.

Tier 0 validation: Fast staleness detection (<5ms for 100K bars).
Run on every cache read when microstructure features are requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

from rangebar.constants import MICROSTRUCTURE_COLUMNS

logger = logging.getLogger(__name__)

# Semantic version has 3 parts
_VERSION_PARTS = 3


@dataclass
class StalenessResult:
    """Result of cache staleness detection.

    Attributes
    ----------
    is_stale : bool
        True if cached data is detected as stale and should be invalidated.
    reason : str | None
        Human-readable description of why data is stale (None if not stale).
    confidence : Literal["high", "medium", "low"]
        Confidence level of staleness detection.
    checks_passed : dict[str, bool]
        Individual validation checks and their results.
    recommendations : list[str]
        Suggested actions to resolve staleness.
    """

    is_stale: bool
    reason: str | None = None
    confidence: Literal["high", "medium", "low"] = "high"
    checks_passed: dict[str, bool] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


def _check_vwap(
    df: pd.DataFrame,
    checks: dict[str, bool],
    reasons: list[str],
) -> None:
    """Validate VWAP is within [Low, High] and not all zeros."""
    if "vwap" not in df.columns:
        return

    vwap_all_zero = (df["vwap"] == 0).all()
    checks["vwap_not_all_zero"] = not vwap_all_zero

    if vwap_all_zero:
        reasons.append("All VWAP values are zero (pre-v7.0 cache data)")
    else:
        vwap_valid = (df["vwap"] >= df["Low"]) & (df["vwap"] <= df["High"])
        checks["vwap_bounded"] = vwap_valid.all()
        if not vwap_valid.all():
            invalid_count = (~vwap_valid).sum()
            reasons.append(f"VWAP outside [Low, High] for {invalid_count} bars")


def _check_bounded_columns(
    df: pd.DataFrame,
    checks: dict[str, bool],
    reasons: list[str],
) -> None:
    """Validate bounded microstructure columns are within expected ranges."""
    # OFI in [-1, 1]
    if "ofi" in df.columns:
        ofi_bounded = df["ofi"].between(-1, 1).all()
        checks["ofi_bounded"] = ofi_bounded
        if not ofi_bounded:
            reasons.append("OFI values outside [-1, 1] range")

    # Turnover imbalance in [-1, 1]
    if "turnover_imbalance" in df.columns:
        ti_bounded = df["turnover_imbalance"].between(-1, 1).all()
        checks["turnover_imbalance_bounded"] = ti_bounded
        if not ti_bounded:
            reasons.append("Turnover imbalance outside [-1, 1] range")

    # Duration non-negative
    if "duration_us" in df.columns:
        duration_valid = (df["duration_us"] >= 0).all()
        checks["duration_non_negative"] = duration_valid
        if not duration_valid:
            reasons.append("Negative duration values detected")

    # Aggregation density >= 1
    if "aggregation_density" in df.columns:
        agg_valid = (df["aggregation_density"] >= 1).all()
        checks["aggregation_density_valid"] = agg_valid
        if not agg_valid:
            reasons.append("Aggregation density < 1 detected")


def _check_volume_consistency(
    df: pd.DataFrame,
    checks: dict[str, bool],
    reasons: list[str],
) -> None:
    """Validate buy_volume + sell_volume == Volume."""
    required_cols = {"buy_volume", "sell_volume", "Volume"}
    if not required_cols.issubset(df.columns):
        return

    vol_sum = df["buy_volume"] + df["sell_volume"]
    vol_diff = (vol_sum - df["Volume"]).abs()
    vol_match = (vol_diff < 1e-6 * df["Volume"].abs().clip(lower=1e-10)).all()
    checks["volume_consistency"] = vol_match
    if not vol_match:
        reasons.append("buy_volume + sell_volume != Volume")


def _check_trade_counts(
    df: pd.DataFrame,
    checks: dict[str, bool],
    reasons: list[str],
) -> None:
    """Validate trade count columns have valid values."""
    if "individual_trade_count" in df.columns:
        counts_valid = (df["individual_trade_count"] >= 1).all()
        checks["trade_counts_valid"] = counts_valid
        if not counts_valid:
            reasons.append("Invalid trade count values (< 1)")


def _check_all_microstructure_zero(
    df: pd.DataFrame,
    checks: dict[str, bool],
    reasons: list[str],
) -> None:
    """Check if all microstructure columns are zero (indicates stale data)."""
    micro_cols_present = [c for c in MICROSTRUCTURE_COLUMNS if c in df.columns]
    if not micro_cols_present:
        return

    all_micro_zero = all((df[col] == 0).all() for col in micro_cols_present)
    checks["microstructure_not_all_zero"] = not all_micro_zero
    if all_micro_zero:
        reasons.append("All microstructure columns are zero (pre-v7.0 cache data)")


def _determine_staleness(
    checks: dict[str, bool],
    reasons: list[str],
) -> StalenessResult:
    """Determine final staleness result from individual checks."""
    high_confidence_checks = [
        "vwap_bounded",
        "vwap_not_all_zero",
        "ofi_bounded",
        "turnover_imbalance_bounded",
        "duration_non_negative",
        "aggregation_density_valid",
        "trade_counts_valid",
        "microstructure_not_all_zero",
    ]

    high_conf_failures = [
        k for k in high_confidence_checks if k in checks and not checks[k]
    ]

    is_stale = len(high_conf_failures) > 0

    # Determine confidence level
    confidence: Literal["high", "medium", "low"] = "high"
    if is_stale and not high_conf_failures:
        confidence = "medium"

    # Build recommendations
    recommendations: list[str] = []
    if is_stale:
        recommendations.append(
            "Invalidate cache entry and recompute with current version"
        )
        if "vwap_not_all_zero" in high_conf_failures:
            recommendations.append("Data appears to be from pre-v7.0")
        if "microstructure_not_all_zero" in high_conf_failures:
            recommendations.append("Data appears to be from pre-v7.0")
        recommendations.append("Run: get_range_bars(..., use_cache=False)")

    return StalenessResult(
        is_stale=is_stale,
        reason="; ".join(reasons) if reasons else None,
        confidence=confidence,
        checks_passed=checks,
        recommendations=recommendations,
    )


def detect_staleness(
    df: pd.DataFrame,
    require_microstructure: bool = True,
) -> StalenessResult:
    """Detect stale cached data using content-based validation.

    This is Tier 0 validation: fast staleness detection (<5ms for 100K bars).
    Run on every cache read before returning data to caller.

    Parameters
    ----------
    df : pd.DataFrame
        Cached range bar DataFrame, possibly with microstructure columns.
    require_microstructure : bool, default=True
        If True, check for valid microstructure columns.

    Returns
    -------
    StalenessResult
        Detection result with confidence level and specific failures.
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if require_microstructure:
        _check_vwap(df, checks, reasons)
        _check_bounded_columns(df, checks, reasons)
        _check_volume_consistency(df, checks, reasons)
        _check_trade_counts(df, checks, reasons)
        _check_all_microstructure_zero(df, checks, reasons)

    return _determine_staleness(checks, reasons)


def validate_schema_version(
    cached_version: str | None,
    min_version: str,
) -> bool:
    """Check if cached data meets minimum schema version requirement.

    Parameters
    ----------
    cached_version : str | None
        Version string from cached data (e.g., "7.0.0").
    min_version : str
        Minimum required version (e.g., "7.0.0").

    Returns
    -------
    bool
        True if cached_version >= min_version.
    """
    if not cached_version:
        return False

    try:
        cached_parts = [int(x) for x in cached_version.split(".")[:_VERSION_PARTS]]
        min_parts = [int(x) for x in min_version.split(".")[:_VERSION_PARTS]]

        # Pad to 3 parts
        while len(cached_parts) < _VERSION_PARTS:
            cached_parts.append(0)
        while len(min_parts) < _VERSION_PARTS:
            min_parts.append(0)

        return tuple(cached_parts) >= tuple(min_parts)
    except (ValueError, AttributeError):
        logger.warning(
            "Invalid version format: cached=%r, min=%r",
            cached_version,
            min_version,
        )
        return False
