"""Tier 1: Auto-validation suite for microstructure features (<30 sec).

Run on every precompute to catch data quality issues early.
This is the smoke test - fast checks that should always pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Microstructure feature columns (Issue #25)
FEATURE_COLS = [
    "duration_us",
    "ofi",
    "vwap_close_deviation",
    "price_impact",
    "kyle_lambda_proxy",
    "trade_intensity",
    "volume_per_trade",
    "aggression_ratio",
    "aggregation_density",
    "turnover_imbalance",
]

# Validation thresholds
MIN_SAMPLES_FOR_CORRELATION = 50
OFI_CORR_MIN = 0.05
OFI_CORR_MAX = 0.8
OFI_MEAN_THRESHOLD = 0.3
AGGRESSION_RATIO_MAX_MEDIAN = 10


def _check_bounds(df: pd.DataFrame, results: dict) -> None:
    """Check feature bounds and populate results dict."""
    # Duration should be non-negative
    if "duration_us" in df.columns:
        results["duration_positive"] = (df["duration_us"] >= 0).all()
    else:
        results["duration_positive"] = True  # N/A

    # OFI should be bounded [-1, 1]
    if "ofi" in df.columns:
        results["ofi_bounded"] = df["ofi"].between(-1, 1).all()
    else:
        results["ofi_bounded"] = True  # N/A

    # Turnover imbalance should be bounded [-1, 1]
    if "turnover_imbalance" in df.columns:
        results["turnover_imbalance_bounded"] = (
            df["turnover_imbalance"].between(-1, 1).all()
        )
    else:
        results["turnover_imbalance_bounded"] = True  # N/A


def _check_correlation(df: pd.DataFrame, results: dict) -> None:
    """Check OFI-return correlation sanity."""
    if (
        "Close" in df.columns
        and "ofi" in df.columns
        and len(df) > MIN_SAMPLES_FOR_CORRELATION
    ):
        returns = df["Close"].pct_change()
        ofi_return_corr = df["ofi"].corr(returns)
        results["ofi_return_corr"] = (
            float(ofi_return_corr) if not np.isnan(ofi_return_corr) else None
        )
        # OFI should have some correlation with returns
        if results["ofi_return_corr"] is not None:
            corr_abs = abs(results["ofi_return_corr"])
            results["ofi_corr_sane"] = OFI_CORR_MIN < corr_abs < OFI_CORR_MAX
        else:
            results["ofi_corr_sane"] = None
    else:
        results["ofi_return_corr"] = None
        results["ofi_corr_sane"] = None


def _check_distributions(df: pd.DataFrame, results: dict) -> None:
    """Check basic distribution properties."""
    if "ofi" in df.columns:
        results["ofi_mean_near_zero"] = abs(df["ofi"].mean()) < OFI_MEAN_THRESHOLD
    else:
        results["ofi_mean_near_zero"] = True  # N/A

    if "aggression_ratio" in df.columns:
        results["aggression_ratio_reasonable"] = (
            df["aggression_ratio"].median() < AGGRESSION_RATIO_MAX_MEDIAN
        )
    else:
        results["aggression_ratio_reasonable"] = True  # N/A


def validate_tier1(df: pd.DataFrame) -> dict:
    """Auto-validation suite (<30 sec). Run on every precompute.

    Validates basic data quality for microstructure features:
    - No NaN/Inf values
    - Bounded features within expected ranges
    - Basic correlation sanity with returns (if available)

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with microstructure columns

    Returns
    -------
    dict
        Validation results with individual check results and overall pass/fail.
        Keys include:
        - no_nan: bool - No NaN values in feature columns
        - no_inf: bool - No Inf values in feature columns
        - duration_positive: bool - All durations >= 0
        - ofi_bounded: bool - OFI in [-1, 1]
        - turnover_imbalance_bounded: bool - Turnover imbalance in [-1, 1]
        - ofi_return_corr: float | None - OFI-return correlation if computable
        - ofi_corr_sane: bool | None - OFI correlation within expected range
        - ofi_mean_near_zero: bool - OFI mean close to 0 (balanced market)
        - aggression_ratio_reasonable: bool - Median aggression ratio < 10
        - tier1_passed: bool - All critical checks passed

    Examples
    --------
    >>> from rangebar import get_range_bars
    >>> from rangebar.validation.tier1 import validate_tier1
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-02")
    >>> result = validate_tier1(df)
    >>> print("Tier 1:", "PASSED" if result["tier1_passed"] else "FAILED")
    """
    results: dict = {}

    # Check which columns are present
    present_cols = [c for c in FEATURE_COLS if c in df.columns]

    if not present_cols:
        results["features_present"] = False
        results["tier1_passed"] = False
        results["error"] = "No microstructure feature columns found"
        return results

    results["features_present"] = True
    results["features_found"] = present_cols

    # 1. NaN/Inf checks
    results["no_nan"] = not df[present_cols].isna().any().any()
    results["no_inf"] = (
        not np.isinf(df[present_cols].select_dtypes(include=[np.number])).any().any()
    )

    # 2. Bounds checks
    _check_bounds(df, results)

    # 3. Correlation sanity
    _check_correlation(df, results)

    # 4. Distribution checks
    _check_distributions(df, results)

    # 5. Overall pass/fail
    critical_checks = [
        results["no_nan"],
        results["no_inf"],
        results["duration_positive"],
        results["ofi_bounded"],
        results["turnover_imbalance_bounded"],
    ]
    results["tier1_passed"] = all(critical_checks)

    return results
