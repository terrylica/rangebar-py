"""Tier 2: Statistical validation before production ML (~10 min).

MANDATORY before ML training. Validates:
- Stationarity (ADF test)
- Predictive power (Spearman with forward returns)
- Mutual information with target
- Feature correlation matrix (redundancy check)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .tier1 import FEATURE_COLS, validate_tier1

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Validation thresholds
MIN_SAMPLES_FOR_ADF = 20
MIN_SAMPLES_FOR_SPEARMAN = 100
MIN_SAMPLES_FOR_MI = 50
ADF_SIGNIFICANCE_LEVEL = 0.05
SPEARMAN_MIN_CORRELATION = 0.02
SPEARMAN_SIGNIFICANCE_LEVEL = 0.05
HIGH_CORRELATION_THRESHOLD = 0.8
MIN_SIGNIFICANT_FEATURES = 3


def _run_stationarity_tests(
    data: pd.DataFrame,
    present_cols: list[str],
) -> dict:
    """Run ADF stationarity tests on feature columns."""
    stationarity: dict = {}
    try:
        from statsmodels.tsa.stattools import adfuller

        for col in present_cols:
            series = data[col].dropna()
            if len(series) < MIN_SAMPLES_FOR_ADF:
                continue
            try:
                adf_result = adfuller(series, maxlag=5)
                stationarity[col] = {
                    "adf_statistic": float(adf_result[0]),
                    "adf_p_value": float(adf_result[1]),
                    "is_stationary": adf_result[1] < ADF_SIGNIFICANCE_LEVEL,
                }
            except Exception as e:
                logger.warning("ADF test failed for %s: %s", col, e)
                stationarity[col] = {"error": str(e)}
    except ImportError:
        logger.warning("statsmodels not installed, skipping stationarity tests")
        stationarity = {"error": "statsmodels not installed"}

    return stationarity


def _run_predictive_power_tests(
    data: pd.DataFrame,
    present_cols: list[str],
    target_col: str,
) -> dict:
    """Run Spearman correlation tests for predictive power."""
    predictive: dict = {}
    if target_col not in data.columns:
        return predictive

    try:
        from scipy.stats import spearmanr

        for col in present_cols:
            valid = data[[col, target_col]].dropna()
            if len(valid) > MIN_SAMPLES_FOR_SPEARMAN:
                try:
                    rho, p = spearmanr(valid[col], valid[target_col])
                    is_significant = (
                        abs(rho) > SPEARMAN_MIN_CORRELATION
                        and p < SPEARMAN_SIGNIFICANCE_LEVEL
                        if not np.isnan(rho)
                        else False
                    )
                    predictive[col] = {
                        "spearman_rho": float(rho) if not np.isnan(rho) else None,
                        "p_value": float(p) if not np.isnan(p) else None,
                        "significant": is_significant,
                    }
                except Exception as e:
                    logger.warning("Spearman test failed for %s: %s", col, e)
                    predictive[col] = {"error": str(e)}
    except ImportError:
        logger.warning("scipy not installed, skipping Spearman tests")
        predictive = {"error": "scipy not installed"}

    return predictive


def _run_mutual_information(
    data: pd.DataFrame,
    present_cols: list[str],
    target_col: str,
) -> dict:
    """Compute mutual information scores."""
    mutual_info: dict = {}
    if target_col not in data.columns:
        return mutual_info

    try:
        from sklearn.feature_selection import mutual_info_regression

        feature_data = data[present_cols].dropna()
        target_data = data.loc[feature_data.index, target_col].dropna()
        common_idx = feature_data.index.intersection(target_data.index)

        if len(common_idx) > MIN_SAMPLES_FOR_SPEARMAN:
            feature_data = feature_data.loc[common_idx]
            target_data = target_data.loc[common_idx]
            # Handle any remaining NaN after alignment
            valid_mask = ~(feature_data.isna().any(axis=1) | target_data.isna())
            feature_data = feature_data[valid_mask]
            target_data = target_data[valid_mask]

            if len(feature_data) > MIN_SAMPLES_FOR_MI:
                mi_scores = mutual_info_regression(
                    feature_data, target_data, random_state=42
                )
                mutual_info = dict(
                    zip(present_cols, [float(s) for s in mi_scores], strict=False)
                )
    except ImportError:
        logger.warning("sklearn not installed, skipping mutual information")
        mutual_info = {"error": "sklearn not installed"}
    except Exception as e:
        logger.warning("Mutual information failed: %s", e)
        mutual_info = {"error": str(e)}

    return mutual_info


def _find_high_correlation_pairs(
    data: pd.DataFrame,
    present_cols: list[str],
) -> list:
    """Find highly correlated feature pairs."""
    high_corr_pairs: list = []
    try:
        corr_matrix = data[present_cols].corr(method="spearman")
        for i, col1 in enumerate(present_cols):
            for col2 in present_cols[i + 1 :]:
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[col1, col2]
                    if (
                        not np.isnan(corr_val)
                        and abs(corr_val) > HIGH_CORRELATION_THRESHOLD
                    ):
                        high_corr_pairs.append((col1, col2, float(corr_val)))
    except Exception as e:
        logger.warning("Correlation matrix failed: %s", e)

    return high_corr_pairs


def validate_tier2(
    df: pd.DataFrame,
    target_col: str = "forward_return",
) -> dict:
    """Statistical validation (~10 min). MANDATORY before ML training.

    Extends Tier 1 with statistical tests:
    - Stationarity (Augmented Dickey-Fuller test)
    - Predictive power (Spearman correlation with forward returns)
    - Mutual information scores
    - Feature correlation matrix (redundancy detection)

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with microstructure columns.
        Should include forward_return column or Close for computing returns.
    target_col : str, default="forward_return"
        Column name for the prediction target.
        If not present, will be computed from Close if available.

    Returns
    -------
    dict
        Validation results including:
        - All Tier 1 results
        - stationarity: dict[str, dict] - ADF results per feature
        - predictive_power: dict[str, dict] - Spearman correlation results
        - mutual_info: dict[str, float] - MI scores (if sklearn available)
        - high_correlation_pairs: list[tuple] - Highly correlated feature pairs
        - tier2_passed: bool - Tier 2 validation passed

    Examples
    --------
    >>> from rangebar import get_range_bars
    >>> from rangebar.validation.tier2 import validate_tier2
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-07")
    >>> df["forward_return"] = df["Close"].shift(-1) / df["Close"] - 1
    >>> result = validate_tier2(df)
    >>> print("Tier 2:", "PASSED" if result["tier2_passed"] else "FAILED")
    """
    # Start with Tier 1 results
    results = validate_tier1(df)

    if not results.get("tier1_passed", False):
        results["tier2_passed"] = False
        return results

    # Check which columns are present
    present_cols = [c for c in FEATURE_COLS if c in df.columns]

    if not present_cols:
        results["tier2_passed"] = False
        return results

    # Compute forward return if not present
    working_df = df
    if target_col not in df.columns and "Close" in df.columns:
        working_df = df.copy()
        working_df[target_col] = working_df["Close"].shift(-1) / working_df["Close"] - 1

    # 1. Stationarity tests (ADF)
    results["stationarity"] = _run_stationarity_tests(working_df, present_cols)

    # 2. Predictive power (Spearman with forward returns)
    predictive = _run_predictive_power_tests(working_df, present_cols, target_col)
    results["predictive_power"] = predictive

    # 3. Mutual information
    results["mutual_info"] = _run_mutual_information(
        working_df, present_cols, target_col
    )

    # 4. Correlation matrix (check redundancy)
    results["high_correlation_pairs"] = _find_high_correlation_pairs(
        working_df, present_cols
    )

    # 5. Tier 2 pass criteria
    # At least 3 features should have significant predictive power
    significant_features = sum(
        1
        for v in predictive.values()
        if isinstance(v, dict) and v.get("significant", False)
    )
    results["significant_feature_count"] = significant_features

    results["tier2_passed"] = (
        results.get("tier1_passed", False)
        and significant_features >= MIN_SIGNIFICANT_FEATURES
    )

    return results
