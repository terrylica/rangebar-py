#!/usr/bin/env python3
"""Portable validation for GPU workstation (Linux). Run after re-precompute.

This script validates microstructure features (Issue #25) on any machine
with rangebar-py installed and ClickHouse access. Designed to be run
after re-precomputing all cached bars with the new features.

Usage:
    python scripts/validate_microstructure_features.py

Exit codes:
    0 - All validations passed
    1 - Tier 1 or Tier 2 validation failed
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Feature columns for validation
MICROSTRUCTURE_FEATURE_COLS = [
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


def _print_stationarity_results(stationarity: dict) -> None:
    """Print stationarity test results."""
    if not stationarity or (isinstance(stationarity, dict) and "error" in stationarity):
        return

    print("  Stationarity (ADF p-value < 0.05 = stationary):")
    for col, stats in stationarity.items():
        if isinstance(stats, dict) and "is_stationary" in stats:
            status = "stationary" if stats["is_stationary"] else "non-stationary"
            p_val = stats.get("adf_p_value", "N/A")
            print(f"    {col}: p={p_val:.4f} ({status})")


def _print_predictive_power_results(predictive: dict) -> None:
    """Print predictive power test results."""
    if not predictive or (isinstance(predictive, dict) and "error" in predictive):
        return

    print()
    print("  Predictive Power (Spearman with forward_return):")
    for col, stats in predictive.items():
        if isinstance(stats, dict) and "spearman_rho" in stats:
            sig = "*" if stats.get("significant", False) else ""
            rho = stats.get("spearman_rho", "N/A")
            p_val = stats.get("p_value", "N/A")
            if isinstance(rho, float) and isinstance(p_val, float):
                print(f"    {col}: rho={rho:.4f}, p={p_val:.4f} {sig}")


def _print_high_correlation_pairs(high_corr: list) -> None:
    """Print high correlation pairs."""
    if not high_corr:
        return

    print()
    print("  High Correlation Pairs (|rho| > 0.8):")
    for col1, col2, corr in high_corr:
        print(f"    {col1} <-> {col2}: {corr:.4f}")


def _print_header() -> None:
    """Print script header."""
    print("=" * 60)
    print("Microstructure Feature Validation (Issue #25)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print()


def _print_tier1_results(t1_result: dict) -> bool:
    """Print Tier 1 results and return pass status."""
    print()
    print("--- Tier 1 Validation (Smoke Test) ---")

    exclude_keys = {"tier1_passed", "features_found"}
    for key, value in t1_result.items():
        if key not in exclude_keys:
            print(f"  {key}: {value}")

    t1_passed = t1_result.get("tier1_passed", False)
    print(f"Result: {'PASSED' if t1_passed else 'FAILED'}")

    if not t1_passed:
        print("ERROR: Tier 1 validation failed. Fix data quality issues first.")

    return t1_passed


def _print_tier2_results(t2_result: dict) -> bool:
    """Print Tier 2 results and return pass status."""
    print()
    print("--- Tier 2 Validation (Statistical) ---")

    _print_stationarity_results(t2_result.get("stationarity", {}))
    _print_predictive_power_results(t2_result.get("predictive_power", {}))
    _print_high_correlation_pairs(t2_result.get("high_correlation_pairs", []))

    significant_count = t2_result.get("significant_feature_count", 0)
    t2_passed = t2_result.get("tier2_passed", False)
    print()
    print(f"Significant features: {significant_count} (need >= 3)")
    print(f"Result: {'PASSED' if t2_passed else 'FAILED'}")

    return t2_passed


def _print_feature_statistics(bars_df: pd.DataFrame) -> None:
    """Print feature statistics summary."""
    print()
    print("--- Feature Statistics ---")
    available_cols = [c for c in MICROSTRUCTURE_FEATURE_COLS if c in bars_df.columns]

    if available_cols:
        stats = bars_df[available_cols].describe().T[["mean", "std", "min", "max"]]
        print(stats.to_string())


def _print_overall_result(passed: bool) -> None:
    """Print overall result."""
    print()
    print("=" * 60)
    print(f"OVERALL: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)


def _fetch_bars(
    symbol: str, start: str, end: str, threshold: int
) -> pd.DataFrame | None:
    """Fetch range bars with microstructure features."""
    try:
        from rangebar import get_range_bars

        bars_df = get_range_bars(
            symbol,
            start,
            end,
            threshold_decimal_bps=threshold,
            include_microstructure=True,
        )
    except ImportError as e:
        print(f"ERROR: Could not import rangebar: {e}")
        print("Make sure rangebar-py is installed: pip install rangebar-py")
        return None
    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"ERROR: Failed to fetch bars: {e}")
        return None
    else:
        print(f"Loaded {len(bars_df)} bars")
        print(f"Columns: {list(bars_df.columns)}")
        return bars_df


def main() -> bool:
    """Run microstructure feature validation."""
    _print_header()

    # Configuration
    symbol = "BTCUSDT"
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    threshold_decimal_bps = 250

    print(f"Symbol: {symbol}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Threshold: {threshold_decimal_bps} (25bps = 0.25%)")
    print()

    # Fetch bars
    print("--- Fetching Range Bars ---")
    bars_df = _fetch_bars(symbol, start_date, end_date, threshold_decimal_bps)

    if bars_df is None or bars_df.empty:
        print("WARNING: No bars returned. Check if data is available.")
        return False

    # Import validation modules
    try:
        from rangebar.validation.tier1 import validate_tier1
        from rangebar.validation.tier2 import validate_tier2
    except ImportError as e:
        print(f"ERROR: Could not import validation modules: {e}")
        return False

    # Compute forward return for Tier 2
    bars_df["forward_return"] = bars_df["Close"].shift(-1) / bars_df["Close"] - 1

    # Run validations
    t1_passed = _print_tier1_results(validate_tier1(bars_df))
    if not t1_passed:
        return False

    t2_passed = _print_tier2_results(
        validate_tier2(bars_df, target_col="forward_return")
    )

    # Summary
    _print_feature_statistics(bars_df)
    overall = t1_passed and t2_passed
    _print_overall_result(overall)

    return overall


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
