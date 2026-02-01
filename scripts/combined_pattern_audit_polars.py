#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Adversarial audit of combined RV regime + multi-threshold alignment patterns.

Multi-perspective forensic audit of the 29 universal combined patterns.
Tests for parameter sensitivity, OOS validation, and bootstrap confidence intervals.

GitHub Issues: #54 (RV regime), #55 (multi-threshold)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import polars as pl

# Transaction cost (high VIP tier)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001 / 100

# Symbols and date range
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2026-01-31"

SYMBOL_START_DATES = {
    "BTCUSDT": "2022-01-01",
    "ETHUSDT": "2022-01-01",
    "SOLUSDT": "2023-06-01",
    "BNBUSDT": "2022-01-01",
}

# Parameter variations for sensitivity testing
PARAM_VARIATIONS = [
    {"name": "baseline", "rv_window": 20, "low_pct": 0.25, "high_pct": 0.75},
    {"name": "shorter_rv", "rv_window": 10, "low_pct": 0.25, "high_pct": 0.75},
    {"name": "longer_rv", "rv_window": 30, "low_pct": 0.25, "high_pct": 0.75},
    {"name": "tighter_pct", "rv_window": 20, "low_pct": 0.33, "high_pct": 0.67},
    {"name": "wider_pct", "rv_window": 20, "low_pct": 0.20, "high_pct": 0.80},
]


# =============================================================================
# NDJSON Logging
# =============================================================================


def log_json(level: str, message: str, **kwargs: object) -> None:
    """Log a message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def log_info(message: str, **kwargs: object) -> None:
    """Log INFO level message."""
    log_json("INFO", message, **kwargs)


# =============================================================================
# Data Loading and Processing
# =============================================================================


def load_range_bars_polars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    """Load range bars from cache as Polars LazyFrame."""
    from rangebar import get_range_bars

    pdf = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    pdf = pdf.reset_index()
    pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})

    return pl.from_pandas(pdf).lazy()


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


def compute_rv_and_regime(
    df: pl.LazyFrame,
    rv_window: int,
    low_pct: float,
    high_pct: float,
) -> pl.LazyFrame:
    """Compute RV and classify regime."""
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=rv_window, min_samples=rv_window).alias("rv")
    ).drop("_log_ret")

    df = df.with_columns(
        pl.col("rv").rolling_quantile(low_pct, window_size=100, min_samples=50).alias("rv_p_low"),
        pl.col("rv").rolling_quantile(high_pct, window_size=100, min_samples=50).alias("rv_p_high"),
    )

    df = df.with_columns(
        pl.when(pl.col("rv") < pl.col("rv_p_low"))
        .then(pl.lit("quiet"))
        .when(pl.col("rv") > pl.col("rv_p_high"))
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("rv_regime")
    )

    return df.drop(["rv_p_low", "rv_p_high"])


def align_thresholds(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars across thresholds."""
    df_50 = df_50.select([
        pl.col("timestamp").alias("timestamp_50"),
        pl.col("direction").alias("direction_50"),
    ])

    df_200 = df_200.select([
        pl.col("timestamp").alias("timestamp_200"),
        pl.col("direction").alias("direction_200"),
    ])

    df_100 = df_100.sort("timestamp")
    df_50 = df_50.sort("timestamp_50")
    df_200 = df_200.sort("timestamp_200")

    aligned = df_100.join_asof(
        df_50,
        left_on="timestamp",
        right_on="timestamp_50",
        strategy="backward",
    )

    aligned = aligned.join_asof(
        df_200,
        left_on="timestamp",
        right_on="timestamp_200",
        strategy="backward",
    )

    return aligned


def classify_alignment(df: pl.DataFrame) -> pl.DataFrame:
    """Classify multi-threshold alignment."""
    df = df.with_columns(
        pl.when(
            (pl.col("direction") == "U") &
            (pl.col("direction_50") == "U") &
            (pl.col("direction_200") == "U")
        ).then(pl.lit("aligned_up"))
        .when(
            (pl.col("direction") == "D") &
            (pl.col("direction_50") == "D") &
            (pl.col("direction_200") == "D")
        ).then(pl.lit("aligned_down"))
        .when(
            ((pl.col("direction") == "U").cast(pl.Int32) +
             (pl.col("direction_50") == "U").cast(pl.Int32) +
             (pl.col("direction_200") == "U").cast(pl.Int32)) >= 2
        ).then(pl.lit("partial_up"))
        .when(
            ((pl.col("direction") == "D").cast(pl.Int32) +
             (pl.col("direction_50") == "D").cast(pl.Int32) +
             (pl.col("direction_200") == "D").cast(pl.Int32)) >= 2
        ).then(pl.lit("partial_down"))
        .otherwise(pl.lit("mixed"))
        .alias("alignment")
    )

    return df


def create_combined_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Create combined pattern column."""
    return df.with_columns(
        (pl.col("rv_regime") + "_" + pl.col("alignment") + "|" + pl.col("pattern_2bar"))
        .alias("combined_pattern")
    )


# =============================================================================
# OOD Robustness Testing
# =============================================================================


def compute_t_stat(returns: pl.Series) -> float:
    """Compute t-statistic."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = returns.mean()
    std = returns.std()

    if std is None or std == 0 or mean is None:
        return 0.0

    return float(mean / (std / math.sqrt(n)))


def test_ood_robustness(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> list[str]:
    """Test OOD robustness and return robust patterns."""
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()
    robust_patterns = []

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_df = df.filter(pl.col(pattern_col) == pattern)
        periods = pattern_df.select("period").unique().to_series().drop_nulls().to_list()

        period_stats = []
        for period_label in periods:
            period_data = pattern_df.filter(pl.col("period") == period_label)
            returns = period_data.select(return_col).drop_nulls().to_series()

            if len(returns) < min_samples:
                continue

            t_stat = compute_t_stat(returns)
            period_stats.append(t_stat)

        if len(period_stats) < 2:
            continue

        all_significant = all(abs(t) >= min_t_stat for t in period_stats)
        same_sign = all(t > 0 for t in period_stats) or all(t < 0 for t in period_stats)

        if all_significant and same_sign:
            robust_patterns.append(pattern)

    return robust_patterns


# =============================================================================
# Audit 1: Parameter Sensitivity
# =============================================================================


def audit_parameter_sensitivity() -> dict:
    """Test if patterns are robust across parameter variations."""
    log_info("Starting parameter sensitivity audit")

    all_results = {}

    for params in PARAM_VARIATIONS:
        log_info("Testing parameters", **params)

        universal_patterns: set[str] = set()
        first_symbol = True

        for symbol in SYMBOLS:
            start = SYMBOL_START_DATES.get(symbol, START_DATE)

            # Load data
            df_50 = load_range_bars_polars(symbol, 50, start, END_DATE)
            df_100 = load_range_bars_polars(symbol, 100, start, END_DATE)
            df_200 = load_range_bars_polars(symbol, 200, start, END_DATE)

            # Process
            df_50 = add_bar_direction(df_50).collect()
            df_100 = add_bar_direction(df_100)
            df_100 = add_patterns(df_100)
            df_100 = compute_rv_and_regime(
                df_100,
                params["rv_window"],
                params["low_pct"],
                params["high_pct"],
            )
            df_100 = df_100.with_columns(
                (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
            ).collect()
            df_200 = add_bar_direction(df_200).collect()

            # Align and classify
            aligned = align_thresholds(df_50, df_100, df_200)
            aligned = classify_alignment(aligned)
            aligned = create_combined_pattern(aligned)

            # Test robustness
            robust = test_ood_robustness(aligned, "combined_pattern", "fwd_return")

            if first_symbol:
                universal_patterns = set(robust)
                first_symbol = False
            else:
                universal_patterns &= set(robust)

        all_results[params["name"]] = {
            "params": params,
            "universal_count": len(universal_patterns),
            "universal_patterns": sorted(universal_patterns),
        }

        log_info(
            "Parameter set complete",
            name=params["name"],
            universal_count=len(universal_patterns),
        )

    # Find patterns robust across ALL parameter sets
    robust_across_all = None
    for _param_name, result in all_results.items():
        patterns = set(result["universal_patterns"])
        if robust_across_all is None:
            robust_across_all = patterns
        else:
            robust_across_all &= patterns

    return {
        "parameter_results": all_results,
        "robust_across_all": sorted(robust_across_all) if robust_across_all else [],
    }


# =============================================================================
# Audit 2: Out-of-Sample Validation
# =============================================================================


def audit_oos_validation() -> dict:
    """Test if patterns discovered in training period hold in test period."""
    log_info("Starting OOS validation audit")

    train_end = "2024-12-31"
    test_start = "2025-01-01"

    # Get patterns from training period
    train_universal: set[str] = set()
    first_symbol = True

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)

        df_50 = load_range_bars_polars(symbol, 50, start, train_end)
        df_100 = load_range_bars_polars(symbol, 100, start, train_end)
        df_200 = load_range_bars_polars(symbol, 200, start, train_end)

        df_50 = add_bar_direction(df_50).collect()
        df_100 = add_bar_direction(df_100)
        df_100 = add_patterns(df_100)
        df_100 = compute_rv_and_regime(df_100, 20, 0.25, 0.75)
        df_100 = df_100.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df_200 = add_bar_direction(df_200).collect()

        aligned = align_thresholds(df_50, df_100, df_200)
        aligned = classify_alignment(aligned)
        aligned = create_combined_pattern(aligned)

        robust = test_ood_robustness(aligned, "combined_pattern", "fwd_return")

        if first_symbol:
            train_universal = set(robust)
            first_symbol = False
        else:
            train_universal &= set(robust)

    log_info("Training patterns found", count=len(train_universal))

    # Test patterns in test period
    test_universal: set[str] = set()
    first_symbol = True

    for symbol in SYMBOLS:
        df_50 = load_range_bars_polars(symbol, 50, test_start, END_DATE)
        df_100 = load_range_bars_polars(symbol, 100, test_start, END_DATE)
        df_200 = load_range_bars_polars(symbol, 200, test_start, END_DATE)

        df_50 = add_bar_direction(df_50).collect()
        df_100 = add_bar_direction(df_100)
        df_100 = add_patterns(df_100)
        df_100 = compute_rv_and_regime(df_100, 20, 0.25, 0.75)
        df_100 = df_100.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df_200 = add_bar_direction(df_200).collect()

        aligned = align_thresholds(df_50, df_100, df_200)
        aligned = classify_alignment(aligned)
        aligned = create_combined_pattern(aligned)

        robust = test_ood_robustness(aligned, "combined_pattern", "fwd_return")

        if first_symbol:
            test_universal = set(robust)
            first_symbol = False
        else:
            test_universal &= set(robust)

    log_info("Test patterns found", count=len(test_universal))

    # Calculate overlap
    overlap = train_universal & test_universal
    retention_rate = len(overlap) / len(train_universal) * 100 if train_universal else 0

    return {
        "train_patterns": sorted(train_universal),
        "test_patterns": sorted(test_universal),
        "overlap": sorted(overlap),
        "oos_retention_rate": round(retention_rate, 1),
    }


# =============================================================================
# Audit 3: Bootstrap Confidence Intervals
# =============================================================================


def audit_bootstrap_ci(n_bootstrap: int = 500) -> dict:
    """Compute bootstrap confidence intervals for combined patterns."""
    log_info("Starting bootstrap CI audit", n_bootstrap=n_bootstrap)

    # First, collect all returns by pattern across all symbols
    pattern_returns: dict[str, list[float]] = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)

        df_50 = load_range_bars_polars(symbol, 50, start, END_DATE)
        df_100 = load_range_bars_polars(symbol, 100, start, END_DATE)
        df_200 = load_range_bars_polars(symbol, 200, start, END_DATE)

        df_50 = add_bar_direction(df_50).collect()
        df_100 = add_bar_direction(df_100)
        df_100 = add_patterns(df_100)
        df_100 = compute_rv_and_regime(df_100, 20, 0.25, 0.75)
        df_100 = df_100.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df_200 = add_bar_direction(df_200).collect()

        aligned = align_thresholds(df_50, df_100, df_200)
        aligned = classify_alignment(aligned)
        aligned = create_combined_pattern(aligned)

        # Group returns by pattern
        patterns = aligned.select("combined_pattern").unique().to_series().drop_nulls().to_list()

        for pattern in patterns:
            if pattern is None:
                continue

            returns = aligned.filter(pl.col("combined_pattern") == pattern).select("fwd_return").drop_nulls().to_series().to_list()

            if pattern not in pattern_returns:
                pattern_returns[pattern] = []
            pattern_returns[pattern].extend(returns)

    log_info("Returns collected", patterns=len(pattern_returns))

    # Bootstrap CI for patterns with enough samples
    results = {}

    for pattern, returns in pattern_returns.items():
        if len(returns) < 1000:  # Need substantial samples for bootstrap
            continue

        returns_arr = np.array(returns)
        n = len(returns_arr)

        # Bootstrap using Generator API
        rng = np.random.default_rng()
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(returns_arr, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
        mean_return = float(np.mean(returns_arr))

        # Check if CI excludes zero
        excludes_zero = bool((ci_lower > 0) or (ci_upper < 0))

        results[pattern] = {
            "n_samples": n,
            "mean_bps": round(mean_return * 10000, 2),
            "ci_lower_bps": round(ci_lower * 10000, 2),
            "ci_upper_bps": round(ci_upper * 10000, 2),
            "excludes_zero": excludes_zero,
        }

    return results


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    log_info("Starting combined pattern adversarial audit")

    # Audit 1: Parameter sensitivity
    param_results = audit_parameter_sensitivity()
    log_info(
        "Parameter sensitivity complete",
        robust_across_all=len(param_results["robust_across_all"]),
    )

    # Audit 2: OOS validation
    oos_results = audit_oos_validation()
    log_info(
        "OOS validation complete",
        retention_rate=oos_results["oos_retention_rate"],
    )

    # Audit 3: Bootstrap CI
    bootstrap_results = audit_bootstrap_ci()
    patterns_with_ci = [p for p, r in bootstrap_results.items() if r["excludes_zero"]]
    log_info(
        "Bootstrap CI complete",
        patterns_tested=len(bootstrap_results),
        patterns_ci_excludes_zero=len(patterns_with_ci),
    )

    # Summary
    print("\n" + "=" * 80)
    print("COMBINED PATTERN ADVERSARIAL AUDIT RESULTS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("AUDIT 1: Parameter Sensitivity")
    print("-" * 80)
    for name, result in param_results["parameter_results"].items():
        print(f"  {name}: {result['universal_count']} universal patterns")
    print(f"\n  Robust across ALL param sets: {len(param_results['robust_across_all'])}")

    print("\n" + "-" * 80)
    print("AUDIT 2: Out-of-Sample Validation")
    print("-" * 80)
    print(f"  Training (2022-2024): {len(oos_results['train_patterns'])} patterns")
    print(f"  Test (2025-2026): {len(oos_results['test_patterns'])} patterns")
    print(f"  Overlap: {len(oos_results['overlap'])} patterns")
    print(f"  OOS Retention Rate: {oos_results['oos_retention_rate']}%")

    print("\n" + "-" * 80)
    print("AUDIT 3: Bootstrap Confidence Intervals")
    print("-" * 80)
    print(f"  Patterns tested: {len(bootstrap_results)}")
    print(f"  CI excludes zero: {len(patterns_with_ci)}")

    # Overall verdict
    print("\n" + "-" * 80)
    print("OVERALL VERDICT")
    print("-" * 80)

    param_pass = len(param_results["robust_across_all"]) >= 10
    oos_pass = oos_results["oos_retention_rate"] >= 50
    ci_pass = len(patterns_with_ci) >= 10

    print(f"  Parameter Sensitivity: {'PASS' if param_pass else 'FAIL'}")
    print(f"  OOS Validation: {'PASS' if oos_pass else 'FAIL'}")
    print(f"  Bootstrap CI: {'PASS' if ci_pass else 'FAIL'}")
    print(f"\n  OVERALL: {'VALIDATED' if (param_pass and oos_pass and ci_pass) else 'NEEDS REVIEW'}")

    log_info(
        "Audit complete",
        param_pass=param_pass,
        oos_pass=oos_pass,
        ci_pass=ci_pass,
        overall_pass=param_pass and oos_pass and ci_pass,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
