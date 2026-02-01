#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Adversarial audit of 3-bar pattern OOD robustness.

Multi-perspective forensic audit of the 8 universal 3-bar patterns and
24 RV + 3-bar patterns. Tests for parameter sensitivity, OOS validation,
and bootstrap confidence intervals.

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


def add_3bar_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 3-bar pattern column."""
    return df.with_columns(
        (
            pl.col("direction") +
            pl.col("direction").shift(-1) +
            pl.col("direction").shift(-2)
        ).alias("pattern_3bar")
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


def create_rv_3bar_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Create RV + 3-bar combined pattern column."""
    return df.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("pattern_3bar"))
        .alias("rv_3bar_pattern")
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
# Audit 1: Parameter Sensitivity for 3-bar patterns
# =============================================================================


def audit_parameter_sensitivity_3bar() -> dict:
    """Test if 3-bar patterns are robust across parameter variations."""
    log_info("Starting parameter sensitivity audit for 3-bar patterns")

    all_results = {}

    for params in PARAM_VARIATIONS:
        log_info("Testing parameters", **params)

        universal_3bar: set[str] = set()
        universal_rv_3bar: set[str] = set()
        first_symbol = True

        for symbol in SYMBOLS:
            start = SYMBOL_START_DATES.get(symbol, START_DATE)

            # Load data
            df = load_range_bars_polars(symbol, 100, start, END_DATE)

            # Process
            df = add_bar_direction(df)
            df = add_3bar_patterns(df)
            df = compute_rv_and_regime(
                df,
                params["rv_window"],
                params["low_pct"],
                params["high_pct"],
            )
            df = df.with_columns(
                (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
            ).collect()

            # Create RV + 3-bar pattern
            df = create_rv_3bar_pattern(df)

            # Test robustness for both pattern types
            robust_3bar = test_ood_robustness(df, "pattern_3bar", "fwd_return")
            robust_rv_3bar = test_ood_robustness(df, "rv_3bar_pattern", "fwd_return")

            if first_symbol:
                universal_3bar = set(robust_3bar)
                universal_rv_3bar = set(robust_rv_3bar)
                first_symbol = False
            else:
                universal_3bar &= set(robust_3bar)
                universal_rv_3bar &= set(robust_rv_3bar)

        all_results[params["name"]] = {
            "params": params,
            "universal_3bar_count": len(universal_3bar),
            "universal_rv_3bar_count": len(universal_rv_3bar),
            "universal_3bar": sorted(universal_3bar),
            "universal_rv_3bar": sorted(universal_rv_3bar),
        }

        log_info(
            "Parameter set complete",
            name=params["name"],
            universal_3bar=len(universal_3bar),
            universal_rv_3bar=len(universal_rv_3bar),
        )

    # Find patterns robust across ALL parameter sets
    robust_3bar_all = None
    robust_rv_3bar_all = None

    for _param_name, result in all_results.items():
        patterns_3bar = set(result["universal_3bar"])
        patterns_rv_3bar = set(result["universal_rv_3bar"])

        if robust_3bar_all is None:
            robust_3bar_all = patterns_3bar
            robust_rv_3bar_all = patterns_rv_3bar
        else:
            robust_3bar_all &= patterns_3bar
            robust_rv_3bar_all &= patterns_rv_3bar

    return {
        "parameter_results": all_results,
        "robust_3bar_across_all": sorted(robust_3bar_all) if robust_3bar_all else [],
        "robust_rv_3bar_across_all": sorted(robust_rv_3bar_all) if robust_rv_3bar_all else [],
    }


# =============================================================================
# Audit 2: Out-of-Sample Validation
# =============================================================================


def audit_oos_validation_3bar() -> dict:
    """Test if 3-bar patterns discovered in training period hold in test period."""
    log_info("Starting OOS validation audit for 3-bar patterns")

    train_end = "2024-12-31"
    test_start = "2025-01-01"

    # Get patterns from training period
    train_3bar: set[str] = set()
    train_rv_3bar: set[str] = set()
    first_symbol = True

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)

        df = load_range_bars_polars(symbol, 100, start, train_end)
        df = add_bar_direction(df)
        df = add_3bar_patterns(df)
        df = compute_rv_and_regime(df, 20, 0.25, 0.75)
        df = df.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df = create_rv_3bar_pattern(df)

        robust_3bar = test_ood_robustness(df, "pattern_3bar", "fwd_return")
        robust_rv_3bar = test_ood_robustness(df, "rv_3bar_pattern", "fwd_return")

        if first_symbol:
            train_3bar = set(robust_3bar)
            train_rv_3bar = set(robust_rv_3bar)
            first_symbol = False
        else:
            train_3bar &= set(robust_3bar)
            train_rv_3bar &= set(robust_rv_3bar)

    log_info("Training patterns found", train_3bar=len(train_3bar), train_rv_3bar=len(train_rv_3bar))

    # Test patterns in test period
    test_3bar: set[str] = set()
    test_rv_3bar: set[str] = set()
    first_symbol = True

    for symbol in SYMBOLS:
        df = load_range_bars_polars(symbol, 100, test_start, END_DATE)
        df = add_bar_direction(df)
        df = add_3bar_patterns(df)
        df = compute_rv_and_regime(df, 20, 0.25, 0.75)
        df = df.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df = create_rv_3bar_pattern(df)

        robust_3bar = test_ood_robustness(df, "pattern_3bar", "fwd_return")
        robust_rv_3bar = test_ood_robustness(df, "rv_3bar_pattern", "fwd_return")

        if first_symbol:
            test_3bar = set(robust_3bar)
            test_rv_3bar = set(robust_rv_3bar)
            first_symbol = False
        else:
            test_3bar &= set(robust_3bar)
            test_rv_3bar &= set(robust_rv_3bar)

    log_info("Test patterns found", test_3bar=len(test_3bar), test_rv_3bar=len(test_rv_3bar))

    # Calculate overlap
    overlap_3bar = train_3bar & test_3bar
    overlap_rv_3bar = train_rv_3bar & test_rv_3bar

    retention_3bar = len(overlap_3bar) / len(train_3bar) * 100 if train_3bar else 0
    retention_rv_3bar = len(overlap_rv_3bar) / len(train_rv_3bar) * 100 if train_rv_3bar else 0

    return {
        "train_3bar": sorted(train_3bar),
        "test_3bar": sorted(test_3bar),
        "overlap_3bar": sorted(overlap_3bar),
        "retention_3bar": round(retention_3bar, 1),
        "train_rv_3bar": sorted(train_rv_3bar),
        "test_rv_3bar": sorted(test_rv_3bar),
        "overlap_rv_3bar": sorted(overlap_rv_3bar),
        "retention_rv_3bar": round(retention_rv_3bar, 1),
    }


# =============================================================================
# Audit 3: Bootstrap Confidence Intervals
# =============================================================================


def audit_bootstrap_ci_3bar(n_bootstrap: int = 500) -> dict:
    """Compute bootstrap confidence intervals for 3-bar patterns."""
    log_info("Starting bootstrap CI audit for 3-bar patterns", n_bootstrap=n_bootstrap)

    # Collect all returns by pattern across all symbols
    pattern_3bar_returns: dict[str, list[float]] = {}
    pattern_rv_3bar_returns: dict[str, list[float]] = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)

        df = load_range_bars_polars(symbol, 100, start, END_DATE)
        df = add_bar_direction(df)
        df = add_3bar_patterns(df)
        df = compute_rv_and_regime(df, 20, 0.25, 0.75)
        df = df.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()
        df = create_rv_3bar_pattern(df)

        # Collect 3-bar pattern returns
        patterns_3bar = df.select("pattern_3bar").unique().to_series().drop_nulls().to_list()
        for pattern in patterns_3bar:
            if pattern is None:
                continue
            returns = df.filter(pl.col("pattern_3bar") == pattern).select("fwd_return").drop_nulls().to_series().to_list()
            if pattern not in pattern_3bar_returns:
                pattern_3bar_returns[pattern] = []
            pattern_3bar_returns[pattern].extend(returns)

        # Collect RV + 3-bar pattern returns
        patterns_rv_3bar = df.select("rv_3bar_pattern").unique().to_series().drop_nulls().to_list()
        for pattern in patterns_rv_3bar:
            if pattern is None:
                continue
            returns = df.filter(pl.col("rv_3bar_pattern") == pattern).select("fwd_return").drop_nulls().to_series().to_list()
            if pattern not in pattern_rv_3bar_returns:
                pattern_rv_3bar_returns[pattern] = []
            pattern_rv_3bar_returns[pattern].extend(returns)

    log_info("Returns collected", patterns_3bar=len(pattern_3bar_returns), patterns_rv_3bar=len(pattern_rv_3bar_returns))

    def compute_bootstrap(returns_dict: dict[str, list[float]], min_samples: int = 1000) -> dict:
        """Compute bootstrap CI for patterns."""
        results = {}
        rng = np.random.default_rng()

        for pattern, returns in returns_dict.items():
            if len(returns) < min_samples:
                continue

            returns_arr = np.array(returns)
            n = len(returns_arr)

            # Bootstrap
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

    results_3bar = compute_bootstrap(pattern_3bar_returns)
    results_rv_3bar = compute_bootstrap(pattern_rv_3bar_returns)

    return {
        "3bar": results_3bar,
        "rv_3bar": results_rv_3bar,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    log_info("Starting 3-bar pattern adversarial audit")

    # Audit 1: Parameter sensitivity
    param_results = audit_parameter_sensitivity_3bar()
    log_info(
        "Parameter sensitivity complete",
        robust_3bar=len(param_results["robust_3bar_across_all"]),
        robust_rv_3bar=len(param_results["robust_rv_3bar_across_all"]),
    )

    # Audit 2: OOS validation
    oos_results = audit_oos_validation_3bar()
    log_info(
        "OOS validation complete",
        retention_3bar=oos_results["retention_3bar"],
        retention_rv_3bar=oos_results["retention_rv_3bar"],
    )

    # Audit 3: Bootstrap CI
    bootstrap_results = audit_bootstrap_ci_3bar()
    patterns_3bar_ci = [p for p, r in bootstrap_results["3bar"].items() if r["excludes_zero"]]
    patterns_rv_3bar_ci = [p for p, r in bootstrap_results["rv_3bar"].items() if r["excludes_zero"]]
    log_info(
        "Bootstrap CI complete",
        patterns_3bar_ci=len(patterns_3bar_ci),
        patterns_rv_3bar_ci=len(patterns_rv_3bar_ci),
    )

    # Summary
    print("\n" + "=" * 80)
    print("3-BAR PATTERN ADVERSARIAL AUDIT RESULTS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("AUDIT 1: Parameter Sensitivity")
    print("-" * 80)
    print(f"{'Parameter Set':<20} {'3-bar':<15} {'RV+3-bar':<15}")
    print("-" * 80)
    for name, result in param_results["parameter_results"].items():
        print(f"{name:<20} {result['universal_3bar_count']:<15} {result['universal_rv_3bar_count']:<15}")
    print("-" * 80)
    print(f"{'Robust across ALL':<20} {len(param_results['robust_3bar_across_all']):<15} {len(param_results['robust_rv_3bar_across_all']):<15}")

    print("\n" + "-" * 80)
    print("AUDIT 2: Out-of-Sample Validation")
    print("-" * 80)
    print("3-bar patterns:")
    print(f"  Training (2022-2024): {len(oos_results['train_3bar'])} patterns")
    print(f"  Test (2025-2026): {len(oos_results['test_3bar'])} patterns")
    print(f"  Overlap: {len(oos_results['overlap_3bar'])} patterns")
    print(f"  OOS Retention Rate: {oos_results['retention_3bar']}%")
    print("\nRV + 3-bar patterns:")
    print(f"  Training (2022-2024): {len(oos_results['train_rv_3bar'])} patterns")
    print(f"  Test (2025-2026): {len(oos_results['test_rv_3bar'])} patterns")
    print(f"  Overlap: {len(oos_results['overlap_rv_3bar'])} patterns")
    print(f"  OOS Retention Rate: {oos_results['retention_rv_3bar']}%")

    print("\n" + "-" * 80)
    print("AUDIT 3: Bootstrap Confidence Intervals")
    print("-" * 80)
    print(f"3-bar patterns tested: {len(bootstrap_results['3bar'])}")
    print(f"3-bar CI excludes zero: {len(patterns_3bar_ci)}")
    print(f"\nRV+3-bar patterns tested: {len(bootstrap_results['rv_3bar'])}")
    print(f"RV+3-bar CI excludes zero: {len(patterns_rv_3bar_ci)}")

    # Top patterns by mean return
    if bootstrap_results["3bar"]:
        print("\n" + "-" * 80)
        print("TOP 3-BAR PATTERNS BY MEAN RETURN (CI excludes zero)")
        print("-" * 80)
        print(f"{'Pattern':<12} {'N':>12} {'Mean (bps)':>12} {'CI Low':>12} {'CI High':>12}")
        print("-" * 80)
        sorted_3bar = sorted(
            [(p, r) for p, r in bootstrap_results["3bar"].items() if r["excludes_zero"]],
            key=lambda x: x[1]["mean_bps"],
            reverse=True
        )
        for pattern, stats in sorted_3bar:
            print(f"{pattern:<12} {stats['n_samples']:>12,} {stats['mean_bps']:>12.2f} {stats['ci_lower_bps']:>12.2f} {stats['ci_upper_bps']:>12.2f}")

    # Overall verdict
    print("\n" + "-" * 80)
    print("OVERALL VERDICT")
    print("-" * 80)

    param_pass_3bar = len(param_results["robust_3bar_across_all"]) >= 4
    param_pass_rv_3bar = len(param_results["robust_rv_3bar_across_all"]) >= 10
    oos_pass_3bar = oos_results["retention_3bar"] >= 50
    oos_pass_rv_3bar = oos_results["retention_rv_3bar"] >= 50
    ci_pass_3bar = len(patterns_3bar_ci) >= 4
    ci_pass_rv_3bar = len(patterns_rv_3bar_ci) >= 10

    print("3-bar patterns:")
    print(f"  Parameter Sensitivity: {'PASS' if param_pass_3bar else 'FAIL'} ({len(param_results['robust_3bar_across_all'])} patterns)")
    print(f"  OOS Validation: {'PASS' if oos_pass_3bar else 'FAIL'} ({oos_results['retention_3bar']}% retention)")
    print(f"  Bootstrap CI: {'PASS' if ci_pass_3bar else 'FAIL'} ({len(patterns_3bar_ci)} patterns)")
    overall_3bar = param_pass_3bar and oos_pass_3bar and ci_pass_3bar
    print(f"  OVERALL: {'VALIDATED' if overall_3bar else 'NEEDS REVIEW'}")

    print("\nRV + 3-bar patterns:")
    print(f"  Parameter Sensitivity: {'PASS' if param_pass_rv_3bar else 'FAIL'} ({len(param_results['robust_rv_3bar_across_all'])} patterns)")
    print(f"  OOS Validation: {'PASS' if oos_pass_rv_3bar else 'FAIL'} ({oos_results['retention_rv_3bar']}% retention)")
    print(f"  Bootstrap CI: {'PASS' if ci_pass_rv_3bar else 'FAIL'} ({len(patterns_rv_3bar_ci)} patterns)")
    overall_rv_3bar = param_pass_rv_3bar and oos_pass_rv_3bar and ci_pass_rv_3bar
    print(f"  OVERALL: {'VALIDATED' if overall_rv_3bar else 'NEEDS REVIEW'}")

    log_info(
        "Audit complete",
        overall_3bar=overall_3bar,
        overall_rv_3bar=overall_rv_3bar,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
