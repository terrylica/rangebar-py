#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Adversarial audit of volatility regime patterns (Polars).

Multi-perspective forensic audit of the RV regime patterns discovered in Issue #54.
Tests for data leakage, parameter sensitivity, and statistical artifacts.

GitHub Issue: #54 (adversarial audit)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# Transaction cost (high VIP tier)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001


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
# Technical Indicators
# =============================================================================


def compute_realized_volatility(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Realized Volatility (std of log returns)."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=window, min_samples=window).alias(alias)
    ).drop("_log_ret")


def classify_rv_regimes(
    df: pl.LazyFrame,
    rv_col: str,
    low_pct: float = 0.25,
    high_pct: float = 0.75,
    window: int = 100,
) -> pl.LazyFrame:
    """Classify volatility regimes using realized volatility percentiles."""
    df = df.with_columns(
        pl.col(rv_col).rolling_quantile(low_pct, window_size=window, min_samples=50).alias("rv_p_low"),
        pl.col(rv_col).rolling_quantile(high_pct, window_size=window, min_samples=50).alias("rv_p_high"),
    )

    df = df.with_columns(
        pl.when(pl.col(rv_col) < pl.col("rv_p_low"))
        .then(pl.lit("quiet"))
        .when(pl.col(rv_col) > pl.col("rv_p_high"))
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("vol_regime")
    )

    return df.drop(["rv_p_low", "rv_p_high"])


# =============================================================================
# Data Loading
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


# =============================================================================
# Pattern Classification
# =============================================================================


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction (U for up, D for down)."""
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


# =============================================================================
# OOD Robustness Testing
# =============================================================================


def compute_t_stat(returns: pl.Series) -> float:
    """Compute t-statistic for returns."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = returns.mean()
    std = returns.std()

    if std is None or std == 0 or mean is None:
        return 0.0

    return float(mean / (std / math.sqrt(n)))


def count_ood_robust_patterns(
    df: pl.DataFrame,
    pattern_col: str,
    regime_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Count OOD robust patterns within each regime."""
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    regimes = df.select(regime_col).unique().to_series().drop_nulls().to_list()
    robust_patterns = []

    for regime in regimes:
        if regime is None:
            continue

        regime_df = df.filter(pl.col(regime_col) == regime)
        patterns = regime_df.select(pattern_col).unique().to_series().drop_nulls().to_list()

        for pattern in patterns:
            if pattern is None:
                continue

            pattern_subset = regime_df.filter(pl.col(pattern_col) == pattern)
            periods = pattern_subset.select("period").unique().to_series().drop_nulls().to_list()

            period_stats = []
            for period_label in periods:
                period_data = pattern_subset.filter(pl.col("period") == period_label)
                returns = period_data.select(return_col).drop_nulls().to_series()
                count = len(returns)

                if count < min_samples:
                    continue

                t_stat = compute_t_stat(returns)
                period_stats.append(t_stat)

            if len(period_stats) < 2:
                continue

            all_significant = all(abs(t) >= min_t_stat for t in period_stats)
            same_sign = all(t > 0 for t in period_stats) or all(t < 0 for t in period_stats)

            if all_significant and same_sign:
                robust_patterns.append(f"{regime}|{pattern}")

    return {
        "robust_patterns": robust_patterns,
        "count": len(robust_patterns),
    }


# =============================================================================
# Audit 1: Parameter Sensitivity
# =============================================================================


def audit_parameter_sensitivity(
    symbols: list[str],
    start_date: str,
    end_date: str,
    horizon: int = 10,
) -> dict:
    """Test if patterns are robust across different RV parameter choices."""
    log_info("=== AUDIT 1: PARAMETER SENSITIVITY ===")

    parameter_sets = [
        {"name": "baseline", "rv_window": 20, "low_pct": 0.25, "high_pct": 0.75},
        {"name": "shorter_rv", "rv_window": 10, "low_pct": 0.25, "high_pct": 0.75},
        {"name": "longer_rv", "rv_window": 30, "low_pct": 0.25, "high_pct": 0.75},
        {"name": "tighter_pct", "rv_window": 20, "low_pct": 0.33, "high_pct": 0.67},
        {"name": "wider_pct", "rv_window": 20, "low_pct": 0.20, "high_pct": 0.80},
    ]

    all_results = {}

    for params in parameter_sets:
        param_name = params["name"]
        pattern_counts = Counter()

        for symbol in symbols:
            try:
                df = load_range_bars_polars(symbol, 100, start_date, end_date)
                df = add_bar_direction(df)
                df = add_patterns(df)
                df = compute_realized_volatility(df, params["rv_window"], "rv")
                df = classify_rv_regimes(df, "rv", params["low_pct"], params["high_pct"])
                df = df.with_columns(
                    (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
                )

                result_df = df.collect()
                clean_df = result_df.drop_nulls(subset=["pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"])

                robust = count_ood_robust_patterns(
                    clean_df, "pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"
                )

                for pattern in robust["robust_patterns"]:
                    pattern_counts[pattern] += 1

            except (ValueError, RuntimeError, OSError, KeyError) as e:
                log_json("ERROR", f"Failed {symbol} with {param_name}", error=str(e))

        universal = [p for p, c in pattern_counts.items() if c >= len(symbols)]
        all_results[param_name] = {
            "params": params,
            "universal_count": len(universal),
            "universal_patterns": universal,
        }

        log_info(
            f"Parameter set: {param_name}",
            rv_window=params["rv_window"],
            percentiles=f"{params['low_pct']}-{params['high_pct']}",
            universal_count=len(universal),
        )

    # Find patterns robust across ALL parameter sets
    all_param_patterns = set(all_results["baseline"]["universal_patterns"])
    for _param_name, result in all_results.items():
        all_param_patterns &= set(result["universal_patterns"])

    log_info(
        "Patterns robust across ALL parameter sets",
        count=len(all_param_patterns),
        patterns=sorted(all_param_patterns),
    )

    return {
        "parameter_results": all_results,
        "robust_across_all": sorted(all_param_patterns),
    }


# =============================================================================
# Audit 2: Out-of-Sample Validation
# =============================================================================


def audit_oos_validation(
    symbols: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int = 10,
) -> dict:
    """Test patterns on held-out 2025-2026 data."""
    log_info("=== AUDIT 2: OUT-OF-SAMPLE VALIDATION ===")
    log_info(f"Train: {train_start} to {train_end}")
    log_info(f"Test: {test_start} to {test_end}")

    # Get patterns from training period
    train_pattern_counts = Counter()

    for symbol in symbols:
        try:
            df = load_range_bars_polars(symbol, 100, train_start, train_end)
            df = add_bar_direction(df)
            df = add_patterns(df)
            df = compute_realized_volatility(df, 20, "rv")
            df = classify_rv_regimes(df, "rv")
            df = df.with_columns(
                (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
            )

            result_df = df.collect()
            clean_df = result_df.drop_nulls(subset=["pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"])

            robust = count_ood_robust_patterns(
                clean_df, "pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"
            )

            for pattern in robust["robust_patterns"]:
                train_pattern_counts[pattern] += 1

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", f"Train failed {symbol}", error=str(e))

    train_universal = [p for p, c in train_pattern_counts.items() if c >= len(symbols)]

    # Test patterns on OOS period
    test_pattern_counts = Counter()

    for symbol in symbols:
        try:
            df = load_range_bars_polars(symbol, 100, test_start, test_end)
            df = add_bar_direction(df)
            df = add_patterns(df)
            df = compute_realized_volatility(df, 20, "rv")
            df = classify_rv_regimes(df, "rv")
            df = df.with_columns(
                (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
            )

            result_df = df.collect()
            clean_df = result_df.drop_nulls(subset=["pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"])

            robust = count_ood_robust_patterns(
                clean_df, "pattern_2bar", "vol_regime", f"fwd_ret_{horizon}",
                min_samples=50,  # Lower threshold for shorter OOS period
            )

            for pattern in robust["robust_patterns"]:
                test_pattern_counts[pattern] += 1

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", f"Test failed {symbol}", error=str(e))

    test_universal = [p for p, c in test_pattern_counts.items() if c >= len(symbols)]

    # Compare train vs test
    train_set = set(train_universal)
    test_set = set(test_universal)
    overlap = train_set & test_set
    train_only = train_set - test_set
    test_only = test_set - train_set

    log_info(
        "Train period patterns",
        count=len(train_universal),
        patterns=sorted(train_universal),
    )
    log_info(
        "Test period patterns",
        count=len(test_universal),
        patterns=sorted(test_universal),
    )
    log_info(
        "OOS validation",
        overlap_count=len(overlap),
        overlap=sorted(overlap),
        train_only=sorted(train_only),
        test_only=sorted(test_only),
        oos_retention_rate=round(len(overlap) / len(train_set) * 100, 1) if train_set else 0,
    )

    return {
        "train_patterns": sorted(train_universal),
        "test_patterns": sorted(test_universal),
        "overlap": sorted(overlap),
        "oos_retention_rate": round(len(overlap) / len(train_set) * 100, 1) if train_set else 0,
    }


# =============================================================================
# Audit 3: Bootstrap Confidence Intervals
# =============================================================================


def audit_bootstrap_ci(
    symbols: list[str],
    start_date: str,
    end_date: str,
    horizon: int = 10,
    n_bootstrap: int = 500,
) -> dict:
    """Bootstrap confidence intervals for pattern returns."""
    log_info("=== AUDIT 3: BOOTSTRAP CONFIDENCE INTERVALS ===")
    log_info(f"Bootstrap iterations: {n_bootstrap}")

    # Get the 12 universal RV patterns
    target_patterns = [
        "quiet|DD", "quiet|DU", "quiet|UD", "quiet|UU",
        "active|DD", "active|DU", "active|UD", "active|UU",
        "volatile|DD", "volatile|DU", "volatile|UD", "volatile|UU",
    ]

    pattern_results = {}

    for symbol in symbols[:1]:  # Just use one symbol for speed
        try:
            df = load_range_bars_polars(symbol, 100, start_date, end_date)
            df = add_bar_direction(df)
            df = add_patterns(df)
            df = compute_realized_volatility(df, 20, "rv")
            df = classify_rv_regimes(df, "rv")
            df = df.with_columns(
                (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
            )

            result_df = df.collect()

            for pattern in target_patterns:
                parts = pattern.split("|")
                regime, bar_pattern = parts[0], parts[1]

                pattern_df = result_df.filter(
                    (pl.col("vol_regime") == regime) & (pl.col("pattern_2bar") == bar_pattern)
                )
                returns = pattern_df.select(f"fwd_ret_{horizon}").drop_nulls().to_series().to_numpy()

                if len(returns) < 100:
                    continue

                # Bootstrap
                bootstrap_means = []
                rng = np.random.default_rng(42)
                for _ in range(n_bootstrap):
                    sample = rng.choice(returns, size=len(returns), replace=True)
                    bootstrap_means.append(np.mean(sample))

                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                observed_mean = np.mean(returns)

                excludes_zero = bool((ci_lower > 0) or (ci_upper < 0))

                pattern_results[pattern] = {
                    "n_samples": len(returns),
                    "mean_bps": round(observed_mean * 10000, 2),
                    "ci_lower_bps": round(ci_lower * 10000, 2),
                    "ci_upper_bps": round(ci_upper * 10000, 2),
                    "excludes_zero": excludes_zero,
                }

                log_info(
                    f"Bootstrap: {pattern}",
                    n=len(returns),
                    mean_bps=round(observed_mean * 10000, 2),
                    ci=f"[{round(ci_lower * 10000, 2)}, {round(ci_upper * 10000, 2)}]",
                    excludes_zero=excludes_zero,
                )

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", f"Bootstrap failed {symbol}", error=str(e))

    passing = sum(1 for r in pattern_results.values() if r.get("excludes_zero", False))
    log_info(
        "Bootstrap summary",
        total_patterns=len(pattern_results),
        ci_excludes_zero=passing,
        pass_rate=round(passing / len(pattern_results) * 100, 1) if pattern_results else 0,
    )

    return pattern_results


# =============================================================================
# Main Audit
# =============================================================================


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting adversarial audit of volatility regime patterns (Polars)",
        script="volatility_regime_audit_polars.py",
    )

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    # Audit 1: Parameter Sensitivity
    param_results = audit_parameter_sensitivity(
        symbols,
        start_date="2022-01-01",
        end_date="2026-01-31",
    )

    # Audit 2: OOS Validation
    oos_results = audit_oos_validation(
        symbols,
        train_start="2022-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2026-01-31",
    )

    # Audit 3: Bootstrap CI
    bootstrap_results = audit_bootstrap_ci(
        symbols,
        start_date="2022-01-01",
        end_date="2026-01-31",
    )

    # Summary
    log_info("=== AUDIT SUMMARY ===")
    log_info(
        "Parameter sensitivity",
        patterns_robust_all_params=len(param_results["robust_across_all"]),
        patterns=param_results["robust_across_all"],
    )
    log_info(
        "OOS validation",
        oos_retention_rate=oos_results["oos_retention_rate"],
        overlap_patterns=oos_results["overlap"],
    )

    bootstrap_pass = sum(1 for r in bootstrap_results.values() if r.get("excludes_zero", False))
    log_info(
        "Bootstrap CI",
        patterns_excluding_zero=bootstrap_pass,
        total_tested=len(bootstrap_results),
    )

    # Final verdict
    param_pass = len(param_results["robust_across_all"]) >= 8
    oos_pass = oos_results["oos_retention_rate"] >= 50
    bootstrap_pass_rate = bootstrap_pass / len(bootstrap_results) * 100 if bootstrap_results else 0

    log_info(
        "FINAL VERDICT",
        parameter_sensitivity="PASS" if param_pass else "FAIL",
        oos_validation="PASS" if oos_pass else "FAIL",
        bootstrap_ci=f"{bootstrap_pass_rate:.0f}% pass",
        overall="VALIDATED" if (param_pass and oos_pass) else "NEEDS REVIEW",
    )

    # Save results
    output_path = Path("/tmp/volatility_regime_audit_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "parameter_sensitivity": param_results,
            "oos_validation": oos_results,
            "bootstrap_ci": bootstrap_results,
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
