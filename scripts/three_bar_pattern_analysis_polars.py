#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""3-bar pattern analysis for enhanced signals (Polars).

Extends validated 2-bar patterns to 3-bar formations to test if longer
patterns provide better predictive power.

GitHub Issue: #54, #55
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

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


def add_2bar_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
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


def compute_rv_and_regime(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute RV and classify regime."""
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=20, min_samples=20).alias("rv")
    ).drop("_log_ret")

    df = df.with_columns(
        pl.col("rv").rolling_quantile(0.25, window_size=100, min_samples=50).alias("rv_p25"),
        pl.col("rv").rolling_quantile(0.75, window_size=100, min_samples=50).alias("rv_p75"),
    )

    df = df.with_columns(
        pl.when(pl.col("rv") < pl.col("rv_p25"))
        .then(pl.lit("quiet"))
        .when(pl.col("rv") > pl.col("rv_p75"))
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("rv_regime")
    )

    return df


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
) -> dict:
    """Test OOD robustness and return robust patterns with stats."""
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()
    robust_patterns = []
    pattern_stats = []

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_df = df.filter(pl.col(pattern_col) == pattern)
        periods = pattern_df.select("period").unique().to_series().drop_nulls().to_list()

        period_stats = []
        total_samples = 0
        for period_label in periods:
            period_data = pattern_df.filter(pl.col("period") == period_label)
            returns = period_data.select(return_col).drop_nulls().to_series()
            count = len(returns)
            total_samples += count

            if count < min_samples:
                continue

            t_stat = compute_t_stat(returns)
            period_stats.append(t_stat)

        if len(period_stats) < 2:
            continue

        all_significant = all(abs(t) >= min_t_stat for t in period_stats)
        same_sign = all(t > 0 for t in period_stats) or all(t < 0 for t in period_stats)

        all_returns = pattern_df.select(return_col).drop_nulls().to_series()
        mean_return = float(all_returns.mean()) if all_returns.mean() is not None else 0.0
        net_return = mean_return - ROUND_TRIP_COST_PCT
        overall_t = compute_t_stat(all_returns)

        pattern_stats.append({
            "pattern": pattern,
            "n_periods": len(period_stats),
            "n_samples": total_samples,
            "mean_bps": round(mean_return * 10000, 2),
            "net_bps": round(net_return * 10000, 2),
            "overall_t": round(overall_t, 2),
            "is_robust": all_significant and same_sign,
        })

        if all_significant and same_sign:
            robust_patterns.append(pattern)

    return {
        "robust_patterns": robust_patterns,
        "count": len(robust_patterns),
        "pattern_stats": pattern_stats,
    }


# =============================================================================
# Main Analysis
# =============================================================================


def run_pattern_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run 2-bar and 3-bar pattern analysis."""
    log_info("Loading data", symbol=symbol)

    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_2bar_patterns(df)
    df = add_3bar_patterns(df)
    df = compute_rv_and_regime(df)

    # Add forward return
    df = df.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    )

    df_collected = df.collect()
    log_info("Data loaded", symbol=symbol, bars=len(df_collected))

    # Test 2-bar patterns
    results_2bar = test_ood_robustness(df_collected, "pattern_2bar", "fwd_return")

    # Test 3-bar patterns
    results_3bar = test_ood_robustness(df_collected, "pattern_3bar", "fwd_return")

    # Test 3-bar patterns with RV regime
    df_collected = df_collected.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("pattern_3bar")).alias("rv_3bar_pattern")
    )
    results_rv_3bar = test_ood_robustness(df_collected, "rv_3bar_pattern", "fwd_return")

    return {
        "symbol": symbol,
        "2bar": results_2bar,
        "3bar": results_3bar,
        "rv_3bar": results_rv_3bar,
    }


def aggregate_results(all_results: dict[str, dict]) -> dict:
    """Aggregate results across symbols."""
    # Count universal patterns for each type
    counts_2bar: dict[str, int] = {}
    counts_3bar: dict[str, int] = {}
    counts_rv_3bar: dict[str, int] = {}

    stats_2bar: dict[str, list] = {}
    stats_3bar: dict[str, list] = {}
    stats_rv_3bar: dict[str, list] = {}

    for _symbol, result in all_results.items():
        for pattern in result["2bar"]["robust_patterns"]:
            counts_2bar[pattern] = counts_2bar.get(pattern, 0) + 1
        for pattern in result["3bar"]["robust_patterns"]:
            counts_3bar[pattern] = counts_3bar.get(pattern, 0) + 1
        for pattern in result["rv_3bar"]["robust_patterns"]:
            counts_rv_3bar[pattern] = counts_rv_3bar.get(pattern, 0) + 1

        # Collect stats
        for stat in result["2bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_2bar:
                stats_2bar[p] = []
            stats_2bar[p].append(stat)

        for stat in result["3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_3bar:
                stats_3bar[p] = []
            stats_3bar[p].append(stat)

        for stat in result["rv_3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_rv_3bar:
                stats_rv_3bar[p] = []
            stats_rv_3bar[p].append(stat)

    # Universal = present in all 4 symbols
    universal_2bar = [p for p, c in counts_2bar.items() if c >= 4]
    universal_3bar = [p for p, c in counts_3bar.items() if c >= 4]
    universal_rv_3bar = [p for p, c in counts_rv_3bar.items() if c >= 4]

    # Compute aggregate stats for universal patterns
    def aggregate_stats(universal: list[str], stats: dict[str, list]) -> list[dict]:
        result = []
        for pattern in universal:
            if pattern not in stats:
                continue
            stats_list = stats[pattern]
            total_n = sum(s["n_samples"] for s in stats_list)
            avg_net = sum(s["net_bps"] * s["n_samples"] for s in stats_list) / total_n if total_n > 0 else 0
            result.append({
                "pattern": pattern,
                "total_n": total_n,
                "avg_net_bps": round(avg_net, 2),
            })
        return result

    return {
        "2bar": {
            "universal": universal_2bar,
            "count": len(universal_2bar),
            "stats": aggregate_stats(universal_2bar, stats_2bar),
        },
        "3bar": {
            "universal": universal_3bar,
            "count": len(universal_3bar),
            "stats": aggregate_stats(universal_3bar, stats_3bar),
        },
        "rv_3bar": {
            "universal": universal_rv_3bar,
            "count": len(universal_rv_3bar),
            "stats": aggregate_stats(universal_rv_3bar, stats_rv_3bar),
        },
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting 3-bar pattern analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_pattern_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol complete",
            symbol=symbol,
            robust_2bar=results["2bar"]["count"],
            robust_3bar=results["3bar"]["count"],
            robust_rv_3bar=results["rv_3bar"]["count"],
        )

    # Aggregate
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        universal_2bar=aggregated["2bar"]["count"],
        universal_3bar=aggregated["3bar"]["count"],
        universal_rv_3bar=aggregated["rv_3bar"]["count"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("3-BAR PATTERN ANALYSIS RESULTS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("PATTERN COUNT COMPARISON")
    print("-" * 80)
    print(f"{'Pattern Type':<25} {'Universal Patterns':>20}")
    print("-" * 80)
    print(f"{'2-bar patterns':<25} {aggregated['2bar']['count']:>20}")
    print(f"{'3-bar patterns':<25} {aggregated['3bar']['count']:>20}")
    print(f"{'RV + 3-bar patterns':<25} {aggregated['rv_3bar']['count']:>20}")

    # Print 3-bar patterns
    if aggregated["3bar"]["stats"]:
        print("\n" + "-" * 80)
        print("UNIVERSAL 3-BAR PATTERNS")
        print("-" * 80)
        print(f"{'Pattern':<12} {'N':>12} {'Net (bps)':>12}")
        print("-" * 80)
        for stat in sorted(aggregated["3bar"]["stats"], key=lambda x: x["avg_net_bps"], reverse=True):
            print(f"{stat['pattern']:<12} {stat['total_n']:>12,} {stat['avg_net_bps']:>12.2f}")

    # Print RV + 3-bar patterns
    if aggregated["rv_3bar"]["stats"]:
        print("\n" + "-" * 80)
        print("TOP RV + 3-BAR PATTERNS")
        print("-" * 80)
        print(f"{'Pattern':<25} {'N':>12} {'Net (bps)':>12}")
        print("-" * 80)
        sorted_stats = sorted(aggregated["rv_3bar"]["stats"], key=lambda x: x["avg_net_bps"], reverse=True)
        for stat in sorted_stats[:15]:
            print(f"{stat['pattern']:<25} {stat['total_n']:>12,} {stat['avg_net_bps']:>12.2f}")

    # Comparison analysis
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    log_info(
        "Pattern comparison",
        improvement_3bar_vs_2bar=aggregated["3bar"]["count"] - aggregated["2bar"]["count"],
        improvement_rv3bar_vs_3bar=aggregated["rv_3bar"]["count"] - aggregated["3bar"]["count"],
    )

    log_info("Analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
