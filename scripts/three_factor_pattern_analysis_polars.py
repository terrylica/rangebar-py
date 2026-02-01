#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Three-factor pattern analysis: RV regime + 3-bar + multi-threshold alignment (Polars).

Combines all three validated factors into a single signal:
- RV regime (quiet/active/volatile)
- 3-bar patterns (8 patterns)
- Multi-threshold alignment (5 states)

Tests if the three-factor combination yields more robust patterns than two-factor.

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

    return df.drop(["rv_p25", "rv_p75"])


def align_thresholds(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars across thresholds using asof join."""
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


def create_three_factor_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Create three-factor pattern column: rv_regime|alignment|3bar_pattern."""
    return df.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("alignment") + "|" + pl.col("pattern_3bar"))
        .alias("three_factor_pattern")
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


def run_three_factor_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run three-factor pattern analysis."""
    log_info("Loading data", symbol=symbol)

    # Load all thresholds
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Process 50 and 200 dbps - just need direction
    df_50 = add_bar_direction(df_50).collect()
    df_200 = add_bar_direction(df_200).collect()

    # Process 100 dbps - add direction, 3-bar patterns, and RV regime
    df_100 = add_bar_direction(df_100)
    df_100 = add_3bar_patterns(df_100)
    df_100 = compute_rv_and_regime(df_100)

    # Add forward return before collecting
    df_100 = df_100.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    ).collect()

    log_info("Data loaded", symbol=symbol, bars_100=len(df_100))

    # Align thresholds
    aligned = align_thresholds(df_50, df_100, df_200)
    aligned = classify_alignment(aligned)
    aligned = create_three_factor_pattern(aligned)

    # Test two-factor baselines
    # RV + 3-bar
    aligned = aligned.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("pattern_3bar")).alias("rv_3bar_pattern")
    )
    results_rv_3bar = test_ood_robustness(aligned, "rv_3bar_pattern", "fwd_return")

    # Alignment + 3-bar
    aligned = aligned.with_columns(
        (pl.col("alignment") + "|" + pl.col("pattern_3bar")).alias("alignment_3bar_pattern")
    )
    results_alignment_3bar = test_ood_robustness(aligned, "alignment_3bar_pattern", "fwd_return")

    # Three-factor
    results_three_factor = test_ood_robustness(aligned, "three_factor_pattern", "fwd_return")

    return {
        "symbol": symbol,
        "rv_3bar": results_rv_3bar,
        "alignment_3bar": results_alignment_3bar,
        "three_factor": results_three_factor,
    }


def aggregate_results(all_results: dict[str, dict]) -> dict:
    """Aggregate results across symbols."""
    # Count universal patterns for each type
    counts_rv_3bar: dict[str, int] = {}
    counts_alignment_3bar: dict[str, int] = {}
    counts_three_factor: dict[str, int] = {}

    stats_rv_3bar: dict[str, list] = {}
    stats_alignment_3bar: dict[str, list] = {}
    stats_three_factor: dict[str, list] = {}

    for _symbol, result in all_results.items():
        for pattern in result["rv_3bar"]["robust_patterns"]:
            counts_rv_3bar[pattern] = counts_rv_3bar.get(pattern, 0) + 1
        for pattern in result["alignment_3bar"]["robust_patterns"]:
            counts_alignment_3bar[pattern] = counts_alignment_3bar.get(pattern, 0) + 1
        for pattern in result["three_factor"]["robust_patterns"]:
            counts_three_factor[pattern] = counts_three_factor.get(pattern, 0) + 1

        # Collect stats
        for stat in result["rv_3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_rv_3bar:
                stats_rv_3bar[p] = []
            stats_rv_3bar[p].append(stat)

        for stat in result["alignment_3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_alignment_3bar:
                stats_alignment_3bar[p] = []
            stats_alignment_3bar[p].append(stat)

        for stat in result["three_factor"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_three_factor:
                stats_three_factor[p] = []
            stats_three_factor[p].append(stat)

    # Universal = present in all 4 symbols
    universal_rv_3bar = [p for p, c in counts_rv_3bar.items() if c >= 4]
    universal_alignment_3bar = [p for p, c in counts_alignment_3bar.items() if c >= 4]
    universal_three_factor = [p for p, c in counts_three_factor.items() if c >= 4]

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
        "rv_3bar": {
            "universal": universal_rv_3bar,
            "count": len(universal_rv_3bar),
            "stats": aggregate_stats(universal_rv_3bar, stats_rv_3bar),
        },
        "alignment_3bar": {
            "universal": universal_alignment_3bar,
            "count": len(universal_alignment_3bar),
            "stats": aggregate_stats(universal_alignment_3bar, stats_alignment_3bar),
        },
        "three_factor": {
            "universal": universal_three_factor,
            "count": len(universal_three_factor),
            "stats": aggregate_stats(universal_three_factor, stats_three_factor),
        },
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting three-factor pattern analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_three_factor_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol complete",
            symbol=symbol,
            robust_rv_3bar=results["rv_3bar"]["count"],
            robust_alignment_3bar=results["alignment_3bar"]["count"],
            robust_three_factor=results["three_factor"]["count"],
        )

    # Aggregate
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        universal_rv_3bar=aggregated["rv_3bar"]["count"],
        universal_alignment_3bar=aggregated["alignment_3bar"]["count"],
        universal_three_factor=aggregated["three_factor"]["count"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("THREE-FACTOR PATTERN ANALYSIS RESULTS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("PATTERN COUNT COMPARISON")
    print("-" * 80)
    print(f"{'Pattern Type':<35} {'Universal Patterns':>20}")
    print("-" * 80)
    print(f"{'RV + 3-bar (two-factor)':<35} {aggregated['rv_3bar']['count']:>20}")
    print(f"{'Alignment + 3-bar (two-factor)':<35} {aggregated['alignment_3bar']['count']:>20}")
    print(f"{'RV + Alignment + 3-bar (three-factor)':<35} {aggregated['three_factor']['count']:>20}")

    # Calculate improvements
    best_two_factor = max(aggregated["rv_3bar"]["count"], aggregated["alignment_3bar"]["count"])
    improvement = aggregated["three_factor"]["count"] - best_two_factor
    ratio = aggregated["three_factor"]["count"] / best_two_factor if best_two_factor > 0 else 0
    print(f"\n  Improvement over best two-factor: +{improvement} patterns ({ratio:.1f}x)")

    # Print top three-factor patterns
    if aggregated["three_factor"]["stats"]:
        print("\n" + "-" * 80)
        print("TOP THREE-FACTOR PATTERNS (by net return)")
        print("-" * 80)
        print(f"{'Pattern':<45} {'N':>10} {'Net (bps)':>12}")
        print("-" * 80)
        sorted_stats = sorted(aggregated["three_factor"]["stats"], key=lambda x: x["avg_net_bps"], reverse=True)
        for stat in sorted_stats[:25]:
            print(f"{stat['pattern']:<45} {stat['total_n']:>10,} {stat['avg_net_bps']:>12.2f}")

    # Key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    # Count profitable patterns
    profitable_rv_3bar = len([s for s in aggregated["rv_3bar"]["stats"] if s["avg_net_bps"] > 0])
    profitable_alignment_3bar = len([s for s in aggregated["alignment_3bar"]["stats"] if s["avg_net_bps"] > 0])
    profitable_three_factor = len([s for s in aggregated["three_factor"]["stats"] if s["avg_net_bps"] > 0])

    print(f"  Profitable RV + 3-bar: {profitable_rv_3bar}/{aggregated['rv_3bar']['count']}")
    print(f"  Profitable Alignment + 3-bar: {profitable_alignment_3bar}/{aggregated['alignment_3bar']['count']}")
    print(f"  Profitable Three-factor: {profitable_three_factor}/{aggregated['three_factor']['count']}")

    log_info(
        "Analysis complete",
        improvement=improvement,
        ratio=ratio,
        profitable_three_factor=profitable_three_factor,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
