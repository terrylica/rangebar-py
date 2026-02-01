#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""3-bar pattern + multi-threshold alignment analysis (Polars).

Combines validated 3-bar patterns with multi-threshold alignment (50/100/200 dbps)
to create a two-factor signal. Tests if the combination yields more robust patterns.

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


def create_combined_3bar_alignment_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Create combined 3-bar + alignment pattern column."""
    return df.with_columns(
        (pl.col("alignment") + "|" + pl.col("pattern_3bar"))
        .alias("alignment_3bar_pattern")
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


def run_combined_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run 3-bar + alignment combined analysis."""
    log_info("Loading data", symbol=symbol)

    # Load all thresholds
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Process 50 and 200 dbps - just need direction
    df_50 = add_bar_direction(df_50).collect()
    df_200 = add_bar_direction(df_200).collect()

    # Process 100 dbps - add direction and 3-bar patterns
    df_100 = add_bar_direction(df_100)
    df_100 = add_3bar_patterns(df_100)

    # Add forward return before collecting
    df_100 = df_100.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    ).collect()

    log_info("Data loaded", symbol=symbol, bars_100=len(df_100))

    # Align thresholds
    aligned = align_thresholds(df_50, df_100, df_200)
    aligned = classify_alignment(aligned)
    aligned = create_combined_3bar_alignment_pattern(aligned)

    # Test standalone 3-bar patterns
    results_3bar = test_ood_robustness(aligned, "pattern_3bar", "fwd_return")

    # Test combined alignment + 3-bar patterns
    results_combined = test_ood_robustness(aligned, "alignment_3bar_pattern", "fwd_return")

    return {
        "symbol": symbol,
        "3bar": results_3bar,
        "alignment_3bar": results_combined,
    }


def aggregate_results(all_results: dict[str, dict]) -> dict:
    """Aggregate results across symbols."""
    # Count universal patterns for each type
    counts_3bar: dict[str, int] = {}
    counts_combined: dict[str, int] = {}

    stats_3bar: dict[str, list] = {}
    stats_combined: dict[str, list] = {}

    for _symbol, result in all_results.items():
        for pattern in result["3bar"]["robust_patterns"]:
            counts_3bar[pattern] = counts_3bar.get(pattern, 0) + 1
        for pattern in result["alignment_3bar"]["robust_patterns"]:
            counts_combined[pattern] = counts_combined.get(pattern, 0) + 1

        # Collect stats
        for stat in result["3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_3bar:
                stats_3bar[p] = []
            stats_3bar[p].append(stat)

        for stat in result["alignment_3bar"]["pattern_stats"]:
            p = stat["pattern"]
            if p not in stats_combined:
                stats_combined[p] = []
            stats_combined[p].append(stat)

    # Universal = present in all 4 symbols
    universal_3bar = [p for p, c in counts_3bar.items() if c >= 4]
    universal_combined = [p for p, c in counts_combined.items() if c >= 4]

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
        "3bar": {
            "universal": universal_3bar,
            "count": len(universal_3bar),
            "stats": aggregate_stats(universal_3bar, stats_3bar),
        },
        "alignment_3bar": {
            "universal": universal_combined,
            "count": len(universal_combined),
            "stats": aggregate_stats(universal_combined, stats_combined),
        },
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting 3-bar + alignment combined analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_combined_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol complete",
            symbol=symbol,
            robust_3bar=results["3bar"]["count"],
            robust_combined=results["alignment_3bar"]["count"],
        )

    # Aggregate
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        universal_3bar=aggregated["3bar"]["count"],
        universal_combined=aggregated["alignment_3bar"]["count"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("3-BAR + ALIGNMENT COMBINED ANALYSIS RESULTS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("PATTERN COUNT COMPARISON")
    print("-" * 80)
    print(f"{'Pattern Type':<30} {'Universal Patterns':>20}")
    print("-" * 80)
    print(f"{'3-bar patterns (baseline)':<30} {aggregated['3bar']['count']:>20}")
    print(f"{'Alignment + 3-bar (combined)':<30} {aggregated['alignment_3bar']['count']:>20}")

    improvement = aggregated["alignment_3bar"]["count"] - aggregated["3bar"]["count"]
    ratio = aggregated["alignment_3bar"]["count"] / aggregated["3bar"]["count"] if aggregated["3bar"]["count"] > 0 else 0
    print(f"\n  Improvement: +{improvement} patterns ({ratio:.1f}x)")

    # Print 3-bar patterns
    if aggregated["3bar"]["stats"]:
        print("\n" + "-" * 80)
        print("UNIVERSAL 3-BAR PATTERNS")
        print("-" * 80)
        print(f"{'Pattern':<12} {'N':>12} {'Net (bps)':>12}")
        print("-" * 80)
        for stat in sorted(aggregated["3bar"]["stats"], key=lambda x: x["avg_net_bps"], reverse=True):
            print(f"{stat['pattern']:<12} {stat['total_n']:>12,} {stat['avg_net_bps']:>12.2f}")

    # Print combined patterns
    if aggregated["alignment_3bar"]["stats"]:
        print("\n" + "-" * 80)
        print("TOP ALIGNMENT + 3-BAR COMBINED PATTERNS")
        print("-" * 80)
        print(f"{'Pattern':<30} {'N':>12} {'Net (bps)':>12}")
        print("-" * 80)
        sorted_stats = sorted(aggregated["alignment_3bar"]["stats"], key=lambda x: x["avg_net_bps"], reverse=True)
        for stat in sorted_stats[:20]:
            print(f"{stat['pattern']:<30} {stat['total_n']:>12,} {stat['avg_net_bps']:>12.2f}")

    # Key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    # Count profitable patterns
    profitable_3bar = len([s for s in aggregated["3bar"]["stats"] if s["avg_net_bps"] > 0])
    profitable_combined = len([s for s in aggregated["alignment_3bar"]["stats"] if s["avg_net_bps"] > 0])

    print(f"  Profitable 3-bar patterns: {profitable_3bar}/{aggregated['3bar']['count']}")
    print(f"  Profitable combined patterns: {profitable_combined}/{aggregated['alignment_3bar']['count']}")

    log_info(
        "Analysis complete",
        improvement=improvement,
        ratio=ratio,
        profitable_3bar=profitable_3bar,
        profitable_combined=profitable_combined,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
