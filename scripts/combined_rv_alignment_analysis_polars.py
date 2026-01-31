#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Combined RV regime + multi-threshold alignment analysis (Polars).

Tests whether combining RV regime filter with multi-threshold alignment
produces stronger OOD robust patterns than either alone.

Hypothesis: Patterns in favorable RV regime AND aligned across thresholds
may show higher returns and better robustness.

GitHub Issues: #54 (RV regime), #55 (multi-threshold)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

import polars as pl

# Transaction cost (high VIP tier)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001 / 100  # Convert to decimal

# Symbols and date range
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2026-01-31"

# SOL has later start
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
# Technical Indicators
# =============================================================================


def compute_realized_volatility(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Realized Volatility (std of log returns)."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=window, min_samples=window).alias(alias)
    ).drop("_log_ret")


def classify_rv_regime(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify RV regime using rolling percentiles."""
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
# Multi-Threshold Alignment
# =============================================================================


def align_thresholds(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars across thresholds using timestamps."""
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
    """Classify multi-threshold alignment patterns."""
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


def test_ood_robustness(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Test OOD robustness of patterns."""
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
            "min_t": round(min(period_stats), 2),
            "max_t": round(max(period_stats), 2),
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
    """Run combined RV regime + alignment analysis."""
    log_info("Loading data", symbol=symbol)

    # Load at all thresholds
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Add directions
    df_50 = add_bar_direction(df_50)
    df_100 = add_bar_direction(df_100)
    df_200 = add_bar_direction(df_200)

    # Add patterns to 100 dbps
    df_100 = add_patterns(df_100)

    # Add RV regime to 100 dbps
    df_100 = compute_realized_volatility(df_100, 20, "rv")
    df_100 = classify_rv_regime(df_100)

    # Add forward return
    df_100 = df_100.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    )

    # Collect
    df_50_collected = df_50.collect()
    df_100_collected = df_100.collect()
    df_200_collected = df_200.collect()

    log_info("Data loaded", symbol=symbol, bars_100=len(df_100_collected))

    # Align thresholds
    aligned = align_thresholds(df_50_collected, df_100_collected, df_200_collected)

    # Classify alignment
    aligned = classify_alignment(aligned)

    # Create combined pattern: rv_regime + alignment + pattern_2bar
    aligned = aligned.with_columns(
        (pl.col("rv_regime") + "_" + pl.col("alignment") + "|" + pl.col("pattern_2bar"))
        .alias("combined_pattern")
    )

    # Also create rv_only and alignment_only patterns for comparison
    aligned = aligned.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("pattern_2bar")).alias("rv_only_pattern"),
        (pl.col("alignment") + "|" + pl.col("pattern_2bar")).alias("alignment_only_pattern"),
    )

    log_info("Patterns created", symbol=symbol, rows=len(aligned))

    # Test OOD robustness for each pattern type
    combined_results = test_ood_robustness(aligned, "combined_pattern", "fwd_return")
    rv_only_results = test_ood_robustness(aligned, "rv_only_pattern", "fwd_return")
    alignment_only_results = test_ood_robustness(aligned, "alignment_only_pattern", "fwd_return")

    return {
        "symbol": symbol,
        "combined": combined_results,
        "rv_only": rv_only_results,
        "alignment_only": alignment_only_results,
    }


def aggregate_results(all_results: dict[str, dict]) -> dict:
    """Aggregate results across symbols."""
    # Count universal patterns for each type
    combined_counts: dict[str, int] = {}
    rv_only_counts: dict[str, int] = {}
    alignment_only_counts: dict[str, int] = {}

    for _symbol, result in all_results.items():
        for pattern in result["combined"]["robust_patterns"]:
            combined_counts[pattern] = combined_counts.get(pattern, 0) + 1
        for pattern in result["rv_only"]["robust_patterns"]:
            rv_only_counts[pattern] = rv_only_counts.get(pattern, 0) + 1
        for pattern in result["alignment_only"]["robust_patterns"]:
            alignment_only_counts[pattern] = alignment_only_counts.get(pattern, 0) + 1

    # Universal = present in all 4 symbols
    combined_universal = [p for p, c in combined_counts.items() if c >= 4]
    rv_only_universal = [p for p, c in rv_only_counts.items() if c >= 4]
    alignment_only_universal = [p for p, c in alignment_only_counts.items() if c >= 4]

    # Aggregate stats for universal patterns
    combined_stats = []
    for pattern in combined_universal:
        stats_list = []
        for _symbol, result in all_results.items():
            stat = next((s for s in result["combined"]["pattern_stats"] if s["pattern"] == pattern), None)
            if stat:
                stats_list.append(stat)

        if stats_list:
            total_n = sum(s["n_samples"] for s in stats_list)
            avg_net = sum(s["net_bps"] * s["n_samples"] for s in stats_list) / total_n
            combined_stats.append({
                "pattern": pattern,
                "total_n": total_n,
                "avg_net_bps": round(avg_net, 2),
            })

    return {
        "combined_universal": combined_universal,
        "combined_count": len(combined_universal),
        "combined_stats": combined_stats,
        "rv_only_universal": rv_only_universal,
        "rv_only_count": len(rv_only_universal),
        "alignment_only_universal": alignment_only_universal,
        "alignment_only_count": len(alignment_only_universal),
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting combined RV regime + alignment analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_combined_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol complete",
            symbol=symbol,
            combined_robust=results["combined"]["count"],
            rv_only_robust=results["rv_only"]["count"],
            alignment_only_robust=results["alignment_only"]["count"],
        )

    # Aggregate
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        combined_universal=aggregated["combined_count"],
        rv_only_universal=aggregated["rv_only_count"],
        alignment_only_universal=aggregated["alignment_only_count"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("COMBINED RV REGIME + MULTI-THRESHOLD ALIGNMENT ANALYSIS")
    print("=" * 80 + "\n")

    print("-" * 80)
    print("PATTERN COUNT COMPARISON")
    print("-" * 80)
    print(f"{'Pattern Type':<25} {'Universal Patterns':>20}")
    print("-" * 80)
    print(f"{'RV Regime Only':<25} {aggregated['rv_only_count']:>20}")
    print(f"{'Alignment Only':<25} {aggregated['alignment_only_count']:>20}")
    print(f"{'Combined (RV + Align)':<25} {aggregated['combined_count']:>20}")

    # Print combined patterns with best returns
    if aggregated["combined_stats"]:
        print("\n" + "-" * 80)
        print("TOP COMBINED PATTERNS (by net return)")
        print("-" * 80)
        print(f"{'Pattern':<45} {'N':>12} {'Net (bps)':>12}")
        print("-" * 80)

        sorted_stats = sorted(aggregated["combined_stats"], key=lambda x: x["avg_net_bps"], reverse=True)
        for stat in sorted_stats[:15]:
            print(f"{stat['pattern']:<45} {stat['total_n']:>12,} {stat['avg_net_bps']:>12.2f}")

    # Key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    # Profitable combined patterns
    profitable = [s for s in aggregated["combined_stats"] if s["avg_net_bps"] > 0]
    unprofitable = [s for s in aggregated["combined_stats"] if s["avg_net_bps"] <= 0]

    log_info(
        "Profitability summary",
        profitable_combined=len(profitable),
        unprofitable_combined=len(unprofitable),
    )

    # Compare improvement from combining
    log_info(
        "Combination effect",
        rv_only_count=aggregated["rv_only_count"],
        alignment_only_count=aggregated["alignment_only_count"],
        combined_count=aggregated["combined_count"],
        improvement_vs_rv=aggregated["combined_count"] - aggregated["rv_only_count"],
        improvement_vs_align=aggregated["combined_count"] - aggregated["alignment_only_count"],
    )

    log_info("Analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
