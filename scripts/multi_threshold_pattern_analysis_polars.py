#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-threshold pattern analysis for OOD robust patterns (Polars).

Tests whether patterns that align across multiple range bar thresholds
(50, 100, 200 dbps) are more robust than single-threshold patterns.

Hypothesis: Aligned patterns (same direction at multiple granularities)
may provide stronger confirmation signals.

GitHub Issue: #55 (multi-threshold research)
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

# Thresholds to analyze
THRESHOLDS = [50, 100, 200]
BASE_THRESHOLD = 100  # Primary threshold for returns


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


def add_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


# =============================================================================
# Multi-Threshold Analysis
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


def align_thresholds(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars across thresholds using timestamps.

    For each 100 dbps bar, find the most recent 50 dbps and 200 dbps pattern.
    This creates a multi-threshold view at the 100 dbps resolution.
    """
    # Add suffixes to distinguish columns
    df_50 = df_50.select([
        pl.col("timestamp").alias("timestamp_50"),
        pl.col("direction").alias("direction_50"),
        pl.col("pattern_2bar").alias("pattern_50"),
    ])

    df_200 = df_200.select([
        pl.col("timestamp").alias("timestamp_200"),
        pl.col("direction").alias("direction_200"),
        pl.col("pattern_2bar").alias("pattern_200"),
    ])

    # For each 100 bar, find most recent 50 and 200 patterns using asof join
    df_100 = df_100.sort("timestamp")
    df_50 = df_50.sort("timestamp_50")
    df_200 = df_200.sort("timestamp_200")

    # Asof join: find most recent 50 dbps bar for each 100 dbps bar
    aligned = df_100.join_asof(
        df_50,
        left_on="timestamp",
        right_on="timestamp_50",
        strategy="backward",
    )

    # Asof join: find most recent 200 dbps bar for each 100 dbps bar
    aligned = aligned.join_asof(
        df_200,
        left_on="timestamp",
        right_on="timestamp_200",
        strategy="backward",
    )

    return aligned


def classify_alignment(df: pl.DataFrame) -> pl.DataFrame:
    """Classify multi-threshold alignment patterns."""
    # Alignment types:
    # - "aligned_up": U at all thresholds
    # - "aligned_down": D at all thresholds
    # - "partial_up": U at 2/3 thresholds
    # - "partial_down": D at 2/3 thresholds
    # - "mixed": No clear alignment

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

    # Combined pattern: pattern_100 + alignment
    df = df.with_columns(
        (pl.col("pattern_2bar") + "_" + pl.col("alignment")).alias("aligned_pattern")
    )

    return df


def analyze_aligned_patterns(
    df: pl.DataFrame,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Analyze OOD robustness of aligned patterns."""
    # Add period for quarterly analysis
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    patterns = df.select("aligned_pattern").unique().to_series().drop_nulls().to_list()
    robust_patterns = []
    pattern_stats = []

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_df = df.filter(pl.col("aligned_pattern") == pattern)
        periods = pattern_df.select("period").unique().to_series().drop_nulls().to_list()

        period_stats = []
        total_samples = 0
        for period_label in periods:
            period_data = pattern_df.filter(pl.col("period") == period_label)
            returns = period_data.select("fwd_return").drop_nulls().to_series()
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

        # Compute overall statistics
        all_returns = pattern_df.select("fwd_return").drop_nulls().to_series()
        mean_return = float(all_returns.mean()) if all_returns.mean() is not None else 0.0
        std_return = float(all_returns.std()) if all_returns.std() is not None else 0.0
        overall_t = compute_t_stat(all_returns)

        # Net return after costs
        net_mean = mean_return - ROUND_TRIP_COST_PCT

        pattern_stats.append({
            "pattern": pattern,
            "n_periods": len(period_stats),
            "n_samples": total_samples,
            "mean_return_bps": round(mean_return * 10000, 2),
            "net_return_bps": round(net_mean * 10000, 2),
            "std_bps": round(std_return * 10000, 2),
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


def compare_alignment_effect(df: pl.DataFrame) -> dict:
    """Compare returns between aligned and non-aligned patterns."""
    # Group by base pattern and alignment
    base_patterns = ["DD", "DU", "UD", "UU"]
    comparison = []

    for base in base_patterns:
        # Filter by base pattern
        base_df = df.filter(pl.col("pattern_2bar") == base)
        if len(base_df) < 100:
            continue

        # Compare aligned vs partial vs mixed
        for alignment in ["aligned_up", "aligned_down", "partial_up", "partial_down", "mixed"]:
            aligned_df = base_df.filter(pl.col("alignment") == alignment)
            if len(aligned_df) < 100:
                continue

            returns = aligned_df.select("fwd_return").drop_nulls().to_series()
            mean_ret = float(returns.mean()) if returns.mean() is not None else 0.0
            t_stat = compute_t_stat(returns)

            comparison.append({
                "base_pattern": base,
                "alignment": alignment,
                "n": len(aligned_df),
                "mean_bps": round(mean_ret * 10000, 2),
                "net_bps": round((mean_ret - ROUND_TRIP_COST_PCT) * 10000, 2),
                "t_stat": round(t_stat, 2),
            })

    return {"comparison": comparison}


# =============================================================================
# Main Analysis
# =============================================================================


def run_multi_threshold_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run multi-threshold analysis for a symbol."""
    log_info("Loading data at multiple thresholds", symbol=symbol)

    # Load data at each threshold
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Add directions and patterns
    df_50 = add_bar_direction(df_50)
    df_50 = add_patterns(df_50)

    df_100 = add_bar_direction(df_100)
    df_100 = add_patterns(df_100)

    df_200 = add_bar_direction(df_200)
    df_200 = add_patterns(df_200)

    # Collect
    df_50_collected = df_50.collect()
    df_100_collected = df_100.collect()
    df_200_collected = df_200.collect()

    log_info(
        "Data loaded",
        symbol=symbol,
        bars_50=len(df_50_collected),
        bars_100=len(df_100_collected),
        bars_200=len(df_200_collected),
    )

    # Add forward return to 100 dbps (base threshold)
    df_100_collected = df_100_collected.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    )

    # Align thresholds
    aligned = align_thresholds(df_50_collected, df_100_collected, df_200_collected)

    # Classify alignment
    aligned = classify_alignment(aligned)

    log_info("Alignment complete", symbol=symbol, aligned_bars=len(aligned))

    # Analyze OOD robustness
    robustness = analyze_aligned_patterns(aligned)

    # Compare alignment effect
    comparison = compare_alignment_effect(aligned)

    return {
        "symbol": symbol,
        "bars_50": len(df_50_collected),
        "bars_100": len(df_100_collected),
        "bars_200": len(df_200_collected),
        "aligned_bars": len(aligned),
        "robustness": robustness,
        "comparison": comparison,
    }


def aggregate_results(all_results: dict[str, dict]) -> dict:
    """Aggregate results across all symbols."""
    # Count universal patterns (present in all symbols)
    pattern_counts: dict[str, int] = {}
    pattern_stats: dict[str, list] = {}

    for symbol, result in all_results.items():
        for stat in result["robustness"]["pattern_stats"]:
            pattern = stat["pattern"]
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
                pattern_stats[pattern] = []

            if stat["is_robust"]:
                pattern_counts[pattern] += 1
                pattern_stats[pattern].append({
                    "symbol": symbol,
                    **stat,
                })

    # Universal = robust in all 4 symbols
    universal = [p for p, count in pattern_counts.items() if count >= 4]

    # Aggregate comparison across symbols
    comparison_agg: dict[str, dict] = {}
    for _symbol, result in all_results.items():
        for comp in result["comparison"]["comparison"]:
            key = f"{comp['base_pattern']}_{comp['alignment']}"
            if key not in comparison_agg:
                comparison_agg[key] = {
                    "base_pattern": comp["base_pattern"],
                    "alignment": comp["alignment"],
                    "total_n": 0,
                    "weighted_mean": 0.0,
                    "symbols": 0,
                }
            comparison_agg[key]["total_n"] += comp["n"]
            comparison_agg[key]["weighted_mean"] += comp["mean_bps"] * comp["n"]
            comparison_agg[key]["symbols"] += 1

    # Normalize weighted means
    for _key, value in comparison_agg.items():
        if value["total_n"] > 0:
            value["weighted_mean"] /= value["total_n"]
            value["weighted_mean"] = round(value["weighted_mean"], 2)

    return {
        "universal_patterns": universal,
        "universal_count": len(universal),
        "pattern_counts": pattern_counts,
        "comparison_agg": list(comparison_agg.values()),
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting multi-threshold pattern analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_multi_threshold_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            robust_patterns=results["robustness"]["count"],
        )

    # Aggregate across symbols
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        universal_patterns=aggregated["universal_count"],
        universal_list=aggregated["universal_patterns"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-THRESHOLD PATTERN ANALYSIS RESULTS")
    print("=" * 80 + "\n")

    # Universal patterns
    print("-" * 80)
    print("UNIVERSAL ALIGNED PATTERNS (Robust in ALL 4 symbols)")
    print("-" * 80)
    for pattern in sorted(aggregated["universal_patterns"]):
        print(f"  {pattern}")
    print(f"\nTotal: {aggregated['universal_count']} universal patterns")

    # Alignment effect comparison
    print("\n" + "-" * 80)
    print("ALIGNMENT EFFECT ON RETURNS")
    print("-" * 80)
    print(f"{'Base':<6} {'Alignment':<14} {'N':>10} {'Mean(bps)':>10} {'Symbols':>8}")
    print("-" * 80)

    for comp in sorted(aggregated["comparison_agg"], key=lambda x: (x["base_pattern"], x["alignment"])):
        print(
            f"{comp['base_pattern']:<6} "
            f"{comp['alignment']:<14} "
            f"{comp['total_n']:>10,} "
            f"{comp['weighted_mean']:>10.2f} "
            f"{comp['symbols']:>8}"
        )

    # Key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    # Compare aligned vs non-aligned for each base pattern
    for base in ["DD", "DU", "UD", "UU"]:
        aligned_up = next((c for c in aggregated["comparison_agg"]
                          if c["base_pattern"] == base and c["alignment"] == "aligned_up"), None)
        aligned_down = next((c for c in aggregated["comparison_agg"]
                            if c["base_pattern"] == base and c["alignment"] == "aligned_down"), None)
        mixed = next((c for c in aggregated["comparison_agg"]
                     if c["base_pattern"] == base and c["alignment"] == "mixed"), None)

        if aligned_up and mixed:
            diff = aligned_up["weighted_mean"] - mixed["weighted_mean"]
            log_info(
                "Alignment effect",
                base_pattern=base,
                aligned_up_mean=aligned_up["weighted_mean"],
                mixed_mean=mixed["weighted_mean"],
                difference=round(diff, 2),
            )
        if aligned_down and mixed:
            diff = aligned_down["weighted_mean"] - mixed["weighted_mean"]
            log_info(
                "Alignment effect",
                base_pattern=base,
                aligned_down_mean=aligned_down["weighted_mean"],
                mixed_mean=mixed["weighted_mean"],
                difference=round(diff, 2),
            )

    log_info("Analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
