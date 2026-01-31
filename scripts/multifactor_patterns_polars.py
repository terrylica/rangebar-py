#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-factor range bar pattern analysis (Polars-optimized).

This script combines range bars at different thresholds (50, 100, 200 dbps)
as multi-factor signals to find ODD robust patterns.

Approach:
- 100 dbps: Primary signal (2-bar patterns)
- 50 dbps: Finer granularity confirmation (recent direction)
- 200 dbps: Trend filter (higher timeframe direction)

Key hypothesis: Multi-timeframe alignment may reveal stronger signals
than single-timeframe patterns.

User priorities (from AskUserQuestion):
- Focus on multi-factor patterns
- All 4 crypto symbols must pass ODD robustness
- Use |t-stat| >= 3 threshold

Polars optimizations:
- Vectorized classification (no .apply())
- join_asof for O(N log M) alignment
- Lazy evaluation with predicate pushdown
- Parallel groupby operations

GitHub Issue: #52
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass


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


def log_warn(message: str, **kwargs: object) -> None:
    """Log WARNING level message."""
    log_json("WARNING", message, **kwargs)


def log_error(message: str, **kwargs: object) -> None:
    """Log ERROR level message."""
    log_json("ERROR", message, **kwargs)


# =============================================================================
# Multi-Timeframe Alignment (Polars Vectorized)
# =============================================================================


def classify_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify each bar as 'U' (up) or 'D' (down)."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def compute_htf_trend(df: pl.LazyFrame, lookback: int = 3) -> pl.LazyFrame:
    """Compute higher-timeframe trend direction (vectorized).

    Returns 'up' if majority of last N bars are up, 'down' if majority down.
    """
    df = classify_bar_direction(df)
    return df.with_columns(
        (pl.col("direction") == "U")
        .cast(pl.Int32)
        .rolling_sum(window_size=lookback, min_samples=lookback)
        .alias("up_count")
    ).with_columns(
        pl.when(pl.col("up_count").is_null())
        .then(pl.lit("neutral"))
        .when(pl.col("up_count") >= lookback * 0.67)
        .then(pl.lit("up"))
        .when(pl.col("up_count") <= lookback * 0.33)
        .then(pl.lit("down"))
        .otherwise(pl.lit("neutral"))
        .alias("htf_trend")
    )


def compute_fine_direction(df: pl.LazyFrame, lookback_n: int = 10) -> pl.LazyFrame:
    """Compute fine-granularity direction (vectorized).

    Returns 'up' if >60% of last N bars are up, 'down' if <40%.
    """
    df = classify_bar_direction(df)
    return df.with_columns(
        (pl.col("direction") == "U")
        .cast(pl.Float64)
        .rolling_mean(window_size=lookback_n, min_samples=1)
        .alias("up_ratio")
    ).with_columns(
        pl.when(pl.col("up_ratio").is_null())
        .then(pl.lit("neutral"))
        .when(pl.col("up_ratio") > 0.6)
        .then(pl.lit("up"))
        .when(pl.col("up_ratio") < 0.4)
        .then(pl.lit("down"))
        .otherwise(pl.lit("neutral"))
        .alias("fine_dir")
    )


def align_htf_to_signal(
    signal_df: pl.LazyFrame,
    htf_df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Align HTF trend to signal bars using join_asof."""
    # Ensure timestamp column
    htf_with_trend = compute_htf_trend(htf_df, lookback=3).select([
        pl.col("timestamp"),
        pl.col("htf_trend"),
    ])

    return signal_df.join_asof(
        htf_with_trend,
        on="timestamp",
        strategy="backward",
    ).with_columns(
        pl.col("htf_trend").fill_null("neutral")
    )


def align_fine_to_signal(
    signal_df: pl.LazyFrame,
    fine_df: pl.LazyFrame,
    lookback_n: int = 10,
) -> pl.LazyFrame:
    """Align fine direction to signal bars using join_asof."""
    fine_with_dir = compute_fine_direction(fine_df, lookback_n).select([
        pl.col("timestamp"),
        pl.col("fine_dir"),
    ])

    return signal_df.join_asof(
        fine_with_dir,
        on="timestamp",
        strategy="backward",
    ).with_columns(
        pl.col("fine_dir").fill_null("neutral")
    )


# =============================================================================
# ODD Robustness Testing (Polars)
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


def check_multifactor_odd_robustness(
    df: pl.DataFrame,
    min_samples: int = 50,
    min_t_stat: float = 3.0,
) -> dict[str, dict]:
    """Check ODD robustness for multi-factor patterns."""
    results = {}

    # Create combined multi-factor label
    df = df.with_columns(
        (
            pl.col("signal_pattern").cast(pl.Utf8) + "|" +
            pl.col("htf_trend").cast(pl.Utf8) + "|" +
            pl.col("fine_dir").cast(pl.Utf8)
        ).alias("multifactor")
    )

    # Add period column (quarterly)
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    # Get unique multi-factor patterns
    patterns = df.select("multifactor").unique().to_series().to_list()

    for mf_pattern in patterns:
        if mf_pattern is None or "null" in str(mf_pattern).lower():
            continue

        subset = df.filter(pl.col("multifactor") == mf_pattern)

        # Group by period and compute stats
        period_stats = []
        for period_label in subset.select("period").unique().to_series().to_list():
            if period_label is None:
                continue

            period_data = subset.filter(pl.col("period") == period_label)
            returns = period_data.select("fwd_ret_1").drop_nulls().to_series()
            count = len(returns)

            if count < min_samples:
                continue

            mean_ret = returns.mean()
            t_stat = compute_t_stat(returns)

            period_stats.append({
                "period": period_label,
                "count": count,
                "mean_return": float(mean_ret) if mean_ret is not None else 0.0,
                "t_stat": t_stat,
            })

        if len(period_stats) < 2:
            continue

        # Check ODD criteria
        t_stats = [s["t_stat"] for s in period_stats]
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)
        same_sign = all(t > 0 for t in t_stats) or all(t < 0 for t in t_stats)

        is_odd_robust = all_significant and same_sign

        # Parse pattern components
        parts = mf_pattern.split("|")
        signal_pat = parts[0] if len(parts) > 0 else ""
        htf_trend = parts[1] if len(parts) > 1 else ""
        fine_dir = parts[2] if len(parts) > 2 else ""

        total_count = sum(s["count"] for s in period_stats)
        avg_return = (
            sum(s["mean_return"] * s["count"] for s in period_stats) / total_count
            if total_count > 0 else 0
        )

        results[mf_pattern] = {
            "multifactor_pattern": mf_pattern,
            "signal_pattern": signal_pat,
            "htf_trend": htf_trend,
            "fine_direction": fine_dir,
            "is_odd_robust": is_odd_robust,
            "n_periods": len(period_stats),
            "period_stats": period_stats,
            "all_significant": all_significant,
            "same_sign": same_sign,
            "min_t_stat": min(abs(t) for t in t_stats),
            "max_t_stat": max(abs(t) for t in t_stats),
            "total_count": total_count,
            "avg_return": avg_return,
        }

    return results


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

    # Load as pandas then convert (rangebar returns pandas)
    pdf = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    # Reset index to get timestamp as column
    pdf = pdf.reset_index()
    pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})

    # Convert to Polars LazyFrame
    return pl.from_pandas(pdf).lazy()


# =============================================================================
# Main Analysis
# =============================================================================


def run_multifactor_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run multi-factor pattern analysis for a symbol."""
    log_info("Starting multi-factor analysis", symbol=symbol)

    # Load all three timeframes
    log_info("Loading 100 dbps (signal)")
    bars_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    bars_100_count = bars_100.select(pl.len()).collect().item()

    log_info("Loading 50 dbps (fine)")
    bars_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    bars_50_count = bars_50.select(pl.len()).collect().item()

    log_info("Loading 200 dbps (HTF)")
    bars_200 = load_range_bars_polars(symbol, 200, start_date, end_date)
    bars_200_count = bars_200.select(pl.len()).collect().item()

    log_info(
        "Data loaded",
        bars_100=bars_100_count,
        bars_50=bars_50_count,
        bars_200=bars_200_count,
    )

    if bars_100_count < 1000:
        log_warn("Insufficient 100 dbps data", count=bars_100_count)
        return {"error": "insufficient_data"}

    # Prepare signal bars (100 dbps)
    log_info("Computing signal patterns")
    signal_bars = classify_bar_direction(bars_100)

    # Add 2-bar pattern (shift previous direction)
    signal_bars = signal_bars.with_columns(
        (pl.col("direction").shift(1) + pl.col("direction")).alias("signal_pattern")
    )

    # Add forward returns
    signal_bars = signal_bars.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
    )

    # Compute and align HTF trend from 200 dbps
    log_info("Aligning HTF trend from 200 dbps")
    if bars_200_count > 10:
        signal_bars = align_htf_to_signal(signal_bars, bars_200)
    else:
        signal_bars = signal_bars.with_columns(
            pl.lit("neutral").alias("htf_trend")
        )

    # Compute and align fine direction from 50 dbps
    log_info("Aligning fine direction from 50 dbps")
    if bars_50_count > 100:
        signal_bars = align_fine_to_signal(signal_bars, bars_50, lookback_n=10)
    else:
        signal_bars = signal_bars.with_columns(
            pl.lit("neutral").alias("fine_dir")
        )

    # Collect to DataFrame for analysis
    log_info("Collecting results")
    signal_df = signal_bars.collect()

    # Distribution check
    htf_dist = signal_df.group_by("htf_trend").len().to_dicts()
    fine_dist = signal_df.group_by("fine_dir").len().to_dicts()
    log_info("HTF trend distribution", distribution=htf_dist)
    log_info("Fine direction distribution", distribution=fine_dist)

    # Test ODD robustness
    log_info("Testing multi-factor ODD robustness (t-stat >= 3)")
    results = check_multifactor_odd_robustness(
        signal_df,
        min_samples=50,
        min_t_stat=3.0,
    )

    # Filter to ODD robust patterns
    robust_patterns = {k: v for k, v in results.items() if v["is_odd_robust"]}

    log_info(
        "Analysis complete",
        symbol=symbol,
        total_patterns=len(results),
        robust_patterns=len(robust_patterns),
    )

    # Log robust patterns
    for pattern, stats in robust_patterns.items():
        log_info(
            "ODD robust multi-factor pattern",
            pattern=pattern,
            signal=stats["signal_pattern"],
            htf=stats["htf_trend"],
            fine=stats["fine_direction"],
            n_periods=stats["n_periods"],
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    return {
        "symbol": symbol,
        "bars_100": bars_100_count,
        "bars_50": bars_50_count,
        "bars_200": bars_200_count,
        "total_patterns": len(results),
        "robust_patterns_count": len(robust_patterns),
        "robust_patterns": list(robust_patterns.keys()),
        "results": results,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting multi-factor pattern analysis (Polars)",
        script="multifactor_patterns_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}
    cross_symbol_robust: list[dict] = []

    for symbol in symbols:
        try:
            results = run_multifactor_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[symbol] = results

            if "robust_patterns" in results:
                for pattern in results["robust_patterns"]:
                    cross_symbol_robust.append({
                        "symbol": symbol,
                        "pattern": pattern,
                    })

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )

    # Cross-symbol validation
    log_info("=== CROSS-SYMBOL VALIDATION ===")

    pattern_counts = Counter(p["pattern"] for p in cross_symbol_robust)
    universal_patterns = [p for p, count in pattern_counts.items() if count >= len(symbols)]

    log_info(
        "Universal ODD robust patterns (all 4 symbols)",
        count=len(universal_patterns),
        patterns=universal_patterns,
    )

    # Patterns present in at least 3 symbols
    strong_patterns = [p for p, count in pattern_counts.items() if count >= 3]
    log_info(
        "Strong patterns (3+ symbols)",
        count=len(strong_patterns),
        patterns=strong_patterns,
    )

    # Save results
    output_path = Path("/tmp/multifactor_patterns_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Multi-factor pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
