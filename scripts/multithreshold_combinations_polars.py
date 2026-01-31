#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-threshold range bar combination analysis (Polars).

Test COMBINATIONS of range bar patterns at different thresholds (50, 100, 200 dbps)
as multi-factor signals.

Hypothesis: When patterns at multiple granularities align, the combined signal
may be more robust than single-threshold patterns.

GitHub Issue: #52 (multi-factor extension)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

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

    log_info("Loading range bars", symbol=symbol, threshold_dbps=threshold_dbps,
             start=start_date, end=end_date)

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


def add_2bar_pattern(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


def add_trend_direction(df: pl.LazyFrame, lookback: int = 3) -> pl.LazyFrame:
    """Add trend direction based on majority of last N bars."""
    # Count up bars in lookback window
    up_count_expr = pl.lit(0)
    for i in range(lookback):
        up_count_expr = up_count_expr + pl.when(pl.col("direction").shift(i) == "U").then(pl.lit(1)).otherwise(pl.lit(0))

    return df.with_columns(
        up_count_expr.alias("_up_count")
    ).with_columns(
        pl.when(pl.col("_up_count") >= (lookback // 2 + 1))
        .then(pl.lit("up"))
        .when(pl.col("_up_count") <= (lookback // 2 - 1))
        .then(pl.lit("down"))
        .otherwise(pl.lit("mixed"))
        .alias("trend")
    ).drop("_up_count")


# =============================================================================
# ODD Robustness Testing
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


def check_odd_robustness(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Check ODD robustness for a pattern column."""
    # Add quarterly period
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()
    results = {}

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_subset = df.filter(pl.col(pattern_col) == pattern)
        periods = pattern_subset.select("period").unique().to_series().drop_nulls().to_list()

        period_stats = []
        for period_label in periods:
            period_data = pattern_subset.filter(pl.col("period") == period_label)
            returns = period_data.select(return_col).drop_nulls().to_series()
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
            results[pattern] = {
                "is_odd_robust": False,
                "reason": "insufficient_periods",
                "n_periods": len(period_stats),
            }
            continue

        t_stats = [s["t_stat"] for s in period_stats]
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)
        same_sign = all(t > 0 for t in t_stats) or all(t < 0 for t in t_stats)

        is_odd_robust = all_significant and same_sign

        total_count = sum(s["count"] for s in period_stats)
        avg_return = (
            sum(s["mean_return"] * s["count"] for s in period_stats) / total_count
            if total_count > 0 else 0
        )

        results[pattern] = {
            "is_odd_robust": is_odd_robust,
            "n_periods": len(period_stats),
            "all_significant": all_significant,
            "same_sign": same_sign,
            "min_t_stat": min(abs(t) for t in t_stats),
            "max_t_stat": max(abs(t) for t in t_stats),
            "total_count": total_count,
            "avg_return_bps": avg_return * 10000,
        }

    return results


# =============================================================================
# Main Analysis
# =============================================================================


def run_multithreshold_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run multi-threshold combination analysis for a symbol."""
    log_info("Starting multi-threshold analysis", symbol=symbol, horizon=horizon)

    # Load data at three granularities
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Add directions and patterns
    df_50 = add_bar_direction(df_50)
    df_50 = add_2bar_pattern(df_50)

    df_100 = add_bar_direction(df_100)
    df_100 = add_2bar_pattern(df_100)
    # Add forward returns to 100 dbps (primary signal)
    df_100 = df_100.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    df_200 = add_bar_direction(df_200)
    df_200 = add_trend_direction(df_200, lookback=3)

    # Collect with relevant columns
    df_50_c = df_50.select(["timestamp", "pattern_2bar"]).collect()
    df_100_c = df_100.select(["timestamp", "Close", "pattern_2bar", f"fwd_ret_{horizon}"]).collect()
    df_200_c = df_200.select(["timestamp", "trend"]).collect()

    log_info("Data loaded", symbol=symbol,
             bars_50=len(df_50_c), bars_100=len(df_100_c), bars_200=len(df_200_c))

    # Rename for clarity
    df_50_c = df_50_c.rename({"pattern_2bar": "pattern_50"})
    df_100_c = df_100_c.rename({"pattern_2bar": "pattern_100"})
    df_200_c = df_200_c.rename({"trend": "trend_200"})

    # Sort by timestamp for asof join
    df_50_c = df_50_c.sort("timestamp")
    df_100_c = df_100_c.sort("timestamp")
    df_200_c = df_200_c.sort("timestamp")

    # Join: for each 100 dbps bar, get most recent 50 dbps and 200 dbps patterns
    merged = df_100_c.join_asof(
        df_50_c,
        on="timestamp",
        strategy="backward",
    )

    merged = merged.join_asof(
        df_200_c,
        on="timestamp",
        strategy="backward",
    )

    # Create combined pattern codes
    merged = merged.with_columns([
        # Full combination: 50|100|200
        (
            pl.lit("50:") + pl.col("pattern_50") +
            pl.lit("|100:") + pl.col("pattern_100") +
            pl.lit("|200:") + pl.col("trend_200")
        ).alias("combo_full"),

        # Aligned up: all three show bullish
        pl.when(
            (pl.col("pattern_50").str.ends_with("U")) &
            (pl.col("pattern_100").str.ends_with("U")) &
            (pl.col("trend_200") == "up")
        )
        .then(pl.lit("aligned_up"))
        .when(
            (pl.col("pattern_50").str.ends_with("D")) &
            (pl.col("pattern_100").str.ends_with("D")) &
            (pl.col("trend_200") == "down")
        )
        .then(pl.lit("aligned_down"))
        .otherwise(pl.lit("mixed"))
        .alias("alignment"),

        # 50+200 combo (without 100)
        (
            pl.lit("50:") + pl.col("pattern_50") +
            pl.lit("|200:") + pl.col("trend_200")
        ).alias("combo_50_200"),
    ])

    log_info("Data merged", symbol=symbol, merged_rows=len(merged))

    # Test ODD robustness for different combination types
    return_col = f"fwd_ret_{horizon}"
    results = {
        "symbol": symbol,
        "horizon": horizon,
        "total_bars": len(merged),
    }

    # Test full combinations
    clean_df = merged.drop_nulls(subset=["combo_full", return_col])
    full_combo_results = check_odd_robustness(
        clean_df,
        "combo_full",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    robust_full = sum(1 for v in full_combo_results.values() if v.get("is_odd_robust", False))
    results["combo_full"] = {
        "total_patterns": len(full_combo_results),
        "robust_count": robust_full,
        "patterns": full_combo_results,
    }
    log_info("Full combinations tested", symbol=symbol,
             total=len(full_combo_results), robust=robust_full)

    # Test alignment signals
    alignment_results = check_odd_robustness(
        clean_df,
        "alignment",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    robust_alignment = sum(1 for v in alignment_results.values() if v.get("is_odd_robust", False))
    results["alignment"] = {
        "total_patterns": len(alignment_results),
        "robust_count": robust_alignment,
        "patterns": alignment_results,
    }
    log_info("Alignment signals tested", symbol=symbol,
             total=len(alignment_results), robust=robust_alignment)

    # Test 50+200 combinations
    clean_df_50_200 = merged.drop_nulls(subset=["combo_50_200", return_col])
    combo_50_200_results = check_odd_robustness(
        clean_df_50_200,
        "combo_50_200",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    robust_50_200 = sum(1 for v in combo_50_200_results.values() if v.get("is_odd_robust", False))
    results["combo_50_200"] = {
        "total_patterns": len(combo_50_200_results),
        "robust_count": robust_50_200,
        "patterns": combo_50_200_results,
    }
    log_info("50+200 combinations tested", symbol=symbol,
             total=len(combo_50_200_results), robust=robust_50_200)

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting multi-threshold combination analysis (Polars)",
        script="multithreshold_combinations_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10
    min_samples = 100
    min_t_stat = 5.0

    log_info("=== MULTI-THRESHOLD COMBINATION ANALYSIS ===")
    log_info("Testing 50|100|200 dbps pattern combinations")

    all_results = {}

    for symbol in symbols:
        try:
            result = run_multithreshold_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                min_samples=min_samples,
                min_t_stat=min_t_stat,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL MULTI-THRESHOLD SUMMARY ===")

    # Count patterns robust across all symbols
    for combo_type in ["combo_full", "alignment", "combo_50_200"]:
        pattern_counts = Counter()

        for _symbol, result in all_results.items():
            patterns = result.get(combo_type, {}).get("patterns", {})
            for pattern, data in patterns.items():
                if data.get("is_odd_robust", False):
                    pattern_counts[pattern] += 1

        universal = [p for p, c in pattern_counts.items() if c >= len(symbols)]
        log_info(
            f"Combination type: {combo_type}",
            type=combo_type,
            total_robust_instances=sum(pattern_counts.values()),
            universal_count=len(universal),
            universal_patterns=universal[:10] if universal else [],
        )

    # Final summary
    log_info("=== FINAL MULTI-THRESHOLD SUMMARY ===")

    total_universal_full = 0
    total_universal_alignment = 0
    total_universal_50_200 = 0

    for combo_type in ["combo_full", "alignment", "combo_50_200"]:
        pattern_counts = Counter()
        for _symbol, result in all_results.items():
            patterns = result.get(combo_type, {}).get("patterns", {})
            for pattern, data in patterns.items():
                if data.get("is_odd_robust", False):
                    pattern_counts[pattern] += 1

        universal_count = sum(1 for c in pattern_counts.values() if c >= len(symbols))

        if combo_type == "combo_full":
            total_universal_full = universal_count
        elif combo_type == "alignment":
            total_universal_alignment = universal_count
        elif combo_type == "combo_50_200":
            total_universal_50_200 = universal_count

    log_info(
        "Results",
        universal_full_combos=total_universal_full,
        universal_alignments=total_universal_alignment,
        universal_50_200_combos=total_universal_50_200,
    )

    # Save results
    output_path = Path("/tmp/multithreshold_combinations_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Multi-threshold combination analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
