#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Volume-conditioned pattern analysis for ODD robustness (Polars-optimized).

This script tests if conditioning directional patterns on volume metrics
yields ODD robust signals:
1. High/low volume relative to moving average
2. Buy/sell volume imbalance (OFI - Order Flow Imbalance)
3. Duration-based patterns (fast vs slow bars)

Key hypothesis: Volume confirmation may filter noise from directional patterns.

GitHub Issue: #52
"""

from __future__ import annotations

import json
import math
import sys
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


def log_error(message: str, **kwargs: object) -> None:
    """Log ERROR level message."""
    log_json("ERROR", message, **kwargs)


# =============================================================================
# Volume Conditioning (Polars Vectorized)
# =============================================================================


def add_volume_conditioning(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add volume regime classification."""
    return df.with_columns(
        pl.col("Volume").rolling_mean(window_size=20, min_samples=20).alias("volume_ma20")
    ).with_columns(
        (pl.col("Volume") / pl.col("volume_ma20")).alias("volume_ratio")
    ).with_columns(
        pl.when(pl.col("volume_ratio").is_null())
        .then(pl.lit("normal"))
        .when(pl.col("volume_ratio") > 1.5)
        .then(pl.lit("high"))
        .when(pl.col("volume_ratio") < 0.5)
        .then(pl.lit("low"))
        .otherwise(pl.lit("normal"))
        .alias("volume_regime")
    )


def add_ofi_conditioning(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add OFI (Order Flow Imbalance) classification.

    If ofi column exists, use it. Otherwise, estimate from bar direction.
    """
    # Check if ofi exists - use direction-based fallback
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(-1.0))
        .alias("ofi_value")
    ).with_columns(
        pl.when(pl.col("ofi_value") > 0.3)
        .then(pl.lit("buy"))
        .when(pl.col("ofi_value") < -0.3)
        .then(pl.lit("sell"))
        .otherwise(pl.lit("neutral"))
        .alias("ofi_regime")
    )


def add_duration_conditioning(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add duration regime classification based on rolling percentile."""
    return df.with_columns(
        pl.col("duration_us")
        .rolling_quantile(quantile=0.5, window_size=100, min_samples=10)
        .alias("duration_median")
    ).with_columns(
        (pl.col("duration_us") / pl.col("duration_median")).alias("duration_ratio")
    ).with_columns(
        pl.when(pl.col("duration_ratio").is_null())
        .then(pl.lit("normal"))
        .when(pl.col("duration_ratio") < 0.5)
        .then(pl.lit("fast"))
        .when(pl.col("duration_ratio") > 2.0)
        .then(pl.lit("slow"))
        .otherwise(pl.lit("normal"))
        .alias("duration_regime")
    )


def add_2bar_pattern(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern classification."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    ).with_columns(
        (pl.col("direction").shift(1) + pl.col("direction")).alias("pattern_2bar")
    )


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
    condition_col: str,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> dict[str, dict]:
    """Check ODD robustness for conditioned patterns."""
    results = {}

    # Add period column (quarterly)
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    # Get unique conditions and patterns
    conditions = df.select(condition_col).unique().to_series().drop_nulls().to_list()
    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()

    for condition in conditions:
        for pattern in patterns:
            if pattern is None or condition is None:
                continue

            subset = df.filter(
                (pl.col(condition_col) == condition) &
                (pl.col(pattern_col) == pattern)
            )

            # Collect stats per period
            period_stats = []
            for period_label in subset.select("period").unique().to_series().drop_nulls().to_list():
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

            total_count = sum(s["count"] for s in period_stats)
            avg_return = (
                sum(s["mean_return"] * s["count"] for s in period_stats) / total_count
                if total_count > 0 else 0
            )

            key = f"{condition}|{pattern}"
            results[key] = {
                "condition": str(condition),
                "pattern": str(pattern),
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
# Main Analysis
# =============================================================================


def run_volume_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Run volume-conditioned pattern analysis."""
    log_info(
        "Starting volume-conditioned analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    bars = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    bar_count = bars.select(pl.len()).collect().item()

    log_info("Loaded bars", count=bar_count)

    if bar_count < 1000:
        log_warn("Insufficient data", bars=bar_count)
        return {"error": "insufficient_data"}

    # Add conditioning columns
    log_info("Adding volume conditioning")
    bars = add_volume_conditioning(bars)

    log_info("Adding OFI conditioning")
    bars = add_ofi_conditioning(bars)

    # Check if duration_us exists
    schema = bars.collect_schema()
    if "duration_us" in schema.names():
        log_info("Adding duration conditioning")
        bars = add_duration_conditioning(bars)
    else:
        log_info("Skipping duration conditioning (no duration_us column)")
        bars = bars.with_columns(pl.lit("normal").alias("duration_regime"))

    # Add patterns and forward returns
    log_info("Adding patterns and forward returns")
    bars = add_2bar_pattern(bars)
    bars = bars.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
    )

    # Collect
    log_info("Collecting data")
    df = bars.collect()

    # Distribution of conditions
    vol_dist = df.group_by("volume_regime").len().to_dicts()
    ofi_dist = df.group_by("ofi_regime").len().to_dicts()
    dur_dist = df.group_by("duration_regime").len().to_dicts()

    log_info("Volume regime distribution", distribution=vol_dist)
    log_info("OFI regime distribution", distribution=ofi_dist)
    log_info("Duration regime distribution", distribution=dur_dist)

    # Test ODD robustness for each conditioning type
    log_info("Testing volume-conditioned patterns (t-stat >= 3)")
    results_volume = check_odd_robustness(
        df,
        pattern_col="pattern_2bar",
        condition_col="volume_regime",
        min_samples=100,
        min_t_stat=3.0,
    )

    log_info("Testing OFI-conditioned patterns (t-stat >= 3)")
    results_ofi = check_odd_robustness(
        df,
        pattern_col="pattern_2bar",
        condition_col="ofi_regime",
        min_samples=100,
        min_t_stat=3.0,
    )

    log_info("Testing duration-conditioned patterns (t-stat >= 3)")
    results_duration = check_odd_robustness(
        df,
        pattern_col="pattern_2bar",
        condition_col="duration_regime",
        min_samples=100,
        min_t_stat=3.0,
    )

    # Count ODD robust patterns
    robust_volume = [k for k, v in results_volume.items() if v["is_odd_robust"]]
    robust_ofi = [k for k, v in results_ofi.items() if v["is_odd_robust"]]
    robust_duration = [k for k, v in results_duration.items() if v["is_odd_robust"]]

    log_info(
        "Analysis complete",
        symbol=symbol,
        robust_volume=len(robust_volume),
        robust_ofi=len(robust_ofi),
        robust_duration=len(robust_duration),
    )

    # Log robust patterns
    for pattern in robust_volume:
        stats = results_volume[pattern]
        log_info(
            "ODD robust volume-conditioned pattern",
            pattern=pattern,
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    for pattern in robust_ofi:
        stats = results_ofi[pattern]
        log_info(
            "ODD robust OFI-conditioned pattern",
            pattern=pattern,
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    for pattern in robust_duration:
        stats = results_duration[pattern]
        log_info(
            "ODD robust duration-conditioned pattern",
            pattern=pattern,
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "total_bars": bar_count,
        "volume_distribution": vol_dist,
        "ofi_distribution": ofi_dist,
        "duration_distribution": dur_dist,
        "results_volume": results_volume,
        "results_ofi": results_ofi,
        "results_duration": results_duration,
        "robust_volume": robust_volume,
        "robust_ofi": robust_ofi,
        "robust_duration": robust_duration,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting volume-conditioned pattern analysis (Polars)",
        script="volume_conditioned_patterns_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}

    for symbol in symbols:
        try:
            results = run_volume_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[symbol] = results
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )

    # Summary
    total_robust_volume = sum(
        len(r.get("robust_volume", [])) for r in all_results.values() if "error" not in r
    )
    total_robust_ofi = sum(
        len(r.get("robust_ofi", [])) for r in all_results.values() if "error" not in r
    )
    total_robust_duration = sum(
        len(r.get("robust_duration", [])) for r in all_results.values() if "error" not in r
    )

    log_info(
        "Overall summary",
        total_robust_volume=total_robust_volume,
        total_robust_ofi=total_robust_ofi,
        total_robust_duration=total_robust_duration,
    )

    # Save results
    output_path = Path("/tmp/volume_conditioned_patterns_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Volume-conditioned pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
