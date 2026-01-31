#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Market regime analysis for ODD robust multi-factor range bar patterns (Polars).

This script implements the methodology described in docs/research/market-regime-patterns.md.
It analyzes range bar patterns WITHIN specific market regimes to find ODD robust combinations.

Polars version for better performance and memory efficiency.

GitHub Issue: #52
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


def log_error(message: str, **kwargs: object) -> None:
    """Log ERROR level message."""
    log_json("ERROR", message, **kwargs)


# =============================================================================
# Technical Indicators (Polars Vectorized)
# =============================================================================


def compute_sma(df: pl.LazyFrame, column: str, window: int, alias: str) -> pl.LazyFrame:
    """Compute Simple Moving Average."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=window, min_samples=window).alias(alias)
    )


def compute_rsi(df: pl.LazyFrame, column: str, window: int = 14, alias: str = "rsi14") -> pl.LazyFrame:
    """Compute Relative Strength Index using EWM.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over window
    """
    return df.with_columns(
        pl.col(column).diff().alias("_delta")
    ).with_columns([
        pl.when(pl.col("_delta") > 0)
        .then(pl.col("_delta"))
        .otherwise(pl.lit(0.0))
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0)
        .then(-pl.col("_delta"))
        .otherwise(pl.lit(0.0))
        .alias("_loss"),
    ]).with_columns([
        pl.col("_gain").ewm_mean(alpha=1.0 / window, min_samples=window).alias("_avg_gain"),
        pl.col("_loss").ewm_mean(alpha=1.0 / window, min_samples=window).alias("_avg_loss"),
    ]).with_columns(
        (pl.col("_avg_gain") / pl.col("_avg_loss")).alias("_rs")
    ).with_columns(
        (100.0 - (100.0 / (1.0 + pl.col("_rs")))).alias(alias)
    ).drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss", "_rs"])


def classify_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify market regimes based on SMA crossovers and RSI levels."""
    # SMA regime: uptrend if Close > SMA20 > SMA50, downtrend if opposite
    df = df.with_columns(
        pl.when(
            (pl.col("Close") > pl.col("sma20")) & (pl.col("sma20") > pl.col("sma50"))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("Close") < pl.col("sma20")) & (pl.col("sma20") < pl.col("sma50"))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("consolidation"))
        .alias("sma_regime")
    )

    # RSI regime: overbought > 70, oversold < 30
    df = df.with_columns(
        pl.when(pl.col("rsi14") > 70)
        .then(pl.lit("overbought"))
        .when(pl.col("rsi14") < 30)
        .then(pl.lit("oversold"))
        .otherwise(pl.lit("neutral"))
        .alias("rsi_regime")
    )

    # Combined regime
    df = df.with_columns(
        pl.when(pl.col("sma_regime") == "consolidation")
        .then(pl.lit("chop"))
        .when(
            (pl.col("sma_regime") == "uptrend") & (pl.col("rsi_regime") == "overbought")
        )
        .then(pl.lit("bull_hot"))
        .when(
            (pl.col("sma_regime") == "uptrend") & (pl.col("rsi_regime") == "oversold")
        )
        .then(pl.lit("bull_cold"))
        .when(pl.col("sma_regime") == "uptrend")
        .then(pl.lit("bull_neutral"))
        .when(
            (pl.col("sma_regime") == "downtrend") & (pl.col("rsi_regime") == "overbought")
        )
        .then(pl.lit("bear_hot"))
        .when(
            (pl.col("sma_regime") == "downtrend") & (pl.col("rsi_regime") == "oversold")
        )
        .then(pl.lit("bear_cold"))
        .otherwise(pl.lit("bear_neutral"))
        .alias("regime")
    )

    return df


# =============================================================================
# Pattern Detection (Polars Vectorized)
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
    """Add 2-bar and 3-bar pattern columns."""
    return df.with_columns([
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar"),
        (
            pl.col("direction") +
            pl.col("direction").shift(-1) +
            pl.col("direction").shift(-2)
        ).alias("pattern_3bar"),
    ])


def add_forward_returns(df: pl.LazyFrame, horizons: list[int] | None = None) -> pl.LazyFrame:
    """Add forward returns at multiple horizons."""
    if horizons is None:
        horizons = [1, 3, 5]

    for h in horizons:
        df = df.with_columns(
            (pl.col("Close").shift(-h) / pl.col("Close") - 1).alias(f"fwd_ret_{h}")
        )

    return df


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
    regime_col: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict[str, dict]:
    """Check ODD robustness for patterns within regimes."""
    results = {}

    # Add period column (quarterly)
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    # Get unique regimes and patterns
    regimes = df.select(regime_col).unique().to_series().drop_nulls().to_list()
    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()

    for regime in regimes:
        regime_subset = df.filter(pl.col(regime_col) == regime)

        for pattern in patterns:
            if pattern is None:
                continue

            pattern_subset = regime_subset.filter(pl.col(pattern_col) == pattern)

            # Collect stats per period
            period_stats = []
            periods = pattern_subset.select("period").unique().to_series().drop_nulls().to_list()

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

            key = f"{regime}|{pattern}"
            results[key] = {
                "regime": str(regime),
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

    log_info(
        "Loading range bars",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        start_date=start_date,
        end_date=end_date,
    )

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

    bar_count = len(pdf)
    log_info("Range bars loaded", symbol=symbol, bars=bar_count)

    return pl.from_pandas(pdf).lazy()


# =============================================================================
# Main Analysis
# =============================================================================


def prepare_analysis_df(df: pl.LazyFrame) -> pl.LazyFrame:
    """Prepare LazyFrame for regime analysis."""
    # Technical indicators
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")

    # Regime classifications
    df = classify_regimes(df)

    # Patterns
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Forward returns
    df = add_forward_returns(df, horizons=[1, 3, 5])

    return df


def run_regime_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run full regime analysis for a symbol/threshold combination."""
    log_info(
        "Starting regime analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    # Load data
    df = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    bar_count = df.select(pl.len()).collect().item()

    if bar_count < 1000:
        log_warn("Insufficient data", symbol=symbol, bars=bar_count)
        return {"error": "insufficient_data", "bars": bar_count}

    # Prepare analysis DataFrame
    log_info("Computing technical indicators and patterns")
    df = prepare_analysis_df(df)

    # Collect to DataFrame for groupby operations
    log_info("Collecting data")
    result_df = df.collect()

    # Regime distribution
    regime_counts = result_df.group_by("regime").len().to_dicts()
    log_info("Regime distribution", distribution=regime_counts)

    # Test ODD robustness for 2-bar patterns
    log_info("Testing 2-bar pattern ODD robustness")
    results_2bar = check_odd_robustness(
        result_df,
        pattern_col="pattern_2bar",
        regime_col="regime",
        return_col="fwd_ret_1",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Test ODD robustness for 3-bar patterns
    log_info("Testing 3-bar pattern ODD robustness")
    results_3bar = check_odd_robustness(
        result_df,
        pattern_col="pattern_3bar",
        regime_col="regime",
        return_col="fwd_ret_1",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Count ODD robust patterns
    robust_2bar = [k for k, v in results_2bar.items() if v["is_odd_robust"]]
    robust_3bar = [k for k, v in results_3bar.items() if v["is_odd_robust"]]

    log_info(
        "Analysis complete",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        total_bars=bar_count,
        robust_2bar_patterns=len(robust_2bar),
        robust_3bar_patterns=len(robust_3bar),
    )

    # Log robust patterns
    for pattern in robust_2bar:
        stats = results_2bar[pattern]
        log_info(
            "ODD robust 2-bar pattern",
            pattern=pattern,
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    for pattern in robust_3bar:
        stats = results_3bar[pattern]
        log_info(
            "ODD robust 3-bar pattern",
            pattern=pattern,
            min_t=round(stats["min_t_stat"], 2),
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "start_date": start_date,
        "end_date": end_date,
        "total_bars": bar_count,
        "regime_counts": regime_counts,
        "results_2bar": results_2bar,
        "results_3bar": results_3bar,
        "robust_2bar": robust_2bar,
        "robust_3bar": robust_3bar,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting market regime analysis (Polars)",
        script="regime_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    thresholds = [50, 100]  # Primary signal granularities
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    min_samples = 100
    min_t_stat = 5.0

    # Run analysis for each combination
    all_results = {}
    all_robust_patterns: list[dict] = []

    for symbol in symbols:
        for threshold in thresholds:
            key = f"{symbol}@{threshold}"
            try:
                result = run_regime_analysis(
                    symbol=symbol,
                    threshold_dbps=threshold,
                    start_date=start_date,
                    end_date=end_date,
                    min_samples=min_samples,
                    min_t_stat=min_t_stat,
                )
                all_results[key] = result

                if "robust_2bar" in result:
                    for pattern in result["robust_2bar"]:
                        all_robust_patterns.append({
                            "symbol": symbol,
                            "threshold": threshold,
                            "pattern_type": "2bar",
                            "pattern": pattern,
                        })
                if "robust_3bar" in result:
                    for pattern in result["robust_3bar"]:
                        all_robust_patterns.append({
                            "symbol": symbol,
                            "threshold": threshold,
                            "pattern_type": "3bar",
                            "pattern": pattern,
                        })

            except (ValueError, RuntimeError, OSError, KeyError) as e:
                log_error(
                    "Analysis failed",
                    symbol=symbol,
                    threshold_dbps=threshold,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                all_results[key] = {"error": str(e), "error_type": type(e).__name__}

    # Cross-symbol validation
    log_info("=== CROSS-SYMBOL VALIDATION ===")
    log_info("Total robust patterns found", count=len(all_robust_patterns))

    # Find patterns that are robust across ALL symbols
    if all_robust_patterns:
        pattern_counts = Counter(
            (p["pattern_type"], p["pattern"]) for p in all_robust_patterns
        )
        universal_patterns = [
            {"type": p[0], "pattern": p[1]}
            for p, count in pattern_counts.items()
            if count >= len(symbols) * len(thresholds)
        ]
        log_info(
            "Universal ODD robust patterns (all symbols, all thresholds)",
            count=len(universal_patterns),
            patterns=universal_patterns,
        )

        # Also show patterns robust across all symbols for each threshold
        for threshold in thresholds:
            threshold_patterns = [
                p for p in all_robust_patterns if p["threshold"] == threshold
            ]
            threshold_counts = Counter(
                (p["pattern_type"], p["pattern"]) for p in threshold_patterns
            )
            universal_at_threshold = [
                {"type": p[0], "pattern": p[1]}
                for p, count in threshold_counts.items()
                if count >= len(symbols)
            ]
            log_info(
                f"Universal patterns at {threshold} dbps",
                count=len(universal_at_threshold),
                patterns=universal_at_threshold,
            )
    else:
        log_info("No ODD robust patterns found")

    # Save results
    output_path = Path("/tmp/regime_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Market regime analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
