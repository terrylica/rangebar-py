#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Combined SMA/RSI x RV regime pattern analysis for OOD robust patterns (Polars).

Tests if combining SMA/RSI regimes (trend/momentum) with RV regimes (volatility)
produces more refined OOD robust patterns than either alone.

GitHub Issue: #54 (extension - combined regimes)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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


def compute_sma(df: pl.LazyFrame, column: str, window: int, alias: str) -> pl.LazyFrame:
    """Compute Simple Moving Average."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=window, min_samples=window).alias(alias)
    )


def compute_rsi(df: pl.LazyFrame, column: str, window: int, alias: str) -> pl.LazyFrame:
    """Compute Relative Strength Index using EWM."""
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


def compute_realized_volatility(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Realized Volatility (std of log returns)."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=window, min_samples=window).alias(alias)
    ).drop("_log_ret")


def classify_sma_rsi_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify market regimes using SMA 20/50 and RSI 14."""
    df = df.with_columns(
        pl.when(
            (pl.col("Close") > pl.col("sma_fast")) & (pl.col("sma_fast") > pl.col("sma_slow"))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("Close") < pl.col("sma_fast")) & (pl.col("sma_fast") < pl.col("sma_slow"))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("consolidation"))
        .alias("sma_regime")
    )

    df = df.with_columns(
        pl.when(pl.col("rsi") > 70)
        .then(pl.lit("overbought"))
        .when(pl.col("rsi") < 30)
        .then(pl.lit("oversold"))
        .otherwise(pl.lit("neutral"))
        .alias("rsi_regime")
    )

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
        .alias("trend_regime")
    )

    return df


def classify_rv_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify volatility regimes using realized volatility."""
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
        .alias("vol_regime")
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
    pattern_stats = []

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
            total_samples = 0
            for period_label in periods:
                period_data = pattern_subset.filter(pl.col("period") == period_label)
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

            # Compute overall statistics
            all_returns = pattern_subset.select(return_col).drop_nulls().to_series()
            mean_return = float(all_returns.mean()) if all_returns.mean() is not None else 0.0
            overall_t = compute_t_stat(all_returns)

            pattern_stats.append({
                "regime": regime,
                "pattern": pattern,
                "n_periods": len(period_stats),
                "n_samples": total_samples,
                "mean_return_bps": round(mean_return * 10000, 2),
                "overall_t": round(overall_t, 2),
                "min_t": round(min(period_stats), 2),
                "max_t": round(max(period_stats), 2),
                "is_robust": all_significant and same_sign,
            })

            if all_significant and same_sign:
                robust_patterns.append(f"{regime}|{pattern}")

    return {
        "robust_patterns": robust_patterns,
        "count": len(robust_patterns),
        "pattern_stats": pattern_stats,
    }


# =============================================================================
# Main Analysis
# =============================================================================


def run_combined_regime_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run combined SMA/RSI x RV regime pattern analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Add SMA/RSI indicators and regimes
    df = compute_sma(df, "Close", 20, "sma_fast")
    df = compute_sma(df, "Close", 50, "sma_slow")
    df = compute_rsi(df, "Close", 14, "rsi")
    df = classify_sma_rsi_regimes(df)

    # Add RV indicator and regime
    df = compute_realized_volatility(df, 20, "rv")
    df = classify_rv_regimes(df)

    # Combined regime
    df = df.with_columns(
        (pl.col("trend_regime") + "_" + pl.col("vol_regime")).alias("combined_regime")
    )

    # Add forward returns
    df = df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    result_df = df.collect()
    n_bars = len(result_df)

    log_info(f"Loaded {n_bars} bars", symbol=symbol)

    # Combined regime distribution
    combined_dist = result_df.group_by("combined_regime").len().sort("len", descending=True)
    combined_dict = {row["combined_regime"]: row["len"] for row in combined_dist.iter_rows(named=True)}

    # Count OOD robust patterns for combined regime
    clean_df = result_df.drop_nulls(subset=["pattern_2bar", "combined_regime", f"fwd_ret_{horizon}"])

    robust_combined = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "combined_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Also test trend-only and vol-only for comparison
    robust_trend = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "trend_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    robust_vol = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "vol_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    log_info(
        f"{symbol} results",
        n_bars=n_bars,
        robust_trend=robust_trend["count"],
        robust_vol=robust_vol["count"],
        robust_combined=robust_combined["count"],
    )

    return {
        "symbol": symbol,
        "n_bars": n_bars,
        "combined_regime_distribution": combined_dict,
        "robust_trend": robust_trend,
        "robust_vol": robust_vol,
        "robust_combined": robust_combined,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting combined SMA/RSI x RV regime pattern analysis (Polars)",
        script="combined_regime_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10
    min_samples = 100
    min_t_stat = 5.0

    log_info("=== COMBINED SMA/RSI x RV REGIME PATTERN ANALYSIS ===")
    log_info(
        "Parameters",
        horizon=horizon,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    all_results = {}

    for symbol in symbols:
        try:
            result = run_combined_regime_analysis(
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
    log_info("=== CROSS-SYMBOL SUMMARY ===")

    # Count universal patterns
    trend_pattern_counts = Counter()
    vol_pattern_counts = Counter()
    combined_pattern_counts = Counter()

    for _symbol, result in all_results.items():
        for pattern in result.get("robust_trend", {}).get("robust_patterns", []):
            trend_pattern_counts[pattern] += 1
        for pattern in result.get("robust_vol", {}).get("robust_patterns", []):
            vol_pattern_counts[pattern] += 1
        for pattern in result.get("robust_combined", {}).get("robust_patterns", []):
            combined_pattern_counts[pattern] += 1

    universal_trend = [p for p, c in trend_pattern_counts.items() if c >= len(symbols)]
    universal_vol = [p for p, c in vol_pattern_counts.items() if c >= len(symbols)]
    universal_combined = [p for p, c in combined_pattern_counts.items() if c >= len(symbols)]

    log_info(
        "Universal SMA/RSI regime patterns (all 4 symbols)",
        count=len(universal_trend),
        patterns=sorted(universal_trend),
    )

    log_info(
        "Universal RV regime patterns (all 4 symbols)",
        count=len(universal_vol),
        patterns=sorted(universal_vol),
    )

    log_info(
        "Universal combined regime patterns (all 4 symbols)",
        count=len(universal_combined),
        patterns=sorted(universal_combined),
    )

    # Analyze combined regime breakdown
    log_info("=== COMBINED REGIME BREAKDOWN ===")

    # Group combined patterns by trend regime
    trend_groups = {}
    for pattern in universal_combined:
        parts = pattern.split("|")
        regime = parts[0]
        bar_pattern = parts[1]
        trend_part = regime.rsplit("_", 1)[0]  # e.g., "chop_quiet" -> "chop"
        vol_part = regime.rsplit("_", 1)[1]  # e.g., "chop_quiet" -> "quiet"

        key = f"{trend_part}"
        if key not in trend_groups:
            trend_groups[key] = []
        trend_groups[key].append(f"{vol_part}|{bar_pattern}")

    for trend, patterns in sorted(trend_groups.items()):
        log_info(
            f"Trend regime: {trend}",
            n_combined_patterns=len(patterns),
            patterns=sorted(patterns),
        )

    # Group by vol regime
    vol_groups = {}
    for pattern in universal_combined:
        parts = pattern.split("|")
        regime = parts[0]
        bar_pattern = parts[1]
        trend_part = regime.rsplit("_", 1)[0]
        vol_part = regime.rsplit("_", 1)[1]

        key = f"{vol_part}"
        if key not in vol_groups:
            vol_groups[key] = []
        vol_groups[key].append(f"{trend_part}|{bar_pattern}")

    for vol, patterns in sorted(vol_groups.items()):
        log_info(
            f"Vol regime: {vol}",
            n_combined_patterns=len(patterns),
            patterns=sorted(patterns),
        )

    # Compare pattern counts
    log_info("=== PATTERN COUNT COMPARISON ===")
    log_info(
        "Comparison",
        trend_only=len(universal_trend),
        vol_only=len(universal_vol),
        combined=len(universal_combined),
        theoretical_max_combined=len(universal_trend) * 3,  # 3 vol regimes per trend
    )

    # Per-symbol summary
    log_info("=== PER-SYMBOL RESULTS ===")
    for symbol, result in all_results.items():
        log_info(
            symbol,
            n_bars=result.get("n_bars", 0),
            robust_trend=result.get("robust_trend", {}).get("count", 0),
            robust_vol=result.get("robust_vol", {}).get("count", 0),
            robust_combined=result.get("robust_combined", {}).get("count", 0),
        )

    # Save results
    output_path = Path("/tmp/combined_regime_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "universal_trend": universal_trend,
            "universal_vol": universal_vol,
            "universal_combined": universal_combined,
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Combined regime pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
