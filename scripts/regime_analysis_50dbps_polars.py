#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Market regime pattern analysis at 50 dbps granularity (Polars).

Analyzes whether finer granularity (50 dbps vs 100 dbps) reveals different
or additional OOD robust patterns within market regimes.

GitHub Issue: #52 (regime research extension)
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


def classify_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
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
        .alias("regime")
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
    """Add 2-bar and 3-bar pattern columns."""
    return df.with_columns([
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar"),
        (pl.col("direction") + pl.col("direction").shift(-1) + pl.col("direction").shift(-2)).alias("pattern_3bar"),
    ])


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


def count_odd_robust_patterns(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Count ODD robust patterns within each regime."""
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    regimes = df.select("regime").unique().to_series().drop_nulls().to_list()
    robust_patterns = []
    pattern_stats = []

    for regime in regimes:
        if regime is None:
            continue

        regime_df = df.filter(pl.col("regime") == regime)
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


def run_50dbps_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run 50 dbps regime pattern analysis for a symbol."""
    log_info("Loading 50 dbps data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 50, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Add indicators
    df = compute_sma(df, "Close", 20, "sma_fast")
    df = compute_sma(df, "Close", 50, "sma_slow")
    df = compute_rsi(df, "Close", 14, "rsi")
    df = classify_regimes(df)

    # Add forward returns
    df = df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    result_df = df.collect()
    n_bars = len(result_df)

    log_info(f"Loaded {n_bars} bars at 50 dbps", symbol=symbol)

    # Regime distribution
    regime_dist = result_df.group_by("regime").len().sort("len", descending=True)
    regime_dict = {row["regime"]: row["len"] for row in regime_dist.iter_rows(named=True)}

    # Count 2-bar robust patterns
    clean_df = result_df.drop_nulls(subset=["pattern_2bar", "regime", f"fwd_ret_{horizon}"])
    robust_2bar = count_odd_robust_patterns(
        clean_df,
        "pattern_2bar",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Count 3-bar robust patterns
    clean_df_3bar = result_df.drop_nulls(subset=["pattern_3bar", "regime", f"fwd_ret_{horizon}"])
    robust_3bar = count_odd_robust_patterns(
        clean_df_3bar,
        "pattern_3bar",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    log_info(
        f"{symbol} 50 dbps results",
        n_bars=n_bars,
        robust_2bar=robust_2bar["count"],
        robust_3bar=robust_3bar["count"],
    )

    return {
        "symbol": symbol,
        "threshold_dbps": 50,
        "n_bars": n_bars,
        "regime_distribution": regime_dict,
        "robust_2bar": robust_2bar,
        "robust_3bar": robust_3bar,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting 50 dbps regime pattern analysis (Polars)",
        script="regime_analysis_50dbps_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10
    min_samples = 100
    min_t_stat = 5.0

    log_info("=== 50 DBPS REGIME PATTERN ANALYSIS ===")
    log_info(
        "Parameters",
        horizon=horizon,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    all_results = {}

    for symbol in symbols:
        try:
            result = run_50dbps_analysis(
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
    log_info("=== CROSS-SYMBOL SUMMARY (50 DBPS) ===")

    # Count universal patterns (appear in all symbols)
    pattern_2bar_counts = Counter()
    pattern_3bar_counts = Counter()

    for _symbol, result in all_results.items():
        for pattern in result.get("robust_2bar", {}).get("robust_patterns", []):
            pattern_2bar_counts[pattern] += 1
        for pattern in result.get("robust_3bar", {}).get("robust_patterns", []):
            pattern_3bar_counts[pattern] += 1

    universal_2bar = [p for p, c in pattern_2bar_counts.items() if c >= len(symbols)]
    universal_3bar = [p for p, c in pattern_3bar_counts.items() if c >= len(symbols)]

    log_info(
        "Universal 2-bar patterns (all 4 symbols)",
        count=len(universal_2bar),
        patterns=universal_2bar,
    )

    log_info(
        "Universal 3-bar patterns (all 4 symbols)",
        count=len(universal_3bar),
        patterns=universal_3bar,
    )

    # Comparison with 100 dbps (from prior research)
    known_100dbps_patterns = [
        "chop|DU", "chop|DD", "chop|UU", "chop|UD",
        "bear_neutral|DU", "bear_neutral|DD", "bear_neutral|UU", "bear_neutral|UD",
        "bull_neutral|DU", "bull_neutral|DD", "bull_neutral|UD",
    ]

    overlap = set(universal_2bar) & set(known_100dbps_patterns)
    new_patterns = set(universal_2bar) - set(known_100dbps_patterns)
    missing_patterns = set(known_100dbps_patterns) - set(universal_2bar)

    log_info("=== COMPARISON WITH 100 DBPS ===")
    log_info(
        "Pattern overlap",
        overlap_count=len(overlap),
        overlap=list(overlap),
    )
    log_info(
        "New patterns at 50 dbps (not in 100 dbps)",
        count=len(new_patterns),
        patterns=list(new_patterns),
    )
    log_info(
        "Missing patterns at 50 dbps (in 100 dbps but not 50)",
        count=len(missing_patterns),
        patterns=list(missing_patterns),
    )

    # Per-symbol summary
    log_info("=== PER-SYMBOL RESULTS ===")
    for symbol, result in all_results.items():
        log_info(
            symbol,
            n_bars=result.get("n_bars", 0),
            robust_2bar=result.get("robust_2bar", {}).get("count", 0),
            robust_3bar=result.get("robust_3bar", {}).get("count", 0),
        )

    # Save results
    output_path = Path("/tmp/regime_analysis_50dbps_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "universal_2bar": universal_2bar,
            "universal_3bar": universal_3bar,
            "comparison": {
                "overlap": list(overlap),
                "new_at_50dbps": list(new_patterns),
                "missing_at_50dbps": list(missing_patterns),
            },
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("50 dbps regime pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
