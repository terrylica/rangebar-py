#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Volatility regime pattern analysis for OOD robust patterns (Polars).

Tests if volatility-based regime filters (ATR, realized volatility) reveal
OOD robust patterns that complement or differ from SMA/RSI regime patterns.

GitHub Issue: #54 (volatility regime research)
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


def compute_true_range(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute True Range for ATR calculation."""
    return df.with_columns(
        pl.max_horizontal(
            pl.col("High") - pl.col("Low"),
            (pl.col("High") - pl.col("Close").shift(1)).abs(),
            (pl.col("Low") - pl.col("Close").shift(1)).abs(),
        ).alias("true_range")
    )


def compute_atr(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Average True Range."""
    df = compute_true_range(df)
    return df.with_columns(
        pl.col("true_range").rolling_mean(window_size=window, min_samples=window).alias(alias)
    ).drop("true_range")


def compute_realized_volatility(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Realized Volatility (std of log returns)."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=window, min_samples=window).alias(alias)
    ).drop("_log_ret")


def classify_volatility_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify volatility regimes using ATR and realized volatility."""
    # ATR regime: compare ATR(14) to its SMA(50)
    df = df.with_columns(
        pl.col("atr").rolling_mean(window_size=50, min_samples=50).alias("atr_sma")
    )

    df = df.with_columns(
        (pl.col("atr") / pl.col("atr_sma")).alias("atr_ratio")
    )

    df = df.with_columns(
        pl.when(pl.col("atr_ratio") < 0.7)
        .then(pl.lit("low_vol"))
        .when(pl.col("atr_ratio") > 1.3)
        .then(pl.lit("high_vol"))
        .otherwise(pl.lit("normal_vol"))
        .alias("vol_regime")
    )

    # RV regime: use rolling percentiles
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

    # Combined regime
    df = df.with_columns(
        (pl.col("vol_regime") + "_" + pl.col("rv_regime")).alias("combined_vol_regime")
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
    """Count OOD robust patterns within each volatility regime."""
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


def run_volatility_regime_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run volatility regime pattern analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Add volatility indicators
    df = compute_atr(df, 14, "atr")
    df = compute_realized_volatility(df, 20, "rv")
    df = classify_volatility_regimes(df)

    # Add forward returns
    df = df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    result_df = df.collect()
    n_bars = len(result_df)

    log_info(f"Loaded {n_bars} bars", symbol=symbol)

    # Vol regime distribution
    vol_regime_dist = result_df.group_by("vol_regime").len().sort("len", descending=True)
    vol_regime_dict = {row["vol_regime"]: row["len"] for row in vol_regime_dist.iter_rows(named=True)}

    # RV regime distribution
    rv_regime_dist = result_df.group_by("rv_regime").len().sort("len", descending=True)
    rv_regime_dict = {row["rv_regime"]: row["len"] for row in rv_regime_dist.iter_rows(named=True)}

    # Count OOD robust patterns for each regime type
    clean_df = result_df.drop_nulls(subset=["pattern_2bar", "vol_regime", f"fwd_ret_{horizon}"])

    robust_vol = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "vol_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    robust_rv = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "rv_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    robust_combined = count_ood_robust_patterns(
        clean_df,
        "pattern_2bar",
        "combined_vol_regime",
        f"fwd_ret_{horizon}",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    log_info(
        f"{symbol} results",
        n_bars=n_bars,
        robust_vol=robust_vol["count"],
        robust_rv=robust_rv["count"],
        robust_combined=robust_combined["count"],
    )

    return {
        "symbol": symbol,
        "n_bars": n_bars,
        "vol_regime_distribution": vol_regime_dict,
        "rv_regime_distribution": rv_regime_dict,
        "robust_vol": robust_vol,
        "robust_rv": robust_rv,
        "robust_combined": robust_combined,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting volatility regime pattern analysis (Polars)",
        script="volatility_regime_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10
    min_samples = 100
    min_t_stat = 5.0

    log_info("=== VOLATILITY REGIME PATTERN ANALYSIS ===")
    log_info(
        "Parameters",
        horizon=horizon,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    all_results = {}

    for symbol in symbols:
        try:
            result = run_volatility_regime_analysis(
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

    # Count universal patterns (appear in all symbols)
    vol_pattern_counts = Counter()
    rv_pattern_counts = Counter()
    combined_pattern_counts = Counter()

    for _symbol, result in all_results.items():
        for pattern in result.get("robust_vol", {}).get("robust_patterns", []):
            vol_pattern_counts[pattern] += 1
        for pattern in result.get("robust_rv", {}).get("robust_patterns", []):
            rv_pattern_counts[pattern] += 1
        for pattern in result.get("robust_combined", {}).get("robust_patterns", []):
            combined_pattern_counts[pattern] += 1

    universal_vol = [p for p, c in vol_pattern_counts.items() if c >= len(symbols)]
    universal_rv = [p for p, c in rv_pattern_counts.items() if c >= len(symbols)]
    universal_combined = [p for p, c in combined_pattern_counts.items() if c >= len(symbols)]

    log_info(
        "Universal ATR-based vol regime patterns (all 4 symbols)",
        count=len(universal_vol),
        patterns=universal_vol,
    )

    log_info(
        "Universal RV regime patterns (all 4 symbols)",
        count=len(universal_rv),
        patterns=universal_rv,
    )

    log_info(
        "Universal combined vol regime patterns (all 4 symbols)",
        count=len(universal_combined),
        patterns=universal_combined,
    )

    # Comparison with SMA/RSI patterns (from Issue #52)
    sma_rsi_patterns = [
        "chop|DU", "chop|DD", "chop|UU", "chop|UD",
        "bear_neutral|DU", "bear_neutral|DD", "bear_neutral|UU", "bear_neutral|UD",
        "bull_neutral|DU", "bull_neutral|DD", "bull_neutral|UD",
    ]

    # Extract pattern names (without regime) for comparison
    vol_pattern_names = {p.split("|")[1] for p in universal_vol}
    rv_pattern_names = {p.split("|")[1] for p in universal_rv}
    sma_rsi_pattern_names = {p.split("|")[1] for p in sma_rsi_patterns}

    log_info("=== COMPARISON WITH SMA/RSI PATTERNS ===")
    log_info(
        "SMA/RSI patterns (Issue #52)",
        count=len(sma_rsi_patterns),
        pattern_types=list(sma_rsi_pattern_names),
    )
    log_info(
        "Vol regime patterns",
        count=len(universal_vol),
        pattern_types=list(vol_pattern_names),
    )
    log_info(
        "RV regime patterns",
        count=len(universal_rv),
        pattern_types=list(rv_pattern_names),
    )

    # Per-symbol summary
    log_info("=== PER-SYMBOL RESULTS ===")
    for symbol, result in all_results.items():
        log_info(
            symbol,
            n_bars=result.get("n_bars", 0),
            robust_vol=result.get("robust_vol", {}).get("count", 0),
            robust_rv=result.get("robust_rv", {}).get("count", 0),
            robust_combined=result.get("robust_combined", {}).get("count", 0),
        )

    # Save results
    output_path = Path("/tmp/volatility_regime_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "universal_vol": universal_vol,
            "universal_rv": universal_rv,
            "universal_combined": universal_combined,
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Volatility regime pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
