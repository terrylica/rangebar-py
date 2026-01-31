#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-bar forward returns analysis for ODD robust regime patterns (Polars).

Research question: Do universal ODD robust patterns show predictive power
at LONGER horizons (3-bar, 5-bar returns)?

If patterns are ODD robust at multi-bar horizons = genuine predictive power
If only 1-bar is robust = mechanical/transient alpha

GitHub Issue: #52 (extension)
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
# Technical Indicators (Polars Vectorized)
# =============================================================================


def compute_sma(df: pl.LazyFrame, column: str, window: int, alias: str) -> pl.LazyFrame:
    """Compute Simple Moving Average."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=window, min_samples=window).alias(alias)
    )


def compute_rsi(df: pl.LazyFrame, column: str, window: int = 14, alias: str = "rsi14") -> pl.LazyFrame:
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
    """Classify market regimes based on SMA crossovers and RSI levels."""
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

    df = df.with_columns(
        pl.when(pl.col("rsi14") > 70)
        .then(pl.lit("overbought"))
        .when(pl.col("rsi14") < 30)
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


def add_forward_returns(df: pl.LazyFrame, horizons: list[int]) -> pl.LazyFrame:
    """Add forward returns at multiple horizons."""
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


def check_odd_robustness_multibar(
    df: pl.DataFrame,
    pattern_col: str,
    regime_col: str,
    horizons: list[int],
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict[str, dict]:
    """Check ODD robustness for patterns at multiple forward return horizons."""
    results = {}

    # Add period column (quarterly)
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    regimes = df.select(regime_col).unique().to_series().drop_nulls().to_list()
    patterns = df.select(pattern_col).unique().to_series().drop_nulls().to_list()

    for regime in regimes:
        regime_subset = df.filter(pl.col(regime_col) == regime)

        for pattern in patterns:
            if pattern is None:
                continue

            pattern_subset = regime_subset.filter(pl.col(pattern_col) == pattern)

            # Test each horizon
            horizon_results = {}
            for horizon in horizons:
                return_col = f"fwd_ret_{horizon}"

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
                    horizon_results[horizon] = {
                        "is_odd_robust": False,
                        "reason": "insufficient_periods",
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

                horizon_results[horizon] = {
                    "is_odd_robust": is_odd_robust,
                    "n_periods": len(period_stats),
                    "all_significant": all_significant,
                    "same_sign": same_sign,
                    "min_t_stat": min(abs(t) for t in t_stats),
                    "max_t_stat": max(abs(t) for t in t_stats),
                    "total_count": total_count,
                    "avg_return_bps": avg_return * 10000,
                }

            key = f"{regime}|{pattern}"
            results[key] = {
                "regime": str(regime),
                "pattern": str(pattern),
                "horizons": horizon_results,
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

    log_info("Loading range bars", symbol=symbol, threshold_dbps=threshold_dbps)

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


def run_multibar_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
    horizons: list[int],
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run multi-bar forward return analysis."""
    log_info("Starting multi-bar analysis", symbol=symbol, threshold_dbps=threshold_dbps)

    # Load data
    df = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    bar_count = df.select(pl.len()).collect().item()

    if bar_count < 1000:
        log_warn("Insufficient data", bars=bar_count)
        return {"error": "insufficient_data", "bars": bar_count}

    # Prepare analysis DataFrame
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")
    df = classify_regimes(df)
    df = add_bar_direction(df)
    df = add_patterns(df)
    df = add_forward_returns(df, horizons)

    # Collect
    result_df = df.collect()

    # Test ODD robustness at each horizon
    log_info("Testing ODD robustness at multiple horizons", horizons=horizons)
    results = check_odd_robustness_multibar(
        result_df,
        pattern_col="pattern_2bar",
        regime_col="regime",
        horizons=horizons,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Count robust patterns per horizon
    robust_by_horizon = dict.fromkeys(horizons, 0)
    for _pattern_key, pattern_data in results.items():
        for horizon, horizon_data in pattern_data["horizons"].items():
            if horizon_data.get("is_odd_robust", False):
                robust_by_horizon[horizon] += 1

    log_info(
        "Analysis complete",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        robust_by_horizon=robust_by_horizon,
    )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "total_bars": bar_count,
        "robust_by_horizon": robust_by_horizon,
        "results": results,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting multi-bar forward returns analysis (Polars)",
        script="multibar_forward_returns_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100  # Focus on 100 dbps
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizons = [1, 3, 5, 10]  # Test multiple horizons
    min_samples = 100
    min_t_stat = 5.0

    all_results = {}

    for symbol in symbols:
        try:
            result = run_multibar_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
                horizons=horizons,
                min_samples=min_samples,
                min_t_stat=min_t_stat,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json(
                "ERROR",
                "Analysis failed",
                symbol=symbol,
                error=str(e),
            )

    # Cross-symbol comparison
    log_info("=== CROSS-SYMBOL MULTI-HORIZON COMPARISON ===")

    # Count patterns robust at each horizon across all symbols
    patterns_robust_all_symbols = {h: [] for h in horizons}

    for _symbol, result in all_results.items():
        if "error" in result:
            continue
        for pattern_key, pattern_data in result.get("results", {}).items():
            for horizon, horizon_data in pattern_data["horizons"].items():
                if horizon_data.get("is_odd_robust", False):
                    patterns_robust_all_symbols[horizon].append(pattern_key)

    for horizon in horizons:
        pattern_counts = Counter(patterns_robust_all_symbols[horizon])
        universal = [p for p, c in pattern_counts.items() if c >= len(symbols)]
        log_info(
            f"Horizon {horizon} results",
            horizon=horizon,
            total_robust=len(patterns_robust_all_symbols[horizon]),
            universal_count=len(universal),
            universal_patterns=universal[:10] if universal else [],
        )

    # Save results
    output_path = Path("/tmp/multibar_forward_returns_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Multi-bar forward returns analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
