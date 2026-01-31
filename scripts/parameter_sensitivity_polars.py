#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Parameter sensitivity analysis for regime definitions (Polars).

Test if OOD robust patterns are sensitive to regime parameter choices.
This addresses the adversarial audit concern about "local bias" and parameter snooping.

If patterns only work with exact SMA 20/50 + RSI 14 but fail with alternative
parameters, they may be overfit to specific parameter choices.

GitHub Issue: #52 (adversarial audit extension)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Parameter sets to test
PARAMETER_SETS = [
    {"name": "baseline", "sma_fast": 20, "sma_slow": 50, "rsi_period": 14},
    {"name": "shorter_sma", "sma_fast": 15, "sma_slow": 40, "rsi_period": 14},
    {"name": "longer_sma", "sma_fast": 25, "sma_slow": 60, "rsi_period": 14},
    {"name": "fast_sma", "sma_fast": 10, "sma_slow": 30, "rsi_period": 14},
    {"name": "shorter_rsi", "sma_fast": 20, "sma_slow": 50, "rsi_period": 10},
    {"name": "longer_rsi", "sma_fast": 20, "sma_slow": 50, "rsi_period": 20},
]

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
# Technical Indicators with Parameters
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


def classify_regimes(df: pl.LazyFrame, sma_fast_col: str, sma_slow_col: str, rsi_col: str) -> pl.LazyFrame:
    """Classify market regimes using specified column names."""
    df = df.with_columns(
        pl.when(
            (pl.col("Close") > pl.col(sma_fast_col)) & (pl.col(sma_fast_col) > pl.col(sma_slow_col))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("Close") < pl.col(sma_fast_col)) & (pl.col(sma_fast_col) < pl.col(sma_slow_col))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("consolidation"))
        .alias("sma_regime")
    )

    df = df.with_columns(
        pl.when(pl.col(rsi_col) > 70)
        .then(pl.lit("overbought"))
        .when(pl.col(rsi_col) < 30)
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
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
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


def count_odd_robust_patterns(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Count ODD robust patterns within each regime."""
    # Add quarterly period
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    regimes = df.select("regime").unique().to_series().drop_nulls().to_list()
    robust_patterns = []

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
            for period_label in periods:
                period_data = pattern_subset.filter(pl.col("period") == period_label)
                returns = period_data.select(return_col).drop_nulls().to_series()
                count = len(returns)

                if count < min_samples:
                    continue

                t_stat = compute_t_stat(returns)
                period_stats.append(t_stat)

            if len(period_stats) < 2:
                continue

            all_significant = all(abs(t) >= min_t_stat for t in period_stats)
            same_sign = all(t > 0 for t in period_stats) or all(t < 0 for t in period_stats)

            if all_significant and same_sign:
                robust_patterns.append(f"{regime}|{pattern}")

    return {"robust_patterns": robust_patterns, "count": len(robust_patterns)}


# =============================================================================
# Main Analysis
# =============================================================================


def run_parameter_sensitivity_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run parameter sensitivity analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load data once
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)
    df = df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    results = {"symbol": symbol, "parameter_sets": {}}

    for params in PARAMETER_SETS:
        name = params["name"]
        sma_fast = params["sma_fast"]
        sma_slow = params["sma_slow"]
        rsi_period = params["rsi_period"]

        log_info(f"Testing {name}", symbol=symbol, params=params)

        # Add indicators with these parameters
        df_param = compute_sma(df, "Close", sma_fast, "sma_fast")
        df_param = compute_sma(df_param, "Close", sma_slow, "sma_slow")
        df_param = compute_rsi(df_param, "Close", rsi_period, "rsi")
        df_param = classify_regimes(df_param, "sma_fast", "sma_slow", "rsi")

        result_df = df_param.collect()

        # Count robust patterns
        clean_df = result_df.drop_nulls(subset=["pattern_2bar", "regime", f"fwd_ret_{horizon}"])
        robust_result = count_odd_robust_patterns(
            clean_df,
            "pattern_2bar",
            f"fwd_ret_{horizon}",
            min_samples=min_samples,
            min_t_stat=min_t_stat,
        )

        results["parameter_sets"][name] = {
            "params": params,
            "robust_count": robust_result["count"],
            "robust_patterns": robust_result["robust_patterns"],
        }

        log_info(f"{name} results", symbol=symbol, robust_count=robust_result["count"])

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting parameter sensitivity analysis (Polars)",
        script="parameter_sensitivity_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10
    min_samples = 100
    min_t_stat = 5.0

    log_info("=== PARAMETER SENSITIVITY ANALYSIS ===")
    log_info("Testing if patterns are robust across different SMA/RSI parameters")

    all_results = {}

    for symbol in symbols:
        try:
            result = run_parameter_sensitivity_analysis(
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

    # Cross-parameter summary
    log_info("=== CROSS-PARAMETER SUMMARY ===")

    for params in PARAMETER_SETS:
        name = params["name"]
        pattern_counts = Counter()

        for _symbol, result in all_results.items():
            param_data = result.get("parameter_sets", {}).get(name, {})
            for pattern in param_data.get("robust_patterns", []):
                pattern_counts[pattern] += 1

        universal = [p for p, c in pattern_counts.items() if c >= len(symbols)]
        log_info(
            f"Parameter set: {name}",
            name=name,
            total_robust_instances=sum(pattern_counts.values()),
            universal_count=len(universal),
        )

    # Find patterns robust across MULTIPLE parameter sets
    log_info("=== PARAMETER-AGNOSTIC PATTERNS ===")

    pattern_param_counts = Counter()
    for _symbol, result in all_results.items():
        for _name, param_data in result.get("parameter_sets", {}).items():
            for pattern in param_data.get("robust_patterns", []):
                pattern_param_counts[pattern] += 1

    # Patterns that appear in at least 4 parameter sets * 4 symbols = 16 times
    robust_across_params = [p for p, c in pattern_param_counts.items() if c >= len(PARAMETER_SETS) * len(symbols)]

    log_info(
        "Patterns robust across ALL parameter sets",
        count=len(robust_across_params),
        patterns=robust_across_params[:10] if robust_across_params else [],
    )

    # Detailed comparison
    log_info("=== PARAMETER COMPARISON TABLE ===")
    for name in [p["name"] for p in PARAMETER_SETS]:
        universal_counts = []
        for _symbol, result in all_results.items():
            param_data = result.get("parameter_sets", {}).get(name, {})
            universal_counts.append(param_data.get("robust_count", 0))

        log_info(
            f"{name}",
            by_symbol=universal_counts,
            total=sum(universal_counts),
        )

    # Save results
    output_path = Path("/tmp/parameter_sensitivity_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "robust_across_params": robust_across_params,
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Parameter sensitivity analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
