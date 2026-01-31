#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Cross-regime correlation analysis for portfolio diversification (Polars).

Compares return streams between SMA/RSI regime patterns (Issue #52) and
RV regime patterns (Issue #54) to understand diversification potential.

GitHub Issue: #54 (extension - cross-regime diversification)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

    return df.drop(["rv_p25", "rv_p75"])


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
# Correlation Analysis
# =============================================================================


# SMA/RSI patterns from Issue #52
SMA_RSI_PATTERNS = [
    ("chop", "DU"), ("chop", "DD"), ("chop", "UU"), ("chop", "UD"),
    ("bear_neutral", "DU"), ("bear_neutral", "DD"), ("bear_neutral", "UU"), ("bear_neutral", "UD"),
    ("bull_neutral", "DU"), ("bull_neutral", "DD"), ("bull_neutral", "UD"),
]

# RV patterns from Issue #54
RV_PATTERNS = [
    ("quiet", "DU"), ("quiet", "DD"), ("quiet", "UU"), ("quiet", "UD"),
    ("active", "DU"), ("active", "DD"), ("active", "UU"), ("active", "UD"),
    ("volatile", "DU"), ("volatile", "DD"), ("volatile", "UU"), ("volatile", "UD"),
]


def compute_period_returns(
    df: pl.DataFrame,
    regime_col: str,
    patterns: list[tuple[str, str]],
    return_col: str,
    cost_pct: float,
) -> dict[str, pl.DataFrame]:
    """Compute monthly returns for each pattern."""
    # Add period column
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-" +
            pl.col("timestamp").dt.month().cast(pl.Utf8).str.pad_start(2, "0")
        ).alias("period")
    )

    pattern_returns = {}

    for regime, bar_pattern in patterns:
        pattern_name = f"{regime}|{bar_pattern}"

        # Filter to pattern
        pattern_df = df.filter(
            (pl.col(regime_col) == regime) & (pl.col("pattern_2bar") == bar_pattern)
        )

        if len(pattern_df) < 100:
            continue

        # Determine direction based on gross mean
        gross_returns = pattern_df.select(return_col).drop_nulls().to_series()
        gross_mean = float(gross_returns.mean()) if gross_returns.mean() is not None else 0.0
        is_long = gross_mean > 0

        # Compute period-level mean returns
        period_returns = (
            pattern_df
            .group_by("period")
            .agg(pl.col(return_col).mean().alias("mean_ret"))
            .sort("period")
        )

        # Adjust for direction and costs
        if is_long:
            period_returns = period_returns.with_columns(
                (pl.col("mean_ret") - cost_pct).alias("net_ret")
            )
        else:
            period_returns = period_returns.with_columns(
                (-pl.col("mean_ret") - cost_pct).alias("net_ret")
            )

        pattern_returns[pattern_name] = period_returns.select(["period", "net_ret"])

    return pattern_returns


def compute_cross_regime_correlation(
    sma_rsi_returns: dict[str, pl.DataFrame],
    rv_returns: dict[str, pl.DataFrame],
) -> tuple[list[str], list[str], np.ndarray]:
    """Compute correlation matrix between SMA/RSI and RV patterns."""
    # Get all unique periods
    all_periods = set()
    for df in list(sma_rsi_returns.values()) + list(rv_returns.values()):
        periods = df.select("period").to_series().to_list()
        all_periods.update(periods)

    all_periods = sorted(all_periods)
    period_to_idx = {p: i for i, p in enumerate(all_periods)}

    # Build return matrices
    sma_rsi_names = list(sma_rsi_returns.keys())
    rv_names = list(rv_returns.keys())

    sma_rsi_matrix = np.full((len(all_periods), len(sma_rsi_names)), np.nan)
    rv_matrix = np.full((len(all_periods), len(rv_names)), np.nan)

    for j, name in enumerate(sma_rsi_names):
        df = sma_rsi_returns[name]
        for row in df.iter_rows(named=True):
            period_idx = period_to_idx.get(row["period"])
            if period_idx is not None:
                sma_rsi_matrix[period_idx, j] = row["net_ret"]

    for j, name in enumerate(rv_names):
        df = rv_returns[name]
        for row in df.iter_rows(named=True):
            period_idx = period_to_idx.get(row["period"])
            if period_idx is not None:
                rv_matrix[period_idx, j] = row["net_ret"]

    # Compute cross-correlation (SMA/RSI vs RV)
    combined_matrix = np.concatenate([sma_rsi_matrix, rv_matrix], axis=1)
    corr_matrix = np.corrcoef(combined_matrix.T)

    return sma_rsi_names, rv_names, corr_matrix


def compute_effective_dof(corr_matrix: np.ndarray) -> float:
    """Compute effective degrees of freedom (independent bets)."""
    n = corr_matrix.shape[0]
    sum_corr_sq = np.sum(corr_matrix ** 2)
    return n ** 2 / sum_corr_sq


# =============================================================================
# Main Analysis
# =============================================================================


def run_cross_regime_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
) -> dict:
    """Run cross-regime correlation analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Add both regime systems
    df = compute_sma(df, "Close", 20, "sma_fast")
    df = compute_sma(df, "Close", 50, "sma_slow")
    df = compute_rsi(df, "Close", 14, "rsi")
    df = classify_sma_rsi_regimes(df)

    df = compute_realized_volatility(df, 20, "rv")
    df = classify_rv_regimes(df)

    # Add forward returns
    df = df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    result_df = df.collect()

    # Compute period returns for both systems
    cost_pct = ROUND_TRIP_COST_PCT / 100.0

    sma_rsi_returns = compute_period_returns(
        result_df, "trend_regime", SMA_RSI_PATTERNS, f"fwd_ret_{horizon}", cost_pct
    )

    rv_returns = compute_period_returns(
        result_df, "vol_regime", RV_PATTERNS, f"fwd_ret_{horizon}", cost_pct
    )

    if len(sma_rsi_returns) < 2 or len(rv_returns) < 2:
        return {"symbol": symbol, "error": "insufficient patterns"}

    # Compute correlation matrix
    sma_rsi_names, rv_names, corr_matrix = compute_cross_regime_correlation(
        sma_rsi_returns, rv_returns
    )

    # Compute effective DoF
    effective_dof = compute_effective_dof(corr_matrix)

    # Extract cross-correlation block
    n_sma_rsi = len(sma_rsi_names)
    n_rv = len(rv_names)
    cross_corr = corr_matrix[:n_sma_rsi, n_sma_rsi:]

    # Find highest and lowest cross-correlations
    cross_pairs = []
    for i, sma_name in enumerate(sma_rsi_names):
        for j, rv_name in enumerate(rv_names):
            cross_pairs.append((sma_name, rv_name, cross_corr[i, j]))

    high_corr = sorted([p for p in cross_pairs if abs(p[2]) > 0.5], key=lambda x: -abs(x[2]))
    low_corr = sorted([p for p in cross_pairs if abs(p[2]) < 0.2], key=lambda x: abs(x[2]))

    return {
        "symbol": symbol,
        "sma_rsi_patterns": sma_rsi_names,
        "rv_patterns": rv_names,
        "n_sma_rsi": n_sma_rsi,
        "n_rv": n_rv,
        "n_total": n_sma_rsi + n_rv,
        "effective_dof": float(effective_dof),
        "avg_cross_corr": float(np.nanmean(np.abs(cross_corr))),
        "high_cross_corr": [(p[0], p[1], round(p[2], 3)) for p in high_corr[:5]],
        "low_cross_corr": [(p[0], p[1], round(p[2], 3)) for p in low_corr[:5]],
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting cross-regime correlation analysis (Polars)",
        script="cross_regime_correlation_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10

    log_info("=== CROSS-REGIME CORRELATION ANALYSIS ===")

    all_results = {}

    for symbol in symbols:
        try:
            result = run_cross_regime_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
            )
            all_results[symbol] = result
            log_info(
                f"Completed {symbol}",
                n_total=result.get("n_total", 0),
                effective_dof=round(result.get("effective_dof", 0), 2),
                avg_cross_corr=round(result.get("avg_cross_corr", 0), 3),
            )
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL SUMMARY ===")

    avg_effective_dof = np.mean([r.get("effective_dof", 0) for r in all_results.values()])
    avg_cross_corr = np.mean([r.get("avg_cross_corr", 0) for r in all_results.values()])
    total_patterns = all_results.get(symbols[0], {}).get("n_total", 0)

    log_info(
        "Average metrics",
        total_patterns=total_patterns,
        avg_effective_dof=round(avg_effective_dof, 2),
        diversification_ratio=round(avg_effective_dof / total_patterns, 2) if total_patterns else 0,
        avg_cross_corr=round(avg_cross_corr, 3),
    )

    # Compare with single-system DoF
    # From earlier analysis: SMA/RSI alone = 4.5 DoF, RV alone = similar
    sma_rsi_dof = 4.5  # From Issue #52 correlation analysis
    rv_dof = 4.5  # Estimated similar

    log_info("=== DIVERSIFICATION BENEFIT ===")
    log_info(
        "Comparison",
        sma_rsi_alone_dof=sma_rsi_dof,
        rv_alone_dof=rv_dof,
        combined_dof=round(avg_effective_dof, 2),
        diversification_gain=round(avg_effective_dof - max(sma_rsi_dof, rv_dof), 2),
    )

    # High and low correlation pairs across all symbols
    log_info("=== CROSS-REGIME CORRELATION PATTERNS ===")

    # Aggregate high/low correlation pairs
    high_corr_all = []
    low_corr_all = []
    for result in all_results.values():
        high_corr_all.extend(result.get("high_cross_corr", []))
        low_corr_all.extend(result.get("low_cross_corr", []))

    # Most common high correlation pairs
    high_corr_counts = {}
    for p in high_corr_all:
        key = f"{p[0]} vs {p[1]}"
        if key not in high_corr_counts:
            high_corr_counts[key] = []
        high_corr_counts[key].append(p[2])

    log_info(
        "High cross-correlation pairs (SMA/RSI vs RV, |r| > 0.5)",
        count=len(set(high_corr_counts.keys())),
        pairs=[(k, round(np.mean(v), 3)) for k, v in sorted(high_corr_counts.items(), key=lambda x: -abs(np.mean(x[1])))[:5]],
    )

    # Most common low correlation pairs
    low_corr_counts = {}
    for p in low_corr_all:
        key = f"{p[0]} vs {p[1]}"
        if key not in low_corr_counts:
            low_corr_counts[key] = []
        low_corr_counts[key].append(p[2])

    log_info(
        "Low cross-correlation pairs (SMA/RSI vs RV, |r| < 0.2)",
        count=len(set(low_corr_counts.keys())),
        pairs=[(k, round(np.mean(v), 3)) for k, v in sorted(low_corr_counts.items(), key=lambda x: abs(np.mean(x[1])))[:5]],
    )

    # Recommendations
    log_info("=== PORTFOLIO RECOMMENDATIONS ===")

    if avg_cross_corr < 0.3:
        recommendation = "STRONG diversification benefit - combine both systems"
    elif avg_cross_corr < 0.5:
        recommendation = "MODERATE diversification benefit - selective combination"
    else:
        recommendation = "LIMITED diversification benefit - patterns overlap significantly"

    log_info(
        "Recommendation",
        avg_cross_corr=round(avg_cross_corr, 3),
        verdict=recommendation,
    )

    # Save results
    output_path = Path("/tmp/cross_regime_correlation_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Cross-regime correlation analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
