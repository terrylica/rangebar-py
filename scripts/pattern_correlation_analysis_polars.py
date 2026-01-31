#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Pattern correlation analysis for OOD robust regime patterns (Polars).

Analyzes correlation structure between the 11 universal patterns to understand
diversification benefits and portfolio construction implications.

GitHub Issue: #52 (regime research extension)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# The 11 universal OOD robust patterns
UNIVERSAL_PATTERNS = [
    ("chop", "DU"),
    ("chop", "DD"),
    ("chop", "UU"),
    ("chop", "UD"),
    ("bear_neutral", "DU"),
    ("bear_neutral", "DD"),
    ("bear_neutral", "UU"),
    ("bear_neutral", "UD"),
    ("bull_neutral", "DU"),
    ("bull_neutral", "DD"),
    ("bull_neutral", "UD"),
]

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
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


# =============================================================================
# Correlation Analysis
# =============================================================================


def compute_rolling_returns(
    df: pl.DataFrame,
    regime: str,
    pattern: str,
    horizon: int,
    is_long: bool,
    cost_pct: float,
) -> pl.Series:
    """Get net returns for a pattern, indexed by period."""
    # Filter to pattern
    pattern_df = df.filter(
        (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
    )

    if len(pattern_df) == 0:
        return pl.Series("net_ret", [], dtype=pl.Float64)

    return_col = f"fwd_ret_{horizon}"
    returns = pattern_df.select(return_col).drop_nulls().to_series()

    # Adjust for direction and costs
    net_returns = returns - cost_pct if is_long else -returns - cost_pct

    return net_returns


def compute_period_returns(
    df: pl.DataFrame,
    horizon: int,
    cost_pct: float,
) -> dict[str, pl.Series]:
    """Compute returns for all patterns, binned by period for correlation."""
    # Add period column
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-" +
            pl.col("timestamp").dt.month().cast(pl.Utf8).str.pad_start(2, "0")
        ).alias("period")
    )

    pattern_returns = {}

    for regime, pattern in UNIVERSAL_PATTERNS:
        pattern_name = f"{regime}|{pattern}"

        # Filter to pattern
        pattern_df = df.filter(
            (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
        )

        if len(pattern_df) < 100:
            continue

        return_col = f"fwd_ret_{horizon}"

        # Determine direction
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


def compute_correlation_matrix(pattern_returns: dict[str, pl.DataFrame]) -> tuple:
    """Compute correlation matrix between patterns."""
    pattern_names = list(pattern_returns.keys())
    n_patterns = len(pattern_names)

    # Get all unique periods
    all_periods = set()
    for _name, df in pattern_returns.items():
        periods = df.select("period").to_series().to_list()
        all_periods.update(periods)

    all_periods = sorted(all_periods)

    # Build return matrix (periods x patterns)
    return_matrix = np.full((len(all_periods), n_patterns), np.nan)

    period_to_idx = {p: i for i, p in enumerate(all_periods)}

    for j, name in enumerate(pattern_names):
        df = pattern_returns[name]
        for row in df.iter_rows(named=True):
            period_idx = period_to_idx.get(row["period"])
            if period_idx is not None:
                return_matrix[period_idx, j] = row["net_ret"]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(return_matrix.T)

    return pattern_names, corr_matrix


def compute_effective_dof(corr_matrix: np.ndarray) -> float:
    """Compute effective degrees of freedom (independent bets).

    Uses the formula: N_eff = N^2 / sum(corr^2)
    """
    n = corr_matrix.shape[0]
    sum_corr_sq = np.sum(corr_matrix ** 2)
    return n ** 2 / sum_corr_sq


# =============================================================================
# Main Analysis
# =============================================================================


def run_correlation_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
) -> dict:
    """Run correlation analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
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

    # Compute period returns for all patterns
    cost_pct = ROUND_TRIP_COST_PCT / 100.0
    pattern_returns = compute_period_returns(result_df, horizon, cost_pct)

    if len(pattern_returns) < 2:
        return {"symbol": symbol, "error": "insufficient patterns"}

    # Compute correlation matrix
    pattern_names, corr_matrix = compute_correlation_matrix(pattern_returns)

    # Compute effective DoF
    effective_dof = compute_effective_dof(corr_matrix)

    return {
        "symbol": symbol,
        "pattern_names": pattern_names,
        "correlation_matrix": corr_matrix.tolist(),
        "effective_dof": float(effective_dof),
        "n_patterns": len(pattern_names),
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting correlation analysis (Polars)",
        script="pattern_correlation_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10

    log_info("=== PATTERN CORRELATION ANALYSIS ===")
    log_info(
        "Parameters",
        horizon=horizon,
        cost_dbps=ROUND_TRIP_COST_DBPS,
    )

    all_results = {}
    all_corr_matrices = []

    for symbol in symbols:
        try:
            result = run_correlation_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
            )
            all_results[symbol] = result
            if "correlation_matrix" in result:
                all_corr_matrices.append(np.array(result["correlation_matrix"]))
            log_info(
                f"Completed {symbol}",
                n_patterns=result.get("n_patterns", 0),
                effective_dof=round(result.get("effective_dof", 0), 2),
            )
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Average correlation matrix across symbols
    if all_corr_matrices:
        avg_corr_matrix = np.mean(all_corr_matrices, axis=0)
        pattern_names = all_results[symbols[0]]["pattern_names"]

        log_info("=== AVERAGE CORRELATION MATRIX ===")

        # Print correlation pairs
        high_corr_pairs = []
        low_corr_pairs = []

        for i in range(len(pattern_names)):
            for j in range(i + 1, len(pattern_names)):
                corr = avg_corr_matrix[i, j]
                pair = (pattern_names[i], pattern_names[j], corr)
                if abs(corr) > 0.5:
                    high_corr_pairs.append(pair)
                elif abs(corr) < 0.2:
                    low_corr_pairs.append(pair)

        log_info(
            "Highly correlated pairs (|r| > 0.5)",
            count=len(high_corr_pairs),
            pairs=[(p[0], p[1], round(p[2], 3)) for p in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:10]],
        )

        log_info(
            "Low correlation pairs (|r| < 0.2)",
            count=len(low_corr_pairs),
            pairs=[(p[0], p[1], round(p[2], 3)) for p in sorted(low_corr_pairs, key=lambda x: abs(x[2]))[:10]],
        )

        # Average effective DoF
        avg_effective_dof = np.mean([r.get("effective_dof", 0) for r in all_results.values()])
        log_info(
            "Average effective degrees of freedom",
            n_patterns=len(pattern_names),
            effective_dof=round(avg_effective_dof, 2),
            diversification_ratio=round(avg_effective_dof / len(pattern_names), 2),
        )

        # Clustering analysis (simple grouping by correlation)
        log_info("=== PATTERN CLUSTERS ===")

        # Group by regime
        chop_patterns = [p for p in pattern_names if p.startswith("chop")]
        bear_patterns = [p for p in pattern_names if p.startswith("bear")]
        bull_patterns = [p for p in pattern_names if p.startswith("bull")]

        # Compute within-cluster correlations
        for cluster_name, cluster in [("chop", chop_patterns), ("bear_neutral", bear_patterns), ("bull_neutral", bull_patterns)]:
            if len(cluster) < 2:
                continue

            cluster_corrs = []
            for i, p1 in enumerate(cluster):
                for p2 in cluster[i + 1:]:
                    idx1 = pattern_names.index(p1)
                    idx2 = pattern_names.index(p2)
                    cluster_corrs.append(avg_corr_matrix[idx1, idx2])

            if cluster_corrs:
                log_info(
                    f"Cluster: {cluster_name}",
                    n_patterns=len(cluster),
                    avg_within_corr=round(np.mean(cluster_corrs), 3),
                    patterns=cluster,
                )

    # Save results
    output_path = Path("/tmp/pattern_correlation_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Correlation analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
