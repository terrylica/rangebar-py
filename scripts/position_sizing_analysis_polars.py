#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Position sizing analysis for OOD robust regime patterns (Polars).

Analyzes optimal position sizing using Kelly criterion for the 11 universal
patterns. Tests whether Kelly-based sizing improves risk-adjusted returns.

GitHub Issue: #52 (regime research extension)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
# Kelly Criterion Analysis
# =============================================================================


def compute_kelly_fraction(
    returns: pl.Series,
    is_long: bool,
    cost_pct: float,
) -> dict:
    """Compute Kelly fraction for a pattern's returns.

    Kelly = (p * b - q) / b
    where:
        p = win probability
        q = 1 - p = loss probability
        b = odds = avg_win / avg_loss
    """
    if len(returns) < 100:
        return {"kelly": 0.0, "valid": False}

    # Adjust returns for direction and costs
    net_returns = returns - cost_pct if is_long else -returns - cost_pct

    # Split into wins and losses
    wins = net_returns.filter(net_returns > 0)
    losses = net_returns.filter(net_returns <= 0)

    if len(wins) == 0 or len(losses) == 0:
        return {"kelly": 0.0, "valid": False, "reason": "no wins or no losses"}

    # Win probability
    p = len(wins) / len(net_returns)
    q = 1 - p

    # Average win and loss
    avg_win = float(wins.mean())
    avg_loss = abs(float(losses.mean()))

    if avg_loss == 0:
        return {"kelly": 0.0, "valid": False, "reason": "zero avg loss"}

    # Odds (b)
    b = avg_win / avg_loss

    # Kelly fraction
    kelly = (p * b - q) / b if b > 0 else 0.0

    # Return/risk ratio (not Sharpe - no time weighting for range bars)
    # allow-simple-sharpe: using return/std ratio as relative comparison metric only
    mean_ret = float(net_returns.mean())
    std_ret = float(net_returns.std())
    return_risk_ratio = mean_ret / std_ret if std_ret > 0 else 0.0

    return {
        "kelly": round(kelly, 4),
        "half_kelly": round(kelly / 2, 4),
        "quarter_kelly": round(kelly / 4, 4),
        "win_rate": round(p, 4),
        "avg_win_bps": round(avg_win * 10000, 2),
        "avg_loss_bps": round(avg_loss * 10000, 2),
        "odds": round(b, 4),
        "mean_return_bps": round(mean_ret * 10000, 2),
        "std_return_bps": round(std_ret * 10000, 2),
        "return_risk_ratio": round(return_risk_ratio, 4),
        "n_trades": len(net_returns),
        "valid": True,
    }


def analyze_pattern_kelly(
    df: pl.DataFrame,
    regime: str,
    pattern: str,
    horizon: int,
) -> dict:
    """Analyze Kelly criterion for a single pattern."""
    # Filter to pattern
    pattern_df = df.filter(
        (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
    )

    if len(pattern_df) == 0:
        return {"regime": regime, "pattern": pattern, "valid": False}

    # Get returns
    return_col = f"fwd_ret_{horizon}"
    returns = pattern_df.select(return_col).drop_nulls().to_series()

    # Determine direction (based on gross mean)
    gross_mean = float(returns.mean()) if returns.mean() is not None else 0.0
    is_long = gross_mean > 0

    # Compute Kelly
    kelly_stats = compute_kelly_fraction(returns, is_long, ROUND_TRIP_COST_PCT / 100.0)

    return {
        "regime": regime,
        "pattern": pattern,
        "direction": "LONG" if is_long else "SHORT",
        **kelly_stats,
    }


# =============================================================================
# Main Analysis
# =============================================================================


def run_position_sizing_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 10,
) -> list[dict]:
    """Run Kelly criterion analysis for a symbol."""
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
    results = []

    for regime, pattern in UNIVERSAL_PATTERNS:
        result = analyze_pattern_kelly(result_df, regime, pattern, horizon)
        result["symbol"] = symbol
        results.append(result)

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting position sizing analysis (Polars)",
        script="position_sizing_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10

    log_info("=== POSITION SIZING ANALYSIS ===")
    log_info(
        "Parameters",
        horizon=horizon,
        cost_dbps=ROUND_TRIP_COST_DBPS,
    )

    all_results = []

    for symbol in symbols:
        try:
            results = run_position_sizing_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
            )
            all_results.extend(results)
            log_info(f"Completed {symbol}", n_results=len(results))
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Convert to DataFrame for aggregation
    results_df = pl.DataFrame(all_results)

    # Filter to valid results
    valid_df = results_df.filter(pl.col("valid") == True)  # noqa: E712

    # Aggregate by pattern across symbols
    log_info("=== KELLY CRITERION BY PATTERN (10-bar horizon) ===")

    agg_results = (
        valid_df
        .group_by(["regime", "pattern", "direction"])
        .agg([
            pl.col("kelly").mean().alias("avg_kelly"),
            pl.col("half_kelly").mean().alias("avg_half_kelly"),
            pl.col("win_rate").mean().alias("avg_win_rate"),
            pl.col("return_risk_ratio").mean().alias("avg_return_risk"),
            pl.col("n_trades").sum().alias("total_trades"),
            pl.col("mean_return_bps").mean().alias("avg_return_bps"),
        ])
        .sort("avg_kelly", descending=True)
    )

    for row in agg_results.iter_rows(named=True):
        log_info(
            f"{row['regime']}|{row['pattern']}",
            direction=row["direction"],
            avg_kelly=round(row["avg_kelly"], 4),
            avg_half_kelly=round(row["avg_half_kelly"], 4),
            avg_win_rate=round(row["avg_win_rate"], 4),
            avg_return_risk=round(row["avg_return_risk"], 4),
            total_trades=row["total_trades"],
            avg_return_bps=round(row["avg_return_bps"], 2),
        )

    # Recommendations
    log_info("=== POSITION SIZING RECOMMENDATIONS ===")

    # Patterns with positive Kelly
    positive_kelly = agg_results.filter(pl.col("avg_kelly") > 0)
    negative_kelly = agg_results.filter(pl.col("avg_kelly") <= 0)

    log_info(
        "Patterns with positive Kelly (tradeable)",
        count=len(positive_kelly),
        patterns=[f"{r['regime']}|{r['pattern']}" for r in positive_kelly.iter_rows(named=True)],
    )

    log_info(
        "Patterns with negative/zero Kelly (avoid)",
        count=len(negative_kelly),
        patterns=[f"{r['regime']}|{r['pattern']}" for r in negative_kelly.iter_rows(named=True)],
    )

    # Ranking by return/risk
    log_info("=== RANKING BY RETURN/RISK RATIO ===")
    by_return_risk = agg_results.sort("avg_return_risk", descending=True)

    for i, row in enumerate(by_return_risk.iter_rows(named=True), 1):
        log_info(
            f"Rank {i}: {row['regime']}|{row['pattern']}",
            return_risk=round(row["avg_return_risk"], 4),
            kelly=round(row["avg_kelly"], 4),
        )

    # Save results
    output_path = Path("/tmp/position_sizing_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Position sizing analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
