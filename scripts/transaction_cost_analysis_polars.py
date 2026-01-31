#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Transaction cost analysis for OOD robust regime patterns (Polars).

Analyzes whether validated patterns are profitable after realistic trading costs.
Tests break-even thresholds and net profitability for the 11 universal patterns.

GitHub Issue: #52 (adversarial audit extension)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# The 11 universal OOD robust patterns (validated across all symbols, horizons, parameters)
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

# Transaction cost assumptions
# User specified: 15 dbps (decimal basis points) round-trip
# 1 dbps = 0.001% = 0.00001
# 15 dbps = 0.015% = 0.00015
ROUND_TRIP_COST_DBPS = 15  # Round-trip cost in decimal basis points
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001  # Convert to percentage (0.015%)


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
# Transaction Cost Analysis
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


def analyze_pattern_profitability(
    df: pl.DataFrame,
    regime: str,
    pattern: str,
    horizon: int,
    round_trip_cost_pct: float,
) -> dict:
    """Analyze profitability of a single pattern after transaction costs."""
    # Filter to pattern
    pattern_df = df.filter(
        (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
    )

    if len(pattern_df) == 0:
        return {"regime": regime, "pattern": pattern, "n_trades": 0}

    # Get return column
    return_col = f"fwd_ret_{horizon}"
    returns = pattern_df.select(return_col).drop_nulls().to_series()

    if len(returns) < 100:
        return {"regime": regime, "pattern": pattern, "n_trades": len(returns), "insufficient_data": True}

    # Gross statistics (before costs)
    gross_mean_bps = float(returns.mean()) * 10000 if returns.mean() is not None else 0.0
    gross_std_bps = float(returns.std()) * 10000 if returns.std() is not None else 0.0
    gross_t_stat = compute_t_stat(returns)

    # Determine trade direction based on gross mean
    is_long = gross_mean_bps > 0

    # Net returns after costs
    # For LONG: profit = return - costs (e.g., +11bps - 30bps = -19bps)
    # For SHORT: profit = -return - costs (e.g., -(-11bps) - 30bps = -19bps)
    # In both cases, we need |gross_return| > costs to be profitable
    cost_decimal = round_trip_cost_pct / 100.0
    # For LONG: profit = return - costs; For SHORT: profit = -return - costs
    net_returns = returns - cost_decimal if is_long else -returns - cost_decimal

    net_mean_bps = float(net_returns.mean()) * 10000 if net_returns.mean() is not None else 0.0
    net_t_stat = compute_t_stat(net_returns)

    # Win rate (after costs)
    win_rate = float((net_returns > 0).sum()) / len(net_returns) if is_long else float((net_returns < 0).sum()) / len(net_returns)

    # Profit factor (after costs)
    if is_long:
        gross_wins = float(net_returns.filter(net_returns > 0).sum()) if len(net_returns.filter(net_returns > 0)) > 0 else 0.0
        gross_losses = abs(float(net_returns.filter(net_returns < 0).sum())) if len(net_returns.filter(net_returns < 0)) > 0 else 0.001
    else:
        # For short patterns, wins are negative returns, losses are positive
        gross_wins = abs(float(net_returns.filter(net_returns < 0).sum())) if len(net_returns.filter(net_returns < 0)) > 0 else 0.0
        gross_losses = float(net_returns.filter(net_returns > 0).sum()) if len(net_returns.filter(net_returns > 0)) > 0 else 0.001

    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

    # Break-even cost (max cost where pattern remains profitable)
    break_even_cost_bps = abs(gross_mean_bps)  # in bps

    return {
        "regime": regime,
        "pattern": pattern,
        "horizon": horizon,
        "n_trades": len(returns),
        "direction": "LONG" if is_long else "SHORT",
        "gross_mean_bps": round(gross_mean_bps, 2),
        "gross_std_bps": round(gross_std_bps, 2),
        "gross_t_stat": round(gross_t_stat, 2),
        "cost_bps": round(round_trip_cost_pct * 100, 2),  # Convert to bps
        "net_mean_bps": round(net_mean_bps, 2),
        "net_t_stat": round(net_t_stat, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "break_even_cost_bps": round(break_even_cost_bps, 2),
        "profitable": net_mean_bps > 0,  # net_returns already adjusted for direction
    }


# =============================================================================
# Main Analysis
# =============================================================================


def run_transaction_cost_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    horizons: list[int],
) -> list[dict]:
    """Run transaction cost analysis for a symbol."""
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

    # Add forward returns for all horizons
    for h in horizons:
        df = df.with_columns(
            (pl.col("Close").shift(-h) / pl.col("Close") - 1).alias(f"fwd_ret_{h}")
        )

    result_df = df.collect()
    results = []

    for regime, pattern in UNIVERSAL_PATTERNS:
        for horizon in horizons:
            result = analyze_pattern_profitability(
                result_df,
                regime,
                pattern,
                horizon,
                ROUND_TRIP_COST_PCT,
            )
            result["symbol"] = symbol
            results.append(result)

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting transaction cost analysis (Polars)",
        script="transaction_cost_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizons = [1, 3, 5, 10]

    log_info("=== TRANSACTION COST ANALYSIS ===")
    log_info(
        "Cost assumptions (high VIP tier)",
        round_trip_dbps=ROUND_TRIP_COST_DBPS,
        round_trip_pct=ROUND_TRIP_COST_PCT,
    )

    all_results = []

    for symbol in symbols:
        try:
            results = run_transaction_cost_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                horizons=horizons,
            )
            all_results.extend(results)
            log_info(f"Completed {symbol}", n_results=len(results))
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Convert to DataFrame for analysis
    results_df = pl.DataFrame(all_results)

    # Summary: profitable patterns at each horizon
    log_info("=== PROFITABILITY SUMMARY BY HORIZON ===")

    for horizon in horizons:
        horizon_df = results_df.filter(pl.col("horizon") == horizon)
        profitable_df = horizon_df.filter(pl.col("profitable") == True)  # noqa: E712

        # Count profitable across ALL symbols
        pattern_counts = (
            profitable_df
            .group_by(["regime", "pattern"])
            .agg(pl.len().alias("profitable_symbols"))
        )

        universal_profitable = pattern_counts.filter(
            pl.col("profitable_symbols") >= len(symbols)
        )

        log_info(
            f"Horizon {horizon}-bar",
            total_patterns=len(UNIVERSAL_PATTERNS),
            profitable_any=len(profitable_df.select(["regime", "pattern"]).unique()),
            profitable_all_symbols=len(universal_profitable),
        )

    # Detailed results for 10-bar horizon (most relevant for trading)
    log_info("=== DETAILED RESULTS (10-BAR HORIZON) ===")

    horizon_10 = results_df.filter(pl.col("horizon") == 10)

    # Aggregate across symbols
    agg_results = (
        horizon_10
        .group_by(["regime", "pattern", "direction"])
        .agg([
            pl.col("n_trades").sum(),
            pl.col("gross_mean_bps").mean().alias("avg_gross_bps"),
            pl.col("net_mean_bps").mean().alias("avg_net_bps"),
            pl.col("win_rate_pct").mean().alias("avg_win_rate"),
            pl.col("profit_factor").mean().alias("avg_pf"),
            pl.col("profitable").sum().alias("profitable_count"),
        ])
        .sort("avg_net_bps", descending=True)
    )

    for row in agg_results.iter_rows(named=True):
        log_info(
            f"{row['regime']}|{row['pattern']}",
            direction=row["direction"],
            n_trades=row["n_trades"],
            avg_gross_bps=round(row["avg_gross_bps"], 2),
            avg_net_bps=round(row["avg_net_bps"], 2),
            avg_win_rate=round(row["avg_win_rate"], 2),
            avg_pf=round(row["avg_pf"], 2),
            profitable_symbols=row["profitable_count"],
        )

    # Final summary
    log_info("=== TRADING RECOMMENDATIONS ===")

    # Patterns profitable on ALL symbols at 10-bar
    profitable_all = agg_results.filter(pl.col("profitable_count") >= len(symbols))
    profitable_majority = agg_results.filter(pl.col("profitable_count") >= 3)

    log_info(
        "Patterns profitable on ALL symbols (10-bar)",
        count=len(profitable_all),
        patterns=[f"{r['regime']}|{r['pattern']}" for r in profitable_all.iter_rows(named=True)],
    )

    log_info(
        "Patterns profitable on 3+ symbols (10-bar)",
        count=len(profitable_majority),
        patterns=[f"{r['regime']}|{r['pattern']}" for r in profitable_majority.iter_rows(named=True)],
    )

    # Break-even analysis
    log_info("=== BREAK-EVEN COST ANALYSIS ===")

    breakeven_summary = (
        results_df
        .filter(pl.col("horizon") == 10)
        .group_by(["regime", "pattern"])
        .agg(pl.col("break_even_cost_bps").mean().alias("avg_breakeven_bps"))
        .sort("avg_breakeven_bps", descending=True)
    )

    for row in breakeven_summary.head(11).iter_rows(named=True):
        log_info(
            f"{row['regime']}|{row['pattern']}",
            avg_breakeven_bps=round(row["avg_breakeven_bps"], 2),
            covers_costs=row["avg_breakeven_bps"] > ROUND_TRIP_COST_PCT * 100,
        )

    # Save results
    output_path = Path("/tmp/transaction_cost_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Transaction cost analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
