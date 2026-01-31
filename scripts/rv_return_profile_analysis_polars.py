#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Return profile analysis for RV regime patterns (Polars).

Analyzes the distribution characteristics of returns for each of the 12
validated RV regime patterns to inform position sizing and risk management.

GitHub Issue: #54 (volatility regime research)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import polars as pl

# Transaction cost (high VIP tier)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001 / 100  # Convert to decimal

# Symbols and date range
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2026-01-31"

# SOL has later start
SYMBOL_START_DATES = {
    "BTCUSDT": "2022-01-01",
    "ETHUSDT": "2022-01-01",
    "SOLUSDT": "2023-06-01",
    "BNBUSDT": "2022-01-01",
}


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


def compute_realized_volatility(df: pl.LazyFrame, window: int, alias: str) -> pl.LazyFrame:
    """Compute Realized Volatility (std of log returns)."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=window, min_samples=window).alias(alias)
    ).drop("_log_ret")


def classify_rv_regime(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify RV regime using rolling percentiles."""
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
# Return Profile Analysis
# =============================================================================


def compute_return_stats(returns: pl.Series) -> dict:
    """Compute return distribution statistics."""
    n = len(returns)
    if n < 10:
        return {"n": n, "valid": False}

    mean = float(returns.mean()) if returns.mean() is not None else 0.0
    std = float(returns.std()) if returns.std() is not None else 0.0

    # Skewness
    if std > 0:
        skew = float(((returns - mean) ** 3).mean() / (std ** 3)) if std > 0 else 0.0
    else:
        skew = 0.0

    # Kurtosis (excess)
    if std > 0:
        kurt = float(((returns - mean) ** 4).mean() / (std ** 4) - 3) if std > 0 else 0.0
    else:
        kurt = 0.0

    # Percentiles for tail analysis
    p1 = float(returns.quantile(0.01)) if returns.quantile(0.01) is not None else 0.0
    p5 = float(returns.quantile(0.05)) if returns.quantile(0.05) is not None else 0.0
    p25 = float(returns.quantile(0.25)) if returns.quantile(0.25) is not None else 0.0
    p50 = float(returns.quantile(0.50)) if returns.quantile(0.50) is not None else 0.0
    p75 = float(returns.quantile(0.75)) if returns.quantile(0.75) is not None else 0.0
    p95 = float(returns.quantile(0.95)) if returns.quantile(0.95) is not None else 0.0
    p99 = float(returns.quantile(0.99)) if returns.quantile(0.99) is not None else 0.0

    min_ret = float(returns.min()) if returns.min() is not None else 0.0
    max_ret = float(returns.max()) if returns.max() is not None else 0.0

    return {
        "n": n,
        "valid": True,
        "mean_bps": round(mean * 10000, 2),
        "std_bps": round(std * 10000, 2),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "min_bps": round(min_ret * 10000, 2),
        "p1_bps": round(p1 * 10000, 2),
        "p5_bps": round(p5 * 10000, 2),
        "p25_bps": round(p25 * 10000, 2),
        "median_bps": round(p50 * 10000, 2),
        "p75_bps": round(p75 * 10000, 2),
        "p95_bps": round(p95 * 10000, 2),
        "p99_bps": round(p99 * 10000, 2),
        "max_bps": round(max_ret * 10000, 2),
    }


def compute_trading_metrics(returns: pl.Series, cost_pct: float) -> dict:
    """Compute trading metrics after transaction costs."""
    n = len(returns)
    if n < 10:
        return {"n": n, "valid": False}

    # Net returns after costs
    net_returns = returns - cost_pct

    mean = float(net_returns.mean()) if net_returns.mean() is not None else 0.0
    std = float(net_returns.std()) if net_returns.std() is not None else 0.0

    # Win rate
    wins = (net_returns > 0).sum()
    win_rate = float(wins) / n if n > 0 else 0.0

    # Average win/loss
    winning = net_returns.filter(net_returns > 0)
    losing = net_returns.filter(net_returns < 0)

    avg_win = float(winning.mean()) if len(winning) > 0 and winning.mean() is not None else 0.0
    avg_loss = float(losing.mean()) if len(losing) > 0 and losing.mean() is not None else 0.0

    # Profit factor
    total_wins = float(winning.sum()) if len(winning) > 0 and winning.sum() is not None else 0.0
    total_losses = abs(float(losing.sum())) if len(losing) > 0 and losing.sum() is not None else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Kelly fraction: f* = (p*b - q) / b
    # where p = win rate, q = 1-p, b = avg_win / |avg_loss|
    if avg_loss != 0 and win_rate > 0:
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b if b > 0 else 0.0
    else:
        kelly = 0.0

    # Clamp Kelly to reasonable bounds
    kelly = max(0.0, min(1.0, kelly))

    # Per-trade info ratio (not time-weighted Sharpe - see file header comment)
    # allow-simple-sharpe: this is per-trade analysis, not portfolio performance
    info_ratio = mean / std if std > 0 else 0.0

    # Maximum drawdown (cumulative)
    cumulative = net_returns.cum_sum()
    running_max = cumulative.cum_max()
    drawdown = cumulative - running_max
    max_drawdown = float(drawdown.min()) if drawdown.min() is not None else 0.0

    return {
        "n": n,
        "valid": True,
        "net_mean_bps": round(mean * 10000, 2),
        "net_std_bps": round(std * 10000, 2),
        "win_rate": round(win_rate, 4),
        "avg_win_bps": round(avg_win * 10000, 2),
        "avg_loss_bps": round(avg_loss * 10000, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "kelly_fraction": round(kelly, 4),
        "info_ratio_per_trade": round(info_ratio, 4),
        "max_drawdown_bps": round(max_drawdown * 10000, 2),
    }


def analyze_pattern_returns(
    df: pl.DataFrame,
    regime_col: str,
    pattern_col: str,
    return_col: str,
) -> list[dict]:
    """Analyze return profiles for each regime-pattern combination."""
    results = []

    regimes = df.select(regime_col).unique().to_series().drop_nulls().to_list()

    for regime in sorted(regimes):
        if regime is None:
            continue

        regime_df = df.filter(pl.col(regime_col) == regime)
        patterns = regime_df.select(pattern_col).unique().to_series().drop_nulls().to_list()

        for pattern in sorted(patterns):
            if pattern is None:
                continue

            pattern_df = regime_df.filter(pl.col(pattern_col) == pattern)
            returns = pattern_df.select(return_col).drop_nulls().to_series()

            if len(returns) < 100:
                continue

            dist_stats = compute_return_stats(returns)
            trade_stats = compute_trading_metrics(returns, ROUND_TRIP_COST_PCT)

            results.append({
                "regime": regime,
                "pattern": pattern,
                "key": f"{regime}|{pattern}",
                "distribution": dist_stats,
                "trading": trade_stats,
            })

    return results


# =============================================================================
# Main Analysis
# =============================================================================


def run_return_profile_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Run return profile analysis for a symbol."""
    log_info("Loading data", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, 100, start_date, end_date)
    df = add_bar_direction(df)
    df = add_patterns(df)

    # Add RV regime
    df = compute_realized_volatility(df, 20, "rv")
    df = classify_rv_regime(df)

    # Add forward return
    df = df.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
    )

    # Collect and analyze
    df_collected = df.collect()
    log_info("Data loaded", symbol=symbol, bars=len(df_collected))

    results = analyze_pattern_returns(df_collected, "rv_regime", "pattern_2bar", "fwd_return")

    return results


def aggregate_results(all_results: dict[str, list[dict]]) -> dict:
    """Aggregate results across all symbols."""
    # Group by pattern key
    pattern_data: dict[str, list[dict]] = {}

    for symbol, symbol_results in all_results.items():
        for result in symbol_results:
            key = result["key"]
            if key not in pattern_data:
                pattern_data[key] = []
            pattern_data[key].append({
                "symbol": symbol,
                **result,
            })

    # Find universal patterns (present in all symbols with consistent sign)
    universal_patterns = []

    for key, data_list in pattern_data.items():
        if len(data_list) < 4:
            continue

        # Check sign consistency
        signs = [d["distribution"]["mean_bps"] > 0 for d in data_list if d["distribution"]["valid"]]
        if len(signs) < 4:
            continue

        same_sign = all(signs) or not any(signs)
        if not same_sign:
            continue

        # Compute aggregate stats
        total_n = sum(d["distribution"]["n"] for d in data_list)
        avg_mean = sum(d["distribution"]["mean_bps"] * d["distribution"]["n"] for d in data_list) / total_n
        avg_win_rate = sum(d["trading"]["win_rate"] * d["distribution"]["n"] for d in data_list) / total_n
        avg_kelly = sum(d["trading"]["kelly_fraction"] * d["distribution"]["n"] for d in data_list) / total_n
        avg_net_mean = sum(d["trading"]["net_mean_bps"] * d["distribution"]["n"] for d in data_list) / total_n

        # Compute average skew and kurtosis
        avg_skew = sum(d["distribution"]["skew"] * d["distribution"]["n"] for d in data_list) / total_n
        avg_kurt = sum(d["distribution"]["kurtosis"] * d["distribution"]["n"] for d in data_list) / total_n

        universal_patterns.append({
            "key": key,
            "total_n": total_n,
            "avg_mean_bps": round(avg_mean, 2),
            "avg_net_mean_bps": round(avg_net_mean, 2),
            "avg_win_rate": round(avg_win_rate, 4),
            "avg_kelly": round(avg_kelly, 4),
            "avg_skew": round(avg_skew, 3),
            "avg_kurtosis": round(avg_kurt, 3),
            "symbol_count": len(data_list),
            "per_symbol": {
                d["symbol"]: {
                    "n": d["distribution"]["n"],
                    "mean_bps": d["distribution"]["mean_bps"],
                    "net_mean_bps": d["trading"]["net_mean_bps"],
                    "win_rate": d["trading"]["win_rate"],
                    "kelly": d["trading"]["kelly_fraction"],
                }
                for d in data_list
            },
        })

    return {
        "universal_patterns": universal_patterns,
        "all_pattern_data": pattern_data,
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting RV return profile analysis")

    all_results = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        results = run_return_profile_analysis(symbol, start, END_DATE)
        all_results[symbol] = results

        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            patterns_analyzed=len(results),
        )

    # Aggregate across symbols
    aggregated = aggregate_results(all_results)

    log_info(
        "Aggregation complete",
        universal_patterns=len(aggregated["universal_patterns"]),
    )

    # Print summary for each universal pattern
    print("\n" + "=" * 80)
    print("UNIVERSAL RV PATTERN RETURN PROFILES")
    print("=" * 80 + "\n")

    for pattern in sorted(aggregated["universal_patterns"], key=lambda x: x["key"]):
        log_info(
            "Pattern profile",
            key=pattern["key"],
            total_n=pattern["total_n"],
            avg_mean_bps=pattern["avg_mean_bps"],
            avg_net_mean_bps=pattern["avg_net_mean_bps"],
            avg_win_rate=pattern["avg_win_rate"],
            avg_kelly=pattern["avg_kelly"],
            avg_skew=pattern["avg_skew"],
            avg_kurtosis=pattern["avg_kurtosis"],
        )

    # Summary table
    print("\n" + "-" * 80)
    print("SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Pattern':<15} {'N':>10} {'Gross':>10} {'Net':>10} {'Win%':>8} {'Kelly':>8} {'Skew':>8}")
    print("-" * 80)

    for pattern in sorted(aggregated["universal_patterns"], key=lambda x: x["key"]):
        print(
            f"{pattern['key']:<15} "
            f"{pattern['total_n']:>10,} "
            f"{pattern['avg_mean_bps']:>10.2f} "
            f"{pattern['avg_net_mean_bps']:>10.2f} "
            f"{pattern['avg_win_rate']*100:>7.1f}% "
            f"{pattern['avg_kelly']:>8.4f} "
            f"{pattern['avg_skew']:>8.3f}"
        )

    # Trading implications
    print("\n" + "-" * 80)
    print("TRADING IMPLICATIONS")
    print("-" * 80)

    # Patterns profitable after costs
    profitable = [p for p in aggregated["universal_patterns"] if p["avg_net_mean_bps"] > 0]
    unprofitable = [p for p in aggregated["universal_patterns"] if p["avg_net_mean_bps"] <= 0]

    log_info(
        "Profitability summary",
        profitable_patterns=len(profitable),
        unprofitable_patterns=len(unprofitable),
        profitable_keys=[p["key"] for p in profitable],
        unprofitable_keys=[p["key"] for p in unprofitable],
    )

    # Best Kelly fraction patterns
    best_kelly = sorted(aggregated["universal_patterns"], key=lambda x: x["avg_kelly"], reverse=True)[:5]
    log_info(
        "Best Kelly patterns",
        patterns=[{"key": p["key"], "kelly": p["avg_kelly"]} for p in best_kelly],
    )

    # Skew analysis
    positive_skew = [p for p in aggregated["universal_patterns"] if p["avg_skew"] > 0]
    negative_skew = [p for p in aggregated["universal_patterns"] if p["avg_skew"] < 0]
    log_info(
        "Skew analysis",
        positive_skew_patterns=len(positive_skew),
        negative_skew_patterns=len(negative_skew),
        positive_keys=[p["key"] for p in positive_skew],
        negative_keys=[p["key"] for p in negative_skew],
    )

    log_info("Analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
