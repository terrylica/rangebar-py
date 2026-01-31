#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Historical formation patterns with regime filtering (Polars).

Retry historical lookback formation analysis WITH regime filtering.
Previous analysis (historical_formation_patterns_polars.py) found ZERO ODD robust patterns globally.

Hypothesis: Historical formation patterns at 50/200 dbps may predict 100 dbps
forward returns WITHIN specific market regimes, even if they fail globally.

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
    """Classify market regimes."""
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
# Formation Pattern Analysis
# =============================================================================


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction (U for up, D for down)."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_formation_patterns(df: pl.LazyFrame, lookback: int = 3) -> pl.LazyFrame:
    """Add historical formation patterns (lookback bars before current).

    This uses PAST bars only (shift with positive values), so no look-ahead bias.
    """
    # Build formation from previous bars (not including current)
    cols = [pl.col("direction").shift(i) for i in range(1, lookback + 1)]

    # Concatenate from oldest to newest: shift(3) + shift(2) + shift(1)
    formation_expr = cols[-1]  # oldest
    for col in reversed(cols[:-1]):
        formation_expr = formation_expr + col

    return df.with_columns(formation_expr.alias(f"formation_{lookback}"))


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


def check_odd_robustness_by_regime(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 50,
    min_t_stat: float = 5.0,
) -> dict:
    """Check ODD robustness for patterns within each regime."""
    # Add quarterly period
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    regimes = df.select("regime").unique().to_series().drop_nulls().to_list()
    results = {}

    for regime in regimes:
        if regime is None:
            continue

        regime_df = df.filter(pl.col("regime") == regime)
        patterns = regime_df.select(pattern_col).unique().to_series().drop_nulls().to_list()

        regime_results = {}
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

                mean_ret = returns.mean()
                t_stat = compute_t_stat(returns)

                period_stats.append({
                    "period": period_label,
                    "count": count,
                    "mean_return": float(mean_ret) if mean_ret is not None else 0.0,
                    "t_stat": t_stat,
                })

            if len(period_stats) < 2:
                continue

            t_stats = [s["t_stat"] for s in period_stats]
            all_significant = all(abs(t) >= min_t_stat for t in t_stats)
            same_sign = all(t > 0 for t in t_stats) or all(t < 0 for t in t_stats)

            is_odd_robust = all_significant and same_sign

            if is_odd_robust:
                total_count = sum(s["count"] for s in period_stats)
                avg_return = (
                    sum(s["mean_return"] * s["count"] for s in period_stats) / total_count
                    if total_count > 0 else 0
                )

                key = f"{regime}|{pattern}"
                regime_results[key] = {
                    "is_odd_robust": True,
                    "n_periods": len(period_stats),
                    "min_t_stat": min(abs(t) for t in t_stats),
                    "max_t_stat": max(abs(t) for t in t_stats),
                    "total_count": total_count,
                    "avg_return_bps": avg_return * 10000,
                }

        results[regime] = regime_results

    return results


# =============================================================================
# Main Analysis
# =============================================================================


def run_historical_formation_regime_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    lookback: int = 3,
    horizon: int = 10,
    min_samples: int = 50,
    min_t_stat: float = 5.0,
) -> dict:
    """Run historical formation analysis within market regimes."""
    log_info("Starting historical formation + regime analysis", symbol=symbol,
             lookback=lookback, horizon=horizon)

    # Load data at three granularities
    df_50 = load_range_bars_polars(symbol, 50, start_date, end_date)
    df_100 = load_range_bars_polars(symbol, 100, start_date, end_date)
    df_200 = load_range_bars_polars(symbol, 200, start_date, end_date)

    # Process 50 dbps - add historical formation (PAST bars only)
    df_50 = add_bar_direction(df_50)
    df_50 = add_formation_patterns(df_50, lookback)

    # Process 100 dbps with regime classification and forward returns
    df_100 = compute_sma(df_100, "Close", 20, "sma20")
    df_100 = compute_sma(df_100, "Close", 50, "sma50")
    df_100 = compute_rsi(df_100, "Close", 14, "rsi14")
    df_100 = classify_regimes(df_100)
    df_100 = df_100.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )

    # Process 200 dbps - add historical formation (PAST bars only)
    df_200 = add_bar_direction(df_200)
    df_200 = add_formation_patterns(df_200, lookback)

    # Collect with relevant columns
    df_50_c = df_50.select(["timestamp", f"formation_{lookback}"]).collect()
    df_100_c = df_100.select(["timestamp", "Close", "regime", f"fwd_ret_{horizon}"]).collect()
    df_200_c = df_200.select(["timestamp", f"formation_{lookback}"]).collect()

    log_info("Data loaded", symbol=symbol,
             bars_50=len(df_50_c), bars_100=len(df_100_c), bars_200=len(df_200_c))

    # Rename for clarity
    df_50_c = df_50_c.rename({f"formation_{lookback}": f"formation_50_{lookback}"})
    df_200_c = df_200_c.rename({f"formation_{lookback}": f"formation_200_{lookback}"})

    # Sort and join
    df_50_c = df_50_c.sort("timestamp")
    df_100_c = df_100_c.sort("timestamp")
    df_200_c = df_200_c.sort("timestamp")

    merged = df_100_c.join_asof(df_50_c, on="timestamp", strategy="backward")
    merged = merged.join_asof(df_200_c, on="timestamp", strategy="backward")

    # Create combined formation code
    merged = merged.with_columns(
        (
            pl.lit("50:") + pl.col(f"formation_50_{lookback}") +
            pl.lit("|200:") + pl.col(f"formation_200_{lookback}")
        ).alias("combined_formation")
    )

    log_info("Data merged", symbol=symbol, merged_rows=len(merged))

    # Test ODD robustness within each regime
    return_col = f"fwd_ret_{horizon}"
    results = {
        "symbol": symbol,
        "lookback": lookback,
        "horizon": horizon,
        "total_bars": len(merged),
    }

    # Test 50 dbps formations within regimes
    clean_50 = merged.drop_nulls(subset=[f"formation_50_{lookback}", "regime", return_col])
    formation_50_results = check_odd_robustness_by_regime(
        clean_50,
        f"formation_50_{lookback}",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    total_50 = sum(len(patterns) for patterns in formation_50_results.values())
    results["formation_50"] = {"total_robust": total_50, "by_regime": formation_50_results}
    log_info("50 dbps formations tested", symbol=symbol, total_robust=total_50)

    # Test 200 dbps formations within regimes
    clean_200 = merged.drop_nulls(subset=[f"formation_200_{lookback}", "regime", return_col])
    formation_200_results = check_odd_robustness_by_regime(
        clean_200,
        f"formation_200_{lookback}",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    total_200 = sum(len(patterns) for patterns in formation_200_results.values())
    results["formation_200"] = {"total_robust": total_200, "by_regime": formation_200_results}
    log_info("200 dbps formations tested", symbol=symbol, total_robust=total_200)

    # Test combined formations within regimes
    clean_combined = merged.drop_nulls(subset=["combined_formation", "regime", return_col])
    combined_results = check_odd_robustness_by_regime(
        clean_combined,
        "combined_formation",
        return_col,
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )
    total_combined = sum(len(patterns) for patterns in combined_results.values())
    results["formation_combined"] = {"total_robust": total_combined, "by_regime": combined_results}
    log_info("Combined formations tested", symbol=symbol, total_robust=total_combined)

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting historical formation + regime analysis (Polars)",
        script="historical_formation_regime_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    lookback = 3  # 3-bar historical formation
    horizon = 10
    min_samples = 50
    min_t_stat = 5.0

    log_info("=== HISTORICAL FORMATION + REGIME ANALYSIS ===")
    log_info("Testing if historical lookback patterns work WITHIN regimes")

    all_results = {}

    for symbol in symbols:
        try:
            result = run_historical_formation_regime_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback,
                horizon=horizon,
                min_samples=min_samples,
                min_t_stat=min_t_stat,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL HISTORICAL FORMATION + REGIME SUMMARY ===")

    # Collect all robust patterns and count universal ones
    for formation_type in ["formation_50", "formation_200", "formation_combined"]:
        pattern_counts = Counter()
        for _symbol, result in all_results.items():
            data = result.get(formation_type, {})
            for _regime, patterns in data.get("by_regime", {}).items():
                for pattern_key in patterns:
                    pattern_counts[pattern_key] += 1

        universal = [p for p, c in pattern_counts.items() if c >= len(symbols)]
        log_info(
            f"Formation type: {formation_type}",
            type=formation_type,
            total_robust_instances=sum(pattern_counts.values()),
            universal_count=len(universal),
            universal_patterns=universal[:10] if universal else [],
        )

    # Compare to baseline (0 patterns found globally)
    log_info("=== COMPARISON TO BASELINE ===")
    log_info("Global historical formation (no regime): 0 universal patterns")

    total_universal_with_regime = 0
    for formation_type in ["formation_50", "formation_200", "formation_combined"]:
        pattern_counts = Counter()
        for _symbol, result in all_results.items():
            data = result.get(formation_type, {})
            for _regime, patterns in data.get("by_regime", {}).items():
                for pattern_key in patterns:
                    pattern_counts[pattern_key] += 1
        universal_count = sum(1 for c in pattern_counts.values() if c >= len(symbols))
        total_universal_with_regime += universal_count

    log_info(
        "With regime filtering",
        total_universal=total_universal_with_regime,
        improvement="Regime filtering reveals patterns not visible globally" if total_universal_with_regime > 0 else "No patterns found",
    )

    # Save results
    output_path = Path("/tmp/historical_formation_regime_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Historical formation + regime analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
