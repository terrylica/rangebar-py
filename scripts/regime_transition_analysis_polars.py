#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Regime transition timing analysis (Polars).

Research question: Can we predict when market regimes will flip?

Analyzes:
1. Average duration of each regime (bars and time)
2. Transition probabilities (which regime follows which)
3. Patterns that precede regime transitions
4. Test if transition signals are ODD robust

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
    """Add 2-bar and 3-bar pattern columns."""
    return df.with_columns([
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar"),
        (pl.col("direction") + pl.col("direction").shift(-1) + pl.col("direction").shift(-2)).alias("pattern_3bar"),
    ])


# =============================================================================
# Regime Transition Analysis
# =============================================================================


def identify_regime_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Identify contiguous regime runs and their properties."""
    # Add regime change indicator
    df = df.with_columns([
        (pl.col("regime") != pl.col("regime").shift(1)).alias("regime_change"),
    ])

    # Create run ID using cumulative sum of changes
    df = df.with_columns([
        pl.col("regime_change").cum_sum().alias("run_id"),
    ])

    return df


def compute_regime_durations(df: pl.DataFrame) -> dict:
    """Compute average duration of each regime."""
    # Group by run_id to get run lengths
    runs = df.group_by(["run_id", "regime"]).agg([
        pl.len().alias("run_length"),
        pl.col("timestamp").min().alias("start_time"),
        pl.col("timestamp").max().alias("end_time"),
    ])

    # Compute duration in microseconds
    runs = runs.with_columns([
        ((pl.col("end_time") - pl.col("start_time")).dt.total_microseconds()).alias("duration_us"),
    ])

    # Aggregate by regime
    durations = {}
    for regime in runs.select("regime").unique().to_series().to_list():
        regime_runs = runs.filter(pl.col("regime") == regime)
        if len(regime_runs) > 0:
            durations[regime] = {
                "count": len(regime_runs),
                "mean_bars": float(regime_runs.select("run_length").mean().item()),
                "median_bars": float(regime_runs.select("run_length").median().item()),
                "max_bars": int(regime_runs.select("run_length").max().item()),
                "mean_duration_sec": float(regime_runs.select("duration_us").mean().item() / 1_000_000) if regime_runs.select("duration_us").mean().item() else 0,
            }

    return durations


def compute_transition_matrix(df: pl.DataFrame) -> dict:
    """Compute regime transition probabilities."""
    # Add next regime
    df = df.with_columns([
        pl.col("regime").shift(-1).alias("next_regime"),
    ])

    # Filter to regime change points only
    transitions = df.filter(pl.col("regime") != pl.col("next_regime")).drop_nulls(subset=["next_regime"])

    # Count transitions
    transition_counts = transitions.group_by(["regime", "next_regime"]).agg([
        pl.len().alias("count"),
    ])

    # Compute probabilities
    total_by_regime = transition_counts.group_by("regime").agg([
        pl.col("count").sum().alias("total"),
    ])

    transition_counts = transition_counts.join(total_by_regime, on="regime")
    transition_counts = transition_counts.with_columns([
        (pl.col("count") / pl.col("total")).alias("probability"),
    ])

    # Convert to nested dict
    matrix = {}
    for row in transition_counts.iter_rows(named=True):
        from_regime = row["regime"]
        to_regime = row["next_regime"]
        prob = row["probability"]
        count = row["count"]

        if from_regime not in matrix:
            matrix[from_regime] = {}
        matrix[from_regime][to_regime] = {
            "probability": round(prob, 4),
            "count": count,
        }

    return matrix


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


def analyze_pre_transition_patterns(
    df: pl.DataFrame,
    lookback: int = 5,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Analyze patterns that precede regime transitions."""
    # Add next regime and regime change flag
    df = df.with_columns([
        pl.col("regime").shift(-1).alias("next_regime"),
        (pl.col("regime") != pl.col("regime").shift(-1)).alias("is_transition"),
    ])

    # Add quarterly period for ODD testing
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8) + "-Q" +
            ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    # Focus on transitions to specific regimes
    results = {}
    regimes = df.select("regime").unique().to_series().drop_nulls().to_list()
    patterns_2bar = ["UU", "UD", "DU", "DD"]

    for from_regime in regimes:
        for to_regime in regimes:
            if from_regime == to_regime:
                continue

            # Get bars just before this specific transition
            transition_mask = (
                (pl.col("regime") == from_regime) &
                (pl.col("next_regime") == to_regime) &
                pl.col("is_transition")
            )

            # Test each 2-bar pattern
            for pattern in patterns_2bar:
                subset = df.filter(
                    transition_mask &
                    (pl.col("pattern_2bar") == pattern)
                )

                total_count = len(subset)
                if total_count < min_samples:
                    continue

                # ODD robustness test across periods
                periods = subset.select("period").unique().to_series().drop_nulls().to_list()
                period_stats = []

                for period_label in periods:
                    period_data = subset.filter(pl.col("period") == period_label)
                    count = len(period_data)

                    if count < 30:  # Lower threshold for transition events
                        continue

                    # Use transition rate as the "return" signal
                    # This is 1 for transition, which is what we're counting
                    period_stats.append({
                        "period": period_label,
                        "count": count,
                    })

                if len(period_stats) < 2:
                    continue

                # For transitions, we use chi-square like test
                # Check if pattern frequency is stable across periods
                counts = [s["count"] for s in period_stats]
                total = sum(counts)
                mean_count = total / len(counts)
                std_count = (sum((c - mean_count) ** 2 for c in counts) / len(counts)) ** 0.5

                cv = std_count / mean_count if std_count > 0 else 0

                key = f"{from_regime}→{to_regime}|{pattern}"
                results[key] = {
                    "from_regime": from_regime,
                    "to_regime": to_regime,
                    "pattern": pattern,
                    "total_count": total_count,
                    "n_periods": len(period_stats),
                    "cv": round(cv, 4),
                    "is_stable": cv < 0.5,  # Less than 50% variation = stable
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


def run_transition_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Run regime transition analysis for a symbol."""
    log_info("Starting transition analysis", symbol=symbol, threshold_dbps=threshold_dbps)

    # Load and prepare data
    df = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")
    df = classify_regimes(df)
    df = add_bar_direction(df)
    df = add_patterns(df)

    result_df = df.collect()
    total_bars = len(result_df)

    log_info("Data loaded", symbol=symbol, total_bars=total_bars)

    # Identify regime runs
    result_df = identify_regime_runs(result_df)

    # Compute regime durations
    durations = compute_regime_durations(result_df)
    log_info("Regime durations computed", symbol=symbol, regimes=list(durations.keys()))

    # Compute transition matrix
    transitions = compute_transition_matrix(result_df)
    log_info("Transition matrix computed", symbol=symbol)

    # Analyze pre-transition patterns
    pre_patterns = analyze_pre_transition_patterns(result_df)
    stable_patterns = {k: v for k, v in pre_patterns.items() if v.get("is_stable", False)}
    log_info("Pre-transition patterns analyzed", symbol=symbol, total=len(pre_patterns), stable=len(stable_patterns))

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "total_bars": total_bars,
        "regime_durations": durations,
        "transition_matrix": transitions,
        "pre_transition_patterns": pre_patterns,
        "stable_patterns": stable_patterns,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting regime transition analysis (Polars)",
        script="regime_transition_analysis_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}

    for symbol in symbols:
        try:
            result = run_transition_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL REGIME DURATION SUMMARY ===")

    regimes = ["chop", "bull_neutral", "bear_neutral", "bull_hot", "bear_cold", "bull_cold", "bear_hot"]

    for regime in regimes:
        durations = []
        for _symbol, result in all_results.items():
            dur = result.get("regime_durations", {}).get(regime, {})
            if dur:
                durations.append(dur.get("mean_bars", 0))

        if durations:
            avg_duration = sum(durations) / len(durations)
            log_info(
                f"Regime {regime}",
                regime=regime,
                avg_duration_bars=round(avg_duration, 1),
                n_symbols=len(durations),
            )

    # Cross-symbol transition probabilities
    log_info("=== CROSS-SYMBOL TRANSITION PROBABILITIES ===")

    # Aggregate transition probabilities
    transition_probs = {}
    for _symbol, result in all_results.items():
        for from_regime, to_dict in result.get("transition_matrix", {}).items():
            if from_regime not in transition_probs:
                transition_probs[from_regime] = {}
            for to_regime, data in to_dict.items():
                if to_regime not in transition_probs[from_regime]:
                    transition_probs[from_regime][to_regime] = []
                transition_probs[from_regime][to_regime].append(data["probability"])

    # Log top transitions
    top_transitions = []
    for from_regime, to_dict in transition_probs.items():
        for to_regime, probs in to_dict.items():
            if len(probs) >= 3:  # At least 3 symbols
                avg_prob = sum(probs) / len(probs)
                top_transitions.append({
                    "from": from_regime,
                    "to": to_regime,
                    "avg_prob": avg_prob,
                    "n_symbols": len(probs),
                })

    top_transitions.sort(key=lambda x: x["avg_prob"], reverse=True)

    for t in top_transitions[:15]:
        log_info(
            f"Transition {t['from']} → {t['to']}",
            transition=f"{t['from']}→{t['to']}",
            avg_probability=round(t["avg_prob"], 4),
            n_symbols=t["n_symbols"],
        )

    # Cross-symbol stable pre-transition patterns
    log_info("=== STABLE PRE-TRANSITION PATTERNS ===")

    stable_pattern_counts = Counter()
    for _symbol, result in all_results.items():
        for pattern_key in result.get("stable_patterns", {}):
            stable_pattern_counts[pattern_key] += 1

    # Universal patterns (all symbols)
    universal_patterns = [p for p, c in stable_pattern_counts.items() if c >= len(symbols)]
    log_info(
        "Universal stable patterns",
        count=len(universal_patterns),
        patterns=universal_patterns[:20],
    )

    # Save results
    output_path = Path("/tmp/regime_transition_analysis_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Regime transition analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
