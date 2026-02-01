#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-factor range bar pattern analysis.

This script combines range bars at different thresholds (50, 100, 200 dbps)
as multi-factor signals to find ODD robust patterns.

Approach:
- 100 dbps: Primary signal (2-bar patterns)
- 50 dbps: Finer granularity confirmation (recent direction)
- 200 dbps: Trend filter (higher timeframe direction)

Key hypothesis: Multi-timeframe alignment may reveal stronger signals
than single-timeframe patterns.

User priorities (from AskUserQuestion):
- Focus on multi-factor patterns
- All 4 crypto symbols must pass ODD robustness
- Use |t-stat| >= 3 threshold

GitHub Issue: #52
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def log_error(message: str, **kwargs: object) -> None:
    """Log ERROR level message."""
    log_json("ERROR", message, **kwargs)


# =============================================================================
# Multi-Timeframe Alignment
# =============================================================================


def classify_bar_direction(bars: pd.DataFrame) -> pd.Series:
    """Classify each bar as 'U' (up) or 'D' (down)."""
    return (bars["Close"] >= bars["Open"]).map({True: "U", False: "D"})


def compute_htf_trend(bars: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """Compute higher-timeframe trend direction.

    Returns 'up' if majority of last N bars are up, 'down' if majority down.
    """
    directions = classify_bar_direction(bars)
    up_count = (directions == "U").rolling(window=lookback, min_periods=lookback).sum()

    def classify(count: float) -> str:
        if pd.isna(count):
            return "neutral"
        if count >= lookback * 0.67:
            return "up"
        if count <= lookback * 0.33:
            return "down"
        return "neutral"

    return up_count.apply(classify)


def align_htf_to_signal(
    signal_bars: pd.DataFrame,
    htf_bars: pd.DataFrame,
    htf_col: str,
) -> pd.Series:
    """Align higher-timeframe data to signal bars by timestamp.

    For each signal bar, find the most recent HTF bar that completed before it.
    """
    if htf_col not in htf_bars.columns:
        return pd.Series("neutral", index=signal_bars.index)

    # Get HTF values with timestamps
    htf_series = pd.Series(
        htf_bars[htf_col].values,
        index=pd.to_datetime(htf_bars.index),
    )

    # Align to signal timestamps
    signal_times = pd.to_datetime(signal_bars.index)
    aligned = []

    for sig_time in signal_times:
        prior = htf_series[htf_series.index < sig_time]
        if len(prior) > 0:
            aligned.append(prior.iloc[-1])
        else:
            aligned.append("neutral")

    return pd.Series(aligned, index=signal_bars.index)


def align_fine_to_signal(
    signal_bars: pd.DataFrame,
    fine_bars: pd.DataFrame,
    lookback_n: int = 10,
) -> pd.Series:
    """Align finer granularity direction to signal bars (VECTORIZED).

    For each signal bar, look at the direction of the last N fine bars.
    Uses merge_asof for efficient O(N log M) alignment instead of O(N*M).
    """
    # Compute rolling up ratio on fine bars
    fine_dirs = classify_bar_direction(fine_bars)
    fine_up = (fine_dirs == "U").astype(float)
    fine_up_ratio = fine_up.rolling(window=lookback_n, min_periods=1).mean()

    # Classify
    def classify_ratio(ratio: float) -> str:
        if pd.isna(ratio):
            return "neutral"
        if ratio > 0.6:
            return "up"
        if ratio < 0.4:
            return "down"
        return "neutral"

    fine_trend = fine_up_ratio.apply(classify_ratio)

    # Prepare for merge_asof
    fine_df = pd.DataFrame({
        "timestamp": pd.to_datetime(fine_bars.index),
        "fine_dir": fine_trend.to_numpy(),
    }).sort_values("timestamp")

    signal_df = pd.DataFrame({
        "timestamp": pd.to_datetime(signal_bars.index),
    }).sort_values("timestamp")

    # Merge: for each signal bar, find the most recent fine bar
    merged = pd.merge_asof(
        signal_df,
        fine_df,
        on="timestamp",
        direction="backward",
    )

    # Reindex to original signal bars order
    result = pd.Series(merged["fine_dir"].values, index=signal_bars.index)
    return result.fillna("neutral")


# =============================================================================
# ODD Robustness Testing
# =============================================================================


def ttest_1samp(data: pd.Series, popmean: float) -> tuple[float, float]:
    """One-sample t-test."""
    n = len(data)
    if n < 2:
        return 0.0, 1.0

    mean = data.mean()
    std = data.std(ddof=1)

    if std == 0:
        return 0.0, 1.0

    t_stat = (mean - popmean) / (std / math.sqrt(n))
    return t_stat, 0.0


def compute_rolling_periods(
    data: pd.DataFrame, period_months: int = 3
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Split data into rolling quarterly periods."""
    bars = data.copy()
    if "timestamp" not in bars.columns and bars.index.name != "timestamp":
        if isinstance(bars.index, pd.DatetimeIndex):
            bars = bars.reset_index()
            bars = bars.rename(columns={bars.columns[0]: "timestamp"})
        else:
            return

    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp")
    bars["period"] = bars.index.to_period(f"{period_months}M")

    for period, group in bars.groupby("period"):
        yield str(period), group


def check_multifactor_odd_robustness(
    bars_df: pd.DataFrame,
    signal_pattern_col: str,
    htf_trend_col: str,
    fine_dir_col: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 50,
    min_t_stat: float = 3.0,
) -> dict[str, dict]:
    """Check ODD robustness for multi-factor patterns.

    A pattern is: signal_pattern + htf_trend + fine_direction
    """
    results = {}

    # Create combined multi-factor label
    bars_df = bars_df.copy()
    bars_df["multifactor"] = (
        bars_df[signal_pattern_col].astype(str) + "|" +
        bars_df[htf_trend_col].astype(str) + "|" +
        bars_df[fine_dir_col].astype(str)
    )

    for mf_pattern in bars_df["multifactor"].unique():
        if pd.isna(mf_pattern) or "nan" in str(mf_pattern).lower():
            continue

        subset = bars_df[bars_df["multifactor"] == mf_pattern]

        # Collect stats per period
        period_stats = []
        for period_label, period_data in compute_rolling_periods(subset.reset_index()):
            returns = period_data[return_col].dropna()
            count = len(returns)

            if count < min_samples:
                continue

            mean_ret = returns.mean()
            t_stat, _ = ttest_1samp(returns, 0)

            period_stats.append({
                "period": period_label,
                "count": count,
                "mean_return": mean_ret,
                "t_stat": t_stat,
            })

        if len(period_stats) < 2:
            continue

        # Check ODD criteria
        t_stats = [s["t_stat"] for s in period_stats]
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)
        same_sign = all(t > 0 for t in t_stats) or all(t < 0 for t in t_stats)

        is_odd_robust = all_significant and same_sign

        # Parse pattern components
        parts = mf_pattern.split("|")
        signal_pat = parts[0] if len(parts) > 0 else ""
        htf_trend = parts[1] if len(parts) > 1 else ""
        fine_dir = parts[2] if len(parts) > 2 else ""

        results[mf_pattern] = {
            "multifactor_pattern": mf_pattern,
            "signal_pattern": signal_pat,
            "htf_trend": htf_trend,
            "fine_direction": fine_dir,
            "is_odd_robust": is_odd_robust,
            "n_periods": len(period_stats),
            "period_stats": period_stats,
            "all_significant": all_significant,
            "same_sign": same_sign,
            "min_t_stat": min(abs(t) for t in t_stats),
            "max_t_stat": max(abs(t) for t in t_stats),
            "total_count": sum(s["count"] for s in period_stats),
            "avg_return": sum(s["mean_return"] * s["count"] for s in period_stats) /
                         sum(s["count"] for s in period_stats) if period_stats else 0,
        }

    return results


# =============================================================================
# Data Loading
# =============================================================================


def load_range_bars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load range bars from cache."""
    from rangebar import get_range_bars

    return get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )


# =============================================================================
# Main Analysis
# =============================================================================


def run_multifactor_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run multi-factor pattern analysis for a symbol."""
    log_info("Starting multi-factor analysis", symbol=symbol)

    # Load all three timeframes
    log_info("Loading 100 dbps (signal)")
    bars_100 = load_range_bars(symbol, 100, start_date, end_date)

    log_info("Loading 50 dbps (fine)")
    bars_50 = load_range_bars(symbol, 50, start_date, end_date)

    log_info("Loading 200 dbps (HTF)")
    bars_200 = load_range_bars(symbol, 200, start_date, end_date)

    log_info(
        "Data loaded",
        bars_100=len(bars_100),
        bars_50=len(bars_50),
        bars_200=len(bars_200),
    )

    if len(bars_100) < 1000:
        log_warn("Insufficient 100 dbps data", count=len(bars_100))
        return {"error": "insufficient_data"}

    # Prepare signal bars (100 dbps)
    signal_bars = bars_100.copy()

    # Add 2-bar pattern
    directions = classify_bar_direction(signal_bars)
    signal_bars["signal_pattern"] = directions.shift(1) + directions

    # Add forward returns
    signal_bars["fwd_ret_1"] = signal_bars["Close"].shift(-1) / signal_bars["Close"] - 1

    # Compute HTF trend from 200 dbps
    if len(bars_200) > 10:
        bars_200["htf_trend"] = compute_htf_trend(bars_200, lookback=3)
        signal_bars["htf_trend"] = align_htf_to_signal(signal_bars, bars_200, "htf_trend")
    else:
        signal_bars["htf_trend"] = "neutral"

    # Compute fine direction from 50 dbps
    if len(bars_50) > 100:
        signal_bars["fine_dir"] = align_fine_to_signal(signal_bars, bars_50, lookback_seconds=300)
    else:
        signal_bars["fine_dir"] = "neutral"

    # Distribution check
    htf_dist = signal_bars["htf_trend"].value_counts().to_dict()
    fine_dist = signal_bars["fine_dir"].value_counts().to_dict()
    log_info("HTF trend distribution", **htf_dist)
    log_info("Fine direction distribution", **fine_dist)

    # Test ODD robustness
    log_info("Testing multi-factor ODD robustness (t-stat >= 3)")
    results = check_multifactor_odd_robustness(
        signal_bars,
        signal_pattern_col="signal_pattern",
        htf_trend_col="htf_trend",
        fine_dir_col="fine_dir",
        min_samples=50,
        min_t_stat=3.0,
    )

    # Filter to ODD robust patterns
    robust_patterns = {k: v for k, v in results.items() if v["is_odd_robust"]}

    log_info(
        "Analysis complete",
        symbol=symbol,
        total_patterns=len(results),
        robust_patterns=len(robust_patterns),
    )

    # Log robust patterns
    for pattern, stats in robust_patterns.items():
        log_info(
            "ODD robust multi-factor pattern",
            pattern=pattern,
            signal=stats["signal_pattern"],
            htf=stats["htf_trend"],
            fine=stats["fine_direction"],
            n_periods=stats["n_periods"],
            min_t=stats["min_t_stat"],
            avg_return_bps=round(stats["avg_return"] * 10000, 2),
        )

    return {
        "symbol": symbol,
        "bars_100": len(bars_100),
        "bars_50": len(bars_50),
        "bars_200": len(bars_200),
        "total_patterns": len(results),
        "robust_patterns_count": len(robust_patterns),
        "robust_patterns": list(robust_patterns.keys()),
        "results": results,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting multi-factor pattern analysis",
        script="multifactor_patterns.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}
    cross_symbol_robust = []

    for symbol in symbols:
        try:
            results = run_multifactor_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[symbol] = results

            if "robust_patterns" in results:
                for pattern in results["robust_patterns"]:
                    cross_symbol_robust.append({
                        "symbol": symbol,
                        "pattern": pattern,
                    })

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )

    # Cross-symbol validation
    log_info("=== CROSS-SYMBOL VALIDATION ===")

    from collections import Counter

    pattern_counts = Counter(p["pattern"] for p in cross_symbol_robust)
    universal_patterns = [p for p, count in pattern_counts.items() if count >= len(symbols)]

    log_info(
        "Universal ODD robust patterns (all 4 symbols)",
        count=len(universal_patterns),
        patterns=universal_patterns,
    )

    # Patterns present in at least 3 symbols
    strong_patterns = [p for p, count in pattern_counts.items() if count >= 3]
    log_info(
        "Strong patterns (3+ symbols)",
        count=len(strong_patterns),
        patterns=strong_patterns,
    )

    # Save results
    from pathlib import Path

    output_path = Path("/tmp/multifactor_patterns.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Multi-factor pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
