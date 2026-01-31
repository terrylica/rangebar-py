#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Multi-bar continuation probability analysis.

This script investigates genuine predictive signals by analyzing whether
the current bar direction predicts FUTURE bar directions beyond the
immediate next bar (which is mechanically determined by construction).

Key question: Does DD predict DDD? Does UU predict UUU?
If continuation probability > 50%, there is genuine momentum.

GitHub Issue: #52
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


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
# Continuation Analysis
# =============================================================================


@dataclass
class ContinuationStats:
    """Statistics for continuation probability."""

    pattern: str
    n_lookback: int
    total_count: int
    continuation_count: int
    continuation_prob: float
    reversal_prob: float
    z_score: float  # Statistical significance


def compute_z_score(p: float, n: int, p0: float = 0.5) -> float:
    """Compute z-score for proportion test.

    Tests if observed probability p differs from null hypothesis p0.
    """
    import math

    if n < 2:
        return 0.0

    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 0.0

    return (p - p0) / se


def classify_bar_direction(bars: pd.DataFrame) -> pd.Series:
    """Classify each bar as 'U' (up) or 'D' (down)."""
    return (bars["Close"] >= bars["Open"]).map({True: "U", False: "D"})


def analyze_continuation(
    bars: pd.DataFrame,
    lookback: int = 2,
    forward: int = 1,
) -> list[ContinuationStats]:
    """Analyze continuation probability for directional patterns.

    Args:
        bars: DataFrame with OHLC data
        lookback: Number of bars in the pattern (e.g., 2 for DD, UU)
        forward: Number of bars forward to predict

    Returns:
        List of ContinuationStats for each pattern
    """
    directions = classify_bar_direction(bars)

    # Build lookback patterns
    pattern_cols = []
    for i in range(lookback):
        shifted = directions.shift(i)
        pattern_cols.append(shifted)

    # Combine into pattern string (e.g., "DD", "UU")
    pattern = pattern_cols[0]
    for i in range(1, lookback):
        pattern = pattern_cols[i] + pattern  # Prepend older bars

    # Get forward direction
    forward_dir = directions.shift(-forward)

    # Create analysis DataFrame
    analysis = pd.DataFrame({
        "pattern": pattern,
        "forward_dir": forward_dir,
    }).dropna()

    results = []

    # Analyze each uniform pattern (DD, UU, DDD, UUU, etc.)
    for base_dir in ["D", "U"]:
        uniform_pattern = base_dir * lookback

        subset = analysis[analysis["pattern"] == uniform_pattern]
        total = len(subset)

        if total < 100:
            continue

        # Count continuations (same direction forward)
        continuation_count = (subset["forward_dir"] == base_dir).sum()
        continuation_prob = continuation_count / total
        reversal_prob = 1 - continuation_prob

        # z-score for statistical significance (H0: p = 0.5)
        z_score = compute_z_score(continuation_prob, total)

        results.append(
            ContinuationStats(
                pattern=uniform_pattern,
                n_lookback=lookback,
                total_count=total,
                continuation_count=continuation_count,
                continuation_prob=round(continuation_prob, 4),
                reversal_prob=round(reversal_prob, 4),
                z_score=round(z_score, 2),
            )
        )

    return results


def analyze_regime_continuation(
    bars: pd.DataFrame,
    regime_col: str = "regime",
    lookback: int = 2,
    forward: int = 1,
) -> dict[str, list[ContinuationStats]]:
    """Analyze continuation probability within each regime."""
    results = {}

    for regime in bars[regime_col].unique():
        if pd.isna(regime):
            continue

        regime_bars = bars[bars[regime_col] == regime]
        if len(regime_bars) < 1000:
            continue

        stats = analyze_continuation(regime_bars, lookback, forward)
        if stats:
            results[regime] = stats

    return results


# =============================================================================
# Data Loading
# =============================================================================


def load_and_prepare_data(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load range bars and add regime classifications."""
    from rangebar import get_range_bars

    log_info(
        "Loading range bars",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    bars = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    log_info("Loaded bars", count=len(bars))

    # Add regime classification
    from regime_analysis import (
        RSIRegime,
        SMARegime,
        classify_combined_regime,
        classify_rsi_regime,
        classify_sma_regime,
        compute_rsi,
        compute_sma,
    )

    bars["sma20"] = compute_sma(bars["Close"], 20)
    bars["sma50"] = compute_sma(bars["Close"], 50)
    bars["rsi14"] = compute_rsi(bars["Close"], 14)

    bars["sma_regime"] = bars.apply(
        lambda row: classify_sma_regime(
            row["Close"], row["sma20"], row["sma50"]
        ).value,
        axis=1,
    )
    bars["rsi_regime"] = bars["rsi14"].apply(
        lambda rsi: classify_rsi_regime(rsi).value
    )
    bars["regime"] = bars.apply(
        lambda row: classify_combined_regime(
            SMARegime(row["sma_regime"]),
            RSIRegime(row["rsi_regime"]),
        ).value,
        axis=1,
    )

    return bars


# =============================================================================
# Main Analysis
# =============================================================================


def run_continuation_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Run multi-bar continuation analysis."""
    log_info(
        "Starting continuation analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    bars = load_and_prepare_data(symbol, threshold_dbps, start_date, end_date)

    if len(bars) < 1000:
        log_warn("Insufficient data", bars=len(bars))
        return {"error": "insufficient_data"}

    # Overall continuation (no regime filter)
    log_info("Analyzing overall continuation probabilities")

    overall_results = {}
    for lookback in [2, 3, 4, 5]:
        stats = analyze_continuation(bars, lookback=lookback, forward=1)
        for s in stats:
            log_info(
                "Overall continuation",
                pattern=s.pattern,
                prob=s.continuation_prob,
                z_score=s.z_score,
                count=s.total_count,
            )
        overall_results[f"lookback_{lookback}"] = [vars(s) for s in stats]

    # Per-regime continuation
    log_info("Analyzing per-regime continuation probabilities")

    regime_results = {}
    for lookback in [2, 3]:
        regime_stats = analyze_regime_continuation(
            bars, regime_col="regime", lookback=lookback, forward=1
        )
        for regime, stats in regime_stats.items():
            for s in stats:
                if abs(s.z_score) >= 2:  # Significant at p < 0.05
                    log_info(
                        "Regime continuation (significant)",
                        regime=regime,
                        pattern=s.pattern,
                        prob=s.continuation_prob,
                        z_score=s.z_score,
                    )
            regime_results[f"{regime}_lookback_{lookback}"] = [vars(s) for s in stats]

    log_info(
        "Continuation analysis complete",
        symbol=symbol,
    )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "total_bars": len(bars),
        "overall": overall_results,
        "by_regime": regime_results,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting multi-bar continuation analysis",
        script="multibar_continuation.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}

    for symbol in symbols:
        try:
            results = run_continuation_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[symbol] = results
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )

    # Summary: find patterns with significant momentum
    log_info("=== MOMENTUM SUMMARY ===")

    significant_momentum = []
    for symbol, results in all_results.items():
        if "error" in results:
            continue

        for _key, stats_list in results.get("overall", {}).items():
            for stats in stats_list:
                if abs(stats.get("z_score", 0)) >= 3:  # p < 0.001
                    significant_momentum.append({
                        "symbol": symbol,
                        "pattern": stats["pattern"],
                        "continuation_prob": stats["continuation_prob"],
                        "z_score": stats["z_score"],
                        "count": stats["total_count"],
                    })

    if significant_momentum:
        log_info(
            "Patterns with significant momentum (|z| >= 3)",
            count=len(significant_momentum),
            patterns=significant_momentum,
        )
    else:
        log_info("No patterns with significant momentum found")

    # Save results
    from pathlib import Path

    output_path = Path("/tmp/multibar_continuation.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Multi-bar continuation analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
