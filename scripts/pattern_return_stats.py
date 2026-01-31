#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses 1-bar forward returns without duration tracking
"""Compute return statistics for ODD robust patterns per regime.

This script computes detailed return statistics (mean, median, std,
win rate, profit factor) for each ODD robust pattern/regime combination
identified in the regime analysis.

NOTE: This is pattern-based research on 1-bar forward returns, not trading
strategy PnL with duration tracking. Simple Sharpe is appropriate here
because we're measuring statistical properties of patterns, not time-weighted
trading performance.

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
# Return Statistics
# =============================================================================


@dataclass
class PatternReturnStats:
    """Detailed return statistics for a pattern/regime combination."""

    regime: str
    pattern: str
    count: int
    # Return statistics (in basis points)
    mean_return_bps: float
    median_return_bps: float
    std_return_bps: float
    # Performance metrics
    win_rate: float  # % of positive returns
    profit_factor: float  # sum(gains) / sum(losses)
    # Tail risk
    max_gain_bps: float
    max_loss_bps: float
    var_95_bps: float  # 95% Value at Risk
    # Period consistency
    n_periods: int
    min_period_winrate: float
    max_period_winrate: float


def compute_profit_factor(returns: pd.Series) -> float:
    """Compute profit factor = sum(gains) / sum(losses)."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def analyze_pattern_returns(
    bars: pd.DataFrame,
    regime: str,
    pattern: str,
    pattern_col: str,
    regime_col: str,
    return_col: str = "fwd_ret_1",
) -> PatternReturnStats | None:
    """Compute detailed return statistics for a pattern/regime combination."""
    # Filter to specific regime and pattern
    mask = (bars[regime_col] == regime) & (bars[pattern_col] == pattern)
    subset = bars[mask]

    returns = subset[return_col].dropna()
    count = len(returns)

    if count < 100:
        return None

    # Convert to basis points (1 bp = 0.0001)
    returns_bps = returns * 10000

    # Basic stats
    mean_ret = returns_bps.mean()
    median_ret = returns_bps.median()
    std_ret = returns_bps.std()

    # Performance
    win_rate = (returns > 0).mean()
    profit_factor = compute_profit_factor(returns)

    # Tail risk
    max_gain = returns_bps.max()
    max_loss = returns_bps.min()
    var_95 = returns_bps.quantile(0.05)  # 5th percentile

    # Period consistency (quarterly)
    period_winrates = []
    subset_with_ts = subset.copy()
    if "timestamp" not in subset_with_ts.columns and isinstance(
        subset_with_ts.index, pd.DatetimeIndex
    ):
        subset_with_ts = subset_with_ts.reset_index()
        subset_with_ts = subset_with_ts.rename(
            columns={subset_with_ts.columns[0]: "timestamp"}
        )

    if "timestamp" in subset_with_ts.columns:
        subset_with_ts["timestamp"] = pd.to_datetime(subset_with_ts["timestamp"])
        subset_with_ts["quarter"] = subset_with_ts["timestamp"].dt.to_period("Q")

        for _, qtr_data in subset_with_ts.groupby("quarter"):
            qtr_returns = qtr_data[return_col].dropna()
            if len(qtr_returns) >= 30:
                period_winrates.append((qtr_returns > 0).mean())

    n_periods = len(period_winrates)
    min_period_wr = min(period_winrates) if period_winrates else 0.0
    max_period_wr = max(period_winrates) if period_winrates else 0.0

    return PatternReturnStats(
        regime=regime,
        pattern=pattern,
        count=count,
        mean_return_bps=round(mean_ret, 2),
        median_return_bps=round(median_ret, 2),
        std_return_bps=round(std_ret, 2),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        max_gain_bps=round(max_gain, 2),
        max_loss_bps=round(max_loss, 2),
        var_95_bps=round(var_95, 2),
        n_periods=n_periods,
        min_period_winrate=round(min_period_wr, 4),
        max_period_winrate=round(max_period_wr, 4),
    )


# =============================================================================
# Data Loading and Preparation
# =============================================================================


def load_and_prepare_data(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load range bars and prepare for analysis."""
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

    # Import regime analysis functions
    from regime_analysis import (
        RSIRegime,
        SMARegime,
        classify_combined_regime,
        classify_rsi_regime,
        classify_sma_regime,
        compute_forward_returns,
        compute_rsi,
        compute_sma,
        detect_2bar_patterns,
        detect_3bar_patterns,
    )

    # Add indicators
    bars["sma20"] = compute_sma(bars["Close"], 20)
    bars["sma50"] = compute_sma(bars["Close"], 50)
    bars["rsi14"] = compute_rsi(bars["Close"], 14)

    # Add regimes
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

    # Add patterns
    bars["pattern_2bar"] = detect_2bar_patterns(bars)
    bars["pattern_3bar"] = detect_3bar_patterns(bars)

    # Add forward returns
    bars = compute_forward_returns(bars, horizons=[1, 3, 5])

    return bars


# =============================================================================
# Universal ODD Robust Patterns (from prior analysis)
# =============================================================================

UNIVERSAL_2BAR_PATTERNS = {
    "chop": ["DD", "DU", "UD", "UU"],
    "bull_neutral": ["DD", "DU", "UD", "UU"],
    "bear_neutral": ["DD", "DU", "UD", "UU"],
    "bull_hot": ["UD"],
    "bear_cold": ["DD", "DU"],
}

UNIVERSAL_3BAR_PATTERNS = {
    "chop": ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"],
    "bull_neutral": ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"],
    "bear_neutral": ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"],
    "bull_hot": ["UDD", "UUU"],
    "bear_cold": ["DDD", "DDU", "DUD", "DUU"],
}


# =============================================================================
# Main Analysis
# =============================================================================


def run_return_stats_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Run return statistics analysis for universal ODD robust patterns."""
    log_info(
        "Starting return stats analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    bars = load_and_prepare_data(symbol, threshold_dbps, start_date, end_date)

    if len(bars) < 1000:
        log_warn("Insufficient data", bars=len(bars))
        return []

    results = []

    # Analyze 2-bar patterns
    for regime, patterns in UNIVERSAL_2BAR_PATTERNS.items():
        for pattern in patterns:
            stats = analyze_pattern_returns(
                bars,
                regime=regime,
                pattern=pattern,
                pattern_col="pattern_2bar",
                regime_col="regime",
                return_col="fwd_ret_1",
            )
            if stats:
                result_dict = {
                    "symbol": symbol,
                    "threshold_dbps": threshold_dbps,
                    "pattern_type": "2bar",
                    **vars(stats),
                }
                results.append(result_dict)
                log_info(
                    "Pattern stats computed",
                    regime=regime,
                    pattern=pattern,
                    mean_bps=stats.mean_return_bps,
                    win_rate=stats.win_rate,
                )

    # Analyze 3-bar patterns
    for regime, patterns in UNIVERSAL_3BAR_PATTERNS.items():
        for pattern in patterns:
            stats = analyze_pattern_returns(
                bars,
                regime=regime,
                pattern=pattern,
                pattern_col="pattern_3bar",
                regime_col="regime",
                return_col="fwd_ret_1",
            )
            if stats:
                result_dict = {
                    "symbol": symbol,
                    "threshold_dbps": threshold_dbps,
                    "pattern_type": "3bar",
                    **vars(stats),
                }
                results.append(result_dict)

    log_info(
        "Return stats analysis complete",
        symbol=symbol,
        patterns_analyzed=len(results),
    )

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting pattern return statistics analysis",
        script="pattern_return_stats.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = []

    for symbol in symbols:
        try:
            results = run_return_stats_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
            )
            all_results.extend(results)
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )

    # Summary statistics
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Best performing patterns by win rate
        best_by_winrate = results_df.nlargest(10, "win_rate")[
            ["regime", "pattern", "symbol", "win_rate", "mean_return_bps", "profit_factor"]
        ]
        log_info(
            "Top 10 patterns by win rate",
            patterns=best_by_winrate.to_dict("records"),
        )

        # Highest mean return patterns
        best_by_return = results_df.nlargest(10, "mean_return_bps")[
            ["regime", "pattern", "symbol", "mean_return_bps", "win_rate", "profit_factor"]
        ]
        log_info(
            "Top 10 patterns by mean return",
            patterns=best_by_return.to_dict("records"),
        )

        # Save results
        from pathlib import Path

        output_path = Path("/tmp/pattern_return_stats.json")
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log_info("Results saved", path=output_path, total_patterns=len(all_results))

    log_info("Pattern return statistics analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
