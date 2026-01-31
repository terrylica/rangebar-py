#!/usr/bin/env python3
"""200 dbps Trend Filter Confirmation Analysis.

This script tests if adding a 200 dbps trend filter to the regime-based
patterns improves ODD robustness.

Methodology:
1. Load 100 dbps bars for pattern detection and regime classification
2. Load 200 dbps bars for higher-timeframe trend determination
3. Align 200 dbps trend to each 100 dbps bar (last completed 200 dbps bar)
4. Test if patterns filtered by BOTH regime AND trend are more robust

GitHub Issue: #52
"""

from __future__ import annotations

import json
import sys
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
# Trend Filter Logic
# =============================================================================


def compute_trend_direction(bars: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """Compute trend direction from the last N bars.

    Args:
        bars: DataFrame with Open, Close columns
        lookback: Number of bars to consider

    Returns:
        Series with trend direction: 'up', 'down', or 'neutral'
    """
    # Count up vs down bars in lookback window
    directions = (bars["Close"] >= bars["Open"]).astype(int)

    # Rolling sum of up bars
    up_count = directions.rolling(window=lookback, min_periods=lookback).sum()

    # Classify trend
    def classify_trend(up: float) -> str:
        if pd.isna(up):
            return "neutral"
        if up >= lookback * 0.67:  # 2+ of 3 bars up
            return "up"
        if up <= lookback * 0.33:  # 2+ of 3 bars down
            return "down"
        return "neutral"

    return up_count.apply(classify_trend)


def align_htf_trend(
    signal_bars: pd.DataFrame,
    htf_bars: pd.DataFrame,
    htf_trend: pd.Series,
) -> pd.Series:
    """Align higher-timeframe trend to signal bars.

    For each signal bar, find the most recent completed HTF bar
    and use its trend direction.

    Args:
        signal_bars: Lower-timeframe bars (100 dbps)
        htf_bars: Higher-timeframe bars (200 dbps)
        htf_trend: Trend direction for each HTF bar

    Returns:
        Series aligned to signal_bars index
    """
    # Get HTF bar open times as index
    htf_trend_series = pd.Series(
        htf_trend.values,
        index=pd.to_datetime(htf_bars.index),
    )

    # For each signal bar, find the most recent HTF bar
    aligned = []
    signal_times = pd.to_datetime(signal_bars.index)

    for sig_time in signal_times:
        # Get all HTF bars that opened before this signal bar
        prior_htf = htf_trend_series[htf_trend_series.index < sig_time]
        if len(prior_htf) > 0:
            aligned.append(prior_htf.iloc[-1])
        else:
            aligned.append("neutral")

    return pd.Series(aligned, index=signal_bars.index)


# =============================================================================
# Data Loading
# =============================================================================


def load_range_bars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load range bars from ClickHouse cache."""
    from rangebar import get_range_bars

    log_info(
        "Loading range bars",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        start_date=start_date,
        end_date=end_date,
    )

    bars = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    log_info(
        "Range bars loaded",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        bars=len(bars),
    )

    return bars


# =============================================================================
# Main Analysis
# =============================================================================


def run_trend_filter_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    signal_threshold: int = 100,
    htf_threshold: int = 200,
    lookback: int = 3,
) -> dict:
    """Run trend filter confirmation analysis.

    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        signal_threshold: Signal bar threshold (100 dbps)
        htf_threshold: Higher-timeframe trend threshold (200 dbps)
        lookback: Bars to consider for trend

    Returns:
        Analysis results
    """
    log_info(
        "Starting trend filter analysis",
        symbol=symbol,
        signal_threshold=signal_threshold,
        htf_threshold=htf_threshold,
    )

    # Load both timeframes
    signal_bars = load_range_bars(symbol, signal_threshold, start_date, end_date)
    htf_bars = load_range_bars(symbol, htf_threshold, start_date, end_date)

    if len(signal_bars) < 1000 or len(htf_bars) < 100:
        log_warn(
            "Insufficient data",
            signal_bars=len(signal_bars),
            htf_bars=len(htf_bars),
        )
        return {"error": "insufficient_data"}

    # Compute HTF trend
    htf_trend = compute_trend_direction(htf_bars, lookback=lookback)
    log_info(
        "HTF trend distribution",
        up=int((htf_trend == "up").sum()),
        down=int((htf_trend == "down").sum()),
        neutral=int((htf_trend == "neutral").sum()),
    )

    # Align HTF trend to signal bars
    signal_bars = signal_bars.copy()
    signal_bars["htf_trend"] = align_htf_trend(signal_bars, htf_bars, htf_trend)

    htf_aligned_counts = signal_bars["htf_trend"].value_counts().to_dict()
    log_info("HTF trend aligned to signal bars", **htf_aligned_counts)

    # Import regime analysis functions
    from regime_analysis import (
        RSIRegime,
        SMARegime,
        check_odd_robustness,
        classify_combined_regime,
        classify_rsi_regime,
        classify_sma_regime,
        compute_forward_returns,
        compute_rsi,
        compute_sma,
        detect_2bar_patterns,
        detect_3bar_patterns,
    )

    # Prepare signal bars with regimes
    signal_bars["sma20"] = compute_sma(signal_bars["Close"], 20)
    signal_bars["sma50"] = compute_sma(signal_bars["Close"], 50)
    signal_bars["rsi14"] = compute_rsi(signal_bars["Close"], 14)

    signal_bars["sma_regime"] = signal_bars.apply(
        lambda row: classify_sma_regime(
            row["Close"], row["sma20"], row["sma50"]
        ).value,
        axis=1,
    )
    signal_bars["rsi_regime"] = signal_bars["rsi14"].apply(
        lambda rsi: classify_rsi_regime(rsi).value
    )
    signal_bars["regime"] = signal_bars.apply(
        lambda row: classify_combined_regime(
            SMARegime(row["sma_regime"]),
            RSIRegime(row["rsi_regime"]),
        ).value,
        axis=1,
    )

    # Add combined regime + trend filter
    signal_bars["regime_trend"] = (
        signal_bars["regime"] + "_" + signal_bars["htf_trend"]
    )

    # Add patterns
    signal_bars["pattern_2bar"] = detect_2bar_patterns(signal_bars)
    signal_bars["pattern_3bar"] = detect_3bar_patterns(signal_bars)

    # Add forward returns
    signal_bars = compute_forward_returns(signal_bars, horizons=[1, 3, 5])

    # Test ODD robustness with regime only
    log_info("Testing ODD robustness: regime only")
    results_regime_only = check_odd_robustness(
        signal_bars,
        pattern_col="pattern_2bar",
        regime_col="regime",
        return_col="fwd_ret_1",
        min_samples=100,
        min_t_stat=5.0,
    )
    robust_regime_only = [
        k for k, v in results_regime_only.items() if v["is_odd_robust"]
    ]

    # Test ODD robustness with regime + trend filter
    log_info("Testing ODD robustness: regime + HTF trend")
    results_regime_trend = check_odd_robustness(
        signal_bars,
        pattern_col="pattern_2bar",
        regime_col="regime_trend",
        return_col="fwd_ret_1",
        min_samples=100,
        min_t_stat=5.0,
    )
    robust_regime_trend = [
        k for k, v in results_regime_trend.items() if v["is_odd_robust"]
    ]

    log_info(
        "Comparison: regime only vs regime+trend",
        regime_only_robust=len(robust_regime_only),
        regime_trend_robust=len(robust_regime_trend),
    )

    # Analyze which trend directions improve robustness
    trend_up_patterns = [p for p in robust_regime_trend if "_up" in p]
    trend_down_patterns = [p for p in robust_regime_trend if "_down" in p]
    trend_neutral_patterns = [p for p in robust_regime_trend if "_neutral" in p]

    log_info(
        "Robust patterns by HTF trend",
        up_trend=len(trend_up_patterns),
        down_trend=len(trend_down_patterns),
        neutral_trend=len(trend_neutral_patterns),
    )

    return {
        "symbol": symbol,
        "signal_threshold": signal_threshold,
        "htf_threshold": htf_threshold,
        "signal_bars": len(signal_bars),
        "htf_bars": len(htf_bars),
        "robust_regime_only": robust_regime_only,
        "robust_regime_trend": robust_regime_trend,
        "improvement": len(robust_regime_trend) - len(robust_regime_only),
        "trend_up_patterns": trend_up_patterns,
        "trend_down_patterns": trend_down_patterns,
        "trend_neutral_patterns": trend_neutral_patterns,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting 200 dbps trend filter analysis",
        script="trend_filter_analysis.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}

    for symbol in symbols:
        try:
            result = run_trend_filter_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                signal_threshold=100,
                htf_threshold=200,
                lookback=3,
            )
            all_results[symbol] = result

            log_info(
                "Analysis complete",
                symbol=symbol,
                regime_only=len(result.get("robust_regime_only", [])),
                regime_trend=len(result.get("robust_regime_trend", [])),
                improvement=result.get("improvement", 0),
            )

        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_error(
                "Analysis failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            all_results[symbol] = {"error": str(e)}

    # Summary across all symbols
    total_regime_only = sum(
        len(r.get("robust_regime_only", []))
        for r in all_results.values()
        if "error" not in r
    )
    total_regime_trend = sum(
        len(r.get("robust_regime_trend", []))
        for r in all_results.values()
        if "error" not in r
    )

    log_info(
        "Overall summary",
        total_regime_only=total_regime_only,
        total_regime_trend=total_regime_trend,
        net_improvement=total_regime_trend - total_regime_only,
    )

    log_info("Trend filter analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
