#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Volume-conditioned pattern analysis for ODD robustness.

This script tests if conditioning directional patterns on volume metrics
yields ODD robust signals:
1. High/low volume relative to moving average
2. Buy/sell volume imbalance (OFI - Order Flow Imbalance)
3. Duration-based patterns (fast vs slow bars)

Key hypothesis: Volume confirmation may filter noise from directional patterns.

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
# Volume Conditioning
# =============================================================================


def compute_volume_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Compute volume moving average."""
    return volume.rolling(window=window, min_periods=window).mean()


def classify_volume_regime(volume: float, volume_ma: float) -> str:
    """Classify volume as high/normal/low relative to MA."""
    if pd.isna(volume_ma) or volume_ma == 0:
        return "normal"

    ratio = volume / volume_ma
    if ratio > 1.5:
        return "high"
    if ratio < 0.5:
        return "low"
    return "normal"


def compute_ofi(bars: pd.DataFrame) -> pd.Series:
    """Compute Order Flow Imbalance from microstructure features.

    OFI = (buy_volume - sell_volume) / total_volume
    Range: [-1, 1]
    """
    if "ofi" in bars.columns:
        return bars["ofi"]

    # Fallback: estimate from bar direction and volume
    direction = (bars["Close"] >= bars["Open"]).astype(int)
    # Assume all volume is in the direction of the bar
    return direction * 2 - 1  # Maps 0,1 to -1,1


def classify_ofi_regime(ofi: float) -> str:
    """Classify OFI as buy/neutral/sell."""
    if pd.isna(ofi):
        return "neutral"

    if ofi > 0.3:
        return "buy"
    if ofi < -0.3:
        return "sell"
    return "neutral"


def compute_duration_percentile(bars: pd.DataFrame, window: int = 100) -> pd.Series:
    """Compute rolling percentile of bar duration."""
    if "duration_us" not in bars.columns:
        # Fall back to estimating from timestamp
        if isinstance(bars.index, pd.DatetimeIndex):
            duration = bars.index.to_series().diff().dt.total_seconds() * 1e6
            return duration.rolling(window=window).rank(pct=True)
        return pd.Series(0.5, index=bars.index)

    return bars["duration_us"].rolling(window=window).rank(pct=True)


def classify_duration_regime(duration_pct: float) -> str:
    """Classify bar as fast/normal/slow based on duration percentile."""
    if pd.isna(duration_pct):
        return "normal"

    if duration_pct < 0.25:
        return "fast"
    if duration_pct > 0.75:
        return "slow"
    return "normal"


# =============================================================================
# Pattern Detection
# =============================================================================


def classify_bar_direction(bars: pd.DataFrame) -> pd.Series:
    """Classify each bar as 'U' (up) or 'D' (down)."""
    return (bars["Close"] >= bars["Open"]).map({True: "U", False: "D"})


def detect_2bar_patterns(bars: pd.DataFrame) -> pd.Series:
    """Detect 2-bar patterns (e.g., UU, UD, DU, DD)."""
    directions = classify_bar_direction(bars)
    return directions.shift(1) + directions


# =============================================================================
# ODD Robustness Testing
# =============================================================================


def ttest_1samp(data: pd.Series, popmean: float) -> tuple[float, float]:
    """One-sample t-test (pure Python implementation)."""
    n = len(data)
    if n < 2:
        return 0.0, 1.0

    mean = data.mean()
    std = data.std(ddof=1)

    if std == 0:
        return 0.0, 1.0

    t_stat = (mean - popmean) / (std / math.sqrt(n))
    return t_stat, 0.0  # p-value not needed for this analysis


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
            log_error("No timestamp column found")
            return

    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp")
    bars["period"] = bars.index.to_period(f"{period_months}M")

    for period, group in bars.groupby("period"):
        yield str(period), group


def check_odd_robustness(
    bars_df: pd.DataFrame,
    pattern_col: str,
    condition_col: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict[str, dict]:
    """Check ODD robustness for volume-conditioned patterns."""
    results = {}

    for condition in bars_df[condition_col].unique():
        if pd.isna(condition):
            continue

        cond_subset = bars_df[bars_df[condition_col] == condition]

        for pattern in cond_subset[pattern_col].unique():
            if pd.isna(pattern):
                continue

            pattern_subset = cond_subset[cond_subset[pattern_col] == pattern]

            # Collect stats per period
            period_stats = []
            for period_label, period_data in compute_rolling_periods(
                pattern_subset.reset_index()
            ):
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

            key = f"{condition}|{pattern}"
            results[key] = {
                "condition": str(condition),
                "pattern": str(pattern),
                "is_odd_robust": is_odd_robust,
                "n_periods": len(period_stats),
                "period_stats": period_stats,
                "all_significant": all_significant,
                "same_sign": same_sign,
                "min_t_stat": min(abs(t) for t in t_stats),
                "max_t_stat": max(abs(t) for t in t_stats),
            }

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
    """Load range bars and add volume conditioning columns."""
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
        include_microstructure=True,
    )

    log_info("Loaded bars", count=len(bars))

    # Volume conditioning
    bars["volume_ma20"] = compute_volume_ma(bars["Volume"], 20)
    bars["volume_regime"] = bars.apply(
        lambda row: classify_volume_regime(row["Volume"], row["volume_ma20"]),
        axis=1,
    )

    # OFI conditioning
    bars["ofi_value"] = compute_ofi(bars)
    bars["ofi_regime"] = bars["ofi_value"].apply(classify_ofi_regime)

    # Duration conditioning
    bars["duration_pct"] = compute_duration_percentile(bars)
    bars["duration_regime"] = bars["duration_pct"].apply(classify_duration_regime)

    # Patterns
    bars["pattern_2bar"] = detect_2bar_patterns(bars)

    # Forward returns
    bars["fwd_ret_1"] = bars["Close"].shift(-1) / bars["Close"] - 1

    return bars


# =============================================================================
# Main Analysis
# =============================================================================


def run_volume_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Run volume-conditioned pattern analysis."""
    log_info(
        "Starting volume-conditioned analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
    )

    bars = load_and_prepare_data(symbol, threshold_dbps, start_date, end_date)

    if len(bars) < 1000:
        log_warn("Insufficient data", bars=len(bars))
        return {"error": "insufficient_data"}

    # Distribution of conditions
    vol_dist = bars["volume_regime"].value_counts().to_dict()
    ofi_dist = bars["ofi_regime"].value_counts().to_dict()
    dur_dist = bars["duration_regime"].value_counts().to_dict()

    log_info("Volume regime distribution", **vol_dist)
    log_info("OFI regime distribution", **ofi_dist)
    log_info("Duration regime distribution", **dur_dist)

    # Test ODD robustness for each conditioning type
    log_info("Testing volume-conditioned patterns")
    results_volume = check_odd_robustness(
        bars,
        pattern_col="pattern_2bar",
        condition_col="volume_regime",
        min_samples=100,
        min_t_stat=5.0,
    )

    log_info("Testing OFI-conditioned patterns")
    results_ofi = check_odd_robustness(
        bars,
        pattern_col="pattern_2bar",
        condition_col="ofi_regime",
        min_samples=100,
        min_t_stat=5.0,
    )

    log_info("Testing duration-conditioned patterns")
    results_duration = check_odd_robustness(
        bars,
        pattern_col="pattern_2bar",
        condition_col="duration_regime",
        min_samples=100,
        min_t_stat=5.0,
    )

    # Count ODD robust patterns
    robust_volume = [k for k, v in results_volume.items() if v["is_odd_robust"]]
    robust_ofi = [k for k, v in results_ofi.items() if v["is_odd_robust"]]
    robust_duration = [k for k, v in results_duration.items() if v["is_odd_robust"]]

    log_info(
        "Analysis complete",
        symbol=symbol,
        robust_volume=len(robust_volume),
        robust_ofi=len(robust_ofi),
        robust_duration=len(robust_duration),
    )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "total_bars": len(bars),
        "volume_distribution": vol_dist,
        "ofi_distribution": ofi_dist,
        "duration_distribution": dur_dist,
        "results_volume": results_volume,
        "results_ofi": results_ofi,
        "results_duration": results_duration,
        "robust_volume": robust_volume,
        "robust_ofi": robust_ofi,
        "robust_duration": robust_duration,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting volume-conditioned pattern analysis",
        script="volume_conditioned_patterns.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"

    all_results = {}

    for symbol in symbols:
        try:
            results = run_volume_analysis(
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

    # Summary
    total_robust_volume = sum(
        len(r.get("robust_volume", [])) for r in all_results.values() if "error" not in r
    )
    total_robust_ofi = sum(
        len(r.get("robust_ofi", [])) for r in all_results.values() if "error" not in r
    )
    total_robust_duration = sum(
        len(r.get("robust_duration", [])) for r in all_results.values() if "error" not in r
    )

    log_info(
        "Overall summary",
        total_robust_volume=total_robust_volume,
        total_robust_ofi=total_robust_ofi,
        total_robust_duration=total_robust_duration,
    )

    # Save results
    from pathlib import Path

    output_path = Path("/tmp/volume_conditioned_patterns.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Volume-conditioned pattern analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
