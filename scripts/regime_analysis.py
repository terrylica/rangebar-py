#!/usr/bin/env python3
"""Market regime analysis for ODD robust multi-factor range bar patterns.

This script implements the methodology described in docs/research/market-regime-patterns.md.
It analyzes range bar patterns WITHIN specific market regimes to find ODD robust combinations.

GitHub Issue: #52
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd


def ttest_1samp(data: pd.Series, popmean: float) -> tuple[float, float]:
    """One-sample t-test (pure Python implementation).

    Tests if the mean of data differs from popmean.

    Args:
        data: Sample data
        popmean: Population mean to test against

    Returns:
        Tuple of (t_statistic, p_value)
    """
    n = len(data)
    if n < 2:
        return 0.0, 1.0

    mean = data.mean()
    std = data.std(ddof=1)  # Sample standard deviation

    if std == 0:
        return 0.0, 1.0

    # t = (sample_mean - pop_mean) / (std / sqrt(n))
    t_stat = (mean - popmean) / (std / math.sqrt(n))

    # Approximate p-value using normal distribution for large n
    # For n > 30, t-distribution approaches normal
    degrees_of_freedom = n - 1
    if degrees_of_freedom > 30:
        # Use normal approximation
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    else:
        # Use t-distribution approximation (normal approx for simplicity)
        p_value = 2 * _norm_cdf(-abs(t_stat))

    return t_stat, p_value


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

if TYPE_CHECKING:
    from collections.abc import Iterator


# =============================================================================
# NDJSON Logging
# =============================================================================


def log_json(level: str, message: str, **kwargs: object) -> None:
    """Log a message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(tz=datetime.UTC).isoformat(),
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
# Market Regime Definitions
# =============================================================================


class SMARegime(Enum):
    """SMA-based market regime."""

    UPTREND = "uptrend"  # Price > SMA20 > SMA50
    DOWNTREND = "downtrend"  # Price < SMA20 < SMA50
    CONSOLIDATION = "consolidation"  # Neither uptrend nor downtrend


class RSIRegime(Enum):
    """RSI-based market regime."""

    OVERBOUGHT = "overbought"  # RSI > 70
    OVERSOLD = "oversold"  # RSI < 30
    NEUTRAL = "neutral"  # 30 <= RSI <= 70


class CombinedRegime(Enum):
    """Combined SMA + RSI regime."""

    BULL_HOT = "bull_hot"  # Uptrend + Overbought
    BULL_NEUTRAL = "bull_neutral"  # Uptrend + Neutral
    BULL_COLD = "bull_cold"  # Uptrend + Oversold (rare)
    BEAR_HOT = "bear_hot"  # Downtrend + Overbought (rare)
    BEAR_NEUTRAL = "bear_neutral"  # Downtrend + Neutral
    BEAR_COLD = "bear_cold"  # Downtrend + Oversold
    CHOP = "chop"  # Consolidation + Any RSI


@dataclass
class RegimeStats:
    """Statistics for a pattern within a regime."""

    regime: str
    pattern: str
    count: int
    win_rate: float
    mean_return: float
    std_return: float
    t_stat: float
    p_value: float


# =============================================================================
# Technical Indicators
# =============================================================================


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over window
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def classify_sma_regime(
    close: float, sma20: float, sma50: float
) -> SMARegime:
    """Classify SMA-based regime for a single bar."""
    if pd.isna(sma20) or pd.isna(sma50):
        return SMARegime.CONSOLIDATION

    if close > sma20 > sma50:
        return SMARegime.UPTREND
    if close < sma20 < sma50:
        return SMARegime.DOWNTREND
    return SMARegime.CONSOLIDATION


def classify_rsi_regime(rsi: float) -> RSIRegime:
    """Classify RSI-based regime for a single bar."""
    if pd.isna(rsi):
        return RSIRegime.NEUTRAL

    if rsi > 70:
        return RSIRegime.OVERBOUGHT
    if rsi < 30:
        return RSIRegime.OVERSOLD
    return RSIRegime.NEUTRAL


def classify_combined_regime(
    sma_regime: SMARegime, rsi_regime: RSIRegime
) -> CombinedRegime:
    """Classify combined regime from SMA and RSI regimes."""
    if sma_regime == SMARegime.CONSOLIDATION:
        return CombinedRegime.CHOP

    if sma_regime == SMARegime.UPTREND:
        if rsi_regime == RSIRegime.OVERBOUGHT:
            return CombinedRegime.BULL_HOT
        if rsi_regime == RSIRegime.OVERSOLD:
            return CombinedRegime.BULL_COLD
        return CombinedRegime.BULL_NEUTRAL

    # Downtrend
    if rsi_regime == RSIRegime.OVERBOUGHT:
        return CombinedRegime.BEAR_HOT
    if rsi_regime == RSIRegime.OVERSOLD:
        return CombinedRegime.BEAR_COLD
    return CombinedRegime.BEAR_NEUTRAL


# =============================================================================
# Pattern Detection
# =============================================================================


def classify_bar_direction(bar: pd.Series) -> str:
    """Classify bar as Up (U) or Down (D) based on close vs open."""
    if bar["Close"] >= bar["Open"]:
        return "U"
    return "D"


def detect_2bar_patterns(bars: pd.DataFrame) -> pd.Series:
    """Detect 2-bar patterns (e.g., UU, UD, DU, DD)."""
    directions = bars.apply(classify_bar_direction, axis=1)
    return directions + directions.shift(-1)


def detect_3bar_patterns(bars: pd.DataFrame) -> pd.Series:
    """Detect 3-bar patterns (e.g., UUU, UUD, UDU, etc.)."""
    directions = bars.apply(classify_bar_direction, axis=1)
    return directions + directions.shift(-1) + directions.shift(-2)


# =============================================================================
# Forward Returns
# =============================================================================


def compute_forward_returns(
    df: pd.DataFrame, horizons: list[int] | None = None
) -> pd.DataFrame:
    """Compute forward returns at multiple horizons.

    Args:
        df: DataFrame with Close prices
        horizons: List of bar horizons (default: [1, 3, 5])

    Returns:
        DataFrame with forward return columns
    """
    if horizons is None:
        horizons = [1, 3, 5]

    result = df.copy()
    for h in horizons:
        result[f"fwd_ret_{h}"] = (
            df["Close"].shift(-h) / df["Close"] - 1
        )
    return result


# =============================================================================
# ODD Robustness Testing
# =============================================================================


def compute_rolling_periods(
    data: pd.DataFrame, period_months: int = 3
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Split data into rolling quarterly periods.

    Yields:
        Tuple of (period_label, period_df)
    """
    bars = data.copy()
    if "timestamp" not in bars.columns and bars.index.name != "timestamp":
        # Try to use index as timestamp
        if isinstance(bars.index, pd.DatetimeIndex):
            bars = bars.reset_index()
            bars = bars.rename(columns={bars.columns[0]: "timestamp"})
        else:
            log_error("No timestamp column found")
            return

    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp")

    # Create period labels
    bars["period"] = bars.index.to_period(f"{period_months}M")

    for period, group in bars.groupby("period"):
        yield str(period), group


def compute_pattern_stats(
    data: pd.DataFrame,
    pattern_col: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 100,
) -> list[RegimeStats]:
    """Compute statistics for each pattern.

    Args:
        df: DataFrame with pattern and return columns
        pattern_col: Column name for pattern labels
        return_col: Column name for forward returns
        min_samples: Minimum samples required

    Returns:
        List of RegimeStats for each pattern
    """
    results = []

    for pattern, group in data.groupby(pattern_col):
        if pd.isna(pattern):
            continue

        returns = group[return_col].dropna()
        count = len(returns)

        if count < min_samples:
            continue

        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret > 0:
            t_stat, p_value = ttest_1samp(returns, 0)
        else:
            t_stat = 0.0
            p_value = 1.0

        win_rate = (returns > 0).sum() / count

        results.append(
            RegimeStats(
                regime="all",
                pattern=str(pattern),
                count=count,
                win_rate=win_rate,
                mean_return=mean_ret,
                std_return=std_ret,
                t_stat=t_stat,
                p_value=p_value,
            )
        )

    return results


def check_odd_robustness(
    bars_df: pd.DataFrame,
    pattern_col: str,
    regime_col: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict[str, dict]:
    """Check ODD robustness for patterns within regimes.

    A pattern is ODD robust if:
    1. |t-stat| >= min_t_stat in ALL rolling periods
    2. Same sign (positive or negative) across all periods
    3. At least min_samples per period

    Args:
        df: DataFrame with patterns, regimes, and returns
        pattern_col: Column name for pattern labels
        regime_col: Column name for regime labels
        return_col: Column name for forward returns
        min_samples: Minimum samples per period
        min_t_stat: Minimum t-statistic threshold

    Returns:
        Dict mapping (regime, pattern) to robustness results
    """
    results = {}

    for regime in bars_df[regime_col].unique():
        if pd.isna(regime):
            continue

        regime_subset = bars_df[bars_df[regime_col] == regime]

        for pattern in regime_subset[pattern_col].unique():
            if pd.isna(pattern):
                continue

            pattern_subset = regime_subset[regime_subset[pattern_col] == pattern]

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
                std_ret = returns.std()

                if std_ret > 0:
                    t_stat, _ = ttest_1samp(returns, 0)
                else:
                    t_stat = 0.0

                period_stats.append(
                    {
                        "period": period_label,
                        "count": count,
                        "mean_return": mean_ret,
                        "t_stat": t_stat,
                    }
                )

            if len(period_stats) < 2:
                continue

            # Check ODD criteria
            t_stats = [s["t_stat"] for s in period_stats]
            all_significant = all(abs(t) >= min_t_stat for t in t_stats)
            same_sign = all(t > 0 for t in t_stats) or all(
                t < 0 for t in t_stats
            )

            is_odd_robust = all_significant and same_sign

            key = f"{regime}|{pattern}"
            results[key] = {
                "regime": str(regime),
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


def load_range_bars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load range bars from ClickHouse cache.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        threshold_dbps: Threshold in decimal basis points
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    from rangebar import get_range_bars

    log_info(
        "Loading range bars",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        start_date=start_date,
        end_date=end_date,
    )

    df = get_range_bars(
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
        bars=len(df),
    )

    return df


# =============================================================================
# Main Analysis
# =============================================================================


def prepare_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for regime analysis.

    Adds:
    - SMA(20), SMA(50)
    - RSI(14)
    - Regime classifications
    - 2-bar and 3-bar patterns
    - Forward returns
    """
    result = df.copy()

    # Technical indicators
    result["sma20"] = compute_sma(result["Close"], 20)
    result["sma50"] = compute_sma(result["Close"], 50)
    result["rsi14"] = compute_rsi(result["Close"], 14)

    # Regime classifications
    result["sma_regime"] = result.apply(
        lambda row: classify_sma_regime(
            row["Close"], row["sma20"], row["sma50"]
        ).value,
        axis=1,
    )
    result["rsi_regime"] = result["rsi14"].apply(
        lambda rsi: classify_rsi_regime(rsi).value
    )

    # Combined regime
    result["regime"] = result.apply(
        lambda row: classify_combined_regime(
            SMARegime(row["sma_regime"]),
            RSIRegime(row["rsi_regime"]),
        ).value,
        axis=1,
    )

    # Patterns
    result["pattern_2bar"] = detect_2bar_patterns(result)
    result["pattern_3bar"] = detect_3bar_patterns(result)

    # Forward returns
    result = compute_forward_returns(result, horizons=[1, 3, 5])

    return result


def run_regime_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> dict:
    """Run full regime analysis for a symbol/threshold combination.

    Args:
        symbol: Trading symbol
        threshold_dbps: Threshold in decimal basis points
        start_date: Start date
        end_date: End date
        min_samples: Minimum samples per period
        min_t_stat: Minimum t-statistic for ODD robustness

    Returns:
        Dict with analysis results
    """
    log_info(
        "Starting regime analysis",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        start_date=start_date,
        end_date=end_date,
    )

    # Load data
    df = load_range_bars(symbol, threshold_dbps, start_date, end_date)

    if len(df) < 1000:
        log_warn(
            "Insufficient data for analysis",
            symbol=symbol,
            threshold_dbps=threshold_dbps,
            bars=len(df),
        )
        return {"error": "insufficient_data", "bars": len(df)}

    # Prepare analysis DataFrame
    df = prepare_analysis_df(df)

    # Regime distribution
    regime_counts = df["regime"].value_counts().to_dict()
    log_info("Regime distribution", **regime_counts)

    # Test ODD robustness for 2-bar patterns
    log_info("Testing 2-bar pattern ODD robustness")
    results_2bar = check_odd_robustness(
        df,
        pattern_col="pattern_2bar",
        regime_col="regime",
        return_col="fwd_ret_1",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Test ODD robustness for 3-bar patterns
    log_info("Testing 3-bar pattern ODD robustness")
    results_3bar = check_odd_robustness(
        df,
        pattern_col="pattern_3bar",
        regime_col="regime",
        return_col="fwd_ret_1",
        min_samples=min_samples,
        min_t_stat=min_t_stat,
    )

    # Count ODD robust patterns
    robust_2bar = [
        k for k, v in results_2bar.items() if v["is_odd_robust"]
    ]
    robust_3bar = [
        k for k, v in results_3bar.items() if v["is_odd_robust"]
    ]

    log_info(
        "Analysis complete",
        symbol=symbol,
        threshold_dbps=threshold_dbps,
        total_bars=len(df),
        robust_2bar_patterns=len(robust_2bar),
        robust_3bar_patterns=len(robust_3bar),
    )

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "start_date": start_date,
        "end_date": end_date,
        "total_bars": len(df),
        "regime_counts": regime_counts,
        "results_2bar": results_2bar,
        "results_3bar": results_3bar,
        "robust_2bar": robust_2bar,
        "robust_3bar": robust_3bar,
    }


def main() -> int:
    """Main entry point."""
    log_info("Starting market regime analysis", script="regime_analysis.py")

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    thresholds = [50, 100]  # Primary signal granularities
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    min_samples = 100
    min_t_stat = 5.0

    # Run analysis for each combination
    all_results = {}
    all_robust_patterns = []

    for symbol in symbols:
        for threshold in thresholds:
            key = f"{symbol}@{threshold}"
            try:
                result = run_regime_analysis(
                    symbol=symbol,
                    threshold_dbps=threshold,
                    start_date=start_date,
                    end_date=end_date,
                    min_samples=min_samples,
                    min_t_stat=min_t_stat,
                )
                all_results[key] = result

                if "robust_2bar" in result:
                    for pattern in result["robust_2bar"]:
                        all_robust_patterns.append(
                            {
                                "symbol": symbol,
                                "threshold": threshold,
                                "pattern_type": "2bar",
                                "pattern": pattern,
                            }
                        )
                if "robust_3bar" in result:
                    for pattern in result["robust_3bar"]:
                        all_robust_patterns.append(
                            {
                                "symbol": symbol,
                                "threshold": threshold,
                                "pattern_type": "3bar",
                                "pattern": pattern,
                            }
                        )

            except (ValueError, RuntimeError, OSError, KeyError) as e:
                log_error(
                    "Analysis failed",
                    symbol=symbol,
                    threshold_dbps=threshold,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                all_results[key] = {"error": str(e), "error_type": type(e).__name__}

    # Cross-symbol validation
    log_info(
        "Cross-symbol validation",
        total_robust_patterns=len(all_robust_patterns),
    )

    # Find patterns that are robust across ALL symbols
    if all_robust_patterns:
        from collections import Counter

        pattern_counts = Counter(
            (p["pattern_type"], p["pattern"]) for p in all_robust_patterns
        )
        universal_patterns = [
            p for p, count in pattern_counts.items() if count >= len(symbols)
        ]
        log_info(
            "Universal ODD robust patterns (all symbols)",
            count=len(universal_patterns),
            patterns=universal_patterns,
        )
    else:
        log_info("No ODD robust patterns found")

    log_info("Market regime analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
