#!/usr/bin/env python3
"""Hurst-Adjusted Kelly Fraction Analysis.

Even though patterns fail strict PSR/MinTRL validation after Hurst adjustment,
they may still be tradeable with reduced position sizes.

This script calculates:
1. Standard Kelly fraction based on observed returns
2. Hurst-adjusted Kelly using effective sample size uncertainty
3. Recommended position sizing given long memory

Issue #54: Volatility Regime Filter for ODD Robust Patterns
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Add parent dir for rangebar import
sys.path.insert(0, str(Path(__file__).parent.parent))

from rangebar import get_range_bars


def log_info(message: str, **kwargs: object) -> None:
    """Log info message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))


def compute_hurst_rs(returns: np.ndarray, min_window: int = 20) -> float:
    """Compute Hurst exponent using R/S method."""
    n = len(returns)
    if n < min_window * 4:
        return 0.5

    max_k = int(np.log2(n // min_window))
    if max_k < 2:
        return 0.5

    window_sizes = [min_window * (2**k) for k in range(max_k + 1)]

    log_rs_values = []
    log_n_values = []

    for window in window_sizes:
        num_windows = n // window
        if num_windows < 1:
            continue

        rs_list = []
        for i in range(num_windows):
            start = i * window
            end = start + window
            segment = returns[start:end]

            mean_seg = np.mean(segment)
            cumsum = np.cumsum(segment - mean_seg)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(segment, ddof=1)

            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            avg_rs = np.mean(rs_list)
            log_rs_values.append(np.log(avg_rs))
            log_n_values.append(np.log(window))

    if len(log_rs_values) < 2:
        return 0.5

    slope, _ = np.polyfit(log_n_values, log_rs_values, 1)
    return float(np.clip(slope, 0.0, 1.0))


def compute_kelly_fraction(
    mean_return: float,
    std_return: float,
) -> float:
    """Compute Kelly fraction for position sizing.

    Kelly = mu / sigma^2

    Args:
        mean_return: Expected return per trade
        std_return: Standard deviation of returns

    Returns:
        Kelly fraction (can exceed 1.0 for high-edge strategies)
    """
    if std_return <= 0:
        return 0.0

    variance = std_return**2
    kelly = mean_return / variance

    return float(kelly)


def compute_hurst_adjusted_kelly(
    mean_return: float,
    std_return: float,
    n_samples: int,
    hurst: float,
    confidence: float = 0.95,
) -> float:
    """Compute Kelly fraction adjusted for Hurst-based uncertainty.

    With long memory (H > 0.5), effective sample size is reduced,
    increasing uncertainty in mean estimate. We discount Kelly by
    the ratio of confidence intervals.

    Args:
        mean_return: Expected return per trade
        std_return: Standard deviation of returns
        n_samples: Number of observations
        hurst: Hurst exponent
        confidence: Confidence level for adjustment

    Returns:
        Hurst-adjusted Kelly fraction
    """
    if std_return <= 0 or n_samples < 2:
        return 0.0

    # Effective sample size
    exponent = 2 * (1 - hurst)
    n_eff = n_samples**exponent

    if n_eff < 2:
        return 0.0

    # Standard error of mean with effective samples
    se_eff = std_return / math.sqrt(n_eff)

    # Z-score for confidence
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Lower bound of mean at confidence level
    mean_lower = mean_return - z * se_eff

    if mean_lower <= 0:
        # Lower bound includes zero or negative - no confident edge
        return 0.0

    # Adjusted Kelly using lower bound estimate
    kelly_adj = compute_kelly_fraction(mean_lower, std_return)

    return float(kelly_adj)


def analyze_pattern_kelly(
    df: pl.DataFrame,
    transaction_cost_bps: float = 1.5,
) -> list[dict]:
    """Analyze Kelly fractions for each pattern."""
    # Add direction and 3-bar pattern
    df = df.with_columns(
        pl.when(pl.col("Close") > pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )

    df = df.with_columns(
        (
            pl.col("direction").shift(2)
            + pl.col("direction").shift(1)
            + pl.col("direction")
        ).alias("pattern_3bar")
    )

    # Forward returns
    df = df.with_columns(
        ((pl.col("Close").shift(-1) / pl.col("Close") - 1) * 10000).alias(
            "forward_return_bps"
        )
    )

    patterns = df.filter(pl.col("pattern_3bar").is_not_null())["pattern_3bar"].unique().to_list()

    results = []

    for pattern in patterns:
        pattern_df = df.filter(
            (pl.col("pattern_3bar") == pattern)
            & pl.col("forward_return_bps").is_not_null()
        )

        if len(pattern_df) < 100:
            continue

        returns = pattern_df["forward_return_bps"].to_numpy()
        n = len(returns)

        # Net returns
        gross_mean = float(np.mean(returns))
        net_mean = gross_mean - transaction_cost_bps
        std_ret = float(np.std(returns, ddof=1))

        # Hurst
        hurst = compute_hurst_rs(returns / 10000)

        # Kelly fractions
        kelly_std = compute_kelly_fraction(net_mean, std_ret)
        kelly_adj = compute_hurst_adjusted_kelly(net_mean, std_ret, n, hurst)

        # Win rate
        win_rate = float(np.mean(returns > transaction_cost_bps))

        # Fractional Kelly recommendations
        kelly_half = kelly_std * 0.5
        kelly_quarter = kelly_std * 0.25

        results.append({
            "pattern": pattern,
            "n": n,
            "net_bps": round(net_mean, 2),
            "std_bps": round(std_ret, 2),
            "win_rate": round(win_rate, 4),
            "hurst": round(hurst, 4),
            "kelly_std": round(kelly_std, 4),
            "kelly_adj": round(kelly_adj, 4),
            "kelly_half": round(kelly_half, 4),
            "kelly_quarter": round(kelly_quarter, 4),
            "recommendation": "TRADE" if kelly_adj > 0 else "NO TRADE",
        })

    return results


def main() -> None:
    """Run Hurst-adjusted Kelly analysis."""
    log_info("Starting Hurst-adjusted Kelly analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100
    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    all_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        df = get_range_bars(
            symbol,
            start_date=start_dates[symbol],
            end_date=end_date,
            threshold_decimal_bps=threshold,
            ouroboros="month",
            use_cache=True,
            fetch_if_missing=False,
        )

        if df is None or len(df) == 0:
            continue

        df_pl = pl.from_pandas(df.reset_index())
        results = analyze_pattern_kelly(df_pl)

        for r in results:
            r["symbol"] = symbol
            all_results.append(r)

    if not all_results:
        log_info("No results")
        return

    # Aggregate across symbols
    results_df = pl.DataFrame(all_results)

    agg = (
        results_df.group_by("pattern")
        .agg([
            pl.col("n").sum().alias("total_n"),
            (pl.col("net_bps") * pl.col("n")).sum().alias("weighted_net"),
            (pl.col("std_bps") * pl.col("n")).sum().alias("weighted_std"),
            (pl.col("win_rate") * pl.col("n")).sum().alias("weighted_wr"),
            (pl.col("hurst") * pl.col("n")).sum().alias("weighted_h"),
            (pl.col("kelly_std") * pl.col("n")).sum().alias("weighted_k_std"),
            (pl.col("kelly_adj") * pl.col("n")).sum().alias("weighted_k_adj"),
        ])
        .with_columns([
            (pl.col("weighted_net") / pl.col("total_n")).alias("net_bps"),
            (pl.col("weighted_std") / pl.col("total_n")).alias("std_bps"),
            (pl.col("weighted_wr") / pl.col("total_n")).alias("win_rate"),
            (pl.col("weighted_h") / pl.col("total_n")).alias("hurst"),
            (pl.col("weighted_k_std") / pl.col("total_n")).alias("kelly_std"),
            (pl.col("weighted_k_adj") / pl.col("total_n")).alias("kelly_adj"),
        ])
        .with_columns([
            (pl.col("kelly_std") * 0.5).alias("kelly_half"),
            (pl.col("kelly_std") * 0.25).alias("kelly_quarter"),
        ])
        .select([
            "pattern", "total_n", "net_bps", "std_bps", "win_rate",
            "hurst", "kelly_std", "kelly_adj", "kelly_half", "kelly_quarter"
        ])
        .sort("net_bps", descending=True)
    )

    # Print results
    print("\n" + "=" * 140)
    print("HURST-ADJUSTED KELLY FRACTION ANALYSIS")
    print("=" * 140)
    print("\nKelly = mu / sigma^2")
    print("Hurst-Adjusted = Kelly based on lower confidence bound of mean")
    print("\n" + "-" * 140)
    print(f"{'Pattern':<12} {'N':>10} {'Net(bps)':>10} {'Std':>8} {'WinRate':>8} {'Hurst':>7} {'Kelly':>8} {'Adj':>8} {'Half':>8} {'Qtr':>8}")
    print("-" * 140)

    tradeable = 0
    for row in agg.to_dicts():
        kelly_adj = row["kelly_adj"]
        if kelly_adj > 0:
            tradeable += 1
        print(
            f"{row['pattern']:<12} "
            f"{row['total_n']:>10,} "
            f"{row['net_bps']:>10.2f} "
            f"{row['std_bps']:>8.2f} "
            f"{row['win_rate']:>8.2%} "
            f"{row['hurst']:>7.4f} "
            f"{row['kelly_std']:>8.4f} "
            f"{row['kelly_adj']:>8.4f} "
            f"{row['kelly_half']:>8.4f} "
            f"{row['kelly_quarter']:>8.4f}"
        )

    print("-" * 140)

    # Summary
    print("\n" + "=" * 140)
    print("TRADING RECOMMENDATIONS")
    print("=" * 140)

    print(f"\nPatterns with positive Hurst-adjusted Kelly: {tradeable}/8")

    if tradeable == 0:
        print("\n⚠️  NO PATTERNS have confident positive edge after Hurst adjustment")
        print("\nHowever, half-Kelly or quarter-Kelly may still be appropriate:")
        print("- Use fractional Kelly to account for model uncertainty")
        print("- DU-ending patterns have highest raw edge (+10-13 bps gross)")
        print("- Monitor for regime changes (ADWIN detected none in 4 years)")
    else:
        print(f"\n✓ {tradeable} patterns have confident positive edge")
        print("- Use Hurst-adjusted Kelly for position sizing")
        print("- Consider half-Kelly for additional safety margin")

    # Best patterns for trading
    print("\n" + "=" * 140)
    print("BEST PATTERNS BY RAW EDGE (for fractional Kelly)")
    print("=" * 140)

    best = agg.filter(pl.col("net_bps") > 0).sort("net_bps", descending=True)
    if len(best) > 0:
        for row in best.to_dicts():
            print(
                f"\n{row['pattern']}:"
                f"\n  Net return: {row['net_bps']:.2f} bps"
                f"\n  Win rate: {row['win_rate']:.2%}"
                f"\n  Hurst: {row['hurst']:.4f}"
                f"\n  Quarter-Kelly: {row['kelly_quarter']:.4f}"
                f"\n  Recommended sizing: {row['kelly_quarter'] * 100:.2f}% per trade"
            )
    else:
        print("\nNo patterns with positive net edge after transaction costs")

    log_info("Analysis complete", tradeable=tradeable)


if __name__ == "__main__":
    main()
