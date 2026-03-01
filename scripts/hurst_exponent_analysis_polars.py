#!/usr/bin/env python3
"""Hurst Exponent Analysis for Range Bar Returns.

Implements Generalized Hurst Exponent (GHE) analysis per MRH Framework.

Key concepts:
- H > 0.5: Long memory / trending behavior (positive autocorrelation)
- H = 0.5: Random walk / no memory (IID)
- H < 0.5: Mean-reverting behavior (negative autocorrelation)

Effective sample size adjustment: T_eff = T^(2(1-H))
- If H = 0.5: T_eff = T (no adjustment needed)
- If H = 0.7: T_eff = T^0.6 (fewer effective samples)
- If H = 0.3: T_eff = T^1.4 (more effective samples)

Reference: docs/research/external/time-to-convergence-stationarity-gap.md

Issue #54: Volatility Regime Filter for ODD Robust Patterns
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

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


def compute_rs_hurst(returns: np.ndarray, min_window: int = 20) -> float:
    """Compute Hurst exponent using Rescaled Range (R/S) method.

    Args:
        returns: Array of log returns
        min_window: Minimum window size for R/S calculation

    Returns:
        Estimated Hurst exponent
    """
    n = len(returns)
    if n < min_window * 4:
        return 0.5  # Default to random walk if insufficient data

    # Generate window sizes (powers of 2)
    max_k = int(np.log2(n // min_window))
    if max_k < 2:
        return 0.5

    window_sizes = [min_window * (2**k) for k in range(max_k + 1)]

    log_rs_values = []
    log_n_values = []

    for window in window_sizes:
        # Number of non-overlapping windows
        num_windows = n // window
        if num_windows < 1:
            continue

        rs_list = []

        for i in range(num_windows):
            start = i * window
            end = start + window
            segment = returns[start:end]

            # Mean-adjusted cumulative sum
            mean_seg = np.mean(segment)
            cumsum = np.cumsum(segment - mean_seg)

            # Range
            r = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            s = np.std(segment, ddof=1)

            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            avg_rs = np.mean(rs_list)
            log_rs_values.append(np.log(avg_rs))
            log_n_values.append(np.log(window))

    if len(log_rs_values) < 2:
        return 0.5

    # Linear regression to find Hurst exponent
    # log(R/S) = H * log(n) + c
    slope, _ = np.polyfit(log_n_values, log_rs_values, 1)

    return float(np.clip(slope, 0.0, 1.0))


def compute_dfa_hurst(returns: np.ndarray, min_window: int = 10) -> float:
    """Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).

    More robust than R/S for non-stationary data.

    Args:
        returns: Array of log returns
        min_window: Minimum window size

    Returns:
        Estimated Hurst exponent (alpha parameter)
    """
    n = len(returns)
    if n < min_window * 10:
        return 0.5

    # Cumulative sum (integration)
    y = np.cumsum(returns - np.mean(returns))

    # Generate window sizes
    max_k = int(np.log2(n // min_window))
    if max_k < 2:
        return 0.5

    window_sizes = [min_window * (2**k) for k in range(max_k + 1)]

    log_f_values = []
    log_n_values = []

    for window in window_sizes:
        num_windows = n // window
        if num_windows < 1:
            continue

        f_list = []

        for i in range(num_windows):
            start = i * window
            end = start + window
            segment = y[start:end]

            # Fit linear trend
            x = np.arange(window)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # Fluctuation (RMS of detrended segment)
            f = np.sqrt(np.mean((segment - trend) ** 2))
            f_list.append(f)

        if f_list:
            avg_f = np.mean(f_list)
            if avg_f > 0:
                log_f_values.append(np.log(avg_f))
                log_n_values.append(np.log(window))

    if len(log_f_values) < 2:
        return 0.5

    # Linear regression: log(F) = alpha * log(n) + c
    # alpha = H (Hurst exponent)
    slope, _ = np.polyfit(log_n_values, log_f_values, 1)

    return float(np.clip(slope, 0.0, 1.0))


def compute_effective_sample_size(n: int, hurst: float) -> float:
    """Compute effective sample size given Hurst exponent.

    T_eff = T^(2(1-H))

    Args:
        n: Actual sample size
        hurst: Hurst exponent

    Returns:
        Effective sample size
    """
    if n <= 0 or not (0 < hurst < 1):
        return float(n)

    exponent = 2 * (1 - hurst)
    return float(n**exponent)


def analyze_hurst_by_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Analyze Hurst exponent for returns following each pattern.

    Args:
        df: Range bar DataFrame with patterns

    Returns:
        Aggregated Hurst analysis by pattern
    """
    # Add direction and 3-bar pattern
    df = df.with_columns(
        (
            pl.when(pl.col("Close") > pl.col("Open"))
            .then(pl.lit("U"))
            .otherwise(pl.lit("D"))
        ).alias("direction")
    )

    df = df.with_columns(
        (
            pl.col("direction").shift(2)
            + pl.col("direction").shift(1)
            + pl.col("direction")
        ).alias("pattern_3bar")
    )

    # Compute log returns
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
    )

    # Get unique patterns
    patterns = df.filter(pl.col("pattern_3bar").is_not_null())["pattern_3bar"].unique().to_list()

    results = []

    for pattern in patterns:
        # Get returns following this pattern
        pattern_mask = df["pattern_3bar"] == pattern
        pattern_indices = df.with_row_index().filter(pattern_mask)["index"].to_list()

        # Get forward returns (next bar after pattern)
        forward_returns = []
        log_returns = df["log_return"].to_list()

        for idx in pattern_indices:
            if idx + 1 < len(log_returns):
                ret = log_returns[idx + 1]
                import math
                if ret is not None and not math.isnan(ret):
                    forward_returns.append(ret)

        if len(forward_returns) < 100:
            continue

        returns_arr = np.array(forward_returns)

        # Compute Hurst using both methods
        h_rs = compute_rs_hurst(returns_arr)
        h_dfa = compute_dfa_hurst(returns_arr)

        # Average of both methods
        h_avg = (h_rs + h_dfa) / 2

        # Effective sample size
        n = len(returns_arr)
        t_eff = compute_effective_sample_size(n, h_avg)

        # Interpretation
        if h_avg > 0.55:
            behavior = "trending"
        elif h_avg < 0.45:
            behavior = "mean-reverting"
        else:
            behavior = "random"

        results.append({
            "pattern": pattern,
            "n_samples": n,
            "hurst_rs": round(h_rs, 4),
            "hurst_dfa": round(h_dfa, 4),
            "hurst_avg": round(h_avg, 4),
            "t_effective": round(t_eff, 0),
            "efficiency_ratio": round(t_eff / n, 4),
            "behavior": behavior,
        })

    return pl.DataFrame(results)


def main() -> None:
    """Run Hurst exponent analysis."""
    log_info("Starting Hurst exponent analysis")

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100  # 100 dbps
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

        # Get range bars
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
            log_info("No data for symbol", symbol=symbol)
            continue

        # Convert to Polars
        df_pl = pl.from_pandas(df.reset_index())

        # Compute overall Hurst for the symbol
        log_returns = (
            df_pl.filter(pl.col("Close").is_not_null())
            .select((pl.col("Close") / pl.col("Close").shift(1)).log())
            .drop_nulls()
            .to_numpy()
            .flatten()
        )

        h_rs = compute_rs_hurst(log_returns)
        h_dfa = compute_dfa_hurst(log_returns)
        h_avg = (h_rs + h_dfa) / 2
        t_eff = compute_effective_sample_size(len(log_returns), h_avg)

        log_info(
            "Overall Hurst",
            symbol=symbol,
            n=len(log_returns),
            hurst_rs=round(h_rs, 4),
            hurst_dfa=round(h_dfa, 4),
            hurst_avg=round(h_avg, 4),
            t_effective=round(t_eff, 0),
            efficiency=round(t_eff / len(log_returns), 4),
        )

        # Analyze by pattern
        pattern_results = analyze_hurst_by_pattern(df_pl)

        for row in pattern_results.to_dicts():
            row["symbol"] = symbol
            all_results.append(row)

    # Aggregate across symbols
    if not all_results:
        log_info("No results to aggregate")
        return

    results_df = pl.DataFrame(all_results)

    # Aggregate by pattern
    final_agg = (
        results_df.group_by("pattern")
        .agg(
            [
                pl.col("n_samples").sum().alias("total_n"),
                (pl.col("hurst_avg") * pl.col("n_samples")).sum().alias("weighted_hurst_sum"),
                pl.col("n_samples").sum().alias("weight_sum"),
                pl.col("t_effective").sum().alias("total_t_eff"),
            ]
        )
        .with_columns(
            (pl.col("weighted_hurst_sum") / pl.col("weight_sum")).alias("hurst_avg")
        )
        .with_columns(
            (pl.col("total_t_eff") / pl.col("total_n")).alias("efficiency_ratio")
        )
        .select(["pattern", "total_n", "hurst_avg", "total_t_eff", "efficiency_ratio"])
        .sort("hurst_avg", descending=True)
    )

    # Print results
    print("\n" + "=" * 100)
    print("HURST EXPONENT ANALYSIS: LONG MEMORY EFFECTS BY PATTERN")
    print("=" * 100)
    print("\nFormula: T_eff = T^(2(1-H))")
    print("H > 0.5: Trending (fewer effective samples)")
    print("H = 0.5: Random walk (no adjustment)")
    print("H < 0.5: Mean-reverting (more effective samples)")
    print("\n" + "-" * 100)
    print(f"{'Pattern':<15} {'N':>12} {'Hurst':>10} {'T_eff':>15} {'Efficiency':>12}")
    print("-" * 100)

    for row in final_agg.to_dicts():
        print(
            f"{row['pattern']:<15} "
            f"{row['total_n']:>12,} "
            f"{row['hurst_avg']:>10.4f} "
            f"{row['total_t_eff']:>15,.0f} "
            f"{row['efficiency_ratio']:>12.4f}"
        )

    print("-" * 100)

    # Summary statistics
    avg_hurst = final_agg["hurst_avg"].mean()
    avg_efficiency = final_agg["efficiency_ratio"].mean()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nAverage Hurst across patterns: {avg_hurst:.4f}")
    print(f"Average efficiency ratio: {avg_efficiency:.4f}")

    if avg_hurst > 0.55:
        print("\nConclusion: Returns show TRENDING behavior (positive autocorrelation)")
        print("MinTRL calculations should use reduced effective sample size")
    elif avg_hurst < 0.45:
        print("\nConclusion: Returns show MEAN-REVERTING behavior (negative autocorrelation)")
        print("MinTRL calculations benefit from increased effective sample size")
    else:
        print("\nConclusion: Returns approximate RANDOM WALK (H ~ 0.5)")
        print("No adjustment needed for MinTRL calculations")

    # Impact on PSR/MinTRL validation
    print("\n" + "=" * 100)
    print("IMPACT ON MRH FRAMEWORK VALIDATION")
    print("=" * 100)

    # If H > 0.5, our T_available is effectively smaller
    if avg_hurst > 0.5:
        reduction = 2 * (1 - avg_hurst)
        print(f"\nWith H = {avg_hurst:.4f}, effective sample exponent = {reduction:.4f}")
        print("This means our 36 production-ready patterns may have smaller effective T_available")
        print("However, if gaps are still negative after adjustment, patterns remain valid")
    else:
        print(f"\nWith H = {avg_hurst:.4f} (near or below 0.5), no sample size reduction needed")
        print("The 36 production-ready patterns remain valid")

    log_info("Analysis complete", avg_hurst=float(avg_hurst), avg_efficiency=float(avg_efficiency))


if __name__ == "__main__":
    main()
