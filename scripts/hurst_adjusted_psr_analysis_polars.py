#!/usr/bin/env python3
"""Hurst-Adjusted PSR/MinTRL Analysis.

Re-validates pattern significance using Hurst-corrected effective sample sizes.

Key insight from Hurst analysis:
- Pattern-conditioned returns show H ~ 0.79 (long memory)
- T_eff = T^(2(1-H)) = T^0.41
- This drastically reduces effective sample size

This script checks if patterns remain valid after adjustment.

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


def compute_effective_sample_size(n: int, hurst: float) -> float:
    """Compute effective sample size given Hurst exponent."""
    if n <= 0 or not (0 < hurst < 1):
        return float(n)
    exponent = 2 * (1 - hurst)
    return float(n**exponent)


def compute_psr(
    observed_sr: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio."""
    if n_samples < 2 or observed_sr <= benchmark_sr:
        return 0.0

    sr_variance = (
        1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    ) / (n_samples - 1)

    if sr_variance <= 0:
        return 1.0 if observed_sr > benchmark_sr else 0.0

    sr_std = math.sqrt(sr_variance)
    z_score = (observed_sr - benchmark_sr) / sr_std
    return float(stats.norm.cdf(z_score))


def compute_mintrl(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """Compute Minimum Track Record Length."""
    if observed_sr <= benchmark_sr:
        return float("inf")

    z_alpha = stats.norm.ppf(confidence)
    penalty = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    mintrl = 1 + penalty * (z_alpha / (observed_sr - benchmark_sr)) ** 2

    return float(mintrl)


def analyze_with_hurst_adjustment(
    df: pl.DataFrame,
    transaction_cost_bps: float = 1.5,
) -> list[dict]:
    """Analyze patterns with Hurst-adjusted sample sizes."""
    # Add direction and pattern
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

    # Compute returns
    df = df.with_columns(
        ((pl.col("Close").shift(-1) / pl.col("Close") - 1) * 10000).alias("forward_return_bps")
    )

    # Get unique patterns
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

        # Compute Hurst for this pattern's forward returns
        hurst = compute_hurst_rs(returns / 10000)  # Use decimal returns for Hurst

        # Effective sample size
        n_eff = compute_effective_sample_size(n, hurst)

        # Gross and net returns
        gross_mean = float(np.mean(returns))
        net_mean = gross_mean - transaction_cost_bps

        # Higher moments
        std_ret = float(np.std(returns, ddof=1))
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))

        # Net Sharpe ratio (annualized not needed for comparison)
        net_sr = net_mean / std_ret if std_ret > 0 else 0

        # PSR with ORIGINAL sample size
        psr_original = compute_psr(net_sr, n, skewness, kurtosis)

        # PSR with HURST-ADJUSTED sample size
        psr_adjusted = compute_psr(net_sr, int(n_eff), skewness, kurtosis)

        # MinTRL
        mintrl = compute_mintrl(net_sr, 0.0, skewness, kurtosis)

        # Stationarity gaps
        gap_original = mintrl - n
        gap_adjusted = mintrl - n_eff

        results.append({
            "pattern": pattern,
            "n_original": n,
            "n_effective": int(n_eff),
            "hurst": round(hurst, 4),
            "net_bps": round(net_mean, 2),
            "sharpe": round(net_sr, 4),
            "psr_original": round(psr_original, 4),
            "psr_adjusted": round(psr_adjusted, 4),
            "mintrl": round(mintrl, 0),
            "gap_original": round(gap_original, 0),
            "gap_adjusted": round(gap_adjusted, 0),
            "valid_original": gap_original < 0,
            "valid_adjusted": gap_adjusted < 0,
        })

    return results


def main() -> None:
    """Run Hurst-adjusted PSR analysis."""
    log_info("Starting Hurst-adjusted PSR/MinTRL analysis")

    # Configuration
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
        results = analyze_with_hurst_adjustment(df_pl)

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
            pl.col("n_original").sum().alias("total_n"),
            pl.col("n_effective").sum().alias("total_n_eff"),
            (pl.col("hurst") * pl.col("n_original")).sum().alias("weighted_hurst"),
            (pl.col("net_bps") * pl.col("n_original")).sum().alias("weighted_net"),
            (pl.col("psr_adjusted") * pl.col("n_original")).sum().alias("weighted_psr"),
            pl.col("valid_adjusted").all().alias("all_valid"),
        ])
        .with_columns([
            (pl.col("weighted_hurst") / pl.col("total_n")).alias("hurst"),
            (pl.col("weighted_net") / pl.col("total_n")).alias("net_bps"),
            (pl.col("weighted_psr") / pl.col("total_n")).alias("psr_adjusted"),
        ])
        .select(["pattern", "total_n", "total_n_eff", "hurst", "net_bps", "psr_adjusted", "all_valid"])
        .sort("net_bps", descending=True)
    )

    # Print results
    print("\n" + "=" * 120)
    print("HURST-ADJUSTED PSR/MinTRL ANALYSIS")
    print("=" * 120)
    print("\nKey: Patterns must remain valid after Hurst adjustment (Gap < 0)")
    print("\n" + "-" * 120)
    print(f"{'Pattern':<12} {'N_orig':>12} {'N_eff':>12} {'Hurst':>8} {'Net(bps)':>10} {'PSR_adj':>10} {'Valid':>8}")
    print("-" * 120)

    valid_count = 0
    for row in agg.to_dicts():
        valid_str = "YES" if row["all_valid"] else "NO"
        if row["all_valid"]:
            valid_count += 1
        print(
            f"{row['pattern']:<12} "
            f"{row['total_n']:>12,} "
            f"{row['total_n_eff']:>12,} "
            f"{row['hurst']:>8.4f} "
            f"{row['net_bps']:>10.2f} "
            f"{row['psr_adjusted']:>10.4f} "
            f"{valid_str:>8}"
        )

    print("-" * 120)

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    total_patterns = len(agg)
    print(f"\nTotal patterns analyzed: {total_patterns}")
    print(f"Patterns valid with Hurst adjustment: {valid_count}")
    print(f"Patterns invalidated by Hurst adjustment: {total_patterns - valid_count}")

    if valid_count == 0:
        print("\n⚠️  WARNING: No patterns survive Hurst adjustment!")
        print("This suggests pattern returns have strong long memory (H >> 0.5)")
        print("which reduces effective sample size below MinTRL requirements.")
    elif valid_count < total_patterns:
        print(f"\n⚠️  {total_patterns - valid_count} patterns invalidated by Hurst correction")
        print("These patterns have insufficient effective samples for significance.")
    else:
        print("\n✓ All patterns remain valid after Hurst adjustment")

    log_info("Analysis complete", valid=valid_count, total=total_patterns)


if __name__ == "__main__":
    main()
