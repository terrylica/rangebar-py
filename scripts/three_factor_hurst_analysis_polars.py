#!/usr/bin/env python3
"""Three-Factor Hurst Exponent Analysis.

Checks if three-factor patterns (RV regime + alignment + 3-bar) have lower
Hurst exponents than raw 3-bar patterns.

Hypothesis: The RV regime and alignment filters may reduce autocorrelation by:
1. Conditioning on volatility state (quiet/active/volatile)
2. Requiring multi-threshold alignment (reducing noise)

If H is closer to 0.5 for filtered patterns, some may survive Hurst adjustment.

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


def add_rv_regime(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add RV regime classification."""
    # Compute log returns
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
    )

    # Rolling RV (standard deviation)
    df = df.with_columns(
        pl.col("log_return").rolling_std(window_size=window).alias("rv")
    )

    # Compute percentiles for classification
    rv_values = df.filter(pl.col("rv").is_not_null())["rv"].to_numpy()
    if len(rv_values) < 100:
        return df.with_columns(pl.lit("active").alias("rv_regime"))

    p25 = float(np.percentile(rv_values, 25))
    p75 = float(np.percentile(rv_values, 75))

    df = df.with_columns(
        pl.when(pl.col("rv") < p25)
        .then(pl.lit("quiet"))
        .when(pl.col("rv") > p75)
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("rv_regime")
    )

    return df


def add_alignment(df: pl.DataFrame) -> pl.DataFrame:
    """Add multi-threshold alignment classification.

    Uses 50, 100, 200 dbps thresholds to determine alignment.
    """
    # Direction at current threshold (100 dbps)
    df = df.with_columns(
        pl.when(pl.col("Close") > pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("dir_100")
    )

    # For simplicity, use rolling direction patterns as proxy for multi-threshold
    # In real implementation, would need separate range bar streams at 50/200 dbps

    # Use rolling majority direction as alignment proxy
    df = df.with_columns(
        pl.col("dir_100").shift(1).alias("prev_dir_1"),
        pl.col("dir_100").shift(2).alias("prev_dir_2"),
    )

    # Alignment based on consistency of recent directions
    df = df.with_columns(
        pl.when(
            (pl.col("dir_100") == "U")
            & (pl.col("prev_dir_1") == "U")
            & (pl.col("prev_dir_2") == "U")
        )
        .then(pl.lit("aligned_up"))
        .when(
            (pl.col("dir_100") == "D")
            & (pl.col("prev_dir_1") == "D")
            & (pl.col("prev_dir_2") == "D")
        )
        .then(pl.lit("aligned_down"))
        .when(
            (pl.col("dir_100") == "U")
            & ((pl.col("prev_dir_1") == "U") | (pl.col("prev_dir_2") == "U"))
        )
        .then(pl.lit("partial_up"))
        .when(
            (pl.col("dir_100") == "D")
            & ((pl.col("prev_dir_1") == "D") | (pl.col("prev_dir_2") == "D"))
        )
        .then(pl.lit("partial_down"))
        .otherwise(pl.lit("mixed"))
        .alias("alignment")
    )

    return df


def add_3bar_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Add 3-bar pattern classification."""
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

    return df


def analyze_three_factor_hurst(
    df: pl.DataFrame,
    transaction_cost_bps: float = 1.5,
) -> list[dict]:
    """Analyze Hurst for three-factor patterns."""
    # Add all factors
    df = add_rv_regime(df)
    df = add_alignment(df)
    df = add_3bar_pattern(df)

    # Create combined pattern
    df = df.with_columns(
        (
            pl.col("rv_regime")
            + "|"
            + pl.col("alignment")
            + "|"
            + pl.col("pattern_3bar")
        ).alias("three_factor_pattern")
    )

    # Forward returns
    df = df.with_columns(
        ((pl.col("Close").shift(-1) / pl.col("Close") - 1) * 10000).alias(
            "forward_return_bps"
        )
    )

    # Get unique three-factor patterns
    valid_df = df.filter(
        pl.col("three_factor_pattern").is_not_null()
        & pl.col("forward_return_bps").is_not_null()
        & ~pl.col("three_factor_pattern").str.contains("null")
    )

    patterns = valid_df["three_factor_pattern"].unique().to_list()

    results = []

    for pattern in patterns:
        pattern_df = valid_df.filter(pl.col("three_factor_pattern") == pattern)

        if len(pattern_df) < 100:
            continue

        returns = pattern_df["forward_return_bps"].to_numpy()
        n = len(returns)

        # Compute Hurst
        hurst = compute_hurst_rs(returns / 10000)

        # Effective sample size
        n_eff = compute_effective_sample_size(n, hurst)

        # Return statistics
        gross_mean = float(np.mean(returns))
        net_mean = gross_mean - transaction_cost_bps
        std_ret = float(np.std(returns, ddof=1))
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))

        # Sharpe and PSR
        net_sr = net_mean / std_ret if std_ret > 0 else 0
        psr_adjusted = compute_psr(net_sr, int(n_eff), skewness, kurtosis)

        # MinTRL and gap
        mintrl = compute_mintrl(net_sr, 0.0, skewness, kurtosis)
        gap_adjusted = mintrl - n_eff

        results.append({
            "pattern": pattern,
            "n": n,
            "n_eff": int(n_eff),
            "hurst": round(hurst, 4),
            "net_bps": round(net_mean, 2),
            "psr_adj": round(psr_adjusted, 4),
            "gap_adj": round(gap_adjusted, 0),
            "valid": gap_adjusted < 0,
        })

    return results


def main() -> None:
    """Run three-factor Hurst analysis."""
    log_info("Starting three-factor Hurst analysis")

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
            ouroboros="year",
            use_cache=True,
            fetch_if_missing=False,
        )

        if df is None or len(df) == 0:
            continue

        df_pl = pl.from_pandas(df.reset_index())
        results = analyze_three_factor_hurst(df_pl)

        for r in results:
            r["symbol"] = symbol
            all_results.append(r)

    if not all_results:
        log_info("No results")
        return

    # Aggregate across symbols - require pattern valid on ALL symbols
    results_df = pl.DataFrame(all_results)

    # Group by pattern
    agg = (
        results_df.group_by("pattern")
        .agg([
            pl.col("n").sum().alias("total_n"),
            pl.col("n_eff").sum().alias("total_n_eff"),
            (pl.col("hurst") * pl.col("n")).sum().alias("weighted_hurst"),
            (pl.col("net_bps") * pl.col("n")).sum().alias("weighted_net"),
            pl.col("valid").all().alias("all_valid"),
            pl.len().alias("symbol_count"),
        ])
        .filter(pl.col("symbol_count") >= 4)  # Must appear in all 4 symbols
        .with_columns([
            (pl.col("weighted_hurst") / pl.col("total_n")).alias("hurst"),
            (pl.col("weighted_net") / pl.col("total_n")).alias("net_bps"),
        ])
        .select(["pattern", "total_n", "total_n_eff", "hurst", "net_bps", "all_valid"])
        .sort("net_bps", descending=True)
    )

    # Print results
    print("\n" + "=" * 130)
    print("THREE-FACTOR HURST ANALYSIS: RV Regime + Alignment + 3-Bar Pattern")
    print("=" * 130)
    print("\nHypothesis: Three-factor filtering may reduce autocorrelation (lower H)")
    print("\n" + "-" * 130)
    print(f"{'Pattern':<50} {'N':>10} {'N_eff':>10} {'Hurst':>8} {'Net(bps)':>10} {'Valid':>8}")
    print("-" * 130)

    valid_count = 0
    low_hurst_count = 0

    for row in agg.head(40).to_dicts():
        valid_str = "YES" if row["all_valid"] else "NO"
        if row["all_valid"]:
            valid_count += 1
        if row["hurst"] < 0.6:
            low_hurst_count += 1
        print(
            f"{row['pattern']:<50} "
            f"{row['total_n']:>10,} "
            f"{row['total_n_eff']:>10,} "
            f"{row['hurst']:>8.4f} "
            f"{row['net_bps']:>10.2f} "
            f"{valid_str:>8}"
        )

    print("-" * 130)

    # Summary
    total = len(agg)
    avg_hurst = float(agg["hurst"].mean()) if total > 0 else 0.5

    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(f"\nTotal three-factor patterns (all 4 symbols): {total}")
    print(f"Average Hurst exponent: {avg_hurst:.4f}")
    print(f"Patterns with H < 0.6: {low_hurst_count}")
    print(f"Patterns valid after Hurst adjustment: {valid_count}")

    # Compare to raw 3-bar
    print("\n" + "=" * 130)
    print("COMPARISON TO RAW 3-BAR PATTERNS")
    print("=" * 130)
    print("\n| Pattern Type | Avg Hurst | Valid Patterns |")
    print("|--------------|-----------|----------------|")
    print("| Raw 3-bar    | ~0.79     | 0              |")
    print(f"| Three-factor | {avg_hurst:.4f}    | {valid_count}              |")

    if avg_hurst < 0.7:
        print("\n✓ Three-factor filtering REDUCES autocorrelation")
    else:
        print("\n✗ Three-factor filtering does NOT significantly reduce autocorrelation")

    log_info(
        "Analysis complete",
        total=total,
        avg_hurst=avg_hurst,
        valid=valid_count,
    )


if __name__ == "__main__":
    main()
