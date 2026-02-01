#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Probabilistic Sharpe Ratio (PSR) and Minimum Track Record Length (MinTRL) analysis.

Implements the MRH Framework from the stationarity gap research:
- PSR: Probability that true SR > benchmark, accounting for skewness/kurtosis
- MinTRL: Minimum sample size needed for statistical significance
- Stationarity Gap: T_required - T_available (negative = valid)

Reference: docs/research/external/time-to-convergence-stationarity-gap.md

GitHub Issue: #54, #55
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import polars as pl
from scipy import stats

# Transaction cost (high VIP tier)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001 / 100

# Symbols and date range
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
START_DATE = "2022-01-01"
END_DATE = "2026-01-31"

SYMBOL_START_DATES = {
    "BTCUSDT": "2022-01-01",
    "ETHUSDT": "2022-01-01",
    "SOLUSDT": "2023-06-01",
    "BNBUSDT": "2022-01-01",
}

# Confidence level for significance
CONFIDENCE_LEVEL = 0.95
Z_ALPHA = stats.norm.ppf(CONFIDENCE_LEVEL)  # ~1.645 for 95%


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


# =============================================================================
# Data Loading
# =============================================================================


def load_range_bars_polars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    """Load range bars from cache as Polars LazyFrame."""
    from rangebar import get_range_bars

    pdf = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    pdf = pdf.reset_index()
    pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})

    return pl.from_pandas(pdf).lazy()


# =============================================================================
# Pattern Classification (reused from three_factor)
# =============================================================================


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_3bar_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 3-bar pattern column."""
    return df.with_columns(
        (
            pl.col("direction") +
            pl.col("direction").shift(-1) +
            pl.col("direction").shift(-2)
        ).alias("pattern_3bar")
    )


def compute_rv_and_regime(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute RV and classify regime."""
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("_log_ret")
    ).with_columns(
        pl.col("_log_ret").rolling_std(window_size=20, min_samples=20).alias("rv")
    ).drop("_log_ret")

    df = df.with_columns(
        pl.col("rv").rolling_quantile(0.25, window_size=100, min_samples=50).alias("rv_p25"),
        pl.col("rv").rolling_quantile(0.75, window_size=100, min_samples=50).alias("rv_p75"),
    )

    df = df.with_columns(
        pl.when(pl.col("rv") < pl.col("rv_p25"))
        .then(pl.lit("quiet"))
        .when(pl.col("rv") > pl.col("rv_p75"))
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("rv_regime")
    )

    return df.drop(["rv_p25", "rv_p75"])


def align_thresholds(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars across thresholds."""
    df_50 = df_50.select([
        pl.col("timestamp").alias("timestamp_50"),
        pl.col("direction").alias("direction_50"),
    ])

    df_200 = df_200.select([
        pl.col("timestamp").alias("timestamp_200"),
        pl.col("direction").alias("direction_200"),
    ])

    df_100 = df_100.sort("timestamp")
    df_50 = df_50.sort("timestamp_50")
    df_200 = df_200.sort("timestamp_200")

    aligned = df_100.join_asof(
        df_50,
        left_on="timestamp",
        right_on="timestamp_50",
        strategy="backward",
    )

    aligned = aligned.join_asof(
        df_200,
        left_on="timestamp",
        right_on="timestamp_200",
        strategy="backward",
    )

    return aligned


def classify_alignment(df: pl.DataFrame) -> pl.DataFrame:
    """Classify multi-threshold alignment."""
    df = df.with_columns(
        pl.when(
            (pl.col("direction") == "U") &
            (pl.col("direction_50") == "U") &
            (pl.col("direction_200") == "U")
        ).then(pl.lit("aligned_up"))
        .when(
            (pl.col("direction") == "D") &
            (pl.col("direction_50") == "D") &
            (pl.col("direction_200") == "D")
        ).then(pl.lit("aligned_down"))
        .when(
            ((pl.col("direction") == "U").cast(pl.Int32) +
             (pl.col("direction_50") == "U").cast(pl.Int32) +
             (pl.col("direction_200") == "U").cast(pl.Int32)) >= 2
        ).then(pl.lit("partial_up"))
        .when(
            ((pl.col("direction") == "D").cast(pl.Int32) +
             (pl.col("direction_50") == "D").cast(pl.Int32) +
             (pl.col("direction_200") == "D").cast(pl.Int32)) >= 2
        ).then(pl.lit("partial_down"))
        .otherwise(pl.lit("mixed"))
        .alias("alignment")
    )

    return df


def create_three_factor_pattern(df: pl.DataFrame) -> pl.DataFrame:
    """Create three-factor pattern column."""
    return df.with_columns(
        (pl.col("rv_regime") + "|" + pl.col("alignment") + "|" + pl.col("pattern_3bar"))
        .alias("three_factor_pattern")
    )


# =============================================================================
# PSR and MinTRL Calculations
# =============================================================================


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    """Compute annualized Sharpe Ratio (assuming daily returns)."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    # For range bars, we don't annualize since bars are not time-uniform
    return float(mean_ret / std_ret)


def compute_psr(
    observed_sr: float,
    benchmark_sr: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Compute Probabilistic Sharpe Ratio.

    PSR = Probability that true SR > benchmark SR given higher moments.

    From Bailey & Lopez de Prado (2012):
    Var(SR) = (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1)

    Args:
        observed_sr: Observed Sharpe Ratio
        benchmark_sr: Benchmark SR (typically 0)
        n_samples: Number of observations
        skewness: Sample skewness (Fisher's definition)
        kurtosis: Sample kurtosis (Fisher's definition, Normal=3)

    Returns:
        PSR probability [0, 1]
    """
    if n_samples < 2:
        return 0.0

    # Compute variance of SR estimator (non-Normal formula)
    # Var(SR) = (1 - γ₃*SR + (γ₄-1)/4 * SR²) / (n-1)
    sr_variance = (
        1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2
    ) / (n_samples - 1)

    if sr_variance <= 0:
        return 1.0 if observed_sr > benchmark_sr else 0.0

    sr_std = math.sqrt(sr_variance)

    # PSR = Phi((SR_obs - SR_bm) / std_SR)
    z_score = (observed_sr - benchmark_sr) / sr_std
    psr = float(stats.norm.cdf(z_score))

    return psr


def compute_mintrl(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """Compute Minimum Track Record Length.

    MinTRL = 1 + (1 - skew*SR + (kurt-1)/4 * SR^2) * (Z_alpha / (SR - SR_bm))^2

    Args:
        observed_sr: Observed Sharpe Ratio
        benchmark_sr: Benchmark SR (typically 0)
        skewness: Sample skewness
        kurtosis: Sample kurtosis
        confidence: Confidence level (default 0.95)

    Returns:
        MinTRL (minimum samples needed for significance)
    """
    if observed_sr <= benchmark_sr:
        return float("inf")  # Cannot be significant if SR <= benchmark

    z_alpha = stats.norm.ppf(confidence)

    # Higher moments penalty factor
    penalty = 1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr**2

    # MinTRL formula
    mintrl = 1 + penalty * (z_alpha / (observed_sr - benchmark_sr))**2

    return float(mintrl)


def compute_dsr(
    observed_sr: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
) -> float:
    """Compute Deflated Sharpe Ratio accounting for multiple testing.

    Adjusts benchmark SR using expected maximum of N independent trials.
    SR_bm = sqrt(2 * ln(N)) * std_SR

    Args:
        observed_sr: Observed Sharpe Ratio
        n_samples: Number of observations
        skewness: Sample skewness
        kurtosis: Sample kurtosis
        n_trials: Number of backtest trials

    Returns:
        DSR probability [0, 1]
    """
    if n_trials < 1 or n_samples < 2:
        return 0.0

    # Estimate std_SR for trials (assuming SR~0 for null hypothesis)
    sr_std_trials = math.sqrt(1 / (n_samples - 1))

    # Adjusted benchmark (expected max of N normals)
    adjusted_benchmark = math.sqrt(2 * math.log(n_trials)) * sr_std_trials

    # Compute PSR with adjusted benchmark
    return compute_psr(observed_sr, adjusted_benchmark, n_samples, skewness, kurtosis)


def analyze_pattern_returns(returns: np.ndarray, pattern_name: str, n_trials: int = 49) -> dict:
    """Analyze returns for a single pattern using MRH Framework."""
    n = len(returns)
    if n < 30:  # Minimum for reliable moment estimation
        return None

    # Basic statistics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sr = mean_ret / std_ret if std_ret > 0 else 0

    # Higher moments
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns) + 3)  # Convert to Fisher's (Normal=3)

    # Net return after costs
    net_mean = mean_ret - ROUND_TRIP_COST_PCT
    net_sr = net_mean / std_ret if std_ret > 0 else 0

    # PSR calculations
    psr = compute_psr(sr, 0, n, skewness, kurtosis)
    psr_net = compute_psr(net_sr, 0, n, skewness, kurtosis)

    # MinTRL
    mintrl = compute_mintrl(sr, 0, skewness, kurtosis, 0.95)
    mintrl_net = compute_mintrl(net_sr, 0, skewness, kurtosis, 0.95)

    # DSR (accounting for 49 three-factor patterns tested)
    dsr = compute_dsr(sr, n, skewness, kurtosis, n_trials)
    dsr_net = compute_dsr(net_sr, n, skewness, kurtosis, n_trials)

    # Stationarity Gap
    gap = mintrl - n  # Negative = valid
    gap_net = mintrl_net - n

    return {
        "pattern": pattern_name,
        "n_samples": n,
        "mean_bps": round(mean_ret * 10000, 2),
        "net_bps": round(net_mean * 10000, 2),
        "std_bps": round(std_ret * 10000, 2),
        "sharpe_ratio": round(sr, 4),
        "net_sharpe_ratio": round(net_sr, 4),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 2),
        "psr": round(psr, 4),
        "psr_net": round(psr_net, 4),
        "mintrl": round(mintrl, 0),
        "mintrl_net": round(mintrl_net, 0) if mintrl_net != float("inf") else "inf",
        "gap": round(gap, 0),
        "gap_net": round(gap_net, 0) if gap_net != float("inf") else "inf",
        "dsr": round(dsr, 4),
        "dsr_net": round(dsr_net, 4),
        "is_valid": gap < 0,  # Negative gap = statistically valid
        "is_valid_net": gap_net < 0 if gap_net != float("inf") else False,
    }


# =============================================================================
# Main Analysis
# =============================================================================


def run_psr_analysis() -> list[dict]:
    """Run PSR/MinTRL analysis on all three-factor patterns."""
    log_info("Starting PSR/MinTRL analysis")

    # Collect all returns by pattern across symbols
    pattern_returns: dict[str, list[float]] = {}

    for symbol in SYMBOLS:
        start = SYMBOL_START_DATES.get(symbol, START_DATE)
        log_info("Processing symbol", symbol=symbol)

        # Load all thresholds
        df_50 = load_range_bars_polars(symbol, 50, start, END_DATE)
        df_100 = load_range_bars_polars(symbol, 100, start, END_DATE)
        df_200 = load_range_bars_polars(symbol, 200, start, END_DATE)

        # Process
        df_50 = add_bar_direction(df_50).collect()
        df_200 = add_bar_direction(df_200).collect()

        df_100 = add_bar_direction(df_100)
        df_100 = add_3bar_patterns(df_100)
        df_100 = compute_rv_and_regime(df_100)
        df_100 = df_100.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_return")
        ).collect()

        # Align and classify
        aligned = align_thresholds(df_50, df_100, df_200)
        aligned = classify_alignment(aligned)
        aligned = create_three_factor_pattern(aligned)

        # Collect returns by pattern
        patterns = aligned.select("three_factor_pattern").unique().to_series().drop_nulls().to_list()

        for pattern in patterns:
            if pattern is None:
                continue
            returns = (
                aligned.filter(pl.col("three_factor_pattern") == pattern)
                .select("fwd_return")
                .drop_nulls()
                .to_series()
                .to_list()
            )
            if pattern not in pattern_returns:
                pattern_returns[pattern] = []
            pattern_returns[pattern].extend(returns)

    log_info("Returns collected", n_patterns=len(pattern_returns))

    # Analyze each pattern
    results = []
    n_trials = len(pattern_returns)  # Number of patterns tested

    for pattern, returns in pattern_returns.items():
        returns_arr = np.array(returns)
        analysis = analyze_pattern_returns(returns_arr, pattern, n_trials)
        if analysis is not None:
            results.append(analysis)

    log_info("Analysis complete", n_analyzed=len(results))

    return results


def main() -> int:
    """Main entry point."""
    log_info("Starting MRH Framework analysis")

    results = run_psr_analysis()

    # Sort by net return
    results_sorted = sorted(results, key=lambda x: x["net_bps"], reverse=True)

    # Print summary
    print("\n" + "=" * 100)
    print("MRH FRAMEWORK: PSR & MinTRL ANALYSIS")
    print("=" * 100 + "\n")

    print("-" * 100)
    print("TOP PATTERNS BY NET RETURN (with MRH metrics)")
    print("-" * 100)
    print(f"{'Pattern':<40} {'N':>8} {'Net':>8} {'SR':>7} {'Skew':>6} {'Kurt':>6} {'PSR':>6} {'MinTRL':>8} {'Gap':>10} {'Valid':>6}")
    print("-" * 100)

    for r in results_sorted[:30]:
        gap_str = f"{r['gap_net']}" if isinstance(r["gap_net"], int | float) else r["gap_net"]
        valid_str = "YES" if r["is_valid_net"] else "NO"
        print(
            f"{r['pattern']:<40} "
            f"{r['n_samples']:>8,} "
            f"{r['net_bps']:>8.2f} "
            f"{r['net_sharpe_ratio']:>7.4f} "
            f"{r['skewness']:>6.2f} "
            f"{r['kurtosis']:>6.1f} "
            f"{r['psr_net']:>6.3f} "
            f"{r['mintrl_net']:>8} "
            f"{gap_str:>10} "
            f"{valid_str:>6}"
        )

    # Summary statistics
    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)

    valid_patterns = [r for r in results if r["is_valid_net"]]
    high_psr_patterns = [r for r in results if r["psr_net"] >= 0.95]

    print(f"Total patterns analyzed: {len(results)}")
    print(f"Patterns with negative gap (valid): {len(valid_patterns)}")
    print(f"Patterns with PSR >= 0.95: {len(high_psr_patterns)}")

    # DSR analysis
    print("\n" + "-" * 100)
    print("DSR ANALYSIS (accounting for {len(results)} trials)")
    print("-" * 100)

    high_dsr = [r for r in results if r["dsr_net"] >= 0.95]
    print(f"Patterns with DSR >= 0.95: {len(high_dsr)}")

    if high_dsr:
        print("\nTop patterns surviving DSR correction:")
        for r in sorted(high_dsr, key=lambda x: x["net_bps"], reverse=True)[:10]:
            print(f"  {r['pattern']}: net={r['net_bps']:.2f}bps, DSR={r['dsr_net']:.3f}")

    # Print patterns that pass all tests
    print("\n" + "-" * 100)
    print("PRODUCTION-READY PATTERNS (PSR >= 0.95, Gap < 0, DSR >= 0.95)")
    print("-" * 100)

    production_ready = [
        r for r in results
        if r["psr_net"] >= 0.95 and r["is_valid_net"] and r["dsr_net"] >= 0.95
    ]

    if production_ready:
        for r in sorted(production_ready, key=lambda x: x["net_bps"], reverse=True):
            print(
                f"  {r['pattern']}: net={r['net_bps']:.2f}bps, "
                f"PSR={r['psr_net']:.3f}, DSR={r['dsr_net']:.3f}, "
                f"Gap={r['gap_net']}, N={r['n_samples']:,}"
            )
    else:
        print("  None meet all criteria")

    log_info(
        "Analysis complete",
        total=len(results),
        valid=len(valid_patterns),
        high_psr=len(high_psr_patterns),
        high_dsr=len(high_dsr),
        production_ready=len(production_ready),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
