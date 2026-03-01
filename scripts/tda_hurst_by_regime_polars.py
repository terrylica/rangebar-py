#!/usr/bin/env python3
"""TDA Regime Hurst Exponent Analysis - Long Memory by Structural Break Regime.

Computes the Hurst exponent (H) for each TDA-defined regime to test whether
long memory characteristics differ across structural breaks.

Hypothesis: If TDA breaks detect meaningful structural changes, Hurst exponent
should differ between regimes:
- H > 0.5: Trending/persistent behavior (momentum)
- H = 0.5: Random walk (no memory)
- H < 0.5: Mean-reverting behavior (anti-persistent)

Methodology:
1. Segment returns by TDA regime (pre_break, inter_break, post_break)
2. Compute R/S Hurst exponent for each regime
3. Compare H values across regimes with statistical testing
4. Validate whether TDA breaks correspond to memory structure changes

Issue #56: TDA Structural Break Detection for Range Bar Patterns
"""

import json
import sys
from datetime import datetime, timezone
from itertools import pairwise
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


def compute_hurst_rs(returns: np.ndarray, min_window: int = 10) -> float:
    """Compute Hurst exponent using Rescaled Range (R/S) method.

    The R/S method estimates Hurst by computing (R/S)_n ~ n^H for various
    window sizes n, then fitting H via log-log regression.

    Args:
        returns: Array of log returns
        min_window: Minimum window size for R/S calculation

    Returns:
        Estimated Hurst exponent H
    """
    n = len(returns)
    if n < min_window * 4:
        return np.nan

    # Window sizes: powers of 2 up to n/4
    max_k = int(np.log2(n / 4))
    if max_k < 2:
        return np.nan

    window_sizes = [2**k for k in range(int(np.log2(min_window)), max_k + 1)]

    rs_values = []
    for window in window_sizes:
        n_windows = n // window
        if n_windows < 2:
            continue

        rs_list = []
        for i in range(n_windows):
            segment = returns[i * window : (i + 1) * window]

            # Mean-centered cumulative sum
            mean_seg = np.mean(segment)
            centered = segment - mean_seg
            cumsum = np.cumsum(centered)

            # Range
            r = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            s = np.std(segment, ddof=1)

            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((window, np.mean(rs_list)))

    if len(rs_values) < 3:
        return np.nan

    # Log-log regression: log(R/S) = H * log(n) + c
    log_n = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    # Simple linear regression
    slope, _ = np.polyfit(log_n, log_rs, 1)

    return float(slope)


def takens_embedding(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Create Takens delay embedding of time series."""
    n = len(series)
    n_points = n - (embedding_dim - 1) * delay

    if n_points <= 0:
        return np.array([]).reshape(0, embedding_dim)

    embedding = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        embedding[:, i] = series[i * delay : i * delay + n_points]

    return embedding


def compute_persistence_l2(point_cloud: np.ndarray) -> float:
    """Compute L2 norm of H1 persistence diagram."""
    try:
        from ripser import ripser
    except ImportError:
        return 0.0

    if len(point_cloud) < 4:
        return 0.0

    point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / (
        np.std(point_cloud, axis=0) + 1e-10
    )

    result = ripser(point_cloud, maxdim=1)
    dgm = result["dgms"][1]

    if len(dgm) == 0:
        return 0.0

    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    dgm = dgm[finite_mask]

    if len(dgm) == 0:
        return 0.0

    persistence = dgm[:, 1] - dgm[:, 0]
    return float(np.sqrt(np.sum(persistence**2)))


def detect_tda_breaks_fast(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 50,
    threshold_pct: float = 95,
) -> list[int]:
    """Fast TDA break detection using L2 norm velocity."""
    n = len(returns)
    l2_norms = []
    indices = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = returns[start:end]

        point_cloud = takens_embedding(window, embedding_dim=3, delay=1)

        if len(point_cloud) < 10:
            continue

        l2 = compute_persistence_l2(point_cloud)
        l2_norms.append(l2)
        indices.append((start + end) // 2)

    if len(l2_norms) < 3:
        return []

    l2_norms = np.array(l2_norms)
    velocity = np.diff(l2_norms)

    threshold = np.percentile(np.abs(velocity), threshold_pct)
    break_mask = np.abs(velocity) > threshold

    break_indices = [indices[i + 1] for i in range(len(break_mask)) if break_mask[i]]

    return break_indices


def assign_tda_regimes(
    df: pl.DataFrame,
    break_indices: list[int],
) -> pl.DataFrame:
    """Assign TDA regime labels based on break indices."""
    if len(break_indices) == 0:
        return df.with_columns(pl.lit("stable").alias("tda_regime"))

    breaks = sorted(break_indices)
    df = df.with_row_index("_idx")
    n = len(df)

    regimes = []
    for idx in range(n):
        if idx < breaks[0]:
            regimes.append("pre_break_1")
        elif len(breaks) == 1:
            regimes.append("post_break_1")
        elif idx < breaks[-1]:
            for i, (b1, b2) in enumerate(pairwise(breaks)):
                if b1 <= idx < b2:
                    regimes.append(f"inter_break_{i + 1}")
                    break
            else:
                regimes.append(f"inter_break_{len(breaks) - 1}")
        else:
            regimes.append(f"post_break_{len(breaks)}")

    df = df.with_columns(pl.Series("tda_regime", regimes))
    return df.drop("_idx")


def compute_hurst_by_regime(
    df: pl.DataFrame,
    min_samples: int = 500,
) -> list[dict]:
    """Compute Hurst exponent for each TDA regime."""
    results = []

    regimes = df["tda_regime"].unique().to_list()

    for regime in sorted(regimes):
        regime_df = df.filter(pl.col("tda_regime") == regime)

        if len(regime_df) < min_samples:
            continue

        returns = regime_df["log_return"].drop_nulls().to_numpy()

        if len(returns) < min_samples:
            continue

        h = compute_hurst_rs(returns)

        if np.isnan(h):
            continue

        # Interpretation
        if h > 0.55:
            interpretation = "trending/persistent"
        elif h < 0.45:
            interpretation = "mean-reverting"
        else:
            interpretation = "random walk"

        results.append({
            "regime": regime,
            "n_bars": len(returns),
            "hurst_h": round(h, 4),
            "interpretation": interpretation,
            "mean_return_bps": round(float(np.mean(returns)) * 10000, 2),
            "std_return_bps": round(float(np.std(returns)) * 10000, 2),
        })

    return results


def compare_hurst_across_regimes(
    results: list[dict],
) -> dict:
    """Compare Hurst exponents across regimes with statistical testing."""
    if len(results) < 2:
        return {"error": "insufficient regimes for comparison"}

    hurst_values = [r["hurst_h"] for r in results]

    return {
        "min_hurst": round(min(hurst_values), 4),
        "max_hurst": round(max(hurst_values), 4),
        "range": round(max(hurst_values) - min(hurst_values), 4),
        "mean_hurst": round(float(np.mean(hurst_values)), 4),
        "std_hurst": round(float(np.std(hurst_values)), 4),
    }


def main() -> None:
    """Run TDA regime Hurst exponent analysis."""
    log_info("Starting TDA regime Hurst exponent analysis")

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
    all_comparisons = []

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
            log_info("No data", symbol=symbol)
            continue

        df_pl = pl.from_pandas(df.reset_index())

        # Compute log returns
        df_pl = df_pl.with_columns(
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
        )

        log_returns = (
            df_pl.filter(pl.col("log_return").is_not_null())["log_return"]
            .to_numpy()
        )

        log_info("Detecting TDA breaks", symbol=symbol, n_returns=len(log_returns))

        # Subsample for tractable CPU computation
        subsample_factor = max(1, len(log_returns) // 10000)
        subsampled = log_returns[::subsample_factor]

        # Detect TDA breaks
        break_indices_subsampled = detect_tda_breaks_fast(
            subsampled,
            window_size=100,
            step_size=50,
            threshold_pct=95,
        )

        # Scale break indices back to original
        break_indices = [idx * subsample_factor for idx in break_indices_subsampled]

        log_info(
            "TDA breaks detected",
            symbol=symbol,
            n_breaks=len(break_indices),
        )

        if len(break_indices) == 0:
            log_info("No TDA breaks detected", symbol=symbol)
            continue

        # Assign TDA regimes
        df_pl = assign_tda_regimes(df_pl, break_indices)

        # Compute Hurst by regime
        hurst_results = compute_hurst_by_regime(df_pl, min_samples=500)

        for r in hurst_results:
            r["symbol"] = symbol
            all_results.append(r)

        # Compare across regimes
        comparison = compare_hurst_across_regimes(hurst_results)
        comparison["symbol"] = symbol
        comparison["n_regimes"] = len(hurst_results)
        all_comparisons.append(comparison)

        log_info(
            "Hurst analysis complete",
            symbol=symbol,
            n_regimes=len(hurst_results),
        )

    # Print results
    print("\n" + "=" * 120)
    print("TDA REGIME HURST EXPONENT ANALYSIS")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("HURST EXPONENT BY TDA REGIME")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'N Bars':>10} {'Hurst H':>10} "
        f"{'Interpretation':<20} {'Mean(bps)':>12} {'Std(bps)':>12}"
    )
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: (x["symbol"], x["regime"])):
        print(
            f"{r['symbol']:<10} {r['regime']:<18} {r['n_bars']:>10,} {r['hurst_h']:>10.4f} "
            f"{r['interpretation']:<20} {r['mean_return_bps']:>12.2f} {r['std_return_bps']:>12.2f}"
        )

    print("\n" + "-" * 120)
    print("CROSS-REGIME HURST COMPARISON")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'N Regimes':>12} {'Min H':>10} {'Max H':>10} "
        f"{'Range':>10} {'Mean H':>10} {'Std H':>10}"
    )
    print("-" * 120)

    for c in all_comparisons:
        if "error" in c:
            print(f"{c['symbol']:<10} {c['error']}")
            continue
        print(
            f"{c['symbol']:<10} {c['n_regimes']:>12} {c['min_hurst']:>10.4f} "
            f"{c['max_hurst']:>10.4f} {c['range']:>10.4f} "
            f"{c['mean_hurst']:>10.4f} {c['std_hurst']:>10.4f}"
        )

    # Key findings
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    # Count regimes by interpretation
    trending = sum(1 for r in all_results if r["interpretation"] == "trending/persistent")
    mean_rev = sum(1 for r in all_results if r["interpretation"] == "mean-reverting")
    random_walk = sum(1 for r in all_results if r["interpretation"] == "random walk")

    print(f"\nTotal regimes analyzed: {len(all_results)}")
    print(f"  Trending/persistent (H > 0.55): {trending}")
    print(f"  Random walk (0.45 ≤ H ≤ 0.55): {random_walk}")
    print(f"  Mean-reverting (H < 0.45): {mean_rev}")

    # Check if Hurst varies significantly across regimes
    all_hurst = [r["hurst_h"] for r in all_results]
    if all_hurst:
        overall_range = max(all_hurst) - min(all_hurst)
        print(f"\nOverall Hurst range: {overall_range:.4f}")

        if overall_range > 0.1:
            print("  → Hurst varies significantly across TDA regimes")
            print("  → TDA breaks correspond to changes in long memory structure")
        else:
            print("  → Hurst is relatively stable across TDA regimes")
            print("  → TDA breaks may detect other structural changes (not memory)")

    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)
    print("\n- Hurst > 0.5 indicates trending behavior (momentum)")
    print("- Hurst = 0.5 indicates random walk (no memory)")
    print("- Hurst < 0.5 indicates mean-reversion (anti-persistent)")
    print("- If Hurst changes across TDA regimes, patterns may need regime-specific calibration")

    log_info(
        "TDA regime Hurst analysis complete",
        total_regimes=len(all_results),
        trending=trending,
        mean_reverting=mean_rev,
        random_walk=random_walk,
    )


if __name__ == "__main__":
    main()
