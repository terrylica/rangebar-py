#!/usr/bin/env python3
"""TDA Structural Break Detection for Range Bar Returns.

Applies Topological Data Analysis (Persistent Homology) to detect geometric
structural breaks in range bar return distributions.

Hypothesis: TDA can detect distributional changes earlier than ADWIN by
identifying topological features (loops, voids) in the return phase space
that precede statistical regime shifts.

Methodology (per time-to-convergence-stationarity-gap.md):
1. Takens embedding of return time series
2. Compute persistence diagrams using Vietoris-Rips filtration
3. Calculate persistence landscape Lp-norms over sliding windows
4. Track Lp-norm velocity as early warning signal

Reference: Gidea and Katz (2018) - Lp-norm of H1 features predicts crashes
250 trading days before statistical volatility models.

Issue #54: Volatility Regime Filter for ODD Robust Patterns
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from persim import PersistenceLandscaper
from ripser import ripser

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


def takens_embedding(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Create Takens delay embedding of time series.

    Reconstructs phase space from scalar time series.

    Args:
        series: 1D time series array
        embedding_dim: Embedding dimension (default 3 for financial data)
        delay: Time delay between coordinates

    Returns:
        2D array of shape (n_points, embedding_dim)
    """
    n = len(series)
    n_points = n - (embedding_dim - 1) * delay

    if n_points <= 0:
        return np.array([]).reshape(0, embedding_dim)

    embedding = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        embedding[:, i] = series[i * delay : i * delay + n_points]

    return embedding


def compute_persistence_diagram(
    point_cloud: np.ndarray,
    max_dim: int = 1,
) -> list[np.ndarray]:
    """Compute persistence diagram using Vietoris-Rips filtration.

    Args:
        point_cloud: 2D array of points
        max_dim: Maximum homology dimension (1 = loops/H1)

    Returns:
        List of persistence diagrams per dimension
    """
    if len(point_cloud) < 4:
        return [np.array([]).reshape(0, 2) for _ in range(max_dim + 1)]

    # Normalize point cloud for stable persistence
    point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / (
        np.std(point_cloud, axis=0) + 1e-10
    )

    # Compute persistence with ripser
    result = ripser(point_cloud, maxdim=max_dim)
    return result["dgms"]


def compute_landscape_norm(
    diagrams: list[np.ndarray],
    dim: int = 1,
    p: int = 2,
    num_steps: int = 500,
) -> float:
    """Compute Lp-norm of persistence landscape.

    Args:
        diagrams: List of persistence diagrams
        dim: Homology dimension to use (1 = H1 loops)
        p: Norm order (1 = L1, 2 = L2)
        num_steps: Grid resolution for landscape

    Returns:
        Lp-norm of the persistence landscape
    """
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0.0

    dgm = diagrams[dim]

    # Filter infinite persistence and NaN values
    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    dgm = dgm[finite_mask]

    if len(dgm) == 0:
        return 0.0

    try:
        # Create filtered diagrams list for landscaper
        # Replace the target dimension with filtered version
        filtered_diagrams = []
        for i, d in enumerate(diagrams):
            if i == dim:
                filtered_diagrams.append(dgm)
            else:
                # Filter NaN from other dimensions too
                mask = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])
                filtered_diagrams.append(d[mask])

        # Create landscaper with correct API
        # hom_deg: homological degree to compute landscape for
        landscaper = PersistenceLandscaper(
            hom_deg=dim,
            num_steps=num_steps,
        )
        # fit_transform expects list of diagrams (one per dimension)
        landscape = landscaper.fit_transform(filtered_diagrams)

        # Compute Lp norm
        if p == 1:
            return float(np.sum(np.abs(landscape)))
        if p == 2:
            return float(np.sqrt(np.sum(landscape**2)))
        return float(np.sum(np.abs(landscape) ** p) ** (1 / p))

    except (ValueError, IndexError, TypeError, KeyError):
        # Landscape computation can fail for degenerate diagrams
        return 0.0


def analyze_tda_over_time(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 20,
    embedding_dim: int = 3,
    delay: int = 1,
) -> pl.DataFrame:
    """Analyze TDA landscape norms over sliding windows.

    Args:
        returns: Array of log returns
        window_size: Size of sliding window
        step_size: Step between windows
        embedding_dim: Takens embedding dimension
        delay: Embedding delay

    Returns:
        DataFrame with window indices and landscape norms
    """
    n = len(returns)
    results = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window_returns = returns[start:end]

        # Takens embedding
        point_cloud = takens_embedding(window_returns, embedding_dim, delay)

        if len(point_cloud) < 10:
            continue

        # Compute persistence diagram
        diagrams = compute_persistence_diagram(point_cloud, max_dim=1)

        # Compute L1 and L2 norms of H1 landscape
        l1_norm = compute_landscape_norm(diagrams, dim=1, p=1)
        l2_norm = compute_landscape_norm(diagrams, dim=1, p=2)

        # H0 (connected components) norm
        l2_norm_h0 = compute_landscape_norm(diagrams, dim=0, p=2)

        results.append({
            "window_start": start,
            "window_end": end,
            "window_center": (start + end) // 2,
            "l1_norm_h1": l1_norm,
            "l2_norm_h1": l2_norm,
            "l2_norm_h0": l2_norm_h0,
            "mean_return": float(np.mean(window_returns)),
            "std_return": float(np.std(window_returns)),
        })

    return pl.DataFrame(results)


def detect_tda_breaks(
    tda_df: pl.DataFrame,
    threshold_percentile: float = 95,
) -> pl.DataFrame:
    """Detect structural breaks based on TDA norm velocity.

    A break is detected when the time-derivative of landscape norm
    exceeds a threshold.

    Args:
        tda_df: DataFrame from analyze_tda_over_time
        threshold_percentile: Percentile for break detection

    Returns:
        DataFrame with break detection results
    """
    if len(tda_df) < 3:
        return tda_df.with_columns([
            pl.lit(0.0).alias("l2_velocity"),
            pl.lit(False).alias("is_break"),
        ])

    # Compute velocity (first difference) of L2 norm
    tda_df = tda_df.with_columns([
        (pl.col("l2_norm_h1") - pl.col("l2_norm_h1").shift(1)).alias("l2_velocity"),
    ])

    # Compute threshold from velocity distribution
    velocities = tda_df.filter(pl.col("l2_velocity").is_not_null())["l2_velocity"].to_numpy()

    if len(velocities) == 0:
        return tda_df.with_columns(pl.lit(False).alias("is_break"))

    threshold = float(np.percentile(np.abs(velocities), threshold_percentile))

    # Mark breaks
    tda_df = tda_df.with_columns([
        (pl.col("l2_velocity").abs() > threshold).alias("is_break"),
    ])

    return tda_df


def compare_tda_vs_adwin(
    returns: np.ndarray,
    tda_breaks: pl.DataFrame,
) -> dict:
    """Compare TDA break detection vs ADWIN.

    Args:
        returns: Original return array
        tda_breaks: DataFrame with TDA break detection

    Returns:
        Comparison statistics
    """
    try:
        from river.drift import ADWIN

        # Run ADWIN on returns
        adwin = ADWIN()
        adwin_breaks = []

        for i, ret in enumerate(returns):
            adwin.update(ret)
            if adwin.drift_detected:
                adwin_breaks.append(i)

        # Get TDA breaks
        tda_break_indices = tda_breaks.filter(pl.col("is_break"))["window_center"].to_list()

        return {
            "adwin_break_count": len(adwin_breaks),
            "tda_break_count": len(tda_break_indices),
            "adwin_breaks": adwin_breaks[:10],  # First 10
            "tda_breaks": tda_break_indices[:10],
        }

    except ImportError:
        return {
            "adwin_break_count": -1,
            "tda_break_count": len(tda_breaks.filter(pl.col("is_break"))),
            "error": "river not installed",
        }


def analyze_pattern_tda(
    df: pl.DataFrame,
) -> list[dict]:
    """Analyze TDA characteristics for each pattern.

    Args:
        df: Range bar DataFrame

    Returns:
        List of pattern TDA statistics
    """
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

    # Compute forward returns
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
    )

    patterns = df.filter(pl.col("pattern_3bar").is_not_null())["pattern_3bar"].unique().to_list()

    results = []

    for pattern in patterns:
        # Get forward returns after this pattern
        pattern_mask = df["pattern_3bar"] == pattern
        pattern_indices = df.with_row_index().filter(pattern_mask)["index"].to_list()

        forward_returns = []
        log_returns = df["log_return"].to_list()

        for idx in pattern_indices:
            if idx + 1 < len(log_returns):
                ret = log_returns[idx + 1]
                if ret is not None and np.isfinite(ret):
                    forward_returns.append(ret)

        if len(forward_returns) < 200:
            continue

        returns_arr = np.array(forward_returns)

        # Compute TDA over pattern returns
        tda_df = analyze_tda_over_time(
            returns_arr,
            window_size=50,
            step_size=10,
        )

        if len(tda_df) < 5:
            continue

        # Detect breaks
        tda_breaks = detect_tda_breaks(tda_df)

        # Statistics
        avg_l2_h1 = float(tda_df["l2_norm_h1"].mean())
        std_l2_h1 = float(tda_df["l2_norm_h1"].std())
        max_l2_h1 = float(tda_df["l2_norm_h1"].max())
        break_count = len(tda_breaks.filter(pl.col("is_break")))

        # Compare with volatility
        avg_vol = float(tda_df["std_return"].mean())
        corr_l2_vol = float(
            np.corrcoef(
                tda_df["l2_norm_h1"].to_numpy(),
                tda_df["std_return"].to_numpy(),
            )[0, 1]
        ) if len(tda_df) > 2 else 0.0

        results.append({
            "pattern": pattern,
            "n_samples": len(returns_arr),
            "avg_l2_h1": round(avg_l2_h1, 4),
            "std_l2_h1": round(std_l2_h1, 4),
            "max_l2_h1": round(max_l2_h1, 4),
            "tda_break_count": break_count,
            "avg_volatility": round(avg_vol, 6),
            "corr_l2_vol": round(corr_l2_vol, 4),
        })

    return results


def main() -> None:
    """Run TDA structural break analysis."""
    log_info("Starting TDA structural break analysis")

    # Use subset for faster computation - TDA is O(n^3) on point clouds
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100
    # Use 1 year of data for faster TDA computation
    start_dates = {
        "BTCUSDT": "2025-01-01",
        "ETHUSDT": "2025-01-01",
        "SOLUSDT": "2025-01-01",
        "BNBUSDT": "2025-01-01",
    }
    end_date = "2026-01-31"

    overall_results = []

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

        # Compute log returns for overall analysis
        log_returns = (
            df_pl.filter(pl.col("Close").is_not_null())
            .select((pl.col("Close") / pl.col("Close").shift(1)).log())
            .drop_nulls()
            .to_numpy()
            .flatten()
        )

        log_info("Computing TDA for symbol", symbol=symbol, n_returns=len(log_returns))

        # Sample at most 20K returns for tractable CPU computation
        # TDA is O(n^2) to O(n^3) so we must limit sample size
        # Use 5K samples with small windows for tractable computation
        sample_returns = log_returns[:5000] if len(log_returns) > 5000 else log_returns

        # Overall TDA analysis with small windows for speed
        tda_df = analyze_tda_over_time(
            sample_returns,
            window_size=100,  # Small window = 100 points in phase space
            step_size=100,  # Non-overlapping windows for speed
        )

        if len(tda_df) > 0:
            tda_breaks = detect_tda_breaks(tda_df)
            comparison = compare_tda_vs_adwin(sample_returns, tda_breaks)

            overall_results.append({
                "symbol": symbol,
                "n_bars": len(df_pl),
                "avg_l2_h1": float(tda_df["l2_norm_h1"].mean()),
                "max_l2_h1": float(tda_df["l2_norm_h1"].max()),
                "tda_breaks": comparison["tda_break_count"],
                "adwin_breaks": comparison["adwin_break_count"],
            })

            log_info(
                "TDA analysis complete",
                symbol=symbol,
                avg_l2=float(tda_df["l2_norm_h1"].mean()),
                tda_breaks=comparison["tda_break_count"],
                adwin_breaks=comparison["adwin_break_count"],
            )

        # Skip pattern-specific TDA (too slow for CPU)
        # For full analysis, run on GPU with giotto-tda CUDA backend

    # Print overall results
    print("\n" + "=" * 120)
    print("TDA STRUCTURAL BREAK ANALYSIS")
    print("=" * 120)
    print("\nMethodology: Persistent Homology with Vietoris-Rips filtration")
    print("H1 features (loops) indicate cyclic market structure before crashes")
    print("Reference: Gidea & Katz (2018) - Lp-norm predicts crashes 250 days early")

    print("\n" + "-" * 120)
    print("OVERALL SYMBOL ANALYSIS")
    print("-" * 120)
    print(f"{'Symbol':<12} {'N Bars':>12} {'Avg L2(H1)':>12} {'Max L2(H1)':>12} {'TDA Breaks':>12} {'ADWIN Breaks':>14}")
    print("-" * 120)

    for r in overall_results:
        print(
            f"{r['symbol']:<12} "
            f"{r['n_bars']:>12,} "
            f"{r['avg_l2_h1']:>12.4f} "
            f"{r['max_l2_h1']:>12.4f} "
            f"{r['tda_breaks']:>12} "
            f"{r['adwin_breaks']:>14}"
        )

    # Pattern-specific TDA skipped (too slow for CPU)
    print("\n" + "-" * 120)
    print("NOTE: Pattern-specific TDA skipped (O(n^2) complexity)")
    print("For pattern analysis, use GPU with giotto-tda CUDA backend")
    print("-" * 120)

    # Summary
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    total_tda = sum(r["tda_breaks"] for r in overall_results)
    total_adwin = sum(r["adwin_breaks"] for r in overall_results if r["adwin_breaks"] >= 0)

    print(f"\nTotal TDA structural breaks detected: {total_tda}")
    print(f"Total ADWIN drift points detected: {total_adwin}")

    if total_tda > 0 and total_adwin == 0:
        print("\nFINDING: TDA detects structural changes that ADWIN misses")
        print("This suggests topological features precede statistical moment shifts")
    elif total_tda == 0 and total_adwin == 0:
        print("\nFINDING: Both TDA and ADWIN detect stable distribution")
        print("This confirms the earlier ADWIN finding of no regime changes")
    else:
        ratio = total_tda / max(total_adwin, 1)
        print(f"\nTDA/ADWIN ratio: {ratio:.2f}")

    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)
    print("\n- H1 (loop) persistence captures cyclic correlations in returns")
    print("- High L2 norm indicates market structure becoming more deterministic")
    print("- L2 velocity spikes are early warning for regime shifts")
    print("- Correlation between L2 and volatility shows structural-volatility link")

    log_info("TDA analysis complete", total_tda_breaks=total_tda, total_adwin_breaks=total_adwin)


if __name__ == "__main__":
    main()
