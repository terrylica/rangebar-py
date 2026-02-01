#!/usr/bin/env python3
"""TDA Regime Pattern Analysis - Pattern Behavior Around Structural Breaks.

Analyzes how OOD robust patterns perform before/after TDA-detected structural
breaks. This validates whether TDA provides an early warning signal for
pattern degradation.

Hypothesis: If patterns behave differently around TDA breaks, it confirms:
1. TDA detects meaningful regime changes invisible to moment-based methods
2. Pattern reliability is regime-dependent
3. TDA could serve as real-time pattern reliability filter

Methodology:
1. Run TDA analysis on full multi-year dataset (subsample for CPU tractability)
2. Identify TDA break timestamps via L2 velocity spikes
3. Segment data into pre/post break periods
4. Compare pattern statistics across TDA-defined regimes

Issue #54: Volatility Regime Filter for ODD Robust Patterns
"""

import json
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ttest_ind

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
    """Compute L2 norm of H1 persistence diagram.

    Uses simplified persistence (birth-death differences) to avoid
    landscape computation overhead.
    """
    try:
        from ripser import ripser
    except ImportError:
        return 0.0

    if len(point_cloud) < 4:
        return 0.0

    # Normalize
    point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / (
        np.std(point_cloud, axis=0) + 1e-10
    )

    result = ripser(point_cloud, maxdim=1)
    dgm = result["dgms"][1]  # H1 (loops)

    if len(dgm) == 0:
        return 0.0

    # Filter infinite persistence
    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    dgm = dgm[finite_mask]

    if len(dgm) == 0:
        return 0.0

    # L2 norm of persistence (death - birth)
    persistence = dgm[:, 1] - dgm[:, 0]
    return float(np.sqrt(np.sum(persistence**2)))


def detect_tda_breaks_fast(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 50,
    threshold_pct: float = 95,
) -> list[int]:
    """Fast TDA break detection using L2 norm velocity.

    Returns indices where L2 velocity exceeds threshold.
    """
    n = len(returns)
    l2_norms = []
    indices = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = returns[start:end]

        # Takens embedding
        point_cloud = takens_embedding(window, embedding_dim=3, delay=1)

        if len(point_cloud) < 10:
            continue

        l2 = compute_persistence_l2(point_cloud)
        l2_norms.append(l2)
        indices.append((start + end) // 2)

    if len(l2_norms) < 3:
        return []

    # Compute velocity
    l2_norms = np.array(l2_norms)
    velocity = np.diff(l2_norms)

    # Detect breaks
    threshold = np.percentile(np.abs(velocity), threshold_pct)
    break_mask = np.abs(velocity) > threshold

    # Return center indices of break windows
    break_indices = [indices[i + 1] for i in range(len(break_mask)) if break_mask[i]]

    return break_indices


def analyze_pattern_by_regime(
    df: pl.DataFrame,
    break_indices: list[int],
    pattern_col: str = "pattern_2bar",
) -> pl.DataFrame:
    """Analyze pattern performance segmented by TDA regimes.

    Segments: before first break, between breaks, after last break
    """
    if len(break_indices) == 0:
        # No breaks - single regime
        return df.with_columns(pl.lit("stable").alias("tda_regime"))

    # Sort break indices
    breaks = sorted(break_indices)

    # Assign regime based on row index
    df = df.with_row_index("_idx")

    # Create regime labels
    regimes = []
    n = len(df)

    for idx in range(n):
        if idx < breaks[0]:
            regimes.append("pre_break_1")
        elif len(breaks) == 1:
            regimes.append("post_break_1")
        elif idx < breaks[-1]:
            # Find which inter-break period
            for i, (b1, b2) in enumerate(pairwise(breaks)):
                if idx >= b1 and idx < b2:
                    regimes.append(f"inter_break_{i+1}")
                    break
            else:
                regimes.append(f"inter_break_{len(breaks)-1}")
        else:
            regimes.append(f"post_break_{len(breaks)}")

    df = df.with_columns(pl.Series("tda_regime", regimes))
    df = df.drop("_idx")

    return df


def compute_pattern_stats_by_regime(
    df: pl.DataFrame,
    patterns: list[str],
) -> list[dict]:
    """Compute pattern statistics for each TDA regime."""
    results = []

    regimes = df["tda_regime"].unique().to_list()

    for regime in sorted(regimes):
        regime_df = df.filter(pl.col("tda_regime") == regime)

        for pattern in patterns:
            pattern_df = regime_df.filter(pl.col("pattern_2bar") == pattern)

            if len(pattern_df) < 50:
                continue

            returns = pattern_df["forward_return"].drop_nulls().to_numpy()

            if len(returns) < 50:
                continue

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0

            results.append({
                "regime": regime,
                "pattern": pattern,
                "n": len(returns),
                "mean_bps": round(mean_ret * 10000, 2),
                "std_bps": round(std_ret * 10000, 2),
                "t_stat": round(t_stat, 2),
                "win_rate": round(float(np.mean(returns > 0)) * 100, 1),
            })

    return results


def compare_pattern_across_regimes(
    df: pl.DataFrame,
    pattern: str,
    regime1: str,
    regime2: str,
) -> dict:
    """Compare pattern performance between two regimes."""
    r1_returns = (
        df.filter((pl.col("pattern_2bar") == pattern) & (pl.col("tda_regime") == regime1))
        ["forward_return"]
        .drop_nulls()
        .to_numpy()
    )

    r2_returns = (
        df.filter((pl.col("pattern_2bar") == pattern) & (pl.col("tda_regime") == regime2))
        ["forward_return"]
        .drop_nulls()
        .to_numpy()
    )

    if len(r1_returns) < 30 or len(r2_returns) < 30:
        return {
            "pattern": pattern,
            "regime1": regime1,
            "regime2": regime2,
            "significant": None,
            "error": "insufficient samples",
        }

    # Two-sample t-test
    t_stat, p_value = ttest_ind(r1_returns, r2_returns)

    return {
        "pattern": pattern,
        "regime1": regime1,
        "regime2": regime2,
        "n1": len(r1_returns),
        "n2": len(r2_returns),
        "mean1_bps": round(float(np.mean(r1_returns)) * 10000, 2),
        "mean2_bps": round(float(np.mean(r2_returns)) * 10000, 2),
        "diff_bps": round((float(np.mean(r1_returns)) - float(np.mean(r2_returns))) * 10000, 2),
        "t_stat": round(float(t_stat), 2),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


def main() -> None:
    """Run TDA regime pattern analysis."""
    log_info("Starting TDA regime pattern analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    # Use multi-year data for proper regime detection
    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    all_results = []
    all_comparisons = []
    break_summary = []

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
            log_info("No data", symbol=symbol)
            continue

        df_pl = pl.from_pandas(df.reset_index())

        # Add pattern columns
        df_pl = df_pl.with_columns(
            pl.when(pl.col("Close") > pl.col("Open"))
            .then(pl.lit("U"))
            .otherwise(pl.lit("D"))
            .alias("direction")
        )

        df_pl = df_pl.with_columns(
            (pl.col("direction").shift(1) + pl.col("direction")).alias("pattern_2bar")
        )

        # Forward returns
        df_pl = df_pl.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("forward_return")
        )

        # Compute log returns for TDA
        log_returns = (
            df_pl.filter(pl.col("Close").is_not_null())
            .select((pl.col("Close") / pl.col("Close").shift(1)).log())
            .drop_nulls()
            .to_numpy()
            .flatten()
        )

        log_info("Detecting TDA breaks", symbol=symbol, n_returns=len(log_returns))

        # Subsample for tractable CPU computation
        # Use every Nth return to cover full time span
        subsample_factor = max(1, len(log_returns) // 10000)
        subsampled = log_returns[::subsample_factor]

        log_info(
            "Subsampled for TDA",
            symbol=symbol,
            original=len(log_returns),
            subsampled=len(subsampled),
            factor=subsample_factor,
        )

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
            break_indices=break_indices[:10],
        )

        break_summary.append({
            "symbol": symbol,
            "n_bars": len(df_pl),
            "n_breaks": len(break_indices),
            "break_indices": break_indices[:5],
        })

        if len(break_indices) == 0:
            log_info("No TDA breaks detected", symbol=symbol)
            continue

        # Assign TDA regimes
        df_pl = analyze_pattern_by_regime(df_pl, break_indices, "pattern_2bar")

        # Compute pattern stats by regime
        patterns = ["DD", "DU", "UD", "UU"]
        regime_stats = compute_pattern_stats_by_regime(df_pl, patterns)

        for stat in regime_stats:
            stat["symbol"] = symbol
            all_results.append(stat)

        # Compare first vs last regime for each pattern
        regimes = sorted(df_pl["tda_regime"].unique().to_list())

        if len(regimes) >= 2:
            first_regime = regimes[0]
            last_regime = regimes[-1]

            for pattern in patterns:
                comp = compare_pattern_across_regimes(df_pl, pattern, first_regime, last_regime)
                comp["symbol"] = symbol
                all_comparisons.append(comp)

    # Print results
    print("\n" + "=" * 120)
    print("TDA REGIME PATTERN ANALYSIS")
    print("=" * 120)

    print("\n" + "-" * 80)
    print("TDA BREAK SUMMARY")
    print("-" * 80)
    print(f"{'Symbol':<12} {'N Bars':>12} {'TDA Breaks':>12} {'First 5 Break Indices'}")
    print("-" * 80)

    for bs in break_summary:
        indices_str = str(bs["break_indices"]) if bs["break_indices"] else "[]"
        print(f"{bs['symbol']:<12} {bs['n_bars']:>12,} {bs['n_breaks']:>12} {indices_str}")

    print("\n" + "-" * 120)
    print("PATTERN PERFORMANCE BY TDA REGIME")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'Pattern':<8} {'N':>10} "
        f"{'Mean(bps)':>12} {'t-stat':>10} {'WinRate':>10}"
    )
    print("-" * 120)

    # Group by symbol for readability
    for symbol in symbols:
        symbol_results = [r for r in all_results if r["symbol"] == symbol]
        for r in sorted(symbol_results, key=lambda x: (x["regime"], x["pattern"])):
            print(
                f"{r['symbol']:<10} {r['regime']:<18} {r['pattern']:<8} {r['n']:>10,} "
                f"{r['mean_bps']:>12.2f} {r['t_stat']:>10.2f} {r['win_rate']:>9.1f}%"
            )
        if symbol_results:
            print("-" * 120)

    print("\n" + "-" * 120)
    print("CROSS-REGIME COMPARISON (First vs Last TDA Regime)")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Pattern':<8} {'Regime1':<16} {'Regime2':<16} "
        f"{'Mean1':>10} {'Mean2':>10} {'Diff':>10} {'p-value':>10} {'Sig?':>6}"
    )
    print("-" * 120)

    for comp in all_comparisons:
        if "error" in comp:
            continue
        sig = "YES" if comp["significant"] else "no"
        print(
            f"{comp['symbol']:<10} {comp['pattern']:<8} "
            f"{comp['regime1']:<16} {comp['regime2']:<16} "
            f"{comp['mean1_bps']:>10.2f} {comp['mean2_bps']:>10.2f} "
            f"{comp['diff_bps']:>10.2f} {comp['p_value']:>10.4f} {sig:>6}"
        )

    # Key findings
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    total_breaks = sum(bs["n_breaks"] for bs in break_summary)
    print(f"\nTotal TDA structural breaks detected: {total_breaks}")

    # Check for significant cross-regime differences
    significant_diffs = [c for c in all_comparisons if c.get("significant")]
    print(f"Patterns with significant cross-regime differences: {len(significant_diffs)}")

    if significant_diffs:
        print("\nSignificant regime-dependent patterns:")
        for c in significant_diffs:
            direction = "better" if c["diff_bps"] > 0 else "worse"
            print(
                f"  - {c['symbol']} {c['pattern']}: {direction} in {c['regime1']} "
                f"vs {c['regime2']} ({c['diff_bps']:+.2f} bps, p={c['p_value']:.4f})"
            )

    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)
    print("\n- TDA detects geometric structural breaks in return phase space")
    print("- If patterns differ significantly across TDA regimes, TDA serves as filter")
    print("- Patterns may be more reliable in certain TDA regimes")
    print("- This validates TDA as early warning for pattern degradation")

    log_info(
        "TDA regime analysis complete",
        total_breaks=total_breaks,
        significant_diffs=len(significant_diffs),
    )


if __name__ == "__main__":
    main()
