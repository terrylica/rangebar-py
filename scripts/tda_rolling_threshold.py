#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""TDA Regime Detection with Rolling Threshold - No Data Leakage.

This script implements CORRECTED TDA break detection that does NOT use future information.

PREVIOUS (INCORRECT):
    threshold = np.percentile(all_velocities, 95)  # Uses ENTIRE dataset

CORRECTED (THIS SCRIPT):
    threshold[i] = np.percentile(velocities[:i], 95)  # Uses ONLY past data

This ensures:
- Break detection at time i uses only velocity data from times < i
- No future volatility information leaks into regime assignments

Issue #52, #56: Adversarial Audit Data Leakage Fix
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


def detect_tda_breaks_rolling(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 50,
    threshold_pct: float = 95,
    min_history: int = 20,
) -> list[int]:
    """TDA break detection using ROLLING threshold (no data leakage).

    CORRECTED: At each point i, threshold is computed using only velocities[0:i].

    Args:
        returns: Log returns array
        window_size: Size of rolling window for TDA embedding
        step_size: Step size between windows
        threshold_pct: Percentile for break threshold (computed on past only)
        min_history: Minimum velocity observations before computing threshold

    Returns:
        List of break indices (in original series coordinates)
    """
    n = len(returns)
    l2_norms = []
    indices = []

    # Phase 1: Compute all L2 norms
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

    # Phase 2: Detect breaks using ROLLING threshold
    break_indices = []

    for i in range(len(velocity)):
        if i < min_history:
            # Not enough history to compute reliable threshold
            continue

        # Threshold computed using ONLY past velocities (0 to i-1)
        historical_velocities = np.abs(velocity[:i])
        threshold = np.percentile(historical_velocities, threshold_pct)

        # Compare current velocity against historical threshold
        if np.abs(velocity[i]) > threshold:
            break_indices.append(indices[i + 1])

    return break_indices


def detect_tda_breaks_global(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 50,
    threshold_pct: float = 95,
) -> list[int]:
    """TDA break detection using GLOBAL threshold (HAS data leakage - for comparison)."""
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

    # LEAKY: Uses entire dataset
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


def test_odd_within_regime(
    df: pl.DataFrame,
    regime: str,
    pattern_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> list[dict]:
    """Test ODD robustness within a single TDA regime using sub-periods."""
    regime_df = df.filter(pl.col("tda_regime") == regime)

    if len(regime_df) < min_samples * 4:
        return []

    # Split regime into quarters for ODD testing
    regime_df = regime_df.with_row_index("_row_idx")
    n = len(regime_df)
    quarter_size = n // 4

    if quarter_size < min_samples:
        return []

    # Assign sub-periods
    regime_df = regime_df.with_columns(
        (pl.col("_row_idx") // quarter_size).clip(0, 3).cast(pl.Utf8).alias("sub_period")
    )

    results = []
    patterns = regime_df[pattern_col].drop_nulls().unique().to_list()

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_df = regime_df.filter(pl.col(pattern_col) == pattern)
        sub_periods = pattern_df["sub_period"].unique().to_list()

        if len(sub_periods) < 4:
            continue

        period_stats = []
        for sp in sorted(sub_periods):
            sp_df = pattern_df.filter(pl.col("sub_period") == sp)
            returns = sp_df["fwd_ret_1"].drop_nulls().to_numpy()

            if len(returns) < min_samples:
                continue

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0

            period_stats.append({
                "sub_period": sp,
                "n": len(returns),
                "mean_bps": mean_ret * 10000,
                "t_stat": t_stat,
            })

        if len(period_stats) < 4:
            continue

        t_stats = [p["t_stat"] for p in period_stats]
        signs = [1 if t > 0 else -1 for t in t_stats]

        all_same_sign = len(set(signs)) == 1
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)

        is_odd_robust = all_same_sign and all_significant

        total_n = sum(p["n"] for p in period_stats)
        mean_t = float(np.mean(t_stats))

        results.append({
            "regime": regime,
            "pattern": pattern,
            "n_sub_periods": len(period_stats),
            "total_n": total_n,
            "mean_t_stat": round(mean_t, 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "same_sign": all_same_sign,
            "all_significant": all_significant,
            "is_odd_robust": is_odd_robust,
        })

    return results


def main() -> None:
    """Compare global vs rolling TDA threshold for break detection."""
    log_info("Starting TDA rolling threshold comparison analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    global_results = []
    rolling_results = []

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

        # Add direction and temporal-safe pattern (past only)
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
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
        )

        # Compute log returns for TDA
        log_returns = (
            df_pl.filter(pl.col("Close").is_not_null())
            .select((pl.col("Close") / pl.col("Close").shift(1)).log())
            .drop_nulls()
            .to_numpy()
            .flatten()
        )

        # Subsample for TDA
        subsample_factor = max(1, len(log_returns) // 10000)
        subsampled = log_returns[::subsample_factor]

        # Detect TDA breaks with GLOBAL threshold (leaky)
        global_breaks_sub = detect_tda_breaks_global(
            subsampled,
            window_size=100,
            step_size=50,
            threshold_pct=95,
        )
        global_breaks = [idx * subsample_factor for idx in global_breaks_sub]

        # Detect TDA breaks with ROLLING threshold (no leakage)
        rolling_breaks_sub = detect_tda_breaks_rolling(
            subsampled,
            window_size=100,
            step_size=50,
            threshold_pct=95,
            min_history=20,
        )
        rolling_breaks = [idx * subsample_factor for idx in rolling_breaks_sub]

        log_info(
            "TDA breaks detected",
            symbol=symbol,
            global_breaks=len(global_breaks),
            rolling_breaks=len(rolling_breaks),
        )

        # Test ODD robustness with GLOBAL regimes
        if len(global_breaks) > 0:
            df_global = assign_tda_regimes(df_pl, global_breaks)
            regimes = df_global["tda_regime"].unique().to_list()

            for regime in sorted(regimes):
                regime_results = test_odd_within_regime(
                    df_global, regime, "pattern_2bar", min_samples=100, min_t_stat=5.0
                )
                for r in regime_results:
                    r["symbol"] = symbol
                    r["threshold_type"] = "global"
                    global_results.append(r)

        # Test ODD robustness with ROLLING regimes
        if len(rolling_breaks) > 0:
            df_rolling = assign_tda_regimes(df_pl, rolling_breaks)
            regimes = df_rolling["tda_regime"].unique().to_list()

            for regime in sorted(regimes):
                regime_results = test_odd_within_regime(
                    df_rolling, regime, "pattern_2bar", min_samples=100, min_t_stat=5.0
                )
                for r in regime_results:
                    r["symbol"] = symbol
                    r["threshold_type"] = "rolling"
                    rolling_results.append(r)

        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            global_odd=sum(1 for r in global_results if r["symbol"] == symbol and r["is_odd_robust"]),
            rolling_odd=sum(1 for r in rolling_results if r["symbol"] == symbol and r["is_odd_robust"]),
        )

    # Print results
    print("\n" + "=" * 140)
    print("TDA THRESHOLD COMPARISON: GLOBAL (LEAKY) vs ROLLING (TEMPORAL-SAFE)")
    print("=" * 140)

    print("\n" + "-" * 140)
    print("GLOBAL THRESHOLD RESULTS (HAS DATA LEAKAGE)")
    print("-" * 140)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'Pattern':<8} {'N Sub-P':>8} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'ODD':>6}"
    )
    print("-" * 140)

    for r in sorted(global_results, key=lambda x: (x["symbol"], x["regime"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['regime']:<18} {r['pattern'] or 'N/A':<8} "
            f"{r['n_sub_periods']:>8} {r['total_n']:>12,} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} "
            f"{'YES' if r['is_odd_robust'] else 'no':>6}"
        )

    print("\n" + "-" * 140)
    print("ROLLING THRESHOLD RESULTS (TEMPORAL-SAFE)")
    print("-" * 140)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'Pattern':<8} {'N Sub-P':>8} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'ODD':>6}"
    )
    print("-" * 140)

    for r in sorted(rolling_results, key=lambda x: (x["symbol"], x["regime"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['regime']:<18} {r['pattern'] or 'N/A':<8} "
            f"{r['n_sub_periods']:>8} {r['total_n']:>12,} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} "
            f"{'YES' if r['is_odd_robust'] else 'no':>6}"
        )

    # Summary
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    global_odd = sum(1 for r in global_results if r["is_odd_robust"])
    rolling_odd = sum(1 for r in rolling_results if r["is_odd_robust"])

    print("\nGlobal threshold (leaky):")
    print(f"  Total pattern-regime combinations: {len(global_results)}")
    print(f"  ODD robust: {global_odd} ({100*global_odd/max(len(global_results), 1):.1f}%)")

    print("\nRolling threshold (temporal-safe):")
    print(f"  Total pattern-regime combinations: {len(rolling_results)}")
    print(f"  ODD robust: {rolling_odd} ({100*rolling_odd/max(len(rolling_results), 1):.1f}%)")

    # Key findings
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    if rolling_odd == 0 and global_odd > 0:
        print(f"\n✗ Global threshold: {global_odd} ODD robust patterns")
        print(f"✗ Rolling threshold: {rolling_odd} ODD robust patterns")
        print("  → The {global_odd} patterns from global threshold are artifacts of data leakage")
    elif rolling_odd > 0:
        print(f"\n✓ Rolling threshold finds {rolling_odd} genuine ODD robust patterns")
        print("  → These are valid patterns using only historical information")
    else:
        print("\n✗ Neither threshold approach finds ODD robust patterns")
        print("  → Pattern predictability does not exist in this data")

    log_info(
        "Analysis complete",
        global_tested=len(global_results),
        global_odd=global_odd,
        rolling_tested=len(rolling_results),
        rolling_odd=rolling_odd,
    )


if __name__ == "__main__":
    main()
