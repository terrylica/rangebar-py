#!/usr/bin/env python3
"""TDA Regime-Conditioned Pattern Analysis.

Tests whether patterns achieve ODD robustness WITHIN TDA-defined stable regimes
rather than across all periods unconditionally.

Hypothesis: If we condition on TDA regimes (excluding break periods), patterns
may show stable sign consistency that fails in unconditional testing.

Methodology:
1. Segment data by TDA regimes (pre_break, inter_break, post_break)
2. Test pattern ODD robustness within each stable regime
3. Compare: unconditional vs regime-conditioned performance
4. Identify which regimes produce stable patterns

Issue #52: Market Regime Filter for ODD Robust Patterns
Issue #56: TDA Structural Break Detection
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
    """Run TDA regime-conditioned pattern analysis."""
    log_info("Starting TDA regime-conditioned pattern analysis")

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

        # Add direction and pattern columns
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

        # Detect TDA breaks
        break_indices_subsampled = detect_tda_breaks_fast(
            subsampled,
            window_size=100,
            step_size=50,
            threshold_pct=95,
        )

        break_indices = [idx * subsample_factor for idx in break_indices_subsampled]

        log_info(
            "TDA breaks detected",
            symbol=symbol,
            n_breaks=len(break_indices),
        )

        if len(break_indices) == 0:
            continue

        # Assign TDA regimes
        df_pl = assign_tda_regimes(df_pl, break_indices)

        # Test ODD within each regime
        regimes = df_pl["tda_regime"].unique().to_list()

        for regime in sorted(regimes):
            regime_results = test_odd_within_regime(
                df_pl, regime, "pattern_2bar", min_samples=100, min_t_stat=5.0
            )

            for r in regime_results:
                r["symbol"] = symbol
                all_results.append(r)

        log_info(
            "Regime analysis complete",
            symbol=symbol,
            n_regimes=len(regimes),
        )

    # Print results
    print("\n" + "=" * 140)
    print("TDA REGIME-CONDITIONED PATTERN ANALYSIS")
    print("=" * 140)

    print("\n" + "-" * 140)
    print("PATTERN ODD ROBUSTNESS WITHIN TDA REGIMES")
    print("-" * 140)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'Pattern':<8} {'N Sub-P':>8} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'All Sig':>8} {'ODD':>6}"
    )
    print("-" * 140)

    for r in sorted(all_results, key=lambda x: (x["symbol"], x["regime"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['regime']:<18} {r['pattern'] or 'N/A':<8} "
            f"{r['n_sub_periods']:>8} {r['total_n']:>12,} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} {'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['all_significant'] else 'no':>8} "
            f"{'YES' if r['is_odd_robust'] else 'no':>6}"
        )

    # Summary
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    total_tested = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal pattern-regime combinations tested: {total_tested}")
    print(f"Patterns with same sign across sub-periods: {same_sign} ({100*same_sign/max(total_tested,1):.1f}%)")
    print(f"ODD robust patterns (same sign + |t| >= 5): {odd_robust} ({100*odd_robust/max(total_tested,1):.1f}%)")

    # Find patterns that are ODD robust in multiple symbols
    if odd_robust > 0:
        print("\n" + "-" * 140)
        print("ODD ROBUST PATTERNS BY SYMBOL")
        print("-" * 140)

        robust_by_regime_pattern = {}
        for r in all_results:
            if r["is_odd_robust"]:
                key = (r["regime"], r["pattern"])
                if key not in robust_by_regime_pattern:
                    robust_by_regime_pattern[key] = []
                robust_by_regime_pattern[key].append(r["symbol"])

        for (regime, pattern), syms in sorted(robust_by_regime_pattern.items()):
            print(f"  {regime}|{pattern}: {', '.join(sorted(syms))} ({len(syms)} symbols)")

    # Key findings
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    if odd_robust > 0:
        print(f"\n✓ {odd_robust} patterns achieve ODD robustness within TDA regimes")
        print("  → TDA regime conditioning DOES improve stability")
    else:
        print("\n✗ No patterns achieve ODD robustness even within TDA regimes")
        print("  → Pattern instability is more fundamental than regime-driven")

    if same_sign > odd_robust:
        print(f"\n{same_sign - odd_robust} patterns have same sign but insufficient t-stat")
        print("  → Directional consistency exists but signal strength varies")

    log_info(
        "Analysis complete",
        total_tested=total_tested,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
