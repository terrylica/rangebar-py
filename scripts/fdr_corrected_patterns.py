#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Pattern Testing with Benjamini-Hochberg FDR Correction.

This script implements CORRECTED statistical testing for ODD robustness
with multiple testing correction.

PREVIOUS (INCORRECT):
    - 176+ tests with |t| >= 5.0 threshold
    - No correction for multiple comparisons
    - Expected false discovery rate: 8.4x (p=0.0000006 * 176 = 0.0001 expected FDR)

CORRECTED (THIS SCRIPT):
    - Benjamini-Hochberg FDR control at alpha = 0.05
    - Bonferroni-corrected threshold as comparison
    - Reports both raw and corrected significance

Issue #52, #56: Adversarial Audit Statistical Validity Fix
"""

import json
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


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values from multiple tests
        alpha: Desired FDR control level

    Returns:
        Boolean array indicating which tests pass FDR control
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH threshold: p(k) <= k/n * alpha
    thresholds = np.arange(1, n + 1) / n * alpha

    # Find largest k where p(k) <= threshold
    passes = sorted_p <= thresholds

    # All tests up to and including the largest passing one are significant
    if not np.any(passes):
        return np.zeros(n, dtype=bool)

    # Find the largest passing index
    last_passing = np.max(np.where(passes)[0])

    # Mark all indices up to last_passing as significant
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices[: last_passing + 1]] = True

    return significant


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Compute Bonferroni-corrected t-statistic threshold.

    Args:
        n_tests: Number of tests being performed
        alpha: Desired family-wise error rate

    Returns:
        t-statistic threshold for significance
    """
    # Two-tailed test, so divide alpha by 2
    corrected_alpha = alpha / (2 * n_tests)
    # Use large df (approximating normal distribution)
    return stats.norm.ppf(1 - corrected_alpha)


def test_pattern_returns(
    returns: np.ndarray,
) -> dict:
    """Compute test statistics for pattern returns.

    Returns:
        Dictionary with mean, std, t-stat, p-value, n
    """
    n = len(returns)
    if n < 10:
        return {
            "n": n,
            "mean_bps": np.nan,
            "std_bps": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    se = std_ret / np.sqrt(n)

    if se == 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = mean_ret / se
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return {
        "n": n,
        "mean_bps": mean_ret * 10000,
        "std_bps": std_ret * 10000,
        "t_stat": round(t_stat, 4),
        "p_value": p_value,
    }


def test_odd_robustness_with_fdr(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    raw_t_threshold: float = 5.0,
    fdr_alpha: float = 0.05,
) -> tuple[list[dict], dict]:
    """Test ODD robustness with FDR correction.

    Returns:
        Tuple of (results list, summary dict)
    """
    # Add period column (year-quarter)
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8)
            + "-Q"
            + ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    results = []
    patterns = df[pattern_col].drop_nulls().unique().to_list()

    for pattern in patterns:
        if pattern is None:
            continue

        pattern_df = df.filter(pl.col(pattern_col) == pattern)
        periods = pattern_df["period"].drop_nulls().unique().to_list()

        if len(periods) < 4:
            continue

        period_stats = []
        for period in sorted(periods):
            period_df = pattern_df.filter(pl.col("period") == period)
            returns = period_df[return_col].drop_nulls().to_numpy()

            if len(returns) < min_samples:
                continue

            stats_dict = test_pattern_returns(returns)
            stats_dict["period"] = period
            period_stats.append(stats_dict)

        if len(period_stats) < 4:
            continue

        # Aggregate statistics
        t_stats = [p["t_stat"] for p in period_stats]
        p_values = [p["p_value"] for p in period_stats]
        signs = [1 if t > 0 else -1 for t in t_stats]

        all_same_sign = len(set(signs)) == 1
        all_raw_significant = all(abs(t) >= raw_t_threshold for t in t_stats)

        results.append({
            "pattern": pattern,
            "n_periods": len(period_stats),
            "total_n": sum(p["n"] for p in period_stats),
            "mean_t_stat": round(float(np.mean(t_stats)), 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "max_p_value": max(p_values),
            "min_p_value": min(p_values),
            "same_sign": all_same_sign,
            "raw_significant": all_raw_significant,
            "raw_odd_robust": all_same_sign and all_raw_significant,
            "period_stats": period_stats,
        })

    # Apply FDR correction across all period tests
    # Collect all p-values for FDR correction
    all_p_values = []
    p_value_mapping = []  # (result_idx, period_idx)

    for r_idx, r in enumerate(results):
        for p_idx, ps in enumerate(r["period_stats"]):
            all_p_values.append(ps["p_value"])
            p_value_mapping.append((r_idx, p_idx))

    if len(all_p_values) > 0:
        all_p_values = np.array(all_p_values)
        fdr_significant = benjamini_hochberg(all_p_values, fdr_alpha)

        # Map back to results
        for i, (r_idx, p_idx) in enumerate(p_value_mapping):
            results[r_idx]["period_stats"][p_idx]["fdr_significant"] = bool(
                fdr_significant[i]
            )

        # Check FDR ODD robustness (same sign + all periods FDR significant)
        for r in results:
            fdr_all_sig = all(ps["fdr_significant"] for ps in r["period_stats"])
            r["fdr_all_significant"] = fdr_all_sig
            r["fdr_odd_robust"] = r["same_sign"] and fdr_all_sig
    else:
        for r in results:
            r["fdr_all_significant"] = False
            r["fdr_odd_robust"] = False

    # Compute Bonferroni threshold for comparison
    n_tests = len(all_p_values)
    bonf_t_threshold = bonferroni_threshold(n_tests, fdr_alpha) if n_tests > 0 else 5.0

    summary = {
        "n_patterns": len(results),
        "n_period_tests": len(all_p_values),
        "raw_t_threshold": raw_t_threshold,
        "bonferroni_t_threshold": round(bonf_t_threshold, 2),
        "fdr_alpha": fdr_alpha,
        "raw_odd_robust": sum(1 for r in results if r["raw_odd_robust"]),
        "fdr_odd_robust": sum(1 for r in results if r["fdr_odd_robust"]),
    }

    return results, summary


def main() -> None:
    """Run FDR-corrected pattern analysis."""
    log_info("Starting FDR-corrected pattern analysis")

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
    total_period_tests = 0

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

        # Add direction and TEMPORAL-SAFE pattern (past only)
        df_pl = df_pl.with_columns(
            pl.when(pl.col("Close") > pl.col("Open"))
            .then(pl.lit("U"))
            .otherwise(pl.lit("D"))
            .alias("direction")
        )

        # Temporal-safe: shift(1) uses past bar, not shift(-1) which uses future
        df_pl = df_pl.with_columns(
            (pl.col("direction").shift(1) + pl.col("direction")).alias("pattern_2bar")
        )

        # Forward returns (this is what we predict)
        df_pl = df_pl.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
        )

        log_info("Features computed", symbol=symbol, n_bars=len(df_pl))

        # Test with FDR correction
        results, summary = test_odd_robustness_with_fdr(
            df_pl,
            "pattern_2bar",
            "fwd_ret_1",
            min_samples=100,
            raw_t_threshold=5.0,
            fdr_alpha=0.05,
        )

        for r in results:
            r["symbol"] = symbol

        all_results.extend(results)
        total_period_tests += summary["n_period_tests"]

        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            raw_odd=summary["raw_odd_robust"],
            fdr_odd=summary["fdr_odd_robust"],
        )

    # Compute global Bonferroni threshold
    if total_period_tests == 0:
        log_info("No data loaded - ensure range bar data is available")
        return
    global_bonf_t = bonferroni_threshold(total_period_tests, 0.05)

    # Print results
    print("\n" + "=" * 130)
    print("FDR-CORRECTED PATTERN ANALYSIS (TEMPORAL-SAFE)")
    print("=" * 130)

    print("\n" + "-" * 130)
    print("PATTERN RESULTS BY SYMBOL")
    print("-" * 130)
    print(
        f"{'Symbol':<10} {'Pattern':<10} {'N Periods':>10} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'Raw ODD':>8} {'FDR ODD':>8}"
    )
    print("-" * 130)

    for r in sorted(all_results, key=lambda x: (x["symbol"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['pattern'] or 'N/A':<10} {r['n_periods']:>10} {r['total_n']:>12,} "
            f"{r['mean_t_stat']:>10.2f} {r['min_abs_t']:>10.2f} "
            f"{'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['raw_odd_robust'] else 'no':>8} "
            f"{'YES' if r['fdr_odd_robust'] else 'no':>8}"
        )

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    raw_odd = sum(1 for r in all_results if r["raw_odd_robust"])
    fdr_odd = sum(1 for r in all_results if r["fdr_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal patterns tested: {len(all_results)}")
    print(f"Total period tests: {total_period_tests}")
    print("\nRaw t-statistic threshold: 5.0")
    print(f"Bonferroni-corrected threshold: {global_bonf_t:.2f}")
    print("FDR alpha: 0.05")

    print(f"\nPatterns with same sign: {same_sign} ({100*same_sign/max(len(all_results), 1):.1f}%)")
    print(f"Raw ODD robust (|t| >= 5): {raw_odd} ({100*raw_odd/max(len(all_results), 1):.1f}%)")
    print(f"FDR ODD robust (BH correction): {fdr_odd} ({100*fdr_odd/max(len(all_results), 1):.1f}%)")

    # Cross-symbol analysis
    print("\n" + "-" * 130)
    print("CROSS-SYMBOL UNIVERSAL PATTERNS")
    print("-" * 130)

    pattern_by_symbol_raw = {}
    pattern_by_symbol_fdr = {}

    for r in all_results:
        p = r["pattern"]
        if r["raw_odd_robust"]:
            if p not in pattern_by_symbol_raw:
                pattern_by_symbol_raw[p] = []
            pattern_by_symbol_raw[p].append(r["symbol"])
        if r["fdr_odd_robust"]:
            if p not in pattern_by_symbol_fdr:
                pattern_by_symbol_fdr[p] = []
            pattern_by_symbol_fdr[p].append(r["symbol"])

    universal_raw = [p for p, syms in pattern_by_symbol_raw.items() if len(syms) >= 4]
    universal_fdr = [p for p, syms in pattern_by_symbol_fdr.items() if len(syms) >= 4]

    print(f"\nUniversal patterns (Raw ODD on 4 symbols): {len(universal_raw)}")
    for p in universal_raw:
        print(f"  - {p}")

    print(f"\nUniversal patterns (FDR ODD on 4 symbols): {len(universal_fdr)}")
    for p in universal_fdr:
        print(f"  - {p}")

    # Key findings
    print("\n" + "=" * 130)
    print("KEY FINDINGS")
    print("=" * 130)

    if fdr_odd == 0:
        print(f"\n✗ Raw threshold found {raw_odd} ODD robust patterns")
        print("✗ FDR correction eliminates ALL patterns (0 survive)")
        print(f"  → The {raw_odd} 'significant' patterns are false discoveries")
        print(f"  → Expected false discoveries at p=0.0000006 with {total_period_tests} tests:")
        print(f"     ~{total_period_tests * 0.0000006:.3f} by chance")
    elif fdr_odd < raw_odd:
        reduction = 100 * (raw_odd - fdr_odd) / raw_odd
        print(f"\n⚠ Raw threshold: {raw_odd} ODD robust patterns")
        print(f"⚠ FDR correction: {fdr_odd} ODD robust patterns")
        print(f"  → {reduction:.0f}% reduction after multiple testing correction")
    else:
        print(f"\n✓ {fdr_odd} patterns survive FDR correction")
        print("  → These are statistically valid patterns")

    log_info(
        "Analysis complete",
        total_patterns=len(all_results),
        total_period_tests=total_period_tests,
        raw_odd=raw_odd,
        fdr_odd=fdr_odd,
        universal_raw=len(universal_raw),
        universal_fdr=len(universal_fdr),
    )


if __name__ == "__main__":
    main()
