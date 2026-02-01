#!/usr/bin/env python3
"""TDA Regime Hurst Analysis - Hurst Exponent Per TDA-Detected Regime.

Analyzes whether the Hurst exponent (H ~ 0.79) finding is stable across
TDA-detected regimes or varies significantly.

Hypothesis:
- If H is stable across regimes: Long memory is a global market property
- If H varies across regimes: Long memory is itself regime-dependent

This is an adversarial audit of the MRH Framework findings.

Issue #54, #56: Volatility Regime Filter / TDA Structural Break Detection
"""

import json
import sys
from datetime import datetime, timezone
from itertools import pairwise
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
        return np.nan  # Insufficient data

    # Generate window sizes (powers of 2)
    max_k = int(np.log2(n // min_window))
    if max_k < 2:
        return np.nan

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
        return np.nan

    slope, _ = np.polyfit(log_n_values, log_rs_values, 1)
    return float(np.clip(slope, 0.0, 1.0))


def compute_dfa_hurst(returns: np.ndarray, min_window: int = 10) -> float:
    """Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).

    More robust than R/S for non-stationary data.
    """
    n = len(returns)
    if n < min_window * 10:
        return np.nan

    y = np.cumsum(returns - np.mean(returns))

    max_k = int(np.log2(n // min_window))
    if max_k < 2:
        return np.nan

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

            x = np.arange(window)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend

            rms = np.sqrt(np.mean(detrended**2))
            f_list.append(rms)

        if f_list:
            avg_f = np.mean(f_list)
            if avg_f > 0:
                log_f_values.append(np.log(avg_f))
                log_n_values.append(np.log(window))

    if len(log_f_values) < 2:
        return np.nan

    slope, _ = np.polyfit(log_n_values, log_f_values, 1)
    return float(np.clip(slope, 0.0, 1.5))


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


def assign_tda_regimes(n: int, break_indices: list[int]) -> list[str]:
    """Assign regime labels based on TDA break indices."""
    if len(break_indices) == 0:
        return ["stable"] * n

    breaks = sorted(break_indices)
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

    return regimes


def bootstrap_hurst_ci(
    returns: np.ndarray,
    n_bootstrap: int = 100,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for Hurst exponent.

    Returns:
        Tuple of (H, lower_bound, upper_bound)
    """
    if len(returns) < 100:
        return (np.nan, np.nan, np.nan)

    h_samples = []
    n = len(returns)
    rng = np.random.default_rng()

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample = returns[indices]
        h = compute_rs_hurst(sample)
        if not np.isnan(h):
            h_samples.append(h)

    if len(h_samples) < 10:
        return (np.nan, np.nan, np.nan)

    h_mean = np.mean(h_samples)
    alpha = (1 - ci) / 2
    lower = np.percentile(h_samples, alpha * 100)
    upper = np.percentile(h_samples, (1 - alpha) * 100)

    return (h_mean, lower, upper)


def main() -> None:
    """Run TDA regime Hurst analysis."""
    log_info("Starting TDA regime Hurst analysis")

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
            ouroboros="year",
            use_cache=True,
            fetch_if_missing=False,
        )

        if df is None or len(df) == 0:
            log_info("No data", symbol=symbol)
            continue

        df_pl = pl.from_pandas(df.reset_index())

        # Compute log returns
        log_returns = (
            df_pl.filter(pl.col("Close").is_not_null())
            .select((pl.col("Close") / pl.col("Close").shift(1)).log())
            .drop_nulls()
            .to_numpy()
            .flatten()
        )

        log_info("Computing TDA breaks", symbol=symbol, n_returns=len(log_returns))

        # Subsample for TDA computation
        subsample_factor = max(1, len(log_returns) // 10000)
        subsampled = log_returns[::subsample_factor]

        # Detect TDA breaks
        break_indices_subsampled = detect_tda_breaks_fast(
            subsampled,
            window_size=100,
            step_size=50,
            threshold_pct=95,
        )

        # Scale back to original indices
        break_indices = [idx * subsample_factor for idx in break_indices_subsampled]

        log_info(
            "TDA breaks detected",
            symbol=symbol,
            n_breaks=len(break_indices),
        )

        if len(break_indices) == 0:
            log_info("No TDA breaks, computing global Hurst", symbol=symbol)
            h = compute_rs_hurst(log_returns)
            all_results.append({
                "symbol": symbol,
                "regime": "global",
                "n_samples": len(log_returns),
                "hurst_rs": round(h, 4) if not np.isnan(h) else None,
            })
            continue

        # Assign regimes
        regimes = assign_tda_regimes(len(log_returns), break_indices)

        # Compute Hurst per regime
        regime_returns = {}
        for idx, regime in enumerate(regimes):
            if regime not in regime_returns:
                regime_returns[regime] = []
            regime_returns[regime].append(log_returns[idx])

        regime_hurst = []
        for regime, returns in sorted(regime_returns.items()):
            returns_arr = np.array(returns)

            if len(returns_arr) < 500:
                continue

            h_rs = compute_rs_hurst(returns_arr)
            h_dfa = compute_dfa_hurst(returns_arr)

            # Bootstrap CI for R/S Hurst
            _h_mean, h_lower, h_upper = bootstrap_hurst_ci(returns_arr, n_bootstrap=50)

            result = {
                "symbol": symbol,
                "regime": regime,
                "n_samples": len(returns_arr),
                "hurst_rs": round(h_rs, 4) if not np.isnan(h_rs) else None,
                "hurst_dfa": round(h_dfa, 4) if not np.isnan(h_dfa) else None,
                "h_ci_lower": round(h_lower, 4) if not np.isnan(h_lower) else None,
                "h_ci_upper": round(h_upper, 4) if not np.isnan(h_upper) else None,
            }
            all_results.append(result)
            regime_hurst.append(result)

        # Compare first vs last regime Hurst
        if len(regime_hurst) >= 2:
            first_regime = regime_hurst[0]
            last_regime = regime_hurst[-1]

            if (
                first_regime["hurst_rs"] is not None
                and last_regime["hurst_rs"] is not None
            ):
                # Welch's t-test for bootstrap samples would be more rigorous
                # Here we use the point estimates
                diff = last_regime["hurst_rs"] - first_regime["hurst_rs"]

                all_comparisons.append({
                    "symbol": symbol,
                    "regime1": first_regime["regime"],
                    "regime2": last_regime["regime"],
                    "h1": first_regime["hurst_rs"],
                    "h2": last_regime["hurst_rs"],
                    "diff": round(diff, 4),
                    "significant": abs(diff) > 0.05,  # Heuristic threshold
                })

    # Print results
    print("\n" + "=" * 120)
    print("TDA REGIME HURST ANALYSIS")
    print("=" * 120)
    print("\nHypothesis: Does Hurst exponent vary across TDA-detected regimes?")
    print("If H varies significantly, long memory is itself regime-dependent.")

    print("\n" + "-" * 120)
    print("HURST EXPONENT BY TDA REGIME")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Regime':<18} {'N':>12} "
        f"{'H (R/S)':>10} {'H (DFA)':>10} {'95% CI':>20}"
    )
    print("-" * 120)

    for r in all_results:
        ci_str = ""
        if r.get("h_ci_lower") is not None and r.get("h_ci_upper") is not None:
            ci_str = f"[{r['h_ci_lower']:.3f}, {r['h_ci_upper']:.3f}]"

        h_rs = f"{r['hurst_rs']:.4f}" if r.get("hurst_rs") is not None else "N/A"
        h_dfa = f"{r['hurst_dfa']:.4f}" if r.get("hurst_dfa") is not None else "N/A"

        print(
            f"{r['symbol']:<10} {r['regime']:<18} {r['n_samples']:>12,} "
            f"{h_rs:>10} {h_dfa:>10} {ci_str:>20}"
        )

    print("\n" + "-" * 120)
    print("CROSS-REGIME HURST COMPARISON")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Regime1':<16} {'Regime2':<16} "
        f"{'H1':>8} {'H2':>8} {'Diff':>8} {'Sig?':>6}"
    )
    print("-" * 120)

    for c in all_comparisons:
        sig = "YES" if c["significant"] else "no"
        print(
            f"{c['symbol']:<10} {c['regime1']:<16} {c['regime2']:<16} "
            f"{c['h1']:>8.4f} {c['h2']:>8.4f} {c['diff']:>+8.4f} {sig:>6}"
        )

    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)

    valid_h = [
        r["hurst_rs"]
        for r in all_results
        if r.get("hurst_rs") is not None and r["regime"] != "global"
    ]

    if valid_h:
        avg_h = np.mean(valid_h)
        std_h = np.std(valid_h)
        min_h = np.min(valid_h)
        max_h = np.max(valid_h)

        print("\nAcross all TDA regimes:")
        print(f"  Mean Hurst:   {avg_h:.4f}")
        print(f"  Std Dev:      {std_h:.4f}")
        print(f"  Range:        [{min_h:.4f}, {max_h:.4f}]")
        print(f"  N regimes:    {len(valid_h)}")

        # Compare to global finding H ~ 0.79
        global_h = 0.79
        deviation = abs(avg_h - global_h)

        print(f"\nComparison to global H = {global_h}:")
        print(f"  Deviation:    {deviation:.4f}")

        if std_h < 0.05:
            print(f"\n  Conclusion: Hurst is STABLE across regimes (std = {std_h:.4f})")
            print("  Long memory is a global market property, not regime-dependent.")
        else:
            print(f"\n  Conclusion: Hurst VARIES across regimes (std = {std_h:.4f})")
            print("  Long memory may itself be regime-dependent.")

        # Test for significant variation
        if len(valid_h) >= 3:
            _, p_value = stats.normaltest(valid_h)
            print(f"\n  D'Agostino-Pearson normality test p-value: {p_value:.4f}")

    # Key findings
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    sig_diffs = [c for c in all_comparisons if c["significant"]]
    print(f"\nSignificant Hurst changes across regimes: {len(sig_diffs)}/{len(all_comparisons)}")

    if sig_diffs:
        print("\nRegimes with significant Hurst changes:")
        for c in sig_diffs:
            direction = "increased" if c["diff"] > 0 else "decreased"
            print(
                f"  - {c['symbol']}: H {direction} from {c['h1']:.4f} to {c['h2']:.4f} "
                f"({c['diff']:+.4f})"
            )

    print("\n" + "=" * 120)
    print("IMPLICATIONS FOR MRH FRAMEWORK")
    print("=" * 120)
    print("""
If Hurst is stable:
  - T_eff = T^(2(1-H)) adjustment applies globally
  - MRH Framework's H ~ 0.79 finding is robust

If Hurst varies:
  - T_eff adjustment should be regime-specific
  - Pattern validation may need regime-conditional Hurst
  - This complicates the PSR/MinTRL calculations
""")

    log_info(
        "Analysis complete",
        n_regimes=len(all_results),
        n_significant_changes=len(sig_diffs),
    )


if __name__ == "__main__":
    main()
