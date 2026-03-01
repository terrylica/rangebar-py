#!/usr/bin/env python3
# allow-simple-sharpe: volatility research uses forward volatility without return tracking
"""TDA Velocity → Forward Volatility Forecast.

Tests if high TDA H1 velocity precedes increased realized volatility.

Hypothesis: TDA detects geometric instability BEFORE statistical volatility rises.
Use case: Risk management - reduce exposure when TDA signals regime change.

Methodology:
1. Compute rolling TDA H1 L2 velocity on range bar returns
2. Classify velocity into terciles (LOW/MID/HIGH)
3. Compute forward realized volatility (std of next N bar returns)
4. Test if HIGH velocity → HIGH forward RV
5. Validate ODD robustness across quarterly periods

Key difference from directional patterns:
- Not predicting direction, just volatility magnitude
- Use case: position sizing, stop placement, hedging

Issue #52, #56: Post-Audit Volatility Research
"""

import json
import sys
from datetime import datetime, timezone
from math import sqrt
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


def compute_tda_velocities(
    returns: np.ndarray,
    window_size: int = 100,
    step_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling TDA H1 L2 velocity.

    Returns:
        indices: Center indices of windows
        velocities: Change in L2 norm between consecutive windows
    """
    n = len(returns)

    # Subsample for tractable computation
    subsample_factor = max(1, n // 5000)
    subsampled = returns[::subsample_factor]

    l2_norms = []
    indices = []

    for start in range(0, len(subsampled) - window_size + 1, step_size):
        end = start + window_size
        window = subsampled[start:end]

        point_cloud = takens_embedding(window, embedding_dim=3, delay=1)

        if len(point_cloud) < 10:
            continue

        l2 = compute_persistence_l2(point_cloud)
        l2_norms.append(l2)
        center_idx = ((start + end) // 2) * subsample_factor
        indices.append(center_idx)

    if len(l2_norms) < 3:
        return np.array([]), np.array([])

    l2_norms = np.array(l2_norms)
    indices = np.array(indices)

    # Velocity = change in L2 norm
    velocities = np.abs(np.diff(l2_norms))
    indices = indices[1:]  # Velocity indices

    return indices, velocities


def analyze_tda_volatility_forecast(
    symbol: str,
    threshold: int = 100,
    forward_window: int = 50,
    min_samples: int = 10,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze if TDA velocity predicts forward realized volatility."""
    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }

    df = get_range_bars(
        symbol,
        start_date=start_dates[symbol],
        end_date="2026-01-31",
        threshold_decimal_bps=threshold,
        ouroboros="month",
        use_cache=True,
        fetch_if_missing=False,
    )

    if df is None or len(df) == 0:
        return []

    # Compute log returns
    close_prices = df["Close"].values
    log_returns = np.log(close_prices[1:] / close_prices[:-1])

    # Compute TDA velocities
    tda_indices, tda_velocities = compute_tda_velocities(log_returns)

    if len(tda_velocities) == 0:
        return []

    # Compute tercile thresholds for TDA velocity
    vel_q33 = np.percentile(tda_velocities, 33)
    vel_q67 = np.percentile(tda_velocities, 67)

    # For each TDA velocity observation, compute forward realized volatility
    results_data = []

    for idx, vel in zip(tda_indices, tda_velocities, strict=False):
        # Classify TDA velocity
        if vel <= vel_q33:
            tda_tercile = "LOW"
        elif vel <= vel_q67:
            tda_tercile = "MID"
        else:
            tda_tercile = "HIGH"

        # Compute forward realized volatility
        if idx + forward_window >= len(log_returns):
            continue

        forward_returns = log_returns[idx : idx + forward_window]
        forward_rv = np.std(forward_returns)

        # Get quarter
        timestamp = df.index[min(idx + 1, len(df) - 1)]
        year = timestamp.year
        quarter = (timestamp.month - 1) // 3 + 1

        results_data.append({
            "tda_tercile": tda_tercile,
            "forward_rv": forward_rv,
            "year": year,
            "quarter": quarter,
        })

    if not results_data:
        return []

    # Convert to Polars for aggregation
    df_results = pl.DataFrame(results_data)

    # Compute tercile thresholds for forward RV
    rv_q33 = df_results["forward_rv"].quantile(0.33)
    rv_q67 = df_results["forward_rv"].quantile(0.67)

    # Add RV tercile
    df_results = df_results.with_columns(
        pl.when(pl.col("forward_rv") <= rv_q33)
        .then(pl.lit("LOW_RV"))
        .when(pl.col("forward_rv") <= rv_q67)
        .then(pl.lit("MID_RV"))
        .otherwise(pl.lit("HIGH_RV"))
        .alias("rv_tercile")
    )

    # Aggregate by TDA tercile and period
    df_results = df_results.with_columns(
        (pl.col("year").cast(str) + "-Q" + pl.col("quarter").cast(str)).alias("period")
    )

    # Compute P(HIGH_RV | tda_tercile) by period
    grouped = (
        df_results.group_by(["tda_tercile", "period"])
        .agg(
            pl.count().alias("n"),
            (pl.col("rv_tercile") == "HIGH_RV").sum().alias("high_rv_count"),
        )
        .with_columns(
            (pl.col("high_rv_count") / pl.col("n")).alias("prob_high_rv")
        )
        .filter(pl.col("n") >= min_samples)
        .sort(["tda_tercile", "period"])
    )

    # Debug: print grouped data summary
    log_info(
        "Grouped data",
        symbol=symbol,
        n_rows=len(grouped),
        periods=grouped["period"].unique().to_list() if len(grouped) > 0 else [],
    )

    # Group by TDA tercile and compute ODD robustness
    results = []
    for tda_tercile in ["LOW", "MID", "HIGH"]:
        tercile_data = grouped.filter(pl.col("tda_tercile") == tda_tercile)

        if len(tercile_data) < 4:
            log_info(
                "Insufficient periods",
                symbol=symbol,
                tercile=tda_tercile,
                n_periods=len(tercile_data),
            )
            continue

        probs = tercile_data["prob_high_rv"].to_numpy()
        total_n = tercile_data["n"].sum()

        mean_prob = float(np.mean(probs))
        null_prob = 1/3

        std_prob = float(np.std(probs, ddof=1)) if len(probs) > 1 else 0

        if std_prob > 0:
            se = std_prob / sqrt(len(probs))
            t_stat = (mean_prob - null_prob) / se
        else:
            t_stat = 0

        signs = [1 if p > null_prob else -1 for p in probs]
        all_same_sign = len(set(signs)) == 1
        is_odd_robust = all_same_sign and abs(t_stat) >= min_t_stat

        results.append({
            "tda_tercile": tda_tercile,
            "n_periods": len(probs),
            "total_n": int(total_n),
            "mean_prob_high_rv": round(mean_prob, 3),
            "min_prob": round(float(min(probs)), 3),
            "max_prob": round(float(max(probs)), 3),
            "t_stat_vs_null": round(t_stat, 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if mean_prob > null_prob else "-",
        })

    return results


def main() -> None:
    """Run TDA volatility forecast analysis."""
    log_info("Starting TDA volatility forecast analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        results = analyze_tda_volatility_forecast(symbol)

        for r in results:
            r["symbol"] = symbol
            all_results.append(r)

        symbol_odd = sum(
            1 for r in all_results
            if r["symbol"] == symbol and r["is_odd_robust"]
        )
        log_info("Symbol complete", symbol=symbol, odd_robust=symbol_odd)

    # Print results
    print("\n" + "=" * 130)
    print("TDA VELOCITY → FORWARD VOLATILITY FORECAST")
    print("Hypothesis: HIGH TDA velocity → HIGH forward realized volatility")
    print("=" * 130)

    print("\n" + "-" * 130)
    print("P(HIGH_RV | TDA_tercile) - Null hypothesis: P = 0.33")
    print("-" * 130)
    print(
        f"{'TDA Tercile':<12} {'Symbol':<10} {'Periods':>8} {'Total N':>12} "
        f"{'P(HIGH_RV)':>12} {'Min P':>8} {'Max P':>8} {'t-stat':>10} {'Sign':>5} {'ODD':>5}"
    )
    print("-" * 130)

    for r in sorted(
        all_results,
        key=lambda x: (x["tda_tercile"], x["symbol"]),
    ):
        print(
            f"{r['tda_tercile']:<12} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['total_n']:>12} {r['mean_prob_high_rv']:>12.3f} "
            f"{r['min_prob']:>8.3f} {r['max_prob']:>8.3f} "
            f"{r['t_stat_vs_null']:>10.2f} {r['direction']:>5} "
            f"{'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal TDA tercile-symbol combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Check hypothesis
    if odd_robust > 0:
        print("\n" + "-" * 130)
        print("HYPOTHESIS CHECK")
        print("-" * 130)

        for tercile in ["LOW", "MID", "HIGH"]:
            tercile_results = [r for r in all_results if r["tda_tercile"] == tercile and r["is_odd_robust"]]
            if tercile_results:
                avg_prob = sum(r["mean_prob_high_rv"] for r in tercile_results) / len(tercile_results)
                direction = tercile_results[0]["direction"]
                print(f"  {tercile}: {len(tercile_results)}/4 symbols ODD robust")
                print(f"    Direction: {direction}, avg P(HIGH_RV) = {avg_prob:.3f}")

        # Check if HIGH TDA → HIGH RV universally
        high_tda_high_rv = [
            r for r in all_results
            if r["tda_tercile"] == "HIGH" and r["is_odd_robust"] and r["direction"] == "+"
        ]
        if len(high_tda_high_rv) >= 4:
            avg = sum(r["mean_prob_high_rv"] for r in high_tda_high_rv) / len(high_tda_high_rv)
            print("\n  ✓ UNIVERSAL: HIGH TDA velocity → HIGH forward RV")
            print(f"    Average P(HIGH_RV | HIGH TDA) = {avg:.3f} vs null 0.333")
        else:
            print(f"\n  ✗ Not universal: Only {len(high_tda_high_rv)}/4 symbols")

        # Check if LOW TDA → LOW RV universally
        low_tda_low_rv = [
            r for r in all_results
            if r["tda_tercile"] == "LOW" and r["is_odd_robust"] and r["direction"] == "-"
        ]
        if len(low_tda_low_rv) >= 4:
            avg = sum(r["mean_prob_high_rv"] for r in low_tda_low_rv) / len(low_tda_low_rv)
            print("\n  ✓ UNIVERSAL: LOW TDA velocity → LOW forward RV")
            print(f"    Average P(HIGH_RV | LOW TDA) = {avg:.3f} vs null 0.333")
    else:
        print("\n  ✗ No ODD robust TDA volatility forecast patterns found")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
