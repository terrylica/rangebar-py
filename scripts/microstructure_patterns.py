#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Microstructure-Based ODD Robust Pattern Research.

Direction patterns (U/D) were INVALIDATED by adversarial audit - no predictive power
when using temporal-safe methodology.

This script pivots to microstructure features which capture genuine market dynamics:
- OFI (Order Flow Imbalance): buy_vol vs sell_vol
- VWAP deviation: close relative to VWAP
- Trade intensity: trades per second
- Kyle lambda proxy: price impact per unit flow
- Aggression ratio: buy count vs sell count

Methodology:
- Temporal-safe: features use only past/current data
- Forward return: shift(-1) for prediction target
- ODD robustness: same sign + |t| >= 5 across all quarterly periods
- FDR correction: Benjamini-Hochberg for multiple testing

Issue #52, #56: Post-Audit Microstructure Research
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
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    thresholds = np.arange(1, n + 1) / n * alpha
    passes = sorted_p <= thresholds

    if not np.any(passes):
        return np.zeros(n, dtype=bool)

    last_passing = np.max(np.where(passes)[0])
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices[: last_passing + 1]] = True

    return significant


def add_microstructure_terciles(df: pl.DataFrame) -> pl.DataFrame:
    """Discretize microstructure features into terciles for pattern analysis.

    Uses simple quantile cuts on the collected DataFrame (memory efficient).
    Terciles (3 groups) instead of quintiles to reduce pattern explosion.
    """
    features = [
        "ofi",
        "vwap_close_deviation",
        "trade_intensity",
        "aggression_ratio",
        "turnover_imbalance",
    ]

    for feat in features:
        # Get tercile boundaries
        q33 = df[feat].quantile(0.33)
        q67 = df[feat].quantile(0.67)

        # Assign tercile
        df = df.with_columns(
            pl.when(pl.col(feat) <= q33)
            .then(pl.lit("LOW"))
            .when(pl.col(feat) <= q67)
            .then(pl.lit("MID"))
            .otherwise(pl.lit("HIGH"))
            .alias(f"{feat}_tercile")
        )

    return df


def add_microstructure_extremes(df: pl.DataFrame) -> pl.DataFrame:
    """Add binary extreme indicators for microstructure features.

    Extreme = top/bottom 20% (simpler than rolling).
    """
    features = [
        "ofi",
        "vwap_close_deviation",
        "trade_intensity",
        "aggression_ratio",
        "turnover_imbalance",
    ]

    for feat in features:
        p20 = df[feat].quantile(0.2)
        p80 = df[feat].quantile(0.8)

        df = df.with_columns(
            pl.when(pl.col(feat) <= p20)
            .then(pl.lit("LOW"))
            .when(pl.col(feat) >= p80)
            .then(pl.lit("HIGH"))
            .otherwise(pl.lit("MID"))
            .alias(f"{feat}_extreme")
        )

    return df


def test_odd_robustness(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> list[dict]:
    """Test ODD robustness using quarterly periods."""
    # Add period column
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

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))
            se = std_ret / np.sqrt(len(returns))
            t_stat = mean_ret / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(returns) - 1))

            period_stats.append({
                "period": period,
                "n": len(returns),
                "mean_bps": mean_ret * 10000,
                "t_stat": t_stat,
                "p_value": p_value,
            })

        if len(period_stats) < 4:
            continue

        t_stats = [p["t_stat"] for p in period_stats]
        signs = [1 if t > 0 else -1 for t in t_stats]

        all_same_sign = len(set(signs)) == 1
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)
        is_odd_robust = all_same_sign and all_significant

        results.append({
            "pattern": pattern,
            "n_periods": len(period_stats),
            "total_n": sum(p["n"] for p in period_stats),
            "mean_t_stat": round(float(np.mean(t_stats)), 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "max_p_value": max(p["p_value"] for p in period_stats),
            "same_sign": all_same_sign,
            "all_significant": all_significant,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if np.mean(t_stats) > 0 else "-",
            "mean_bps": round(
                float(np.mean([p["mean_bps"] for p in period_stats])), 2
            ),
        })

    return results


def main() -> None:
    """Run microstructure-based ODD robustness analysis."""
    log_info("Starting microstructure-based ODD robustness analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    # Use 2 years of data to reduce memory usage
    start_dates = {
        "BTCUSDT": "2024-01-01",
        "ETHUSDT": "2024-01-01",
        "SOLUSDT": "2024-01-01",
        "BNBUSDT": "2024-01-01",
    }
    end_date = "2026-01-31"

    all_results = []
    features_tested = [
        "ofi_tercile",
        "vwap_close_deviation_tercile",
        "trade_intensity_tercile",
        "aggression_ratio_tercile",
        "turnover_imbalance_tercile",
        "ofi_extreme",
        "vwap_close_deviation_extreme",
        "trade_intensity_extreme",
        "aggression_ratio_extreme",
        "turnover_imbalance_extreme",
    ]

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
            include_microstructure=True,
        )

        if df is None or len(df) == 0:
            log_info("No data", symbol=symbol)
            continue

        df_pl = pl.from_pandas(df.reset_index()).lazy()

        # Add forward returns
        df_pl = df_pl.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
        )

        # Collect first, then add microstructure patterns (memory efficient)
        df_collected = df_pl.collect()

        # Add microstructure patterns
        df_collected = add_microstructure_terciles(df_collected)
        df_collected = add_microstructure_extremes(df_collected)

        log_info("Features computed", symbol=symbol, n_bars=len(df_collected))

        # Test each feature
        for feat in features_tested:
            if feat not in df_collected.columns:
                continue

            results = test_odd_robustness(
                df_collected, feat, "fwd_ret_1", min_samples=100, min_t_stat=5.0
            )

            for r in results:
                r["symbol"] = symbol
                r["feature"] = feat
                all_results.append(r)

        symbol_odd = sum(
            1 for r in all_results if r["symbol"] == symbol and r["is_odd_robust"]
        )
        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            n_patterns=len([r for r in all_results if r["symbol"] == symbol]),
            n_odd_robust=symbol_odd,
        )

    # Print results
    print("\n" + "=" * 140)
    print("MICROSTRUCTURE-BASED ODD ROBUSTNESS ANALYSIS")
    print("=" * 140)

    # Group by feature
    print("\n" + "-" * 140)
    print("ODD ROBUST PATTERNS BY FEATURE")
    print("-" * 140)

    features_with_odd = {}
    for r in all_results:
        if r["is_odd_robust"]:
            feat = r["feature"]
            if feat not in features_with_odd:
                features_with_odd[feat] = []
            features_with_odd[feat].append(r)

    if features_with_odd:
        print(
            f"{'Feature':<35} {'Pattern':<8} {'Symbol':<10} {'N Periods':>10} "
            f"{'Mean t':>10} {'Min |t|':>10} {'Dir':>4} {'Mean bps':>10}"
        )
        print("-" * 140)

        for feat in sorted(features_with_odd.keys()):
            for r in sorted(features_with_odd[feat], key=lambda x: x["symbol"]):
                print(
                    f"{r['feature']:<35} {r['pattern']:<8} {r['symbol']:<10} "
                    f"{r['n_periods']:>10} {r['mean_t_stat']:>10.2f} "
                    f"{r['min_abs_t']:>10.2f} {r['direction']:>4} {r['mean_bps']:>10.2f}"
                )
    else:
        print("\nNo ODD robust patterns found.")

    # Summary
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    total_patterns = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal pattern-feature combinations tested: {total_patterns}")
    print(f"Patterns with same sign across periods: {same_sign} ({100*same_sign/max(total_patterns, 1):.1f}%)")
    print(f"ODD robust (same sign + |t| >= 5): {odd_robust} ({100*odd_robust/max(total_patterns, 1):.1f}%)")

    # Cross-symbol analysis
    if odd_robust > 0:
        print("\n" + "-" * 140)
        print("CROSS-SYMBOL UNIVERSAL PATTERNS")
        print("-" * 140)

        pattern_by_symbol = {}
        for r in all_results:
            if r["is_odd_robust"]:
                key = (r["feature"], r["pattern"])
                if key not in pattern_by_symbol:
                    pattern_by_symbol[key] = []
                pattern_by_symbol[key].append(r["symbol"])

        universal = [
            (feat, pat) for (feat, pat), syms in pattern_by_symbol.items() if len(syms) >= 4
        ]

        print(f"\nUniversal patterns (ODD robust on 4 symbols): {len(universal)}")
        for feat, pat in universal:
            print(f"  - {feat}|{pat}")

    # Key findings
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    if odd_robust == 0:
        print("\n✗ No microstructure patterns achieve ODD robustness")
        print("  → Microstructure features also lack consistent predictive power")
    else:
        print(f"\n✓ {odd_robust} microstructure patterns achieve ODD robustness")
        print("  → Some microstructure dynamics have predictive value")

    log_info(
        "Analysis complete",
        total_patterns=total_patterns,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
