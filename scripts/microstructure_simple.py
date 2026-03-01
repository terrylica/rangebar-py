#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Simple Microstructure Pattern Analysis - Memory Efficient.

Tests if microstructure features (OFI, aggression ratio, etc.) have predictive power
for forward returns. Uses simple tercile discretization.

Issue #52, #56: Post-Audit Microstructure Research
"""

import json
import sys
from datetime import datetime, timezone
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


def analyze_single_feature(
    df: pl.DataFrame,
    feature: str,
    return_col: str = "fwd_ret_1",
    min_samples: int = 100,
    min_t_stat: float = 3.0,  # Reduced threshold for exploratory
) -> list[dict]:
    """Analyze predictive power of a single microstructure feature."""
    # Add period column
    df = df.with_columns(
        (
            pl.col("timestamp").dt.year().cast(pl.Utf8)
            + "-Q"
            + ((pl.col("timestamp").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    # Create terciles
    q33 = df[feature].quantile(0.33)
    q67 = df[feature].quantile(0.67)

    df = df.with_columns(
        pl.when(pl.col(feature) <= q33)
        .then(pl.lit("LOW"))
        .when(pl.col(feature) <= q67)
        .then(pl.lit("MID"))
        .otherwise(pl.lit("HIGH"))
        .alias("tercile")
    )

    results = []

    for tercile in ["LOW", "MID", "HIGH"]:
        tercile_df = df.filter(pl.col("tercile") == tercile)
        periods = tercile_df["period"].drop_nulls().unique().to_list()

        if len(periods) < 4:
            continue

        period_stats = []
        for period in sorted(periods):
            period_df = tercile_df.filter(pl.col("period") == period)
            returns = period_df[return_col].drop_nulls().to_numpy()

            if len(returns) < min_samples:
                continue

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))
            se = std_ret / np.sqrt(len(returns))
            t_stat = mean_ret / se if se > 0 else 0

            period_stats.append({
                "period": period,
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

        results.append({
            "feature": feature,
            "tercile": tercile,
            "n_periods": len(period_stats),
            "total_n": sum(p["n"] for p in period_stats),
            "mean_t_stat": round(float(np.mean(t_stats)), 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if np.mean(t_stats) > 0 else "-",
            "mean_bps": round(
                float(np.mean([p["mean_bps"] for p in period_stats])), 2
            ),
        })

    return results


def main() -> None:
    """Run simple microstructure analysis."""
    log_info("Starting simple microstructure pattern analysis")

    # Process one symbol at a time to reduce memory
    configs = [
        ("BTCUSDT", "2024-01-01"),
        ("ETHUSDT", "2024-01-01"),
        ("SOLUSDT", "2024-01-01"),
        ("BNBUSDT", "2024-01-01"),
    ]
    end_date = "2026-01-31"

    features = [
        "ofi",
        "vwap_close_deviation",
        "trade_intensity",
        "aggression_ratio",
        "turnover_imbalance",
    ]

    all_results = []

    for symbol, start_date in configs:
        log_info("Processing symbol", symbol=symbol)

        df = get_range_bars(
            symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=100,
            ouroboros="month",
            use_cache=True,
            fetch_if_missing=False,
            include_microstructure=True,
        )

        if df is None or len(df) == 0:
            log_info("No data", symbol=symbol)
            continue

        df_pl = pl.from_pandas(df.reset_index())

        # Add forward returns
        df_pl = df_pl.with_columns(
            (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("fwd_ret_1")
        )

        log_info("Data loaded", symbol=symbol, n_bars=len(df_pl))

        # Analyze each feature
        for feat in features:
            if feat not in df_pl.columns:
                continue

            results = analyze_single_feature(df_pl, feat, min_t_stat=3.0)

            for r in results:
                r["symbol"] = symbol
                all_results.append(r)

        # Clear memory
        del df, df_pl

        log_info(
            "Symbol complete",
            symbol=symbol,
            odd_robust=sum(
                1 for r in all_results
                if r["symbol"] == symbol and r["is_odd_robust"]
            ),
        )

    # Print results
    print("\n" + "=" * 120)
    print("MICROSTRUCTURE PATTERN ANALYSIS")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("RESULTS BY FEATURE (|t| >= 3.0 threshold)")
    print("-" * 120)
    print(
        f"{'Feature':<25} {'Tercile':<8} {'Symbol':<10} {'Periods':>8} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Dir':>4} {'Mean bps':>10} {'ODD':>5}"
    )
    print("-" * 120)

    for r in sorted(
        all_results,
        key=lambda x: (x["feature"], x["tercile"], x["symbol"]),
    ):
        print(
            f"{r['feature']:<25} {r['tercile']:<8} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} {r['direction']:>4} "
            f"{r['mean_bps']:>10.2f} {'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal feature-tercile combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Cross-symbol patterns
    if odd_robust > 0:
        print("\n" + "-" * 120)
        print("CROSS-SYMBOL PATTERNS")
        print("-" * 120)

        pattern_by_symbol = {}
        for r in all_results:
            if r["is_odd_robust"]:
                key = (r["feature"], r["tercile"])
                if key not in pattern_by_symbol:
                    pattern_by_symbol[key] = []
                pattern_by_symbol[key].append(r["symbol"])

        universal = [
            (f, t) for (f, t), syms in pattern_by_symbol.items() if len(syms) >= 4
        ]

        print(f"\nUniversal patterns (4 symbols): {len(universal)}")
        for f, t in universal:
            print(f"  - {f}|{t}")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
