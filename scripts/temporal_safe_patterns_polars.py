#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Temporal-Safe Pattern Analysis - No Data Leakage.

This script implements CORRECTED pattern calculation that does NOT use future information.

PREVIOUS (INCORRECT):
    pattern_2bar[i] = direction[i] + direction[i+1]  # Uses FUTURE bar direction
    pattern_3bar[i] = direction[i] + direction[i+1] + direction[i+2]  # Uses 2 FUTURE bars

CORRECTED (THIS SCRIPT):
    pattern_2bar[i] = direction[i-1] + direction[i]  # Uses PAST bars only
    pattern_3bar[i] = direction[i-2] + direction[i-1] + direction[i]  # Uses PAST bars only

Prediction setup:
- At time i, we observe completed bars up to and including bar i
- Pattern is formed from historical bars (shift(1), shift(2), etc.)
- Forward return predicts bar i+1 (shift(-1))

This ensures:
- All features (pattern) are available at decision time
- Target (forward return) is genuinely future information

Issue #52, #56: Adversarial Audit Data Leakage Fix
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


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction column (U = Up, D = Down)."""
    return df.with_columns(
        pl.when(pl.col("Close") > pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_temporal_safe_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add temporal-safe 2-bar and 3-bar patterns using ONLY past data.

    CORRECT: Pattern at bar i uses only information from bars <= i.
    - 2-bar: direction[i-1] + direction[i]
    - 3-bar: direction[i-2] + direction[i-1] + direction[i]

    The forward return is still shift(-1), predicting bar i+1 from bar i.
    """
    return df.with_columns([
        # 2-bar pattern: previous bar + current bar
        (pl.col("direction").shift(1) + pl.col("direction")).alias("pattern_2bar"),
        # 3-bar pattern: 2-bars-ago + previous + current
        (
            pl.col("direction").shift(2) +
            pl.col("direction").shift(1) +
            pl.col("direction")
        ).alias("pattern_3bar"),
    ])


def add_forward_returns(df: pl.LazyFrame, horizons: list[int] | None = None) -> pl.LazyFrame:
    """Add forward returns at multiple horizons.

    Forward return at bar i = (Close[i+h] / Close[i]) - 1
    This is the return we're predicting using the pattern at bar i.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    for h in horizons:
        df = df.with_columns(
            (pl.col("Close").shift(-h) / pl.col("Close") - 1).alias(f"fwd_ret_{h}")
        )

    return df


def compute_sma_regime(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute SMA-based regime using ONLY past data.

    SMA at bar i uses bars [i-window+1, i] - no future data.
    Regime at bar i is determined by comparing current close to historical SMAs.
    """
    # SMA 20 and 50 using rolling windows (past only)
    df = df.with_columns([
        pl.col("Close").rolling_mean(window_size=20, min_samples=20).alias("sma20"),
        pl.col("Close").rolling_mean(window_size=50, min_samples=50).alias("sma50"),
    ])

    # Regime classification based on SMA crossovers
    df = df.with_columns(
        pl.when(
            (pl.col("Close") > pl.col("sma20")) & (pl.col("sma20") > pl.col("sma50"))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("Close") < pl.col("sma20")) & (pl.col("sma20") < pl.col("sma50"))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("chop"))
        .alias("sma_regime")
    )

    return df


def test_odd_robustness(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> list[dict]:
    """Test ODD robustness using quarterly periods.

    ODD = Out-of-Distribution across time periods.
    Pattern must show same sign and |t| >= threshold in ALL periods.
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

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0

            period_stats.append({
                "period": period,
                "n": len(returns),
                "mean_bps": mean_ret * 10000,
                "t_stat": t_stat,
            })

        if len(period_stats) < 4:
            continue

        # Check ODD robustness
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
            "same_sign": all_same_sign,
            "all_significant": all_significant,
            "is_odd_robust": is_odd_robust,
        })

    return results


def main() -> None:
    """Run temporal-safe pattern analysis."""
    log_info("Starting temporal-safe pattern analysis (no data leakage)")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    all_2bar_results = []
    all_3bar_results = []

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

        df_pl = pl.from_pandas(df.reset_index()).lazy()

        # Add features using temporal-safe methods
        df_pl = add_bar_direction(df_pl)
        df_pl = add_temporal_safe_patterns(df_pl)  # CORRECTED pattern calculation
        df_pl = add_forward_returns(df_pl, horizons=[1, 3, 5, 10])
        df_pl = compute_sma_regime(df_pl)

        # Collect for analysis
        df_collected = df_pl.collect()

        log_info("Features computed", symbol=symbol, n_bars=len(df_collected))

        # Test 2-bar patterns
        results_2bar = test_odd_robustness(
            df_collected, "pattern_2bar", "fwd_ret_1", min_samples=100, min_t_stat=5.0
        )
        for r in results_2bar:
            r["symbol"] = symbol
            r["pattern_type"] = "2bar"
            all_2bar_results.append(r)

        # Test 3-bar patterns
        results_3bar = test_odd_robustness(
            df_collected, "pattern_3bar", "fwd_ret_1", min_samples=100, min_t_stat=5.0
        )
        for r in results_3bar:
            r["symbol"] = symbol
            r["pattern_type"] = "3bar"
            all_3bar_results.append(r)

        log_info(
            "Symbol analysis complete",
            symbol=symbol,
            odd_2bar=sum(1 for r in results_2bar if r["is_odd_robust"]),
            odd_3bar=sum(1 for r in results_3bar if r["is_odd_robust"]),
        )

    # Print results
    print("\n" + "=" * 120)
    print("TEMPORAL-SAFE PATTERN ANALYSIS (NO DATA LEAKAGE)")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("2-BAR PATTERNS (shift(1) + current)")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Pattern':<10} {'N Periods':>10} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'ODD Robust':>10}"
    )
    print("-" * 120)

    for r in sorted(all_2bar_results, key=lambda x: (x["symbol"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['pattern'] or 'N/A':<10} {r['n_periods']:>10} {r['total_n']:>12,} "
            f"{r['mean_t_stat']:>10.2f} {r['min_abs_t']:>10.2f} "
            f"{'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['is_odd_robust'] else 'no':>10}"
        )

    print("\n" + "-" * 120)
    print("3-BAR PATTERNS (shift(2) + shift(1) + current)")
    print("-" * 120)
    print(
        f"{'Symbol':<10} {'Pattern':<10} {'N Periods':>10} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'ODD Robust':>10}"
    )
    print("-" * 120)

    for r in sorted(all_3bar_results, key=lambda x: (x["symbol"], x["pattern"] or "")):
        print(
            f"{r['symbol']:<10} {r['pattern'] or 'N/A':<10} {r['n_periods']:>10} {r['total_n']:>12,} "
            f"{r['mean_t_stat']:>10.2f} {r['min_abs_t']:>10.2f} "
            f"{'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['is_odd_robust'] else 'no':>10}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    odd_2bar = sum(1 for r in all_2bar_results if r["is_odd_robust"])
    odd_3bar = sum(1 for r in all_3bar_results if r["is_odd_robust"])

    print(f"\n2-bar patterns tested: {len(all_2bar_results)}")
    print(f"2-bar ODD robust: {odd_2bar}")

    print(f"\n3-bar patterns tested: {len(all_3bar_results)}")
    print(f"3-bar ODD robust: {odd_3bar}")

    # Cross-symbol universal patterns
    pattern_by_symbol_2bar = {}
    for r in all_2bar_results:
        if r["is_odd_robust"]:
            p = r["pattern"]
            if p not in pattern_by_symbol_2bar:
                pattern_by_symbol_2bar[p] = []
            pattern_by_symbol_2bar[p].append(r["symbol"])

    pattern_by_symbol_3bar = {}
    for r in all_3bar_results:
        if r["is_odd_robust"]:
            p = r["pattern"]
            if p not in pattern_by_symbol_3bar:
                pattern_by_symbol_3bar[p] = []
            pattern_by_symbol_3bar[p].append(r["symbol"])

    print("\n" + "-" * 120)
    print("UNIVERSAL PATTERNS (ODD robust on ALL 4 symbols)")
    print("-" * 120)

    universal_2bar = [p for p, syms in pattern_by_symbol_2bar.items() if len(syms) >= 4]
    universal_3bar = [p for p, syms in pattern_by_symbol_3bar.items() if len(syms) >= 4]

    print(f"\nUniversal 2-bar patterns: {len(universal_2bar)}")
    for p in universal_2bar:
        print(f"  - {p}")

    print(f"\nUniversal 3-bar patterns: {len(universal_3bar)}")
    for p in universal_3bar:
        print(f"  - {p}")

    # Key findings
    print("\n" + "=" * 120)
    print("KEY FINDINGS (TEMPORAL-SAFE)")
    print("=" * 120)

    if odd_2bar == 0 and odd_3bar == 0:
        print("\n✗ NO patterns achieve ODD robustness with temporal-safe calculation")
        print("  → This confirms the prior findings were artifacts of data leakage")
    else:
        print(f"\n✓ {odd_2bar + odd_3bar} patterns achieve ODD robustness with temporal-safe calculation")
        print("  → These are genuine predictive patterns")

    log_info(
        "Analysis complete",
        tested_2bar=len(all_2bar_results),
        odd_2bar=odd_2bar,
        tested_3bar=len(all_3bar_results),
        odd_3bar=odd_3bar,
        universal_2bar=len(universal_2bar),
        universal_3bar=len(universal_3bar),
    )


if __name__ == "__main__":
    main()
