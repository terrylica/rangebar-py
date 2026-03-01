#!/usr/bin/env python3
"""Multi-Factor Multi-Granularity Range Bar Pattern Analysis.

Tests combinations of range bars at different thresholds (50, 100, 200 dbps)
as factors for pattern detection. Uses higher timeframe trend filters with
lower timeframe pattern signals.

Hypothesis: Higher timeframe trend (200 dbps) + lower timeframe pattern
(50/100 dbps) may reveal more stable trading signals than single-factor.

Methodology:
1. Load range bars at 50, 100, 200 dbps for all 4 symbols
2. Compute direction at each granularity
3. Create multi-factor signals (e.g., HTF trend + LTF pattern)
4. Test ODD robustness of multi-factor vs single-factor patterns
5. Compare across all symbols simultaneously

Issue #52: Market Regime Filter for ODD Robust Patterns
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


def load_range_bars_polars(
    symbol: str,
    threshold: int,
    start_date: str,
    end_date: str,
) -> pl.DataFrame | None:
    """Load range bars as Polars DataFrame."""
    df = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold,
        ouroboros="month",
        use_cache=True,
        fetch_if_missing=False,
    )

    if df is None or len(df) == 0:
        return None

    df_pl = pl.from_pandas(df.reset_index())

    # Add direction column
    df_pl = df_pl.with_columns(
        pl.when(pl.col("Close") > pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )

    # Add threshold identifier
    df_pl = df_pl.with_columns(pl.lit(threshold).alias("threshold_dbps"))

    return df_pl


def compute_htf_trend(
    df: pl.DataFrame,
    lookback: int = 3,
) -> pl.DataFrame:
    """Compute higher timeframe trend from recent bar directions.

    Trend is determined by majority of last N bars:
    - "up": majority Up bars
    - "down": majority Down bars
    - "neutral": tied
    """
    # Create shifted direction columns
    shifts = []
    for i in range(lookback):
        shifts.append(
            pl.when(pl.col("direction").shift(i) == "U")
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
            .alias(f"_dir_{i}")
        )

    df = df.with_columns(shifts)

    # Sum to get trend
    dir_cols = [f"_dir_{i}" for i in range(lookback)]
    df = df.with_columns(
        pl.sum_horizontal(dir_cols).alias("_trend_sum")
    )

    df = df.with_columns(
        pl.when(pl.col("_trend_sum") > 0)
        .then(pl.lit("up"))
        .when(pl.col("_trend_sum") < 0)
        .then(pl.lit("down"))
        .otherwise(pl.lit("neutral"))
        .alias("htf_trend")
    )

    # Clean up temp columns
    df = df.drop([*dir_cols, "_trend_sum"])

    return df


def align_granularities(
    df_50: pl.DataFrame,
    df_100: pl.DataFrame,
    df_200: pl.DataFrame,
) -> pl.DataFrame:
    """Align bars from different granularities by timestamp.

    For each 100 dbps bar, find the most recent 200 dbps bar (HTF trend)
    and the next 50 dbps bar (LTF entry).
    """
    # Rename columns to avoid conflicts
    df_50 = df_50.select([
        pl.col("timestamp").alias("ts_50"),
        pl.col("direction").alias("dir_50"),
        pl.col("Close").alias("close_50"),
    ])

    df_100 = df_100.select([
        pl.col("timestamp").alias("ts_100"),
        pl.col("direction").alias("dir_100"),
        pl.col("Open").alias("open_100"),
        pl.col("Close").alias("close_100"),
    ])

    df_200 = df_200.select([
        pl.col("timestamp").alias("ts_200"),
        pl.col("direction").alias("dir_200"),
    ])

    # Sort by timestamp
    df_50 = df_50.sort("ts_50")
    df_100 = df_100.sort("ts_100")
    df_200 = df_200.sort("ts_200")

    # For each 100 dbps bar, find most recent 200 dbps bar
    # Use asof join (join on nearest timestamp <= target)
    aligned = df_100.join_asof(
        df_200,
        left_on="ts_100",
        right_on="ts_200",
        strategy="backward",
    )

    # Add forward return from 100 dbps
    aligned = aligned.with_columns(
        (pl.col("close_100").shift(-1) / pl.col("close_100") - 1).alias("fwd_ret_1")
    )

    return aligned


def compute_multifactor_patterns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute multi-factor patterns combining HTF trend and LTF pattern."""
    # 2-bar pattern at 100 dbps
    df = df.with_columns(
        (pl.col("dir_100").shift(1) + pl.col("dir_100")).alias("pattern_100")
    )

    # Multi-factor signal: HTF trend + LTF pattern
    df = df.with_columns(
        (pl.col("dir_200") + "|" + pl.col("pattern_100")).alias("multifactor_signal")
    )

    return df


def test_odd_robustness(
    df: pl.DataFrame,
    signal_col: str,
    min_samples: int = 100,
    min_t_stat: float = 5.0,
) -> list[dict]:
    """Test ODD robustness of patterns using quarterly periods."""
    # Add period column (year-quarter)
    df = df.with_columns(
        (
            pl.col("ts_100").dt.year().cast(pl.Utf8)
            + "-Q"
            + ((pl.col("ts_100").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("period")
    )

    results = []
    signals = df[signal_col].drop_nulls().unique().to_list()

    for signal in signals:
        if signal is None or not isinstance(signal, str):
            continue

        signal_df = df.filter(pl.col(signal_col) == signal)
        periods = signal_df["period"].drop_nulls().unique().to_list()

        if len(periods) < 4:
            continue

        period_stats = []
        for period in sorted(periods):
            period_df = signal_df.filter(pl.col("period") == period)
            returns = period_df["fwd_ret_1"].drop_nulls().to_numpy()

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

        # Check ODD robustness: same sign and |t| >= threshold across all periods
        t_stats = [p["t_stat"] for p in period_stats]
        signs = [1 if t > 0 else -1 for t in t_stats]

        all_same_sign = len(set(signs)) == 1
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)

        is_odd_robust = all_same_sign and all_significant

        # Aggregate stats
        total_n = sum(p["n"] for p in period_stats)
        mean_t = float(np.mean(t_stats))

        results.append({
            "signal": signal,
            "n_periods": len(period_stats),
            "total_n": total_n,
            "mean_t_stat": round(mean_t, 2),
            "min_t_stat": round(min(abs(t) for t in t_stats), 2),
            "same_sign": all_same_sign,
            "all_significant": all_significant,
            "is_odd_robust": is_odd_robust,
        })

    return results


def main() -> None:
    """Run multi-factor multi-granularity pattern analysis."""
    log_info("Starting multi-factor multi-granularity pattern analysis")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    all_single_results = []
    all_multi_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        # Load all granularities
        df_50 = load_range_bars_polars(symbol, 50, start_dates[symbol], end_date)
        df_100 = load_range_bars_polars(symbol, 100, start_dates[symbol], end_date)
        df_200 = load_range_bars_polars(symbol, 200, start_dates[symbol], end_date)

        if df_50 is None or df_100 is None or df_200 is None:
            log_info("Missing data for granularity", symbol=symbol)
            continue

        log_info(
            "Data loaded",
            symbol=symbol,
            n_50=len(df_50),
            n_100=len(df_100),
            n_200=len(df_200),
        )

        # Compute HTF trend on 200 dbps
        df_200 = compute_htf_trend(df_200, lookback=3)

        # Align granularities
        aligned = align_granularities(df_50, df_100, df_200)

        # Compute multi-factor patterns
        aligned = compute_multifactor_patterns(aligned)

        log_info("Aligned bars", symbol=symbol, n_aligned=len(aligned))

        # Test single-factor ODD robustness (100 dbps patterns only)
        single_results = test_odd_robustness(aligned, "pattern_100")
        for r in single_results:
            r["symbol"] = symbol
            r["factor_type"] = "single"
            all_single_results.append(r)

        # Test multi-factor ODD robustness
        multi_results = test_odd_robustness(aligned, "multifactor_signal")
        for r in multi_results:
            r["symbol"] = symbol
            r["factor_type"] = "multi"
            all_multi_results.append(r)

        log_info(
            "ODD testing complete",
            symbol=symbol,
            single_patterns=len(single_results),
            multi_patterns=len(multi_results),
        )

    # Print results
    print("\n" + "=" * 140)
    print("MULTI-FACTOR MULTI-GRANULARITY PATTERN ANALYSIS")
    print("=" * 140)

    print("\n" + "-" * 140)
    print("SINGLE-FACTOR PATTERNS (100 dbps only)")
    print("-" * 140)
    print(
        f"{'Symbol':<10} {'Pattern':<10} {'N Periods':>10} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'All Sig':>10} {'ODD Robust':>10}"
    )
    print("-" * 140)

    for r in sorted(all_single_results, key=lambda x: (x["symbol"], x["signal"] or "")):
        print(
            f"{r['symbol']:<10} {r['signal'] or 'N/A':<10} {r['n_periods']:>10} {r['total_n']:>12,} "
            f"{r['mean_t_stat']:>10.2f} {r['min_t_stat']:>10.2f} "
            f"{'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['all_significant'] else 'no':>10} "
            f"{'YES' if r['is_odd_robust'] else 'no':>10}"
        )

    print("\n" + "-" * 140)
    print("MULTI-FACTOR PATTERNS (200 dbps trend + 100 dbps pattern)")
    print("-" * 140)
    print(
        f"{'Symbol':<10} {'Signal':<15} {'N Periods':>10} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Same Sign':>10} {'All Sig':>10} {'ODD Robust':>10}"
    )
    print("-" * 140)

    for r in sorted(all_multi_results, key=lambda x: (x["symbol"], x["signal"] or "")):
        signal_str = r["signal"][:15] if r["signal"] else "N/A"
        print(
            f"{r['symbol']:<10} {signal_str:<15} {r['n_periods']:>10} {r['total_n']:>12,} "
            f"{r['mean_t_stat']:>10.2f} {r['min_t_stat']:>10.2f} "
            f"{'YES' if r['same_sign'] else 'no':>10} "
            f"{'YES' if r['all_significant'] else 'no':>10} "
            f"{'YES' if r['is_odd_robust'] else 'no':>10}"
        )

    # Summary comparison
    print("\n" + "=" * 140)
    print("SUMMARY COMPARISON")
    print("=" * 140)

    single_robust = sum(1 for r in all_single_results if r["is_odd_robust"])
    multi_robust = sum(1 for r in all_multi_results if r["is_odd_robust"])

    print(f"\nSingle-factor patterns tested: {len(all_single_results)}")
    print(f"Single-factor ODD robust: {single_robust}")

    print(f"\nMulti-factor patterns tested: {len(all_multi_results)}")
    print(f"Multi-factor ODD robust: {multi_robust}")

    # Cross-symbol universal patterns
    single_by_pattern = {}
    for r in all_single_results:
        if r["is_odd_robust"]:
            p = r["signal"]
            if p not in single_by_pattern:
                single_by_pattern[p] = []
            single_by_pattern[p].append(r["symbol"])

    multi_by_signal = {}
    for r in all_multi_results:
        if r["is_odd_robust"]:
            s = r["signal"]
            if s not in multi_by_signal:
                multi_by_signal[s] = []
            multi_by_signal[s].append(r["symbol"])

    print("\n" + "-" * 140)
    print("UNIVERSAL PATTERNS (ODD robust on all 4 symbols)")
    print("-" * 140)

    universal_single = [p for p, syms in single_by_pattern.items() if len(syms) >= 4]
    universal_multi = [s for s, syms in multi_by_signal.items() if len(syms) >= 4]

    print(f"\nUniversal single-factor: {len(universal_single)}")
    for p in universal_single:
        print(f"  - {p}")

    print(f"\nUniversal multi-factor: {len(universal_multi)}")
    for s in universal_multi:
        print(f"  - {s}")

    # Key findings
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    if multi_robust > single_robust:
        print("\n✓ Multi-factor patterns show MORE ODD robust signals than single-factor")
        print(f"  Improvement: {multi_robust - single_robust} additional robust patterns")
    elif multi_robust < single_robust:
        print("\n✗ Multi-factor patterns show FEWER ODD robust signals than single-factor")
        print(f"  Reduction: {single_robust - multi_robust} fewer robust patterns")
    else:
        print("\n= Multi-factor patterns show SAME number of ODD robust signals")

    if len(universal_multi) > len(universal_single):
        print("\n✓ Multi-factor improves cross-symbol universality")
    elif len(universal_multi) < len(universal_single):
        print("\n✗ Multi-factor reduces cross-symbol universality")

    log_info(
        "Analysis complete",
        single_tested=len(all_single_results),
        single_robust=single_robust,
        multi_tested=len(all_multi_results),
        multi_robust=multi_robust,
        universal_single=len(universal_single),
        universal_multi=len(universal_multi),
    )


if __name__ == "__main__":
    main()
