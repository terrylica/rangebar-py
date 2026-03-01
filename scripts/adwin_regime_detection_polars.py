#!/usr/bin/env python3
"""ADWIN Regime Detection for Range Bar Patterns.

Implements Adaptive Windowing (ADWIN) for parameter-free regime shift detection.
Per MRH Framework, ADWIN provides the "supply side" (T_available) of the
stationarity gap inequality: Supply(ADWIN) >= Demand(MinTRL).

Key features:
- Parameter-free detection of distributional changes
- Feeds squared returns (for volatility regime detection)
- Window length is a meta-signal (expanding = stable, shrinking = shift)
- Compare to fixed-window RV regime approach

Reference: docs/research/external/time-to-convergence-stationarity-gap.md

Issue #54: Volatility Regime Filter for ODD Robust Patterns
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from river import drift

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


def compute_adwin_regimes(
    df: pl.DataFrame,
    delta: float = 0.002,
) -> pl.DataFrame:
    """Compute ADWIN-based volatility regimes.

    Args:
        df: Range bar DataFrame with Close prices
        delta: ADWIN confidence parameter (default 0.002)
              Lower delta = more sensitive to drift

    Returns:
        DataFrame with ADWIN regime columns added
    """
    # Compute log returns
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
    )

    # Initialize ADWIN detector for volatility (squared returns)
    adwin = drift.ADWIN(delta=delta)

    # Track window sizes and regime changes
    window_sizes = []
    regime_ids = []
    current_regime = 0

    # Process each bar
    log_returns = df["log_return"].to_list()

    for log_ret in log_returns:
        import math
        if log_ret is None or (isinstance(log_ret, float) and math.isnan(log_ret)):
            window_sizes.append(0)
            regime_ids.append(current_regime)
            continue

        # Feed squared return (volatility proxy)
        squared_return = log_ret**2
        drift_detected = adwin.update(squared_return)

        if drift_detected:
            current_regime += 1

        window_sizes.append(int(adwin.width))
        regime_ids.append(current_regime)

    # Add columns
    df = df.with_columns(
        [
            pl.Series("adwin_window", window_sizes),
            pl.Series("adwin_regime_id", regime_ids),
        ]
    )

    return df


def analyze_adwin_vs_fixed_window(
    df: pl.DataFrame,
    fixed_window: int = 20,
) -> dict:
    """Compare ADWIN adaptive window to fixed window approach.

    Args:
        df: DataFrame with ADWIN and fixed-window RV regimes
        fixed_window: Size of fixed window for comparison

    Returns:
        Comparison metrics
    """
    # Compute fixed-window RV (standard deviation of log returns)
    df = df.with_columns(
        pl.col("log_return")
        .rolling_std(window_size=fixed_window)
        .alias("fixed_rv")
    )

    # Compute RV percentiles for fixed window
    rv_values = df["fixed_rv"].drop_nulls().to_numpy()
    if len(rv_values) < 100:
        return {"error": "Not enough data for fixed-window analysis"}

    p25 = float(rv_values[int(len(rv_values) * 0.25)])
    p75 = float(rv_values[int(len(rv_values) * 0.75)])

    # Classify fixed-window regimes
    df = df.with_columns(
        pl.when(pl.col("fixed_rv") < p25)
        .then(pl.lit("quiet"))
        .when(pl.col("fixed_rv") > p75)
        .then(pl.lit("volatile"))
        .otherwise(pl.lit("active"))
        .alias("fixed_rv_regime")
    )

    # Compare regime stability
    adwin_regime_changes = df["adwin_regime_id"].n_unique() - 1
    fixed_regime_changes = (
        (df["fixed_rv_regime"] != df["fixed_rv_regime"].shift(1))
        .sum()
    )

    # Average ADWIN window size
    avg_adwin_window = df["adwin_window"].mean()
    min_adwin_window = df["adwin_window"].min()
    max_adwin_window = df["adwin_window"].max()

    return {
        "adwin_regime_changes": int(adwin_regime_changes),
        "fixed_regime_changes": int(fixed_regime_changes),
        "avg_adwin_window": float(avg_adwin_window) if avg_adwin_window else 0,
        "min_adwin_window": int(min_adwin_window) if min_adwin_window else 0,
        "max_adwin_window": int(max_adwin_window) if max_adwin_window else 0,
        "fixed_window_size": fixed_window,
    }


def compute_adwin_pattern_returns(
    df: pl.DataFrame,
    pattern_col: str = "pattern_3bar",
) -> pl.DataFrame:
    """Compute returns by pattern within ADWIN regimes.

    Args:
        df: DataFrame with patterns and ADWIN regimes
        pattern_col: Column containing pattern labels

    Returns:
        Aggregated pattern returns by ADWIN regime state
    """
    # Add pattern column if not present
    if pattern_col not in df.columns:
        df = df.with_columns(
            (
                pl.when(pl.col("Close") > pl.col("Open"))
                .then(pl.lit("U"))
                .otherwise(pl.lit("D"))
            ).alias("direction")
        )

        # Create 3-bar pattern
        df = df.with_columns(
            (
                pl.col("direction").shift(2)
                + pl.col("direction").shift(1)
                + pl.col("direction")
            ).alias(pattern_col)
        )

    # Forward return
    df = df.with_columns(
        (pl.col("Close").shift(-1) / pl.col("Close") - 1).alias("forward_return")
    )

    # Classify ADWIN regime state based on window dynamics
    # Expanding window = stable, shrinking = transitional
    df = df.with_columns(
        (pl.col("adwin_window") - pl.col("adwin_window").shift(1)).alias(
            "window_delta"
        )
    )

    df = df.with_columns(
        pl.when(pl.col("window_delta") > 0)
        .then(pl.lit("expanding"))
        .when(pl.col("window_delta") < 0)
        .then(pl.lit("shrinking"))
        .otherwise(pl.lit("stable"))
        .alias("adwin_state")
    )

    # Aggregate by pattern and ADWIN state
    agg = (
        df.filter(pl.col(pattern_col).is_not_null())
        .filter(pl.col("forward_return").is_not_null())
        .filter(pl.col("adwin_state").is_not_null())
        .group_by([pattern_col, "adwin_state"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("forward_return").mean().alias("mean_return"),
                pl.col("forward_return").std().alias("std_return"),
                pl.col("adwin_window").mean().alias("avg_window"),
            ]
        )
        .sort([pattern_col, "adwin_state"])
    )

    return agg


def main() -> None:
    """Run ADWIN regime detection analysis."""
    log_info("Starting ADWIN regime detection analysis")

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100  # 100 dbps
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

        # Get range bars
        df = get_range_bars(
            symbol,
            start_date=start_dates[symbol],
            end_date=end_date,
            threshold_decimal_bps=threshold,
            ouroboros="month",
            use_cache=True,
            fetch_if_missing=False,
        )

        if df is None or len(df) == 0:
            log_info("No data for symbol", symbol=symbol)
            continue

        # Convert to Polars
        df_pl = pl.from_pandas(df.reset_index())

        # Compute ADWIN regimes
        # Try different delta values for sensitivity analysis
        # Lower delta = more sensitive to drift
        df_pl = compute_adwin_regimes(df_pl, delta=0.0001)

        # Compare ADWIN vs fixed window
        comparison = analyze_adwin_vs_fixed_window(df_pl, fixed_window=20)
        comparison["symbol"] = symbol
        comparison["n_bars"] = len(df_pl)

        log_info("ADWIN vs fixed window comparison", **comparison)

        # Compute pattern returns by ADWIN state
        pattern_returns = compute_adwin_pattern_returns(df_pl)

        # Store results
        for row in pattern_returns.to_dicts():
            row["symbol"] = symbol
            all_results.append(row)

    # Aggregate across symbols
    if not all_results:
        log_info("No results to aggregate")
        return

    results_df = pl.DataFrame(all_results)

    # Aggregate by pattern and ADWIN state
    final_agg = (
        results_df.group_by(["pattern_3bar", "adwin_state"])
        .agg(
            [
                pl.col("n").sum().alias("total_n"),
                (pl.col("mean_return") * pl.col("n")).sum().alias("weighted_return_sum"),
                pl.col("n").sum().alias("weight_sum"),
            ]
        )
        .with_columns(
            (pl.col("weighted_return_sum") / pl.col("weight_sum") * 10000).alias(
                "mean_return_bps"
            )
        )
        .select(["pattern_3bar", "adwin_state", "total_n", "mean_return_bps"])
        .sort(["adwin_state", "mean_return_bps"], descending=[False, True])
    )

    # Print results
    print("\n" + "=" * 100)
    print("ADWIN REGIME DETECTION: PATTERN RETURNS BY REGIME STATE")
    print("=" * 100)
    print("\n" + "-" * 100)
    print(f"{'Pattern':<15} {'ADWIN State':<15} {'N':>12} {'Mean (bps)':>12}")
    print("-" * 100)

    for row in final_agg.to_dicts():
        print(
            f"{row['pattern_3bar']:<15} "
            f"{row['adwin_state']:<15} "
            f"{row['total_n']:>12,} "
            f"{row['mean_return_bps']:>12.2f}"
        )

    print("-" * 100)

    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Best patterns by ADWIN state
    for state in ["expanding", "stable", "shrinking"]:
        state_df = final_agg.filter(pl.col("adwin_state") == state)
        if len(state_df) > 0:
            best = state_df.sort("mean_return_bps", descending=True).head(3)
            print(f"\nTop patterns in {state.upper()} regime:")
            for row in best.to_dicts():
                print(
                    f"  {row['pattern_3bar']}: {row['mean_return_bps']:.2f} bps "
                    f"(N={row['total_n']:,})"
                )

    log_info("Analysis complete")


if __name__ == "__main__":
    main()
