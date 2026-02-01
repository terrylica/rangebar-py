#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Cross-asset class correlation analysis for OOD robust patterns.

# Issue #145: Explore multi-asset multi-factor patterns with Forex

Tests if crypto-forex cross-asset correlations reveal OOD robust patterns
that single-asset-class validation missed. Key hypothesis: patterns that
appear in uncorrelated asset classes are more likely to be genuine.

Methodology:
1. Load 5 symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT (crypto) + EURUSD (forex)
2. Compute rolling correlation between crypto and forex range bar returns
3. Test if correlation regimes (high/low correlation) predict returns
4. Validate OOD robustness via quarterly cross-validation
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import polars as pl

# Transaction cost (high VIP tier for crypto, lower for forex)
ROUND_TRIP_COST_DBPS = 15
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_DBPS * 0.001


# =============================================================================
# NDJSON Logging
# =============================================================================


def log_json(level: str, message: str, **kwargs: object) -> None:
    """Log a message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def log_info(message: str, **kwargs: object) -> None:
    """Log INFO level message."""
    log_json("INFO", message, **kwargs)


def log_warn(message: str, **kwargs: object) -> None:
    """Log WARN level message."""
    log_json("WARN", message, **kwargs)


# =============================================================================
# Data Loading
# =============================================================================


def load_range_bars_polars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    """Load range bars from cache as Polars LazyFrame.

    For Binance symbols (BTCUSDT etc), uses get_range_bars().
    For Forex symbols (EURUSD), queries ClickHouse directly.
    """
    from datetime import datetime, timezone

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    if symbol == "EURUSD":
        # Query ClickHouse directly for EURUSD
        from rangebar.clickhouse.cache import RangeBarCache

        cache = RangeBarCache()
        pdf = cache.get_bars_by_timestamp_range(
            symbol=symbol,
            threshold_decimal_bps=threshold_dbps,
            start_ts=start_ts,
            end_ts=end_ts,
            include_microstructure=False,
            ouroboros_mode="none",  # EURUSD was uploaded with ouroboros_mode="none"
        )

        if pdf is None or len(pdf) == 0:
            # Return empty LazyFrame with correct schema
            return pl.DataFrame({
                "timestamp": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "Open": pl.Series([], dtype=pl.Float64),
                "High": pl.Series([], dtype=pl.Float64),
                "Low": pl.Series([], dtype=pl.Float64),
                "Close": pl.Series([], dtype=pl.Float64),
                "Volume": pl.Series([], dtype=pl.Float64),
            }).lazy()

        pdf = pdf.reset_index()
        pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})
        return pl.from_pandas(pdf).lazy()

    # For Binance symbols, use the standard API
    from rangebar import get_range_bars

    pdf = get_range_bars(
        symbol,
        start_date=start_date,
        end_date=end_date,
        threshold_decimal_bps=threshold_dbps,
        use_cache=True,
        fetch_if_missing=False,
    )

    pdf = pdf.reset_index()
    pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})

    return pl.from_pandas(pdf).lazy()


# =============================================================================
# Feature Engineering
# =============================================================================


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction (U for up, D for down)."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_log_returns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add log returns for correlation calculation."""
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_ret")
    )


def add_forward_return(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add forward return (next bar's return)."""
    return df.with_columns(
        ((pl.col("Close").shift(-1) - pl.col("Close")) / pl.col("Close")).alias("fwd_ret")
    )


def add_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


# =============================================================================
# Cross-Asset Alignment
# =============================================================================


def align_bars_by_hour(
    crypto_df: pl.DataFrame,
    forex_df: pl.DataFrame,
) -> pl.DataFrame:
    """Align crypto and forex bars by hourly buckets.

    Range bars are irregular, so we need to resample to a common time grid.
    We use hourly buckets to capture meaningful cross-asset correlation.
    """
    # Add hour bucket to both
    crypto_with_hour = crypto_df.with_columns(
        pl.col("timestamp").dt.truncate("1h").alias("hour_bucket")
    )

    forex_with_hour = forex_df.with_columns(
        pl.col("timestamp").dt.truncate("1h").alias("hour_bucket")
    )

    # Aggregate to hourly: use last close, sum volume, direction of last bar
    crypto_hourly = crypto_with_hour.group_by("hour_bucket").agg([
        pl.col("Close").last().alias("crypto_close"),
        pl.col("log_ret").sum().alias("crypto_ret"),
        pl.col("direction").last().alias("crypto_dir"),
        pl.col("Volume").sum().alias("crypto_volume"),
    ])

    forex_hourly = forex_with_hour.group_by("hour_bucket").agg([
        pl.col("Close").last().alias("forex_close"),
        pl.col("log_ret").sum().alias("forex_ret"),
        pl.col("direction").last().alias("forex_dir"),
        pl.col("Volume").sum().alias("forex_volume"),
    ])

    # Inner join on hour bucket
    aligned = crypto_hourly.join(forex_hourly, on="hour_bucket", how="inner")

    return aligned.sort("hour_bucket")


def compute_rolling_correlation(
    df: pl.DataFrame,
    window: int = 24,  # 24 hours = 1 day
) -> pl.DataFrame:
    """Compute rolling correlation between crypto and forex returns.

    Uses rolling_cov / (rolling_std_x * rolling_std_y) formula.
    """
    # Compute rolling means
    df = df.with_columns([
        pl.col("crypto_ret").rolling_mean(window_size=window, min_samples=window // 2).alias("crypto_mean"),
        pl.col("forex_ret").rolling_mean(window_size=window, min_samples=window // 2).alias("forex_mean"),
        pl.col("crypto_ret").rolling_std(window_size=window, min_samples=window // 2).alias("crypto_std"),
        pl.col("forex_ret").rolling_std(window_size=window, min_samples=window // 2).alias("forex_std"),
    ])

    # Compute deviations
    df = df.with_columns([
        (pl.col("crypto_ret") - pl.col("crypto_mean")).alias("crypto_dev"),
        (pl.col("forex_ret") - pl.col("forex_mean")).alias("forex_dev"),
    ])

    # Compute rolling covariance
    df = df.with_columns(
        (pl.col("crypto_dev") * pl.col("forex_dev"))
        .rolling_mean(window_size=window, min_samples=window // 2)
        .alias("rolling_cov")
    )

    # Compute correlation
    df = df.with_columns(
        (pl.col("rolling_cov") / (pl.col("crypto_std") * pl.col("forex_std")))
        .alias("rolling_corr")
    )

    # Clean up temporary columns
    return df.drop(["crypto_mean", "forex_mean", "crypto_std", "forex_std", "crypto_dev", "forex_dev", "rolling_cov"])


def classify_correlation_regime(df: pl.DataFrame) -> pl.DataFrame:
    """Classify correlation into regimes: positive, negative, neutral."""
    # Use static thresholds for correlation regimes
    return df.with_columns(
        pl.when(pl.col("rolling_corr") > 0.3)
        .then(pl.lit("positive_corr"))
        .when(pl.col("rolling_corr") < -0.3)
        .then(pl.lit("negative_corr"))
        .otherwise(pl.lit("neutral_corr"))
        .alias("corr_regime")
    )


# =============================================================================
# OOD Robustness Testing
# =============================================================================


def compute_t_stat(returns: list[float]) -> float:
    """Compute t-statistic for returns."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean_val = sum(returns) / n
    variance = sum((r - mean_val) ** 2 for r in returns) / (n - 1) if n > 1 else 0
    std_val = math.sqrt(variance)

    if std_val == 0:
        return 0.0

    return mean_val / (std_val / math.sqrt(n))


def quarterly_ood_test(
    df: pl.DataFrame,
    pattern_col: str,
    return_col: str,
    min_samples: int = 100,
) -> dict:
    """Test if patterns are OOD robust across quarterly periods."""
    # Add quarter column
    df_with_q = df.with_columns(
        (pl.col("hour_bucket").dt.year().cast(pl.Utf8) + "_Q" +
         ((pl.col("hour_bucket").dt.month() - 1) // 3 + 1).cast(pl.Utf8)).alias("quarter")
    )

    quarters = df_with_q.select("quarter").unique().sort("quarter").to_series().to_list()

    # Get unique patterns
    patterns = df_with_q.select(pattern_col).unique().to_series().to_list()
    patterns = [p for p in patterns if p is not None]

    results = {}

    for pattern in patterns:
        pattern_data = df_with_q.filter(pl.col(pattern_col) == pattern)

        quarterly_stats = []
        all_tstats_pass = True
        all_same_sign = True
        first_sign = None

        for quarter in quarters:
            q_data = pattern_data.filter(pl.col("quarter") == quarter)
            returns = q_data.select(return_col).to_series().to_list()

            n = len(returns)
            if n < min_samples:
                continue

            t_stat = compute_t_stat(returns)
            mean_ret = sum(returns) / n if n > 0 else 0

            quarterly_stats.append({
                "quarter": quarter,
                "n": n,
                "mean_ret_bps": mean_ret * 10000,
                "t_stat": t_stat,
            })

            # Check OOD criteria: |t| >= 5
            if abs(t_stat) < 5:
                all_tstats_pass = False

            # Check sign consistency
            if first_sign is None and mean_ret != 0:
                first_sign = 1 if mean_ret > 0 else -1
            elif mean_ret != 0:
                current_sign = 1 if mean_ret > 0 else -1
                if current_sign != first_sign:
                    all_same_sign = False

        if len(quarterly_stats) >= 4:  # Need at least 4 quarters
            results[pattern] = {
                "quarterly_stats": quarterly_stats,
                "ood_robust": all_tstats_pass and all_same_sign,
                "n_quarters": len(quarterly_stats),
            }

    return results


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_crypto_forex_correlation():
    """Main analysis: crypto-forex correlation patterns."""
    log_info("Starting cross-asset correlation analysis")

    # Parameters
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    threshold_dbps = 100  # Primary threshold per encouraged guidance

    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    forex_symbol = "EURUSD"

    # Load EURUSD first
    log_info("Loading EURUSD data", threshold=threshold_dbps)
    forex_df = load_range_bars_polars(forex_symbol, threshold_dbps, start_date, end_date)
    forex_df = add_bar_direction(forex_df)
    forex_df = add_log_returns(forex_df)
    forex_df = forex_df.collect()

    log_info("EURUSD loaded", rows=len(forex_df))

    # Analyze each crypto symbol against EURUSD
    all_results = {}

    for crypto_symbol in crypto_symbols:
        log_info(f"Analyzing {crypto_symbol} vs EURUSD", symbol=crypto_symbol)

        # Load crypto data
        crypto_df = load_range_bars_polars(crypto_symbol, threshold_dbps, start_date, end_date)
        crypto_df = add_bar_direction(crypto_df)
        crypto_df = add_log_returns(crypto_df)
        crypto_df = crypto_df.collect()

        log_info(f"{crypto_symbol} loaded", rows=len(crypto_df))

        # Align by hour
        aligned = align_bars_by_hour(crypto_df, forex_df)
        log_info("Aligned data", aligned_rows=len(aligned))

        if len(aligned) < 1000:
            log_warn("Insufficient aligned data", aligned_rows=len(aligned))
            continue

        # Compute rolling correlation
        aligned = compute_rolling_correlation(aligned, window=24)

        # Classify correlation regime
        aligned = classify_correlation_regime(aligned)

        # Add forward return (next hour's crypto return)
        aligned = aligned.with_columns(
            pl.col("crypto_ret").shift(-1).alias("fwd_crypto_ret")
        )

        # Create pattern: correlation regime + crypto direction
        aligned = aligned.with_columns(
            (pl.col("corr_regime") + "|" + pl.col("crypto_dir")).alias("corr_pattern")
        )

        # Drop nulls
        aligned = aligned.filter(
            pl.col("fwd_crypto_ret").is_not_null() &
            pl.col("corr_pattern").is_not_null() &
            pl.col("rolling_corr").is_not_null()
        )

        # Test OOD robustness
        ood_results = quarterly_ood_test(
            aligned,
            pattern_col="corr_pattern",
            return_col="fwd_crypto_ret",
            min_samples=20,  # Lower threshold for hourly data
        )

        # Log pattern distribution for debugging
        pattern_dist = aligned.group_by("corr_pattern").agg(pl.len().alias("count"))
        log_info(
            "Pattern distribution",
            symbol=crypto_symbol,
            patterns=pattern_dist.to_dicts(),
        )

        # Count OOD robust patterns
        ood_robust_count = sum(1 for r in ood_results.values() if r.get("ood_robust", False))

        log_info(
            f"{crypto_symbol} OOD analysis complete",
            total_patterns=len(ood_results),
            ood_robust=ood_robust_count,
        )

        all_results[crypto_symbol] = ood_results

        # Log correlation regime distribution
        regime_dist = aligned.group_by("corr_regime").agg(pl.len().alias("count"))
        for row in regime_dist.iter_rows(named=True):
            log_info(
                "Correlation regime distribution",
                symbol=crypto_symbol,
                regime=row["corr_regime"],
                count=row["count"],
            )

    # Summary across all symbols
    log_info("=" * 50)
    log_info("Cross-Asset Correlation Analysis Summary")
    log_info("=" * 50)

    total_ood_robust = 0
    universal_patterns = {}

    for symbol, results in all_results.items():
        ood_robust = [p for p, r in results.items() if r.get("ood_robust", False)]
        total_ood_robust += len(ood_robust)

        for pattern in ood_robust:
            if pattern not in universal_patterns:
                universal_patterns[pattern] = []
            universal_patterns[pattern].append(symbol)

        log_info(
            f"{symbol} OOD robust patterns",
            patterns=ood_robust,
            count=len(ood_robust),
        )

    # Find patterns that are OOD robust across multiple symbols
    multi_symbol_patterns = {
        p: symbols for p, symbols in universal_patterns.items()
        if len(symbols) >= 2
    }

    log_info(
        "Multi-symbol OOD robust patterns",
        patterns=list(multi_symbol_patterns.keys()),
        count=len(multi_symbol_patterns),
    )

    # Final verdict
    log_info("=" * 50)
    log_info(
        "FINAL RESULT",
        total_ood_robust=total_ood_robust,
        multi_symbol_patterns=len(multi_symbol_patterns),
    )

    if len(multi_symbol_patterns) > 0:
        log_info("SUCCESS: Found OOD robust cross-asset patterns")
    else:
        log_info("NO OOD robust cross-asset patterns found")

    return all_results


if __name__ == "__main__":
    analyze_crypto_forex_correlation()
