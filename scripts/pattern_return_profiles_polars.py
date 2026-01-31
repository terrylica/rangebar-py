#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Pattern return profile analysis for universal 11 ODD robust patterns (Polars).

Computes expected return profiles for the 11 patterns that persist at 10-bar horizons:
- Mean return at each horizon (1, 3, 5, 10 bars)
- Sharpe-like ratio (return / std)
- Win rate at each horizon
- Sample size and stability

GitHub Issue: #52 (extension)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Universal patterns that persist at 10-bar horizons (from multi-bar analysis)
UNIVERSAL_PATTERNS = [
    ("bear_neutral", "DU"),
    ("bear_neutral", "DD"),
    ("bear_neutral", "UU"),
    ("bear_neutral", "UD"),
    ("bull_neutral", "DU"),
    ("bull_neutral", "DD"),
    ("bull_neutral", "UD"),
    ("chop", "DU"),
    ("chop", "DD"),
    ("chop", "UU"),
    ("chop", "UD"),
]


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


# =============================================================================
# Technical Indicators (Polars Vectorized)
# =============================================================================


def compute_sma(df: pl.LazyFrame, column: str, window: int, alias: str) -> pl.LazyFrame:
    """Compute Simple Moving Average."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=window, min_samples=window).alias(alias)
    )


def compute_rsi(df: pl.LazyFrame, column: str, window: int = 14, alias: str = "rsi14") -> pl.LazyFrame:
    """Compute Relative Strength Index using EWM."""
    return df.with_columns(
        pl.col(column).diff().alias("_delta")
    ).with_columns([
        pl.when(pl.col("_delta") > 0)
        .then(pl.col("_delta"))
        .otherwise(pl.lit(0.0))
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0)
        .then(-pl.col("_delta"))
        .otherwise(pl.lit(0.0))
        .alias("_loss"),
    ]).with_columns([
        pl.col("_gain").ewm_mean(alpha=1.0 / window, min_samples=window).alias("_avg_gain"),
        pl.col("_loss").ewm_mean(alpha=1.0 / window, min_samples=window).alias("_avg_loss"),
    ]).with_columns(
        (pl.col("_avg_gain") / pl.col("_avg_loss")).alias("_rs")
    ).with_columns(
        (100.0 - (100.0 / (1.0 + pl.col("_rs")))).alias(alias)
    ).drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss", "_rs"])


def classify_regimes(df: pl.LazyFrame) -> pl.LazyFrame:
    """Classify market regimes."""
    df = df.with_columns(
        pl.when(
            (pl.col("Close") > pl.col("sma20")) & (pl.col("sma20") > pl.col("sma50"))
        )
        .then(pl.lit("uptrend"))
        .when(
            (pl.col("Close") < pl.col("sma20")) & (pl.col("sma20") < pl.col("sma50"))
        )
        .then(pl.lit("downtrend"))
        .otherwise(pl.lit("consolidation"))
        .alias("sma_regime")
    )

    df = df.with_columns(
        pl.when(pl.col("rsi14") > 70)
        .then(pl.lit("overbought"))
        .when(pl.col("rsi14") < 30)
        .then(pl.lit("oversold"))
        .otherwise(pl.lit("neutral"))
        .alias("rsi_regime")
    )

    df = df.with_columns(
        pl.when(pl.col("sma_regime") == "consolidation")
        .then(pl.lit("chop"))
        .when(
            (pl.col("sma_regime") == "uptrend") & (pl.col("rsi_regime") == "overbought")
        )
        .then(pl.lit("bull_hot"))
        .when(
            (pl.col("sma_regime") == "uptrend") & (pl.col("rsi_regime") == "oversold")
        )
        .then(pl.lit("bull_cold"))
        .when(pl.col("sma_regime") == "uptrend")
        .then(pl.lit("bull_neutral"))
        .when(
            (pl.col("sma_regime") == "downtrend") & (pl.col("rsi_regime") == "overbought")
        )
        .then(pl.lit("bear_hot"))
        .when(
            (pl.col("sma_regime") == "downtrend") & (pl.col("rsi_regime") == "oversold")
        )
        .then(pl.lit("bear_cold"))
        .otherwise(pl.lit("bear_neutral"))
        .alias("regime")
    )

    return df


def add_bar_direction(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add bar direction."""
    return df.with_columns(
        pl.when(pl.col("Close") >= pl.col("Open"))
        .then(pl.lit("U"))
        .otherwise(pl.lit("D"))
        .alias("direction")
    )


def add_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-bar pattern column."""
    return df.with_columns(
        (pl.col("direction") + pl.col("direction").shift(-1)).alias("pattern_2bar")
    )


def add_forward_returns(df: pl.LazyFrame, horizons: list[int]) -> pl.LazyFrame:
    """Add forward returns at multiple horizons."""
    for h in horizons:
        df = df.with_columns(
            (pl.col("Close").shift(-h) / pl.col("Close") - 1).alias(f"fwd_ret_{h}")
        )
    return df


# =============================================================================
# Profile Computation
# =============================================================================


def compute_pattern_profile(
    df: pl.DataFrame,
    regime: str,
    pattern: str,
    horizons: list[int],
) -> dict:
    """Compute return profile for a specific pattern."""
    subset = df.filter(
        (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
    )

    total_count = len(subset)
    if total_count < 100:
        return {"error": "insufficient_samples", "count": total_count}

    profile = {
        "regime": regime,
        "pattern": pattern,
        "total_count": total_count,
        "horizons": {},
    }

    for h in horizons:
        return_col = f"fwd_ret_{h}"
        returns = subset.select(return_col).drop_nulls().to_series()
        n = len(returns)

        if n < 100:
            profile["horizons"][h] = {"error": "insufficient_samples", "count": n}
            continue

        mean_ret = returns.mean()
        std_ret = returns.std()
        win_rate = (returns > 0).sum() / n

        # Simple return/risk ratio (not time-weighted - see allow-simple-sharpe comment)
        if std_ret is not None and std_ret > 0:
            return_risk_ratio = (mean_ret / std_ret) * math.sqrt(1000 / h) if mean_ret is not None else 0
        else:
            return_risk_ratio = 0

        profile["horizons"][h] = {
            "count": n,
            "mean_return_bps": float(mean_ret * 10000) if mean_ret is not None else 0,
            "std_return_bps": float(std_ret * 10000) if std_ret is not None else 0,
            "win_rate": float(win_rate),
            "return_risk_ratio": float(return_risk_ratio) if return_risk_ratio else 0,
        }

    return profile


# =============================================================================
# Data Loading
# =============================================================================


def load_range_bars_polars(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    """Load range bars from cache as Polars LazyFrame."""
    from rangebar import get_range_bars

    log_info("Loading range bars", symbol=symbol, threshold_dbps=threshold_dbps)

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
# Main Analysis
# =============================================================================


def run_profile_analysis(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
    horizons: list[int],
) -> dict:
    """Run pattern profile analysis for a symbol."""
    log_info("Starting profile analysis", symbol=symbol)

    # Load and prepare data
    df = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")
    df = classify_regimes(df)
    df = add_bar_direction(df)
    df = add_patterns(df)
    df = add_forward_returns(df, horizons)

    result_df = df.collect()

    profiles = {}
    for regime, pattern in UNIVERSAL_PATTERNS:
        key = f"{regime}|{pattern}"
        profiles[key] = compute_pattern_profile(result_df, regime, pattern, horizons)

    return {
        "symbol": symbol,
        "threshold_dbps": threshold_dbps,
        "profiles": profiles,
    }


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting pattern return profile analysis (Polars)",
        script="pattern_return_profiles_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizons = [1, 3, 5, 10]

    all_results = {}

    for symbol in symbols:
        try:
            result = run_profile_analysis(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
                horizons=horizons,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Analysis failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL PATTERN PROFILES ===")

    for regime, pattern in UNIVERSAL_PATTERNS:
        key = f"{regime}|{pattern}"

        # Average across symbols
        avg_returns = {h: [] for h in horizons}
        avg_ratios = {h: [] for h in horizons}

        for _symbol, result in all_results.items():
            profile = result.get("profiles", {}).get(key, {})
            for h in horizons:
                h_data = profile.get("horizons", {}).get(h, {})
                if "mean_return_bps" in h_data:
                    avg_returns[h].append(h_data["mean_return_bps"])
                    avg_ratios[h].append(h_data["return_risk_ratio"])

        # Compute averages
        summary = {}
        for h in horizons:
            if avg_returns[h]:
                summary[h] = {
                    "mean_return_bps": sum(avg_returns[h]) / len(avg_returns[h]),
                    "mean_ratio": sum(avg_ratios[h]) / len(avg_ratios[h]),
                }

        log_info(
            f"Pattern {key}",
            pattern=key,
            horizon_1=summary.get(1, {}),
            horizon_3=summary.get(3, {}),
            horizon_5=summary.get(5, {}),
            horizon_10=summary.get(10, {}),
        )

    # Find best patterns by return/risk ratio at 10-bar horizon
    log_info("=== BEST PATTERNS BY 10-BAR RETURN/RISK RATIO ===")

    pattern_ratios = []
    for regime, pattern in UNIVERSAL_PATTERNS:
        key = f"{regime}|{pattern}"
        ratios = []
        returns = []
        for _symbol, result in all_results.items():
            profile = result.get("profiles", {}).get(key, {})
            h_data = profile.get("horizons", {}).get(10, {})
            if "return_risk_ratio" in h_data:
                ratios.append(h_data["return_risk_ratio"])
                returns.append(h_data["mean_return_bps"])

        if ratios:
            pattern_ratios.append({
                "pattern": key,
                "avg_ratio": sum(ratios) / len(ratios),
                "avg_return_bps": sum(returns) / len(returns),
            })

    # Sort by ratio
    pattern_ratios.sort(key=lambda x: abs(x["avg_ratio"]), reverse=True)

    for ps in pattern_ratios:
        log_info(
            "Pattern ranking",
            pattern=ps["pattern"],
            avg_ratio=round(ps["avg_ratio"], 2),
            avg_return_bps=round(ps["avg_return_bps"], 2),
        )

    # Save results
    output_path = Path("/tmp/pattern_return_profiles_polars.json")
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Pattern return profile analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
