#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Out-of-Sample validation for OOD robust patterns (Polars).

ADVERSARIAL AUDIT: Test if patterns discovered on 2022-2024 data
hold on completely held-out 2025-2026 data.

This is a critical test for local bias / overfitting detection.

GitHub Issue: #52 (adversarial audit)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Universal patterns claimed to be OOD robust (from prior analysis)
CLAIMED_PATTERNS = [
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


def log_warn(message: str, **kwargs: object) -> None:
    """Log WARNING level message."""
    log_json("WARNING", message, **kwargs)


# =============================================================================
# Technical Indicators (Same as original analysis)
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
# Validation Functions
# =============================================================================


def compute_t_stat(returns: pl.Series) -> float:
    """Compute t-statistic for returns."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = returns.mean()
    std = returns.std()

    if std is None or std == 0 or mean is None:
        return 0.0

    return float(mean / (std / math.sqrt(n)))


def validate_pattern_oos(
    df: pl.DataFrame,
    regime: str,
    pattern: str,
    horizons: list[int],
    min_samples: int = 50,
) -> dict:
    """Validate a single pattern on out-of-sample data."""
    subset = df.filter(
        (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
    )

    total_count = len(subset)
    if total_count < min_samples:
        return {"error": "insufficient_samples", "count": total_count}

    result = {
        "regime": regime,
        "pattern": pattern,
        "total_count": total_count,
        "horizons": {},
    }

    for h in horizons:
        return_col = f"fwd_ret_{h}"
        returns = subset.select(return_col).drop_nulls().to_series()
        n = len(returns)

        if n < min_samples:
            result["horizons"][h] = {"error": "insufficient_samples", "count": n}
            continue

        mean_ret = returns.mean()
        std_ret = returns.std()
        t_stat = compute_t_stat(returns)
        win_rate = (returns > 0).sum() / n

        result["horizons"][h] = {
            "count": n,
            "mean_return_bps": float(mean_ret * 10000) if mean_ret is not None else 0,
            "std_return_bps": float(std_ret * 10000) if std_ret is not None else 0,
            "t_stat": round(t_stat, 2),
            "win_rate": round(float(win_rate), 4),
            "is_significant": abs(t_stat) >= 5.0,
            "sign": "positive" if (mean_ret or 0) > 0 else "negative",
        }

    return result


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

    log_info("Loading range bars", symbol=symbol, threshold_dbps=threshold_dbps,
             start=start_date, end=end_date)

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


def run_oos_validation(
    symbol: str,
    threshold_dbps: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizons: list[int],
) -> dict:
    """Run out-of-sample validation for a symbol."""
    log_info("Starting OOS validation", symbol=symbol,
             train=f"{train_start} to {train_end}",
             test=f"{test_start} to {test_end}")

    # Load TEST data only (we're validating, not training)
    df = load_range_bars_polars(symbol, threshold_dbps, test_start, test_end)
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")
    df = classify_regimes(df)
    df = add_bar_direction(df)
    df = add_patterns(df)
    df = add_forward_returns(df, horizons)

    result_df = df.collect()
    test_bars = len(result_df)

    log_info("Test data loaded", symbol=symbol, test_bars=test_bars)

    # Validate each claimed pattern
    results = {
        "symbol": symbol,
        "test_period": f"{test_start} to {test_end}",
        "test_bars": test_bars,
        "patterns": {},
    }

    for regime, pattern in CLAIMED_PATTERNS:
        key = f"{regime}|{pattern}"
        validation = validate_pattern_oos(result_df, regime, pattern, horizons)
        results["patterns"][key] = validation

        # Check if still significant at 10-bar horizon
        h10 = validation.get("horizons", {}).get(10, {})
        is_still_robust = h10.get("is_significant", False)

        log_info(
            f"Pattern {key}",
            symbol=symbol,
            pattern=key,
            test_count=validation.get("total_count", 0),
            t_stat_10bar=h10.get("t_stat", 0),
            is_still_robust=is_still_robust,
        )

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting Out-of-Sample validation (Polars)",
        script="oos_validation_polars.py",
    )

    # Configuration - STRICT train/test split
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    train_start = "2022-01-01"
    train_end = "2024-12-31"
    test_start = "2025-01-01"
    test_end = "2026-01-31"
    horizons = [1, 3, 5, 10]

    log_info("=== ADVERSARIAL AUDIT: OUT-OF-SAMPLE VALIDATION ===")
    log_info("Train period (patterns discovered)", start=train_start, end=train_end)
    log_info("Test period (completely held out)", start=test_start, end=test_end)

    all_results = {}

    for symbol in symbols:
        try:
            result = run_oos_validation(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizons=horizons,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Validation failed", symbol=symbol, error=str(e))

    # Cross-symbol OOS summary
    log_info("=== CROSS-SYMBOL OOS RESULTS ===")

    # Count patterns that remain significant at 10-bar horizon across ALL symbols
    pattern_oos_status = {}
    for regime, pattern in CLAIMED_PATTERNS:
        key = f"{regime}|{pattern}"
        still_robust_count = 0
        t_stats = []

        for _symbol, result in all_results.items():
            pat_data = result.get("patterns", {}).get(key, {})
            h10 = pat_data.get("horizons", {}).get(10, {})
            if h10.get("is_significant", False):
                still_robust_count += 1
            t_stats.append(h10.get("t_stat", 0))

        is_universal_oos = still_robust_count >= len(symbols)
        pattern_oos_status[key] = {
            "still_robust_count": still_robust_count,
            "is_universal_oos": is_universal_oos,
            "t_stats": t_stats,
            "avg_t_stat": sum(t_stats) / len(t_stats) if t_stats else 0,
        }

        status = "PASS" if is_universal_oos else "FAIL"
        log_info(
            f"OOS {status}: {key}",
            pattern=key,
            status=status,
            robust_symbols=still_robust_count,
            total_symbols=len(symbols),
            avg_t_stat=round(pattern_oos_status[key]["avg_t_stat"], 2),
        )

    # Final summary
    passed = sum(1 for v in pattern_oos_status.values() if v["is_universal_oos"])
    failed = len(pattern_oos_status) - passed

    log_info("=== OOS VALIDATION SUMMARY ===")
    log_info(
        "Final results",
        total_patterns=len(CLAIMED_PATTERNS),
        passed_oos=passed,
        failed_oos=failed,
        pass_rate=round(passed / len(CLAIMED_PATTERNS) * 100, 1),
    )

    if failed > 0:
        log_warn(
            "AUDIT FINDING: Some patterns failed OOS validation",
            failed_count=failed,
            implication="Possible overfitting to in-sample period",
        )
    else:
        log_info(
            "All patterns passed OOS validation",
            implication="Patterns appear genuinely robust",
        )

    # Save results
    output_path = Path("/tmp/oos_validation_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "pattern_oos_status": pattern_oos_status,
            "summary": {
                "total_patterns": len(CLAIMED_PATTERNS),
                "passed_oos": passed,
                "failed_oos": failed,
            },
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Out-of-Sample validation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
