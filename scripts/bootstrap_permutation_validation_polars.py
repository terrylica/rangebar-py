#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Bootstrap and permutation statistical validation for OOD robust patterns (Polars).

ADVERSARIAL AUDIT: Validate that observed t-stats are not statistical artifacts
using bootstrap resampling and permutation testing.

1. Bootstrap (1000 iterations): Compute 95% confidence intervals for mean returns
2. Permutation (1000 iterations): Establish null distribution of t-stats
3. Compare observed vs null to compute empirical p-values

GitHub Issue: #52 (adversarial audit)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# 11 Universal OOD robust patterns from prior analysis
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


def log_warn(message: str, **kwargs: object) -> None:
    """Log WARNING level message."""
    log_json("WARNING", message, **kwargs)


# =============================================================================
# Technical Indicators (Same as prior analysis)
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


def add_forward_returns(df: pl.LazyFrame, horizon: int = 10) -> pl.LazyFrame:
    """Add forward returns at specified horizon."""
    return df.with_columns(
        (pl.col("Close").shift(-horizon) / pl.col("Close") - 1).alias(f"fwd_ret_{horizon}")
    )


# =============================================================================
# Statistical Functions
# =============================================================================


def compute_t_stat(returns: np.ndarray) -> float:
    """Compute t-statistic for returns."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)

    if std == 0:
        return 0.0

    return float(mean / (std / math.sqrt(n)))


def bootstrap_mean_ci(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute bootstrap confidence interval for mean return."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(returns)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return {
        "mean": float(np.mean(returns)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


def permutation_test_t_stat(
    returns: np.ndarray,
    observed_t: float,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute permutation test for t-statistic.

    Null hypothesis: returns have zero mean (randomize signs).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(returns)
    null_t_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Randomize signs to simulate null (zero mean)
        signs = rng.choice([-1, 1], size=n)
        permuted = returns * signs
        null_t_stats[i] = compute_t_stat(permuted)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_t_stats) >= np.abs(observed_t))

    return {
        "observed_t": float(observed_t),
        "null_mean_t": float(np.mean(null_t_stats)),
        "null_std_t": float(np.std(null_t_stats)),
        "null_percentile_95": float(np.percentile(np.abs(null_t_stats), 95)),
        "null_percentile_99": float(np.percentile(np.abs(null_t_stats), 99)),
        "empirical_p_value": float(p_value),
        "n_permutations": n_permutations,
        "is_significant_05": p_value < 0.05,
        "is_significant_01": p_value < 0.01,
    }


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


def run_bootstrap_permutation_validation(
    symbol: str,
    threshold_dbps: int,
    start_date: str,
    end_date: str,
    horizon: int = 10,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
) -> dict:
    """Run bootstrap and permutation validation for a symbol."""
    log_info("Starting bootstrap/permutation validation", symbol=symbol,
             horizon=horizon, n_bootstrap=n_bootstrap, n_permutations=n_permutations)

    # Load and prepare data
    df = load_range_bars_polars(symbol, threshold_dbps, start_date, end_date)
    df = compute_sma(df, "Close", 20, "sma20")
    df = compute_sma(df, "Close", 50, "sma50")
    df = compute_rsi(df, "Close", 14, "rsi14")
    df = classify_regimes(df)
    df = add_bar_direction(df)
    df = add_patterns(df)
    df = add_forward_returns(df, horizon)

    result_df = df.collect()
    log_info("Data loaded", symbol=symbol, total_bars=len(result_df))

    # Fixed RNG for reproducibility
    rng = np.random.default_rng(42)

    results = {
        "symbol": symbol,
        "horizon": horizon,
        "total_bars": len(result_df),
        "patterns": {},
    }

    for regime, pattern in UNIVERSAL_PATTERNS:
        key = f"{regime}|{pattern}"

        subset = result_df.filter(
            (pl.col("regime") == regime) & (pl.col("pattern_2bar") == pattern)
        )

        return_col = f"fwd_ret_{horizon}"
        returns = subset.select(return_col).drop_nulls().to_series().to_numpy()
        n = len(returns)

        if n < 50:
            results["patterns"][key] = {"error": "insufficient_samples", "count": n}
            continue

        # Compute observed t-stat
        observed_t = compute_t_stat(returns)

        # Bootstrap confidence interval
        bootstrap_result = bootstrap_mean_ci(returns, n_bootstrap=n_bootstrap, rng=rng)

        # Permutation test
        permutation_result = permutation_test_t_stat(
            returns, observed_t, n_permutations=n_permutations, rng=rng
        )

        # Combine results
        results["patterns"][key] = {
            "count": n,
            "mean_return_bps": float(np.mean(returns) * 10000),
            "observed_t_stat": round(observed_t, 2),
            "bootstrap": {
                "ci_lower_bps": round(bootstrap_result["ci_lower"] * 10000, 2),
                "ci_upper_bps": round(bootstrap_result["ci_upper"] * 10000, 2),
                "ci_excludes_zero": (bootstrap_result["ci_lower"] > 0 or bootstrap_result["ci_upper"] < 0),
            },
            "permutation": {
                "empirical_p_value": round(permutation_result["empirical_p_value"], 4),
                "null_95_percentile": round(permutation_result["null_percentile_95"], 2),
                "null_99_percentile": round(permutation_result["null_percentile_99"], 2),
                "is_significant_05": permutation_result["is_significant_05"],
                "is_significant_01": permutation_result["is_significant_01"],
            },
        }

        log_info(
            f"Validated {key}",
            symbol=symbol,
            pattern=key,
            count=n,
            t_stat=round(observed_t, 2),
            bootstrap_ci_excludes_zero=results["patterns"][key]["bootstrap"]["ci_excludes_zero"],
            permutation_p_value=round(permutation_result["empirical_p_value"], 4),
        )

    return results


def main() -> int:
    """Main entry point."""
    log_info(
        "Starting bootstrap and permutation validation (Polars)",
        script="bootstrap_permutation_validation_polars.py",
    )

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold_dbps = 100
    start_date = "2022-01-01"
    end_date = "2026-01-31"
    horizon = 10  # Focus on 10-bar returns (genuine alpha)
    n_bootstrap = 1000
    n_permutations = 1000

    log_info("=== STATISTICAL ARTIFACT AUDIT ===")
    log_info("Bootstrap resampling for confidence intervals")
    log_info("Permutation testing for null distribution")

    all_results = {}

    for symbol in symbols:
        try:
            result = run_bootstrap_permutation_validation(
                symbol=symbol,
                threshold_dbps=threshold_dbps,
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
            )
            all_results[symbol] = result
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            log_json("ERROR", "Validation failed", symbol=symbol, error=str(e))

    # Cross-symbol summary
    log_info("=== CROSS-SYMBOL STATISTICAL VALIDATION SUMMARY ===")

    pattern_validation = {}
    for regime, pattern in UNIVERSAL_PATTERNS:
        key = f"{regime}|{pattern}"
        bootstrap_pass = 0
        permutation_pass = 0
        total_symbols = 0

        for _symbol, result in all_results.items():
            pat_data = result.get("patterns", {}).get(key, {})
            if "error" in pat_data:
                continue
            total_symbols += 1

            if pat_data.get("bootstrap", {}).get("ci_excludes_zero", False):
                bootstrap_pass += 1

            if pat_data.get("permutation", {}).get("is_significant_01", False):
                permutation_pass += 1

        pattern_validation[key] = {
            "bootstrap_pass": bootstrap_pass,
            "permutation_pass": permutation_pass,
            "total_symbols": total_symbols,
            "universal_bootstrap": bootstrap_pass >= total_symbols,
            "universal_permutation": permutation_pass >= total_symbols,
        }

        status = "PASS" if (bootstrap_pass >= total_symbols and permutation_pass >= total_symbols) else "PARTIAL"
        log_info(
            f"Statistical validation: {key}",
            pattern=key,
            status=status,
            bootstrap_pass=f"{bootstrap_pass}/{total_symbols}",
            permutation_pass=f"{permutation_pass}/{total_symbols}",
        )

    # Final summary
    fully_validated = sum(
        1 for v in pattern_validation.values()
        if v["universal_bootstrap"] and v["universal_permutation"]
    )

    log_info("=== FINAL STATISTICAL VALIDATION SUMMARY ===")
    log_info(
        "Results",
        total_patterns=len(UNIVERSAL_PATTERNS),
        fully_validated=fully_validated,
        validation_rate=round(fully_validated / len(UNIVERSAL_PATTERNS) * 100, 1),
    )

    if fully_validated < len(UNIVERSAL_PATTERNS):
        log_warn(
            "Some patterns failed statistical validation",
            failed_count=len(UNIVERSAL_PATTERNS) - fully_validated,
        )
    else:
        log_info("All patterns passed both bootstrap and permutation tests")

    # Save results
    output_path = Path("/tmp/bootstrap_permutation_validation_polars.json")
    with output_path.open("w") as f:
        json.dump({
            "all_results": all_results,
            "pattern_validation": pattern_validation,
            "summary": {
                "total_patterns": len(UNIVERSAL_PATTERNS),
                "fully_validated": fully_validated,
            },
        }, f, indent=2, default=str)
    log_info("Results saved", path=str(output_path))

    log_info("Bootstrap and permutation validation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
