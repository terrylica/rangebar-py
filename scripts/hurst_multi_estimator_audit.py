"""Re-audit Hurst exponent finding with multiple estimators.

Issue #57 / Task #148: Validate H~0.79 finding using:
1. R/S (Rescaled Range) - classical method
2. DFA (Detrended Fluctuation Analysis) - robust to trends
3. Variance-time plot - aggregated variance scaling

If estimators disagree significantly, the effective sample size
reduction may be less severe than originally concluded.
"""

import json
from datetime import UTC, datetime

import numpy as np


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def hurst_rs(series: np.ndarray, min_window: int = 10, max_window: int | None = None) -> float:
    """Compute Hurst exponent using R/S (Rescaled Range) method.

    H = 0.5: random walk (Brownian motion)
    H > 0.5: long memory / persistence
    H < 0.5: mean reversion / anti-persistence
    """
    n = len(series)
    if max_window is None:
        max_window = n // 4

    # Generate log-spaced window sizes
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
    )

    rs_values = []
    for w in window_sizes:
        n_windows = n // w
        if n_windows < 2:
            continue

        rs_list = []
        for i in range(n_windows):
            segment = series[i * w : (i + 1) * w]
            mean_seg = np.mean(segment)
            cumdev = np.cumsum(segment - mean_seg)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((w, np.mean(rs_list)))

    if len(rs_values) < 3:
        return np.nan

    log_w = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    # Linear regression: log(R/S) = H * log(n) + c
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


def hurst_dfa(series: np.ndarray, min_window: int = 10, max_window: int | None = None) -> float:
    """Compute Hurst exponent using Detrended Fluctuation Analysis.

    More robust to trends and non-stationarities than R/S.
    """
    n = len(series)
    if max_window is None:
        max_window = n // 4

    # Cumulative sum (profile)
    profile = np.cumsum(series - np.mean(series))

    # Generate log-spaced window sizes
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
    )

    fluctuations = []
    for w in window_sizes:
        n_windows = n // w
        if n_windows < 2:
            continue

        f2_list = []
        for i in range(n_windows):
            segment = profile[i * w : (i + 1) * w]
            # Linear detrending
            x = np.arange(w)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            f2_list.append(np.mean((segment - trend) ** 2))

        if f2_list:
            fluctuations.append((w, np.sqrt(np.mean(f2_list))))

    if len(fluctuations) < 3:
        return np.nan

    log_w = np.log([x[0] for x in fluctuations])
    log_f = np.log([x[1] for x in fluctuations])

    # Linear regression: log(F) = H * log(n) + c
    slope, _ = np.polyfit(log_w, log_f, 1)
    return float(slope)


def hurst_variance_time(series: np.ndarray, max_agg: int = 100) -> float:
    """Compute Hurst exponent using variance-time plot.

    Based on: Var(X^(m)) ~ m^(2H-2) where X^(m) is aggregated series.
    """
    variances = []
    agg_levels = np.unique(np.logspace(0, np.log10(max_agg), num=20).astype(int))

    for m in agg_levels:
        if m >= len(series) // 2:
            continue
        # Aggregate series
        n_agg = len(series) // m
        agg_series = np.array([np.mean(series[i * m : (i + 1) * m]) for i in range(n_agg)])
        if len(agg_series) < 10:
            continue
        variances.append((m, np.var(agg_series, ddof=1)))

    if len(variances) < 3:
        return np.nan

    log_m = np.log([x[0] for x in variances])
    log_var = np.log([x[1] for x in variances])

    # Linear regression: log(Var) = (2H-2) * log(m) + c
    slope, _ = np.polyfit(log_m, log_var, 1)
    h = (slope + 2) / 2
    return float(h)


def main() -> None:
    """Run multi-estimator Hurst analysis on range bar returns."""
    log("INFO", "Starting multi-estimator Hurst audit")

    # Import here to avoid slow startup
    import clickhouse_connect
    from rangebar.clickhouse.tunnel import SSHTunnel

    symbol = "BTCUSDT"
    threshold = 100

    log("INFO", "Connecting to bigblack ClickHouse")

    with SSHTunnel("bigblack") as local_port:
        client = clickhouse_connect.get_client(
            host="localhost",
            port=local_port,
            database="rangebar_cache",
        )

        # Query returns
        query = f"""
        SELECT
            (close - open) / open * 10000 AS return_dbps
        FROM range_bars
        WHERE symbol = '{symbol}'
          AND threshold_decimal_bps = {threshold}
          AND ouroboros_mode = 'year'
        ORDER BY close_time_ms
        """

        log("INFO", "Querying range bar returns", symbol=symbol, threshold=threshold)
        result = client.query(query)
        returns = np.array([row[0] for row in result.result_rows], dtype=np.float64)

        log("INFO", "Data loaded", n_bars=len(returns))

        # Check return distribution first
        at_boundary = np.sum(np.abs(np.abs(returns) - threshold) < 1) / len(returns) * 100
        log(
            "INFO",
            "Return distribution",
            pct_at_boundary=round(at_boundary, 1),
            mean_return=round(np.mean(returns), 3),
            std_return=round(np.std(returns), 3),
        )

        # Compute Hurst with multiple estimators
        log("INFO", "Computing Hurst exponent with R/S method")
        h_rs = hurst_rs(returns)
        log("INFO", "R/S Hurst", H=round(h_rs, 3) if not np.isnan(h_rs) else "NaN")

        log("INFO", "Computing Hurst exponent with DFA method")
        h_dfa = hurst_dfa(returns)
        log("INFO", "DFA Hurst", H=round(h_dfa, 3) if not np.isnan(h_dfa) else "NaN")

        log("INFO", "Computing Hurst exponent with variance-time method")
        h_vt = hurst_variance_time(returns)
        log("INFO", "Variance-Time Hurst", H=round(h_vt, 3) if not np.isnan(h_vt) else "NaN")

        # Summary
        estimators = {
            "R/S": h_rs,
            "DFA": h_dfa,
            "Variance-Time": h_vt,
        }

        valid_h = [h for h in estimators.values() if not np.isnan(h)]
        mean_h = np.mean(valid_h) if valid_h else np.nan
        std_h = np.std(valid_h) if len(valid_h) > 1 else 0

        log(
            "INFO",
            "Hurst estimator summary",
            estimators={k: round(v, 3) if not np.isnan(v) else None for k, v in estimators.items()},
            mean_H=round(mean_h, 3) if not np.isnan(mean_h) else None,
            std_H=round(std_h, 3),
        )

        # Interpretation
        if mean_h > 0.6:
            interpretation = "LONG_MEMORY - effective sample size reduced"
            t_eff_factor = 2 * (1 - mean_h)
            log(
                "INFO",
                "Long memory confirmed",
                interpretation=interpretation,
                T_eff_exponent=round(t_eff_factor, 3),
                example_T_10000=round(10000 ** t_eff_factor),
            )
        elif mean_h < 0.4:
            interpretation = "MEAN_REVERTING - anti-persistence"
            log("INFO", "Mean reversion detected", interpretation=interpretation)
        else:
            interpretation = "NEAR_RANDOM_WALK - H ~ 0.5"
            log("INFO", "Near random walk", interpretation=interpretation)

        # Check if estimators agree
        if std_h > 0.1:
            log(
                "WARN",
                "Estimators disagree significantly",
                std_H=round(std_h, 3),
                recommendation="Results inconclusive - further investigation needed",
            )
        else:
            log(
                "INFO",
                "Estimators agree",
                std_H=round(std_h, 3),
                confidence="HIGH",
            )

        # Final verdict
        original_h = 0.79
        h_diff = abs(mean_h - original_h)
        if h_diff < 0.1:
            verdict = "CONFIRMED - original H~0.79 finding validated"
        elif mean_h < 0.6:
            verdict = "REFUTED - H closer to 0.5, effective sample size reduction less severe"
        else:
            verdict = f"MODIFIED - H~{round(mean_h, 2)} (was {original_h})"

        log("INFO", "Audit verdict", verdict=verdict, original_H=original_h, new_H=round(mean_h, 3))


if __name__ == "__main__":
    main()
