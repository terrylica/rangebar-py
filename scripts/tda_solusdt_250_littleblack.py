"""TDA re-audit on SOLUSDT 250 dbps - GPU version.

Issue #57 / Task #149: Test TDA features with alternative parameters
on focused dataset (SOLUSDT 250 dbps - highest frequency).

Tests if TDA H1 features predict forward realized volatility.
Runs on bigblack (RTX 4090) with GPU-accelerated ripser++ connecting to local ClickHouse.
"""

import json
from datetime import datetime, timezone

import numpy as np


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def time_delay_embedding(series: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Create time-delay embedding of a 1D series."""
    n = len(series) - (dim - 1) * delay
    if n <= 0:
        return np.array([])

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = series[i * delay : i * delay + n]
    return embedded


def compute_tda_features_gpu(embedded: np.ndarray) -> dict[str, float]:
    """Compute TDA features from embedded point cloud using GPU-accelerated ripser++."""
    try:
        import ripserplusplus as rpp
    except ImportError:
        log("ERROR", "ripserplusplus not installed")
        return {}

    if len(embedded) < 10:
        return {}

    # ripser++ can handle larger point clouds efficiently with GPU
    max_points = 500  # Can be larger than CPU ripser
    if len(embedded) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embedded), max_points, replace=False)
        embedded = embedded[idx]

    # Compute persistent homology using GPU
    # ripser++ returns dict: {0: [(birth, death), ...], 1: [(birth, death), ...]}
    result = rpp.run("--format point-cloud --dim 1", embedded)

    features = {}

    # ripser++ returns structured arrays with (birth, death) tuples
    # Convert to 2D array for processing
    def to_2d_array(dgm: np.ndarray) -> np.ndarray:
        """Convert structured array of (birth, death) to 2D array."""
        if len(dgm) == 0:
            return np.array([]).reshape(0, 2)
        # Extract birth and death from structured array
        births = np.array([p[0] for p in dgm])
        deaths = np.array([p[1] for p in dgm])
        return np.column_stack([births, deaths])

    # H0 features (connected components)
    if 0 in result and len(result[0]) > 0:
        h0 = to_2d_array(result[0])
        h0_finite = h0[np.isfinite(h0[:, 1])]
        if len(h0_finite) > 0:
            lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
            features["h0_max"] = float(np.max(lifetimes))
            features["h0_mean"] = float(np.mean(lifetimes))

    # H1 features (loops/holes) - key for market structure
    if 1 in result and len(result[1]) > 0:
        h1 = to_2d_array(result[1])
        lifetimes = h1[:, 1] - h1[:, 0]
        features["h1_max"] = float(np.max(lifetimes))
        features["h1_mean"] = float(np.mean(lifetimes))
        features["h1_count"] = len(h1)
        features["h1_total"] = float(np.sum(lifetimes))
        features["h1_l2"] = float(np.sqrt(np.sum(lifetimes**2)))
    else:
        features["h1_max"] = 0.0
        features["h1_mean"] = 0.0
        features["h1_count"] = 0
        features["h1_total"] = 0.0
        features["h1_l2"] = 0.0

    return features


def main() -> None:
    """Run TDA audit on SOLUSDT 250 dbps with GPU acceleration."""
    log("INFO", "Starting TDA audit on SOLUSDT 250 dbps", host="bigblack", gpu="ripser++")

    import clickhouse_connect

    symbol = "SOLUSDT"
    threshold = 250

    # Extended parameter combinations to test (GPU allows more)
    params = [
        {"dim": 2, "delay": 1, "window": 50},
        {"dim": 3, "delay": 1, "window": 50},
        {"dim": 3, "delay": 2, "window": 100},
        {"dim": 5, "delay": 1, "window": 100},
        {"dim": 5, "delay": 2, "window": 200},
        {"dim": 7, "delay": 1, "window": 200},  # Higher dim with GPU
        {"dim": 7, "delay": 3, "window": 300},  # Longer delay
    ]

    # Connect to local ClickHouse on littleblack (no SSH tunnel)
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        database="rangebar_cache",
    )

    # Query returns - limit to recent data for speed
    query = f"""
    SELECT
        timestamp_ms,
        (close - open) / open * 10000 AS return_dbps
    FROM rangebar_cache.range_bars
    WHERE symbol = '{symbol}'
      AND threshold_decimal_bps = {threshold}
      AND ouroboros_mode = 'year'
      AND timestamp_ms >= 1704067200000  -- 2024-01-01
    ORDER BY timestamp_ms
    """

    log("INFO", "Querying data", symbol=symbol, threshold=threshold)
    result = client.query(query)
    returns = np.array([row[1] for row in result.result_rows], dtype=np.float64)

    log("INFO", "Data loaded", n_bars=len(returns))

    # Compute forward realized volatility (20-bar rolling std)
    rv_window = 20
    forward_rv = np.full(len(returns), np.nan)
    for i in range(len(returns) - rv_window):
        forward_rv[i] = np.std(returns[i + 1 : i + 1 + rv_window])

    all_results = []

    for p in params:
        dim, delay, window = p["dim"], p["delay"], p["window"]
        log("INFO", "Testing parameters", dim=dim, delay=delay, window=window)

        n_windows = (len(returns) - rv_window) // window
        if n_windows < 50:
            log("WARN", "Too few windows", n_windows=n_windows)
            continue

        tda_features = []
        rv_values = []

        # Process more windows with GPU (up to 1000)
        for w in range(min(n_windows, 1000)):
            start = w * window
            end = start + window

            if end + rv_window > len(returns):
                break

            # Time-delay embedding
            window_returns = returns[start:end]
            embedded = time_delay_embedding(window_returns, dim, delay)

            if len(embedded) < 10:
                continue

            # Compute TDA with GPU
            features = compute_tda_features_gpu(embedded)
            if not features or "h1_l2" not in features:
                continue

            # Forward RV
            fwd_rv = np.nanmean(forward_rv[end : end + window])
            if np.isnan(fwd_rv):
                continue

            tda_features.append(features)
            rv_values.append(fwd_rv)

        if len(tda_features) < 30:
            log("WARN", "Too few valid windows", n_valid=len(tda_features))
            continue

        rv_array = np.array(rv_values)
        log("INFO", "Windows processed", n_valid=len(tda_features), dim=dim, delay=delay, window=window)

        # Test each TDA feature
        for feat_name in ["h1_l2", "h1_count", "h1_total", "h1_max"]:
            feat_vals = np.array([f.get(feat_name, 0) for f in tda_features])

            if np.std(feat_vals) < 1e-10:
                continue

            # Tercile analysis
            low_thresh = np.percentile(feat_vals, 33)
            high_thresh = np.percentile(feat_vals, 67)

            low_mask = feat_vals <= low_thresh
            high_mask = feat_vals >= high_thresh

            if np.sum(high_mask) < 10 or np.sum(low_mask) < 10:
                continue

            # Compare RV in high vs low TDA terciles
            rv_high = rv_array[high_mask]
            rv_low = rv_array[low_mask]

            mean_diff = np.mean(rv_high) - np.mean(rv_low)
            pooled_std = np.sqrt(
                (np.var(rv_high) + np.var(rv_low)) / 2
            )

            if pooled_std > 1e-10:
                t_stat = mean_diff / (pooled_std * np.sqrt(2 / len(rv_high)))
            else:
                t_stat = 0

            result_entry = {
                "dim": dim,
                "delay": delay,
                "window": window,
                "feature": feat_name,
                "rv_high_mean": round(float(np.mean(rv_high)), 2),
                "rv_low_mean": round(float(np.mean(rv_low)), 2),
                "mean_diff": round(float(mean_diff), 2),
                "t_stat": round(float(t_stat), 2),
                "n_samples": len(rv_high),
            }

            all_results.append(result_entry)

            log("INFO", "Result", **result_entry)

    # Summary
    log("INFO", "Audit complete", total_tests=len(all_results), gpu="ripser++")

    if all_results:
        best = max(all_results, key=lambda x: abs(x["t_stat"]))
        log("INFO", "Best result", **best)

        significant = [r for r in all_results if abs(r["t_stat"]) >= 2.0]
        if significant:
            log(
                "INFO",
                "Potentially significant results (|t| >= 2.0)",
                n_significant=len(significant),
                results=significant,
            )
        else:
            log(
                "INFO",
                "NO significant TDA-RV relationship found",
                max_t=best["t_stat"],
                verdict="TDA null result CONFIRMED on SOLUSDT 250 dbps",
            )


if __name__ == "__main__":
    main()
