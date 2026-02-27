"""TDA Walk-Forward Optimization audit on SOLUSDT 250 dbps.

Issue #57 / Task #149: Proper OOS validation using rolling walk-forward.

Train on window N, test on window N+1, roll forward.
This validates if TDA features have REAL predictive power.
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
    """Compute TDA features using GPU-accelerated ripser++."""
    try:
        import ripserplusplus as rpp
    except ImportError:
        log("ERROR", "ripserplusplus not installed")
        return {}

    if len(embedded) < 10:
        return {}

    max_points = 500
    if len(embedded) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embedded), max_points, replace=False)
        embedded = embedded[idx]

    result = rpp.run("--format point-cloud --dim 1", embedded)

    def to_2d_array(dgm: np.ndarray) -> np.ndarray:
        if len(dgm) == 0:
            return np.array([]).reshape(0, 2)
        births = np.array([p[0] for p in dgm])
        deaths = np.array([p[1] for p in dgm])
        return np.column_stack([births, deaths])

    features = {}

    if 0 in result and len(result[0]) > 0:
        h0 = to_2d_array(result[0])
        h0_finite = h0[np.isfinite(h0[:, 1])]
        if len(h0_finite) > 0:
            lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
            features["h0_max"] = float(np.max(lifetimes))
            features["h0_mean"] = float(np.mean(lifetimes))

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
    """Run WFO TDA audit on SOLUSDT 250 dbps."""
    log("INFO", "Starting TDA WFO audit on SOLUSDT 250 dbps", host="bigblack", method="walk-forward")

    import clickhouse_connect

    symbol = "SOLUSDT"
    threshold = 250

    # Best params from initial audit
    dim, delay, window = 3, 2, 100

    # WFO parameters - use more folds for statistical power
    # With 7751 windows, use rolling approach where each fold advances by step_size
    train_periods = 200  # Number of windows to train on
    test_periods = 50    # Number of windows to test on
    step_size = 50       # Roll forward by this many windows each fold
    # Calculate max folds: (7751 - 200 - 50) / 50 = ~150 folds possible
    n_folds = 100        # Number of walk-forward folds

    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        database="rangebar_cache",
    )

    query = f"""
    SELECT
        close_time_ms,
        (close - open) / open * 10000 AS return_dbps
    FROM rangebar_cache.range_bars
    WHERE symbol = '{symbol}'
      AND threshold_decimal_bps = {threshold}
      AND ouroboros_mode = 'year'
      AND close_time_ms >= 1704067200000  -- 2024-01-01
    ORDER BY close_time_ms
    """

    log("INFO", "Querying data", symbol=symbol, threshold=threshold)
    result = client.query(query)
    returns = np.array([row[1] for row in result.result_rows], dtype=np.float64)

    log("INFO", "Data loaded", n_bars=len(returns))

    # Compute forward RV
    rv_window = 20
    forward_rv = np.full(len(returns), np.nan)
    for i in range(len(returns) - rv_window):
        forward_rv[i] = np.std(returns[i + 1 : i + 1 + rv_window])

    # Total windows available
    total_windows = (len(returns) - rv_window) // window

    log("INFO", "WFO setup",
        total_windows=total_windows,
        train_periods=train_periods,
        test_periods=test_periods,
        n_folds=n_folds,
        dim=dim, delay=delay, window=window)

    # Check if enough data with rolling step approach
    # Need: train_periods + test_periods + (n_folds - 1) * step_size windows
    needed_windows = train_periods + test_periods + (n_folds - 1) * step_size
    if total_windows < needed_windows:
        log("ERROR", "Not enough data for WFO",
            needed=needed_windows,
            available=total_windows)
        return

    all_oos_results = []

    for fold in range(n_folds):
        fold_start = fold * step_size  # Roll forward by step_size each fold
        train_start = fold_start
        train_end = train_start + train_periods
        test_start = train_end
        test_end = test_start + test_periods

        log("INFO", f"Fold {fold + 1}/{n_folds}",
            train_windows=f"{train_start}-{train_end}",
            test_windows=f"{test_start}-{test_end}")

        # Compute TDA features for training windows
        train_features = []
        train_rv = []

        for w in range(train_start, train_end):
            start = w * window
            end = start + window

            if end + rv_window > len(returns):
                break

            window_returns = returns[start:end]
            embedded = time_delay_embedding(window_returns, dim, delay)

            if len(embedded) < 10:
                continue

            features = compute_tda_features_gpu(embedded)
            if not features or "h1_count" not in features:
                continue

            fwd_rv = np.nanmean(forward_rv[end : end + window])
            if np.isnan(fwd_rv):
                continue

            train_features.append(features["h1_count"])
            train_rv.append(fwd_rv)

        if len(train_features) < 50:
            log("WARN", "Too few training samples", n_samples=len(train_features))
            continue

        # Determine threshold from training data (tercile)
        train_features_arr = np.array(train_features)
        high_thresh = np.percentile(train_features_arr, 67)
        low_thresh = np.percentile(train_features_arr, 33)

        log("INFO", "Training thresholds",
            high_thresh=round(high_thresh, 2),
            low_thresh=round(low_thresh, 2),
            n_train=len(train_features))

        # Test on OOS data
        test_features = []
        test_rv = []
        test_predictions = []  # 1 = high RV predicted, -1 = low RV predicted, 0 = neutral

        for w in range(test_start, test_end):
            start = w * window
            end = start + window

            if end + rv_window > len(returns):
                break

            window_returns = returns[start:end]
            embedded = time_delay_embedding(window_returns, dim, delay)

            if len(embedded) < 10:
                continue

            features = compute_tda_features_gpu(embedded)
            if not features or "h1_count" not in features:
                continue

            fwd_rv = np.nanmean(forward_rv[end : end + window])
            if np.isnan(fwd_rv):
                continue

            h1_count = features["h1_count"]
            test_features.append(h1_count)
            test_rv.append(fwd_rv)

            # Predict based on training thresholds
            if h1_count >= high_thresh:
                test_predictions.append(1)  # Predict high RV
            elif h1_count <= low_thresh:
                test_predictions.append(-1)  # Predict low RV
            else:
                test_predictions.append(0)  # Neutral

        if len(test_features) < 20:
            log("WARN", "Too few test samples", n_samples=len(test_features))
            continue

        test_rv_arr = np.array(test_rv)
        test_pred_arr = np.array(test_predictions)

        # Evaluate OOS performance
        high_pred_mask = test_pred_arr == 1
        low_pred_mask = test_pred_arr == -1

        if np.sum(high_pred_mask) > 5 and np.sum(low_pred_mask) > 5:
            rv_when_high_pred = test_rv_arr[high_pred_mask]
            rv_when_low_pred = test_rv_arr[low_pred_mask]

            oos_diff = np.mean(rv_when_high_pred) - np.mean(rv_when_low_pred)
            pooled_std = np.sqrt((np.var(rv_when_high_pred) + np.var(rv_when_low_pred)) / 2)

            if pooled_std > 1e-10:
                n_high = len(rv_when_high_pred)
                n_low = len(rv_when_low_pred)
                oos_t = oos_diff / (pooled_std * np.sqrt(1/n_high + 1/n_low))
            else:
                oos_t = 0

            fold_result = {
                "fold": fold + 1,
                "n_test": len(test_features),
                "n_high_pred": int(np.sum(high_pred_mask)),
                "n_low_pred": int(np.sum(low_pred_mask)),
                "rv_high_pred": round(float(np.mean(rv_when_high_pred)), 2),
                "rv_low_pred": round(float(np.mean(rv_when_low_pred)), 2),
                "oos_diff": round(float(oos_diff), 2),
                "oos_t": round(float(oos_t), 2),
            }

            all_oos_results.append(fold_result)
            log("INFO", "OOS result", **fold_result)
        else:
            log("WARN", "Too few predictions in terciles",
                n_high=int(np.sum(high_pred_mask)),
                n_low=int(np.sum(low_pred_mask)))

    # Summary
    log("INFO", "WFO audit complete", total_folds=len(all_oos_results))

    if all_oos_results:
        avg_oos_t = np.mean([r["oos_t"] for r in all_oos_results])
        avg_oos_diff = np.mean([r["oos_diff"] for r in all_oos_results])
        positive_folds = sum(1 for r in all_oos_results if r["oos_t"] > 0)
        significant_folds = sum(1 for r in all_oos_results if abs(r["oos_t"]) >= 2.0)

        summary = {
            "avg_oos_t": round(float(avg_oos_t), 2),
            "avg_oos_diff": round(float(avg_oos_diff), 2),
            "positive_folds": f"{positive_folds}/{len(all_oos_results)}",
            "significant_folds": f"{significant_folds}/{len(all_oos_results)}",
        }

        log("INFO", "WFO summary", **summary)

        if avg_oos_t >= 2.0 and positive_folds >= len(all_oos_results) * 0.7:
            log("INFO", "VERDICT: TDA h1_count shows OOS predictive power",
                verdict="POTENTIAL ALPHA - needs multi-symbol validation")
        elif avg_oos_t > 0:
            log("INFO", "VERDICT: TDA shows weak OOS signal",
                verdict="MARGINAL - likely not robust")
        else:
            log("INFO", "VERDICT: TDA fails OOS validation",
                verdict="NULL RESULT - no predictive power")


if __name__ == "__main__":
    main()
