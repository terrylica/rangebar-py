"""Re-audit TDA volatility forecast with alternative embedding parameters.

Issue #57 / Task #149: The original TDA analysis used fixed parameters.
Re-test with parameter sweeps to check if the null result is robust.

Original finding: TDA H1 L2 velocity tercile does NOT predict forward RV
(t-stats -1.67 to +1.01, threshold ±3.0)
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


def time_delay_embedding(series: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Create time-delay embedding of a 1D series.

    Args:
        series: 1D array of values
        dim: Embedding dimension
        delay: Time delay between coordinates

    Returns:
        2D array of shape (n_points, dim)
    """
    n = len(series) - (dim - 1) * delay
    if n <= 0:
        return np.array([])

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = series[i * delay : i * delay + n]
    return embedded


def compute_tda_features(embedded: np.ndarray) -> dict[str, float]:
    """Compute TDA features from embedded point cloud.

    Uses ripser for persistent homology computation.
    """
    try:
        from ripser import ripser
    except ImportError:
        log("ERROR", "ripser not installed, skipping TDA computation")
        return {}

    if len(embedded) < 10:
        return {}

    # Subsample if too large (ripser is O(n^3))
    max_points = 500
    if len(embedded) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embedded), max_points, replace=False)
        embedded = embedded[idx]

    # Compute persistent homology
    result = ripser(embedded, maxdim=1)
    dgms = result["dgms"]

    features = {}

    # H0 features (connected components)
    h0 = dgms[0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    if len(h0_finite) > 0:
        lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        features["h0_max_lifetime"] = float(np.max(lifetimes))
        features["h0_mean_lifetime"] = float(np.mean(lifetimes))
        features["h0_n_components"] = len(h0_finite)

    # H1 features (loops/holes)
    if len(dgms) > 1:
        h1 = dgms[1]
        if len(h1) > 0:
            lifetimes = h1[:, 1] - h1[:, 0]
            features["h1_max_lifetime"] = float(np.max(lifetimes))
            features["h1_mean_lifetime"] = float(np.mean(lifetimes))
            features["h1_n_holes"] = len(h1)
            features["h1_total_persistence"] = float(np.sum(lifetimes))
            # L2 norm of persistence diagram
            features["h1_l2_norm"] = float(np.sqrt(np.sum(lifetimes**2)))
        else:
            features["h1_max_lifetime"] = 0.0
            features["h1_mean_lifetime"] = 0.0
            features["h1_n_holes"] = 0
            features["h1_total_persistence"] = 0.0
            features["h1_l2_norm"] = 0.0

    return features


def main() -> None:
    """Run TDA parameter sweep audit."""
    log("INFO", "Starting TDA parameter sweep audit")

    import clickhouse_connect
    from rangebar.clickhouse.tunnel import SSHTunnel

    symbol = "BTCUSDT"
    threshold = 100

    # Parameter sweep
    embedding_dims = [2, 3, 5]
    delays = [1, 2, 5]
    window_sizes = [50, 100, 200]  # bars per TDA window

    log("INFO", "Connecting to bigblack ClickHouse")

    with SSHTunnel("bigblack") as local_port:
        client = clickhouse_connect.get_client(
            host="localhost",
            port=local_port,
            database="rangebar_cache",
        )

        # Query returns and realized volatility
        query = f"""
        SELECT
            timestamp_ms,
            (close - open) / open * 10000 AS return_dbps
        FROM range_bars
        WHERE symbol = '{symbol}'
          AND threshold_decimal_bps = {threshold}
          AND ouroboros_mode = 'year'
        ORDER BY timestamp_ms
        """

        log("INFO", "Querying range bar data", symbol=symbol, threshold=threshold)
        result = client.query(query)
        returns = np.array([row[1] for row in result.result_rows], dtype=np.float64)

        log("INFO", "Data loaded", n_bars=len(returns))

        # Compute forward realized volatility (20-bar rolling std)
        rv_window = 20
        forward_rv = np.zeros(len(returns))
        for i in range(len(returns) - rv_window):
            forward_rv[i] = np.std(returns[i + 1 : i + 1 + rv_window])

        # Test each parameter combination
        results = []

        for dim in embedding_dims:
            for delay in delays:
                for window in window_sizes:
                    log(
                        "INFO",
                        "Testing parameters",
                        dim=dim,
                        delay=delay,
                        window=window,
                    )

                    # Compute TDA features for each window
                    n_windows = (len(returns) - rv_window) // window
                    if n_windows < 100:
                        log("WARN", "Too few windows", n_windows=n_windows)
                        continue

                    tda_features_list = []
                    rv_list = []

                    for w in range(n_windows):
                        start_idx = w * window
                        end_idx = start_idx + window

                        if end_idx + rv_window > len(returns):
                            break

                        # Time-delay embedding
                        window_returns = returns[start_idx:end_idx]
                        embedded = time_delay_embedding(window_returns, dim, delay)

                        if len(embedded) < 10:
                            continue

                        # Compute TDA features
                        features = compute_tda_features(embedded)
                        if not features:
                            continue

                        # Forward RV (mean of next window)
                        fwd_rv = np.mean(forward_rv[end_idx : end_idx + window])

                        tda_features_list.append(features)
                        rv_list.append(fwd_rv)

                    if len(tda_features_list) < 50:
                        log("WARN", "Too few valid windows", n_valid=len(tda_features_list))
                        continue

                    # Test each TDA feature for RV prediction
                    rv_array = np.array(rv_list)

                    # Classify RV into terciles
                    rv_terciles = np.zeros(len(rv_array), dtype=int)
                    rv_terciles[rv_array <= np.percentile(rv_array, 33)] = 0
                    rv_terciles[rv_array >= np.percentile(rv_array, 67)] = 2
                    rv_terciles[(rv_terciles != 0) & (rv_terciles != 2)] = 1

                    for feature_name in ["h1_l2_norm", "h1_n_holes", "h1_total_persistence"]:
                        feature_values = np.array(
                            [f.get(feature_name, 0) for f in tda_features_list]
                        )

                        if np.std(feature_values) < 1e-10:
                            continue

                        # Classify feature into terciles
                        feat_terciles = np.zeros(len(feature_values), dtype=int)
                        feat_terciles[feature_values <= np.percentile(feature_values, 33)] = 0
                        feat_terciles[feature_values >= np.percentile(feature_values, 67)] = 2
                        feat_terciles[(feat_terciles != 0) & (feat_terciles != 2)] = 1

                        # For HIGH feature tercile, compute P(HIGH RV)
                        high_feat_mask = feat_terciles == 2
                        if np.sum(high_feat_mask) < 20:
                            continue

                        p_high_rv_given_high_feat = np.mean(rv_terciles[high_feat_mask] == 2)
                        n_samples = np.sum(high_feat_mask)

                        # t-test vs null (0.33)
                        se = np.sqrt(0.33 * 0.67 / n_samples)
                        t_stat = (p_high_rv_given_high_feat - 0.33) / se

                        results.append(
                            {
                                "dim": dim,
                                "delay": delay,
                                "window": window,
                                "feature": feature_name,
                                "p_high_rv": round(p_high_rv_given_high_feat, 3),
                                "n_samples": n_samples,
                                "t_stat": round(t_stat, 2),
                            }
                        )

                        log(
                            "INFO",
                            "Result",
                            dim=dim,
                            delay=delay,
                            window=window,
                            feature=feature_name,
                            p_high_rv=round(p_high_rv_given_high_feat, 3),
                            t_stat=round(t_stat, 2),
                        )

        # Summary
        log("INFO", "Parameter sweep complete", total_tests=len(results))

        if results:
            # Find best result
            best = max(results, key=lambda x: abs(x["t_stat"]))
            log(
                "INFO",
                "Best result",
                **best,
            )

            # Check if any t-stat exceeds threshold
            significant = [r for r in results if abs(r["t_stat"]) >= 3.0]
            if significant:
                log(
                    "INFO",
                    "SIGNIFICANT results found",
                    n_significant=len(significant),
                    results=significant,
                )
            else:
                log(
                    "INFO",
                    "NO significant results",
                    max_t=best["t_stat"],
                    threshold=3.0,
                    verdict="TDA null result CONFIRMED across parameter sweep",
                )


if __name__ == "__main__":
    main()
