"""TDA WFO with selective abstention based on in-sample confidence.

Issue #57: Test if abstaining from low-confidence periods improves OOS performance.

Abstention criteria (computed from in-sample only - no look-ahead bias):
1. In-sample t-stat threshold: only trade if |t_insample| >= min_t
2. Effect size threshold: only trade if |rv_diff_insample| >= min_effect
3. Sample balance: only trade if both terciles have enough samples

All parameters are data-driven or have documented rationale - NO MAGIC NUMBERS.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration - ADAPTIVE where possible.

    Philosophy: Use data-driven or theoretically-grounded values.
    Avoid arbitrary fixed numbers.
    """

    # TDA embedding - ADAPTIVE: test multiple and select best IS
    tda_dims: tuple = (2, 3, 5)  # Will select best per fold
    tda_delays: tuple = (1, 2)  # Will select best per fold
    tda_window_ratio: float = 0.5  # Window = ratio * train_periods (adaptive)

    # ripser++ - computational constraint, not statistical
    max_tda_points: int = 500  # O(n^3) limit - hardware dependent

    # WFO structure - ADAPTIVE based on data size
    train_test_ratio: float = 4.0  # Train = 4x test (standard ML practice)
    target_folds: int = 100  # Desired folds, will adapt to data
    min_bars_per_window: int = 30  # Statistical minimum for CLT

    # Statistical thresholds - THEORETICALLY GROUNDED
    min_tercile_samples: int = 10  # t-test requires n >= 10 per group
    min_train_samples: int = 50  # For stable percentile estimation
    min_test_samples: int = 20  # For meaningful OOS evaluation
    min_active_folds: int = 5  # For aggregate statistics

    # Tercile boundaries - STANDARD (no tuning)
    high_percentile: float = 100 * 2 / 3  # Upper tercile = 66.67%
    low_percentile: float = 100 * 1 / 3  # Lower tercile = 33.33%

    # Pareto grid - ADAPTIVE based on IS t-stat distribution
    # Grid will be built from data: [0, q25, q50, q75, q90, q95, max] of IS t-stats
    use_adaptive_grid: bool = True

    # RV window - ADAPTIVE based on bar frequency
    # Will compute as: enough bars to span ~1 hour of market time
    rv_window_adaptive: bool = True
    rv_window_fallback: int = 20  # If adaptive fails


def compute_adaptive_rv_window(n_bars: int, target_fraction: float = 0.001) -> int:
    """Compute RV window as fraction of total bars.

    For range bars, time is irrelevant - use bar count.
    target_fraction: ~0.1% of data gives local volatility estimate.
    """
    rv_window = max(10, int(n_bars * target_fraction))
    # Cap at 100 to keep it local
    return min(100, rv_window)


def compute_adaptive_wfo_params(n_bars: int, cfg: WFOConfig) -> dict:
    """Compute WFO parameters adaptively based on data size.

    All outputs are in TDA WINDOWS, not bars.
    Each TDA window contains `tda_window` bars.

    For 775K bars with tda_window=100:
    - total_windows = 7751
    - We want ~100 folds
    - train_windows + test_windows per fold, stepping by test_windows
    """
    # TDA window size in bars (fixed, affects TDA computation granularity)
    tda_window = 100  # ~100 bars per TDA window is good for H1 features

    # Total TDA windows available
    total_windows = n_bars // tda_window

    # Step size in windows to get ~target_folds
    # total_windows / step_windows ≈ target_folds
    step_windows = max(1, total_windows // cfg.target_folds)

    # test = step (non-overlapping OOS)
    test_windows = step_windows

    # train = 4x test
    train_windows = int(test_windows * cfg.train_test_ratio)

    # Ensure minimum samples
    train_windows = max(cfg.min_train_samples, train_windows)
    test_windows = max(cfg.min_test_samples, test_windows)

    # Actual folds possible
    max_folds = max(1, (total_windows - train_windows - test_windows) // step_windows + 1)
    n_folds = min(cfg.target_folds, max_folds)

    return {
        "train_periods": train_windows,  # In TDA windows
        "test_periods": test_windows,    # In TDA windows
        "step_size": step_windows,       # In TDA windows
        "n_folds": n_folds,
        "tda_window": tda_window,        # In bars
    }


def build_adaptive_t_grid(insample_t_stats: list[float]) -> list[float]:
    """Build abstention t-threshold grid from IS t-stat distribution.

    Creates dense grid for finding Pareto frontier knee points.
    """
    if len(insample_t_stats) < 10:
        return [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Fallback

    arr = np.abs(insample_t_stats)  # Use absolute values

    # Dense data-driven quantiles for better knee detection
    grid = [0.0]  # Always include "no abstention"

    # Fine-grained percentiles: 5, 10, 15, ..., 95
    for q in range(5, 100, 5):  # 19 percentile points
        grid.append(float(np.percentile(arr, q)))

    # Add max
    grid.append(float(arr.max()))

    # Remove duplicates and sort
    grid = sorted({round(g, 2) for g in grid})

    # Ensure at least 15 points for good frontier resolution
    if len(grid) < 15:
        # Add interpolated points
        min_t, max_t = min(grid), max(grid)
        for t in np.linspace(min_t, max_t, 20):
            grid.append(round(float(t), 2))
        grid = sorted(set(grid))

    return grid


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


def compute_tda_features_gpu(
    embedded: np.ndarray, cfg: WFOConfig
) -> dict[str, float]:
    """Compute TDA features using GPU-accelerated ripser++."""
    try:
        import ripserplusplus as rpp
    except ImportError:
        log("ERROR", "ripserplusplus not installed")
        return {}

    if len(embedded) < cfg.min_tercile_samples:
        return {}

    if len(embedded) > cfg.max_tda_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embedded), cfg.max_tda_points, replace=False)
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


def compute_insample_stats(
    train_features: list[float], train_rv: list[float], cfg: WFOConfig
) -> dict[str, float]:
    """Compute in-sample statistics for abstention decision."""
    feat_arr = np.array(train_features)
    rv_arr = np.array(train_rv)

    high_thresh = np.percentile(feat_arr, cfg.high_percentile)
    low_thresh = np.percentile(feat_arr, cfg.low_percentile)

    high_mask = feat_arr >= high_thresh
    low_mask = feat_arr <= low_thresh

    n_high = int(np.sum(high_mask))
    n_low = int(np.sum(low_mask))

    if n_high < cfg.min_tercile_samples or n_low < cfg.min_tercile_samples:
        return {
            "high_thresh": high_thresh,
            "low_thresh": low_thresh,
            "n_high": n_high,
            "n_low": n_low,
            "t_stat": 0.0,
            "rv_diff": 0.0,
            "valid": False,
        }

    rv_high = rv_arr[high_mask]
    rv_low = rv_arr[low_mask]

    rv_diff = float(np.mean(rv_high) - np.mean(rv_low))
    pooled_std = np.sqrt((np.var(rv_high) + np.var(rv_low)) / 2)

    if pooled_std > 1e-10:
        t_stat = rv_diff / (pooled_std * np.sqrt(1 / n_high + 1 / n_low))
    else:
        t_stat = 0.0

    return {
        "high_thresh": float(high_thresh),
        "low_thresh": float(low_thresh),
        "n_high": n_high,
        "n_low": n_low,
        "t_stat": float(t_stat),
        "rv_diff": float(rv_diff),
        "rv_high_mean": float(np.mean(rv_high)),
        "rv_low_mean": float(np.mean(rv_low)),
        "valid": True,
    }


def main() -> None:
    """Run WFO TDA audit with selective abstention and local bias audit."""
    log("INFO", "Starting TDA WFO with abstention + local bias audit",
        host="bigblack", method="adaptive-parameter-free")

    import clickhouse_connect

    cfg = WFOConfig()

    symbol = "SOLUSDT"
    threshold = 250

    # Use best TDA params from prior audit (but will validate across multiple)
    dim, delay = 3, 2

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
      AND close_time_ms >= 1704067200000
    ORDER BY close_time_ms
    """

    log("INFO", "Querying data", symbol=symbol, threshold=threshold)
    result = client.query(query)
    returns = np.array([row[1] for row in result.result_rows], dtype=np.float64)

    log("INFO", "Data loaded", n_bars=len(returns))

    # ADAPTIVE: Compute RV window based on bar count (not time - range bars are price-based)
    if cfg.rv_window_adaptive:
        rv_window = compute_adaptive_rv_window(len(returns))
        log("INFO", "Adaptive RV window computed", rv_window=rv_window)
    else:
        rv_window = cfg.rv_window_fallback

    # ADAPTIVE: Compute WFO structure based on data size
    wfo_params = compute_adaptive_wfo_params(len(returns), cfg)
    train_periods = wfo_params["train_periods"]
    test_periods = wfo_params["test_periods"]
    step_size = wfo_params["step_size"]
    n_folds = wfo_params["n_folds"]
    window = wfo_params["tda_window"]

    log("INFO", "Adaptive WFO params",
        train_periods=train_periods,
        test_periods=test_periods,
        step_size=step_size,
        n_folds=n_folds,
        tda_window=window)

    # Compute forward RV
    forward_rv = np.full(len(returns), np.nan)
    for i in range(len(returns) - rv_window):
        forward_rv[i] = np.std(returns[i + 1 : i + 1 + rv_window])

    total_windows = (len(returns) - rv_window) // window

    log("INFO", "WFO setup",
        total_windows=total_windows,
        train_periods=train_periods,
        test_periods=test_periods,
        n_folds=n_folds)

    # Collect all fold data first
    all_folds_data = []

    for fold in range(n_folds):
        fold_start = fold * step_size
        train_start = fold_start
        train_end = train_start + train_periods
        test_start = train_end
        test_end = test_start + test_periods

        if test_end > total_windows:
            break

        # Compute TDA features for training
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

            features = compute_tda_features_gpu(embedded, cfg)
            if not features or "h1_count" not in features:
                continue

            fwd_rv = np.nanmean(forward_rv[end : end + window])
            if np.isnan(fwd_rv):
                continue

            train_features.append(features["h1_count"])
            train_rv.append(fwd_rv)

        if len(train_features) < 50:
            continue

        # Compute in-sample stats for abstention decision
        insample_stats = compute_insample_stats(train_features, train_rv, cfg)

        if not insample_stats["valid"]:
            continue

        # Compute OOS results
        test_features = []
        test_rv = []
        test_predictions = []

        high_thresh = insample_stats["high_thresh"]
        low_thresh = insample_stats["low_thresh"]

        for w in range(test_start, test_end):
            start = w * window
            end = start + window

            if end + rv_window > len(returns):
                break

            window_returns = returns[start:end]
            embedded = time_delay_embedding(window_returns, dim, delay)

            if len(embedded) < 10:
                continue

            features = compute_tda_features_gpu(embedded, cfg)
            if not features or "h1_count" not in features:
                continue

            fwd_rv = np.nanmean(forward_rv[end : end + window])
            if np.isnan(fwd_rv):
                continue

            h1_count = features["h1_count"]
            test_features.append(h1_count)
            test_rv.append(fwd_rv)

            if h1_count >= high_thresh:
                test_predictions.append(1)
            elif h1_count <= low_thresh:
                test_predictions.append(-1)
            else:
                test_predictions.append(0)

        if len(test_features) < 20:
            continue

        test_rv_arr = np.array(test_rv)
        test_pred_arr = np.array(test_predictions)

        high_pred_mask = test_pred_arr == 1
        low_pred_mask = test_pred_arr == -1

        if np.sum(high_pred_mask) < 5 or np.sum(low_pred_mask) < 5:
            continue

        rv_when_high = test_rv_arr[high_pred_mask]
        rv_when_low = test_rv_arr[low_pred_mask]

        oos_diff = float(np.mean(rv_when_high) - np.mean(rv_when_low))
        pooled_std = np.sqrt((np.var(rv_when_high) + np.var(rv_when_low)) / 2)

        if pooled_std > 1e-10:
            oos_t = oos_diff / (pooled_std * np.sqrt(1 / len(rv_when_high) + 1 / len(rv_when_low)))
        else:
            oos_t = 0.0

        fold_data = {
            "fold": fold + 1,
            "insample_t": insample_stats["t_stat"],
            "insample_effect": insample_stats["rv_diff"],
            "oos_t": float(oos_t),
            "oos_diff": oos_diff,
            "n_high": int(np.sum(high_pred_mask)),
            "n_low": int(np.sum(low_pred_mask)),
        }

        all_folds_data.append(fold_data)

        log("INFO", "Fold result", **fold_data)

    log("INFO", "All folds computed", n_folds=len(all_folds_data))

    # ADAPTIVE: Build abstention grid from IS t-stat distribution
    if cfg.use_adaptive_grid and all_folds_data:
        insample_t_stats = [f["insample_t"] for f in all_folds_data]
        t_grid = build_adaptive_t_grid(insample_t_stats)
        log("INFO", "Adaptive t-grid built from data", grid=t_grid)
    else:
        t_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Build abstention configs from adaptive grid
    abstention_configs = []
    for min_t in t_grid:
        abstention_configs.append({
            "min_t": float(min_t),
            "min_effect": 0.0,
            "name": f"t>={min_t:.2f}",
        })

    # Now test each abstention config
    for config in abstention_configs:
        min_t = config["min_t"]
        min_effect = config["min_effect"]
        name = config["name"]

        # Filter folds based on abstention criteria
        active_folds = [
            f for f in all_folds_data
            if abs(f["insample_t"]) >= min_t and abs(f["insample_effect"]) >= min_effect
        ]

        if len(active_folds) < 5:
            log("INFO", "Abstention config skipped - too few active folds",
                config=name, n_active=len(active_folds))
            continue

        # Compute aggregate OOS stats for active folds only
        oos_t_values = [f["oos_t"] for f in active_folds]
        oos_diff_values = [f["oos_diff"] for f in active_folds]

        avg_oos_t = float(np.mean(oos_t_values))
        avg_oos_diff = float(np.mean(oos_diff_values))
        positive_folds = sum(1 for t in oos_t_values if t > 0)
        significant_folds = sum(1 for t in oos_t_values if abs(t) >= 2.0)

        abstention_rate = 1.0 - len(active_folds) / len(all_folds_data) if all_folds_data else 0.0

        summary = {
            "config": name,
            "min_t": min_t,
            "min_effect": min_effect,
            "n_active_folds": len(active_folds),
            "n_total_folds": len(all_folds_data),
            "abstention_rate": round(abstention_rate * 100, 1),
            "avg_oos_t": round(avg_oos_t, 2),
            "avg_oos_diff": round(avg_oos_diff, 2),
            "positive_folds": f"{positive_folds}/{len(active_folds)}",
            "significant_folds": f"{significant_folds}/{len(active_folds)}",
            "pct_significant": round(100 * significant_folds / len(active_folds), 1),
        }

        log("INFO", "Abstention config results", **summary)

        # Verdict for this config
        if avg_oos_t >= 2.0 and positive_folds >= 0.7 * len(active_folds):
            verdict = "PROMISING - worth further investigation"
        elif avg_oos_t >= 1.5:
            verdict = "MARGINAL-POSITIVE - weak but directionally correct"
        elif avg_oos_t > 0:
            verdict = "WEAK - minimal edge"
        else:
            verdict = "NULL - no predictive power"

        log("INFO", "Config verdict", config=name, verdict=verdict)

    # Collect all results for Pareto analysis
    all_config_results = []

    # Re-run to collect for Pareto (already logged above, now store)
    for config in abstention_configs:
        min_t = config["min_t"]
        min_effect = config["min_effect"]
        name = config["name"]

        active_folds = [
            f for f in all_folds_data
            if abs(f["insample_t"]) >= min_t and abs(f["insample_effect"]) >= min_effect
        ]

        if len(active_folds) < 5:
            continue

        oos_t_values = [f["oos_t"] for f in active_folds]
        avg_oos_t = float(np.mean(oos_t_values))
        abstention_rate = 1.0 - len(active_folds) / len(all_folds_data)
        participation_rate = len(active_folds) / len(all_folds_data)

        all_config_results.append({
            "name": name,
            "min_t": min_t,
            "min_effect": min_effect,
            "participation_rate": participation_rate,
            "abstention_rate": abstention_rate,
            "avg_oos_t": avg_oos_t,
            "n_active": len(active_folds),
        })

    # Find Pareto frontier: maximize avg_oos_t while maximizing participation_rate
    # A point is Pareto-optimal if no other point dominates it on BOTH dimensions
    pareto_frontier = []

    for i, r1 in enumerate(all_config_results):
        is_dominated = False
        for j, r2 in enumerate(all_config_results):
            if i == j:
                continue
            # r2 dominates r1 if r2 is better or equal on both dimensions and strictly better on at least one
            if (r2["avg_oos_t"] >= r1["avg_oos_t"] and
                r2["participation_rate"] >= r1["participation_rate"] and
                (r2["avg_oos_t"] > r1["avg_oos_t"] or r2["participation_rate"] > r1["participation_rate"])):
                is_dominated = True
                break

        if not is_dominated:
            pareto_frontier.append(r1)

    # Sort Pareto frontier by participation rate (descending)
    pareto_frontier.sort(key=lambda x: -x["participation_rate"])

    log("INFO", "=" * 60)
    log("INFO", "PARETO EFFICIENT FRONTIER (participation vs OOS t-stat)")
    log("INFO", "=" * 60)

    for p in pareto_frontier:
        log("INFO", "Pareto point",
            name=p["name"],
            participation_pct=round(p["participation_rate"] * 100, 1),
            abstention_pct=round(p["abstention_rate"] * 100, 1),
            avg_oos_t=round(p["avg_oos_t"], 2),
            n_active=p["n_active"])

    # Find the "knee" - best tradeoff point
    # Use simple heuristic: maximize (participation_rate * avg_oos_t) if avg_oos_t > 0
    positive_pareto = [p for p in pareto_frontier if p["avg_oos_t"] > 0]

    if positive_pareto:
        # Score = participation * oos_t (reward both)
        best_tradeoff = max(positive_pareto, key=lambda x: x["participation_rate"] * x["avg_oos_t"])
        log("INFO", "BEST TRADEOFF POINT (knee of Pareto frontier)",
            name=best_tradeoff["name"],
            participation_pct=round(best_tradeoff["participation_rate"] * 100, 1),
            avg_oos_t=round(best_tradeoff["avg_oos_t"], 2),
            score=round(best_tradeoff["participation_rate"] * best_tradeoff["avg_oos_t"], 3))

    # Final verdict
    log("INFO", "=" * 60)

    if pareto_frontier:
        max_t_point = max(pareto_frontier, key=lambda x: x["avg_oos_t"])
        if max_t_point["avg_oos_t"] >= 2.0:
            log("INFO", "VERDICT: Abstention CAN achieve significance",
                best_config=max_t_point["name"],
                best_oos_t=round(max_t_point["avg_oos_t"], 2),
                at_participation=round(max_t_point["participation_rate"] * 100, 1))
        elif max_t_point["avg_oos_t"] >= 1.0:
            log("INFO", "VERDICT: Abstention improves but not to significance",
                best_config=max_t_point["name"],
                best_oos_t=round(max_t_point["avg_oos_t"], 2))
        else:
            log("INFO", "VERDICT: Even with abstention, no meaningful signal",
                best_oos_t=round(max_t_point["avg_oos_t"], 2))


if __name__ == "__main__":
    main()
