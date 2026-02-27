#!/usr/bin/env python3
"""True GPU-Accelerated TDA using Ripser++.

Ripser++ achieves up to 30x speedup over CPU ripser by using GPU for
the core persistence computation (not just distance matrix).

Prerequisites on littleblack:
    # Our fork with CUDA 12.x and Python 3.13 compatibility:
    git clone https://github.com/terrylica/ripser-plusplus /tmp/ripser-plusplus
    cd /tmp/ripser-plusplus
    uv venv tda_env_py313 --python 3.13
    source tda_env_py313/bin/activate
    uv pip install setuptools wheel cmake numpy scipy polars pyarrow
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    python3 setup.py build_ext --inplace

Usage:
    # Export data from bigblack first:
    ssh bigblack "clickhouse-client --query \"SELECT * FROM rangebar_cache.range_bars
        WHERE symbol = 'BTCUSDT' AND threshold_decimal_bps = 100
        AND ouroboros_mode = 'year' ORDER BY close_time_ms\"
        --format Parquet" > /tmp/btcusdt_100.parquet

    # Run on littleblack:
    scp /tmp/btcusdt_100.parquet littleblack:/tmp/
    ssh littleblack "source /tmp/tda_env_py313/bin/activate && \\
        export LD_LIBRARY_PATH=/tmp/ripser-plusplus/ripserplusplus:\\$LD_LIBRARY_PATH && \\
        PYTHONPATH=/tmp/ripser-plusplus python3 /tmp/tda_ripser_plusplus.py /tmp/btcusdt_100.parquet"

Issue #56: TDA Structural Break Detection for Range Bar Patterns
"""

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Try to import Ripser++
RIPSER_PP_AVAILABLE = False
try:
    import ripserplusplus as rpp

    RIPSER_PP_AVAILABLE = True
except ImportError:
    print("WARNING: ripserplusplus not available. See docstring for installation.")


def log_info(message: str, **kwargs: object) -> None:
    """Log info message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))


def takens_embedding(x: np.ndarray, dim: int = 3, delay: int = 1) -> np.ndarray:
    """Create Takens delay embedding of time series."""
    n = len(x) - (dim - 1) * delay
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * delay : i * delay + n]
    return embedded


def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(points, metric="euclidean")).astype(np.float32)


def persistence_l2_norm(dgm: np.ndarray) -> float:
    """Compute L2 norm of persistence diagram."""
    if len(dgm) == 0:
        return 0.0
    # dgm is structured array with 'birth' and 'death' fields
    if dgm.dtype.names:
        persistence = dgm["death"] - dgm["birth"]
    else:
        persistence = dgm[:, 1] - dgm[:, 0]
    # Filter out infinite persistence
    finite_mask = np.isfinite(persistence)
    persistence = persistence[finite_mask]
    return float(np.sqrt(np.sum(persistence**2)))


def analyze_window_ripser_pp(
    returns: np.ndarray,
    embedding_dim: int = 3,
    embedding_delay: int = 1,
    max_dim: int = 1,
) -> dict:
    """Analyze single window with Ripser++ (GPU-accelerated)."""
    # Takens embedding
    points = takens_embedding(returns, dim=embedding_dim, delay=embedding_delay)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(points)

    # Run Ripser++ on distance matrix (GPU-accelerated!)
    result = rpp.run(f"--format distance --dim {max_dim}", dist_matrix)

    # Compute L2 norms for each dimension
    l2_norms = []
    for dim in range(max_dim + 1):
        dgm = result.get(dim, np.array([]))
        l2 = persistence_l2_norm(dgm)
        l2_norms.append(l2)

    return {
        "h0_l2": l2_norms[0] if len(l2_norms) > 0 else 0.0,
        "h1_l2": l2_norms[1] if len(l2_norms) > 1 else 0.0,
        "total_l2": sum(l2_norms),
    }


def load_parquet_data(filepath: str) -> tuple:
    """Load range bar data from Parquet file."""
    import polars as pl

    df = pl.read_parquet(filepath)
    log_info("Loaded data", rows=len(df), columns=list(df.columns))

    symbol = df["symbol"][0] if "symbol" in df.columns else "UNKNOWN"
    threshold = df["threshold_decimal_bps"][0] if "threshold_decimal_bps" in df.columns else 0

    close = df["close"].to_numpy()
    log_returns = np.diff(np.log(close))

    valid_mask = np.isfinite(log_returns)
    log_returns = log_returns[valid_mask]

    log_info(
        "Computed log returns",
        n_returns=len(log_returns),
        symbol=symbol,
        threshold=int(threshold),
    )

    return symbol, int(threshold), log_returns


def analyze_tda_ripser_pp(
    returns: np.ndarray,
    window_size: int = 200,
    step_size: int = 500,
    subsample: int = 100,
) -> dict:
    """Run TDA analysis using Ripser++ (true GPU acceleration)."""
    if not RIPSER_PP_AVAILABLE:
        return {"error": "ripserplusplus not available"}

    n = len(returns)
    n_windows = (n - window_size) // step_size + 1

    log_info(
        "Starting Ripser++ TDA analysis",
        n_returns=n,
        window_size=window_size,
        step_size=step_size,
        subsample=subsample,
        n_windows=n_windows,
    )

    start_time = time.time()
    l2_norms = []

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = returns[start_idx:end_idx]

        # Subsample if needed
        if len(window) > subsample:
            indices = np.linspace(0, len(window) - 1, subsample, dtype=int)
            window = window[indices]

        window_start = time.time()
        result = analyze_window_ripser_pp(window)
        window_elapsed = time.time() - window_start

        l2_norms.append(result["total_l2"])

        # Progress logging every 100 windows
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_windows - i - 1)
            log_info(
                "Progress",
                window=i + 1,
                total=n_windows,
                pct=round((i + 1) / n_windows * 100, 1),
                elapsed_sec=round(elapsed, 1),
                eta_sec=round(eta, 1),
                last_window_ms=round(window_elapsed * 1000, 1),
            )

    l2_norms = np.array(l2_norms)
    l2_velocity = np.diff(l2_norms)

    # Detect breaks (velocity > 95th percentile)
    if len(l2_velocity) > 0:
        threshold_val = np.percentile(np.abs(l2_velocity), 95)
        breaks = np.where(np.abs(l2_velocity) > threshold_val)[0]
    else:
        breaks = np.array([])

    elapsed = time.time() - start_time
    log_info("TDA analysis complete", elapsed_sec=round(elapsed, 2), n_breaks=len(breaks))

    return {
        "n_windows": n_windows,
        "avg_l2_norm": float(np.mean(l2_norms)),
        "max_l2_norm": float(np.max(l2_norms)),
        "std_l2_norm": float(np.std(l2_norms)),
        "n_breaks": len(breaks),
        "break_indices": breaks.tolist(),
        "elapsed_sec": round(elapsed, 2),
        "windows_per_sec": round(n_windows / elapsed, 2),
    }


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python tda_ripser_plusplus.py <parquet_file>")
        print("\nSee docstring for full setup instructions.")
        sys.exit(1)

    filepath = sys.argv[1]
    log_info("Starting Ripser++ GPU TDA analysis", filepath=filepath)

    if not RIPSER_PP_AVAILABLE:
        log_info("ERROR: ripserplusplus not available")
        sys.exit(1)

    # Load data
    symbol, threshold, returns = load_parquet_data(filepath)

    # Run TDA analysis
    result = analyze_tda_ripser_pp(returns)

    print("\n" + "=" * 80)
    print("RIPSER++ GPU TDA RESULTS")
    print("=" * 80)
    print(f"Symbol: {symbol} @ {threshold} dbps")
    print("Library: Ripser++ (true GPU acceleration)")
    print(f"Returns: {len(returns):,}")
    print(f"Windows: {result.get('n_windows', 0)}")
    print(f"Throughput: {result.get('windows_per_sec', 0):.1f} windows/sec")
    print(f"Avg L2 Norm: {result.get('avg_l2_norm', 0):.4f}")
    print(f"Max L2 Norm: {result.get('max_l2_norm', 0):.4f}")
    print(f"TDA Breaks: {result.get('n_breaks', 0)}")
    print(f"Break Indices: {result.get('break_indices', [])[:10]}...")
    print(f"Total Time: {result.get('elapsed_sec', 0):.1f}s")
    print("=" * 80)

    log_info("Analysis complete", symbol=symbol, threshold=threshold, **result)


if __name__ == "__main__":
    main()
