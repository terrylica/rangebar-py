#!/usr/bin/env python3
"""CuPy-Accelerated TDA Structural Break Detection.

Uses CuPy for GPU-accelerated distance matrix computation,
then feeds to ripser for persistent homology.

The bottleneck in TDA is O(n²) pairwise distance computation.
CuPy can compute this on GPU ~50-100x faster than NumPy on CPU.

Prerequisites on littleblack (CUDA 12.4):
    uv venv tda_env --python 3.11
    source tda_env/bin/activate
    uv pip install cupy-cuda12x ripser polars pyarrow numpy persim

Usage:
    ssh bigblack "clickhouse-client --query \"SELECT * FROM rangebar_cache.range_bars
        WHERE symbol = 'BTCUSDT' AND threshold_decimal_bps = 100
        AND ouroboros_mode = 'year' ORDER BY close_time_ms\"
        --format Parquet" > /tmp/btcusdt_100.parquet

    scp /tmp/btcusdt_100.parquet littleblack:/tmp/
    ssh littleblack "cd /tmp && source tda_env/bin/activate && python3 tda_cupy_accelerated.py /tmp/btcusdt_100.parquet"

Issue #56: TDA Structural Break Detection for Range Bar Patterns
"""

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Check for GPU support
CUPY_AVAILABLE = False
GPU_NAME = None
try:
    import cupy as cp

    # Get device name via CUDA runtime
    device = cp.cuda.Device(0)
    GPU_NAME = cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
    CUPY_AVAILABLE = True
except ImportError:
    pass
except RuntimeError:
    # CUDA not available or device not found
    pass

try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


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
    """Create Takens delay embedding of time series.

    Args:
        x: 1D time series
        dim: Embedding dimension
        delay: Time delay

    Returns:
        2D array of shape (n_points, dim)
    """
    n = len(x) - (dim - 1) * delay
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * delay : i * delay + n]
    return embedded


def compute_distance_matrix_gpu(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix using CuPy on GPU.

    This is the O(n²) bottleneck - GPU gives ~50-100x speedup.

    Args:
        points: 2D array of shape (n_points, n_dims)

    Returns:
        Distance matrix of shape (n_points, n_points)
    """
    if not CUPY_AVAILABLE:
        # Fallback to CPU
        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(points, metric="euclidean"))

    # Transfer to GPU
    points_gpu = cp.asarray(points, dtype=cp.float32)

    # Compute squared distances: ||a-b||² = ||a||² + ||b||² - 2<a,b>
    sq_norms = cp.sum(points_gpu**2, axis=1, keepdims=True)
    distances_sq = sq_norms + sq_norms.T - 2.0 * cp.dot(points_gpu, points_gpu.T)

    # Clamp negative values (numerical errors) and take sqrt
    distances_sq = cp.maximum(distances_sq, 0.0)
    distances = cp.sqrt(distances_sq)

    # Transfer back to CPU
    return cp.asnumpy(distances)


def compute_persistence_landscape_l2(dgm: np.ndarray, n_steps: int = 100) -> float:
    """Compute L2 norm of persistence landscape.

    Args:
        dgm: Persistence diagram array
        n_steps: Resolution for landscape

    Returns:
        L2 norm of the landscape
    """
    if len(dgm) == 0:
        return 0.0

    # Filter out infinite points and NaN
    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    dgm = dgm[finite_mask]

    if len(dgm) == 0:
        return 0.0

    # Compute persistence (death - birth)
    persistence = dgm[:, 1] - dgm[:, 0]
    # L2 norm is sqrt(sum of squares of persistence)
    return float(np.sqrt(np.sum(persistence**2)))


def analyze_window_tda(
    returns: np.ndarray,
    embedding_dim: int = 3,
    embedding_delay: int = 1,
    max_dim: int = 1,
) -> dict:
    """Analyze single window with TDA.

    Args:
        returns: Return series for this window
        embedding_dim: Takens embedding dimension
        embedding_delay: Takens embedding delay
        max_dim: Maximum homology dimension

    Returns:
        Dictionary with TDA metrics
    """
    # Takens embedding
    points = takens_embedding(returns, dim=embedding_dim, delay=embedding_delay)

    # GPU-accelerated distance matrix
    dist_matrix = compute_distance_matrix_gpu(points)

    # Run ripser on distance matrix
    result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)

    # Compute L2 norms for each dimension
    l2_norms = []
    for dim in range(max_dim + 1):
        dgm = result["dgms"][dim]
        l2 = compute_persistence_landscape_l2(dgm)
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


def analyze_tda_sliding_window(
    returns: np.ndarray,
    window_size: int = 200,
    step_size: int = 500,
    subsample: int = 100,
) -> dict:
    """Run TDA analysis over sliding windows.

    Args:
        returns: Full return series
        window_size: Size of each window
        step_size: Step between windows
        subsample: Subsample window to this many points (reduces ripser complexity)

    Returns:
        Dictionary with TDA results
    """
    if not RIPSER_AVAILABLE:
        return {"error": "ripser not available"}

    n = len(returns)
    n_windows = (n - window_size) // step_size + 1

    log_info(
        "Starting TDA analysis",
        n_returns=n,
        window_size=window_size,
        step_size=step_size,
        subsample=subsample,
        n_windows=n_windows,
        gpu_available=CUPY_AVAILABLE,
        gpu_name=GPU_NAME,
    )

    start_time = time.time()
    l2_norms = []

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window = returns[start:end]

        # Subsample if window is larger than subsample size
        if len(window) > subsample:
            indices = np.linspace(0, len(window) - 1, subsample, dtype=int)
            window = window[indices]

        window_start = time.time()
        result = analyze_window_tda(window)
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
        print("Usage: python tda_cupy_accelerated.py <parquet_file>")
        print("\nExport data from bigblack first:")
        print(
            '  ssh bigblack "clickhouse-client --query \'SELECT * FROM rangebar_cache.range_bars'
        )
        print("      WHERE symbol = 'BTCUSDT' AND threshold_decimal_bps = 100")
        print("      AND ouroboros_mode = 'year' ORDER BY close_time_ms'")
        print('      --format Parquet" > /tmp/btcusdt_100.parquet')
        sys.exit(1)

    filepath = sys.argv[1]
    log_info("Starting CuPy-accelerated TDA analysis", filepath=filepath)

    # Report GPU status
    if CUPY_AVAILABLE:
        log_info("GPU available", device=GPU_NAME)
        # Warm up GPU
        _ = cp.zeros(1000)
    else:
        log_info("CuPy not available, using CPU fallback")

    # Load data
    symbol, threshold, returns = load_parquet_data(filepath)

    # Run TDA analysis
    result = analyze_tda_sliding_window(returns)

    print("\n" + "=" * 80)
    print("TDA STRUCTURAL BREAK RESULTS")
    print("=" * 80)
    print(f"Symbol: {symbol} @ {threshold} dbps")
    print(f"GPU: {GPU_NAME if CUPY_AVAILABLE else 'None (CPU mode)'}")
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
