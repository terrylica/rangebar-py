#!/usr/bin/env python3
"""GPU-Accelerated TDA Structural Break Detection.

Uses giotto-tda with CUDA backend for fast persistent homology computation.
Designed to run on littleblack GPU workstation (RTX 2080 Ti).

Prerequisites on littleblack:
    uv add giotto-tda polars numpy

Usage:
    # Export data from bigblack first:
    ssh bigblack "clickhouse-client --query 'SELECT * FROM rangebar_cache.range_bars
        WHERE symbol = \"BTCUSDT\" AND threshold_decimal_bps = 100
        AND ouroboros_mode = \"year\" ORDER BY timestamp_ms'
        --format Parquet" > /tmp/btcusdt_100.parquet

    # Copy to littleblack and run:
    scp /tmp/btcusdt_100.parquet littleblack:/tmp/
    ssh littleblack "python3 tda_gpu_analysis.py /tmp/btcusdt_100.parquet"

Issue #56: TDA Structural Break Detection for Range Bar Patterns
"""

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Check for GPU support
try:
    from gtda.diagrams import Amplitude
    from gtda.homology import VietorisRipsPersistence
    from gtda.time_series import TakensEmbedding

    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    print("WARNING: giotto-tda not available. Install with: uv add giotto-tda")


def log_info(message: str, **kwargs: object) -> None:
    """Log info message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))


def check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            log_info("GPU available", device=device_name)
            return True
        log_info("No CUDA GPU available, using CPU")
        return False
    except ImportError:
        log_info("PyTorch not installed, cannot check GPU")
        return False


def load_parquet_data(filepath: str) -> tuple:
    """Load range bar data from Parquet file.

    Args:
        filepath: Path to Parquet file

    Returns:
        Tuple of (symbol, threshold, log_returns array)
    """
    import polars as pl

    df = pl.read_parquet(filepath)
    log_info("Loaded data", rows=len(df), columns=list(df.columns))

    # Extract symbol and threshold from data
    symbol = df["symbol"][0] if "symbol" in df.columns else "UNKNOWN"
    threshold = df["threshold_decimal_bps"][0] if "threshold_decimal_bps" in df.columns else 0

    # Compute log returns
    close = df["close"].to_numpy()
    log_returns = np.diff(np.log(close))

    # Remove any NaN/Inf
    valid_mask = np.isfinite(log_returns)
    log_returns = log_returns[valid_mask]

    log_info("Computed log returns", n_returns=len(log_returns), symbol=symbol, threshold=int(threshold))

    return symbol, int(threshold), log_returns


def analyze_tda_gpu(
    returns: np.ndarray,
    window_size: int = 500,
    step_size: int = 100,
    embedding_dim: int = 3,
    embedding_delay: int = 1,
) -> dict:
    """Run GPU-accelerated TDA analysis using giotto-tda.

    Args:
        returns: Array of log returns
        window_size: Size of sliding windows
        step_size: Step between windows
        embedding_dim: Takens embedding dimension
        embedding_delay: Takens embedding delay

    Returns:
        Dictionary with TDA results
    """
    if not GTDA_AVAILABLE:
        return {"error": "giotto-tda not available"}

    n = len(returns)
    log_info("Starting GPU TDA analysis", n_returns=n, window_size=window_size)

    start_time = time.time()

    # Create sliding windows
    n_windows = (n - window_size) // step_size + 1
    windows = []
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows.append(returns[start:end])

    windows = np.array(windows)
    log_info("Created sliding windows", n_windows=len(windows))

    # Takens embedding
    embedder = TakensEmbedding(
        time_delay=embedding_delay,
        dimension=embedding_dim,
        flatten=False,
    )
    point_clouds = embedder.fit_transform(windows)
    log_info("Takens embedding complete", shape=list(point_clouds.shape))

    # Vietoris-Rips persistence (GPU-accelerated if available)
    vr = VietorisRipsPersistence(
        homology_dimensions=[0, 1],  # H0 and H1
        n_jobs=-1,  # Use all cores
    )
    diagrams = vr.fit_transform(point_clouds)
    log_info("Persistence diagrams computed", shape=list(diagrams.shape))

    # Compute persistence amplitudes (L2 norm of persistence)
    amplitude = Amplitude(metric="persistence")
    amplitudes = amplitude.fit_transform(diagrams)
    log_info("Landscape amplitudes computed", shape=list(amplitudes.shape))

    # Compute statistics
    l2_norms = np.linalg.norm(amplitudes, axis=1)
    l2_velocity = np.diff(l2_norms)

    # Detect breaks (velocity > 95th percentile)
    if len(l2_velocity) > 0:
        threshold = np.percentile(np.abs(l2_velocity), 95)
        breaks = np.where(np.abs(l2_velocity) > threshold)[0]
    else:
        breaks = np.array([])

    elapsed = time.time() - start_time
    log_info("TDA analysis complete", elapsed_sec=round(elapsed, 2), n_breaks=len(breaks))

    return {
        "n_windows": len(windows),
        "avg_l2_norm": float(np.mean(l2_norms)),
        "max_l2_norm": float(np.max(l2_norms)),
        "std_l2_norm": float(np.std(l2_norms)),
        "n_breaks": len(breaks),
        "break_indices": breaks.tolist(),
        "elapsed_sec": round(elapsed, 2),
    }


def analyze_pattern_tda_gpu(
    returns: np.ndarray,
    patterns: list[str] | None = None,
) -> list[dict]:
    """Analyze TDA for each 3-bar pattern.

    Args:
        returns: Full return series
        patterns: List of patterns to analyze (default: all 8)

    Returns:
        List of pattern TDA results
    """
    if patterns is None:
        patterns = ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"]

    results = []

    # Classify each return into U/D
    directions = np.where(returns > 0, "U", "D")

    # Build 3-bar patterns
    n = len(directions)
    pattern_labels = np.array([
        directions[i-2] + directions[i-1] + directions[i]
        for i in range(2, n)
    ])

    for pattern in patterns:
        # Find indices where this pattern occurs
        pattern_mask = pattern_labels == pattern
        pattern_indices = np.where(pattern_mask)[0] + 2  # Offset for pattern construction

        # Get forward returns after pattern
        forward_indices = pattern_indices + 1
        valid_mask = forward_indices < len(returns)
        forward_returns = returns[forward_indices[valid_mask]]

        if len(forward_returns) < 500:
            log_info("Skipping pattern", pattern=pattern, n=len(forward_returns), reason="insufficient samples")
            continue

        log_info("Analyzing pattern", pattern=pattern, n=len(forward_returns))

        # Run TDA on pattern-specific returns
        tda_result = analyze_tda_gpu(
            forward_returns,
            window_size=100,
            step_size=50,
        )

        results.append({
            "pattern": pattern,
            "n_samples": len(forward_returns),
            **tda_result,
        })

    return results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python tda_gpu_analysis.py <parquet_file>")
        print("\nExport data from bigblack first:")
        print('  ssh bigblack "clickhouse-client --query \'SELECT * FROM rangebar_cache.range_bars')
        print('      WHERE symbol = "BTCUSDT" AND threshold_decimal_bps = 100')
        print('      AND ouroboros_mode = "year" ORDER BY timestamp_ms\'')
        print('      --format Parquet" > /tmp/btcusdt_100.parquet')
        sys.exit(1)

    filepath = sys.argv[1]
    log_info("Starting GPU TDA analysis", filepath=filepath)

    # Check GPU availability
    has_gpu = check_gpu()

    # Load data
    symbol, threshold, returns = load_parquet_data(filepath)

    # Overall TDA analysis
    log_info("Running overall TDA analysis")
    overall_result = analyze_tda_gpu(returns)

    print("\n" + "=" * 80)
    print("OVERALL TDA RESULTS")
    print("=" * 80)
    print(f"Symbol: {symbol} @ {threshold} dbps (GPU: {has_gpu})")
    print(f"Returns: {len(returns):,}")
    print(f"Windows: {overall_result.get('n_windows', 0)}")
    print(f"Avg L2 Norm: {overall_result.get('avg_l2_norm', 0):.4f}")
    print(f"Max L2 Norm: {overall_result.get('max_l2_norm', 0):.4f}")
    print(f"TDA Breaks: {overall_result.get('n_breaks', 0)}")
    print(f"Elapsed: {overall_result.get('elapsed_sec', 0):.2f}s")
    print("=" * 80)

    # Pattern-specific TDA (if enough data)
    if len(returns) > 10000:
        log_info("Running pattern-specific TDA analysis")
        pattern_results = analyze_pattern_tda_gpu(returns)

        if pattern_results:
            print("\n" + "=" * 80)
            print("PATTERN-SPECIFIC TDA RESULTS")
            print("=" * 80)
            print(f"{'Pattern':<10} {'N':>10} {'Avg L2':>10} {'Max L2':>10} {'Breaks':>8}")
            print("-" * 80)

            for r in sorted(pattern_results, key=lambda x: x.get("avg_l2_norm", 0), reverse=True):
                print(
                    f"{r['pattern']:<10} "
                    f"{r['n_samples']:>10,} "
                    f"{r.get('avg_l2_norm', 0):>10.4f} "
                    f"{r.get('max_l2_norm', 0):>10.4f} "
                    f"{r.get('n_breaks', 0):>8}"
                )

            print("=" * 80)
    else:
        log_info("Skipping pattern analysis", reason="insufficient data", n=len(returns))

    log_info("Analysis complete", symbol=symbol, threshold=threshold)


if __name__ == "__main__":
    main()
