#!/usr/bin/env python3
"""Example: Using ClickHouse cache for range bar computation.

This example demonstrates how to use the two-tier ClickHouse caching layer
for efficient range bar computation across multiple runs.

Prerequisites:
- ClickHouse running locally OR configured via mise environment
- mise env vars for remote hosts (optional):
  - RANGEBAR_CH_HOSTS: Comma-separated SSH aliases from ~/.ssh/config
  - RANGEBAR_CH_PRIMARY: Default host for caching

Benefits:
- Tier 1: Raw trades cached (avoid re-downloading from exchange)
- Tier 2: Computed range bars cached (avoid re-computing)
- Second run uses cached results (near-instant)
"""

from __future__ import annotations

import time

import pandas as pd

# Check if ClickHouse is available before running
try:
    from rangebar import (
        ClickHouseNotConfiguredError,
        detect_clickhouse_state,
        get_available_clickhouse_host,
        process_trades_to_dataframe,
        process_trades_to_dataframe_cached,
    )
    from rangebar.clickhouse import InstallationLevel, RangeBarCache
except ImportError as e:
    print(f"Error importing rangebar: {e}")
    print("Make sure rangebar is installed: pip install rangebar")
    raise SystemExit(1)


def generate_synthetic_trades(num_trades: int = 10000) -> pd.DataFrame:
    """Generate synthetic trade data for testing.

    Parameters
    ----------
    num_trades : int
        Number of trades to generate

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp, price, quantity columns
    """
    import numpy as np

    base_timestamp = 1704067200000  # 2024-01-01 00:00:00 UTC
    base_price = 42000.0

    # Generate random price movements (random walk)
    np.random.seed(42)  # For reproducibility
    returns = np.random.randn(num_trades) * 0.001  # 0.1% std dev
    prices = base_price * np.cumprod(1 + returns)

    # Generate random volumes
    volumes = np.random.exponential(1.0, num_trades)

    return pd.DataFrame(
        {
            "timestamp": base_timestamp
            + np.arange(num_trades) * 100,  # 100ms intervals
            "price": prices,
            "quantity": volumes,
        }
    )


def check_preflight() -> bool:
    """Check ClickHouse availability and print status.

    Returns
    -------
    bool
        True if ClickHouse is available for caching
    """
    print("=" * 70)
    print("ClickHouse Preflight Check")
    print("=" * 70)

    # Check installation state
    state = detect_clickhouse_state()
    print(f"\nInstallation Level: {state.level.name}")
    print(f"Message: {state.message}")

    if state.version:
        print(f"Version: {state.version}")
    if state.binary_path:
        print(f"Binary: {state.binary_path}")
    if state.action_required:
        print(f"\nAction Required:\n{state.action_required}")

    # Check if we can proceed
    if state.level < InstallationLevel.RUNNING_NO_SCHEMA:
        print("\n❌ ClickHouse not available for caching")
        return False

    # Try to get available host
    try:
        host = get_available_clickhouse_host()
        print(f"\n✅ Available Host: {host.host}")
        print(f"   Connection Method: {host.method}")
        print(f"   Port: {host.port}")
        return True
    except ClickHouseNotConfiguredError:
        print("\n❌ No ClickHouse hosts available")
        return False


def example_without_cache(trades: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Process trades without caching (baseline).

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data

    Returns
    -------
    tuple[pd.DataFrame, float]
        Range bars DataFrame and processing time
    """
    start = time.perf_counter()
    df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
    elapsed = time.perf_counter() - start
    return df, elapsed


def example_with_cache_first_run(
    trades: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame, float]:
    """Process trades with caching (first run - cache miss).

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data
    symbol : str
        Trading symbol for cache key

    Returns
    -------
    tuple[pd.DataFrame, float]
        Range bars DataFrame and processing time
    """
    start = time.perf_counter()
    df = process_trades_to_dataframe_cached(
        trades, symbol=symbol, threshold_decimal_bps=250
    )
    elapsed = time.perf_counter() - start
    return df, elapsed


def example_with_cache_second_run(
    trades: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame, float]:
    """Process trades with caching (second run - cache hit).

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data
    symbol : str
        Trading symbol for cache key

    Returns
    -------
    tuple[pd.DataFrame, float]
        Range bars DataFrame and processing time
    """
    start = time.perf_counter()
    df = process_trades_to_dataframe_cached(
        trades, symbol=symbol, threshold_decimal_bps=250
    )
    elapsed = time.perf_counter() - start
    return df, elapsed


def example_with_shared_cache(trades: pd.DataFrame, symbol: str) -> None:
    """Demonstrate sharing cache across multiple operations.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data
    symbol : str
        Trading symbol
    """
    print("\n" + "-" * 50)
    print("Example: Shared Cache Instance")
    print("-" * 50)

    # Create cache once, reuse for multiple operations
    with RangeBarCache() as cache:
        # Process multiple thresholds with same cache
        for threshold in [100, 250, 500]:
            start = time.perf_counter()
            df = process_trades_to_dataframe_cached(
                trades, symbol=symbol, threshold_decimal_bps=threshold, cache=cache
            )
            elapsed = time.perf_counter() - start
            print(
                f"   Threshold {threshold:4d} dbps: {len(df):5d} bars in {elapsed*1000:.1f}ms"
            )


def main() -> None:
    """Run ClickHouse cache example."""
    print("\n" + "=" * 70)
    print("rangebar-py: ClickHouse Cache Example")
    print("=" * 70)

    # Step 1: Check preflight
    ch_available = check_preflight()

    if not ch_available:
        print("\n" + "-" * 70)
        print("Running without cache (ClickHouse not available)")
        print("-" * 70)

        # Generate test data
        trades = generate_synthetic_trades(10000)
        print(f"\nGenerated {len(trades)} synthetic trades")

        # Process without cache
        df, elapsed = example_without_cache(trades)
        print(f"Processed to {len(df)} range bars in {elapsed*1000:.1f}ms")
        print("\nTip: Install ClickHouse for caching support:")
        print("  macOS: brew install --cask clickhouse")
        print("  Linux: curl https://clickhouse.com/ | sh")
        return

    # Step 2: Generate test data
    print("\n" + "-" * 50)
    print("Generating Synthetic Trade Data")
    print("-" * 50)

    trades = generate_synthetic_trades(10000)
    print(f"Generated {len(trades)} trades")
    print(f"  Price range: ${trades['price'].min():.2f} - ${trades['price'].max():.2f}")
    print(
        f"  Time span: {(trades['timestamp'].max() - trades['timestamp'].min()) / 1000:.0f} seconds"
    )

    symbol = "SYNTHETIC_TEST"

    # Step 3: Baseline (no cache)
    print("\n" + "-" * 50)
    print("Baseline: Without Cache")
    print("-" * 50)

    df_baseline, time_baseline = example_without_cache(trades)
    print(f"  Processed {len(trades)} trades → {len(df_baseline)} bars")
    print(f"  Time: {time_baseline*1000:.1f}ms")

    # Step 4: First run with cache (cache miss)
    print("\n" + "-" * 50)
    print("First Run: With Cache (Cache Miss)")
    print("-" * 50)

    df_first, time_first = example_with_cache_first_run(trades, symbol)
    print(f"  Processed {len(trades)} trades → {len(df_first)} bars")
    print(f"  Time: {time_first*1000:.1f}ms (includes cache storage)")

    # Step 5: Second run with cache (cache hit)
    print("\n" + "-" * 50)
    print("Second Run: With Cache (Cache Hit)")
    print("-" * 50)

    df_second, time_second = example_with_cache_second_run(trades, symbol)
    print(f"  Retrieved {len(df_second)} bars from cache")
    print(f"  Time: {time_second*1000:.1f}ms")

    # Step 6: Show speedup
    print("\n" + "-" * 50)
    print("Performance Summary")
    print("-" * 50)

    print(f"  Baseline (no cache):     {time_baseline*1000:8.1f}ms")
    print(f"  First run (cache miss):  {time_first*1000:8.1f}ms")
    print(f"  Second run (cache hit):  {time_second*1000:8.1f}ms")

    if time_second > 0:
        speedup = time_baseline / time_second
        print(f"\n  Cache hit speedup: {speedup:.1f}x faster than baseline")

    # Step 7: Verify data consistency
    print("\n" + "-" * 50)
    print("Data Consistency Check")
    print("-" * 50)

    pd.testing.assert_frame_equal(
        df_baseline.reset_index(drop=True),
        df_first.reset_index(drop=True),
        check_names=False,
    )
    print("  ✅ First run matches baseline")

    pd.testing.assert_frame_equal(
        df_baseline.reset_index(drop=True),
        df_second.reset_index(drop=True),
        check_names=False,
    )
    print("  ✅ Second run (cached) matches baseline")

    # Step 8: Shared cache example
    example_with_shared_cache(trades, symbol)

    # Done
    print("\n" + "=" * 70)
    print("✅ ClickHouse cache example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
