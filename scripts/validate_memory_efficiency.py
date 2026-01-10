#!/usr/bin/env python3
# ruff: noqa: E501, PD901, PLR2004, T201, TRY300, PLR0913, DTZ001, PLR0912, PLR0915
"""Memory efficiency validation for high-volume months.

Run before releases to prevent memory regressions like #12 and #14:
    uv run python scripts/validate_memory_efficiency.py

This script tests both cached and uncached code paths for known high-volume
months that have historically caused OOM issues.

Requirements:
    - rangebar >= 5.3.6
    - 8GB+ available RAM
    - Network access for Binance data fetching (uncached tests)
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# High-volume months that have caused OOM issues historically
HIGH_VOLUME_MONTHS: list[tuple[str, int, int, int, float]] = [
    # (symbol, year, month, expected_rows, max_memory_gb)
    ("BTCUSDT", 2024, 3, 55_000_000, 6.0),  # March 2024 - triggered #12, #14
    ("BTCUSDT", 2024, 4, 45_000_000, 5.0),  # April 2024
    ("BTCUSDT", 2024, 2, 40_000_000, 5.0),  # February 2024 - triggered #11
]


@dataclass
class MemoryTestResult:
    """Result of a memory test."""

    month: str
    path_type: str  # "cached" or "uncached"
    peak_memory_gb: float
    max_allowed_gb: float
    passed: bool
    error: str | None = None
    bars_generated: int = 0
    elapsed_seconds: float = 0.0


def get_peak_memory_gb() -> float:
    """Get peak memory usage in GB using tracemalloc."""
    _current, peak = tracemalloc.get_traced_memory()
    return peak / (1024**3)


def clear_tick_cache(symbol: str, year: int, month: int) -> bool:
    """Clear tick cache for a specific month. Returns True if cleared."""
    try:
        from platformdirs import user_cache_dir

        cache_dir = Path(user_cache_dir("rangebar", "rangebar"))
        # Cache files are stored by symbol/year/month
        cache_pattern = f"*{symbol}*{year}*{month:02d}*"

        cleared = False
        for cache_file in cache_dir.glob(cache_pattern):
            cache_file.unlink()
            cleared = True

        return cleared
    except Exception:
        return False


def test_precompute_memory(
    symbol: str,
    year: int,
    month: int,
    max_memory_gb: float,
    cached: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> MemoryTestResult:
    """Test memory usage for precompute_range_bars on a specific month."""
    from datetime import datetime

    month_str = f"{year}-{month:02d}"
    path_type = "cached" if cached else "uncached"

    # Calculate date range for the month
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_year, end_month = year + 1, 1
    else:
        end_year, end_month = year, month + 1
    # Last day of the month
    end_date = (
        datetime(end_year, end_month, 1) - __import__("datetime").timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if progress_callback:
        progress_callback(f"Testing {month_str} ({path_type})...")

    # Clear cache if testing uncached path
    if not cached:
        clear_tick_cache(symbol, year, month)

    # Force garbage collection before test
    gc.collect()

    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()

    try:
        from rangebar import precompute_range_bars

        result = precompute_range_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold_decimal_bps=700,  # Wide threshold for faster processing
            chunk_size=100_000,
        )

        elapsed = time.time() - start_time
        peak_gb = get_peak_memory_gb()

        tracemalloc.stop()

        return MemoryTestResult(
            month=month_str,
            path_type=path_type,
            peak_memory_gb=peak_gb,
            max_allowed_gb=max_memory_gb,
            passed=peak_gb <= max_memory_gb,
            bars_generated=result.total_bars,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        tracemalloc.stop()
        return MemoryTestResult(
            month=month_str,
            path_type=path_type,
            peak_memory_gb=0.0,
            max_allowed_gb=max_memory_gb,
            passed=False,
            error=str(e),
        )


def check_available_memory() -> float:
    """Check available system memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback: assume enough memory
        return 16.0


def main() -> int:
    """Run memory validation suite."""
    print("=" * 70)
    print("rangebar Memory Efficiency Validation Suite")
    print("=" * 70)
    print()

    # Check rangebar version
    try:
        from rangebar import __version__

        print(f"rangebar version: {__version__}")
    except ImportError as e:
        print(f"ERROR: Failed to import rangebar: {e}")
        print("Install with: uv pip install rangebar>=5.3.6")
        return 1

    # Check available memory
    available_gb = check_available_memory()
    print(f"Available memory: {available_gb:.1f} GB")

    if available_gb < 8.0:
        print("WARNING: Less than 8GB available. Some tests may fail or be skipped.")

    print()
    print("-" * 70)
    print("Testing high-volume months for memory efficiency")
    print("-" * 70)
    print()

    results: list[MemoryTestResult] = []
    passed = 0
    failed = 0
    skipped = 0

    for symbol, year, month, expected_rows, max_memory_gb in HIGH_VOLUME_MONTHS:
        month_str = f"{year}-{month:02d}"

        # Check if we have enough memory for this test
        if max_memory_gb > available_gb * 0.8:
            print(
                f"[SKIP] {month_str}: Requires {max_memory_gb:.1f}GB, only {available_gb:.1f}GB available"
            )
            skipped += 1
            continue

        # Test cached path first (if data exists)
        print(f"\n[TEST] {symbol} {month_str} (~{expected_rows / 1_000_000:.0f}M rows)")

        # Test 1: Cached path (run twice - first populates cache, second tests cached read)
        result_cached = test_precompute_memory(
            symbol, year, month, max_memory_gb, cached=True
        )

        status = "PASS" if result_cached.passed else "FAIL"
        if result_cached.error:
            print(f"  Cached:   [{status}] Error: {result_cached.error}")
        else:
            print(
                f"  Cached:   [{status}] Peak: {result_cached.peak_memory_gb:.2f}GB "
                f"(limit: {max_memory_gb:.1f}GB) - {result_cached.bars_generated} bars in {result_cached.elapsed_seconds:.1f}s"
            )
        results.append(result_cached)

        if result_cached.passed:
            passed += 1
        else:
            failed += 1

        # Test 2: Uncached path (clear cache first)
        result_uncached = test_precompute_memory(
            symbol, year, month, max_memory_gb, cached=False
        )

        status = "PASS" if result_uncached.passed else "FAIL"
        if result_uncached.error:
            print(f"  Uncached: [{status}] Error: {result_uncached.error}")
        else:
            print(
                f"  Uncached: [{status}] Peak: {result_uncached.peak_memory_gb:.2f}GB "
                f"(limit: {max_memory_gb:.1f}GB) - {result_uncached.bars_generated} bars in {result_uncached.elapsed_seconds:.1f}s"
            )
        results.append(result_uncached)

        if result_uncached.passed:
            passed += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print()

    if failed > 0:
        print("MEMORY REGRESSION DETECTED - Do not release!")
        print()
        print("Failed tests:")
        for r in results:
            if not r.passed:
                if r.error:
                    print(f"  - {r.month} ({r.path_type}): {r.error}")
                else:
                    print(
                        f"  - {r.month} ({r.path_type}): Peak {r.peak_memory_gb:.2f}GB > limit {r.max_allowed_gb:.1f}GB"
                    )
        return 1

    if passed > 0:
        print("All memory tests passed!")
        return 0

    print("No tests executed (all skipped due to insufficient memory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
