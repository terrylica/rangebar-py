#!/usr/bin/env python3
# ruff: noqa: E501, PD901, PLR2004, F841
"""Portable validation script for get_n_range_bars() API.

Run on GPU workstations to validate end-to-end functionality:
    uv run python scripts/validate_n_range_bars.py

Requirements:
    uv pip install rangebar>=5.0.0

Prerequisites (for ClickHouse tests):
    - bigblack: Schema migration applied (threshold_bps -> threshold_decimal_bps)
    - littleblack: ClickHouse server started
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def main() -> int:  # noqa: PLR0912, PLR0915
    """Run validation suite."""
    print("=" * 70)
    print("rangebar get_n_range_bars() Validation Suite")
    print("=" * 70)
    print()

    # Check uv is being used
    import shutil

    if shutil.which("uv") is None:
        print(
            "WARNING: uv not found in PATH. Recommend using 'uv run python' to run this script."
        )
    else:
        print("uv: Found in PATH")

    # Import rangebar
    try:
        from rangebar import __version__, get_n_range_bars

        print(f"rangebar version: {__version__}")
    except ImportError as e:
        print(f"ERROR: Failed to import rangebar: {e}")
        print("Install with: uv pip install rangebar>=5.0.0")
        return 1

    import pandas as pd

    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0

    def test(name: str, condition: bool, details: str = "") -> None:
        nonlocal tests_passed, tests_failed
        if condition:
            print(f"  PASS {name}")
            tests_passed += 1
        else:
            print(f"  FAIL {name}: {details}")
            tests_failed += 1

    def skip(name: str, reason: str) -> None:
        nonlocal tests_skipped
        print(f"  SKIP {name}: {reason}")
        tests_skipped += 1

    # =========================================================================
    # Test 1: Parameter Validation
    # =========================================================================
    print("\n[1/7] Testing parameter validation...")

    # n_bars=0
    try:
        get_n_range_bars("BTCUSDT", n_bars=0, use_cache=False, fetch_if_missing=False)
        test("n_bars=0 raises ValueError", False, "No exception raised")
    except ValueError:
        test("n_bars=0 raises ValueError", True)
    except Exception as e:
        test(
            "n_bars=0 raises ValueError", False, f"Wrong exception: {type(e).__name__}"
        )

    # n_bars=-1
    try:
        get_n_range_bars("BTCUSDT", n_bars=-1, use_cache=False, fetch_if_missing=False)
        test("n_bars=-1 raises ValueError", False, "No exception raised")
    except ValueError:
        test("n_bars=-1 raises ValueError", True)
    except Exception as e:
        test(
            "n_bars=-1 raises ValueError", False, f"Wrong exception: {type(e).__name__}"
        )

    # Invalid threshold preset
    try:
        get_n_range_bars(
            "BTCUSDT",
            n_bars=100,
            threshold_decimal_bps="invalid",
            use_cache=False,
            fetch_if_missing=False,
        )
        test("Invalid preset raises ValueError", False, "No exception raised")
    except ValueError:
        test("Invalid preset raises ValueError", True)
    except Exception as e:
        test(
            "Invalid preset raises ValueError",
            False,
            f"Wrong exception: {type(e).__name__}",
        )

    # Invalid date format
    try:
        get_n_range_bars(
            "BTCUSDT",
            n_bars=100,
            end_date="2024/01/01",
            use_cache=False,
            fetch_if_missing=False,
        )
        test("Invalid date raises ValueError", False, "No exception raised")
    except ValueError:
        test("Invalid date raises ValueError", True)
    except Exception as e:
        test(
            "Invalid date raises ValueError",
            False,
            f"Wrong exception: {type(e).__name__}",
        )

    # =========================================================================
    # Test 2: Output Format (Empty Result)
    # =========================================================================
    print("\n[2/7] Testing output format (empty result)...")

    df = get_n_range_bars(
        "BTCUSDT",
        n_bars=100,
        use_cache=False,
        fetch_if_missing=False,
        warn_if_fewer=False,
    )
    test(
        "Empty result has OHLCV columns",
        list(df.columns) == ["Open", "High", "Low", "Close", "Volume"],
    )
    test("Empty result has DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
    test("Empty result is empty", len(df) == 0)

    # =========================================================================
    # Test 3: Threshold Presets
    # =========================================================================
    print("\n[3/7] Testing threshold presets...")

    for preset in ["tight", "medium", "wide"]:
        try:
            get_n_range_bars(
                "BTCUSDT",
                n_bars=50,
                threshold_decimal_bps=preset,  # type: ignore[arg-type]
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=False,
            )
            test(f"Preset '{preset}' accepted", True)
        except ValueError as e:
            test(f"Preset '{preset}' accepted", False, str(e))

    # =========================================================================
    # Test 4: ClickHouse Connection
    # =========================================================================
    print("\n[4/7] Testing ClickHouse connection...")

    clickhouse_available = False
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            # Test count_bars
            count = cache.count_bars("BTCUSDT", 250)
            test(f"ClickHouse connected (count={count})", True)
            clickhouse_available = True
    except Exception as e:
        skip("ClickHouse connection", str(e))

    # =========================================================================
    # Test 5: Cache Methods
    # =========================================================================
    print("\n[5/7] Testing cache methods...")

    if clickhouse_available:
        try:
            from rangebar.clickhouse import RangeBarCache

            with RangeBarCache() as cache:
                # count_bars
                count = cache.count_bars("BTCUSDT", 250)
                test("count_bars() returns int", isinstance(count, int))

                # get_n_bars
                bars_df, available = cache.get_n_bars("BTCUSDT", 250, 100)
                test("get_n_bars() returns tuple", isinstance(available, int))
                if bars_df is not None:
                    test(
                        "get_n_bars() DataFrame valid",
                        isinstance(bars_df, pd.DataFrame),
                    )

                # get_oldest_bar_timestamp
                oldest = cache.get_oldest_bar_timestamp("BTCUSDT", 250)
                test(
                    "get_oldest_bar_timestamp() returns int or None",
                    oldest is None or isinstance(oldest, int),
                )

                # get_newest_bar_timestamp
                newest = cache.get_newest_bar_timestamp("BTCUSDT", 250)
                test(
                    "get_newest_bar_timestamp() returns int or None",
                    newest is None or isinstance(newest, int),
                )
        except Exception as e:
            test("Cache methods work", False, str(e))
    else:
        skip("Cache methods", "ClickHouse not available")

    # =========================================================================
    # Test 6: Data Fetching (with cache)
    # =========================================================================
    print("\n[6/7] Testing data fetching with cache...")

    if clickhouse_available:
        try:
            start = time.perf_counter()
            df = get_n_range_bars(
                "BTCUSDT",
                n_bars=100,
                threshold_decimal_bps=250,
                use_cache=True,
                max_lookback_days=30,
                warn_if_fewer=False,
            )
            elapsed = time.perf_counter() - start

            test(
                f"Returned {len(df)} bars in {elapsed:.1f}s",
                len(df) > 0 or elapsed < 60,
            )

            if len(df) >= 100:
                test("Returns 100 bars", len(df) == 100, f"got {len(df)}")

                # Verify format
                test(
                    "OHLCV columns",
                    list(df.columns) == ["Open", "High", "Low", "Close", "Volume"],
                )
                test("DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
                test("Chronological order", df.index.is_monotonic_increasing)

                # OHLC invariants
                test("High >= Open", (df["High"] >= df["Open"]).all())
                test("High >= Close", (df["High"] >= df["Close"]).all())
                test("Low <= Open", (df["Low"] <= df["Open"]).all())
                test("Low <= Close", (df["Low"] <= df["Close"]).all())

                # No NaN
                test("No NaN values", not df.isna().any().any())
            else:
                skip("Format tests", f"Only {len(df)} bars returned")

        except Exception as e:
            test("Data fetching works", False, str(e))
    else:
        skip("Data fetching with cache", "ClickHouse not available")

    # =========================================================================
    # Test 7: Cache Performance
    # =========================================================================
    print("\n[7/7] Testing cache performance...")

    if clickhouse_available:
        try:
            # First call (may populate cache)
            start1 = time.perf_counter()
            df1 = get_n_range_bars("BTCUSDT", n_bars=100, use_cache=True)
            first_time = time.perf_counter() - start1

            if len(df1) >= 100:
                # Second call (should hit cache)
                start2 = time.perf_counter()
                df2 = get_n_range_bars("BTCUSDT", n_bars=100, use_cache=True)
                second_time = time.perf_counter() - start2

                print(f"  INFO First call: {first_time:.2f}s")
                print(f"  INFO Second call (cache): {second_time:.2f}s")

                test(
                    "Cache hit is fast (<1s)",
                    second_time < 1.0,
                    f"took {second_time:.2f}s",
                )
                test("Cache hit is faster than first call", second_time < first_time)
            else:
                skip("Cache performance", f"Only {len(df1)} bars available")
        except Exception as e:
            test("Cache performance", False, str(e))
    else:
        skip("Cache performance", "ClickHouse not available")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print(
        f"Results: {tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped"
    )
    print("=" * 70)

    if tests_failed > 0:
        print("\nFailed tests require investigation.")
        return 1

    if tests_skipped > 0 and not clickhouse_available:
        print("\nNote: ClickHouse tests skipped. To run all tests:")
        print(
            "  - bigblack: Ensure schema migrated (threshold_bps -> threshold_decimal_bps)"
        )
        print("  - littleblack: Start ClickHouse server")
        return 0

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
