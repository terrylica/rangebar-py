"""Regenerate ClickHouse cache after Issue #46 defer_open fix.

The defer_open bug caused the streaming path to double-count breaching trades,
producing incorrect bar boundaries. All cached data must be regenerated with
the fixed processor.

Uses precompute_range_bars() which streams month-by-month to avoid OOM.
Called year-by-year to match ouroboros='year' boundaries (processor resets
at Jan 1 each year). Since precompute processes each year independently
and ouroboros='year' resets at the same boundary, results are equivalent.

The cache write uses ouroboros_mode='year' via store_bars_bulk().

Usage:
    RANGEBAR_CH_HOSTS=bigblack RANGEBAR_CH_PRIMARY=bigblack python scripts/regenerate_cache.py
"""

from __future__ import annotations

import fcntl
import gc
import sys
import time
from pathlib import Path

from rangebar import precompute_range_bars

# Prevent duplicate launches (Issue #47: background task stacking caused 70GB OOM)
_lock_file = Path("/tmp/regenerate_cache.lock").open("w")  # noqa: SIM115
try:
    fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit("ERROR: Another regenerate_cache.py is already running.")

COMBINATIONS: list[tuple[str, int, int]] = [
    # (symbol, threshold_decimal_bps, start_year)
    ("BTCUSDT", 50, 2022),
    ("BTCUSDT", 100, 2022),
    ("BTCUSDT", 150, 2022),
    ("BTCUSDT", 200, 2022),
    ("BTCUSDT", 250, 2022),
    ("BTCUSDT", 700, 2022),
    ("ETHUSDT", 100, 2022),
    ("ETHUSDT", 250, 2022),
    ("ETHUSDT", 700, 2022),
    ("SOLUSDT", 100, 2023),
    ("BNBUSDT", 100, 2022),
]

END_YEAR = 2026
END_DATE_FINAL = "2026-01-29"


def main() -> None:
    total_bars = 0
    total_start = time.time()
    results: list[tuple[str, int, int, float]] = []

    for i, (symbol, threshold, start_year) in enumerate(COMBINATIONS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(COMBINATIONS)}] {symbol} @ {threshold} dbps")
        print(f"{'=' * 60}")

        combo_start = time.time()
        combo_bars = 0

        for year in range(start_year, END_YEAR + 1):
            if symbol == "SOLUSDT" and year == 2023:
                year_start = "2023-06-01"
            else:
                year_start = f"{year}-01-01"

            year_end = END_DATE_FINAL if year == END_YEAR else f"{year}-12-31"

            print(f"  {year_start} -> {year_end} ... ", end="", flush=True)
            year_time = time.time()

            result = precompute_range_bars(
                symbol,
                start_date=year_start,
                end_date=year_end,
                threshold_decimal_bps=threshold,
                invalidate_existing="overlap",
                include_microstructure=True,
                validate_on_complete="warn",
            )
            year_bars = result.total_bars
            combo_bars += year_bars
            year_elapsed = time.time() - year_time
            print(f"{year_bars:,} bars ({year_elapsed:.1f}s)")

            gc.collect()

        elapsed = time.time() - combo_start
        total_bars += combo_bars
        results.append((symbol, threshold, combo_bars, elapsed))
        print(f"  Total: {combo_bars:,} bars in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("REGENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total bars: {total_bars:,}")
    print(f"Total time: {total_elapsed:.1f}s")
    print()
    print(f"{'Symbol':<10} {'Threshold':>10} {'Bars':>12} {'Time':>8}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 12} {'-' * 8}")
    for symbol, threshold, bar_count, elapsed in results:
        print(f"{symbol:<10} {threshold:>10} {bar_count:>12,} {elapsed:>7.1f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
