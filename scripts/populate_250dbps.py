"""Populate ClickHouse cache with 250 dbps range bars - chunked by month.

For run length momentum cross-validation research (Issue #58).

Usage:
    uv run --python 3.13 python scripts/populate_250dbps.py

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/58
"""

import sys
import time
from datetime import UTC, datetime, timedelta

sys.path.insert(0, "/Users/terryli/eon/rangebar-py")

from rangebar import get_range_bars


def date_range_months(start_str: str, end_str: str):
    """Generate monthly date ranges."""
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=UTC)

    current = start
    while current < end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        month_end = min(month_end, end)
        yield current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        current = month_end + timedelta(days=1)


# Symbols to populate at 250 dbps
SYMBOLS = ["ETHUSDT", "BNBUSDT"]
THRESHOLD = 250
START = "2022-01-01"
END = "2025-12-31"

print("=" * 70, flush=True)
print("POPULATING 250 DBPS RANGE BARS", flush=True)
print(f"Symbols: {', '.join(SYMBOLS)}", flush=True)
print(f"Period: {START} to {END}", flush=True)
print("=" * 70, flush=True)

for symbol in SYMBOLS:
    print(f"\n{'='*70}", flush=True)
    print(f"SYMBOL: {symbol} @ {THRESHOLD} dbps", flush=True)
    print("=" * 70, flush=True)

    total_bars = 0
    total_time = 0
    errors = 0

    for chunk_start, chunk_end in date_range_months(START, END):
        t0 = time.time()
        try:
            df = get_range_bars(
                symbol,
                start_date=chunk_start,
                end_date=chunk_end,
                threshold_decimal_bps=THRESHOLD,
                use_cache=True,
                fetch_if_missing=True,
            )
            elapsed = time.time() - t0
            total_bars += len(df)
            total_time += elapsed
            print(f"  {chunk_start} to {chunk_end}: {len(df):>6,} bars ({elapsed:>5.1f}s)", flush=True)
        except (ValueError, RuntimeError, ConnectionError) as e:
            errors += 1
            print(f"  {chunk_start} to {chunk_end}: ERROR - {e}", flush=True)

    print(f"\n  SUMMARY: {total_bars:,} bars in {total_time:.1f}s ({errors} errors)", flush=True)

print("\n" + "=" * 70, flush=True)
print("CACHE POPULATION COMPLETE", flush=True)
print("=" * 70, flush=True)
