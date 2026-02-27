#!/usr/bin/env python3
"""Memory-safe cache population script.

Processes range bars with:
1. Memory limit to prevent OOM kill (get MemoryError instead)
2. Weekly chunks for high-volume months
3. Resume capability from ClickHouse cache

Usage:
    uv run --python 3.13 python scripts/populate_safe.py --symbol BTCUSDT --threshold 250

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/58
SRED-Type: support-work
SRED-Claim: PATTERN-RESEARCH
"""

import argparse
import os
import time
from datetime import UTC, datetime, timedelta
from gc import collect as gc_collect

# Disable auto memory guard - we manage memory explicitly in this script
# RLIMIT_AS on Linux is too aggressive for virtual address space
os.environ["RANGEBAR_NO_MEMORY_GUARD"] = "1"

import clickhouse_connect
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from rangebar import get_range_bars
from rangebar.resource_guard import get_memory_info, set_memory_limit


def get_last_cached_date(symbol: str, threshold: int) -> str | None:
    """Get the last date with cached data for this symbol/threshold."""
    try:
        client = clickhouse_connect.get_client(host="localhost", port=8123)
        result = client.query(f"""
            SELECT max(toDate(fromUnixTimestamp64Milli(close_time_ms))) as last_date
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
        """)
        if result.result_rows and result.result_rows[0][0]:
            last_date = result.result_rows[0][0]
            # Return the day after the last cached date
            next_day = last_date + timedelta(days=1)
            return next_day.strftime("%Y-%m-%d")
    except (DatabaseError, OperationalError, OSError) as e:
        print(f"  Cache check failed: {e}", flush=True)
    return None


def date_range_weekly(start_str: str, end_str: str):
    """Generate weekly date ranges for finer granularity."""
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=UTC)

    current = start
    while current < end:
        week_end = min(current + timedelta(days=6), end)
        yield current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")
        current = week_end + timedelta(days=1)


def populate_symbol(
    symbol: str,
    threshold: int,
    start: str,
    end: str,
    memory_limit_gb: float | None = None,
) -> int:
    """Populate cache for a single symbol with memory safety.

    Returns total bars processed.
    """
    # Memory limit via RLIMIT_AS is problematic on Linux (limits virtual address space,
    # not RSS). Skip it - weekly chunks + resume capability provide safety instead.
    if memory_limit_gb is not None:
        limit = set_memory_limit(max_gb=memory_limit_gb)
        if limit > 0:
            print(f"Memory limit set to {limit / 1024**3:.1f} GB", flush=True)
        else:
            print("Memory limit not set (RLIMIT unavailable)", flush=True)
    else:
        print("Memory limit disabled (using chunking for safety)", flush=True)

    # Check cache for resume point
    resume_date = get_last_cached_date(symbol, threshold)
    if resume_date and resume_date > start:
        print(f"Resuming from {resume_date} (cache has data up to day before)", flush=True)
        start = resume_date

    if start >= end:
        print(f"Already complete (start {start} >= end {end})", flush=True)
        return 0

    print(f"Processing {symbol} @ {threshold} dbps ({start} to {end})", flush=True)

    total_bars = 0
    errors = 0

    for chunk_start, chunk_end in date_range_weekly(start, end):
        t0 = time.time()

        try:
            df = get_range_bars(
                symbol,
                chunk_start,
                chunk_end,
                threshold_decimal_bps=threshold,
                use_cache=True,
                fetch_if_missing=True,
            )
            elapsed = time.time() - t0
            mem_after = get_memory_info()
            total_bars += len(df)

            print(
                f"  {chunk_start} to {chunk_end}: {len(df):>6,} bars "
                f"({elapsed:>5.1f}s, {mem_after.process_rss_mb}MB RSS)",
                flush=True
            )

        except MemoryError:
            print(f"  {chunk_start}: MEMORY ERROR - skipping", flush=True)
            errors += 1
            # Force garbage collection
            gc_collect()
            time.sleep(5)

        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"  {chunk_start}: ERROR - {e}", flush=True)
            errors += 1

    print(f"DONE: {total_bars:,} bars, {errors} errors", flush=True)
    return total_bars


def main():
    parser = argparse.ArgumentParser(description="Memory-safe cache population")
    parser.add_argument("--symbol", required=True, help="Symbol to process")
    parser.add_argument("--threshold", type=int, required=True, help="Threshold in dbps")
    parser.add_argument("--start", default="2022-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--memory-limit", type=float, default=None, help="Memory limit in GB (disabled by default on Linux)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MEMORY-SAFE CACHE POPULATION", flush=True)
    print(f"Symbol: {args.symbol}", flush=True)
    print(f"Threshold: {args.threshold} dbps", flush=True)
    print(f"Period: {args.start} to {args.end}", flush=True)
    print(f"Memory limit: {args.memory_limit} GB", flush=True)
    print("=" * 70, flush=True)

    mem = get_memory_info()
    print(f"System RAM: {mem.system_total_mb / 1024:.1f} GB total, "
          f"{mem.system_available_mb / 1024:.1f} GB available", flush=True)

    total = populate_symbol(
        args.symbol,
        args.threshold,
        args.start,
        args.end,
        args.memory_limit,
    )

    print("=" * 70, flush=True)
    print(f"COMPLETE: {total:,} total bars", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
