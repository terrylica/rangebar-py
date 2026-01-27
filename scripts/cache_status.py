#!/usr/bin/env python3
"""Show ClickHouse cache statistics for range bars."""

from datetime import UTC

from rangebar.clickhouse.cache import RangeBarCache


def main() -> None:
    """Display cache statistics."""
    cache = RangeBarCache()
    print("=== ClickHouse Cache Status ===")

    # Count total bars
    result = cache.client.query("SELECT count() FROM rangebar_cache.range_bars")
    total = result.result_rows[0][0]
    print(f"Total cached bars: {total:,}")

    # Count by symbol and threshold
    query = """
        SELECT symbol, threshold_decimal_bps, count() as bars,
               min(timestamp_ms) as earliest_ms, max(timestamp_ms) as latest_ms
        FROM rangebar_cache.range_bars
        GROUP BY symbol, threshold_decimal_bps
        ORDER BY bars DESC
    """
    result = cache.client.query(query)

    print("\nBy symbol/threshold:")
    for row in result.result_rows[:10]:
        symbol, threshold, bars, earliest_ms, latest_ms = row
        # Convert ms to readable date
        from datetime import datetime

        earliest = datetime.fromtimestamp(earliest_ms / 1000, tz=UTC).strftime(
            "%Y-%m-%d"
        )
        latest = datetime.fromtimestamp(latest_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
        print(f"  {symbol} @ {threshold} dbps: {bars:,} bars ({earliest} to {latest})")

    max_display = 10
    if len(result.result_rows) > max_display:
        print(f"  ... and {len(result.result_rows) - max_display} more")


if __name__ == "__main__":
    main()
