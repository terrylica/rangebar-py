#!/usr/bin/env python3
"""Validate ClickHouse connectivity and cache integrity."""

import sys

from rangebar.clickhouse.cache import RangeBarCache


def main() -> int:
    """Validate ClickHouse connection and schema."""
    print("=== ClickHouse Validation ===")

    try:
        cache = RangeBarCache()
        print(f"URL: {cache.client.url}")
        print(f"Database: {cache.client.database}")

        # Test connectivity
        result = cache.client.query("SELECT 1")
        if result.result_rows[0][0] == 1:
            print("Connectivity: OK")
        else:
            print("Connectivity: FAILED (unexpected response)")
            return 1

        # Check database exists
        result = cache.client.query("SHOW DATABASES")
        databases = [row[0] for row in result.result_rows]
        if "rangebar_cache" in databases:
            print("Database 'rangebar_cache': OK")
        else:
            print("Database 'rangebar_cache': MISSING")
            print("  Run schema setup to create database")
            return 1

        # Check table exists
        result = cache.client.query("SHOW TABLES FROM rangebar_cache")
        tables = [row[0] for row in result.result_rows]
        if "range_bars" in tables:
            print("Table 'range_bars': OK")
        else:
            print("Table 'range_bars': MISSING")
            return 1

        # Check row count
        result = cache.client.query("SELECT count() FROM rangebar_cache.range_bars")
        count = result.result_rows[0][0]
        print(f"Cached bars: {count:,}")

    except Exception as e:
        print(f"ERROR: {e}")
        print("=== FAIL ===")
        return 1
    else:
        print("=== PASS ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
