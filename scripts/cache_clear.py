#!/usr/bin/env python3
"""Clear ClickHouse cache for range bars (interactive)."""

import sys

from rangebar.clickhouse.cache import RangeBarCache


def main() -> None:
    """Clear cache with interactive confirmation."""
    print("=== Clear ClickHouse Cache ===")
    print("WARNING: This will delete cached range bars.")
    print()

    symbol = input("Enter symbol to clear (e.g., BTCUSDT) or 'ALL': ").strip()

    if not symbol:
        print("No symbol entered. Aborted.")
        sys.exit(1)

    cache = RangeBarCache()

    if symbol.upper() == "ALL":
        confirm = input("Are you sure you want to clear ALL cached data? (yes/no): ")
        if confirm.strip().lower() == "yes":
            cache.client.command("TRUNCATE TABLE rangebar_cache.range_bars")
            print("All cache cleared.")
        else:
            print("Aborted.")
            sys.exit(0)
    else:
        # Show current count for this symbol
        result = cache.client.query(
            f"SELECT count() FROM rangebar_cache.range_bars WHERE symbol = '{symbol}'"
        )
        count = result.result_rows[0][0]

        if count == 0:
            print(f"No cached bars found for {symbol}.")
            sys.exit(0)

        confirm = input(f"Delete {count:,} bars for {symbol}? (yes/no): ")
        if confirm.strip().lower() == "yes":
            table = "rangebar_cache.range_bars"
            cache.client.command(
                f"ALTER TABLE {table} DELETE WHERE symbol = '{symbol}'"
            )
            print(f"Cache cleared for {symbol}.")
        else:
            print("Aborted.")


if __name__ == "__main__":
    main()
