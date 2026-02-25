#!/usr/bin/env python3
"""Delete non-TIER1 symbol data and non-standard thresholds from ClickHouse.
# Issue #102: Gap Regression Prevention — cleanup of unmaintained EXTRA symbol data

Removes EXTRA symbol data (ATOM, DOT, ETC, HBAR, PAXG, PEPE, SHIB, TON,
TRX, XLM, ZEC) and any non-standard thresholds (not 250/500/750/1000) from
the rangebar_cache.range_bars table.

Usage:
    python scripts/cleanup_extra_symbols.py --dry-run   # Preview
    python scripts/cleanup_extra_symbols.py              # Execute
"""

from __future__ import annotations

import argparse
import sys

from rangebar.clickhouse.cache import RangeBarCache

EXTRA_SYMBOLS: tuple[str, ...] = (
    "ATOMUSDT", "DOTUSDT", "ETCUSDT", "HBARUSDT", "PAXGUSDT",
    "PEPEUSDT", "SHIBUSDT", "TONUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT",
)

STANDARD_THRESHOLDS: tuple[int, ...] = (250, 500, 750, 1000)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete non-TIER1 symbols and non-standard thresholds from ClickHouse"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    print("=== Cleanup EXTRA Symbols & Non-Standard Thresholds ===")
    print()

    try:
        cache = RangeBarCache()
    except (OSError, RuntimeError, ConnectionError) as e:
        print(f"ERROR: Failed to connect to ClickHouse: {e}")
        return 1

    # Discover all (symbol, threshold, count) in ClickHouse
    result = cache.client.query("""
        SELECT symbol, threshold_decimal_bps, count(*) as bar_count
        FROM rangebar_cache.range_bars
        GROUP BY symbol, threshold_decimal_bps
        ORDER BY symbol, threshold_decimal_bps
    """)

    to_delete: list[tuple[str, int, int]] = []
    for symbol, threshold, bar_count in result.result_rows:
        is_extra = symbol in EXTRA_SYMBOLS
        is_nonstandard = threshold not in STANDARD_THRESHOLDS
        if is_extra or is_nonstandard:
            reason = []
            if is_extra:
                reason.append("EXTRA symbol")
            if is_nonstandard:
                reason.append(f"non-standard threshold ({threshold} dbps)")
            to_delete.append((symbol, threshold, bar_count))

    if not to_delete:
        print("No EXTRA symbols or non-standard thresholds found. Nothing to do.")
        return 0

    # Display what will be deleted
    print("Data to delete:")
    total_bars = 0
    for symbol, threshold, bar_count in to_delete:
        reasons = []
        if symbol in EXTRA_SYMBOLS:
            reasons.append("EXTRA")
        if threshold not in STANDARD_THRESHOLDS:
            reasons.append(f"non-std {threshold}dbps")
        tag = ", ".join(reasons)
        print(f"  {symbol} @ {threshold} dbps: {bar_count:,} bars  [{tag}]")
        total_bars += bar_count
    print()
    print(f"Total: {total_bars:,} bars across {len(to_delete)} pairs")
    print()

    if args.dry_run:
        print("[DRY-RUN] Would delete the above data. Run without --dry-run to execute.")
        return 0

    confirm = input("Proceed with deletion? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return 0

    print()
    for symbol, threshold, bar_count in to_delete:
        print(f"Deleting {symbol} @ {threshold} dbps ({bar_count:,} bars)...", end=" ")
        try:
            cache.client.command(f"""
                ALTER TABLE rangebar_cache.range_bars
                DELETE WHERE symbol = '{symbol}'
                  AND threshold_decimal_bps = {threshold}
            """)
            print("OK")
        except (OSError, RuntimeError) as e:
            print(f"ERROR: {e}")

    print()
    print("Done. Run the following to reclaim disk space:")
    print("  OPTIMIZE TABLE rangebar_cache.range_bars FINAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
