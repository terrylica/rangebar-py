#!/usr/bin/env python3
"""Purge crypto data below minimum threshold from ClickHouse cache.

Issue #62: Crypto minimum threshold enforcement

This script removes cached crypto range bars with thresholds below the configured
minimum (default 1000 dbps = 1%). Thresholds below 1% cannot overcome trading
costs sufficiently for crypto assets.

Usage:
    python scripts/purge_crypto_low_threshold.py

    # Dry-run (show what would be deleted without deleting):
    python scripts/purge_crypto_low_threshold.py --dry-run

The minimum threshold is read from the SSoT environment variable:
    RANGEBAR_CRYPTO_MIN_THRESHOLD (default: 1000 dbps)
"""

from __future__ import annotations

import argparse
import sys

from rangebar.clickhouse.cache import RangeBarCache
from rangebar.threshold import get_min_threshold
from rangebar.validation.gap_classification import AssetClass, detect_asset_class


def main() -> int:
    """Purge crypto data below minimum threshold."""
    parser = argparse.ArgumentParser(
        description="Purge crypto data below minimum threshold from ClickHouse cache"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    print("=== Purge Crypto Low Threshold Data ===")
    print()

    # Get minimum threshold from SSoT (environment variable or fallback)
    crypto_min = get_min_threshold(AssetClass.CRYPTO)
    print(f"Crypto minimum threshold (from SSoT): {crypto_min} dbps")
    print()

    # Connect to ClickHouse
    try:
        cache = RangeBarCache()
    except (OSError, RuntimeError, ConnectionError) as e:
        print(f"ERROR: Failed to connect to ClickHouse: {e}")
        return 1

    # Find all distinct (symbol, threshold) pairs below minimum
    result = cache.client.query("""
        SELECT DISTINCT symbol, threshold_decimal_bps, count(*) as bar_count
        FROM rangebar_cache.range_bars
        GROUP BY symbol, threshold_decimal_bps
        ORDER BY symbol, threshold_decimal_bps
    """)

    # Filter to crypto symbols below minimum
    to_purge: list[tuple[str, int, int]] = []
    for symbol, threshold, bar_count in result.result_rows:
        if detect_asset_class(symbol) == AssetClass.CRYPTO and threshold < crypto_min:
            to_purge.append((symbol, threshold, bar_count))

    if not to_purge:
        print("No crypto data below minimum threshold found.")
        return 0

    # Show what will be purged
    print("Data to purge:")
    total_bars = 0
    for symbol, threshold, bar_count in to_purge:
        print(f"  {symbol} @ {threshold} dbps: {bar_count:,} bars")
        total_bars += bar_count
    print()
    print(f"Total: {total_bars:,} bars across {len(to_purge)} symbol/threshold combinations")
    print()

    if args.dry_run:
        print("[DRY-RUN] Would delete the above data. Run without --dry-run to execute.")
        return 0

    # Confirm before deletion
    confirm = input("Proceed with deletion? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return 0

    # Execute deletion
    print()
    for symbol, threshold, bar_count in to_purge:
        print(f"Purging {symbol} @ {threshold} dbps ({bar_count:,} bars)...", end=" ")
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
