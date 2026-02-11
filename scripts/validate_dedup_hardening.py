#!/usr/bin/env python3
"""Validate Issue #90 dedup hardening layers against live ClickHouse.

Run on bigblack (or any host with ClickHouse) after deploying v12.16.x+.

Usage:
    uv run python scripts/validate_dedup_hardening.py

Validates:
    Layer 1: non_replicated_deduplication_window = 1000 on table
    Layer 2: Duplicate INSERT with same token is silently dropped
    Layer 3: deduplicate_bars() completes without crash
    Layer 4: do_not_merge_across_partitions_select_final accepted
"""

from __future__ import annotations

import hashlib
import sys
import time


def validate_layer1_schema_setting(client) -> bool:
    """Layer 1: Verify non_replicated_deduplication_window is set on the table."""
    print("\n=== Layer 1: Schema Setting ===")
    try:
        # SHOW CREATE TABLE is the authoritative source for per-table settings.
        # system.merge_tree_settings only shows global defaults, not per-table overrides.
        create_sql = str(client.command(
            "SHOW CREATE TABLE rangebar_cache.range_bars"
        ))
        if "non_replicated_deduplication_window = 1000" in create_sql:
            print("  non_replicated_deduplication_window = 1000")
            print("  PASS: Setting found in CREATE TABLE definition")
            return True
        if "non_replicated_deduplication_window" in create_sql:
            print("  WARN: Setting present but not 1000")
            return False
        print("  FAIL: Setting not found (need ALTER TABLE MODIFY SETTING)")
        return False
    except (OSError, RuntimeError) as e:
        print(f"  FAIL: {e}")
        return False


def validate_layer2_insert_dedup_token(client) -> bool:
    """Layer 2: Verify duplicate INSERT with same token is dropped."""
    print("\n=== Layer 2: INSERT Dedup Token ===")
    # Use a unique test token to avoid colliding with real data
    test_token = f"test_dedup_{int(time.time())}"
    test_symbol = "__TEST_DEDUP__"
    test_threshold = 99999

    try:
        # Count before
        before = client.command(
            "SELECT count() FROM rangebar_cache.range_bars "
            f"WHERE symbol = '{test_symbol}' AND threshold_decimal_bps = {test_threshold}"
        )
        before = int(before) if before else 0
        print(f"  Rows before: {before}")

        # Insert test row with dedup token
        cache_key = hashlib.md5(test_token.encode()).hexdigest()
        insert_sql = (
            "INSERT INTO rangebar_cache.range_bars "
            "(symbol, threshold_decimal_bps, timestamp_ms, open, high, low, close, "
            "volume, cache_key, rangebar_version) "
            f"VALUES ('{test_symbol}', {test_threshold}, 1000000000000, "
            f"1.0, 2.0, 0.5, 1.5, 100.0, '{cache_key}', 'test')"
        )

        client.command(
            insert_sql,
            settings={
                "insert_deduplicate": 1,
                "insert_deduplication_token": test_token,
            },
        )
        after1 = int(
            client.command(
                "SELECT count() FROM rangebar_cache.range_bars "
                f"WHERE symbol = '{test_symbol}' AND threshold_decimal_bps = {test_threshold}"
            )
            or 0
        )
        print(f"  Rows after 1st INSERT: {after1}")

        # Insert SAME data with SAME token — should be silently dropped
        client.command(
            insert_sql,
            settings={
                "insert_deduplicate": 1,
                "insert_deduplication_token": test_token,
            },
        )
        after2 = int(
            client.command(
                "SELECT count() FROM rangebar_cache.range_bars "
                f"WHERE symbol = '{test_symbol}' AND threshold_decimal_bps = {test_threshold}"
            )
            or 0
        )
        print(f"  Rows after 2nd INSERT (same token): {after2}")

        # Clean up test data
        client.command(
            "ALTER TABLE rangebar_cache.range_bars DELETE "
            f"WHERE symbol = '{test_symbol}'"
        )

        if after1 == before + 1 and after2 == after1:
            print("  PASS: Duplicate INSERT was silently dropped")
            return True
        if after2 > after1:
            print(f"  FAIL: Duplicate was NOT dropped ({after2} > {after1})")
            return False
        print(f"  WARN: Unexpected counts (before={before}, after1={after1}, after2={after2})")
        return False

    except (OSError, RuntimeError) as e:
        print(f"  FAIL: {e}")
        return False


def validate_layer3_fire_and_forget_optimize(cache) -> bool:
    """Layer 3: Verify deduplicate_bars() completes without crash."""
    print("\n=== Layer 3: Fire-and-Forget OPTIMIZE ===")
    try:
        # Use BTCUSDT@1000 (small, fast) for the test
        start = time.monotonic()
        cache.deduplicate_bars("BTCUSDT", 1000, timeout=30)
        elapsed = time.monotonic() - start
        print(f"  deduplicate_bars() completed in {elapsed:.1f}s")
        print("  PASS: No crash, no timeout exception")
        return True
    except (OSError, RuntimeError) as e:
        print(f"  FAIL: deduplicate_bars() raised: {e}")
        return False


def validate_layer4_final_read_settings(client) -> bool:
    """Layer 4: Verify do_not_merge_across_partitions_select_final is accepted."""
    print("\n=== Layer 4: FINAL Read Optimization ===")
    try:
        result = client.query(
            "SELECT count() FROM rangebar_cache.range_bars FINAL "
            "WHERE symbol = 'BTCUSDT' AND threshold_decimal_bps = 1000",
            settings={"do_not_merge_across_partitions_select_final": 1},
        )
        count = result.result_rows[0][0] if result.result_rows else 0
        print(f"  Query with FINAL + parallel partition dedup returned {count:,} rows")
        print("  PASS: Setting accepted by ClickHouse")
        return True
    except (OSError, RuntimeError) as e:
        print(f"  FAIL: {e}")
        return False


def main() -> int:
    from rangebar.clickhouse import RangeBarCache

    print("Issue #90: Dedup Hardening Validation")
    print("=" * 50)

    with RangeBarCache() as cache:
        results = {
            "Layer 1: Schema Setting": validate_layer1_schema_setting(cache.client),
            "Layer 2: INSERT Dedup Token": validate_layer2_insert_dedup_token(cache.client),
            "Layer 3: Fire-and-Forget OPTIMIZE": validate_layer3_fire_and_forget_optimize(cache),
            "Layer 4: FINAL Read Optimization": validate_layer4_final_read_settings(cache.client),
        }

    print("\n" + "=" * 50)
    print("RESULTS:")
    all_pass = True
    for layer, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {layer}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL LAYERS VALIDATED SUCCESSFULLY")
    else:
        print("SOME LAYERS FAILED — see details above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
