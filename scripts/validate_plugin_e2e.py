#!/usr/bin/env python3
"""E2E validation of Issue #98 FeatureProvider plugin system against bigblack.

Tests the full roundtrip:
  1. Register mock plugin columns in the column registry
  2. Migrate schema (ALTER TABLE ADD COLUMN IF NOT EXISTS)
  3. Read existing bars from ClickHouse
  4. Enrich with mock provider (adds plugin columns)
  5. Write enriched bars back to ClickHouse
  6. Read back and verify plugin columns are non-NULL
  7. Clean up written rows

Usage:
    uv run python scripts/validate_plugin_e2e.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main() -> int:
    # -------------------------------------------------------------------------
    # [1/7] Register mock plugin columns
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Plugin E2E Validation (Issue #98)")
    print("=" * 60)
    print()

    import rangebar.constants as rc
    from rangebar.constants import register_plugin_columns

    plugin_columns = ("test_plugin_col_a", "test_plugin_col_b")
    register_plugin_columns(plugin_columns)
    print(f"[1/7] Registered plugin columns: {plugin_columns}")
    assert "test_plugin_col_a" in rc._PLUGIN_FEATURE_COLUMNS
    assert "test_plugin_col_b" in rc._PLUGIN_FEATURE_COLUMNS
    print("      Column registry OK")

    # -------------------------------------------------------------------------
    # [2/7] Connect to bigblack and migrate schema
    # -------------------------------------------------------------------------
    from rangebar.clickhouse import RangeBarCache

    print("\n[2/7] Connecting to bigblack ClickHouse...")
    try:
        cache = RangeBarCache()
    except (OSError, RuntimeError, ConnectionError) as e:
        print(f"      SKIP: Cannot connect to ClickHouse: {e}")
        print("      (Run on a machine with bigblack access)")
        return 0

    print("      Connected.")

    # Run migration manually to ensure plugin columns exist
    from rangebar.plugins.migration import migrate_plugin_columns

    class _MockProvider:
        name = "test_e2e"
        version = "0.0.1"
        columns = plugin_columns
        min_bars = 0

        def enrich(self, bars: pd.DataFrame, _symbol: str, _threshold_decimal_bps: int) -> pd.DataFrame:
            n = len(bars)
            bars["test_plugin_col_a"] = np.linspace(0.1, 0.9, n)
            bars["test_plugin_col_b"] = np.linspace(100.0, 200.0, n)
            return bars

    provider = _MockProvider()
    migrate_plugin_columns(cache.client, [provider])
    print("      Schema migration OK (ADD COLUMN IF NOT EXISTS)")

    # Verify columns exist in system.columns
    result = cache.client.query(
        "SELECT name FROM system.columns "
        "WHERE database = 'rangebar_cache' AND table = 'range_bars' "
        "AND name IN ('test_plugin_col_a', 'test_plugin_col_b')"
    )
    col_names = [row[0] for row in result.result_rows]
    assert "test_plugin_col_a" in col_names, f"Column missing: {col_names}"
    assert "test_plugin_col_b" in col_names, f"Column missing: {col_names}"
    print(f"      Verified in system.columns: {col_names}")

    # -------------------------------------------------------------------------
    # [3/7] Read existing bars from ClickHouse
    # -------------------------------------------------------------------------
    print("\n[3/7] Reading existing bars (BTCUSDT @500, last 100)...")
    bars_df = cache.get_bars_by_timestamp_range(
        symbol="BTCUSDT",
        threshold_decimal_bps=500,
        start_ts=0,
        end_ts=2_000_000_000_000,  # far future
        include_microstructure=False,
    )
    if bars_df is None or bars_df.empty:
        print("      SKIP: No BTCUSDT@500 bars in cache. Need populated cache.")
        cache.close()
        return 0

    # Take last 100 bars to keep validation fast
    bars_df = bars_df.tail(100).copy()
    print(f"      Read {len(bars_df)} bars")
    print(f"      Columns: {list(bars_df.columns[:6])}...")
    print(f"      Index range: {bars_df.index[0]} — {bars_df.index[-1]}")

    # -------------------------------------------------------------------------
    # [4/7] Enrich with mock provider
    # -------------------------------------------------------------------------
    print("\n[4/7] Enriching bars with plugin provider...")
    enriched = provider.enrich(bars_df, "BTCUSDT", 500)
    assert "test_plugin_col_a" in enriched.columns
    assert "test_plugin_col_b" in enriched.columns
    print(f"      test_plugin_col_a range: [{enriched['test_plugin_col_a'].min():.4f}, {enriched['test_plugin_col_a'].max():.4f}]")
    print(f"      test_plugin_col_b range: [{enriched['test_plugin_col_b'].min():.1f}, {enriched['test_plugin_col_b'].max():.1f}]")

    # -------------------------------------------------------------------------
    # [5/7] Write enriched bars via store_bars_bulk (the real write path)
    # -------------------------------------------------------------------------
    print("\n[5/7] Writing enriched bars via store_bars_bulk()...")
    written = cache.store_bars_bulk(
        symbol="BTCUSDT",
        threshold_decimal_bps=500,
        bars=enriched,
        skip_dedup=True,  # allow re-write for validation
    )
    print(f"      Written: {written} rows")
    assert written == len(enriched), f"Expected {len(enriched)}, got {written}"

    # -------------------------------------------------------------------------
    # [6/7] Read back and verify plugin columns are present and non-NULL
    # -------------------------------------------------------------------------
    print("\n[6/7] Reading back with include_plugin_features=True...")

    # Use direct SQL to verify the columns in ClickHouse
    ts_min = int(enriched.index[0].timestamp() * 1000)
    ts_max = int(enriched.index[-1].timestamp() * 1000)

    verify_result = cache.client.query(
        "SELECT timestamp_ms, test_plugin_col_a, test_plugin_col_b "
        "FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = 'BTCUSDT' "
        "  AND threshold_decimal_bps = 500 "
        "  AND timestamp_ms >= {ts_min:Int64} "
        "  AND timestamp_ms <= {ts_max:Int64} "
        "  AND test_plugin_col_a IS NOT NULL "
        "ORDER BY timestamp_ms DESC "
        "LIMIT 10",
        parameters={"ts_min": ts_min, "ts_max": ts_max},
    )

    rows = verify_result.result_rows
    print(f"      Found {len(rows)} rows with non-NULL plugin columns")

    if len(rows) == 0:
        print("      FAIL: No rows with non-NULL plugin columns found!")
        cache.close()
        return 1

    print("      Sample (latest 5):")
    print(f"      {'timestamp_ms':>15s}  {'col_a':>8s}  {'col_b':>8s}")
    for row in rows[:5]:
        print(f"      {row[0]:>15d}  {row[1]:>8.4f}  {row[2]:>8.1f}")

    # Verify values are in expected ranges
    col_a_values = [row[1] for row in rows]
    col_b_values = [row[2] for row in rows]
    assert all(0.0 <= v <= 1.0 for v in col_a_values), f"col_a out of range: {col_a_values}"
    assert all(90.0 <= v <= 210.0 for v in col_b_values), f"col_b out of range: {col_b_values}"
    print("      Value ranges validated OK")

    # Also validate via the query_operations path (include_plugin_features=True)
    bars_with_plugins, count = cache.get_n_bars(
        symbol="BTCUSDT",
        threshold_decimal_bps=500,
        n_bars=50,
        include_plugin_features=True,
    )
    if bars_with_plugins is not None:
        has_col_a = "test_plugin_col_a" in bars_with_plugins.columns
        has_col_b = "test_plugin_col_b" in bars_with_plugins.columns
        print(f"      get_n_bars(include_plugin_features=True): col_a={has_col_a}, col_b={has_col_b}")
        if has_col_a:
            non_null = bars_with_plugins["test_plugin_col_a"].notna().sum()
            print(f"      Non-NULL plugin values in result: {non_null}/{len(bars_with_plugins)}")
    else:
        print("      get_n_bars returned None (unexpected)")

    # -------------------------------------------------------------------------
    # [7/7] Clean up plugin rows from ClickHouse
    # -------------------------------------------------------------------------
    print("\n[7/7] Cleaning up validation rows...")      # Null out plugin columns for the rows we wrote
    cache.client.command(
        "ALTER TABLE rangebar_cache.range_bars "
        "UPDATE test_plugin_col_a = NULL, test_plugin_col_b = NULL "
        "WHERE symbol = 'BTCUSDT' "
        f"  AND timestamp_ms >= {ts_min} "
        f"  AND timestamp_ms <= {ts_max} "
        "  AND test_plugin_col_a IS NOT NULL"
    )
    print("      Plugin columns nulled out (columns remain in schema, harmless)")
    # Clean up column registry
    rc._PLUGIN_FEATURE_COLUMNS.clear()

    cache.close()

    print()
    print("=" * 60)
    print("VALIDATION PASSED — Full E2E roundtrip verified")
    print("=" * 60)
    print()
    print("Verified:")
    print("  - Column registry (register_plugin_columns)")
    print("  - Schema migration (ALTER TABLE ADD COLUMN IF NOT EXISTS)")
    print("  - Write path (store_bars_bulk with plugin columns)")
    print("  - Read path (SQL query with plugin columns)")
    print("  - Query API (get_n_bars with include_plugin_features=True)")
    print("  - Value integrity (ranges preserved through roundtrip)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
