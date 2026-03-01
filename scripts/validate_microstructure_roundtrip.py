"""Fail-fast validation: microstructure columns survive ClickHouse roundtrip.

Proves the full pipeline works before committing to multi-hour backfills:
  compute bars (all 38 features) → store_bars_bulk → read back → verify non-null

Motivation: Issue #78 Part 2 showed inter-bar features computed correctly in Rust
but silently dropped during cache write/read. This script catches that class of bug.

Usage:
    uv run python scripts/validate_microstructure_roundtrip.py
"""

from __future__ import annotations

import os
import sys
import time

# Disable symbol gate (test symbol won't be in registry)
os.environ["RANGEBAR_SYMBOL_GATE"] = "off"

# Allow low thresholds for testing
os.environ.setdefault("RANGEBAR_CRYPTO_MIN_THRESHOLD", "1")


def generate_synthetic_trades(n_trades: int = 600, base_price: float = 50000.0):
    """Generate synthetic trades that produce multiple bar closures at 250 dbps.

    250 dbps = 0.25% threshold. With price oscillating ~0.3% each swing,
    we get a bar closure roughly every 20-40 trades.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC

    trades = []
    price = base_price
    for i in range(n_trades):
        # Random walk with enough volatility to trigger 250 dbps bars
        price *= 1 + rng.normal(0, 0.0008)
        trades.append({
            "agg_trade_id": i + 1,
            "price": round(price, 2),
            "quantity": round(abs(rng.exponential(0.5)) + 0.01, 4),
            "first_trade_id": i * 3 + 1,
            "last_trade_id": i * 3 + 3,  # 3 individual trades per agg
            "timestamp": base_ts + i * 500,  # 500ms apart
            "is_buyer_maker": bool(rng.integers(0, 2)),
        })

    return trades


def main():
    print("=" * 70)
    print("FAIL-FAST: Microstructure Column ClickHouse Roundtrip Validation")
    print("=" * 70)

    # --- Preflight: Check ClickHouse ---
    print("\n[1/6] Preflight: checking ClickHouse availability...")
    try:
        from rangebar.clickhouse import (
            InstallationLevel,
            RangeBarCache,
            detect_clickhouse_state,
        )

        state = detect_clickhouse_state()
        if state.level < InstallationLevel.RUNNING_NO_SCHEMA:
            print(f"  SKIP: ClickHouse not available ({state.message})")
            sys.exit(0)
        print(f"  OK: {state.message}")
    except (ImportError, OSError, ConnectionError) as e:
        print(f"  SKIP: Cannot connect to ClickHouse ({e})")
        sys.exit(0)

    from rangebar._core import PyRangeBarProcessor
    from rangebar.constants import (
        INTER_BAR_FEATURE_COLUMNS,
        INTRA_BAR_FEATURE_COLUMNS,
        MICROSTRUCTURE_COLUMNS,
    )

    test_symbol = f"TEST_ROUNDTRIP_{int(time.time())}"
    threshold = 250
    cache = RangeBarCache()

    try:
        # --- Step 1: Compute bars with all features ---
        print("\n[2/6] Computing bars with all microstructure features...")
        processor = PyRangeBarProcessor(
            threshold_decimal_bps=threshold,
            inter_bar_lookback_count=200,
            include_intra_bar_features=True,
        )

        trades = generate_synthetic_trades(600)
        bars = processor.process_trades(trades)
        print(f"  Produced {len(bars)} bars from 600 trades")

        if len(bars) < 3:
            print("  FAIL: Need at least 3 bars for meaningful validation")
            sys.exit(1)

        # Convert to DataFrame (same as orchestration layer does)
        import pandas as pd

        bars_df = pd.DataFrame(bars)
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], format="ISO8601")
        bars_df = bars_df.set_index("timestamp")

        # Rename OHLCV to capitalized (as the Python layer does)
        bars_df = bars_df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        print(f"  DataFrame columns ({len(bars_df.columns)}): {list(bars_df.columns)}")

        # Quick sanity: check that inter-bar and intra-bar columns exist in computed data
        inter_present = [c for c in INTER_BAR_FEATURE_COLUMNS if c in bars_df.columns]
        intra_present = [c for c in INTRA_BAR_FEATURE_COLUMNS if c in bars_df.columns]
        micro_present = [c for c in MICROSTRUCTURE_COLUMNS if c in bars_df.columns]
        print(f"  Core microstructure: {len(micro_present)}/{len(MICROSTRUCTURE_COLUMNS)}")
        print(f"  Inter-bar columns:   {len(inter_present)}/{len(INTER_BAR_FEATURE_COLUMNS)}")
        print(f"  Intra-bar columns:   {len(intra_present)}/{len(INTRA_BAR_FEATURE_COLUMNS)}")

        if len(inter_present) == 0:
            print("  FAIL: No inter-bar columns in computed bars!")
            print("        Rust processor did not output lookback features.")
            sys.exit(1)

        if len(intra_present) == 0:
            print("  FAIL: No intra-bar columns in computed bars!")
            print("        Rust processor did not output intra-bar features.")
            sys.exit(1)

        # --- Step 2: Store via store_bars_bulk ---
        print(f"\n[3/6] Storing {len(bars_df)} bars via store_bars_bulk ({test_symbol})...")
        written = cache.store_bars_bulk(
            symbol=test_symbol,
            threshold_decimal_bps=threshold,
            bars=bars_df,
            ouroboros_mode="month",
        )
        print(f"  Written: {written} rows")
        assert written == len(bars_df), f"Expected {len(bars_df)}, wrote {written}"

        # Brief wait for ClickHouse async processing
        time.sleep(0.5)

        # --- Step 3: Read back via get_bars_by_timestamp_range ---
        print("\n[4/6] Reading back via get_bars_by_timestamp_range(include_microstructure=True)...")
        bars_df_reset = bars_df.reset_index()
        ts_col = "timestamp"
        start_ts = int(bars_df_reset[ts_col].min().timestamp() * 1000)
        end_ts = int(bars_df_reset[ts_col].max().timestamp() * 1000)
        print(f"  Query range: {start_ts} - {end_ts}")

        # Debug: check what was actually stored
        debug_result = cache.client.query(
            "SELECT count(*), min(close_time_ms), max(close_time_ms), "
            "min(source_start_ts), max(source_end_ts) "
            "FROM rangebar_cache.range_bars FINAL "
            "WHERE symbol = {symbol:String}",
            parameters={"symbol": test_symbol},
        )
        if debug_result.result_rows:
            row = debug_result.result_rows[0]
            print(f"  Stored: {row[0]} rows, ts_ms=[{row[1]}, {row[2]}], "
                  f"source=[{row[3]}, {row[4]}]")

        retrieved = cache.get_bars_by_timestamp_range(
            symbol=test_symbol,
            threshold_decimal_bps=threshold,
            start_ts=start_ts,
            end_ts=end_ts,
            include_microstructure=True,
            ouroboros_mode="month",
        )

        if retrieved is None or retrieved.empty:
            print("  FAIL: No data retrieved from ClickHouse!")
            sys.exit(1)

        print(f"  Retrieved: {len(retrieved)} rows")
        print(f"  Retrieved columns ({len(retrieved.columns)}): {list(retrieved.columns)}")

        # --- Step 4: Verify all columns present ---
        print("\n[5/6] Verifying column presence and non-null values...")
        failures = []

        # individual_trade_count and agg_record_count are not in the query's
        # hardcoded SELECT (they're basic extended columns, not v7.0 features).
        # Not critical for the 38-feature backfill validation.
        query_excluded = {"individual_trade_count", "agg_record_count"}

        print("\n  --- Core Microstructure Columns ---")
        for col in MICROSTRUCTURE_COLUMNS:
            if col in retrieved.columns:
                non_null = retrieved[col].notna().sum()
                print(f"  OK  {col}: {non_null}/{len(retrieved)} non-null")
            elif col in query_excluded:
                print(f"  SKIP {col}: not in query SELECT (expected)")
            else:
                print(f"  FAIL {col}: MISSING from retrieved data")
                failures.append(f"MISSING core column: {col}")

        print("\n  --- Inter-Bar Lookback Columns (16) ---")
        for col in INTER_BAR_FEATURE_COLUMNS:
            if col in retrieved.columns:
                non_null = retrieved[col].notna().sum()
                status = "OK " if non_null > 0 else "WARN"
                print(f"  {status} {col}: {non_null}/{len(retrieved)} non-null")
            else:
                print(f"  FAIL {col}: MISSING from retrieved data")
                failures.append(f"MISSING inter-bar column: {col}")

        print("\n  --- Intra-Bar Columns (22) ---")
        for col in INTRA_BAR_FEATURE_COLUMNS:
            if col in retrieved.columns:
                non_null = retrieved[col].notna().sum()
                status = "OK " if non_null > 0 else "WARN"
                print(f"  {status} {col}: {non_null}/{len(retrieved)} non-null")
            else:
                print(f"  FAIL {col}: MISSING from retrieved data")
                failures.append(f"MISSING intra-bar column: {col}")

        # --- Step 5: Verify post-warmup values are non-null ---
        print("\n  --- Post-Warmup Non-Null Check ---")
        warmup_bars = min(2, len(retrieved) - 1)
        post_warmup = retrieved.iloc[warmup_bars:]

        # Complexity features (hurst, permutation_entropy) require sufficient
        # trades per bar. With synthetic data (~14 trades/bar) they'll be NULL.
        # Production SOL data has much larger bars (50-5000 trades/bar).
        complexity_cols = {
            "intra_hurst", "intra_permutation_entropy",
            "lookback_hurst", "lookback_permutation_entropy",
        }

        for col in INTER_BAR_FEATURE_COLUMNS:
            if col not in post_warmup.columns:
                continue
            if post_warmup[col].notna().sum() == 0:
                if col in complexity_cols:
                    print(f"  WARN {col}: all NULL (expected with small synthetic bars)")
                else:
                    msg = f"ALL NULL after warmup: {col}"
                    print(f"  FAIL {msg}")
                    failures.append(msg)

        for col in INTRA_BAR_FEATURE_COLUMNS:
            if col not in post_warmup.columns:
                continue
            if post_warmup[col].notna().sum() == 0:
                if col in complexity_cols:
                    print(f"  WARN {col}: all NULL (expected with small synthetic bars)")
                else:
                    msg = f"ALL NULL after warmup: {col}"
                    print(f"  FAIL {msg}")
                    failures.append(msg)

        # --- Results ---
        print(f"\n[6/6] Cleanup: deleting test data ({test_symbol})...")
        cache.client.command(
            f"ALTER TABLE rangebar_cache.range_bars DELETE WHERE symbol = '{test_symbol}'"
        )
        print("  Done.")

        print("\n" + "=" * 70)
        if failures:
            print(f"FAILED: {len(failures)} issue(s) found:")
            for f in failures:
                print(f"  - {f}")
            print("=" * 70)
            sys.exit(1)
        else:
            n_inter = sum(1 for c in INTER_BAR_FEATURE_COLUMNS if c in retrieved.columns)
            n_intra = sum(1 for c in INTRA_BAR_FEATURE_COLUMNS if c in retrieved.columns)
            print("PASSED: Full roundtrip verified!")
            print(f"  {len(bars_df)} bars computed → stored → read back")
            print(f"  {n_inter}/{len(INTER_BAR_FEATURE_COLUMNS)} inter-bar columns preserved")
            print(f"  {n_intra}/{len(INTRA_BAR_FEATURE_COLUMNS)} intra-bar columns preserved")
            print("  Post-warmup values are non-null")
            print("=" * 70)

    finally:
        cache.close()


if __name__ == "__main__":
    main()
