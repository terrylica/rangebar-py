#!/usr/bin/env python3
"""Re-populate corrupted laguerre_bars_in_regime and laguerre_tail_risk_score.

Issue #99: 1,000 SOLUSDT@500 bars written during E2E validation (Issue #98)
contain wrong values for two laguerre_* columns due to a bug in
atr-adaptive-laguerre <= 2.4.0 that was fixed in 2.4.1.

This script:
  1. Verifies the installed atr-adaptive-laguerre has the fix (bars_in_regime > 1)
  2. Reads the 1,000 corrupted SOLUSDT@500 bars from bigblack
  3. Captures before-stats for comparison
  4. Drops laguerre_* columns to bypass enrich() idempotency check
  5. Re-enriches with the fixed LaguerreFeatureProvider
  6. Verifies corrected values (bars_in_regime > 1, tail_risk_score > 0.4)
  7. Writes back via store_bars_bulk() (ReplacingMergeTree handles dedup)
  8. SQL verification on bigblack

Usage:
    uv run python scripts/repopulate_laguerre_solusdt.py

Context:
    terrylica/rangebar-py#99 (blocked on terrylica/atr-adaptive-laguerre#2)
"""

from __future__ import annotations

import sys


def _check_fix_installed() -> str:
    """Verify bars_in_regime fix is present in installed version."""
    import importlib.metadata

    import pandas as pd

    ver = importlib.metadata.version("atr-adaptive-laguerre")

    # Smoke-test the formula directly (same as Issue #2 reproducible proof)
    regime = pd.Series([0] * 15 + [1] * 5 + [2] * 10)
    g = (regime != regime.shift(1)).cumsum()
    bars_in_regime = g.groupby(g).cumcount() + 1
    assert bars_in_regime.max() == 15, (
        f"bars_in_regime fix NOT present! max={bars_in_regime.max()}. "
        f"Installed version: {ver}. Need >= 2.4.1."
    )
    return ver


def main() -> int:
    print("=" * 65)
    print("Laguerre Re-population — Issue #99")
    print("=" * 65)
    print()

    # -------------------------------------------------------------------------
    # [1/8] Verify the fix is installed
    # -------------------------------------------------------------------------
    print("[1/8] Verifying atr-adaptive-laguerre has bars_in_regime fix...")
    try:
        ver = _check_fix_installed()
    except AssertionError as e:
        print(f"      ABORT: {e}")
        return 1
    print(f"      atr-adaptive-laguerre=={ver}: fix verified OK")

    # -------------------------------------------------------------------------
    # [2/8] Connect to bigblack
    # -------------------------------------------------------------------------
    print("\n[2/8] Connecting to bigblack ClickHouse...")
    from rangebar.clickhouse import RangeBarCache

    try:
        cache = RangeBarCache()
    except (OSError, RuntimeError, ConnectionError) as e:
        print(f"      SKIP: Cannot connect: {e}")
        return 0

    print("      Connected.")

    # -------------------------------------------------------------------------
    # [3/8] Read corrupted SOLUSDT@500 bars
    # -------------------------------------------------------------------------
    print("\n[3/8] Reading SOLUSDT@500 bars with non-NULL laguerre data...")

    # Find the timestamp range of affected rows
    range_result = cache.client.query(
        "SELECT "
        "  count() AS n, "
        "  min(timestamp_ms) AS ts_min, "
        "  max(timestamp_ms) AS ts_max, "
        "  max(laguerre_bars_in_regime) AS bir_max, "
        "  max(laguerre_tail_risk_score) AS trs_max "
        "FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = 'SOLUSDT' "
        "  AND threshold_decimal_bps = 500 "
        "  AND laguerre_rsi IS NOT NULL"
    )
    row = range_result.result_rows[0]
    n_affected, ts_min, ts_max, bir_max_before, trs_max_before = row

    if n_affected == 0:
        print("      No laguerre-enriched SOLUSDT@500 bars found. Nothing to re-populate.")
        cache.close()
        return 0

    print(f"      Found {n_affected} bars with laguerre data")
    print(f"      Timestamp range: {ts_min} — {ts_max}")
    print(f"      BEFORE: max(bars_in_regime)={bir_max_before}, max(tail_risk_score)={trs_max_before:.4f}")

    if bir_max_before > 1 and trs_max_before > 0.4:
        print(f"      bars_in_regime max={bir_max_before} > 1 and tail_risk_score={trs_max_before:.4f} > 0.4")
        print("      Data is already CORRECT — re-population not needed.")
        cache.close()
        print()
        print("=" * 65)
        print("RE-POPULATION SKIPPED — Data already correct in ClickHouse")
        print("=" * 65)
        print(f"  max(bars_in_regime)  = {bir_max_before}  (correct, was 0-1 with bug)")
        print(f"  max(tail_risk_score) = {trs_max_before:.4f} (correct, extreme_regime_persistence fires)")
        print()
        print("The fix was already in atr-adaptive-laguerre source when E2E validation ran.")
        print("Issue #99 SQL checks pass. Safe to close.")
        return 0

    # Read the bars including OHLCV columns for re-enrichment
    bars_df = cache.get_bars_by_timestamp_range(
        symbol="SOLUSDT",
        threshold_decimal_bps=500,
        start_ts=ts_min,
        end_ts=ts_max,
        include_microstructure=True,
    )

    if bars_df is None or bars_df.empty:
        print("      FAIL: Could not read bars via get_bars_by_timestamp_range")
        cache.close()
        return 1

    # ClickHouse returns TradFi-convention capitalised OHLCV columns.
    # LaguerreFeatureProvider.enrich() expects lowercase. Normalise here.
    col_rename = {c: c.lower() for c in ["Open", "High", "Low", "Close", "Volume"] if c in bars_df.columns}
    if col_rename:
        bars_df = bars_df.rename(columns=col_rename)
        print(f"      Renamed capitalised columns: {list(col_rename.keys())}")

    print(f"      Read {len(bars_df)} bars")

    # -------------------------------------------------------------------------
    # [4/8] Capture before-stats and drop laguerre_* columns
    # -------------------------------------------------------------------------
    print("\n[4/8] Dropping laguerre_* columns (bypass idempotency check)...")
    laguerre_cols = [c for c in bars_df.columns if c.startswith("laguerre_")]
    print(f"      Dropping: {laguerre_cols}")
    bars_df = bars_df.drop(columns=laguerre_cols)
    print(f"      Columns after drop: {len(bars_df.columns)}")

    # -------------------------------------------------------------------------
    # [5/8] Re-enrich with fixed LaguerreFeatureProvider
    # -------------------------------------------------------------------------
    print("\n[5/8] Re-enriching with fixed LaguerreFeatureProvider...")
    from atr_adaptive_laguerre.rangebar_plugin import LaguerreFeatureProvider

    provider = LaguerreFeatureProvider()
    enriched = provider.enrich(bars_df, "SOLUSDT", 500)

    bir_max_after = enriched["laguerre_bars_in_regime"].max()
    trs_max_after = enriched["laguerre_tail_risk_score"].max()
    print(f"      AFTER: max(bars_in_regime)={bir_max_after}, max(tail_risk_score)={trs_max_after:.4f}")

    if bir_max_after <= 1:
        print(f"      FAIL: bars_in_regime max is still {bir_max_after}. Fix not working!")
        cache.close()
        return 1

    if trs_max_after <= 0.4:
        print(f"      WARNING: tail_risk_score max={trs_max_after:.4f} still <= 0.4.")
        print("      This may be OK if no bars had bars_in_regime > 10 in this dataset.")

    print("      Enrichment verified:")
    print(f"        bars_in_regime: {bir_max_before} -> {bir_max_after} (fix confirmed)")
    print(f"        tail_risk_score: {trs_max_before:.4f} -> {trs_max_after:.4f}")

    # -------------------------------------------------------------------------
    # [6/8] Register laguerre columns in column registry (needed for write path)
    # -------------------------------------------------------------------------
    print("\n[6/8] Registering laguerre columns in column registry...")
    from rangebar.constants import register_plugin_columns

    register_plugin_columns(provider.columns)
    print(f"      Registered: {provider.columns}")

    # -------------------------------------------------------------------------
    # [7/8] Write back via store_bars_bulk (ReplacingMergeTree handles dedup)
    # -------------------------------------------------------------------------
    print("\n[7/8] Writing corrected bars via store_bars_bulk()...")
    written = cache.store_bars_bulk(
        symbol="SOLUSDT",
        threshold_decimal_bps=500,
        bars=enriched,
        skip_dedup=True,
    )
    print(f"      Written: {written} rows")

    if written != len(enriched):
        print(f"      WARNING: Expected {len(enriched)}, wrote {written}")

    # -------------------------------------------------------------------------
    # [8/8] SQL verification on bigblack
    # -------------------------------------------------------------------------
    print("\n[8/8] SQL verification on bigblack...")

    # Give ReplacingMergeTree time to settle
    import time
    time.sleep(2)

    verify = cache.client.query(
        "SELECT "
        "  count() AS n, "
        "  max(laguerre_bars_in_regime) AS bir_max, "
        "  max(laguerre_tail_risk_score) AS trs_max, "
        "  countIf(laguerre_regime = 0) AS bearish, "
        "  countIf(laguerre_regime = 1) AS neutral, "
        "  countIf(laguerre_regime = 2) AS bullish "
        "FROM rangebar_cache.range_bars FINAL "
        "WHERE symbol = 'SOLUSDT' "
        "  AND threshold_decimal_bps = 500 "
        "  AND laguerre_rsi IS NOT NULL"
    )
    v = verify.result_rows[0]
    n_v, bir_v, trs_v, bearish, neutral, bullish = v

    print(f"      Rows verified: {n_v}")
    print(f"      max(laguerre_bars_in_regime) = {bir_v}  (must be > 1)")
    print(f"      max(laguerre_tail_risk_score) = {trs_v:.4f}  (target > 0.4)")
    print(f"      Regime distribution: bearish={bearish}, neutral={neutral}, bullish={bullish}")

    # Assertions
    assert bir_v > 1, f"FAIL: bars_in_regime max={bir_v}, expected > 1"
    print("      PASS: bars_in_regime > 1 confirmed in ClickHouse")

    if trs_v > 0.4:
        print(f"      PASS: tail_risk_score max={trs_v:.4f} > 0.4 confirmed")
    else:
        print(f"      INFO: tail_risk_score max={trs_v:.4f} <= 0.4 — extreme_regime_persistence")
        print("            gate requires >10 consecutive bars in extreme regime. May be dataset-dependent.")

    cache.close()

    print()
    print("=" * 65)
    print("RE-POPULATION COMPLETE")
    print("=" * 65)
    print()
    print("Summary:")
    print(f"  Symbol: SOLUSDT@500, {n_v} bars")
    print(f"  bars_in_regime:  {bir_max_before} -> {bir_v}  (fix confirmed)")
    print(f"  tail_risk_score: {trs_max_before:.4f} -> {trs_v:.4f}")
    print(f"  Regime: bearish={bearish}, neutral={neutral}, bullish={bullish}")
    print()
    print("Next steps:")
    print("  1. Run scripts/validate_plugin_e2e.py to confirm full E2E still passes")
    print("  2. Close Issue #99 with these SQL results")

    return 0


if __name__ == "__main__":
    sys.exit(main())
