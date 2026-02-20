#!/usr/bin/env python3
"""Re-populate SOLUSDT@500 laguerre_* columns with corrected values (Issue #99).

This script fixes ClickHouse rows that were written with the buggy
bars_in_regime formula (always 0 or 1). It uses the real LaguerreFeatureProvider
discovered via entry points, which must be the fixed version where
bars_in_regime = g.groupby(g).cumcount() + 1.

Validation assertions:
  - bars_in_regime max > 1  (proves consecutive count is working)
  - tail_risk_score max > 0.4  (proves extreme_regime_persistence now fires)

Usage:
    uv run python scripts/repopulate_laguerre_corrected.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 65)
    print("Laguerre Re-population — Issue #99 Fix Verification")
    print("=" * 65)
    print()

    # -------------------------------------------------------------------------
    # [1/6] Discover real LaguerreFeatureProvider via entry points
    # -------------------------------------------------------------------------
    from rangebar.plugins.loader import discover_providers

    providers = discover_providers()
    print(f"[1/6] Discovered {len(providers)} provider(s)")

    laguerre = None
    for p in providers:
        print(f"      - {p.name} v{p.version} ({len(p.columns)} columns, min_bars={p.min_bars})")
        if p.name == "laguerre":
            laguerre = p

    if laguerre is None:
        print("\n  FAIL: LaguerreFeatureProvider not discovered.")
        print("  Install atr-adaptive-laguerre editable: uv add --editable ~/eon/atr-adaptive-laguerre")
        return 1

    # Verify the fix is present
    import inspect

    import atr_adaptive_laguerre.features.feature_expander as fe
    src = inspect.getsource(fe.FeatureExpander._extract_regimes)
    if "cumcount" not in src:
        print("\n  FAIL: bars_in_regime fix not present — 'cumcount' not found in _extract_regimes.")
        print("  Install the fixed version from ~/eon/atr-adaptive-laguerre (v2.4.1+)")
        return 1
    print("\n      Fix confirmed: 'cumcount' found in _extract_regimes ✓")

    # -------------------------------------------------------------------------
    # [2/6] Connect to ClickHouse
    # -------------------------------------------------------------------------
    from rangebar.clickhouse import RangeBarCache

    print("\n[2/6] Connecting to ClickHouse (bigblack)...")
    try:
        cache = RangeBarCache()
    except (OSError, RuntimeError, ConnectionError) as e:
        print(f"      SKIP: Cannot connect: {e}")
        return 0
    print("      Connected.")

    # -------------------------------------------------------------------------
    # [3/6] Read 1,000 SOLUSDT@500 bars
    # -------------------------------------------------------------------------
    print("\n[3/6] Reading 1,000 SOLUSDT@500 bars...")
    bars_df, total = cache.get_n_bars(
        symbol="SOLUSDT",
        threshold_decimal_bps=500,
        n_bars=1000,
        include_plugin_features=False,
    )
    if bars_df is None or bars_df.empty:
        print("      SKIP: No SOLUSDT@500 bars available.")
        cache.close()
        return 0

    print(f"      Read {len(bars_df)} bars ({bars_df.index[0]} → {bars_df.index[-1]})")

    # -------------------------------------------------------------------------
    # [4/6] Enrich with corrected LaguerreFeatureProvider
    # -------------------------------------------------------------------------
    print("\n[4/6] Enriching with corrected LaguerreFeatureProvider...")

    # Drop any existing laguerre_* columns to bypass idempotency check
    laguerre_cols_present = [c for c in bars_df.columns if c.startswith("laguerre_")]
    if laguerre_cols_present:
        bars_df = bars_df.drop(columns=laguerre_cols_present)
        print(f"      Dropped pre-existing laguerre columns: {laguerre_cols_present}")

    # enrich() expects lowercase column names (open/high/low/close/volume)
    # get_n_bars() returns capitalized (Open/High/Low/Close/Volume) — normalize here
    bars_df.columns = [c.lower() for c in bars_df.columns]

    enriched = laguerre.enrich(bars_df, "SOLUSDT", 500)

    # Check all 6 columns are present
    for col in laguerre.columns:
        assert col in enriched.columns, f"Missing column: {col}"

    nonnull = enriched["laguerre_rsi"].notna().sum()
    print(f"      Enriched: {nonnull}/{len(enriched)} non-null rows (warmup = {laguerre.min_bars} bars)")

    # Key assertions for the fix
    bir_max = enriched["laguerre_bars_in_regime"].max()
    trs_max = enriched["laguerre_tail_risk_score"].max()
    print()
    print(f"      laguerre_bars_in_regime  max = {bir_max}  (must be > 1)")
    print(f"      laguerre_tail_risk_score max = {trs_max:.4f}  (must be > 0.4)")
    print()

    bir_ok = bir_max > 1
    trs_ok = trs_max > 0.4
    print(f"      bars_in_regime fix:    {'✓ PASS' if bir_ok else '✗ FAIL'}")
    print(f"      tail_risk_score fix:   {'✓ PASS' if trs_ok else '✗ FAIL'}")

    if not (bir_ok and trs_ok):
        print("\n  FAIL: Fix assertions failed. Do NOT write to ClickHouse.")
        cache.close()
        return 1

    # Full column stats
    print()
    print("      Full column stats:")
    for col in laguerre.columns:
        s = enriched[col].dropna()
        print(f"        {col:<35s} min={s.min():.4f}  max={s.max():.4f}  avg={s.mean():.4f}  non-null={len(s)}")

    # -------------------------------------------------------------------------
    # [5/6] Write corrected values back to ClickHouse
    # -------------------------------------------------------------------------
    print("\n[5/6] Writing corrected bars to ClickHouse...")

    # Register laguerre columns in the column registry before writing
    from rangebar.constants import register_plugin_columns
    register_plugin_columns(laguerre.columns)

    # Migrate schema (idempotent — columns already exist)
    from rangebar.plugins.migration import migrate_plugin_columns
    migrate_plugin_columns(cache.client, [laguerre])
    print("      Schema migration: OK (columns already exist, no-op)")

    written = cache.store_bars_bulk(
        symbol="SOLUSDT",
        threshold_decimal_bps=500,
        bars=enriched,
        skip_dedup=True,
    )
    print(f"      Written: {written} rows")
    assert written == len(enriched), f"Write count mismatch: expected {len(enriched)}, got {written}"

    # -------------------------------------------------------------------------
    # [6/6] SQL verification via FINAL
    # -------------------------------------------------------------------------
    print("\n[6/6] SQL verification on BigBlack (FINAL)...")

    result = cache.client.query("""
        SELECT
            count()                              AS total_nonnull,
            max(laguerre_bars_in_regime)         AS bir_max,
            max(laguerre_tail_risk_score)        AS trs_max,
            min(laguerre_rsi)                    AS rsi_min,
            max(laguerre_rsi)                    AS rsi_max,
            countIf(laguerre_regime = 0)         AS regime_bearish,
            countIf(laguerre_regime = 1)         AS regime_neutral,
            countIf(laguerre_regime = 2)         AS regime_bullish
        FROM rangebar_cache.range_bars FINAL
        WHERE symbol = 'SOLUSDT'
          AND threshold_decimal_bps = 500
          AND laguerre_rsi IS NOT NULL
    """)
    row = result.result_rows[0]
    (total_nonnull, bir_max_sql, trs_max_sql,
     rsi_min, rsi_max, bearish, neutral, bullish) = row

    print()
    print("      ClickHouse SQL verification (FINAL):")
    print(f"        total non-null rows:          {total_nonnull}")
    print(f"        max(laguerre_bars_in_regime): {bir_max_sql}  {'✓' if bir_max_sql > 1 else '✗ FAIL'}")
    print(f"        max(laguerre_tail_risk_score):{trs_max_sql:.4f}  {'✓' if trs_max_sql > 0.4 else '✗ FAIL'}")
    print(f"        laguerre_rsi range:           [{rsi_min:.4f}, {rsi_max:.4f}]  {'✓' if rsi_min >= 0 and rsi_max <= 1 else '✗'}")
    print(f"        regime dist:                  bearish={bearish} neutral={neutral} bullish={bullish}")

    sql_ok = (bir_max_sql > 1) and (trs_max_sql > 0.4) and (rsi_min >= 0) and (rsi_max <= 1)

    cache.close()

    print()
    print("=" * 65)
    if sql_ok:
        print("RE-POPULATION PASSED — All Issue #99 SQL assertions verified")
        print()
        print("Results to paste in Issue #99 closing comment:")
        print(f"  max(laguerre_bars_in_regime)  = {bir_max_sql}")
        print(f"  max(laguerre_tail_risk_score) = {trs_max_sql:.4f}")
        print(f"  regime: bearish={bearish}, neutral={neutral}, bullish={bullish}")
        print(f"  total non-null bars: {total_nonnull}")
    else:
        print("RE-POPULATION FAILED — SQL assertions not satisfied")
    print("=" * 65)

    return 0 if sql_ok else 1


if __name__ == "__main__":
    sys.exit(main())
