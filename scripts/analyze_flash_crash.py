#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pandas", "numpy"]
# ///
"""
Flash Crash Analysis: Oct 10, 2025 (Issue #36)

Statistical comparison of timestamp-gated (v9) vs legacy (v8) range bar behavior
during the Oct 10, 2025 BTCUSDT flash crash (4.3% drop in sub-millisecond).

Usage:
    uv run scripts/analyze_flash_crash.py
"""

from __future__ import annotations

import warnings


def analyze_flash_crash() -> tuple:
    """Full statistical analysis of timestamp gating impact on Oct 10 flash crash."""

    import rangebar

    print("=" * 70)
    print("FLASH CRASH ANALYSIS: Oct 10, 2025")
    print("=" * 70)
    print("\nFetching data and generating bars (this may take a few minutes)...")

    # Generate both behaviors
    try:
        df_legacy = rangebar.get_range_bars(
            "BTCUSDT",
            "2025-10-10",
            "2025-10-11",
            threshold_decimal_bps=100,
            prevent_same_timestamp_close=False,  # Legacy v8 behavior
            include_microstructure=True,  # Include duration_us
        )
    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"\n[ERROR] Failed to fetch legacy bars: {e}")
        print("Make sure you have network connectivity and tick data cached.")
        return None, None

    print(f"Legacy (v8) bars fetched: {len(df_legacy):,}")

    try:
        df_new = rangebar.get_range_bars(
            "BTCUSDT",
            "2025-10-10",
            "2025-10-11",
            threshold_decimal_bps=100,
            prevent_same_timestamp_close=True,  # New v9 behavior (default)
            include_microstructure=True,  # Include duration_us
        )
    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"\n[ERROR] Failed to fetch new bars: {e}")
        return None, None

    print(f"New (v9) bars fetched: {len(df_new):,}")

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # 1. Bar count comparison
    print("\n1. BAR COUNT COMPARISON")
    print(f"   Legacy (v8):     {len(df_legacy):,} bars")
    print(f"   New (v9):        {len(df_new):,} bars")
    reduction = (1 - len(df_new) / len(df_legacy)) * 100 if len(df_legacy) > 0 else 0
    print(f"   Reduction:       {reduction:.1f}%")

    # 2. Duplicate timestamp analysis
    print("\n2. DUPLICATE TIMESTAMPS")
    dup_legacy = df_legacy.index.duplicated().sum()
    dup_new = df_new.index.duplicated().sum()
    print(f"   Legacy:          {dup_legacy:,} duplicates")
    print(f"   New:             {dup_new:,} duplicates (should be 0)")
    if dup_new > 0:
        print("   [WARNING] New behavior should have NO duplicate timestamps!")
    else:
        print("   [PASS] New behavior has unique timestamps")

    # 3. Hourly breakdown (if we have hour info)
    print("\n3. HOURLY BREAKDOWN")
    for hour in range(24):
        legacy_h = len(df_legacy[df_legacy.index.hour == hour])
        new_h = len(df_new[df_new.index.hour == hour])
        if legacy_h > 100 or new_h > 100:  # Only show significant hours
            print(f"   Hour {hour:02d}:  Legacy={legacy_h:>6,}, New={new_h:>6,}")

    # 4. Max bars at single timestamp
    print("\n4. MAX BARS AT SINGLE TIMESTAMP")
    max_legacy = df_legacy.index.value_counts().max() if len(df_legacy) > 0 else 0
    max_new = df_new.index.value_counts().max() if len(df_new) > 0 else 0
    print(f"   Legacy:          {max_legacy} bars at same timestamp")
    print(f"   New:             {max_new} bar(s) at same timestamp")
    if max_new > 1:
        print("   [WARNING] New behavior should have at most 1 bar per timestamp!")
    else:
        print("   [PASS] New behavior has unique timestamps")

    # 5. Duration distribution
    print("\n5. DURATION DISTRIBUTION (microseconds)")
    if "duration_us" in df_legacy.columns and "duration_us" in df_new.columns:
        zero_dur_legacy = (df_legacy["duration_us"] == 0).sum()
        zero_dur_new = (df_new["duration_us"] == 0).sum()
        print(
            f"   Zero-duration bars (Legacy): {zero_dur_legacy:,} "
            f"({zero_dur_legacy/len(df_legacy)*100:.2f}%)"
        )
        print(
            f"   Zero-duration bars (New):    {zero_dur_new:,} "
            f"({zero_dur_new/len(df_new)*100:.2f}%)"
        )

        print("\n   Legacy duration percentiles:")
        print(f"      p50: {df_legacy['duration_us'].quantile(0.5):,.0f} us")
        print(f"      p90: {df_legacy['duration_us'].quantile(0.9):,.0f} us")
        print(f"      p99: {df_legacy['duration_us'].quantile(0.99):,.0f} us")

        print("\n   New duration percentiles:")
        print(f"      p50: {df_new['duration_us'].quantile(0.5):,.0f} us")
        print(f"      p90: {df_new['duration_us'].quantile(0.9):,.0f} us")
        print(f"      p99: {df_new['duration_us'].quantile(0.99):,.0f} us")
    else:
        print(
            "   [INFO] duration_us column not present (use include_microstructure=True)"
        )

    # 6. Range size distribution (dbps)
    print("\n6. RANGE SIZE DISTRIBUTION (decimal basis points)")
    df_legacy_copy = df_legacy.copy()
    df_new_copy = df_new.copy()
    # Calculate range in dbps (1 dbps = 0.001% = 0.00001)
    df_legacy_copy["range_dbps"] = (
        (df_legacy_copy["High"] - df_legacy_copy["Low"])
        / df_legacy_copy["Open"]
        * 100000  # Convert to dbps
    )
    df_new_copy["range_dbps"] = (
        (df_new_copy["High"] - df_new_copy["Low"]) / df_new_copy["Open"] * 100000
    )

    print(f"   Legacy max range: {df_legacy_copy['range_dbps'].max():.1f} dbps")
    print(f"   New max range:    {df_new_copy['range_dbps'].max():.1f} dbps")
    print("   (Larger range in new = consolidated cascade bars)")

    # 7. Price coverage verification
    print("\n7. PRICE COVERAGE")
    print(
        f"   Legacy price range: ${df_legacy['Low'].min():,.2f} - "
        f"${df_legacy['High'].max():,.2f}"
    )
    print(
        f"   New price range:    ${df_new['Low'].min():,.2f} - "
        f"${df_new['High'].max():,.2f}"
    )
    print("   (Should be identical - no price data lost)")

    # 8. Volume coverage
    print("\n8. VOLUME VERIFICATION")
    total_vol_legacy = df_legacy["Volume"].sum()
    total_vol_new = df_new["Volume"].sum()
    vol_diff_pct = abs(total_vol_legacy - total_vol_new) / total_vol_legacy * 100
    print(f"   Legacy total volume: {total_vol_legacy:,.2f}")
    print(f"   New total volume:    {total_vol_new:,.2f}")
    print(f"   Difference:          {vol_diff_pct:.4f}%")

    # 9. Index uniqueness assertion
    print("\n9. INDEX UNIQUENESS")
    is_unique = df_new.index.is_unique
    print(f"   df_new.index.is_unique = {is_unique}")
    if is_unique:
        print("   [PASS] New behavior has unique DatetimeIndex")
    else:
        print("   [FAIL] New behavior must have unique index")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return df_legacy, df_new


def main() -> int:
    """Main entry point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_legacy, df_new = analyze_flash_crash()

    if df_legacy is None or df_new is None:
        return 1

    print("\n[INFO] DataFrames returned for further analysis:")
    print(f"  df_legacy.shape = {df_legacy.shape}")
    print(f"  df_new.shape = {df_new.shape}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
