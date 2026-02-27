"""Tests for Issue #112: Oversized bar write guard.

Validates that _guard_bar_range_invariant() rejects bars exceeding
the threshold * multiplier range invariant at the ClickHouse write boundary.
"""

import pandas as pd
import pytest


def test_guard_rejects_oversized_bar():
    """Bar with range > 3x threshold should be rejected."""
    from rangebar.clickhouse.bulk_operations import _guard_bar_range_invariant

    # 250 dbps = 0.25%. 3x = 0.75%. Bar with 1% range should fail.
    df = pd.DataFrame(
        {
            "Open": [50000.0],
            "High": [50500.0],  # 1.0% range (500/50000)
            "Low": [50000.0],
            "Close": [50450.0],
            "Volume": [10.0],
        }
    )
    with pytest.raises(ValueError, match="Issue #112"):
        _guard_bar_range_invariant(df, threshold_decimal_bps=250, multiplier=3)


def test_guard_accepts_valid_bar():
    """Bar within threshold should pass."""
    from rangebar.clickhouse.bulk_operations import _guard_bar_range_invariant

    # 250 dbps = 0.25%. 3x = 0.75%. Bar with 0.2% range should pass.
    df = pd.DataFrame(
        {
            "Open": [50000.0],
            "High": [50100.0],  # 0.2% range
            "Low": [50000.0],
            "Close": [50090.0],
            "Volume": [10.0],
        }
    )
    # Should not raise
    _guard_bar_range_invariant(df, threshold_decimal_bps=250, multiplier=3)


def test_guard_skips_when_no_threshold():
    """When threshold_decimal_bps is None, skip validation."""
    from rangebar.clickhouse.bulk_operations import _guard_bar_range_invariant

    df = pd.DataFrame(
        {
            "Open": [50000.0],
            "High": [60000.0],  # 20% range — would fail any threshold
            "Low": [50000.0],
        }
    )
    # Should not raise
    _guard_bar_range_invariant(df, threshold_decimal_bps=None)


def test_guard_works_with_lowercase_columns():
    """Internal DataFrames use lowercase OHLC columns."""
    from rangebar.clickhouse.bulk_operations import _guard_bar_range_invariant

    df = pd.DataFrame(
        {
            "open": [50000.0],
            "high": [50500.0],  # 1.0% range
            "low": [50000.0],
            "close": [50450.0],
        }
    )
    with pytest.raises(ValueError, match="Issue #112"):
        _guard_bar_range_invariant(df, threshold_decimal_bps=250, multiplier=3)


def test_run_write_guards_includes_bar_range():
    """_run_write_guards() should call _guard_bar_range_invariant()."""
    from rangebar.clickhouse.bulk_operations import _run_write_guards

    df = pd.DataFrame(
        {
            "Open": [50000.0],
            "High": [50500.0],  # 1.0% range
            "Low": [50000.0],
            "Close": [50450.0],
            "Volume": [10.0],
            "close_time_ms": [1640995200000],
        }
    )
    with pytest.raises(ValueError, match="Issue #112"):
        _run_write_guards(df, threshold_decimal_bps=250)
