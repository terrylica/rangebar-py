"""Tests for MEM-013: Long range date protection (Issue #69).

MEM-013 guards prevent OOM on long date ranges (>30 days) by forcing
ClickHouse-first access patterns. These tests verify the guard behaviors.
"""

from __future__ import annotations

import pytest
from rangebar import LONG_RANGE_DAYS, get_range_bars


class TestMEM013LongRangeGuard:
    """Test MEM-013 guard behaviors."""

    def test_long_range_requires_cache_enabled(self) -> None:
        """Long ranges with use_cache=False raise ValueError."""
        with pytest.raises(ValueError, match="use_cache=True"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-12-31",
                threshold_decimal_bps=1000,
                use_cache=False,
            )

    def test_long_range_empty_cache_raises(self) -> None:
        """Long ranges with empty cache raise ValueError with guidance."""
        # Use a threshold unlikely to have cached data (99999 dbps)
        with pytest.raises(ValueError, match="populate_cache_resumable"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-12-31",
                threshold_decimal_bps=99999,
            )

    def test_short_range_accepts_use_cache_false(self) -> None:
        """Short ranges (<=30 days) do not raise MEM-013 guard.

        Note: This test only verifies the MEM-013 guard doesn't trigger.
        Actual data fetch may fail due to missing tick data, which is fine.
        We use pytest.raises with does_not_match to ensure MEM-013 isn't triggered.
        """
        # 30 days is exactly at the threshold - should not raise MEM-013 guard
        # If ValueError is raised, it must NOT be the MEM-013 guard message
        with pytest.raises(
            (ValueError, FileNotFoundError),
            match=r"^(?!.*use_cache=True).*$",
        ):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-30",  # Exactly 30 days
                threshold_decimal_bps=1000,
                use_cache=False,
                fetch_if_missing=False,
            )

    def test_long_range_days_constant_value(self) -> None:
        """LONG_RANGE_DAYS constant is accessible and equals 30."""
        assert LONG_RANGE_DAYS == 30

    def test_long_range_days_constant_type(self) -> None:
        """LONG_RANGE_DAYS is an integer."""
        assert isinstance(LONG_RANGE_DAYS, int)

    def test_exactly_31_days_triggers_guard(self) -> None:
        """31-day range triggers MEM-013 guard."""
        with pytest.raises(ValueError, match="use_cache=True"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-31",  # 31 days (> LONG_RANGE_DAYS)
                threshold_decimal_bps=1000,
                use_cache=False,
            )


class TestMEM013ErrorMessages:
    """Test MEM-013 error message quality."""

    def test_error_mentions_populate_cache_resumable(self) -> None:
        """Error message guides user to populate_cache_resumable()."""
        with pytest.raises(ValueError, match="populate_cache_resumable"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-12-31",
                threshold_decimal_bps=99999,  # Uncached threshold
            )

    def test_error_mentions_day_count(self) -> None:
        """Error message includes the actual day count."""
        with pytest.raises(ValueError, match=r"365.*days|days.*365"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-12-31",
                threshold_decimal_bps=1000,
                use_cache=False,
            )
