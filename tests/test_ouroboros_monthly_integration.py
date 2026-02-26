"""Test monthly ouroboros boundary logic and consistency. Issue #97.

Tests cover:
- Monthly boundary detection for all 12 months
- Monthly segment iteration
- Week boundary consistency (Sunday, not Monday) — regression for Gap 5 fix
- February leap year boundary
- Negative tests for non-day-1 dates
"""

from datetime import date

import pytest
from rangebar.checkpoint import _is_ouroboros_boundary
from rangebar.ouroboros import (
    get_ouroboros_boundaries,
    iter_ouroboros_segments,
)


class TestMonthlyBoundaryDetectionAllMonths:
    """Test _is_ouroboros_boundary returns True on day 1, False on days 2-31."""

    @pytest.mark.parametrize("month", range(1, 13))
    def test_day1_is_boundary(self, month: int):
        """Day 1 of every month is a monthly boundary."""
        date_str = f"2024-{month:02d}-01"
        assert _is_ouroboros_boundary(date_str, "month") is True

    @pytest.mark.parametrize("month", range(1, 13))
    def test_day15_is_not_boundary(self, month: int):
        """Day 15 of every month is NOT a monthly boundary."""
        date_str = f"2024-{month:02d}-15"
        assert _is_ouroboros_boundary(date_str, "month") is False

    def test_day1_is_not_year_boundary(self):
        """Feb 1 is a month boundary but NOT a year boundary."""
        assert _is_ouroboros_boundary("2024-02-01", "month") is True
        assert _is_ouroboros_boundary("2024-02-01", "year") is False

    def test_jan1_is_both_month_and_year_boundary(self):
        """Jan 1 is both a month boundary and a year boundary."""
        assert _is_ouroboros_boundary("2024-01-01", "month") is True
        assert _is_ouroboros_boundary("2024-01-01", "year") is True


class TestMonthlySegmentIteration:
    """Test iter_ouroboros_segments produces correct date ranges for monthly mode."""

    def test_three_month_span(self):
        """3-month span produces 3 segments."""
        segments = list(iter_ouroboros_segments(
            date(2024, 1, 1), date(2024, 3, 31), "month"
        ))
        assert len(segments) == 3

        # First segment starts Jan 1
        seg_start, _, boundary = segments[0]
        assert seg_start.month == 1
        assert seg_start.day == 1
        assert boundary is not None
        assert boundary.reason == "month_boundary"

        # Second segment starts Feb 1
        seg_start, _, boundary = segments[1]
        assert seg_start.month == 2
        assert seg_start.day == 1

        # Third segment starts Mar 1
        seg_start, _, boundary = segments[2]
        assert seg_start.month == 3
        assert seg_start.day == 1

    def test_single_month(self):
        """A date range within one month produces 1 segment."""
        segments = list(iter_ouroboros_segments(
            date(2024, 3, 5), date(2024, 3, 25), "month"
        ))
        # No boundaries within range (Mar 1 is before start)
        assert len(segments) == 1
        _, _, boundary = segments[0]
        assert boundary is None  # No boundary at start (mid-month)

    def test_full_year_produces_12_boundaries(self):
        """Full year should have 12 month boundaries."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 12, 31), "month"
        )
        assert len(boundaries) == 12
        for i, b in enumerate(boundaries):
            assert b.timestamp.month == i + 1
            assert b.timestamp.day == 1
            assert b.reason == "month_boundary"


class TestWeekBoundaryConsistency:
    """Regression test for Gap 5 fix: week boundaries use Sunday (weekday=6)."""

    def test_checkpoint_boundary_is_sunday(self):
        """_is_ouroboros_boundary uses Sunday for week mode."""
        # 2024-01-07 is a Sunday
        assert _is_ouroboros_boundary("2024-01-07", "week") is True
        # 2024-01-08 is a Monday — should NOT be a boundary
        assert _is_ouroboros_boundary("2024-01-08", "week") is False
        # 2024-01-06 is a Saturday — should NOT be a boundary
        assert _is_ouroboros_boundary("2024-01-06", "week") is False

    def test_ouroboros_module_agrees_on_sunday(self):
        """get_ouroboros_boundaries also uses Sunday for week mode."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 1, 14), "week"
        )
        for b in boundaries:
            # All week boundaries should be Sundays
            assert b.timestamp.weekday() == 6, (
                f"Week boundary {b.timestamp} is weekday {b.timestamp.weekday()}, "
                f"expected 6 (Sunday)"
            )

    def test_checkpoint_and_ouroboros_agree(self):
        """_is_ouroboros_boundary and get_ouroboros_boundaries agree on Sundays."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        boundaries = get_ouroboros_boundaries(start, end, "week")

        boundary_dates = {b.timestamp.date() for b in boundaries}

        # Check every day in range
        from datetime import timedelta

        d = start
        while d <= end:
            date_str = d.strftime("%Y-%m-%d")
            is_boundary = _is_ouroboros_boundary(date_str, "week")
            in_ouroboros = d in boundary_dates
            assert is_boundary == in_ouroboros, (
                f"Disagreement on {date_str}: checkpoint says {is_boundary}, "
                f"ouroboros says {in_ouroboros}"
            )
            d += timedelta(days=1)


class TestFebruaryLeapYearBoundary:
    """Feb 1 is a boundary in both leap and non-leap years."""

    def test_leap_year_2024(self):
        assert _is_ouroboros_boundary("2024-02-01", "month") is True
        # Feb 29 exists in leap year but is NOT a boundary
        assert _is_ouroboros_boundary("2024-02-29", "month") is False

    def test_non_leap_year_2023(self):
        assert _is_ouroboros_boundary("2023-02-01", "month") is True
        assert _is_ouroboros_boundary("2023-02-28", "month") is False

    def test_march_1_after_leap_feb(self):
        """Mar 1 is always a boundary regardless of Feb length."""
        assert _is_ouroboros_boundary("2024-03-01", "month") is True
        assert _is_ouroboros_boundary("2023-03-01", "month") is True


class TestMonthlyBoundaryNegative:
    """Comprehensive negative tests: random days 2-28 should not be boundaries."""

    @pytest.mark.parametrize("day", [2, 5, 10, 14, 15, 20, 25, 28])
    @pytest.mark.parametrize("month", [1, 3, 6, 9, 12])
    def test_non_day1_is_not_boundary(self, month: int, day: int):
        date_str = f"2024-{month:02d}-{day:02d}"
        assert _is_ouroboros_boundary(date_str, "month") is False, (
            f"{date_str} should NOT be a monthly boundary"
        )

    def test_last_days_of_months_are_not_boundaries(self):
        """Last day of each month is not a boundary."""
        last_days = [
            "2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30",
            "2024-05-31", "2024-06-30", "2024-07-31", "2024-08-31",
            "2024-09-30", "2024-10-31", "2024-11-30", "2024-12-31",
        ]
        for day_str in last_days:
            assert _is_ouroboros_boundary(day_str, "month") is False
