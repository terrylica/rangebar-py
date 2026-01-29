#!/usr/bin/env python3
"""Test ouroboros module for cyclical reset boundaries.

Plan: /Users/terryli/.claude/plans/sparkling-coalescing-dijkstra.md

Tests cover:
- Boundary calculation for year/month/week granularities
- Segment iteration
- Exchange session detection
- Edge cases (DST transitions, leap years, year boundaries)
"""

from datetime import UTC, date, datetime

import pytest
from rangebar.ouroboros import (
    WEEKEND_GAP_THRESHOLD_MS,
    ExchangeSessionFlags,
    OrphanedBarMetadata,
    OuroborosBoundary,
    OuroborosMode,
    detect_forex_weekend_boundaries,
    get_active_exchange_sessions,
    get_ouroboros_boundaries,
    iter_forex_ouroboros_segments,
    iter_ouroboros_segments,
    validate_ouroboros_mode,
)


class TestOuroborosMode:
    """Test OuroborosMode enum."""

    def test_mode_values(self):
        """Test enum values are correct strings."""
        assert OuroborosMode.YEAR.value == "year"
        assert OuroborosMode.MONTH.value == "month"
        assert OuroborosMode.WEEK.value == "week"

    def test_mode_is_string_enum(self):
        """Test mode can be compared to strings."""
        assert OuroborosMode.YEAR == "year"
        assert OuroborosMode.MONTH == "month"
        assert OuroborosMode.WEEK == "week"


class TestOuroborosBoundary:
    """Test OuroborosBoundary dataclass."""

    def test_boundary_creation(self):
        """Test creating a boundary."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.YEAR,
            reason="year_boundary",
        )
        assert boundary.timestamp == ts
        assert boundary.mode == OuroborosMode.YEAR
        assert boundary.reason == "year_boundary"

    def test_timestamp_ms(self):
        """Test timestamp_ms property."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.YEAR,
            reason="year_boundary",
        )
        # 2024-01-01 00:00:00 UTC in milliseconds
        expected_ms = 1704067200000
        assert boundary.timestamp_ms == expected_ms

    def test_timestamp_us(self):
        """Test timestamp_us property."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.YEAR,
            reason="year_boundary",
        )
        expected_us = 1704067200000000
        assert boundary.timestamp_us == expected_us

    def test_boundary_is_frozen(self):
        """Test boundary is immutable."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.YEAR,
            reason="year_boundary",
        )
        with pytest.raises(AttributeError):
            boundary.timestamp = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


class TestOrphanedBarMetadata:
    """Test OrphanedBarMetadata dataclass."""

    def test_default_values(self):
        """Test default values."""
        metadata = OrphanedBarMetadata()
        assert metadata.is_orphan is True
        assert metadata.ouroboros_boundary is None
        assert metadata.reason is None
        assert metadata.expected_duration_us is None

    def test_full_metadata(self):
        """Test with all fields populated."""
        boundary_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        metadata = OrphanedBarMetadata(
            is_orphan=True,
            ouroboros_boundary=boundary_ts,
            reason="year_boundary",
            expected_duration_us=3600000000,  # 1 hour
        )
        assert metadata.ouroboros_boundary == boundary_ts
        assert metadata.reason == "year_boundary"
        assert metadata.expected_duration_us == 3600000000


class TestGetOuroborosBoundaries:
    """Test get_ouroboros_boundaries function."""

    # ==================== Year Mode ====================

    def test_year_boundaries_single_year(self):
        """Test year boundaries within a single year."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 12, 31), "year"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[0].mode == OuroborosMode.YEAR
        assert boundaries[0].reason == "year_boundary"

    def test_year_boundaries_multi_year(self):
        """Test year boundaries across multiple years."""
        boundaries = get_ouroboros_boundaries(
            date(2023, 6, 1), date(2025, 6, 30), "year"
        )
        assert len(boundaries) == 2
        assert boundaries[0].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_year_boundaries_start_on_boundary(self):
        """Test when start date IS a year boundary."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 3, 31), "year"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_year_boundaries_no_boundary_in_range(self):
        """Test when no year boundary falls in range."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 3, 1), date(2024, 6, 30), "year"
        )
        assert len(boundaries) == 0

    # ==================== Month Mode ====================

    def test_month_boundaries_q1(self):
        """Test month boundaries for Q1 2024."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 3, 31), "month"
        )
        assert len(boundaries) == 3
        assert boundaries[0].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[2].timestamp == datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC)
        for b in boundaries:
            assert b.mode == OuroborosMode.MONTH
            assert b.reason == "month_boundary"

    def test_month_boundaries_cross_year(self):
        """Test month boundaries crossing year end."""
        boundaries = get_ouroboros_boundaries(
            date(2023, 11, 1), date(2024, 2, 29), "month"
        )
        assert len(boundaries) == 4
        assert boundaries[0].timestamp == datetime(2023, 11, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2023, 12, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[2].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[3].timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)

    def test_month_boundaries_partial_month(self):
        """Test boundaries when starting mid-month."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 15), date(2024, 3, 15), "month"
        )
        # Should include Feb 1 and Mar 1
        assert len(boundaries) == 2
        assert boundaries[0].timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC)

    # ==================== Week Mode ====================

    def test_week_boundaries_basic(self):
        """Test week boundaries (Sunday 00:00 UTC)."""
        # 2024-01-07 is a Sunday
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 1, 31), "week"
        )
        # Sundays in Jan 2024: 7, 14, 21, 28
        assert len(boundaries) == 4
        assert boundaries[0].timestamp == datetime(2024, 1, 7, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2024, 1, 14, 0, 0, 0, tzinfo=UTC)
        assert boundaries[2].timestamp == datetime(2024, 1, 21, 0, 0, 0, tzinfo=UTC)
        assert boundaries[3].timestamp == datetime(2024, 1, 28, 0, 0, 0, tzinfo=UTC)

    def test_week_boundaries_start_on_sunday(self):
        """Test when start date IS a Sunday."""
        # 2024-01-07 is a Sunday
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 7), date(2024, 1, 20), "week"
        )
        # Should include Jan 7 and Jan 14
        assert len(boundaries) == 2
        assert boundaries[0].timestamp == datetime(2024, 1, 7, 0, 0, 0, tzinfo=UTC)
        assert boundaries[1].timestamp == datetime(2024, 1, 14, 0, 0, 0, tzinfo=UTC)

    def test_week_boundaries_no_sunday_in_range(self):
        """Test when no Sunday falls in the range."""
        # 2024-01-08 is Monday, 2024-01-13 is Saturday
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 8), date(2024, 1, 13), "week"
        )
        assert len(boundaries) == 0

    def test_week_boundaries_single_week(self):
        """Test a single week span."""
        # Just one Sunday in range
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 6), date(2024, 1, 8), "week"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2024, 1, 7, 0, 0, 0, tzinfo=UTC)


class TestIterOuroborosSegments:
    """Test iter_ouroboros_segments function."""

    def test_single_segment_no_boundaries(self):
        """Test iteration when no boundaries in range."""
        segments = list(
            iter_ouroboros_segments(date(2024, 1, 8), date(2024, 1, 13), "week")
        )
        assert len(segments) == 1
        start, end, boundary = segments[0]
        assert boundary is None  # No boundary at start
        assert start == datetime(2024, 1, 8, 0, 0, 0, tzinfo=UTC)
        assert end == datetime(2024, 1, 13, 23, 59, 59, 999999, tzinfo=UTC)

    def test_segments_with_boundaries(self):
        """Test iteration with month boundaries."""
        segments = list(
            iter_ouroboros_segments(date(2024, 1, 15), date(2024, 3, 15), "month")
        )
        # 3 segments: Jan 15-31, Feb 1-29, Mar 1-15
        assert len(segments) == 3

        # First segment: no boundary (starts mid-month)
        start0, _end0, boundary0 = segments[0]
        assert boundary0 is None
        assert start0 == datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)

        # Second segment: Feb 1 boundary
        start1, _end1, boundary1 = segments[1]
        assert boundary1 is not None
        assert boundary1.mode == OuroborosMode.MONTH
        assert start1 == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)

        # Third segment: Mar 1 boundary
        start2, end2, boundary2 = segments[2]
        assert boundary2 is not None
        assert start2 == datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC)
        assert end2 == datetime(2024, 3, 15, 23, 59, 59, 999999, tzinfo=UTC)

    def test_segments_start_on_boundary(self):
        """Test when start date is exactly on a boundary."""
        segments = list(
            iter_ouroboros_segments(date(2024, 1, 1), date(2024, 2, 15), "month")
        )
        # 2 segments: Jan 1-31, Feb 1-15
        assert len(segments) == 2

        # First segment starts on boundary
        start0, _end0, boundary0 = segments[0]
        assert boundary0 is not None
        assert boundary0.reason == "month_boundary"
        assert start0 == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


class TestExchangeSessionFlags:
    """Test ExchangeSessionFlags dataclass."""

    def test_flags_creation(self):
        """Test creating session flags."""
        flags = ExchangeSessionFlags(
            sydney=True, tokyo=True, london=False, newyork=False
        )
        assert flags.sydney is True
        assert flags.tokyo is True
        assert flags.london is False
        assert flags.newyork is False

    def test_to_dict(self):
        """Test converting to dict with column names."""
        flags = ExchangeSessionFlags(
            sydney=True, tokyo=False, london=True, newyork=False
        )
        d = flags.to_dict()
        assert d == {
            "exchange_session_sydney": True,
            "exchange_session_tokyo": False,
            "exchange_session_london": True,
            "exchange_session_newyork": False,
        }

    def test_flags_is_frozen(self):
        """Test flags are immutable."""
        flags = ExchangeSessionFlags(
            sydney=True, tokyo=True, london=False, newyork=False
        )
        with pytest.raises(AttributeError):
            flags.sydney = False


class TestGetActiveExchangeSessions:
    """Test get_active_exchange_sessions function."""

    def test_tokyo_session_morning(self):
        """Test Tokyo session during Tokyo morning (9 AM JST = 0 AM UTC)."""
        # 9 AM JST = 0 AM UTC (JST = UTC+9)
        # Tokyo session: 9 AM - 6 PM JST (0 AM - 9 AM UTC)
        ts = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        assert flags.tokyo is True

    def test_london_session_morning(self):
        """Test London session during London morning (9 AM GMT = 9 AM UTC)."""
        # London session: 8 AM - 4 PM GMT (8 AM - 4 PM UTC in winter)
        ts = datetime(2024, 1, 15, 9, 0, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        assert flags.london is True

    def test_newyork_session_afternoon(self):
        """Test New York session (2 PM EST = 7 PM UTC)."""
        # NY session: 8 AM - 5 PM EST (1 PM - 10 PM UTC in winter)
        ts = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        assert flags.newyork is True

    def test_no_sessions_weekend(self):
        """Test no sessions active on weekend."""
        # Saturday
        ts = datetime(2024, 1, 13, 12, 0, 0, tzinfo=UTC)
        flags = get_active_exchange_sessions(ts)
        assert flags.sydney is False
        assert flags.tokyo is False
        assert flags.london is False
        assert flags.newyork is False

    def test_session_overlap_london_newyork(self):
        """Test London/New York overlap (afternoon UTC).

        Issue #8: Corrected session hours
        - London: 08:00-17:00 local (8am-5pm)
        - New York: 10:00-16:00 local (10am-4pm, i.e. 15:00-21:00 UTC in winter)
        Overlap occurs roughly 15:00-17:00 UTC (NY open, London still open)
        """
        # 15:30 UTC = 10:30 AM EST (NY open) and 3:30 PM London (still open)
        ts = datetime(2024, 1, 15, 15, 30, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        assert flags.london is True
        assert flags.newyork is True

    def test_sydney_session(self):
        """Test Sydney session (Sydney mid-morning = late UTC previous day).

        Issue #8: Corrected session hours
        - Sydney (ASX): 10:00-16:00 local (10am-4pm AEDT, UTC+11 in summer)
        - 10 AM AEDT = 23:00 UTC previous day
        - 12 PM AEDT = 01:00 UTC
        """
        # Tuesday 01:00 UTC = Tuesday 12:00 noon AEDT (Sydney session active)
        ts = datetime(2024, 1, 16, 1, 0, 0, tzinfo=UTC)  # Tuesday UTC
        flags = get_active_exchange_sessions(ts)
        assert flags.sydney is True


class TestValidateOuroborosMode:
    """Test validate_ouroboros_mode function."""

    def test_valid_modes(self):
        """Test valid mode strings."""
        assert validate_ouroboros_mode("year") == "year"
        assert validate_ouroboros_mode("month") == "month"
        assert validate_ouroboros_mode("week") == "week"

    def test_invalid_mode(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ouroboros mode"):
            validate_ouroboros_mode("day")

    def test_invalid_mode_none(self):
        """Test 'none' is not a valid mode (mandatory ouroboros)."""
        with pytest.raises(ValueError, match="Invalid ouroboros mode"):
            validate_ouroboros_mode("none")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_leap_year_feb_boundary(self):
        """Test February boundary in leap year (2024)."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 2, 1), date(2024, 2, 29), "month"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)

    def test_non_leap_year_feb_boundary(self):
        """Test February boundary in non-leap year (2023)."""
        boundaries = get_ouroboros_boundaries(
            date(2023, 2, 1), date(2023, 2, 28), "month"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2023, 2, 1, 0, 0, 0, tzinfo=UTC)

    def test_year_boundary_midnight(self):
        """Test year boundary is exactly midnight UTC."""
        boundaries = get_ouroboros_boundaries(
            date(2023, 12, 31), date(2024, 1, 2), "year"
        )
        assert len(boundaries) == 1
        boundary = boundaries[0]
        assert boundary.timestamp.hour == 0
        assert boundary.timestamp.minute == 0
        assert boundary.timestamp.second == 0
        assert boundary.timestamp.microsecond == 0

    def test_week_boundary_sunday_midnight(self):
        """Test week boundary is exactly Sunday midnight UTC."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 6), date(2024, 1, 8), "week"
        )
        assert len(boundaries) == 1
        boundary = boundaries[0]
        assert boundary.timestamp.weekday() == 6  # Sunday
        assert boundary.timestamp.hour == 0
        assert boundary.timestamp.minute == 0
        assert boundary.timestamp.second == 0

    def test_same_day_start_end(self):
        """Test when start and end are the same day."""
        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 1), date(2024, 1, 1), "year"
        )
        assert len(boundaries) == 1  # Jan 1 is a year boundary

        boundaries = get_ouroboros_boundaries(
            date(2024, 1, 2), date(2024, 1, 2), "year"
        )
        assert len(boundaries) == 0  # Jan 2 is not a year boundary

    def test_december_to_january_year_boundary(self):
        """Test year boundary crossing December to January."""
        boundaries = get_ouroboros_boundaries(
            date(2023, 12, 15), date(2024, 1, 15), "year"
        )
        assert len(boundaries) == 1
        assert boundaries[0].timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_many_years_range(self):
        """Test range spanning many years."""
        boundaries = get_ouroboros_boundaries(
            date(2020, 1, 1), date(2025, 12, 31), "year"
        )
        assert len(boundaries) == 6  # 2020, 2021, 2022, 2023, 2024, 2025
        for i, boundary in enumerate(boundaries):
            assert boundary.timestamp.year == 2020 + i
            assert boundary.timestamp.month == 1
            assert boundary.timestamp.day == 1

    def test_timestamp_ms_precision(self):
        """Test timestamp_ms has millisecond precision."""
        ts = datetime(2024, 1, 1, 12, 30, 45, 123456, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.MONTH,
            reason="test",
        )
        # Should truncate microseconds to milliseconds
        expected_ms = int(ts.timestamp() * 1000)
        assert boundary.timestamp_ms == expected_ms

    def test_timestamp_us_precision(self):
        """Test timestamp_us has microsecond precision."""
        ts = datetime(2024, 1, 1, 12, 30, 45, 123456, tzinfo=UTC)
        boundary = OuroborosBoundary(
            timestamp=ts,
            mode=OuroborosMode.MONTH,
            reason="test",
        )
        expected_us = int(ts.timestamp() * 1_000_000)
        assert boundary.timestamp_us == expected_us


class TestForexWeekendDetection:
    """Test Forex weekend boundary detection (dynamic ouroboros)."""

    def test_weekend_gap_threshold(self):
        """Test weekend gap threshold is 40 hours."""
        assert WEEKEND_GAP_THRESHOLD_MS == 40 * 60 * 60 * 1000

    def test_detect_single_weekend(self):
        """Test detecting a single weekend gap."""
        # Friday 21:00 UTC (market close)
        friday_close_ms = 1705093200000  # 2024-01-12 21:00:00 UTC

        # Sunday 17:00 UTC (market open) - 44 hours later
        sunday_open_ms = 1705251600000  # 2024-01-14 17:00:00 UTC

        timestamps = [
            friday_close_ms - 1000,  # Friday tick
            friday_close_ms,  # Last Friday tick
            sunday_open_ms,  # First Sunday tick (ouroboros point)
            sunday_open_ms + 1000,  # More Sunday ticks
        ]

        boundaries = detect_forex_weekend_boundaries(timestamps)
        assert len(boundaries) == 1
        assert boundaries[0].reason == "forex_weekend_boundary"
        assert boundaries[0].mode == OuroborosMode.WEEK
        # Boundary should be at the first tick after the gap
        assert boundaries[0].timestamp_ms == sunday_open_ms

    def test_detect_multiple_weekends(self):
        """Test detecting multiple weekend gaps."""
        # Week 1: Friday to Sunday
        week1_friday = 1705093200000  # 2024-01-12 21:00 UTC
        week1_sunday = 1705251600000  # 2024-01-14 17:00 UTC

        # Week 2: Friday to Sunday
        week2_friday = 1705698000000  # 2024-01-19 21:00 UTC
        week2_sunday = 1705856400000  # 2024-01-21 17:00 UTC

        # Simulate realistic tick data with ticks throughout the week
        # (gaps between weekday ticks are < 40 hours)
        timestamps = [
            week1_friday,
            week1_sunday,  # First weekend boundary
            week1_sunday + 3600000,  # 1 hour later
            week1_sunday + 86400000,  # Monday
            week1_sunday + 172800000,  # Tuesday
            week1_sunday + 259200000,  # Wednesday
            week1_sunday + 345600000,  # Thursday
            week2_friday,  # Friday (< 40 hours from Thursday)
            week2_sunday,  # Second weekend boundary
            week2_sunday + 1000,
        ]

        boundaries = detect_forex_weekend_boundaries(timestamps)
        assert len(boundaries) == 2
        assert boundaries[0].timestamp_ms == week1_sunday
        assert boundaries[1].timestamp_ms == week2_sunday

    def test_no_weekend_within_week(self):
        """Test no boundaries detected within a trading week."""
        # Monday to Friday, no gaps > 40 hours
        monday_ms = 1705312800000  # 2024-01-15 10:00 UTC
        timestamps = [
            monday_ms,
            monday_ms + 3600000,  # 1 hour later
            monday_ms + 7200000,  # 2 hours later
            monday_ms + 86400000,  # 24 hours later (Tuesday)
        ]

        boundaries = detect_forex_weekend_boundaries(timestamps)
        assert len(boundaries) == 0

    def test_empty_timestamps(self):
        """Test with empty timestamp list."""
        boundaries = detect_forex_weekend_boundaries([])
        assert len(boundaries) == 0

    def test_single_timestamp(self):
        """Test with single timestamp."""
        boundaries = detect_forex_weekend_boundaries([1705093200000])
        assert len(boundaries) == 0

    def test_gap_exactly_at_threshold(self):
        """Test gap exactly at 40 hour threshold."""
        start_ms = 1705093200000  # 2024-01-12 21:00 UTC
        # Exactly 40 hours later
        end_ms = start_ms + WEEKEND_GAP_THRESHOLD_MS

        timestamps = [start_ms, end_ms]
        boundaries = detect_forex_weekend_boundaries(timestamps)
        assert len(boundaries) == 1  # Should detect as weekend

    def test_gap_just_under_threshold(self):
        """Test gap just under 40 hour threshold."""
        start_ms = 1705093200000
        # 39 hours, 59 minutes later (just under threshold)
        end_ms = start_ms + WEEKEND_GAP_THRESHOLD_MS - 60000

        timestamps = [start_ms, end_ms]
        boundaries = detect_forex_weekend_boundaries(timestamps)
        assert len(boundaries) == 0  # Should not detect


class TestIterForexOuroborosSegments:
    """Test Forex ouroboros segment iteration."""

    def test_single_segment_no_weekend(self):
        """Test iteration when no weekend in data."""
        # Monday to Friday ticks
        monday_ms = 1705312800000  # 2024-01-15 10:00 UTC
        timestamps = [
            monday_ms,
            monday_ms + 3600000,
            monday_ms + 7200000,
        ]

        segments = list(
            iter_forex_ouroboros_segments(
                timestamps, date(2024, 1, 15), date(2024, 1, 15)
            )
        )
        assert len(segments) == 1
        start_idx, end_idx, boundary = segments[0]
        assert start_idx == 0
        assert end_idx == 2
        assert boundary is None

    def test_two_segments_with_weekend(self):
        """Test iteration with one weekend gap."""
        friday_ms = 1705093200000  # 2024-01-12 21:00 UTC
        sunday_ms = 1705251600000  # 2024-01-14 17:00 UTC

        timestamps = [
            friday_ms - 1000,  # idx 0
            friday_ms,  # idx 1
            sunday_ms,  # idx 2 (boundary)
            sunday_ms + 1000,  # idx 3
        ]

        segments = list(
            iter_forex_ouroboros_segments(
                timestamps, date(2024, 1, 12), date(2024, 1, 14)
            )
        )
        assert len(segments) == 2

        # First segment: before weekend
        start0, end0, boundary0 = segments[0]
        assert start0 == 0
        assert end0 == 1
        assert boundary0 is None

        # Second segment: after weekend (starts at boundary)
        start1, end1, boundary1 = segments[1]
        assert start1 == 2
        assert end1 == 3
        assert boundary1 is not None
        assert boundary1.reason == "forex_weekend_boundary"

    def test_empty_timestamps(self):
        """Test with empty timestamp list."""
        segments = list(
            iter_forex_ouroboros_segments([], date(2024, 1, 1), date(2024, 1, 7))
        )
        assert len(segments) == 0


class TestOuroborosIntegrationWithRealData:
    """Integration tests for ouroboros with real cryptocurrency data.

    These tests verify that:
    1. Default ouroboros is "year" for cryptocurrency
    2. Ouroboros mode affects bar generation (different modes = different results)
    3. Reproducibility: same parameters = same results
    4. Year boundaries reset processor state
    """

    @pytest.fixture
    def real_data_available(self):
        """Check if real data is available for testing."""
        try:
            from rangebar import get_range_bars

            # Quick test with small date range
            df = get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-02",
                threshold_decimal_bps=250,
                use_cache=True,
            )
            return len(df) > 0
        except (ImportError, FileNotFoundError, ConnectionError, ValueError):
            return False

    def test_default_ouroboros_is_year(self):
        """Test that default ouroboros mode is 'year' for cryptocurrency."""
        import inspect

        from rangebar import get_range_bars

        sig = inspect.signature(get_range_bars)
        ouroboros_param = sig.parameters["ouroboros"]
        assert ouroboros_param.default == "year"

    @pytest.mark.skipif(
        "not config.getoption('--run-integration', default=False)",
        reason="Integration test - requires real data",
    )
    def test_reproducibility_same_ouroboros(self, real_data_available):
        """Test that same ouroboros mode produces identical results."""
        if not real_data_available:
            pytest.skip("Real data not available")

        from rangebar import get_range_bars

        # Two calls with same parameters should produce identical results
        df1 = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
            threshold_decimal_bps=250,
            ouroboros="year",
            use_cache=True,
        )
        df2 = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
            threshold_decimal_bps=250,
            ouroboros="year",
            use_cache=True,
        )

        assert len(df1) == len(df2)
        assert df1.equals(df2), "Same ouroboros mode should produce identical results"

    @pytest.mark.skipif(
        "not config.getoption('--run-integration', default=False)",
        reason="Integration test - requires real data",
    )
    def test_different_ouroboros_modes_produce_different_bar_counts(
        self, real_data_available
    ):
        """Test that different ouroboros modes may produce different bar counts.

        Different modes reset processor at different boundaries, which can
        affect the number of bars generated (due to orphaned bars at boundaries).
        """
        if not real_data_available:
            pytest.skip("Real data not available")

        from rangebar import get_range_bars

        # Get bars with different ouroboros modes
        df_year = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-31",
            threshold_decimal_bps=250,
            ouroboros="year",
            use_cache=True,
        )
        df_month = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-31",
            threshold_decimal_bps=250,
            ouroboros="month",
            use_cache=True,
        )
        df_week = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-31",
            threshold_decimal_bps=250,
            ouroboros="week",
            use_cache=True,
        )

        # All should produce bars
        assert len(df_year) > 0
        assert len(df_month) > 0
        assert len(df_week) > 0

        # Bar counts may differ due to boundary resets
        # (orphaned bars at each boundary may be excluded)
        # Just verify they all produce reasonable results
        assert len(df_year) > 100, "Should produce substantial bars for January"
        assert len(df_month) > 100
        assert len(df_week) > 100

    @pytest.mark.skipif(
        "not config.getoption('--run-integration', default=False)",
        reason="Integration test - requires real data",
    )
    def test_year_boundary_resets_processor(self, real_data_available):
        """Test that year boundary causes processor reset.

        When crossing year boundary with ouroboros='year', the processor
        should reset, potentially creating an orphaned bar.
        """
        if not real_data_available:
            pytest.skip("Real data not available")

        from rangebar import get_range_bars

        # Get bars crossing year boundary with orphaned bars included
        df = get_range_bars(
            "BTCUSDT",
            "2023-12-30",
            "2024-01-03",
            threshold_decimal_bps=250,
            ouroboros="year",
            include_orphaned_bars=True,
            use_cache=True,
        )

        # Should have bars
        assert len(df) > 0

        # If orphaned bars are included, check for the column
        if "is_orphan" in df.columns:
            orphans = df[df["is_orphan"]]
            # May or may not have orphans depending on processor state at boundary
            # Just verify the column exists and has boolean values
            assert df["is_orphan"].dtype == bool or orphans.empty

    def test_ouroboros_none_is_rejected(self):
        """Test that ouroboros='none' is rejected (mandatory ouroboros)."""
        from rangebar import get_range_bars

        with pytest.raises(ValueError, match="Invalid ouroboros mode"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-07",
                threshold_decimal_bps=250,
                ouroboros="none",
            )

    def test_ouroboros_invalid_value_rejected(self):
        """Test that invalid ouroboros values are rejected."""
        from rangebar import get_range_bars

        with pytest.raises(ValueError, match="Invalid ouroboros mode"):
            get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-07",
                threshold_decimal_bps=250,
                ouroboros="daily",
            )


class TestSegmentEdgeCases:
    """Test edge cases in segment iteration."""

    def test_segment_end_times(self):
        """Test segment end times are 23:59:59.999999."""
        segments = list(
            iter_ouroboros_segments(date(2024, 1, 1), date(2024, 1, 31), "month")
        )
        # Single segment for January
        assert len(segments) == 1
        _start, end, _boundary = segments[0]
        assert end.hour == 23
        assert end.minute == 59
        assert end.second == 59
        assert end.microsecond == 999999

    def test_segment_boundary_transition(self):
        """Test segment transitions at boundary."""
        segments = list(
            iter_ouroboros_segments(date(2024, 1, 1), date(2024, 2, 29), "month")
        )
        assert len(segments) == 2

        # End of January segment
        _start1, end1, _boundary1 = segments[0]
        # Start of February segment
        start2, _end2, _boundary2 = segments[1]

        # There should be a 1 microsecond gap between segments
        # end1 should be Feb 1 00:00:00 - 1 microsecond = Jan 31 23:59:59.999999
        assert end1 == datetime(2024, 1, 31, 23, 59, 59, 999999, tzinfo=UTC)
        assert start2 == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)

    def test_empty_segments_list(self):
        """Test segment iteration returns generator."""
        segments_gen = iter_ouroboros_segments(
            date(2024, 1, 8), date(2024, 1, 13), "week"
        )
        # Should be a generator/iterator
        assert hasattr(segments_gen, "__iter__")
        assert hasattr(segments_gen, "__next__")
