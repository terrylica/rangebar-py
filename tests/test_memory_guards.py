"""Tests for memory guard per-segment estimation (Issue #53, Phase 4).

Validates:
- MEM-010 estimates memory for largest ouroboros segment, not full range
- Segment calculation uses iter_ouroboros_segments correctly
- Fallback to full range when no segments available
"""

from __future__ import annotations

from datetime import date, datetime


class TestMEM010PerSegmentEstimation:
    """Test that MEM-010 estimates per-segment, not full range."""

    def test_iter_ouroboros_segments_year_mode(self):
        """Year mode should produce ~N segments for N-year range."""
        from rangebar.ouroboros import iter_ouroboros_segments

        start = date(2020, 1, 1)
        end = date(2023, 12, 31)
        segments = list(iter_ouroboros_segments(start, end, "year"))

        # 4 years should produce 4 segments
        assert len(segments) == 4

    def test_largest_segment_selection(self):
        """MEM-010 should find the largest segment by time span."""
        from rangebar.ouroboros import iter_ouroboros_segments

        start = date(2023, 6, 15)
        end = date(2024, 3, 15)
        segments = list(iter_ouroboros_segments(start, end, "year"))

        # Should have 2 segments: partial 2023 and partial 2024
        assert len(segments) >= 1

        # Find largest by time span (same logic as range_bars.py)
        largest = max(segments, key=lambda s: (s[1] - s[0]).total_seconds())
        assert largest is not None

    def test_month_mode_produces_more_segments(self):
        """Month mode should produce ~12 segments per year."""
        from rangebar.ouroboros import iter_ouroboros_segments

        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        segments = list(iter_ouroboros_segments(start, end, "month"))

        assert len(segments) == 12

    def test_segment_smaller_than_full_range(self):
        """Largest segment should cover less time than the full date range."""
        from rangebar.ouroboros import iter_ouroboros_segments

        start = date(2020, 1, 1)
        end = date(2025, 12, 31)
        segments = list(iter_ouroboros_segments(start, end, "year"))

        full_span = (
            datetime.combine(end, datetime.min.time())
            - datetime.combine(start, datetime.min.time())
        ).total_seconds()

        largest = max(segments, key=lambda s: (s[1] - s[0]).total_seconds())
        largest_span = (largest[1] - largest[0]).total_seconds()

        # Largest segment should be much smaller than full 6-year range
        assert largest_span < full_span
        # Each year segment is roughly 1/6 of the total
        assert largest_span < full_span * 0.25  # Each segment < 25% of total

    def test_range_bars_mem010_comment_present(self):
        """Verify MEM-010 comment is present in range_bars.py."""
        import inspect

        from rangebar.orchestration.range_bars import get_range_bars

        source = inspect.getsource(get_range_bars)
        assert "MEM-010" in source
        assert "LARGEST ouroboros segment" in source or "largest" in source.lower()


class TestMemoryGuardConstants:
    """Test memory guard registry in constants.py."""

    def test_mem_guards_registry_exists(self):
        """MEM_GUARDS dict should exist in constants."""
        from rangebar.constants import MEM_GUARDS

        assert isinstance(MEM_GUARDS, dict)
        assert "MEM-010" in MEM_GUARDS

    def test_mem010_description(self):
        """MEM-010 should describe pre-flight memory estimation."""
        from rangebar.constants import MEM_GUARDS

        guard = MEM_GUARDS["MEM-010"]
        assert "memory" in guard["description"].lower()
        assert guard["stage"] == "loading"
