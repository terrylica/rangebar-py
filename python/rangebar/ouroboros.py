"""Ouroboros: Cyclical reset boundaries for reproducible range bar construction.

Named after the Greek serpent eating its tail (οὐροβόρος), representing the
cyclical nature of year/month/week reset boundaries.

This module provides:
- Boundary calculation for year/month/week granularities
- Orphaned bar metadata for ML filtering
- Exchange session detection (Sydney/Tokyo/London/New York)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

# ============================================================================
# Types
# ============================================================================


class OuroborosMode(str, Enum):
    """Ouroboros granularity modes."""

    YEAR = "year"
    MONTH = "month"
    WEEK = "week"


@dataclass(frozen=True)
class OuroborosBoundary:
    """A single ouroboros reset boundary."""

    timestamp: datetime
    """UTC datetime of the boundary."""

    mode: OuroborosMode
    """Which granularity created this boundary."""

    reason: str
    """Human-readable reason (e.g., 'year_boundary', 'month_boundary')."""

    @property
    def timestamp_ms(self) -> int:
        """Timestamp in milliseconds (for comparison with trade data)."""
        return int(self.timestamp.timestamp() * 1000)

    @property
    def timestamp_us(self) -> int:
        """Timestamp in microseconds."""
        return int(self.timestamp.timestamp() * 1_000_000)


@dataclass
class OrphanedBarMetadata:
    """Metadata for orphaned bars at ouroboros boundaries."""

    is_orphan: bool = True
    """Always True for orphaned bars."""

    ouroboros_boundary: datetime | None = None
    """Which boundary caused the orphan."""

    reason: str | None = None
    """Reason string: 'year_boundary', 'month_boundary', 'week_boundary'."""

    expected_duration_us: int | None = None
    """Expected duration if bar had completed normally."""


# ============================================================================
# Boundary Calculation
# ============================================================================


def get_ouroboros_boundaries(
    start: date,
    end: date,
    mode: Literal["year", "month", "week"],
) -> list[OuroborosBoundary]:
    """Return all ouroboros reset points within the date range.

    Parameters
    ----------
    start : date
        Start date (inclusive)
    end : date
        End date (inclusive)
    mode : {"year", "month", "week"}
        Ouroboros granularity

    Returns
    -------
    list[OuroborosBoundary]
        Sorted list of boundaries within the date range

    Examples
    --------
    >>> from datetime import date
    >>> boundaries = get_ouroboros_boundaries(
    ...     date(2024, 1, 1), date(2024, 3, 31), "month"
    ... )
    >>> len(boundaries)
    3
    >>> boundaries[0].reason
    'month_boundary'
    """
    boundaries: list[OuroborosBoundary] = []

    if mode == "year":
        for year in range(start.year, end.year + 2):
            boundary_date = date(year, 1, 1)
            if start <= boundary_date <= end:
                boundaries.append(
                    OuroborosBoundary(
                        timestamp=datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC),
                        mode=OuroborosMode.YEAR,
                        reason="year_boundary",
                    )
                )

    elif mode == "month":
        current = date(start.year, start.month, 1)
        while current <= end:
            if current >= start:
                boundaries.append(
                    OuroborosBoundary(
                        timestamp=datetime(
                            current.year,
                            current.month,
                            1,
                            0,
                            0,
                            0,
                            tzinfo=UTC,
                        ),
                        mode=OuroborosMode.MONTH,
                        reason="month_boundary",
                    )
                )
            # Next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

    elif mode == "week":
        # Sunday 00:00:00 UTC boundaries
        # Sunday = 6 in Python's weekday()
        days_until_sunday = (6 - start.weekday()) % 7
        if days_until_sunday == 0 and start.weekday() != 6:
            # start is not Sunday, find next Sunday
            days_until_sunday = 7
        current = start + timedelta(days=days_until_sunday)

        # Also include start date if it's a Sunday
        if start.weekday() == 6:
            current = start

        while current <= end:
            boundaries.append(
                OuroborosBoundary(
                    timestamp=datetime(
                        current.year,
                        current.month,
                        current.day,
                        0,
                        0,
                        0,
                        tzinfo=UTC,
                    ),
                    mode=OuroborosMode.WEEK,
                    reason="week_boundary",
                )
            )
            current += timedelta(days=7)

    return boundaries


def iter_ouroboros_segments(
    start: date,
    end: date,
    mode: Literal["year", "month", "week"],
) -> Iterator[tuple[datetime, datetime, OuroborosBoundary | None]]:
    """Iterate over date segments between ouroboros boundaries.

    Yields (segment_start, segment_end, boundary) tuples where boundary
    is the ouroboros boundary at segment_start (None for first segment
    if it doesn't start on a boundary).

    Parameters
    ----------
    start : date
        Start date
    end : date
        End date
    mode : {"year", "month", "week"}
        Ouroboros granularity

    Yields
    ------
    tuple[datetime, datetime, OuroborosBoundary | None]
        (segment_start, segment_end, boundary_at_start)
    """
    boundaries = get_ouroboros_boundaries(start, end, mode)

    # Convert dates to datetimes
    start_dt = datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, 999999, tzinfo=UTC)

    if not boundaries:
        # No boundaries in range - single segment
        yield (start_dt, end_dt, None)
        return

    # First segment: start to first boundary (if start is before first boundary)
    if start_dt < boundaries[0].timestamp:
        yield (start_dt, boundaries[0].timestamp - timedelta(microseconds=1), None)

    # Middle segments: between consecutive boundaries
    for i, boundary in enumerate(boundaries):
        if i + 1 < len(boundaries):
            segment_end = boundaries[i + 1].timestamp - timedelta(microseconds=1)
        else:
            segment_end = end_dt

        # Only yield if segment start is before segment end
        if boundary.timestamp <= end_dt:
            yield (boundary.timestamp, segment_end, boundary)


# ============================================================================
# Exchange Market Sessions
# ============================================================================

# Market session hours in local time (aligned with actual exchange hours)
# Note: These are approximate for crypto; traditional markets have pre/post sessions
# Issue #8: Exchange sessions integration (corrected per exchange schedules)
EXCHANGE_SESSION_HOURS = {
    "sydney": {"tz": "Australia/Sydney", "start": 10, "end": 16},   # ASX
    "tokyo": {"tz": "Asia/Tokyo", "start": 9, "end": 15},           # TSE
    "london": {"tz": "Europe/London", "start": 8, "end": 17},       # LSE
    "newyork": {"tz": "America/New_York", "start": 10, "end": 16},  # NYSE
}


@dataclass(frozen=True)
class ExchangeSessionFlags:
    """Boolean flags for active exchange market sessions."""

    sydney: bool
    tokyo: bool
    london: bool
    newyork: bool

    def to_dict(self) -> dict[str, bool]:
        """Convert to dict with column names."""
        return {
            "exchange_session_sydney": self.sydney,
            "exchange_session_tokyo": self.tokyo,
            "exchange_session_london": self.london,
            "exchange_session_newyork": self.newyork,
        }


def get_active_exchange_sessions(timestamp_utc: datetime) -> ExchangeSessionFlags:
    """Determine which exchange market sessions are active at a given UTC time.

    Parameters
    ----------
    timestamp_utc : datetime
        UTC datetime to check (must be timezone-aware)

    Returns
    -------
    ExchangeSessionFlags
        Boolean flags for each session

    Notes
    -----
    This is a simplified implementation that uses fixed hours.
    For production use with DST accuracy, consider using nautilus_trader's
    ForexSession implementation.
    """
    import zoneinfo

    def is_in_session(session_name: str) -> bool:
        info = EXCHANGE_SESSION_HOURS[session_name]
        tz = zoneinfo.ZoneInfo(info["tz"])
        local_time = timestamp_utc.astimezone(tz)
        hour = local_time.hour
        # Skip weekends
        if local_time.weekday() >= 5:
            return False
        return info["start"] <= hour < info["end"]

    return ExchangeSessionFlags(
        sydney=is_in_session("sydney"),
        tokyo=is_in_session("tokyo"),
        london=is_in_session("london"),
        newyork=is_in_session("newyork"),
    )


# ============================================================================
# Dynamic Ouroboros for Forex
# ============================================================================

# Weekend gap threshold: 40 hours in milliseconds
# Forex markets close Friday ~21:00 UTC, reopen Sunday ~17:00 UTC (~44 hours)
# Using 40 hours as threshold to account for slight variations
WEEKEND_GAP_THRESHOLD_MS = 40 * 60 * 60 * 1000  # 40 hours


def detect_forex_weekend_boundaries(
    timestamps_ms: list[int],
) -> list[OuroborosBoundary]:
    """Detect weekend boundaries from tick timestamps.

    For Forex markets, the ouroboros point is the first tick after a weekend gap.
    This handles DST automatically since we use actual data gaps, not calendar.

    Parameters
    ----------
    timestamps_ms : list[int]
        Sorted list of tick timestamps in milliseconds

    Returns
    -------
    list[OuroborosBoundary]
        List of weekend boundaries (first tick after each weekend gap)

    Examples
    --------
    >>> timestamps = [1705057200000, 1705060800000, ...]  # Friday ticks
    >>> # ... weekend gap ...
    >>> timestamps.extend([1705233600000, ...])  # Sunday ticks
    >>> boundaries = detect_forex_weekend_boundaries(timestamps)
    >>> len(boundaries)  # One boundary at Sunday open
    1
    """
    if len(timestamps_ms) < 2:
        return []

    boundaries: list[OuroborosBoundary] = []

    for i in range(1, len(timestamps_ms)):
        gap_ms = timestamps_ms[i] - timestamps_ms[i - 1]

        if gap_ms >= WEEKEND_GAP_THRESHOLD_MS:
            # This is a weekend gap - the current tick is the ouroboros point
            boundary_dt = datetime.fromtimestamp(timestamps_ms[i] / 1000, tz=UTC)
            boundaries.append(
                OuroborosBoundary(
                    timestamp=boundary_dt,
                    mode=OuroborosMode.WEEK,
                    reason="forex_weekend_boundary",
                )
            )

    return boundaries


def iter_forex_ouroboros_segments(
    timestamps_ms: list[int],
    _start_date: date,  # Reserved for future filtering
    _end_date: date,  # Reserved for future filtering
) -> Iterator[tuple[int, int, OuroborosBoundary | None]]:
    """Iterate over segments between Forex weekend boundaries.

    Yields (start_idx, end_idx, boundary) tuples where:
    - start_idx is the first tick index in the segment
    - end_idx is the last tick index in the segment (inclusive)
    - boundary is the OuroborosBoundary at start_idx (None for first segment)

    Parameters
    ----------
    timestamps_ms : list[int]
        Sorted list of tick timestamps in milliseconds
    start_date : date
        Start date (for filtering)
    end_date : date
        End date (for filtering)

    Yields
    ------
    tuple[int, int, OuroborosBoundary | None]
        (start_idx, end_idx, boundary_at_start)
    """
    if not timestamps_ms:
        return

    boundaries = detect_forex_weekend_boundaries(timestamps_ms)

    if not boundaries:
        # No weekend gaps - single segment
        yield (0, len(timestamps_ms) - 1, None)
        return

    # Build boundary index map
    boundary_timestamps = {b.timestamp_ms for b in boundaries}

    current_start = 0
    current_boundary: OuroborosBoundary | None = None

    for i, ts_ms in enumerate(timestamps_ms):
        if ts_ms in boundary_timestamps:
            # End previous segment (if any)
            if i > current_start:
                yield (current_start, i - 1, current_boundary)

            # Start new segment at this boundary
            current_start = i
            current_boundary = next(b for b in boundaries if b.timestamp_ms == ts_ms)

    # Yield final segment
    if current_start < len(timestamps_ms):
        yield (current_start, len(timestamps_ms) - 1, current_boundary)


# ============================================================================
# Validation
# ============================================================================


def validate_ouroboros_mode(mode: str) -> Literal["year", "month", "week"]:
    """Validate ouroboros mode string.

    Parameters
    ----------
    mode : str
        Mode to validate

    Returns
    -------
    Literal["year", "month", "week"]
        Validated mode

    Raises
    ------
    ValueError
        If mode is not valid
    """
    valid_modes = {"year", "month", "week"}
    if mode not in valid_modes:
        msg = f"Invalid ouroboros mode: {mode!r}. Must be one of: {valid_modes}"
        raise ValueError(msg)
    return mode  # type: ignore[return-value]
