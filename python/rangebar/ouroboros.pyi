from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Literal

class OuroborosMode(str, Enum):
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"

@dataclass(frozen=True)
class OuroborosBoundary:
    timestamp: datetime
    mode: OuroborosMode
    reason: str

    @property
    def timestamp_ms(self) -> int:
        ...

    @property
    def timestamp_us(self) -> int:
        ...

@dataclass
class OrphanedBarMetadata:
    is_orphan: bool = True
    ouroboros_boundary: datetime | None = None
    reason: str | None = None
    expected_duration_us: int | None = None

def get_ouroboros_boundaries(
    start: date,
    end: date,
    mode: Literal["year", "month", "week"],
) -> list[OuroborosBoundary]:
    ...
