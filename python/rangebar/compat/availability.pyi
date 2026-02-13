from typing import Any

class SymbolAvailability:
    symbol: str
    asset_class: str
    exchange: str
    tier: int | None
    effective_start: str
    listing_date: str
    thresholds_cached: list[int]
    cached_date_ranges: dict[int, tuple[str, str]]
    bar_counts: dict[int, int]
    has_microstructure: bool
    def to_dict(self) -> dict[str, Any]: ...

def get_cache_coverage(
    symbols: list[str] | None = None,
    *,
    include_bar_counts: bool = True,
) -> dict[str, SymbolAvailability]: ...

def get_available_symbols(
    *,
    asset_class: str | None = None,
    min_tier: int | None = None,
    cached_only: bool = False,
) -> list[dict[str, Any]]: ...
