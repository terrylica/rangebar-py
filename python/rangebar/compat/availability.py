"""Cache availability API for alpha-forge integration (Issue #95).

Reports which symbols have cached range bars, at which thresholds,
and over what date ranges. Graceful fallback when ClickHouse is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rangebar.symbol_registry import SymbolEntry

logger = logging.getLogger(__name__)


@dataclass
class SymbolAvailability:
    """Availability info for a single symbol's cached range bars."""

    symbol: str
    asset_class: str
    exchange: str
    tier: int | None
    effective_start: str
    listing_date: str
    thresholds_cached: list[int] = field(default_factory=list)
    cached_date_ranges: dict[int, tuple[str, str]] = field(default_factory=dict)
    bar_counts: dict[int, int] = field(default_factory=dict)
    has_microstructure: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "exchange": self.exchange,
            "tier": self.tier,
            "effective_start": self.effective_start,
            "listing_date": self.listing_date,
            "thresholds_cached": self.thresholds_cached,
            "cached_date_ranges": {
                str(k): list(v) for k, v in self.cached_date_ranges.items()
            },
            "bar_counts": self.bar_counts,
            "has_microstructure": self.has_microstructure,
        }


def _entry_effective_start(entry: SymbolEntry) -> str:
    """Get effective_start as string, falling back to listing_date."""
    if entry.effective_start is not None:
        return str(entry.effective_start)
    return str(entry.listing_date)


def _entry_to_availability(sym: str, entry: SymbolEntry) -> SymbolAvailability:
    """Convert a SymbolEntry to a SymbolAvailability (no cache data)."""
    return SymbolAvailability(
        symbol=sym,
        asset_class=entry.asset_class,
        exchange=entry.exchange,
        tier=entry.tier,
        effective_start=_entry_effective_start(entry),
        listing_date=str(entry.listing_date),
    )


def _ts_to_date(ts_ms: int) -> str:
    """Convert millisecond timestamp to ISO date string."""
    import datetime

    return datetime.datetime.fromtimestamp(
        ts_ms / 1000, tz=datetime.UTC
    ).strftime("%Y-%m-%d")


def get_cache_coverage(
    symbols: list[str] | None = None,
    *,
    include_bar_counts: bool = True,
) -> dict[str, SymbolAvailability]:
    """Report cached range bar coverage per symbol.

    Returns empty dict (no exception) if ClickHouse is unreachable.

    Parameters
    ----------
    symbols : list[str] | None
        Symbols to query. None = all registered symbols.
    include_bar_counts : bool
        Include per-threshold bar counts (slightly slower).

    Returns
    -------
    dict[str, SymbolAvailability]
        Keyed by symbol name.
    """
    from rangebar.symbol_registry import get_symbol_entries

    entries = get_symbol_entries()

    if symbols is not None:
        # Issue #96 Task #35: Use frozenset for O(1) membership testing
        symbol_set = frozenset(symbols)
        entries = {k: v for k, v in entries.items() if k in symbol_set}

    # Try connecting to ClickHouse
    try:
        from rangebar.clickhouse.cache import RangeBarCache

        cache = RangeBarCache()
    except (ImportError, OSError, RuntimeError):
        logger.debug("ClickHouse unavailable, returning registry-only data")
        return {sym: _entry_to_availability(sym, entry) for sym, entry in entries.items()}

    standard_thresholds = [250, 500, 750, 1000]
    result: dict[str, SymbolAvailability] = {}

    for sym, entry in entries.items():
        avail = _entry_to_availability(sym, entry)

        for threshold in standard_thresholds:
            try:
                count = cache.count_bars(sym, threshold) if include_bar_counts else 0
                if count > 0 or not include_bar_counts:
                    if not include_bar_counts:
                        count = cache.count_bars(sym, threshold)
                    if count > 0:
                        avail.thresholds_cached.append(threshold)
                        avail.bar_counts[threshold] = count

                        oldest = cache.get_oldest_bar_timestamp(sym, threshold)
                        newest = cache.get_newest_bar_timestamp(sym, threshold)
                        if oldest is not None and newest is not None:
                            avail.cached_date_ranges[threshold] = (
                                _ts_to_date(oldest),
                                _ts_to_date(newest),
                            )
            except (OSError, RuntimeError):
                logger.debug("Failed to query %s@%d", sym, threshold)

        result[sym] = avail

    return result


def get_available_symbols(
    *,
    asset_class: str | None = None,
    min_tier: int | None = None,
    cached_only: bool = False,
) -> list[dict[str, Any]]:
    """List symbols with filtering for universe selection.

    Parameters
    ----------
    asset_class : str | None
        Filter by asset class (e.g., "crypto").
    min_tier : int | None
        Minimum tier level (1 = highest liquidity).
    cached_only : bool
        Only include symbols with cached data (requires ClickHouse).

    Returns
    -------
    list[dict[str, Any]]
        List of symbol info dicts.
    """
    from rangebar.symbol_registry import get_symbol_entries

    entries = get_symbol_entries()
    results: list[dict[str, Any]] = []

    coverage = get_cache_coverage() if cached_only else {}
    # Issue #96 Task #35: Pre-compute coverage keys as frozenset for O(1) lookups
    coverage_keys = frozenset(coverage) if coverage else frozenset()

    for sym, entry in entries.items():
        if asset_class and entry.asset_class != asset_class:
            continue
        if min_tier is not None and (entry.tier is None or entry.tier > min_tier):
            continue
        if cached_only and (sym not in coverage_keys or not coverage[sym].thresholds_cached):
            continue

        info: dict[str, Any] = {
            "symbol": sym,
            "asset_class": entry.asset_class,
            "exchange": entry.exchange,
            "tier": entry.tier,
            "listing_date": str(entry.listing_date),
            "effective_start": _entry_effective_start(entry),
        }

        if sym in coverage_keys:
            info["thresholds_cached"] = coverage[sym].thresholds_cached
            info["bar_counts"] = coverage[sym].bar_counts

        results.append(info)

    return results
