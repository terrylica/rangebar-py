from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class SymbolEntry:
    symbol: str
    """Trading symbol (e.g., "BTCUSDT")."""
    enabled: bool
    """Whether this symbol can be processed."""
    asset_class: str
    """Asset class: "crypto", "forex"."""
    exchange: str
    """Exchange: "binance", "exness"."""
    market: str
    """Market type: "spot", "futures-um", "futures-cm"."""
    listing_date: date
    """When symbol was listed on the exchange."""
    effective_start: date | None = ...
    """Override listing_date for data quality (e.g., skip corrupted days)."""
    reason: str | None = ...
    """Why effective_start differs from listing_date."""
    tier: int | None = ...
    """Liquidity tier (1 = highest)."""
    keywords: tuple[str, ...] = ...
    """AI discoverability keywords."""
    reference: str | None = ...
    """Issue or URL reference for data anomaly."""
    first_clean_date: date | None = ...
    """First date with verified clean data."""
    data_anomalies: tuple[str, ...] = ...
    """Known data quality issues."""
    processing_notes: str | None = ...
    """Operational notes from production processing."""

@dataclass(frozen=True)
class SymbolTransition:
    name: str
    """Transition identifier (e.g., "MATIC_TO_POL")."""
    old_symbol: str
    """Previous symbol name."""
    new_symbol: str
    """New symbol name."""
    last_old_date: date
    """Last date with data under old symbol."""
    first_new_date: date
    """First date with data under new symbol."""
    gap_days: int
    """Number of days with no data during transition."""
    reason: str | None = ...
    """Human-readable reason for transition."""
    keywords: tuple[str, ...] = ...
    """AI discoverability keywords."""
    reference: str | None = ...
    """Issue or URL reference."""
    status: str = ...
    """Implementation status: "placeholder", "implemented"."""

def validate_symbol_registered(
    symbol: str,
    *,
    operation: str = "",
) -> None:
    ...

def validate_and_clamp_start_date(
    symbol: str,
    start_date: str,
) -> str:
    ...

def get_effective_start_date(symbol: str) -> str | None:
    ...

def get_registered_symbols(
    *,
    asset_class: str | None = None,
    tier: int | None = None,
    enabled_only: bool = True,
) -> tuple[str, ...]:
    ...

def get_symbol_entries() -> dict[str, SymbolEntry]:
    ...

def get_transitions() -> tuple[SymbolTransition, ...]:
    ...

def clear_symbol_registry_cache() -> None:
    ...
