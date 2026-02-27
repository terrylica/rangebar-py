# FILE-SIZE-OK: registry + gate + telemetry are tightly coupled
"""Unified Symbol Registry with mandatory gating (Issue #79).

SSoT: python/rangebar/data/symbols.toml
Symlink: symbols.toml (repo root) for developer convenience.

Every symbol must have an entry in the registry before it can be processed.
Unregistered symbols raise SymbolNotRegisteredError to prevent inadvertent
processing of symbols whose data quality hasn't been analyzed.

Gate Modes (RANGEBAR_SYMBOL_GATE env var):
- "strict" (default): SymbolNotRegisteredError for unknown symbols
- "warn": UserWarning + continue
- "off": No gating (development/testing only)

Follows threshold.py pattern: @lru_cache, MappingProxyType, clear_*_cache().

Issue #79: https://github.com/terrylica/rangebar-py/issues/79
"""

from __future__ import annotations

import os
import tomllib
import types
import warnings
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

from rangebar.exceptions import SymbolNotRegisteredError

if TYPE_CHECKING:
    pass

__all__ = [
    "KnownGap",
    "SymbolEntry",
    "SymbolTransition",
    "clear_symbol_registry_cache",
    "get_effective_start_date",
    "get_registered_symbols",
    "get_symbol_entries",
    "get_transitions",
    "validate_and_clamp_start_date",
    "validate_symbol_registered",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class KnownGap:
    """A registered gap where missing trade data is expected.

    Used by trade ID continuity checks to suppress alerts for
    known data source gaps (e.g., Binance Vision 404s).
    """

    start_date: date
    end_date: date
    reason: str
    source: str | None = None


@dataclass(frozen=True)
class SymbolEntry:
    """Frozen metadata for a registered symbol.

    Attributes
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    enabled : bool
        Whether the symbol is active for processing.
    asset_class : str
        Asset classification ("crypto", "forex", "equities").
    exchange : str
        Exchange identifier ("binance", "exness").
    market : str
        Market type ("spot", "futures-um", "futures-cm").
    listing_date : date
        When the symbol was listed on the exchange.
    effective_start : date | None
        Override for listing_date when upstream data is corrupted/missing.
        Processing starts from this date instead.
    reason : str | None
        Human-readable explanation for effective_start override.
    tier : int | None
        Liquidity tier (1 = highest).
    keywords : tuple[str, ...]
        AI-discoverable keywords for this symbol.
    reference : str | None
        GitHub issue or documentation URL.
    first_clean_date : date | None
        First date with verified clean data (no corruption, no gaps).
    data_anomalies : tuple[str, ...]
        Known data quality issues (e.g., corrupted CSVs, missing dates).
    processing_notes : str | None
        Operational notes from production processing experience.
    min_threshold : int | None
        Minimum threshold in dbps. SSoT for per-symbol enforcement.
        If None, falls back to asset-class default.
    """

    symbol: str
    enabled: bool
    asset_class: str
    exchange: str
    market: str
    listing_date: date
    effective_start: date | None = None
    reason: str | None = None
    tier: int | None = None
    keywords: tuple[str, ...] = ()
    reference: str | None = None
    first_clean_date: date | None = None
    data_anomalies: tuple[str, ...] = ()
    processing_notes: str | None = None
    min_threshold: int | None = None
    known_gaps: tuple[KnownGap, ...] = ()


@dataclass(frozen=True)
class SymbolTransition:
    """Historical record of a symbol rename/rebrand.

    Not yet used for automated gap handling -- placeholder for future work.
    """

    name: str
    old_symbol: str
    new_symbol: str
    last_old_date: date
    first_new_date: date
    gap_days: int
    reason: str | None = None
    keywords: tuple[str, ...] = ()
    reference: str | None = None
    status: str = "placeholder"


# =============================================================================
# Registry Loading (cached, immutable)
# =============================================================================


@lru_cache(maxsize=1)
def _load_registry() -> types.MappingProxyType:
    """Load TOML registry via importlib.resources (wheel-compatible).

    Returns an immutable MappingProxyType to prevent accidental mutation
    of cached data.
    """
    toml_bytes = (
        resources.files("rangebar.data").joinpath("symbols.toml").read_bytes()
    )
    raw = tomllib.loads(toml_bytes.decode())
    return types.MappingProxyType(raw)


def _parse_keywords(data: dict) -> tuple[str, ...]:
    """Parse keywords list from TOML data into frozen tuple."""
    raw = data.get("keywords", [])
    return tuple(raw) if raw else ()


def _parse_known_gaps(data: dict) -> tuple[KnownGap, ...]:
    """Parse known_gaps list from TOML data into frozen tuple."""
    raw = data.get("known_gaps", [])
    if not raw:
        return ()
    return tuple(
        KnownGap(
            start_date=g["start_date"],
            end_date=g["end_date"],
            reason=g["reason"],
            source=g.get("source"),
        )
        for g in raw
    )


@lru_cache(maxsize=1)
def get_symbol_entries() -> types.MappingProxyType[str, SymbolEntry]:
    """All registered symbols as frozen dataclasses (cached).

    Returns
    -------
    MappingProxyType[str, SymbolEntry]
        Immutable mapping of symbol name to SymbolEntry.

    Examples
    --------
    >>> entries = get_symbol_entries()
    >>> entries["BTCUSDT"].listing_date
    datetime.date(2017, 8, 17)
    """
    registry = _load_registry()
    symbols_raw = registry.get("symbols", {})
    entries: dict[str, SymbolEntry] = {}

    for symbol, data in symbols_raw.items():
        entries[symbol] = SymbolEntry(
            symbol=symbol,
            enabled=data["enabled"],
            asset_class=data["asset_class"],
            exchange=data["exchange"],
            market=data["market"],
            listing_date=data["listing_date"],
            effective_start=data.get("effective_start"),
            reason=data.get("reason"),
            tier=data.get("tier"),
            keywords=_parse_keywords(data),
            reference=data.get("reference"),
            first_clean_date=data.get("first_clean_date"),
            data_anomalies=tuple(data.get("data_anomalies", [])),
            processing_notes=data.get("processing_notes"),
            min_threshold=data.get("min_threshold"),
            known_gaps=_parse_known_gaps(data),
        )

    return types.MappingProxyType(entries)


# =============================================================================
# Mandatory Gate
# =============================================================================


def validate_symbol_registered(symbol: str, *, operation: str = "") -> None:
    """GATE: Raise SymbolNotRegisteredError if symbol not registered.

    Respects RANGEBAR_SYMBOL_GATE env var:
    - "strict" (default): raises SymbolNotRegisteredError
    - "warn": emits UserWarning, continues
    - "off": no gating (development/testing only)

    Parameters
    ----------
    symbol : str
        Trading symbol to validate.
    operation : str
        Name of the calling function (for error messages and telemetry).

    Raises
    ------
    SymbolNotRegisteredError
        If symbol is not in registry or not enabled (strict mode).
    """
    gate_mode = os.environ.get("RANGEBAR_SYMBOL_GATE", "strict")
    if gate_mode == "off":
        return

    entries = get_symbol_entries()
    if symbol in entries and entries[symbol].enabled:
        return

    # Symbol not registered or not enabled
    registered = tuple(sorted(entries))

    if gate_mode == "warn":
        warnings.warn(
            f"Symbol '{symbol}' not in registry (operation: {operation}). "
            f"Add it to symbols.toml before production use.",
            UserWarning,
            stacklevel=2,
        )
        return

    # Emit telemetry before raising (never fails)
    _emit_symbol_gate_telemetry(
        symbol=symbol,
        operation=operation,
        registered_symbols=registered,
    )

    msg = (
        f"Symbol '{symbol}' is not registered. "
        f"Add it to symbols.toml before processing. "
        f"Registered: {', '.join(registered)}"
    )
    raise SymbolNotRegisteredError(
        msg,
        symbol=symbol,
        operation=operation,
    )


# =============================================================================
# Start Date Clamping
# =============================================================================


def validate_and_clamp_start_date(symbol: str, start_date: str) -> str:
    """Gate + clamp start_date to effective_start if applicable.

    Validates that the symbol is registered, then clamps the start_date
    forward to effective_start if the symbol has known data quality issues
    before that date.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    start_date : str
        ISO format date string (e.g., "2017-08-17").

    Returns
    -------
    str
        The later of start_date and effective_start (ISO format).
    """
    validate_symbol_registered(symbol, operation="validate_and_clamp_start_date")
    entry = get_symbol_entries().get(symbol)
    if entry and entry.effective_start:
        effective = entry.effective_start.isoformat()
        if start_date < effective:
            try:
                from loguru import logger

                logger.info(
                    "Clamping start_date {} -> {} for {} (registry: {})",
                    start_date,
                    effective,
                    symbol,
                    entry.reason or "effective_start override",
                )
            except ImportError:
                pass
            return effective
    return start_date


# =============================================================================
# Query Functions
# =============================================================================


def get_effective_start_date(symbol: str) -> str | None:
    """Get effective_start (or listing_date) as ISO string.

    Parameters
    ----------
    symbol : str
        Trading symbol.

    Returns
    -------
    str | None
        ISO date string, or None if symbol not in registry.
    """
    entry = get_symbol_entries().get(symbol)
    if entry is None:
        return None
    if entry.effective_start:
        return entry.effective_start.isoformat()
    return entry.listing_date.isoformat()


def get_registered_symbols(
    *,
    asset_class: str | None = None,
    tier: int | None = None,
    enabled_only: bool = True,
) -> tuple[str, ...]:
    """Get filtered list of registered symbols.

    Parameters
    ----------
    asset_class : str | None
        Filter by asset class (e.g., "crypto", "forex").
    tier : int | None
        Filter by liquidity tier (e.g., 1).
    enabled_only : bool
        If True (default), only return enabled symbols.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of matching symbol names.

    Examples
    --------
    >>> get_registered_symbols(tier=1)
    ('AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', ...)
    """
    entries = get_symbol_entries()
    result = []
    for symbol, entry in entries.items():
        if enabled_only and not entry.enabled:
            continue
        if asset_class is not None and entry.asset_class != asset_class:
            continue
        if tier is not None and entry.tier != tier:
            continue
        result.append(symbol)
    return tuple(sorted(result))


def get_transitions() -> tuple[SymbolTransition, ...]:
    """All recorded symbol transitions.

    Returns
    -------
    tuple[SymbolTransition, ...]
        Tuple of SymbolTransition dataclasses.
    """
    registry = _load_registry()
    transitions_raw = registry.get("transitions", {})
    result = []
    for name, data in transitions_raw.items():
        result.append(
            SymbolTransition(
                name=name,
                old_symbol=data["old_symbol"],
                new_symbol=data["new_symbol"],
                last_old_date=data["last_old_date"],
                first_new_date=data["first_new_date"],
                gap_days=data["gap_days"],
                reason=data.get("reason"),
                keywords=_parse_keywords(data),
                reference=data.get("reference"),
                status=data.get("status", "placeholder"),
            )
        )
    return tuple(result)


# =============================================================================
# Cache Management
# =============================================================================


def clear_symbol_registry_cache() -> None:
    """Clear all caches. Call after editing symbols.toml at runtime.

    Examples
    --------
    >>> clear_symbol_registry_cache()  # Forces reload on next access
    """
    _load_registry.cache_clear()
    get_symbol_entries.cache_clear()


# =============================================================================
# Telemetry (NDJSON logging + Pushover alerts)
# =============================================================================


def _emit_symbol_gate_telemetry(
    symbol: str,
    operation: str,
    registered_symbols: tuple[str, ...],
) -> None:
    """Emit NDJSON event + Pushover alert for gate violation.

    Never raises -- telemetry failures suppressed via contextlib.suppress.
    Discoverable on remote hosts (BigBlack/LittleBlack) via:
        jq 'select(.component == "symbol_registry")' logs/events.jsonl
    """
    import contextlib

    with contextlib.suppress(ImportError, OSError):
        _log_gate_violation_ndjson(symbol, operation, registered_symbols)

    with contextlib.suppress(ImportError, OSError, Exception):
        _send_gate_pushover_alert(symbol, operation)


def _log_gate_violation_ndjson(
    symbol: str,
    operation: str,
    registered_symbols: tuple[str, ...],
) -> None:
    """Log gate violation as NDJSON event to logs/events.jsonl.

    Auto-injected fields (from logging.py context):
      - host: hostname (bigblack, littleblack, etc.)
      - service: "rangebar-py"
      - environment: RANGEBAR_ENV
      - pid: process ID
    """
    from loguru import logger

    logger.bind(
        component="symbol_registry",
        event_type="symbol_not_registered",
        symbol=symbol,
        operation=operation,
        registered_count=len(registered_symbols),
    ).error(
        "Symbol '{}' not registered in symbols.toml. "
        "Add entry before processing. Operation: {}. "
        "Registry has {} symbols.",
        symbol,
        operation,
        len(registered_symbols),
    )


def _send_gate_pushover_alert(symbol: str, operation: str) -> None:
    """Send Pushover EMERGENCY alert for gate violation.

    Hardcoded credentials (safe to share -- Pushover app tokens are not secrets).
    App: "RB runtime failures" on Pushover dashboard.
    Sound: "dune" (custom uploaded sound).
    Priority: 2 (Emergency) -- repeats every 30s until acknowledged, expires 1hr.

    Pushover dashboard: https://pushover.net/apps/aii416kz1pc1rmeftgo58fekifj4fm
    User key source: Pushover account settings page
    Also in 1Password: item hkrtw4piagud72r7ukp64y72ne (vault: Employee)
    """
    # Hardcoded: Pushover "RB runtime failures" app token + user key
    # These are NOT secrets -- Pushover tokens only allow sending notifications
    app_token = "aii416kz1pc1rmeftgo58fekifj4fm"  # App: "RB runtime failures"
    user_key = "ury88s1def6v16seeueoefqn1zbua1"

    import httpx

    httpx.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": app_token,
            "user": user_key,
            "title": f"Symbol Not Registered: {symbol}",
            "message": (
                f"Attempted {operation}('{symbol}') but symbol not in registry.\n"
                f"Fix: Add entry to symbols.toml, then maturin develop."
            ),
            "priority": 2,
            "retry": 30,
            "expire": 3600,
            "sound": "dune",
        },
        timeout=5,
    )
