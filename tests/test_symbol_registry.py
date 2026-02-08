# Issue #79: Unified Symbol Registry with Mandatory Gating
"""Test symbol registry module (Issue #79).

Covers:
- TOML loading and parsing
- SymbolEntry / SymbolTransition data types and immutability
- Gate validation (strict / warn / off modes)
- Start date clamping (effective_start override)
- Query functions (get_registered_symbols, get_effective_start_date)
- Transitions (get_transitions)
- Cache clearing (clear_symbol_registry_cache)
- TIER1_SYMBOLS consistency with registry
- Exports from rangebar package
"""

from __future__ import annotations

import contextlib
import os
import types
import warnings
from collections.abc import Generator
from datetime import date

import pytest
from rangebar import (
    TIER1_SYMBOLS,
    SymbolEntry,
    SymbolTransition,
    clear_symbol_registry_cache,
    get_effective_start_date,
    get_registered_symbols,
    get_symbol_entries,
    get_transitions,
    validate_and_clamp_start_date,
    validate_symbol_registered,
)
from rangebar.exceptions import RangeBarError, SymbolNotRegisteredError

# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def _with_gate_mode(mode: str | None) -> Generator[None]:
    """Context manager to temporarily set RANGEBAR_SYMBOL_GATE env var."""
    original = os.environ.get("RANGEBAR_SYMBOL_GATE")
    if mode is None:
        os.environ.pop("RANGEBAR_SYMBOL_GATE", None)
    else:
        os.environ["RANGEBAR_SYMBOL_GATE"] = mode
    clear_symbol_registry_cache()
    try:
        yield
    finally:
        if original is not None:
            os.environ["RANGEBAR_SYMBOL_GATE"] = original
        else:
            os.environ.pop("RANGEBAR_SYMBOL_GATE", None)
        clear_symbol_registry_cache()


# =============================================================================
# Loading & Data Types
# =============================================================================


class TestSymbolEntryLoading:
    """Test TOML loading and SymbolEntry construction."""

    def test_registry_loads_successfully(self):
        """Registry should load without errors."""
        entries = get_symbol_entries()
        assert len(entries) > 0

    def test_all_entries_are_symbol_entry(self):
        """Every value in the registry should be a SymbolEntry."""
        entries = get_symbol_entries()
        for symbol, entry in entries.items():
            assert isinstance(entry, SymbolEntry), f"{symbol} is not SymbolEntry"

    def test_returns_mapping_proxy(self):
        """get_symbol_entries() should return immutable MappingProxyType."""
        entries = get_symbol_entries()
        assert isinstance(entries, types.MappingProxyType)

    def test_mapping_proxy_is_immutable(self):
        """MappingProxyType should reject item assignment."""
        entries = get_symbol_entries()
        with pytest.raises(TypeError):
            entries["NEWCOIN"] = None  # type: ignore[index]

    def test_symbol_entry_is_frozen(self):
        """SymbolEntry should be frozen (immutable)."""
        entries = get_symbol_entries()
        entry = entries["BTCUSDT"]
        with pytest.raises(AttributeError):
            entry.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_btcusdt_has_expected_fields(self):
        """BTCUSDT should have all expected metadata."""
        entry = get_symbol_entries()["BTCUSDT"]
        assert entry.symbol == "BTCUSDT"
        assert entry.enabled is True
        assert entry.asset_class == "crypto"
        assert entry.exchange == "binance"
        assert entry.market == "spot"
        assert entry.listing_date == date(2017, 8, 17)
        assert entry.effective_start == date(2018, 1, 16)
        assert entry.tier == 1
        assert isinstance(entry.keywords, tuple)
        assert len(entry.keywords) > 0

    def test_btcusdt_has_anomaly_fields(self):
        """BTCUSDT should have oracle-verified data anomaly metadata (schema v2)."""
        entry = get_symbol_entries()["BTCUSDT"]
        assert entry.first_clean_date == date(2018, 1, 16)
        assert isinstance(entry.data_anomalies, tuple)
        assert len(entry.data_anomalies) == 2
        assert "2018-01-14" in entry.data_anomalies[0]
        assert "ghost trades" in entry.data_anomalies[0]
        assert "first_trade_id=-1" in entry.data_anomalies[0]
        assert entry.processing_notes is not None
        assert "ghost trade" in entry.processing_notes.lower()

    def test_symbol_without_anomalies_has_empty_tuple(self):
        """Symbols without data_anomalies should have empty tuple."""
        entry = get_symbol_entries()["XRPUSDT"]
        assert entry.data_anomalies == ()
        assert entry.first_clean_date is None
        assert entry.processing_notes is None

    def test_maticusdt_removed(self):
        """MATICUSDT should not be in registry (delisted)."""
        entries = get_symbol_entries()
        assert "MATICUSDT" not in entries

    def test_symbol_without_effective_start(self):
        """Symbols without effective_start should have None."""
        entry = get_symbol_entries()["XRPUSDT"]
        assert entry.effective_start is None

    def test_atomusdt_has_effective_start(self):
        """ATOMUSDT should have effective_start due to missing data."""
        entry = get_symbol_entries()["ATOMUSDT"]
        assert entry.listing_date == date(2019, 4, 22)
        assert entry.effective_start == date(2019, 4, 29)

    def test_cache_returns_same_object(self):
        """Repeated calls should return same cached object."""
        entries1 = get_symbol_entries()
        entries2 = get_symbol_entries()
        assert entries1 is entries2


# =============================================================================
# Gate Validation
# =============================================================================


class TestValidateSymbolRegistered:
    """Test the mandatory gate function."""

    def test_registered_symbol_passes(self):
        """Known symbol should pass gate without error."""
        with _with_gate_mode("strict"):
            validate_symbol_registered("BTCUSDT")  # Should not raise

    def test_unregistered_symbol_raises_strict(self):
        """Unregistered symbol should raise SymbolNotRegisteredError in strict mode."""
        with _with_gate_mode("strict"):
            with pytest.raises(SymbolNotRegisteredError) as exc_info:
                validate_symbol_registered("FAKECOINUSDT", operation="test_op")
            assert exc_info.value.symbol == "FAKECOINUSDT"
            assert exc_info.value.operation == "test_op"
            assert "FAKECOINUSDT" in str(exc_info.value)
            assert "symbols.toml" in str(exc_info.value)

    def test_error_lists_registered_symbols(self):
        """Error message should list registered symbols."""
        with _with_gate_mode("strict"):
            with pytest.raises(SymbolNotRegisteredError) as exc_info:
                validate_symbol_registered("UNKNOWN")
            msg = str(exc_info.value)
            assert "BTCUSDT" in msg
            assert "Registered:" in msg

    def test_unregistered_symbol_warns_in_warn_mode(self):
        """Unregistered symbol should emit UserWarning in warn mode."""
        with _with_gate_mode("warn"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validate_symbol_registered("FAKECOINUSDT", operation="test_warn")
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "FAKECOINUSDT" in str(w[0].message)

    def test_unregistered_symbol_passes_in_off_mode(self):
        """Unregistered symbol should pass silently in off mode."""
        with _with_gate_mode("off"):
            validate_symbol_registered("TOTALLY_FAKE_SYMBOL")  # Should not raise

    def test_default_gate_mode_is_strict(self):
        """Default gate mode (no env var) should be strict."""
        with _with_gate_mode(None):
            with pytest.raises(SymbolNotRegisteredError):
                validate_symbol_registered("UNKNOWN")

    def test_all_tier1_symbols_pass_gate(self):
        """Every TIER1_SYMBOLS entry should pass the gate."""
        with _with_gate_mode("strict"):
            for base in TIER1_SYMBOLS:
                symbol = f"{base}USDT"
                validate_symbol_registered(symbol)  # Should not raise


# =============================================================================
# Start Date Clamping
# =============================================================================


class TestStartDateClamping:
    """Test validate_and_clamp_start_date() function."""

    def test_clamps_before_effective_start(self):
        """start_date before effective_start should be clamped forward."""
        with _with_gate_mode("strict"):
            result = validate_and_clamp_start_date("BTCUSDT", "2017-08-17")
            assert result == "2018-01-16"

    def test_no_clamp_after_effective_start(self):
        """start_date after effective_start should be unchanged."""
        with _with_gate_mode("strict"):
            result = validate_and_clamp_start_date("BTCUSDT", "2024-01-01")
            assert result == "2024-01-01"

    def test_no_clamp_on_effective_start(self):
        """start_date equal to effective_start should be unchanged."""
        with _with_gate_mode("strict"):
            result = validate_and_clamp_start_date("BTCUSDT", "2018-01-16")
            assert result == "2018-01-16"

    def test_symbol_without_effective_start_no_clamp(self):
        """Symbol without effective_start should return start_date unchanged."""
        with _with_gate_mode("strict"):
            result = validate_and_clamp_start_date("XRPUSDT", "2018-01-01")
            assert result == "2018-01-01"

    def test_atomusdt_clamping(self):
        """ATOMUSDT should clamp to 2019-04-29."""
        with _with_gate_mode("strict"):
            result = validate_and_clamp_start_date("ATOMUSDT", "2019-04-22")
            assert result == "2019-04-29"

    def test_unregistered_symbol_raises(self):
        """Clamping unregistered symbol should raise in strict mode."""
        with _with_gate_mode("strict"):
            with pytest.raises(SymbolNotRegisteredError):
                validate_and_clamp_start_date("UNKNOWN", "2024-01-01")


# =============================================================================
# Query Functions
# =============================================================================


class TestGetEffectiveStartDate:
    """Test get_effective_start_date() function."""

    def test_btcusdt_returns_effective_start(self):
        """BTCUSDT should return effective_start."""
        result = get_effective_start_date("BTCUSDT")
        assert result == "2018-01-16"

    def test_xrpusdt_returns_listing_date(self):
        """XRPUSDT (no effective_start) should return listing_date."""
        result = get_effective_start_date("XRPUSDT")
        assert result == "2018-05-04"

    def test_unknown_returns_none(self):
        """Unknown symbol should return None."""
        result = get_effective_start_date("DOESNOTEXIST")
        assert result is None


class TestGetRegisteredSymbols:
    """Test filtered symbol listing."""

    def test_returns_tuple(self):
        """Should return a tuple."""
        result = get_registered_symbols()
        assert isinstance(result, tuple)

    def test_returns_sorted(self):
        """Symbols should be sorted alphabetically."""
        result = get_registered_symbols()
        assert result == tuple(sorted(result))

    def test_filter_by_tier_1(self):
        """Filtering by tier=1 should return tier-1 symbols."""
        result = get_registered_symbols(tier=1)
        assert len(result) > 0
        for symbol in result:
            entry = get_symbol_entries()[symbol]
            assert entry.tier == 1

    def test_filter_by_asset_class(self):
        """Filtering by asset_class should work."""
        result = get_registered_symbols(asset_class="crypto")
        assert len(result) > 0
        for symbol in result:
            entry = get_symbol_entries()[symbol]
            assert entry.asset_class == "crypto"

    def test_tier1_matches_constants(self):
        """get_registered_symbols(tier=1) should match TIER1_SYMBOLS constant.

        TIER1_SYMBOLS uses base symbols (BTC, ETH) while registry uses
        full symbols (BTCUSDT, ETHUSDT).
        """
        registry_tier1 = {
            s.replace("USDT", "") for s in get_registered_symbols(tier=1)
        }
        assert set(TIER1_SYMBOLS) == registry_tier1, (
            f"Mismatch: {set(TIER1_SYMBOLS) ^ registry_tier1}"
        )


# =============================================================================
# Transitions
# =============================================================================


class TestTransitions:
    """Test symbol transition records."""

    def test_returns_tuple(self):
        """get_transitions() should return a tuple."""
        result = get_transitions()
        assert isinstance(result, tuple)

    def test_has_matic_to_pol(self):
        """Should include MATIC_TO_POL transition."""
        transitions = get_transitions()
        names = [t.name for t in transitions]
        assert "MATIC_TO_POL" in names

    def test_matic_to_pol_fields(self):
        """MATIC_TO_POL should have correct metadata."""
        transitions = get_transitions()
        matic = next(t for t in transitions if t.name == "MATIC_TO_POL")
        assert isinstance(matic, SymbolTransition)
        assert matic.old_symbol == "MATICUSDT"
        assert matic.new_symbol == "POLUSDT"
        assert matic.gap_days == 2
        assert matic.last_old_date == date(2024, 9, 10)
        assert matic.first_new_date == date(2024, 9, 13)
        assert matic.status == "archived"

    def test_matic_transition_old_symbol_not_registered(self):
        """MATIC_TO_POL old_symbol should not be in registry (delisted)."""
        transitions = get_transitions()
        matic = next(t for t in transitions if t.name == "MATIC_TO_POL")
        entries = get_symbol_entries()
        assert matic.old_symbol not in entries

    def test_transition_is_frozen(self):
        """SymbolTransition should be frozen."""
        transitions = get_transitions()
        if transitions:
            with pytest.raises(AttributeError):
                transitions[0].name = "CHANGED"  # type: ignore[misc]


# =============================================================================
# Cache Management
# =============================================================================


class TestCacheManagement:
    """Test cache clearing and reload."""

    def test_clear_forces_reload(self):
        """clear_symbol_registry_cache() should force fresh load."""
        entries1 = get_symbol_entries()
        clear_symbol_registry_cache()
        entries2 = get_symbol_entries()
        # After clearing, should get a new object (not same reference)
        assert entries1 is not entries2
        # But content should be identical
        assert set(entries1.keys()) == set(entries2.keys())


# =============================================================================
# Exports
# =============================================================================


class TestExportedSymbols:
    """Test that all symbol registry symbols are properly exported from rangebar."""

    def test_types_exported(self):
        """SymbolEntry, SymbolTransition, SymbolNotRegisteredError should be exported."""
        assert SymbolEntry is not None
        assert SymbolTransition is not None
        assert SymbolNotRegisteredError is not None

    def test_functions_exported(self):
        """All registry functions should be callable exports."""
        assert callable(validate_symbol_registered)
        assert callable(validate_and_clamp_start_date)
        assert callable(get_effective_start_date)
        assert callable(get_registered_symbols)
        assert callable(get_symbol_entries)
        assert callable(get_transitions)
        assert callable(clear_symbol_registry_cache)

    def test_error_inherits_from_rangebar_error(self):
        """SymbolNotRegisteredError should be a subclass of RangeBarError."""
        assert issubclass(SymbolNotRegisteredError, RangeBarError)
