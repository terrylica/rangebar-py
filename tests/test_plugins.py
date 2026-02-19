# Issue #98: Tests for FeatureProvider plugin system.
"""Tests for plugin discovery, enrichment, column registry, and bulk operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import rangebar.constants as _rc
from rangebar.constants import (
    get_all_columns,
    register_plugin_columns,
)
from rangebar.plugins.loader import (
    enrich_bars,
    reset_provider_cache,
)
from rangebar.plugins.protocol import FeatureProvider

# ---------------------------------------------------------------------------
# Mock FeatureProvider
# ---------------------------------------------------------------------------


class MockLaguerreProvider:
    """Mock provider that adds 2 laguerre columns."""

    @property
    def name(self) -> str:
        return "laguerre"

    @property
    def version(self) -> str:
        return "2.1.0"

    @property
    def columns(self) -> tuple[str, ...]:
        return ("laguerre_rsi", "laguerre_regime")

    @property
    def min_bars(self) -> int:
        return 10

    def enrich(
        self,
        bars: pd.DataFrame,
        symbol: str,
        threshold_decimal_bps: int,
    ) -> pd.DataFrame:
        n = len(bars)
        bars["laguerre_rsi"] = np.linspace(0.1, 0.9, n)
        bars["laguerre_regime"] = np.where(
            np.linspace(0.1, 0.9, n) > 0.5, 2, 0,
        )
        # NaN for warmup period
        bars.loc[bars.index[: self.min_bars], "laguerre_rsi"] = np.nan
        bars.loc[bars.index[: self.min_bars], "laguerre_regime"] = np.nan
        return bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_plugin_state() -> None:  # type: ignore[misc]
    """Reset plugin state before and after each test."""
    _rc._PLUGIN_FEATURE_COLUMNS.clear()
    reset_provider_cache()
    yield
    _rc._PLUGIN_FEATURE_COLUMNS.clear()
    reset_provider_cache()


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Sample range bar DataFrame matching Rust processing output."""
    n = 50
    np.random.seed(42)
    base_price = 42000.0
    returns = np.random.randn(n) * 0.001
    prices = base_price * np.cumprod(1 + returns)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")

    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.001,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.exponential(5.0, n),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_mock_provider_satisfies_protocol(self) -> None:
        provider = MockLaguerreProvider()
        assert isinstance(provider, FeatureProvider)

    def test_non_provider_rejected(self) -> None:
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), FeatureProvider)

    def test_partial_provider_rejected(self) -> None:
        class PartialProvider:
            @property
            def name(self) -> str:
                return "partial"

            # Missing: version, columns, min_bars, enrich

        assert not isinstance(PartialProvider(), FeatureProvider)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


class TestEnrichment:
    def test_enrich_adds_columns(self, sample_bars: pd.DataFrame) -> None:
        provider = MockLaguerreProvider()
        result = provider.enrich(sample_bars, "BTCUSDT", 500)

        assert "laguerre_rsi" in result.columns
        assert "laguerre_regime" in result.columns
        assert len(result) == len(sample_bars)

    def test_enrich_preserves_existing_columns(
        self, sample_bars: pd.DataFrame,
    ) -> None:
        original_cols = list(sample_bars.columns)
        provider = MockLaguerreProvider()
        result = provider.enrich(sample_bars, "BTCUSDT", 500)

        for col in original_cols:
            assert col in result.columns

    def test_enrich_returns_same_object(
        self, sample_bars: pd.DataFrame,
    ) -> None:
        """R3: In-place mutation — same object returned."""
        provider = MockLaguerreProvider()
        result = provider.enrich(sample_bars, "BTCUSDT", 500)
        assert result is sample_bars

    def test_enrich_warmup_nans(self, sample_bars: pd.DataFrame) -> None:
        provider = MockLaguerreProvider()
        result = provider.enrich(sample_bars, "BTCUSDT", 500)

        # First min_bars rows should be NaN
        warmup = result["laguerre_rsi"].iloc[: provider.min_bars]
        assert warmup.isna().all()

        # After warmup, values should be populated
        post_warmup = result["laguerre_rsi"].iloc[provider.min_bars :]
        assert post_warmup.notna().all()

    def test_enrich_empty_df_noop(self) -> None:
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        result = enrich_bars(empty_df, "BTCUSDT", 500)
        assert result is empty_df
        assert len(result.columns) == 5


# ---------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------


class TestColumnRegistry:
    def test_register_plugin_columns(self) -> None:
        register_plugin_columns(("laguerre_rsi", "laguerre_regime"))
        assert "laguerre_rsi" in _rc._PLUGIN_FEATURE_COLUMNS
        assert "laguerre_regime" in _rc._PLUGIN_FEATURE_COLUMNS

    def test_register_idempotent(self) -> None:
        register_plugin_columns(("laguerre_rsi",))
        register_plugin_columns(("laguerre_rsi",))
        assert _rc._PLUGIN_FEATURE_COLUMNS.count("laguerre_rsi") == 1

    def test_get_all_columns_includes_plugins(self) -> None:
        register_plugin_columns(("laguerre_rsi",))
        all_cols = get_all_columns()
        assert "laguerre_rsi" in all_cols


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def _patch_entry_points(self, return_value: object) -> object:  # type: ignore[name-defined]
        """Patch entry_points at the source module (local import in discover_providers)."""
        return patch(
            "importlib.metadata.entry_points", return_value=return_value,
        )

    def test_no_providers_returns_empty(self) -> None:
        from rangebar.plugins.loader import discover_providers

        with self._patch_entry_points([]):
            providers = discover_providers()
            assert providers == []

    def test_discover_loads_valid_provider(self) -> None:
        from rangebar.plugins.loader import discover_providers

        mock_ep = MagicMock()
        mock_ep.name = "laguerre"
        mock_ep.load.return_value = MockLaguerreProvider

        with self._patch_entry_points([mock_ep]):
            providers = discover_providers()
            assert len(providers) == 1
            assert providers[0].name == "laguerre"
            assert "laguerre_rsi" in _rc._PLUGIN_FEATURE_COLUMNS

    def test_discover_skips_invalid_provider(self) -> None:
        from rangebar.plugins.loader import discover_providers

        class BadPlugin:
            pass

        mock_ep = MagicMock()
        mock_ep.name = "bad"
        mock_ep.load.return_value = BadPlugin

        with self._patch_entry_points([mock_ep]):
            providers = discover_providers()
            assert providers == []

    def test_discover_handles_import_error(self) -> None:
        from rangebar.plugins.loader import discover_providers

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("no module")

        with self._patch_entry_points([mock_ep]):
            providers = discover_providers()
            assert providers == []


# ---------------------------------------------------------------------------
# enrich_bars orchestration
# ---------------------------------------------------------------------------


class TestEnrichBarsOrchestration:
    def _patch_entry_points(self, return_value: object) -> object:  # type: ignore[name-defined]
        """Patch entry_points at the source module (local import in discover_providers)."""
        return patch(
            "importlib.metadata.entry_points", return_value=return_value,
        )

    def test_no_providers_noop(self, sample_bars: pd.DataFrame) -> None:
        """Zero overhead when no plugins installed."""
        with self._patch_entry_points([]):
            result = enrich_bars(sample_bars, "BTCUSDT", 500)
            assert result is sample_bars
            assert "laguerre_rsi" not in result.columns

    def test_with_mock_provider(self, sample_bars: pd.DataFrame) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "laguerre"
        mock_ep.load.return_value = MockLaguerreProvider

        with self._patch_entry_points([mock_ep]):
            result = enrich_bars(sample_bars, "BTCUSDT", 500)
            assert "laguerre_rsi" in result.columns
            assert "laguerre_regime" in result.columns

    def test_provider_failure_non_fatal(
        self, sample_bars: pd.DataFrame,
    ) -> None:
        """Provider failure is logged but doesn't crash the pipeline."""

        class FailingProvider:
            name = "failing"
            version = "1.0"
            columns = ("fail_col",)
            min_bars = 0

            def enrich(self, bars, symbol, threshold_decimal_bps) -> None:
                msg = "computation failed"
                raise ValueError(msg)

        mock_ep = MagicMock()
        mock_ep.name = "failing"
        mock_ep.load.return_value = FailingProvider

        with self._patch_entry_points([mock_ep]):
            # Should not raise
            result = enrich_bars(sample_bars, "BTCUSDT", 500)
            assert result is sample_bars
