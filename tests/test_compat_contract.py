"""Category D: Alpha-forge contract tests (Issue #95).

These tests define the interface alpha-forge will code against.
Changing any D-test requires a major version bump discussion.

Run with: pytest tests/test_compat_contract.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_panel():
    """Minimal panel-format DataFrame for contract validation."""
    from rangebar.compat.panel import to_panel_format

    df = pd.DataFrame(
        {
            "Open": [42000.0, 42105.0],
            "High": [42100.0, 42200.0],
            "Low": [41950.0, 42050.0],
            "Close": [42050.0, 42150.0],
            "Volume": [10.5, 8.3],
            "ofi": [0.3, -0.1],
            "vwap": [42020.0, 42120.0],
        },
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00:15", "2024-01-01 00:03:42"],
            name="timestamp",
        ),
    )
    return to_panel_format(df, "BTCUSDT")


@pytest.fixture
def multi_symbol_panel():
    """Two-symbol panel for sort contract tests."""
    from rangebar.compat.panel import to_panel_format

    btc = pd.DataFrame(
        {
            "Open": [42000.0],
            "High": [42100.0],
            "Low": [41950.0],
            "Close": [42050.0],
            "Volume": [10.5],
        },
        index=pd.DatetimeIndex(["2024-01-01 00:00:15"], name="timestamp"),
    )
    eth = pd.DataFrame(
        {
            "Open": [2200.0],
            "High": [2220.0],
            "Low": [2180.0],
            "Close": [2210.0],
            "Volume": [100.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 00:00:10"], name="timestamp"),
    )
    btc_panel = to_panel_format(btc, "BTCUSDT")
    eth_panel = to_panel_format(eth, "ETHUSDT")
    result = pd.concat([eth_panel, btc_panel], ignore_index=True)
    return result.sort_values(["symbol", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# D1-D6: Panel format contract
# ---------------------------------------------------------------------------

def test_panel_required_columns(sample_panel):
    """D1: Output always has ts (datetime64), symbol (str), price.* (float64)."""
    required = {"ts", "symbol", "price.open", "price.high", "price.low", "price.close", "price.volume"}
    assert required <= set(sample_panel.columns)


def test_panel_sorted_by_symbol_ts(multi_symbol_panel):
    """D2: Multi-symbol output sorted by [symbol, ts]."""
    panel = multi_symbol_panel
    resorted = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(panel, resorted)


def test_panel_ts_is_datetime(sample_panel):
    """D3: panel['ts'].dtype is datetime64."""
    assert pd.api.types.is_datetime64_any_dtype(sample_panel["ts"])


def test_panel_symbol_is_string(sample_panel):
    """D4: panel['symbol'].dtype is object (str)."""
    assert sample_panel["symbol"].dtype == object


def test_panel_price_columns_float(sample_panel):
    """D5: All price.* columns are float64."""
    price_cols = [c for c in sample_panel.columns if c.startswith("price.")]
    for col in price_cols:
        assert sample_panel[col].dtype == "float64", f"{col} is {sample_panel[col].dtype}"


def test_panel_feature_columns_float_or_nan(sample_panel):
    """D6: Feature columns are float64, NaN allowed for nullable features."""
    feature_cols = [c for c in sample_panel.columns if c.startswith("feature.")]
    for col in feature_cols:
        assert sample_panel[col].dtype == "float64", f"{col} is {sample_panel[col].dtype}"


# ---------------------------------------------------------------------------
# D7-D8: Manifest contract
# ---------------------------------------------------------------------------

def test_manifest_exposes_all_metadata_fields():
    """D7: FeatureMetadata has required fields."""
    from rangebar.compat.manifest import get_feature_manifest

    registry = get_feature_manifest()
    feat = registry.all_features()[0]

    assert hasattr(feat, "name")
    assert hasattr(feat, "group")
    assert hasattr(feat, "description")
    assert hasattr(feat, "formula")
    assert hasattr(feat, "value_range")
    assert hasattr(feat, "units")
    assert hasattr(feat, "nullable")
    assert hasattr(feat, "category")


def test_manifest_group_enum_values():
    """D8: FeatureGroup enum has exactly 3 values."""
    from rangebar.compat.manifest import FeatureGroup

    values = {g.value for g in FeatureGroup}
    assert values == {"microstructure", "inter_bar", "intra_bar"}


# ---------------------------------------------------------------------------
# D9-D10: Availability contract
# ---------------------------------------------------------------------------

def test_availability_returns_dict():
    """D9: get_cache_coverage() returns dict[str, SymbolAvailability] (even if empty)."""
    from rangebar.compat.availability import get_cache_coverage

    result = get_cache_coverage()
    assert isinstance(result, dict)


def test_availability_symbol_entry_fields():
    """D10: SymbolAvailability has required fields."""
    from rangebar.compat.availability import SymbolAvailability

    avail = SymbolAvailability(
        symbol="TEST",
        asset_class="crypto",
        exchange="binance",
        tier=1,
        effective_start="2020-01-01",
        listing_date="2020-01-01",
    )
    assert hasattr(avail, "symbol")
    assert hasattr(avail, "asset_class")
    assert hasattr(avail, "exchange")
    assert hasattr(avail, "thresholds_cached")
    assert hasattr(avail, "bar_counts")


# ---------------------------------------------------------------------------
# D11: Top-level import contract
# ---------------------------------------------------------------------------

def test_top_level_imports():
    """D11: All compat APIs importable from top-level rangebar package."""
    from rangebar import (
        get_available_symbols,
        get_cache_coverage,
        get_feature_manifest,
        get_range_bars_panel,
        to_panel_format,
    )

    assert callable(get_feature_manifest)
    assert callable(to_panel_format)
    assert callable(get_range_bars_panel)
    assert callable(get_cache_coverage)
    assert callable(get_available_symbols)
