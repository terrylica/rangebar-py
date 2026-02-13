"""Category B: Permanent regression tests for alpha-forge compat layer (Issue #95).

Guards against regressions in manifest parsing, panel conversion, and availability API.
These tests use synthetic data (no network, no ClickHouse).

Run with: pytest tests/test_compat_regression.py -v
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    from rangebar.compat.manifest import get_feature_manifest

    return get_feature_manifest()


@pytest.fixture
def sample_bars():
    """Minimal synthetic range bar DataFrame for panel tests."""
    return pd.DataFrame(
        {
            "Open": [42000.0, 42105.0, 42200.0],
            "High": [42100.0, 42200.0, 42300.0],
            "Low": [41950.0, 42050.0, 42150.0],
            "Close": [42050.0, 42150.0, 42250.0],
            "Volume": [10.5, 8.3, 12.1],
        },
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00:15", "2024-01-01 00:03:42", "2024-01-01 00:07:18"],
            name="timestamp",
        ),
    )


@pytest.fixture
def sample_bars_with_features(sample_bars):
    """Range bars with microstructure feature columns."""
    df = sample_bars.copy()
    df["ofi"] = [0.3, -0.1, 0.5]
    df["vwap"] = [42020.0, 42120.0, 42220.0]
    df["buy_volume"] = [6.5, 3.1, 8.0]
    df["sell_volume"] = [4.0, 5.2, 4.1]
    df["duration_us"] = [15000000, 22000000, 18000000]
    return df


# ---------------------------------------------------------------------------
# B1-B8: Manifest tests
# ---------------------------------------------------------------------------

def test_manifest_names_match_constants(registry):
    """B1: Manifest feature names must match constants.py column tuples."""
    from rangebar.compat.manifest import FeatureGroup
    from rangebar.constants import (
        INTER_BAR_FEATURE_COLUMNS,
        INTRA_BAR_FEATURE_COLUMNS,
        MICROSTRUCTURE_COLUMNS,
    )

    assert set(registry.column_names(FeatureGroup.MICROSTRUCTURE)) == set(MICROSTRUCTURE_COLUMNS)
    assert set(registry.column_names(FeatureGroup.INTER_BAR)) == set(INTER_BAR_FEATURE_COLUMNS)
    assert set(registry.column_names(FeatureGroup.INTRA_BAR)) == set(INTRA_BAR_FEATURE_COLUMNS)


def test_manifest_required_fields(registry):
    """B2: Every feature must have description, group, units, nullable, category, tier."""
    for feat in registry.all_features():
        assert feat.description, f"{feat.name} missing description"
        assert feat.group is not None, f"{feat.name} missing group"
        assert feat.units, f"{feat.name} missing units"
        assert feat.nullable is not None, f"{feat.name} missing nullable"
        assert feat.category, f"{feat.name} missing category"
        assert feat.tier is not None, f"{feat.name} missing tier"


def test_manifest_ofi_range_bounded(registry):
    """B3: OFI must be bounded [-1, 1] and non-nullable."""
    ofi = registry.get("ofi")
    assert ofi is not None
    assert ofi.value_range == (-1.0, 1.0)
    assert ofi.nullable is False


def test_manifest_lookback_nullable(registry):
    """B4: All inter_bar features must be nullable."""
    from rangebar.compat.manifest import FeatureGroup

    for feat in registry.by_group(FeatureGroup.INTER_BAR):
        assert feat.nullable is True, f"{feat.name} should be nullable"


def test_manifest_intra_nullable(registry):
    """B5: All intra_bar features must be nullable."""
    from rangebar.compat.manifest import FeatureGroup

    for feat in registry.by_group(FeatureGroup.INTRA_BAR):
        assert feat.nullable is True, f"{feat.name} should be nullable"


def test_manifest_json_roundtrip(registry):
    """B6: Manifest to_dict() → json.dumps() → json.loads() preserves all fields."""
    d = registry.to_dict()
    serialized = json.dumps(d)
    restored = json.loads(serialized)

    assert restored["schema_version"] == registry.schema_version
    assert len(restored["features"]) == len(registry.all_features())

    for name, fdata in restored["features"].items():
        original = registry.get(name)
        assert original is not None
        assert fdata["description"] == original.description
        assert fdata["units"] == original.units


def test_manifest_filter_by_group(registry):
    """B7: by_group() returns correct counts."""
    from rangebar.compat.manifest import FeatureGroup

    assert len(registry.by_group(FeatureGroup.MICROSTRUCTURE)) == 15
    assert len(registry.by_group(FeatureGroup.INTER_BAR)) == 16


def test_manifest_filter_by_category(registry):
    """B8: by_category() returns features matching category."""
    order_flow = registry.by_category("order_flow")
    names = {f.name for f in order_flow}
    assert "ofi" in names
    assert "aggression_ratio" in names


# ---------------------------------------------------------------------------
# B9-B12: Panel tests
# ---------------------------------------------------------------------------

def test_panel_ohlc_invariants(sample_bars):
    """B9: After to_panel_format(), OHLC invariants hold."""
    from rangebar.compat.panel import to_panel_format

    panel = to_panel_format(sample_bars, "BTCUSDT")
    assert (panel["price.high"] >= panel["price.open"]).all()
    assert (panel["price.high"] >= panel["price.close"]).all()
    assert (panel["price.low"] <= panel["price.open"]).all()
    assert (panel["price.low"] <= panel["price.close"]).all()


def test_panel_no_nan_in_required(sample_bars):
    """B10: ts and symbol columns never NaN."""
    from rangebar.compat.panel import to_panel_format

    panel = to_panel_format(sample_bars, "BTCUSDT")
    assert not panel["ts"].isna().any()
    assert not panel["symbol"].isna().any()


def test_panel_feature_prefix(sample_bars_with_features):
    """B11: Feature columns start with 'feature.' prefix."""
    from rangebar.compat.panel import to_panel_format

    panel = to_panel_format(sample_bars_with_features, "BTCUSDT")
    feature_cols = [c for c in panel.columns if c.startswith("feature.")]
    assert len(feature_cols) > 0
    assert "feature.ofi" in feature_cols


def test_panel_custom_prefix(sample_bars_with_features):
    """B12: Custom feature_prefix applies correctly."""
    from rangebar.compat.panel import to_panel_format

    panel = to_panel_format(sample_bars_with_features, "BTCUSDT", feature_prefix="rb")
    rb_cols = [c for c in panel.columns if c.startswith("rb.")]
    assert len(rb_cols) > 0
    assert "rb.ofi" in rb_cols
    # No feature.* columns should exist
    assert not any(c.startswith("feature.") for c in panel.columns)


# ---------------------------------------------------------------------------
# B13-B14: Availability tests
# ---------------------------------------------------------------------------

def test_availability_symbol_filtering():
    """B13: get_available_symbols() filters by asset_class and min_tier."""
    from rangebar.compat.availability import get_available_symbols

    crypto = get_available_symbols(asset_class="crypto")
    assert all(s["asset_class"] == "crypto" for s in crypto)

    tier1 = get_available_symbols(min_tier=1)
    assert all(s["tier"] is not None and s["tier"] <= 1 for s in tier1)


def test_availability_to_dict():
    """B14: SymbolAvailability.to_dict() is JSON-serializable."""
    from rangebar.compat.availability import SymbolAvailability

    avail = SymbolAvailability(
        symbol="BTCUSDT",
        asset_class="crypto",
        exchange="binance",
        tier=1,
        effective_start="2017-08-17",
        listing_date="2017-08-17",
        thresholds_cached=[250, 500],
        cached_date_ranges={250: ("2019-01-01", "2025-12-31")},
        bar_counts={250: 1000000},
    )

    d = avail.to_dict()
    serialized = json.dumps(d)
    restored = json.loads(serialized)
    assert restored["symbol"] == "BTCUSDT"
    assert restored["thresholds_cached"] == [250, 500]
    assert "250" in restored["cached_date_ranges"]
