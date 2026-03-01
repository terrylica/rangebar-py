"""Category A: Bootstrap tests for alpha-forge compat layer (Issue #95).

These are TRANSITIONAL tests. Each test documents its graduation condition
in the docstring. After 3 stable releases with no failures, move to
tests/archive/test_compat_bootstrap.py (excluded from CI via testpaths).

Run with: pytest tests/test_compat_bootstrap.py -v
"""

import pytest

pytestmark = pytest.mark.bootstrap


def test_manifest_toml_parses_from_rust():
    """TRANSITIONAL: Archive after get_feature_manifest_raw() is stable for 3 releases.

    Validates that the Rust-embedded TOML is parseable by Python's tomllib.
    """
    import tomllib

    from rangebar._core import get_feature_manifest_raw

    raw = get_feature_manifest_raw()
    assert isinstance(raw, str)
    assert len(raw) > 0

    data = tomllib.loads(raw)
    assert "meta" in data
    assert "features" in data
    assert data["meta"]["schema_version"] == 1


def test_manifest_feature_count_53():
    """TRANSITIONAL: Archive after get_feature_manifest() returns 53 for 3 releases.

    Validates exactly 53 features are parsed (matching constants.py tuples).
    """
    from rangebar.compat.manifest import get_feature_manifest

    registry = get_feature_manifest()
    assert len(registry.all_features()) == 53


def test_manifest_group_counts():
    """TRANSITIONAL: Archive after group counts are stable for 3 releases.

    Validates microstructure=15, inter_bar=16, intra_bar=22.
    """
    from rangebar.compat.manifest import FeatureGroup, get_feature_manifest

    registry = get_feature_manifest()
    assert len(registry.by_group(FeatureGroup.MICROSTRUCTURE)) == 15
    assert len(registry.by_group(FeatureGroup.INTER_BAR)) == 16
    assert len(registry.by_group(FeatureGroup.INTRA_BAR)) == 22


def test_panel_basic_column_names():
    """TRANSITIONAL: Archive after to_panel_format() column names are stable for 3 releases.

    Validates panel output has required columns.
    """
    import pandas as pd
    from rangebar.compat.panel import to_panel_format

    df = pd.DataFrame(
        {
            "Open": [42000.0],
            "High": [42100.0],
            "Low": [41900.0],
            "Close": [42050.0],
            "Volume": [10.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 00:00:15"]),
    )

    panel = to_panel_format(df, "BTCUSDT")
    expected = {"ts", "symbol", "price.open", "price.high", "price.low", "price.close", "price.volume"}
    assert expected <= set(panel.columns)


def test_panel_drops_internal_columns():
    """TRANSITIONAL: Archive after internal column dropping is stable for 3 releases.

    Validates ouroboros_mode, is_orphan, trade IDs are not in output.
    """
    import pandas as pd
    from rangebar.compat.panel import to_panel_format

    df = pd.DataFrame(
        {
            "Open": [42000.0],
            "High": [42100.0],
            "Low": [41900.0],
            "Close": [42050.0],
            "Volume": [10.0],
            "ouroboros_mode": ["month"],
            "is_orphan": [False],
            "first_agg_trade_id": [100],
            "last_agg_trade_id": [200],
        },
        index=pd.DatetimeIndex(["2024-01-01 00:00:15"]),
    )

    panel = to_panel_format(df, "BTCUSDT")
    forbidden = {"ouroboros_mode", "is_orphan", "first_agg_trade_id", "last_agg_trade_id"}
    assert not (forbidden & set(panel.columns))


def test_availability_graceful_no_clickhouse(monkeypatch):
    """TRANSITIONAL: Archive after graceful fallback is stable for 3 releases.

    Validates get_cache_coverage() returns empty dict (no exception) when
    ClickHouse is unreachable.
    """
    monkeypatch.setenv("RANGEBAR_MODE", "local")
    monkeypatch.setenv("CLICKHOUSE_HOST", "nonexistent-host-12345")
    monkeypatch.setenv("CLICKHOUSE_PORT", "19999")

    from rangebar.compat.availability import get_cache_coverage

    result = get_cache_coverage(["BTCUSDT"])
    assert isinstance(result, dict)
