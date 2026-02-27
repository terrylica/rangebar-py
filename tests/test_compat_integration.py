"""Category C: Integration tests for alpha-forge compat layer (Issue #95).

Validates against live ClickHouse. Marked @pytest.mark.clickhouse
(skipped in CI, run locally with pytest -m clickhouse).

Run with: pytest tests/test_compat_integration.py -v
"""

from __future__ import annotations

import pytest

try:
    from rangebar.clickhouse import RangeBarCache, get_available_clickhouse_host

    _HOST = get_available_clickhouse_host()
    # Verify schema has close_time_ms (post-migration)
    with RangeBarCache() as _cache:
        _result = _cache.client.query(
            "SELECT name FROM system.columns "
            "WHERE database='rangebar_cache' AND table='range_bars' "
            "AND name='close_time_ms'"
        )
        if not _result.result_rows:
            _CH_AVAILABLE = False
            _skip_reason = "Legacy schema: timestamp_ms not yet migrated to close_time_ms"
        else:
            _CH_AVAILABLE = True
            _skip_reason = ""
except (ImportError, OSError, RuntimeError) as e:
    _CH_AVAILABLE = False
    _HOST = None
    _skip_reason = f"ClickHouse not available: {e}"

pytestmark = [
    pytest.mark.clickhouse,
    pytest.mark.skipif(not _CH_AVAILABLE, reason=_skip_reason),
]


def test_cache_coverage_live():
    """C1: get_cache_coverage() returns thresholds for BTCUSDT."""
    from rangebar.compat.availability import get_cache_coverage

    coverage = get_cache_coverage(["BTCUSDT"])
    assert "BTCUSDT" in coverage
    btc = coverage["BTCUSDT"]
    assert 250 in btc.thresholds_cached


def test_cache_coverage_bar_counts():
    """C2: bar_counts[250] > 0 for BTCUSDT."""
    from rangebar.compat.availability import get_cache_coverage

    coverage = get_cache_coverage(["BTCUSDT"])
    assert coverage["BTCUSDT"].bar_counts[250] > 0


def test_cache_coverage_date_ranges():
    """C3: cached_date_ranges returns valid (start, end) tuple."""
    from rangebar.compat.availability import get_cache_coverage

    coverage = get_cache_coverage(["BTCUSDT"])
    btc = coverage["BTCUSDT"]
    assert 250 in btc.cached_date_ranges
    start, end = btc.cached_date_ranges[250]
    assert start < end
    assert start.startswith("20")


def test_panel_roundtrip_live():
    """C4: get_range_bars_panel with microstructure returns feature.* columns."""
    from rangebar.compat.panel import get_range_bars_panel

    panel = get_range_bars_panel(
        "BTCUSDT",
        "2024-06-01",
        "2024-06-02",
        threshold_decimal_bps=250,
        include_microstructure=True,
    )
    assert len(panel) > 0
    feature_cols = [c for c in panel.columns if c.startswith("feature.")]
    assert len(feature_cols) > 0
    assert "feature.ofi" in feature_cols


def test_multi_symbol_panel_live():
    """C5: Multi-symbol panel is sorted by [symbol, ts]."""
    from rangebar.compat.panel import get_range_bars_panel

    panel = get_range_bars_panel(
        ["BTCUSDT", "ETHUSDT"],
        "2024-06-01",
        "2024-06-02",
        threshold_decimal_bps=250,
    )
    assert len(panel) > 0
    symbols = panel["symbol"].unique()
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    # Verify sorted by symbol, then ts
    assert panel["symbol"].is_monotonic_increasing or (
        panel.equals(panel.sort_values(["symbol", "ts"]).reset_index(drop=True))
    )
