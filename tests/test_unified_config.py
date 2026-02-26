"""Tests for Issue #110: Unified configuration.

Validates Settings singleton, env var loading, and nested config sections.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_settings_singleton():
    """Settings.get() returns the same instance on repeated calls."""
    from rangebar.config import Settings

    Settings.reload()  # Start fresh
    s1 = Settings.get()
    s2 = Settings.get()
    assert s1 is s2


def test_settings_reload_creates_new_instance():
    """Settings.reload() creates a new instance."""
    from rangebar.config import Settings

    s1 = Settings.get()
    s2 = Settings.reload()
    assert s1 is not s2
    # After reload, get() returns the new one
    assert Settings.get() is s2


def test_population_defaults():
    """PopulationConfig has correct defaults without env vars."""
    from rangebar.config import PopulationConfig

    with patch.dict(os.environ, {}, clear=False):
        # Remove any RANGEBAR_ env vars that would override
        env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("RANGEBAR_")
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = PopulationConfig()
            assert cfg.default_threshold == 250
            assert cfg.ouroboros_mode == "year"
            assert cfg.inter_bar_lookback_count == 200
            assert cfg.include_intra_bar_features is True
            assert cfg.symbol_gate == "strict"
            assert cfg.continuity_tolerance == pytest.approx(0.001)


def test_monitoring_defaults():
    """MonitoringConfig has correct defaults without env vars."""
    from rangebar.config import MonitoringConfig

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("RANGEBAR_")
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = MonitoringConfig()
        assert cfg.telegram_token is None
        assert cfg.telegram_chat_id == ""
        assert cfg.recency_fresh_threshold_min == 30
        assert cfg.recency_stale_threshold_min == 120
        assert cfg.recency_critical_threshold_min == 1440
        assert cfg.environment == "development"
        assert cfg.git_sha == "unknown"


def test_settings_loads_from_env():
    """Settings picks up env var overrides."""
    from rangebar.config import Settings

    env_overrides = {
        "RANGEBAR_CRYPTO_MIN_THRESHOLD": "500",
        "RANGEBAR_OUROBOROS_MODE": "month",
        "RANGEBAR_INTER_BAR_LOOKBACK_COUNT": "100",
        "RANGEBAR_INCLUDE_INTRA_BAR_FEATURES": "false",
        "RANGEBAR_TELEGRAM_CHAT_ID": "12345",
        "RANGEBAR_ENV": "production",
        "RANGEBAR_RECENCY_FRESH_THRESHOLD_MIN": "60",
    }
    with patch.dict(os.environ, env_overrides):
        s = Settings.reload()
        assert s.population.default_threshold == 500
        assert s.population.ouroboros_mode == "month"
        assert s.population.inter_bar_lookback_count == 100
        assert s.population.include_intra_bar_features is False
        assert s.monitoring.telegram_chat_id == "12345"
        assert s.monitoring.environment == "production"
        assert s.monitoring.recency_fresh_threshold_min == 60


def test_clickhouse_property_returns_config():
    """Settings.clickhouse returns a ClickHouseConfig instance."""
    from rangebar.clickhouse.config import ClickHouseConfig
    from rangebar.config import Settings

    s = Settings.reload()
    ch = s.clickhouse
    assert isinstance(ch, ClickHouseConfig)
    assert ch.database == "rangebar_cache"


def test_sidecar_property_returns_config():
    """Settings.sidecar returns a SidecarConfig instance."""
    from rangebar.config import Settings
    from rangebar.sidecar import SidecarConfig

    s = Settings.reload()
    sc = s.sidecar
    assert isinstance(sc, SidecarConfig)
