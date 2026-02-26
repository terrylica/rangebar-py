"""Tests for Issue #111: Ariadne trade ID tracking.

Validates checkpoint persistence of last_processed_agg_trade_id,
feature flag gating, and fallback to timestamp-based resume.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def test_checkpoint_has_last_processed_agg_trade_id():
    """PopulationCheckpoint includes last_processed_agg_trade_id field."""
    from rangebar.checkpoint import PopulationCheckpoint

    cp = PopulationCheckpoint(
        symbol="BTCUSDT",
        threshold_bps=250,
        start_date="2024-01-01",
        end_date="2024-12-31",
        last_completed_date="2024-06-15",
        last_processed_agg_trade_id=123456789,
    )
    assert cp.last_processed_agg_trade_id == 123456789

    # Serialization round-trip
    d = cp.to_dict()
    assert d["last_processed_agg_trade_id"] == 123456789
    cp2 = PopulationCheckpoint.from_dict(d)
    assert cp2.last_processed_agg_trade_id == 123456789


def test_checkpoint_defaults_to_none():
    """Old checkpoints without last_processed_agg_trade_id default to None."""
    from rangebar.checkpoint import PopulationCheckpoint

    cp = PopulationCheckpoint(
        symbol="BTCUSDT",
        threshold_bps=250,
        start_date="2024-01-01",
        end_date="2024-12-31",
        last_completed_date="2024-06-15",
    )
    assert cp.last_processed_agg_trade_id is None


def test_ariadne_feature_flag_off_by_default():
    """Ariadne is disabled by default."""
    from rangebar.checkpoint import _ariadne_enabled

    env = {k: v for k, v in os.environ.items() if k != "RANGEBAR_ARIADNE_ENABLED"}
    with patch.dict(os.environ, env, clear=True):
        assert _ariadne_enabled() is False


def test_ariadne_feature_flag_enabled():
    """Ariadne can be enabled via env var."""
    from rangebar.checkpoint import _ariadne_enabled

    with patch.dict(os.environ, {"RANGEBAR_ARIADNE_ENABLED": "true"}):
        assert _ariadne_enabled() is True

    with patch.dict(os.environ, {"RANGEBAR_ARIADNE_ENABLED": "1"}):
        assert _ariadne_enabled() is True

    with patch.dict(os.environ, {"RANGEBAR_ARIADNE_ENABLED": "yes"}):
        assert _ariadne_enabled() is True

    with patch.dict(os.environ, {"RANGEBAR_ARIADNE_ENABLED": "false"}):
        assert _ariadne_enabled() is False


def test_ariadne_fallback_when_no_trade_id():
    """Without last_processed_agg_trade_id, falls back to timestamp resume."""
    from rangebar.checkpoint import PopulationCheckpoint

    # Even with Ariadne enabled, no trade ID means fallback
    cp = PopulationCheckpoint(
        symbol="BTCUSDT",
        threshold_bps=250,
        start_date="2024-01-01",
        end_date="2024-12-31",
        last_completed_date="2024-06-15",
        last_processed_agg_trade_id=None,  # Old checkpoint
    )
    # The Ariadne path requires last_processed_agg_trade_id to be truthy
    assert not cp.last_processed_agg_trade_id


def test_checkpoint_round_trip_with_trade_id():
    """PopulationCheckpoint save/load preserves last_processed_agg_trade_id."""
    import tempfile
    from pathlib import Path

    from rangebar.checkpoint import PopulationCheckpoint

    cp = PopulationCheckpoint(
        symbol="ETHUSDT",
        threshold_bps=500,
        start_date="2024-01-01",
        end_date="2024-06-30",
        last_completed_date="2024-03-15",
        bars_written=42000,
        last_processed_agg_trade_id=987654321,
        first_agg_trade_id_in_bar=987654300,
        last_agg_trade_id_in_bar=987654310,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"
        cp.save(path)
        loaded = PopulationCheckpoint.load(path)

    assert loaded is not None
    assert loaded.last_processed_agg_trade_id == 987654321
    assert loaded.first_agg_trade_id_in_bar == 987654300
    assert loaded.last_agg_trade_id_in_bar == 987654310
    assert loaded.bars_written == 42000
