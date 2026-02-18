"""Tests for sidecar checkpoint injection validation (Issue #96).

Verifies that stale checkpoints are skipped when gap-fill already wrote
authoritative bars for the same symbol/threshold pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from rangebar.sidecar import _inject_checkpoints


@dataclass
class _FakeBackfillResult:
    bars_written: int
    gap_seconds: float = 0.0


def _make_checkpoint(symbol: str = "BTCUSDT", threshold: int = 250) -> dict:
    """Return a minimal valid checkpoint dict."""
    return {
        "symbol": symbol,
        "threshold": threshold,
        "processor_checkpoint": {
            "has_incomplete_bar": True,
            "last_trade_id": 123456,
        },
        "updated_at": 1_700_000_000.0,
    }


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.set_checkpoint = MagicMock()
    return engine


# Patch the source modules that _inject_checkpoints imports from at call time
_PATCHES = [
    patch("rangebar.hooks.emit_hook", new=MagicMock()),
    patch("rangebar.logging.log_checkpoint_event", new=MagicMock()),
]


def _apply_patches() -> None:
    for p in _PATCHES:
        p.start()


def _stop_patches() -> None:
    for p in _PATCHES:
        p.stop()


class TestCheckpointSkipAfterGapFill:
    """Issue #96: checkpoint must be skipped when gap-fill wrote bars."""

    def setup_method(self):
        _apply_patches()

    def teardown_method(self):
        _stop_patches()

    @patch("rangebar.sidecar._load_checkpoint")
    def test_skip_checkpoint_when_gap_fill_wrote_bars(
        self, mock_load, mock_engine,
    ):
        """Gap-fill wrote 71 bars → checkpoint must NOT be injected."""
        mock_load.return_value = _make_checkpoint()

        gap_fill_results = {
            "BTCUSDT@250": _FakeBackfillResult(bars_written=71, gap_seconds=544.0),
        }

        loaded = _inject_checkpoints(
            mock_engine,
            symbols=["BTCUSDT"],
            thresholds=[250],
            trace_id="test-trace",
            gap_fill_results=gap_fill_results,
        )

        assert loaded == 0
        mock_engine.set_checkpoint.assert_not_called()

    @patch("rangebar.sidecar._load_checkpoint")
    def test_inject_checkpoint_when_gap_fill_wrote_zero_bars(
        self, mock_load, mock_engine,
    ):
        """Gap-fill wrote 0 bars (fresh data) → checkpoint IS injected."""
        mock_load.return_value = _make_checkpoint()

        gap_fill_results = {
            "BTCUSDT@250": _FakeBackfillResult(bars_written=0, gap_seconds=30.0),
        }

        loaded = _inject_checkpoints(
            mock_engine,
            symbols=["BTCUSDT"],
            thresholds=[250],
            trace_id="test-trace",
            gap_fill_results=gap_fill_results,
        )

        assert loaded == 1
        mock_engine.set_checkpoint.assert_called_once()

    @patch("rangebar.sidecar._load_checkpoint")
    def test_inject_checkpoint_when_gap_fill_disabled(
        self, mock_load, mock_engine,
    ):
        """gap_fill_on_startup=False → gap_fill_results=None → checkpoint IS injected."""
        mock_load.return_value = _make_checkpoint()

        loaded = _inject_checkpoints(
            mock_engine,
            symbols=["BTCUSDT"],
            thresholds=[250],
            trace_id="test-trace",
            gap_fill_results=None,
        )

        assert loaded == 1
        mock_engine.set_checkpoint.assert_called_once()

    @patch("rangebar.sidecar._load_checkpoint")
    def test_mixed_pairs_selective_skip(
        self, mock_load, mock_engine,
    ):
        """Gap-fill wrote bars for one pair but not another → selective skip."""
        mock_load.return_value = _make_checkpoint()

        gap_fill_results = {
            "BTCUSDT@250": _FakeBackfillResult(bars_written=71),
            "BTCUSDT@500": _FakeBackfillResult(bars_written=0),
        }

        loaded = _inject_checkpoints(
            mock_engine,
            symbols=["BTCUSDT"],
            thresholds=[250, 500],
            trace_id="test-trace",
            gap_fill_results=gap_fill_results,
        )

        # Only threshold=500 should be injected (250 was covered by gap-fill)
        assert loaded == 1
        mock_engine.set_checkpoint.assert_called_once_with(
            "BTCUSDT", 500, _make_checkpoint()["processor_checkpoint"],
        )
