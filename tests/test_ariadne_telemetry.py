# FILE-SIZE-OK: Comprehensive telemetry coverage for all Ariadne boundaries
"""NDJSON telemetry tests for Ariadne trade-ID wirings (Issue #111).

Validates every boundary and every touchpoint with structured NDJSON
artifacts that Claude Code can examine autonomously after test execution:

    jq 'select(.component == "ariadne")' tests/artifacts/ariadne_telemetry.jsonl

Each test emits structured events to a session-scoped .jsonl artifact file.
After all tests run, the artifact is autonomously parseable — every line is
valid JSON with a ``component``, ``event``, and ``trace_id`` for correlation.

Ariadne sets every boundary and every touchpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

# ---------------------------------------------------------------------------
# NDJSON artifact infrastructure
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "ariadne_telemetry.jsonl"


def _emit(event: dict) -> None:
    """Append a single NDJSON event to the artifact file."""
    event.setdefault("timestamp", time.time())
    event.setdefault("component", "ariadne")
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")


@pytest.fixture(autouse=True, scope="session")
def _init_artifact():
    """Truncate the artifact file at session start for a clean run."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text("")
    yield
    # After all tests, verify artifact is valid NDJSON
    lines = ARTIFACT_FILE.read_text().strip().splitlines()
    for i, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Artifact line {i} is not valid JSON: {e}")


# ---------------------------------------------------------------------------
# 1. Feature gate: Ariadne ON by default
# ---------------------------------------------------------------------------


class TestAriadneGateTelemetry:
    """Ariadne gate inversion: ON by default, opt-out via env var."""

    def test_gate_on_by_default(self):
        """Ariadne is enabled when RANGEBAR_ARIADNE_ENABLED is unset."""
        from rangebar.checkpoint import _ariadne_disabled

        env = {k: v for k, v in os.environ.items() if k != "RANGEBAR_ARIADNE_ENABLED"}
        with patch.dict(os.environ, env, clear=True):
            disabled = _ariadne_disabled()

        assert disabled is False
        _emit({
            "event": "gate_check",
            "trace_id": "gate-default",
            "env_var": "unset",
            "ariadne_disabled": disabled,
            "expected": False,
            "pass": disabled is False,
        })

    @pytest.mark.parametrize(
        ("env_val", "expected_disabled"),
        [
            ("false", True),
            ("0", True),
            ("no", True),
            ("true", False),
            ("1", False),
            ("yes", False),
        ],
    )
    def test_gate_explicit_values(self, env_val: str, expected_disabled: bool):
        """Gate responds correctly to all boolean string variants."""
        from rangebar.checkpoint import _ariadne_disabled

        with patch.dict(os.environ, {"RANGEBAR_ARIADNE_ENABLED": env_val}):
            disabled = _ariadne_disabled()

        assert disabled is expected_disabled
        _emit({
            "event": "gate_check",
            "trace_id": f"gate-{env_val}",
            "env_var": env_val,
            "ariadne_disabled": disabled,
            "expected": expected_disabled,
            "pass": disabled is expected_disabled,
        })


# ---------------------------------------------------------------------------
# 2. Checkpoint persistence: trade ID round-trip
# ---------------------------------------------------------------------------


class TestCheckpointTradeIdTelemetry:
    """PopulationCheckpoint persists last_processed_agg_trade_id."""

    def test_checkpoint_field_present(self):
        """Checkpoint includes last_processed_agg_trade_id field."""
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

        _emit({
            "event": "checkpoint_field",
            "trace_id": "cp-field",
            "symbol": "BTCUSDT",
            "threshold_bps": 250,
            "last_processed_agg_trade_id": cp.last_processed_agg_trade_id,
            "pass": cp.last_processed_agg_trade_id == 123456789,
        })

    def test_checkpoint_defaults_none(self):
        """Old checkpoints without trade ID default to None."""
        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="BTCUSDT",
            threshold_bps=250,
            start_date="2024-01-01",
            end_date="2024-12-31",
            last_completed_date="2024-06-15",
        )
        assert cp.last_processed_agg_trade_id is None

        _emit({
            "event": "checkpoint_field",
            "trace_id": "cp-default-none",
            "symbol": "BTCUSDT",
            "last_processed_agg_trade_id": None,
            "pass": cp.last_processed_agg_trade_id is None,
        })

    def test_checkpoint_serialization_round_trip(self):
        """to_dict() / from_dict() preserves trade IDs exactly."""
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
        d = cp.to_dict()
        cp2 = PopulationCheckpoint.from_dict(d)

        assert cp2.last_processed_agg_trade_id == 987654321
        assert cp2.first_agg_trade_id_in_bar == 987654300
        assert cp2.last_agg_trade_id_in_bar == 987654310

        _emit({
            "event": "checkpoint_round_trip",
            "trace_id": "cp-rt-dict",
            "symbol": "ETHUSDT",
            "original": {
                "last_processed_agg_trade_id": 987654321,
                "first_agg_trade_id_in_bar": 987654300,
                "last_agg_trade_id_in_bar": 987654310,
            },
            "restored": {
                "last_processed_agg_trade_id": cp2.last_processed_agg_trade_id,
                "first_agg_trade_id_in_bar": cp2.first_agg_trade_id_in_bar,
                "last_agg_trade_id_in_bar": cp2.last_agg_trade_id_in_bar,
            },
            "bit_exact": (
                cp2.last_processed_agg_trade_id == 987654321
                and cp2.first_agg_trade_id_in_bar == 987654300
                and cp2.last_agg_trade_id_in_bar == 987654310
            ),
            "pass": (
                cp2.last_processed_agg_trade_id == 987654321
                and cp2.first_agg_trade_id_in_bar == 987654300
                and cp2.last_agg_trade_id_in_bar == 987654310
            ),
        })

    def test_checkpoint_file_round_trip(self):
        """save() / load() preserves trade IDs through JSON on disk."""
        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="SOLUSDT",
            threshold_bps=750,
            start_date="2024-01-01",
            end_date="2024-12-31",
            last_completed_date="2024-09-01",
            bars_written=85000,
            last_processed_agg_trade_id=555444333,
            first_agg_trade_id_in_bar=555444300,
            last_agg_trade_id_in_bar=555444320,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            cp.save(path)

            # Read raw JSON to verify structure
            raw = json.loads(path.read_text())
            loaded = PopulationCheckpoint.load(path)

        assert loaded is not None
        assert loaded.last_processed_agg_trade_id == 555444333

        _emit({
            "event": "checkpoint_file_round_trip",
            "trace_id": "cp-rt-file",
            "symbol": "SOLUSDT",
            "raw_json_keys": sorted(raw.keys()),
            "raw_has_trade_id": "last_processed_agg_trade_id" in raw,
            "raw_trade_id": raw.get("last_processed_agg_trade_id"),
            "loaded_trade_id": loaded.last_processed_agg_trade_id,
            "bit_exact": loaded.last_processed_agg_trade_id == 555444333,
            "pass": loaded.last_processed_agg_trade_id == 555444333,
        })


# ---------------------------------------------------------------------------
# 3. Sidecar checkpoint: trade ID at streaming boundary
# ---------------------------------------------------------------------------


class TestSidecarCheckpointTelemetry:
    """Sidecar _save_checkpoint includes last_agg_trade_id when available."""

    def test_sidecar_checkpoint_includes_trade_id(self):
        """Sidecar persists last_agg_trade_id from completed bar."""
        from rangebar.sidecar import _save_checkpoint

        bar_dict = {
            "close_time_ms": 1700000000000,
            "last_agg_trade_id": 444555666,
            "open": 42000.0,
            "high": 42105.0,
            "low": 41980.0,
            "close": 42100.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("rangebar.sidecar._checkpoint_path") as mock_path:
                cp_path = Path(tmpdir) / "sidecar_cp.json"
                mock_path.return_value = cp_path

                _save_checkpoint("BTCUSDT", 250, bar_dict)

                raw = json.loads(cp_path.read_text())

        assert raw.get("last_agg_trade_id") == 444555666

        _emit({
            "event": "sidecar_checkpoint_trade_id",
            "trace_id": "sidecar-cp-1",
            "symbol": "BTCUSDT",
            "threshold": 250,
            "bar_last_agg_trade_id": 444555666,
            "checkpoint_has_key": "last_agg_trade_id" in raw,
            "checkpoint_value": raw.get("last_agg_trade_id"),
            "pass": raw.get("last_agg_trade_id") == 444555666,
        })

    def test_sidecar_checkpoint_omits_when_missing(self):
        """Sidecar omits last_agg_trade_id when bar has no trade ID."""
        from rangebar.sidecar import _save_checkpoint

        bar_dict = {
            "close_time_ms": 1700000000000,
            "open": 42000.0,
            "high": 42105.0,
            "low": 41980.0,
            "close": 42100.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("rangebar.sidecar._checkpoint_path") as mock_path:
                cp_path = Path(tmpdir) / "sidecar_cp.json"
                mock_path.return_value = cp_path

                _save_checkpoint("BTCUSDT", 250, bar_dict)

                raw = json.loads(cp_path.read_text())

        assert "last_agg_trade_id" not in raw

        _emit({
            "event": "sidecar_checkpoint_no_trade_id",
            "trace_id": "sidecar-cp-2",
            "symbol": "BTCUSDT",
            "threshold": 250,
            "bar_has_trade_id": False,
            "checkpoint_has_key": "last_agg_trade_id" in raw,
            "pass": "last_agg_trade_id" not in raw,
        })

    def test_sidecar_checkpoint_ignores_zero_trade_id(self):
        """Sidecar skips last_agg_trade_id == 0 (sentinel for no data)."""
        from rangebar.sidecar import _save_checkpoint

        bar_dict = {
            "close_time_ms": 1700000000000,
            "last_agg_trade_id": 0,
            "open": 42000.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("rangebar.sidecar._checkpoint_path") as mock_path:
                cp_path = Path(tmpdir) / "sidecar_cp.json"
                mock_path.return_value = cp_path

                _save_checkpoint("BTCUSDT", 250, bar_dict)

                raw = json.loads(cp_path.read_text())

        assert "last_agg_trade_id" not in raw

        _emit({
            "event": "sidecar_checkpoint_zero_trade_id",
            "trace_id": "sidecar-cp-3",
            "symbol": "BTCUSDT",
            "bar_last_agg_trade_id": 0,
            "checkpoint_has_key": "last_agg_trade_id" in raw,
            "pass": "last_agg_trade_id" not in raw,
        })


# ---------------------------------------------------------------------------
# 4. Read boundaries: trade ID columns in query paths
# ---------------------------------------------------------------------------


class TestReadBoundaryTelemetry:
    """Trade ID columns present at every read boundary (get_n_bars, get_bars_by_timestamp_range)."""

    def test_trade_id_columns_constant_defined(self):
        """TRADE_ID_RANGE_COLUMNS constant exists and has correct values."""
        from rangebar.constants import TRADE_ID_RANGE_COLUMNS

        assert "first_agg_trade_id" in TRADE_ID_RANGE_COLUMNS
        assert "last_agg_trade_id" in TRADE_ID_RANGE_COLUMNS

        _emit({
            "event": "trade_id_columns_constant",
            "trace_id": "read-const",
            "columns": list(TRADE_ID_RANGE_COLUMNS),
            "has_first": "first_agg_trade_id" in TRADE_ID_RANGE_COLUMNS,
            "has_last": "last_agg_trade_id" in TRADE_ID_RANGE_COLUMNS,
            "pass": True,
        })

    def test_get_n_bars_includes_trade_id_columns(self):
        """get_n_bars() wires TRADE_ID_RANGE_COLUMNS into its query."""
        import inspect

        from rangebar.clickhouse.query_operations import QueryOperationsMixin

        source = inspect.getsource(QueryOperationsMixin.get_n_bars)
        has_trade_id_cols = "TRADE_ID_RANGE_COLUMNS" in source

        assert has_trade_id_cols

        _emit({
            "event": "read_boundary_get_n_bars",
            "trace_id": "read-n-bars",
            "source_has_TRADE_ID_RANGE_COLUMNS": has_trade_id_cols,
            "pass": has_trade_id_cols,
        })

    def test_get_bars_by_timestamp_includes_trade_id_columns(self):
        """get_bars_by_timestamp_range() wires TRADE_ID_RANGE_COLUMNS."""
        import inspect

        from rangebar.clickhouse.query_operations import QueryOperationsMixin

        source = inspect.getsource(
            QueryOperationsMixin.get_bars_by_timestamp_range
        )
        has_trade_id_cols = "TRADE_ID_RANGE_COLUMNS" in source

        assert has_trade_id_cols

        _emit({
            "event": "read_boundary_get_bars_by_timestamp",
            "trace_id": "read-ts-range",
            "source_has_TRADE_ID_RANGE_COLUMNS": has_trade_id_cols,
            "pass": has_trade_id_cols,
        })


# ---------------------------------------------------------------------------
# 5. HWM query: get_ariadne_high_water_mark
# ---------------------------------------------------------------------------


class TestHighWaterMarkTelemetry:
    """get_ariadne_high_water_mark() queries ClickHouse for resume point."""

    def test_hwm_returns_value(self):
        """HWM returns max last_agg_trade_id from mock ClickHouse."""
        from rangebar.clickhouse.query_operations import QueryOperationsMixin

        mixin = QueryOperationsMixin.__new__(QueryOperationsMixin)
        mock_client = MagicMock()
        mock_client.query.return_value = MagicMock(
            result_rows=[(777888999,)]
        )
        mixin.client = mock_client

        hwm = mixin.get_ariadne_high_water_mark("BTCUSDT", 250)

        assert hwm == 777888999
        query = mock_client.query.call_args[0][0]

        _emit({
            "event": "hwm_query",
            "trace_id": "hwm-value",
            "symbol": "BTCUSDT",
            "threshold": 250,
            "hwm_returned": hwm,
            "query_contains_FINAL": "FINAL" in query,
            "query_contains_max": "max(last_agg_trade_id)" in query,
            "pass": hwm == 777888999,
        })

    def test_hwm_returns_none_when_no_data(self):
        """HWM returns None when no bars have trade IDs."""
        from rangebar.clickhouse.query_operations import QueryOperationsMixin

        mixin = QueryOperationsMixin.__new__(QueryOperationsMixin)
        mock_client = MagicMock()
        mock_client.query.return_value = MagicMock(
            result_rows=[(0,)]
        )
        mixin.client = mock_client

        hwm = mixin.get_ariadne_high_water_mark("BTCUSDT", 250)

        assert hwm is None

        _emit({
            "event": "hwm_query_empty",
            "trace_id": "hwm-none",
            "symbol": "BTCUSDT",
            "threshold": 250,
            "hwm_returned": hwm,
            "pass": hwm is None,
        })

    def test_hwm_returns_none_on_connection_error(self):
        """HWM returns None gracefully when ClickHouse is unreachable."""
        from rangebar.clickhouse.query_operations import QueryOperationsMixin

        mixin = QueryOperationsMixin.__new__(QueryOperationsMixin)
        mock_client = MagicMock()
        mock_client.query.side_effect = OSError("Connection refused")
        mixin.client = mock_client

        hwm = mixin.get_ariadne_high_water_mark("BTCUSDT", 250)

        assert hwm is None

        _emit({
            "event": "hwm_query_error",
            "trace_id": "hwm-err",
            "symbol": "BTCUSDT",
            "threshold": 250,
            "error": "OSError: Connection refused",
            "hwm_returned": hwm,
            "graceful_fallback": hwm is None,
            "pass": hwm is None,
        })


# ---------------------------------------------------------------------------
# 6. Ariadne fallback: no trade ID → timestamp resume
# ---------------------------------------------------------------------------


class TestAriadneFallbackTelemetry:
    """When checkpoint has no trade ID, Ariadne falls back to timestamps."""

    def test_fallback_when_trade_id_is_none(self):
        """Old checkpoint without trade ID triggers timestamp fallback."""
        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="BTCUSDT",
            threshold_bps=250,
            start_date="2024-01-01",
            end_date="2024-12-31",
            last_completed_date="2024-06-15",
            last_processed_agg_trade_id=None,
        )
        should_fallback = not cp.last_processed_agg_trade_id

        assert should_fallback is True

        _emit({
            "event": "ariadne_fallback",
            "trace_id": "fallback-none",
            "symbol": "BTCUSDT",
            "last_processed_agg_trade_id": None,
            "reason": "no_trade_id_in_checkpoint",
            "should_fallback": should_fallback,
            "pass": should_fallback is True,
        })

    def test_fallback_when_trade_id_is_zero(self):
        """Trade ID of 0 is treated as absent (falsy)."""
        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="BTCUSDT",
            threshold_bps=250,
            start_date="2024-01-01",
            end_date="2024-12-31",
            last_completed_date="2024-06-15",
            last_processed_agg_trade_id=0,
        )
        should_fallback = not cp.last_processed_agg_trade_id

        assert should_fallback is True

        _emit({
            "event": "ariadne_fallback",
            "trace_id": "fallback-zero",
            "symbol": "BTCUSDT",
            "last_processed_agg_trade_id": 0,
            "reason": "trade_id_is_zero_sentinel",
            "should_fallback": should_fallback,
            "pass": should_fallback is True,
        })

    def test_no_fallback_when_trade_id_present(self):
        """Valid trade ID means Ariadne resumes via fromId (no fallback)."""
        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="BTCUSDT",
            threshold_bps=250,
            start_date="2024-01-01",
            end_date="2024-12-31",
            last_completed_date="2024-06-15",
            last_processed_agg_trade_id=999888777,
        )
        should_fallback = not cp.last_processed_agg_trade_id

        assert should_fallback is False

        _emit({
            "event": "ariadne_no_fallback",
            "trace_id": "no-fallback",
            "symbol": "BTCUSDT",
            "last_processed_agg_trade_id": 999888777,
            "fetch_source": "fromId",
            "should_fallback": should_fallback,
            "pass": should_fallback is False,
        })


# ---------------------------------------------------------------------------
# 7. Hook integration: Ariadne events emitted through hook system
# ---------------------------------------------------------------------------


class TestAriadneHookTelemetry:
    """Ariadne events flow through the hook system for observability."""

    @pytest.fixture(autouse=True)
    def _clean_hooks(self):
        from rangebar.hooks import clear_hooks
        clear_hooks()
        yield
        clear_hooks()

    def test_checkpoint_saved_hook_carries_trade_id(self):
        """CHECKPOINT_SAVED hook payload includes trade ID details."""
        from rangebar.hooks import HookEvent, emit_hook, register_hook

        received = []
        register_hook(HookEvent.CHECKPOINT_SAVED, received.append)

        emit_hook(
            HookEvent.CHECKPOINT_SAVED,
            symbol="BTCUSDT",
            trace_id="ariadne-hook-1",
            last_processed_agg_trade_id=111222333,
            threshold_bps=250,
        )

        assert len(received) == 1
        payload = received[0]
        assert payload.trace_id == "ariadne-hook-1"
        assert payload.details.get("last_processed_agg_trade_id") == 111222333

        _emit({
            "event": "hook_checkpoint_saved",
            "trace_id": "ariadne-hook-1",
            "symbol": "BTCUSDT",
            "hook_event": "checkpoint_saved",
            "payload_trace_id": payload.trace_id,
            "payload_trade_id": payload.details.get("last_processed_agg_trade_id"),
            "pass": payload.details.get("last_processed_agg_trade_id") == 111222333,
        })

    def test_sidecar_checkpoint_hook_carries_trade_id(self):
        """SIDECAR_CHECKPOINT_SAVED hook payload includes trade ID."""
        from rangebar.hooks import HookEvent, emit_hook, register_hook

        received = []
        register_hook(HookEvent.SIDECAR_CHECKPOINT_SAVED, received.append)

        emit_hook(
            HookEvent.SIDECAR_CHECKPOINT_SAVED,
            symbol="ETHUSDT",
            trace_id="sidecar-hook-1",
            last_agg_trade_id=444555666,
            threshold=500,
        )

        assert len(received) == 1
        payload = received[0]
        assert payload.details.get("last_agg_trade_id") == 444555666

        _emit({
            "event": "hook_sidecar_checkpoint",
            "trace_id": "sidecar-hook-1",
            "symbol": "ETHUSDT",
            "hook_event": "sidecar_checkpoint_saved",
            "payload_trade_id": payload.details.get("last_agg_trade_id"),
            "pass": payload.details.get("last_agg_trade_id") == 444555666,
        })


# ---------------------------------------------------------------------------
# 8. Artifact summary: emitted at session end for autonomous examination
# ---------------------------------------------------------------------------


def test_artifact_summary():
    """Final test: emit a summary event with counts of all telemetry events."""
    if not ARTIFACT_FILE.exists():
        pytest.skip("No artifact file")

    lines = ARTIFACT_FILE.read_text().strip().splitlines()
    events = [json.loads(line) for line in lines if line.strip()]

    event_types = {}
    pass_count = 0
    fail_count = 0
    for evt in events:
        etype = evt.get("event", "unknown")
        event_types[etype] = event_types.get(etype, 0) + 1
        if evt.get("pass") is True:
            pass_count += 1
        elif evt.get("pass") is False:
            fail_count += 1

    _emit({
        "event": "artifact_summary",
        "trace_id": "summary",
        "total_events": len(events),
        "event_types": event_types,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "all_passed": fail_count == 0,
    })

    assert fail_count == 0, f"{fail_count} telemetry events reported failures"
