# FILE-SIZE-OK: Single test file covering all watchdog scenarios, splitting would fragment test context
"""Tests for sidecar watchdog logic (Issue #107).

Verifies that the P1 liveness watchdog detects stale trade flow,
restarts the Rust engine, and exits after max restarts.

Issue #117-119: Tests updated to mock RangeBarCache (cached at startup)
and _start_health_server, and use _MonotonicClock helper to handle the
variable number of time.monotonic() calls across startup, notification,
and main loop paths.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

from rangebar.sidecar import (
    SidecarConfig,
    _notify_watchdog_trigger,
    run_sidecar,
)

# =============================================================================
# Helpers
# =============================================================================


class _MonotonicClock:
    """Controllable time.monotonic() replacement for sidecar tests.

    Returns `value` on every call.  Tests advance time by setting `.value`.
    This avoids fragile iterator-based mocking that breaks whenever code
    paths add or remove time.monotonic() calls.
    """

    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def __call__(self) -> float:
        return self.value


def _make_mock_engine(
    trades_sequence: list[int],
    bars: list[dict | None] | None = None,
):
    """Create a mock LiveBarEngine with controlled metrics and bars.

    Parameters
    ----------
    trades_sequence : list[int]
        Sequence of trades_received values returned by successive get_metrics() calls.
    bars : list[dict | None] | None
        Sequence of values returned by next_bar(). None = timeout.
    """
    engine = MagicMock()

    if bars is None:
        bars = [None] * len(trades_sequence)

    engine.next_bar.side_effect = bars
    engine.get_metrics.side_effect = [
        {"trades_received": t} for t in trades_sequence
    ]
    engine.collect_checkpoints.return_value = {}
    return engine


# Common patches for all tests (suppress side effects)
_COMMON_PATCHES = {
    "rangebar.hooks.emit_hook": MagicMock(),
    "rangebar.logging.log_checkpoint_event": MagicMock(),
    "rangebar.logging.generate_trace_id": MagicMock(return_value="test-trace"),
}


def _sidecar_patches(**overrides: object) -> dict[str, object]:
    """Return a dict of all common patches needed for run_sidecar() tests.

    Issue #117-119 additions:
    - rangebar.clickhouse.cache.RangeBarCache: mock constructor (no SSH tunnel)
    - rangebar.sidecar._start_health_server: suppress HTTP server
    """
    defaults = {
        "rangebar.sidecar._gap_fill": MagicMock(return_value={}),
        "rangebar.sidecar._extract_checkpoints": MagicMock(),
        "rangebar.logging.generate_trace_id": MagicMock(return_value="t"),
        "rangebar.hooks.emit_hook": MagicMock(),
        "rangebar.logging.log_checkpoint_event": MagicMock(),
        "signal.signal": MagicMock(),
        "rangebar.clickhouse.cache.RangeBarCache": MagicMock(),
        "rangebar.sidecar._start_health_server": MagicMock(return_value=None),
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# SidecarConfig env parsing
# =============================================================================


class TestSidecarConfigWatchdog:
    def test_from_env_reads_watchdog_timeout(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_STREAMING_SYMBOLS", "BTCUSDT")
        monkeypatch.setenv("RANGEBAR_STREAMING_WATCHDOG_TIMEOUT_S", "120")
        config = SidecarConfig.from_env()
        assert config.watchdog_timeout_s == 120

    def test_from_env_default_watchdog_timeout(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_STREAMING_SYMBOLS", "BTCUSDT")
        monkeypatch.delenv("RANGEBAR_STREAMING_WATCHDOG_TIMEOUT_S", raising=False)
        config = SidecarConfig.from_env()
        assert config.watchdog_timeout_s == 300

    def test_from_env_reads_max_restarts(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_STREAMING_SYMBOLS", "BTCUSDT")
        monkeypatch.setenv("RANGEBAR_STREAMING_MAX_WATCHDOG_RESTARTS", "5")
        config = SidecarConfig.from_env()
        assert config.max_watchdog_restarts == 5

    def test_from_env_default_max_restarts(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_STREAMING_SYMBOLS", "BTCUSDT")
        monkeypatch.delenv("RANGEBAR_STREAMING_MAX_WATCHDOG_RESTARTS", raising=False)
        config = SidecarConfig.from_env()
        assert config.max_watchdog_restarts == 3


# =============================================================================
# Watchdog trigger behavior
# =============================================================================


class TestWatchdogTrigger:
    """Test watchdog detection and engine restart logic.

    Uses _MonotonicClock to control time.monotonic() without fragile iterators.
    The clock starts at 0.0; tests advance .value to simulate elapsed time.
    """

    def test_watchdog_triggers_after_timeout(self):
        """trades_received stays at 0 for >300s → engine restart."""
        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        call_count = [0]

        def next_bar_side_effect(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] == 1:
                # First iteration: not yet timed out
                return
            if call_count[0] == 2:
                # Second iteration: advance time past watchdog timeout
                clock.value = 301.0
                return
            # After restart, exhaust the loop
            raise StopIteration

        engine.next_bar.side_effect = next_bar_side_effect
        engine.get_metrics.return_value = {"trades_received": 0}
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"],
            thresholds=[250],
            watchdog_timeout_s=300,
            max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        engines = [engine, MagicMock()]
        engines[1].next_bar.side_effect = StopIteration
        engines[1].get_metrics.return_value = {"trades_received": 0}
        engines[1].collect_checkpoints.return_value = {}

        patches = _sidecar_patches(
            **{"rangebar.sidecar._notify_watchdog_trigger": MagicMock()},
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", patches["rangebar.sidecar._gap_fill"]),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            mock_create.side_effect = [
                (engine, 0),
                (engines[1], 0),
            ]
            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            assert mock_create.call_count == 2
            mock_notify.assert_called_once()

    def test_watchdog_resets_on_trade_increment(self):
        """trades_received goes 0→50→50→100 — no restart because trades resume."""
        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        call_count = [0]
        metrics_seq = [
            {"trades_received": 50},
            {"trades_received": 50},
            {"trades_received": 100},
        ]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            # Advance time by 150s each iteration (under 300s threshold)
            clock.value = call_count[0] * 150.0
            if call_count[0] > 3:
                raise StopIteration

        engine.next_bar.side_effect = next_bar_effect
        engine.get_metrics.side_effect = metrics_seq
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            mock_notify.assert_not_called()

    def test_watchdog_resets_on_bar_received(self):
        """next_bar() returns a bar dict → watchdog timer resets."""
        bar = {
            "_symbol": "BTCUSDT",
            "_threshold": 250,
            "close": 50000.0,
            "close_time_ms": 1700000000000,
        }

        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        call_count = [0]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] == 1:
                # Bar received at 400s — but trades are flowing so no trigger
                clock.value = 400.0
                return bar.copy()
            raise StopIteration

        engine.next_bar.side_effect = next_bar_effect
        engine.get_metrics.return_value = {"trades_received": 100}
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._write_bars_batch_to_clickhouse") as mock_write,
            patch("rangebar.sidecar._save_checkpoint"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            mock_notify.assert_not_called()
            assert mock_write.call_count >= 1

    def test_watchdog_max_restarts_exit(self):
        """3 consecutive watchdog triggers → loop exits."""
        clock = _MonotonicClock(0.0)
        restart_count = [0]

        engines = []
        for _ in range(4):  # initial + 3 restarts
            e = MagicMock()
            e.collect_checkpoints.return_value = {}
            e.get_metrics.return_value = {"trades_received": 0}

            # Each engine: first next_bar returns None, second triggers timeout
            engine_call = [0]

            def make_next_bar(ec=engine_call):
                ec_ref = [0]
                def _next_bar(timeout_ms=5000):
                    ec_ref[0] += 1
                    if ec_ref[0] == 1:
                        return  # first call: no bar
                    # second call: advance clock past timeout
                    restart_count[0] += 1
                    clock.value = restart_count[0] * 301.0
                    return  # triggers watchdog
                return _next_bar

            e.next_bar.side_effect = make_next_bar()
            engines.append(e)

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            mock_create.side_effect = [(e, 0) for e in engines]
            run_sidecar(config)

            assert mock_create.call_count == 4
            assert mock_notify.call_count == 4

    def test_watchdog_does_not_fire_during_normal_gaps(self):
        """Trades flowing but no bars for 10 min — no restart."""
        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        call_count = [0]
        metrics_seq = [
            {"trades_received": 200},
            {"trades_received": 300},
        ]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            # Advance 300s each time — but trades increment so no trigger
            clock.value = call_count[0] * 300.0
            if call_count[0] > 2:
                raise StopIteration

        engine.next_bar.side_effect = next_bar_effect
        engine.get_metrics.side_effect = metrics_seq
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            mock_notify.assert_not_called()

    def test_watchdog_timeout_s_zero_disables(self):
        """watchdog_timeout_s=0 disables the watchdog entirely."""
        clock = _MonotonicClock(9999.0)

        engine = MagicMock()
        call_count = [0]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] > 3:
                raise StopIteration

        engine.next_bar.side_effect = next_bar_effect
        engine.get_metrics.return_value = {"trades_received": 0}
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=0,  # disabled
            max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            assert engine.get_metrics.call_count == 1  # finally block only
            mock_notify.assert_not_called()


# =============================================================================
# Engine restart sequence
# =============================================================================


class TestEngineRestartSequence:
    def test_restart_extracts_checkpoints_before_shutdown(self):
        """Watchdog triggers → _extract_checkpoints() called before engine.shutdown()."""
        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        engine_call_count = [0]

        def engine_next_bar(timeout_ms=5000):
            engine_call_count[0] += 1
            if engine_call_count[0] >= 2:
                clock.value = 301.0  # trigger watchdog

        engine.next_bar.side_effect = engine_next_bar
        engine.get_metrics.return_value = {"trades_received": 0}
        engine.collect_checkpoints.return_value = {}

        engine2 = MagicMock()
        engine2.next_bar.side_effect = StopIteration
        engine2.get_metrics.return_value = {"trades_received": 0}
        engine2.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        call_order = []

        engine.stop.side_effect = lambda: call_order.append("stop")
        engine.shutdown.side_effect = lambda: call_order.append("shutdown")

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints") as mock_extract,
            patch("rangebar.sidecar._notify_watchdog_trigger"),
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            mock_create.side_effect = [(engine, 0), (engine2, 0)]
            mock_extract.side_effect = lambda *_a, **_kw: call_order.append("extract")

            with contextlib.suppress(StopIteration):
                run_sidecar(config)

            # Verify order: stop → extract → shutdown (for the watchdog restart)
            assert call_order[:3] == ["stop", "extract", "shutdown"]

    def test_restart_preserves_bars_written_count(self):
        """bars_written stays accumulated across watchdog restarts."""
        bar = {
            "_symbol": "BTCUSDT",
            "_threshold": 250,
            "close": 50000.0,
        }

        clock = _MonotonicClock(0.0)

        engine1 = MagicMock()
        call_count = [0]

        def next_bar_1(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] <= 2:
                return bar.copy()
            # After bars, advance time to trigger watchdog
            clock.value = 301.0
            return None

        engine1.next_bar.side_effect = next_bar_1
        engine1.get_metrics.return_value = {"trades_received": 100}
        engine1.collect_checkpoints.return_value = {}

        engine2 = MagicMock()
        engine2_call = [0]

        def next_bar_2(timeout_ms=5000):
            engine2_call[0] += 1
            if engine2_call[0] >= 2:
                # Advance clock past watchdog timeout for engine2
                # After restart, last_trade_increment_time = 301.0
                # so need 301.0 + 301.0 = 602.0
                clock.value = 602.0

        engine2.next_bar.side_effect = next_bar_2
        engine2.get_metrics.return_value = {"trades_received": 0}
        engine2.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=1,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._write_bars_batch_to_clickhouse"),
            patch("rangebar.sidecar._save_checkpoint"),
            patch("rangebar.sidecar._notify_watchdog_trigger"),
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            mock_create.side_effect = [(engine1, 0), (engine2, 0)]
            run_sidecar(config)


# =============================================================================
# Edge cases
# =============================================================================


class TestWatchdogEdgeCases:
    def test_metrics_returns_empty_dict(self):
        """get_metrics() returns {} → treated as 0 trades, no crash."""
        clock = _MonotonicClock(0.0)

        engine = MagicMock()
        call_count = [0]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            clock.value = call_count[0] * 100.0  # advance time modestly
            if call_count[0] > 1:
                raise StopIteration

        engine.next_bar.side_effect = next_bar_effect
        engine.get_metrics.return_value = {}  # empty dict
        engine.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=clock),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger"),
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
            patch("rangebar.clickhouse.cache.RangeBarCache"),
            patch("rangebar.sidecar._start_health_server", return_value=None),
        ):
            with contextlib.suppress(StopIteration):
                run_sidecar(config)
            # No crash — test passes if we get here


# =============================================================================
# Telegram notification
# =============================================================================


class TestWatchdogNotification:
    @patch("rangebar.notify.telegram.send_telegram")
    def test_notify_sends_telegram_on_watchdog(self, mock_send):
        mock_send.return_value = True
        _notify_watchdog_trigger(300.0, 50, 0)
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "WATCHDOG" in msg
        assert "300" in msg

    @patch("rangebar.notify.telegram.send_telegram")
    def test_notify_failure_does_not_crash_watchdog(self, mock_send):
        mock_send.side_effect = Exception("network error")
        # Should not raise
        _notify_watchdog_trigger(300.0, 50, 0)

    def test_notify_disabled_when_import_fails(self):
        with patch.dict("sys.modules", {"rangebar.notify.telegram": None}):
            # Should not raise even when import fails
            _notify_watchdog_trigger(300.0, 50, 0)
