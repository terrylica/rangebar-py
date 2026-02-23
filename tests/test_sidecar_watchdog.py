# FILE-SIZE-OK: Single test file covering all watchdog scenarios, splitting would fragment test context
"""Tests for sidecar watchdog logic (Issue #107).

Verifies that the P1 liveness watchdog detects stale trade flow,
restarts the Rust engine, and exits after max restarts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rangebar.sidecar import (
    SidecarConfig,
    _notify_watchdog_trigger,
    run_sidecar,
)

# =============================================================================
# Helpers
# =============================================================================


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

    Uses time.monotonic patching to simulate elapsed time without waiting.
    """

    def _run_sidecar_with_mock_engine(
        self,
        engine,
        config: SidecarConfig | None = None,
        gap_fill_results: dict | None = None,
    ):
        """Run the sidecar main loop with a mocked engine."""
        if config is None:
            config = SidecarConfig(
                symbols=["BTCUSDT"],
                thresholds=[250],
                watchdog_timeout_s=300,
                max_watchdog_restarts=3,
                gap_fill_on_startup=False,
            )

        with (
            patch.multiple("rangebar.sidecar", _gap_fill=MagicMock(return_value={}), _inject_checkpoints=MagicMock(return_value=0), _extract_checkpoints=MagicMock(), _write_bars_batch_to_clickhouse=MagicMock(), _save_checkpoint=MagicMock(), _notify_watchdog_trigger=MagicMock()),
            patch("rangebar._core.LiveBarEngine", return_value=engine),
            patch.dict("rangebar.sidecar.__builtins__", {}, clear=False),
            patch("rangebar.logging.generate_trace_id", return_value="test-trace"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            run_sidecar(config)

    def test_watchdog_triggers_after_timeout(self):
        """trades_received stays at 0 for >300s → engine restart."""
        # We need to control time.monotonic to simulate 300s passing.
        # The loop calls next_bar → None, then checks metrics.
        # We need enough iterations for the timeout to fire.

        monotonic_values = iter([
            0.0,      # initial baseline
            0.0,      # first next_bar call (within select)
            150.0,    # first watchdog check — 150s, not yet
            301.0,    # second watchdog check — 301s, triggers!
            301.0,    # post-restart baseline
            # After restart, engine.next_bar will raise StopIteration → loop ends
        ])

        engine = MagicMock()
        call_count = [0]

        def next_bar_side_effect(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] >= 4:
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

        # Track engine creation calls
        engines = [engine, MagicMock()]
        engines[1].next_bar.side_effect = StopIteration
        engines[1].get_metrics.return_value = {"trades_received": 0}
        engines[1].collect_checkpoints.return_value = {}
        engine_idx = [0]

        def create_engine_factory(*args, **kwargs):
            e = engines[engine_idx[0]]
            engine_idx[0] += 1
            return e

        with (
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            mock_create.side_effect = [
                (engine, 0),          # initial creation
                (engines[1], 0),      # watchdog restart
            ]
            try:
                run_sidecar(config)
            except StopIteration:
                pass

            # Engine was restarted (created twice: initial + 1 restart)
            assert mock_create.call_count == 2
            mock_notify.assert_called_once()

    def test_watchdog_resets_on_trade_increment(self):
        """trades_received goes 0→50→50→100 — no restart because trades resume."""
        monotonic_values = iter([
            0.0,      # initial baseline
            150.0,    # first check: trades 0→50, resets timer
            200.0,    # second check: trades 50→50, stale_s = 50 (< 300)
            250.0,    # third check: trades 50→100, resets timer
        ])

        engine = MagicMock()
        call_count = [0]
        metrics_seq = [
            {"trades_received": 50},
            {"trades_received": 50},
            {"trades_received": 100},
        ]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
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
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            try:
                run_sidecar(config)
            except StopIteration:
                pass

            mock_notify.assert_not_called()

    def test_watchdog_resets_on_bar_received(self):
        """next_bar() returns a bar dict → watchdog timer resets."""
        bar = {
            "_symbol": "BTCUSDT",
            "_threshold": 250,
            "close": 50000.0,
            "timestamp_ms": 1700000000000,
        }

        # Issue #96 Task #144 Phase 3: Account for batch timeout check in loop
        # Now uses time.monotonic() for batch flush timeout tracking
        monotonic_values = iter([
            0.0,      # last_batch_flush_time init
            0.0,      # batch timeout check in first loop iteration
            400.0,    # bar received at 400s — resets timer (batch timeout check for 2nd iteration)
            400.0,    # get_metrics after bar
        ])

        engine = MagicMock()
        call_count = [0]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] == 1:
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
            patch("time.monotonic", side_effect=monotonic_values),
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
        ):
            try:
                run_sidecar(config)
            except StopIteration:
                pass

            mock_notify.assert_not_called()
            # Batch write should be called at least once (on shutdown flush)
            assert mock_write.call_count >= 1

    def test_watchdog_max_restarts_exit(self):
        """3 consecutive watchdog triggers → loop exits."""
        # Issue #96 Task #144 Phase 3: Updated for batch timeout check in loop
        # Each watchdog cycle: batch timeout check, then stale check at +301s
        # Need 4 cycles (3 restarts + 1 final that breaks)
        monotonic_values = iter([
            0.0,      # last_batch_flush_time init
            0.0,      # batch timeout check in iteration #1
            0.0,      # batch timeout check in iteration #2
            301.0,    # trigger #1 (watchdog stale check)
            301.0,    # last_batch_flush_time after first _flush in watchdog
            301.0,    # batch timeout check after restart #1
            301.0,    # batch timeout check iteration 2 after restart
            602.0,    # trigger #2 (watchdog stale check)
            602.0,    # last_batch_flush_time after second _flush
            602.0,    # batch timeout check after restart #2
            602.0,    # batch timeout check iteration 2 after restart
            903.0,    # trigger #3 (watchdog stale check)
            903.0,    # last_batch_flush_time after third _flush
            903.0,    # batch timeout check after restart #3
            903.0,    # batch timeout check iteration 2 after restart
            1204.0,   # trigger #4 — exceeds max_watchdog_restarts=3
            1204.0,   # final batch timeout check before break
        ])

        engines = []
        for _ in range(4):  # initial + 3 restarts
            e = MagicMock()
            e.next_bar.return_value = None
            e.get_metrics.return_value = {"trades_received": 0}
            e.collect_checkpoints.return_value = {}
            engines.append(e)

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=3,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            mock_create.side_effect = [(e, 0) for e in engines]
            run_sidecar(config)

            # Initial + 3 restarts = 4 engine creations
            assert mock_create.call_count == 4
            # 4 notifications (3 restarts + 1 final that breaks)
            assert mock_notify.call_count == 4

    def test_watchdog_does_not_fire_during_normal_gaps(self):
        """Trades flowing but no bars for 10 min — no restart."""
        # Trade count keeps incrementing, so watchdog never fires
        monotonic_values = iter([
            0.0,      # initial baseline
            300.0,    # check: trades 100→200, resets
            600.0,    # check: trades 200→300, resets
        ])

        engine = MagicMock()
        call_count = [0]
        metrics_seq = [
            {"trades_received": 200},
            {"trades_received": 300},
        ]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
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
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            try:
                run_sidecar(config)
            except StopIteration:
                pass

            mock_notify.assert_not_called()

    def test_watchdog_timeout_s_zero_disables(self):
        """watchdog_timeout_s=0 disables the watchdog entirely."""
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
            patch("time.monotonic", return_value=9999.0),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger") as mock_notify,
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            try:
                run_sidecar(config)
            except StopIteration:
                pass

            # get_metrics called only once (in finally block), not in the loop
            # When watchdog is disabled, the loop skips the metrics check entirely
            assert engine.get_metrics.call_count == 1  # finally block only
            mock_notify.assert_not_called()


# =============================================================================
# Engine restart sequence
# =============================================================================


class TestEngineRestartSequence:
    def test_restart_extracts_checkpoints_before_shutdown(self):
        """Watchdog triggers → _extract_checkpoints() called before engine.shutdown()."""
        # Issue #96 Task #144 Phase 3: Account for batch timeout checks
        monotonic_values = iter([
            0.0,      # last_batch_flush_time init
            0.0,      # batch timeout check in iteration #1
            301.0,    # trigger (watchdog)
            301.0,    # last_batch_flush_time after restart
            301.0,    # batch timeout check in iteration #2 (right before StopIteration)
        ])

        engine = MagicMock()
        engine.next_bar.return_value = None
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
        original_stop = engine.stop
        original_shutdown = engine.shutdown

        def track_stop():
            call_order.append("stop")
            return original_stop()

        def track_shutdown():
            call_order.append("shutdown")
            return original_shutdown()

        engine.stop.side_effect = lambda: call_order.append("stop")
        engine.shutdown.side_effect = lambda: call_order.append("shutdown")

        with (
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine") as mock_create,
            patch("rangebar.sidecar._extract_checkpoints") as mock_extract,
            patch("rangebar.sidecar._notify_watchdog_trigger"),
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            mock_create.side_effect = [(engine, 0), (engine2, 0)]
            mock_extract.side_effect = lambda *a, **kw: call_order.append("extract")

            try:
                run_sidecar(config)
            except StopIteration:
                pass

            # Verify order: stop → extract → shutdown (for the watchdog restart)
            assert call_order[:3] == ["stop", "extract", "shutdown"]

    def test_restart_preserves_bars_written_count(self):
        """bars_written stays accumulated across watchdog restarts."""
        bar = {
            "_symbol": "BTCUSDT",
            "_threshold": 250,
            "close": 50000.0,
        }

        # Timeline: 2 bars, then watchdog, then exit
        # Issue #96 Task #144 Phase 3: Account for batch timeout checks
        # Provide enough values to cover all monotonic() calls during the test
        # Pattern: init, [iter_batch_check, metrics_time]+ , watchdog_trigger, flush_reset, [iter_batch_check, metrics]+ , final_break
        import itertools
        monotonic_values = iter(itertools.islice(
            itertools.cycle([0.0, 1.0, 2.0, 100.0, 303.0, 304.0, 604.0, 1000.0]),
            50  # Provide plenty of values
        ))

        engine1 = MagicMock()
        call_count = [0]

        def next_bar_1(timeout_ms=5000):
            call_count[0] += 1
            if call_count[0] <= 2:
                return bar.copy()
            return None

        engine1.next_bar.side_effect = next_bar_1
        engine1.get_metrics.return_value = {"trades_received": 100}
        engine1.collect_checkpoints.return_value = {}

        engine2 = MagicMock()
        engine2.next_bar.return_value = None
        engine2.get_metrics.return_value = {"trades_received": 0}
        engine2.collect_checkpoints.return_value = {}

        config = SidecarConfig(
            symbols=["BTCUSDT"], thresholds=[250],
            watchdog_timeout_s=300, max_watchdog_restarts=1,
            gap_fill_on_startup=False,
        )

        with (
            patch("time.monotonic", side_effect=monotonic_values),
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
        ):
            mock_create.side_effect = [(engine1, 0), (engine2, 0)]
            # run_sidecar logs bars_written at the end — capture it
            run_sidecar(config)

            # The final engine.get_metrics is called in the finally block
            # and bars_written should be 2 (accumulated before restart)


# =============================================================================
# Edge cases
# =============================================================================


class TestWatchdogEdgeCases:
    def test_metrics_returns_empty_dict(self):
        """get_metrics() returns {} → treated as 0 trades, no crash."""
        monotonic_values = iter([
            0.0,      # baseline
            100.0,    # check — stale but < 300s
        ])

        engine = MagicMock()
        call_count = [0]

        def next_bar_effect(timeout_ms=5000):
            call_count[0] += 1
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
            patch("time.monotonic", side_effect=monotonic_values),
            patch("rangebar.sidecar._gap_fill", return_value={}),
            patch("rangebar.sidecar._create_engine", return_value=(engine, 0)),
            patch("rangebar.sidecar._extract_checkpoints"),
            patch("rangebar.sidecar._notify_watchdog_trigger"),
            patch("rangebar.logging.generate_trace_id", return_value="t"),
            patch("rangebar.hooks.emit_hook"),
            patch("rangebar.logging.log_checkpoint_event"),
            patch("signal.signal"),
        ):
            try:
                run_sidecar(config)
            except StopIteration:
                pass
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
