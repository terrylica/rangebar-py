"""Tests for JSONL diagnostic tracing (Issue #48, Phase 5).

Validates:
- trace_id field on HookPayload
- 4 new HookEvent types (DOWNLOAD_START, DOWNLOAD_COMPLETE, TICK_FETCH_PROGRESS, MIGRATION_PROGRESS)
- generate_trace_id() function
- Hook emissions in bulk_operations.py (CACHE_WRITE_*)
- Hook emissions in migrations.py (MIGRATION_PROGRESS)
"""

from __future__ import annotations

import pytest
from rangebar.hooks import (
    HookEvent,
    HookPayload,
    clear_hooks,
    emit_hook,
    generate_trace_id,
    register_hook,
)


@pytest.fixture(autouse=True)
def _clean_hooks() -> None:
    """Clear all hooks before and after each test."""
    clear_hooks()
    yield
    clear_hooks()


class TestHookEventTypes:
    """Test 4 new HookEvent types added in Phase 5."""

    def test_download_start_event(self):
        """DOWNLOAD_START event should exist."""
        assert HookEvent.DOWNLOAD_START.value == "download_start"

    def test_download_complete_event(self):
        """DOWNLOAD_COMPLETE event should exist."""
        assert HookEvent.DOWNLOAD_COMPLETE.value == "download_complete"

    def test_tick_fetch_progress_event(self):
        """TICK_FETCH_PROGRESS event should exist."""
        assert HookEvent.TICK_FETCH_PROGRESS.value == "tick_fetch_progress"

    def test_migration_progress_event(self):
        """MIGRATION_PROGRESS event should exist."""
        assert HookEvent.MIGRATION_PROGRESS.value == "migration_progress"

    def test_total_event_count(self):
        """Should have at least 12 events (8 original + 4 new)."""
        assert len(HookEvent) >= 12


class TestTraceId:
    """Test trace_id field on HookPayload and generate_trace_id()."""

    def test_trace_id_field_exists(self):
        """HookPayload should have trace_id field."""
        payload = HookPayload(event=HookEvent.CACHE_WRITE_START, symbol="TEST")
        assert hasattr(payload, "trace_id")
        assert payload.trace_id is None  # Default is None

    def test_trace_id_can_be_set(self):
        """trace_id should be settable on construction."""
        payload = HookPayload(
            event=HookEvent.CACHE_WRITE_START,
            symbol="TEST",
            trace_id="abc12345",
        )
        assert payload.trace_id == "abc12345"

    def test_trace_id_in_to_dict(self):
        """trace_id should appear in to_dict() output."""
        payload = HookPayload(
            event=HookEvent.CACHE_WRITE_START,
            symbol="TEST",
            trace_id="abc12345",
        )
        d = payload.to_dict()
        assert "trace_id" in d
        assert d["trace_id"] == "abc12345"

    def test_generate_trace_id_format(self):
        """generate_trace_id() should return 8-char hex string."""
        tid = generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 8
        # Should be valid hex
        int(tid, 16)

    def test_generate_trace_id_unique(self):
        """Each call should produce a unique trace ID."""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100  # All unique


class TestEmitHookWithTraceId:
    """Test emit_hook passes trace_id to callbacks."""

    def test_emit_hook_with_trace_id(self):
        """emit_hook should pass trace_id through to HookPayload."""
        received = []

        def callback(payload) -> None:
            received.append(payload)

        register_hook(HookEvent.DOWNLOAD_START, callback)
        emit_hook(
            HookEvent.DOWNLOAD_START,
            symbol="BTCUSDT",
            trace_id="test1234",
        )

        assert len(received) == 1
        assert received[0].trace_id == "test1234"

    def test_emit_hook_without_trace_id(self):
        """emit_hook without trace_id should default to None."""
        received = []

        def callback(payload) -> None:
            received.append(payload)

        register_hook(HookEvent.CACHE_WRITE_START, callback)
        emit_hook(HookEvent.CACHE_WRITE_START, symbol="BTCUSDT")

        assert len(received) == 1
        assert received[0].trace_id is None


class TestHookEmissionBulkOperations:
    """Test that bulk_operations.py emits CACHE_WRITE hooks."""

    def test_cache_write_hooks_imported(self):
        """bulk_operations should import HookEvent and emit_hook."""
        import inspect

        from rangebar.clickhouse.bulk_operations import BulkStoreMixin

        source = inspect.getsource(BulkStoreMixin)
        assert "CACHE_WRITE_START" in source
        assert "CACHE_WRITE_COMPLETE" in source
        assert "CACHE_WRITE_FAILED" in source

    def test_store_bars_bulk_has_hooks(self):
        """store_bars_bulk should have CACHE_WRITE_START at entry."""
        import inspect

        from rangebar.clickhouse.bulk_operations import BulkStoreMixin

        source = inspect.getsource(BulkStoreMixin.store_bars_bulk)
        assert "emit_hook" in source
        assert "CACHE_WRITE_START" in source


class TestHookEmissionMigrations:
    """Test that migrations.py emits MIGRATION_PROGRESS hook."""

    def test_migration_progress_hook_present(self):
        """migrate_exchange_sessions should emit MIGRATION_PROGRESS."""
        import inspect

        from rangebar.clickhouse.migrations import migrate_exchange_sessions

        source = inspect.getsource(migrate_exchange_sessions)
        assert "MIGRATION_PROGRESS" in source
        assert "emit_hook" in source

    def test_migration_hook_has_step_tracking(self):
        """MIGRATION_PROGRESS hook should include step/total details."""
        import inspect

        from rangebar.clickhouse.migrations import migrate_exchange_sessions

        source = inspect.getsource(migrate_exchange_sessions)
        assert "step=" in source
        assert "total=" in source


class TestGenerateTraceIdExported:
    """Test generate_trace_id is properly exported."""

    def test_in_hooks_all(self):
        """generate_trace_id should be in hooks.__all__."""
        from rangebar import hooks

        assert "generate_trace_id" in hooks.__all__
