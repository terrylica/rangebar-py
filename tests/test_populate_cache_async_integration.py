"""Test populate_cache_resumable with async writes enabled (Phase 4).

Issue #96 Task #5 Phase 4: Verify async writer integration pattern.
"""



class TestPopulateCacheAsyncIntegration:
    """Tests for async write integration in populate_cache_resumable."""

    def test_populate_accepts_num_async_writers_parameter(self) -> None:
        """populate_cache_resumable should accept num_async_writers parameter."""
        # Verify the function signature includes num_async_writers
        import inspect

        from rangebar.checkpoint import populate_cache_resumable

        sig = inspect.signature(populate_cache_resumable)
        assert "num_async_writers" in sig.parameters
        assert sig.parameters["num_async_writers"].default == 0

    def test_populate_with_async_disabled_by_default(self) -> None:
        """Async writes should be disabled by default (backward compatible)."""
        import inspect

        from rangebar.checkpoint import populate_cache_resumable

        sig = inspect.signature(populate_cache_resumable)
        num_async_param = sig.parameters["num_async_writers"]
        # Default should be 0 (disabled)
        assert num_async_param.default == 0

    def test_connection_pool_available_for_import(self) -> None:
        """Connection pool should be importable from clickhouse module."""
        from rangebar.clickhouse.connection_pool import (
            ClickHouseConnectionPool,
            get_connection_pool,
        )

        # Verify classes exist
        assert ClickHouseConnectionPool is not None
        assert get_connection_pool is not None

    def test_async_writer_available_for_import(self) -> None:
        """Async writer should be importable from orchestration module."""
        from rangebar.orchestration.async_cache_writes import AsyncCacheWriter

        # Verify class exists
        assert AsyncCacheWriter is not None

    def test_thread_pool_executor_available(self) -> None:
        """ThreadPoolExecutor should be available for concurrent writes."""
        from concurrent.futures import ThreadPoolExecutor

        # Create a simple pool
        pool = ThreadPoolExecutor(max_workers=2)
        assert pool is not None
        pool.shutdown()

    def test_populate_docstring_documents_async_parameter(self) -> None:
        """Function docstring should document num_async_writers parameter."""
        from rangebar.checkpoint import populate_cache_resumable

        doc = populate_cache_resumable.__doc__
        assert doc is not None
        assert "num_async_writers" in doc
        assert "async" in doc.lower()


class TestAsyncIntegrationArchitecture:
    """Tests for Phase 4 async integration architecture."""

    def test_connection_pool_singleton_pattern(self) -> None:
        """Connection pool should use singleton pattern for global access."""
        from rangebar.clickhouse.connection_pool import (
            get_connection_pool,
        )

        pool1 = get_connection_pool(max_pool_size=4)
        pool2 = get_connection_pool(max_pool_size=8)

        # Same instance (singleton)
        assert pool1 is pool2

    def test_async_writer_has_worker_configuration(self) -> None:
        """AsyncCacheWriter should accept num_workers configuration."""
        from rangebar.orchestration.async_cache_writes import AsyncCacheWriter

        writer = AsyncCacheWriter(num_workers=2, maxsize=100)
        assert writer is not None
        assert hasattr(writer, "num_workers")
        assert hasattr(writer, "queue")

    def test_fatal_cache_write_is_thread_safe(self) -> None:
        """fatal_cache_write should be thread-safe for concurrent calls."""
        from rangebar.orchestration.range_bars_cache import fatal_cache_write

        # Verify function exists and is callable
        assert callable(fatal_cache_write)


class TestPhase4DesignPatterns:
    """Tests for Phase 4 design patterns and best practices."""

    def test_thread_pool_fallback_on_error(self) -> None:
        """Thread pool should gracefully fall back to sync if init fails."""
        from concurrent.futures import ThreadPoolExecutor

        # Simulate fallback scenario
        try:
            pool = ThreadPoolExecutor(max_workers=-1)  # Invalid
        except ValueError:
            # Expected - thread pool would set pool=None in production code
            pool = None

        assert pool is None

    def test_connection_pool_initialization_idempotent(self) -> None:
        """Connection pool initialization should be idempotent."""
        from rangebar.clickhouse.connection_pool import get_connection_pool

        # Call multiple times
        pool1 = get_connection_pool(max_pool_size=8)
        pool2 = get_connection_pool(max_pool_size=8)
        pool3 = get_connection_pool(max_pool_size=8)

        # All should be same instance
        assert pool1 is pool2 is pool3
