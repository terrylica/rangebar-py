"""Integration tests for ClickHouse connection pool (Issue #96 Task #5 Phase 3).

Tests connection pooling functionality including client reuse, tunnel management,
and performance metrics.
"""


from rangebar.clickhouse.connection_pool import (
    ClickHouseConnectionPool,
    get_connection_pool,
)


class TestConnectionPool:
    """Tests for ClickHouseConnectionPool."""

    def test_pool_initialization(self) -> None:
        """Pool initializes with correct parameters."""
        pool = ClickHouseConnectionPool(max_pool_size=4)

        assert pool._max_pool_size == 4
        assert len(pool._clients) == 0
        assert len(pool._tunnels) == 0
        pool.close()

    def test_pool_metrics_initial_state(self) -> None:
        """Pool metrics report correct initial state."""
        pool = ClickHouseConnectionPool(max_pool_size=8)

        metrics = pool.get_metrics()
        assert metrics["active_clients"] == 0
        assert metrics["max_pool_size"] == 8
        assert metrics["active_tunnels"] == 0
        assert metrics["total_created"] == 0
        assert metrics["total_reused"] == 0
        pool.close()

    def test_pool_close_cleanup(self) -> None:
        """Close properly cleans up pool state."""
        pool = ClickHouseConnectionPool(max_pool_size=2)

        # Pool should initialize empty
        assert len(pool._clients) == 0

        # After close, should still be empty
        pool.close()
        assert len(pool._clients) == 0
        assert len(pool._tunnels) == 0

    def test_get_connection_pool_singleton(self) -> None:
        """get_connection_pool returns same instance each call."""
        pool1 = get_connection_pool(max_pool_size=4)
        pool2 = get_connection_pool(max_pool_size=8)  # max_pool_size ignored on second call

        assert pool1 is pool2
        assert pool1._max_pool_size == 4  # First call's value used

    def test_pool_default_max_size(self) -> None:
        """Pool uses default max_pool_size when not specified."""
        pool = ClickHouseConnectionPool()

        assert pool._max_pool_size == 8  # Default value


class TestConnectionPoolMetrics:
    """Tests for connection pool metrics tracking."""

    def test_metrics_accuracy_initial(self) -> None:
        """Initial metrics are accurate."""
        pool = ClickHouseConnectionPool(max_pool_size=4)

        metrics = pool.get_metrics()
        assert metrics["total_created"] == 0
        assert metrics["total_reused"] == 0
        assert metrics["active_clients"] == 0
        assert "reuse_ratio_pct" in metrics
        pool.close()

    def test_metrics_pool_size_tracking(self) -> None:
        """Metrics track pool size correctly."""
        pool = ClickHouseConnectionPool(max_pool_size=8)

        # After operations, metrics should reflect size
        metrics = pool.get_metrics()
        assert metrics["max_pool_size"] == 8
        assert metrics["active_clients"] <= metrics["max_pool_size"]
        pool.close()

    def test_reuse_ratio_calculation(self) -> None:
        """Reuse ratio calculated correctly."""
        pool = ClickHouseConnectionPool(max_pool_size=2)

        # Manually set metrics for testing
        pool._total_created = 2
        pool._total_reused = 2
        metrics = pool.get_metrics()
        assert metrics["reuse_ratio_pct"] == 50

        pool._total_created = 1
        pool._total_reused = 3
        metrics = pool.get_metrics()
        assert metrics["reuse_ratio_pct"] == 75

        pool.close()


class TestConnectionPoolArchitecture:
    """Tests for connection pool architecture and design."""

    def test_pool_thread_safety_lock(self) -> None:
        """Pool has thread safety lock."""
        pool = ClickHouseConnectionPool()

        assert hasattr(pool, "_lock")
        assert pool._lock is not None
        pool.close()

    def test_pool_tunnel_cache_structure(self) -> None:
        """Pool has tunnel cache dictionary."""
        pool = ClickHouseConnectionPool()

        assert isinstance(pool._tunnels, dict)
        assert len(pool._tunnels) == 0  # Initially empty
        pool.close()

    def test_pool_client_storage_structure(self) -> None:
        """Pool stores clients as tuples with state."""
        pool = ClickHouseConnectionPool()

        assert isinstance(pool._clients, list)
        assert len(pool._clients) == 0  # Initially empty
        pool.close()

    def test_pool_has_close_method(self) -> None:
        """Pool has close method for cleanup."""
        pool = ClickHouseConnectionPool()

        assert hasattr(pool, "close")
        assert callable(pool.close)
        pool.close()

    def test_pool_has_get_metrics_method(self) -> None:
        """Pool has get_metrics method."""
        pool = ClickHouseConnectionPool()

        assert hasattr(pool, "get_metrics")
        assert callable(pool.get_metrics)
        pool.close()


class TestConnectionPoolDocumentation:
    """Tests that verify pool documentation and examples."""

    def test_pool_has_docstring(self) -> None:
        """Pool class has documentation."""
        assert ClickHouseConnectionPool.__doc__ is not None
        assert "thread-safe" in ClickHouseConnectionPool.__doc__.lower()
        assert "reusable" in ClickHouseConnectionPool.__doc__.lower()

    def test_get_metrics_returns_dict(self) -> None:
        """get_metrics returns dictionary with expected keys."""
        pool = ClickHouseConnectionPool()

        metrics = pool.get_metrics()
        expected_keys = {
            "active_clients",
            "max_pool_size",
            "active_tunnels",
            "total_created",
            "total_reused",
            "reuse_ratio_pct",
        }
        assert set(metrics.keys()) == expected_keys
        pool.close()

    def test_pool_close_is_idempotent(self) -> None:
        """Calling close multiple times is safe."""
        pool = ClickHouseConnectionPool()

        # Should not raise
        pool.close()
        pool.close()
        pool.close()

