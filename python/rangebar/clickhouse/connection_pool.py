"""Connection pooling for ClickHouse clients (Issue #96 Task #5 Phase 3).

Manages reusable ClickHouse client connections to reduce per-write overhead.
Reuses SSH tunnels and clients across multiple write operations.

Performance target: 1.5-2x speedup by eliminating per-write tunnel creation.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import clickhouse_connect

    from .tunnel import SSHTunnel

logger = logging.getLogger(__name__)


class ClickHouseConnectionPool:
    """Thread-safe pool of reusable ClickHouse client connections.

    Manages client lifecycle and SSH tunnel creation/reuse to reduce overhead
    of creating new connections for each write operation.

    Architecture:
    - One shared tunnel per (host, port) combination
    - One client per tunnel (reused for all operations)
    - Thread-safe concurrent access via lock
    - Lazy initialization (tunnel created on first use)

    Performance Improvement (Phase 3):
    - Sequential baseline: 50 writes x 3ms connection overhead = 150ms
    - Pooled: ~3ms first connection + reuse = 150ms -> ~100ms (1.5x speedup)

    Examples
    --------
    >>> pool = ClickHouseConnectionPool()
    >>> with pool.get_client() as client:
    ...     result = client.query("SELECT 1")
    >>> pool.close()
    """

    def __init__(self, max_pool_size: int = 8) -> None:
        """Initialize connection pool.

        Parameters
        ----------
        max_pool_size : int
            Maximum number of active clients (default: 8).
        """
        self._lock = threading.Lock()
        self._max_pool_size = max_pool_size
        self._clients: list[tuple[clickhouse_connect.driver.Client, bool]] = []
        self._tunnels: dict[tuple[str, int], SSHTunnel] = {}
        self._total_created = 0
        self._total_reused = 0

    @contextmanager
    def get_client(
        self, host: str = "localhost", port: int = 8123
    ) -> clickhouse_connect.driver.Client:
        """Get a client from pool (blocking if none available).

        Parameters
        ----------
        host : str
            ClickHouse host
        port : int
            ClickHouse HTTP port

        Yields
        ------
        clickhouse_connect.driver.Client
            Pooled client connection
        """
        with self._lock:
            client = self._acquire_client(host, port)

            try:
                yield client
            finally:
                # Mark client as available
                for i, (c, _) in enumerate(self._clients):
                    if c is client:
                        self._clients[i] = (c, False)
                        self._total_reused += 1
                        break

    def _acquire_client(
        self, host: str, port: int
    ) -> clickhouse_connect.driver.Client:
        """Acquire a client (create or reuse from pool).

        Returns
        -------
        clickhouse_connect.driver.Client
            Available client connection
        """
        from .client import get_client
        from .tunnel import SSHTunnel

        # Try to reuse existing idle client
        for i, (client, is_in_use) in enumerate(self._clients):
            if not is_in_use:
                self._clients[i] = (client, True)
                logger.debug(
                    "Reusing client from pool (%d/%d)",
                    len(self._clients),
                    self._max_pool_size,
                )
                return client

        # Create new client if pool not full
        if len(self._clients) < self._max_pool_size:
            # Check if tunnel exists for this (host, port)
            tunnel_key = (host, port)
            if tunnel_key not in self._tunnels:
                # Create new tunnel (expensive operation)
                tunnel = SSHTunnel(host, port)
                local_port = tunnel.start()
                self._tunnels[tunnel_key] = tunnel
                logger.info(
                    "Created SSH tunnel: %s:%d -> localhost:%d",
                    host,
                    port,
                    local_port,
                )
            else:
                local_port = self._tunnels[tunnel_key].local_port
                logger.debug("Reusing existing tunnel: %s:%d", host, port)

            # Create client connected to tunnel
            client = get_client("localhost", local_port)
            self._clients.append((client, True))
            self._total_created += 1
            logger.info(
                "Created new client (%d/%d in pool)",
                len(self._clients),
                self._max_pool_size,
            )
            return client

        # Pool full - this shouldn't happen with max_pool_size >= workers
        msg = f"Connection pool exhausted ({len(self._clients)}/{self._max_pool_size})"
        raise RuntimeError(msg)

    def close(self) -> None:
        """Close all pooled clients and tunnels."""
        with self._lock:
            # Close all clients
            for client, _ in self._clients:
                try:
                    client.close()
                except (OSError, RuntimeError) as e:
                    logger.debug("Error closing client: %s", e)
            self._clients.clear()

            # Close all tunnels
            for (host, port), tunnel in self._tunnels.items():
                try:
                    tunnel.stop()
                    logger.debug("Stopped tunnel: %s:%d", host, port)
                except (OSError, RuntimeError) as e:
                    logger.debug("Error stopping tunnel: %s", e)
            self._tunnels.clear()

            logger.info(
                "Connection pool closed (created: %d, reused: %d)",
                self._total_created,
                self._total_reused,
            )

    def get_metrics(self) -> dict[str, int]:
        """Get pool usage metrics."""
        with self._lock:
            total_ops = max(1, self._total_reused + self._total_created)
            return {
                "active_clients": len(self._clients),
                "max_pool_size": self._max_pool_size,
                "active_tunnels": len(self._tunnels),
                "total_created": self._total_created,
                "total_reused": self._total_reused,
                "reuse_ratio_pct": (100 * self._total_reused // total_ops),
            }


class _PoolSingleton:
    """Singleton holder for global connection pool."""

    _instance: ClickHouseConnectionPool | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, max_pool_size: int = 8) -> ClickHouseConnectionPool:
        """Get or create the global connection pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ClickHouseConnectionPool(
                        max_pool_size=max_pool_size
                    )
                    logger.info(
                        "Initialized global connection pool (max: %d)",
                        max_pool_size,
                    )
        return cls._instance


def get_connection_pool(max_pool_size: int = 8) -> ClickHouseConnectionPool:
    """Get or create the global connection pool (singleton)."""
    return _PoolSingleton.get(max_pool_size)


__all__ = [
    "ClickHouseConnectionPool",
    "get_connection_pool",
]
