"""ClickHouse client mixin for rangebar cache classes.

This module provides a mixin for managing ClickHouse client lifecycle,
following the pattern from exness-data-preprocess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .client import get_client

if TYPE_CHECKING:
    import clickhouse_connect


class ClickHouseClientMixin:
    """Mixin for classes that need ClickHouse client access.

    Provides client lifecycle management with ownership tracking.
    If a client is provided externally, the class doesn't own it.
    If created internally, the class owns it and will close it.

    Attributes
    ----------
    _client : Client | None
        ClickHouse client instance
    _owns_client : bool
        Whether this instance owns the client (should close it)

    Examples
    --------
    >>> class MyCache(ClickHouseClientMixin):
    ...     def __init__(self, client=None):
    ...         self._init_client(client)
    ...
    >>> cache = MyCache()  # Creates own client
    >>> cache.client.command("SELECT 1")
    '1'
    >>> cache.close()  # Closes client
    """

    _client: clickhouse_connect.driver.Client | None = None
    _owns_client: bool = False

    def _init_client(
        self,
        client: clickhouse_connect.driver.Client | None = None,
        host: str = "localhost",
        port: int = 8123,
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        client : Client | None
            External client to use. If None, creates a new client.
        host : str
            Host for new client (default: localhost)
        port : int
            Port for new client (default: 8123)
        """
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = None  # Lazy initialization
            self._owns_client = True
            self._client_host = host
            self._client_port = port

    @property
    def client(self) -> clickhouse_connect.driver.Client:
        """Get the ClickHouse client.

        Creates a new client if one doesn't exist and this instance
        owns its client.

        Returns
        -------
        Client
            ClickHouse client

        Raises
        ------
        ClickHouseUnavailableError
            If client creation fails
        """
        if self._client is None:
            if self._owns_client:
                self._client = get_client(
                    host=getattr(self, "_client_host", "localhost"),
                    port=getattr(self, "_client_port", 8123),
                )
            else:
                msg = "No client available and this instance doesn't own one"
                raise RuntimeError(msg)
        return self._client

    def close(self) -> None:
        """Close the client if owned by this instance.

        Safe to call multiple times. Only closes client if this
        instance created it.
        """
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore close errors
            finally:
                self._client = None

    def __enter__(self) -> ClickHouseClientMixin:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - closes client."""
        self.close()
