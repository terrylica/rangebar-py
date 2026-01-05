"""ClickHouse client utilities for rangebar cache.

This module provides client creation and exception handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import clickhouse_connect


class ClickHouseUnavailableError(RuntimeError):
    """ClickHouse not available at specified host:port.

    Raised when connection to ClickHouse fails. Includes actionable guidance.
    """


class ClickHouseQueryError(RuntimeError):
    """Query execution failed.

    Raised when a ClickHouse query fails to execute.
    """


def get_client(
    host: str = "localhost",
    port: int = 8123,
    database: str = "default",
    **kwargs: Any,
) -> clickhouse_connect.driver.Client:
    """Get a ClickHouse client connection.

    Parameters
    ----------
    host : str
        Host to connect to (default: localhost)
    port : int
        HTTP port (default: 8123)
    database : str
        Database to use (default: default)
    **kwargs
        Additional arguments passed to clickhouse_connect.get_client()

    Returns
    -------
    clickhouse_connect.driver.Client
        Connected ClickHouse client

    Raises
    ------
    ClickHouseUnavailableError
        If connection fails

    Examples
    --------
    >>> client = get_client()
    >>> version = client.command("SELECT version()")
    >>> client.close()
    """
    try:
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            **kwargs,
        )

        # Verify connection with a simple query
        client.command("SELECT 1")
        return client

    except ImportError as e:
        msg = (
            "clickhouse-connect not installed. "
            "Install with: pip install clickhouse-connect"
        )
        raise ClickHouseUnavailableError(msg) from e

    except Exception as e:
        msg = f"Failed to connect to ClickHouse at {host}:{port}: {e}"
        raise ClickHouseUnavailableError(msg) from e


def execute_query(
    client: clickhouse_connect.driver.Client,
    query: str,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Execute a ClickHouse query with error handling.

    Parameters
    ----------
    client : Client
        ClickHouse client
    query : str
        SQL query to execute
    parameters : dict, optional
        Query parameters

    Returns
    -------
    Any
        Query result

    Raises
    ------
    ClickHouseQueryError
        If query execution fails
    """
    try:
        if parameters:
            return client.command(query, parameters=parameters)
        return client.command(query)
    except Exception as e:
        msg = f"Query failed: {e}\nQuery: {query[:200]}..."
        raise ClickHouseQueryError(msg) from e
