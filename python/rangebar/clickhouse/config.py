"""ClickHouse configuration for rangebar cache.

This module provides configuration management following the mise SSoT pattern.
All host information comes from environment variables or ~/.ssh/config aliases,
never hardcoded in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ClickHouseConfigError(ValueError):
    """Configuration error for ClickHouse connection."""


class ConnectionMode(str, Enum):
    """Connection mode for ClickHouse cache.

    Controls how rangebar-py connects to ClickHouse:
    - LOCAL: Force localhost:8123 only (no SSH aliases)
    - CLOUD: Require CLICKHOUSE_HOST environment variable
    - AUTO: Auto-detect (try localhost first, then SSH aliases)
    """

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


def get_connection_mode() -> ConnectionMode:
    """Get the connection mode from environment.

    Returns
    -------
    ConnectionMode
        Current connection mode based on RANGEBAR_MODE env var.
        Defaults to AUTO if not set.

    Examples
    --------
    >>> import os
    >>> os.environ["RANGEBAR_MODE"] = "local"
    >>> get_connection_mode()
    <ConnectionMode.LOCAL: 'local'>
    """
    mode_str = os.getenv("RANGEBAR_MODE", "auto").lower()
    try:
        return ConnectionMode(mode_str)
    except ValueError:
        # Invalid mode, default to AUTO
        return ConnectionMode.AUTO


@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection.

    Follows the mise SSoT pattern - all values come from environment variables.
    Host aliases reference ~/.ssh/config entries, never actual IPs/hostnames.

    Parameters
    ----------
    host : str
        Host to connect to (default: localhost)
    port : int
        Port number (default: 8123)
    database : str
        Database name (default: rangebar_cache)
    mode : ConnectionMode
        Connection mode (default: AUTO)

    Environment Variables
    ---------------------
    RANGEBAR_MODE : str
        Connection mode: "local", "cloud", or "auto" (default: auto)
    RANGEBAR_CH_HOSTS : str
        Comma-separated list of SSH aliases from ~/.ssh/config
    RANGEBAR_CH_PRIMARY : str
        Primary host alias to prefer
    CLICKHOUSE_HOST : str
        Direct host override (for localhost only)
    CLICKHOUSE_PORT : str
        Port override
    """

    host: str = field(default_factory=lambda: os.getenv("CLICKHOUSE_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("CLICKHOUSE_PORT", "8123")))
    database: str = "rangebar_cache"
    mode: ConnectionMode = field(default_factory=get_connection_mode)

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        """Create configuration from environment variables.

        Returns
        -------
        ClickHouseConfig
            Configuration instance

        Examples
        --------
        >>> config = ClickHouseConfig.from_env()
        >>> print(config.host, config.port)
        localhost 8123
        """
        return cls(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            database=os.getenv("RANGEBAR_CH_DATABASE", "rangebar_cache"),
            mode=get_connection_mode(),
        )

    def validate(self) -> None:
        """Validate configuration.

        Raises
        ------
        ClickHouseConfigError
            If configuration is invalid
        """
        if self.port < 1 or self.port > 65535:
            msg = f"Invalid port: {self.port}"
            raise ClickHouseConfigError(msg)

        if not self.database:
            msg = "Database name cannot be empty"
            raise ClickHouseConfigError(msg)

    @property
    def connection_string(self) -> str:
        """Get connection string for display/logging.

        Returns
        -------
        str
            Connection string (host:port/database)
        """
        return f"{self.host}:{self.port}/{self.database}"
