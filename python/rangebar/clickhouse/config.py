"""ClickHouse configuration for rangebar cache.

This module provides configuration management following the mise SSoT pattern.
All host information comes from environment variables or ~/.ssh/config aliases,
never hardcoded in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ClickHouseConfigError(ValueError):
    """Configuration error for ClickHouse connection."""


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

    Environment Variables
    ---------------------
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
