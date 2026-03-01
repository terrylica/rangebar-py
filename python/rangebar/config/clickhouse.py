"""ClickHouse configuration for rangebar cache.

Issue #110: Migrated from python/rangebar/clickhouse/config.py to pydantic-settings.

This module provides configuration management following the mise SSoT pattern.
All host information comes from environment variables or ~/.ssh/config aliases,
never hardcoded in the package.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_PORT = 65535


class ClickHouseConfigError(ValueError):
    """Configuration error for ClickHouse connection."""


class ConnectionMode(StrEnum):
    """Connection mode for ClickHouse cache.

    Controls how rangebar-py connects to ClickHouse:
    - LOCAL: Force localhost:8123 only (no SSH aliases)
    - CLOUD: Require CLICKHOUSE_HOST environment variable
    - AUTO: Auto-detect (try localhost first, then SSH aliases)
    - REMOTE: Skip localhost, use RANGEBAR_CH_HOSTS via direct/SSH
    """

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"
    REMOTE = "remote"


class ClickHouseConfig(BaseSettings):
    """Configuration for ClickHouse connection.

    Follows the mise SSoT pattern — all values come from environment variables.
    Host aliases reference ~/.ssh/config entries, never actual IPs/hostnames.

    Environment Variables
    ---------------------
    CLICKHOUSE_HOST : str
        Host to connect to (default: "localhost")
    CLICKHOUSE_PORT : int
        Port number (default: 8123)
    RANGEBAR_CH_DATABASE : str
        Database name (default: "rangebar_cache")
    RANGEBAR_MODE : str
        Connection mode: "local", "cloud", "auto", "remote" (default: "auto")
    RANGEBAR_CH_HOSTS : str
        Comma-separated SSH aliases from ~/.ssh/config
    RANGEBAR_CH_PRIMARY : str
        Primary host alias to prefer
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_CH_",
        case_sensitive=False,
    )

    # Alias: existing env var is CLICKHOUSE_HOST (no RANGEBAR_CH_ prefix)
    host: str = Field(
        "localhost",
        validation_alias=AliasChoices("CLICKHOUSE_HOST", "RANGEBAR_CH_HOST"),
    )
    # Alias: existing env var is CLICKHOUSE_PORT (no RANGEBAR_CH_ prefix)
    port: int = Field(
        8123,
        validation_alias=AliasChoices("CLICKHOUSE_PORT", "RANGEBAR_CH_PORT"),
    )
    database: str = "rangebar_cache"
    # Alias: existing env var is RANGEBAR_MODE (no CH_ infix)
    mode: ConnectionMode = Field(
        ConnectionMode.AUTO,
        validation_alias=AliasChoices("RANGEBAR_MODE", "RANGEBAR_CH_MODE"),
    )
    # Additional fields used by preflight.py
    hosts: str = ""
    primary: str = ""

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v: str | ConnectionMode) -> ConnectionMode:
        if isinstance(v, str):
            try:
                return ConnectionMode(v.lower())
            except ValueError:
                return ConnectionMode.AUTO
        return v

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        """Create configuration from environment variables (backwards-compatible)."""
        return cls()

    def validate(self) -> None:
        """Validate configuration.

        Raises
        ------
        ClickHouseConfigError
            If configuration is invalid
        """
        if self.port < 1 or self.port > MAX_PORT:
            msg = f"Invalid port: {self.port}"
            raise ClickHouseConfigError(msg)

        if not self.database:
            msg = "Database name cannot be empty"
            raise ClickHouseConfigError(msg)

    @property
    def connection_string(self) -> str:
        """Get connection string for display/logging."""
        return f"{self.host}:{self.port}/{self.database}"
