"""Sidecar configuration for the streaming sidecar.

Issue #110: Migrated from python/rangebar/sidecar.py to pydantic-settings.
Issue #96 Task #6: RANGEBAR_MAX_PENDING_BARS backpressure control.
Issue #107: Watchdog reliability parameters.
Issue #109: Health check HTTP port.

Environment variables use RANGEBAR_STREAMING_ prefix (historical convention).
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources.providers.env import EnvSettingsSource


class _CsvEnvSource(EnvSettingsSource):
    """Env source that falls back to CSV splitting for list fields.

    pydantic-settings tries ``json.loads()`` first for complex types. Our
    env vars use comma-separated format (``BTCUSDT,ETHUSDT``), not JSON
    (``["BTCUSDT","ETHUSDT"]``). This source falls back to CSV when JSON
    parsing fails.
    """

    def decode_complex_value(
        self,
        field_name: str,
        field: Any,  # noqa: ANN401
        value: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            # Fall back to CSV parsing for list fields
            if isinstance(value, str):
                return [s.strip() for s in value.split(",") if s.strip()]
            raise


class SidecarConfig(BaseSettings):
    """Configuration for the streaming sidecar.

    Environment Variables
    ---------------------
    RANGEBAR_STREAMING_SYMBOLS : str
        Comma-separated symbol list (default: "")
    RANGEBAR_STREAMING_THRESHOLDS : str
        Comma-separated threshold list (default: "250,500,750,1000")
    RANGEBAR_STREAMING_MICROSTRUCTURE : bool
        Include microstructure features (default: True)
    RANGEBAR_STREAMING_GAP_FILL : bool
        Fill gaps on startup (default: True)
    RANGEBAR_STREAMING_VERBOSE : bool
        Verbose logging (default: False)
    RANGEBAR_STREAMING_TIMEOUT_MS : int
        WebSocket timeout in ms (default: 5000)
    RANGEBAR_STREAMING_WATCHDOG_TIMEOUT_S : int
        Watchdog dead-engine detection (default: 300)
    RANGEBAR_STREAMING_MAX_WATCHDOG_RESTARTS : int
        Max engine restarts before exit (default: 3)
    RANGEBAR_MAX_PENDING_BARS : int
        Backpressure bound (default: 10000)
    RANGEBAR_HEALTH_PORT : int
        Health check HTTP port, 0 to disable (default: 8081)
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_STREAMING_",
        case_sensitive=False,
    )

    symbols: list[str] = Field(default_factory=list)
    thresholds: list[int] = Field(default=[250, 500, 750, 1000])
    # Alias: existing env var is RANGEBAR_STREAMING_MICROSTRUCTURE (no INCLUDE_)
    include_microstructure: bool = Field(
        True,
        validation_alias=AliasChoices(
            "RANGEBAR_STREAMING_MICROSTRUCTURE",
            "RANGEBAR_STREAMING_INCLUDE_MICROSTRUCTURE",
        ),
    )
    # Alias: existing env var is RANGEBAR_STREAMING_GAP_FILL (no ON_STARTUP)
    gap_fill_on_startup: bool = Field(
        True,
        validation_alias=AliasChoices(
            "RANGEBAR_STREAMING_GAP_FILL",
            "RANGEBAR_STREAMING_GAP_FILL_ON_STARTUP",
        ),
    )
    verbose: bool = False
    timeout_ms: int = 5000
    watchdog_timeout_s: int = 300
    max_watchdog_restarts: int = 3
    # Alias: existing env var is RANGEBAR_MAX_PENDING_BARS (no STREAMING_ prefix)
    max_pending_bars: int = Field(
        10_000,
        validation_alias=AliasChoices(
            "RANGEBAR_MAX_PENDING_BARS",
            "RANGEBAR_STREAMING_MAX_PENDING_BARS",
        ),
    )
    # Alias: existing env var is RANGEBAR_HEALTH_PORT (no STREAMING_ prefix)
    health_port: int = Field(
        8081,
        validation_alias=AliasChoices(
            "RANGEBAR_HEALTH_PORT",
            "RANGEBAR_STREAMING_HEALTH_PORT",
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,  # noqa: ANN401
        env_settings: Any,  # noqa: ANN401, ARG003
        dotenv_settings: Any,  # noqa: ANN401
        file_secret_settings: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401, ARG003
    ) -> tuple[Any, ...]:
        """Use CSV-aware env source for comma-separated list fields."""
        return (
            init_settings,
            _CsvEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def from_env(cls) -> SidecarConfig:
        """Create config from environment (backwards-compatible factory)."""
        return cls()
