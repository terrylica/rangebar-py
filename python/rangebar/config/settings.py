"""Unified Settings singleton for rangebar-py.

Issue #110: Thread-safe singleton that composes all config sections. Preserves the
existing ``Settings.get()`` / ``Settings.reload()`` API used by 11+ files.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .monitoring import MonitoringConfig
from .population import PopulationConfig

if TYPE_CHECKING:
    from .algorithm import AlgorithmConfig
    from .clickhouse import ClickHouseConfig
    from .sidecar import SidecarConfig
    from .streaming import StreamingConfig

_lock = threading.Lock()
_instance: Settings | None = None


@dataclass
class Settings:
    """Unified configuration for rangebar-py.

    Consolidates ClickHouseConfig, SidecarConfig, PopulationConfig, and
    MonitoringConfig into a single entry point. All values loaded from
    environment variables (mise SSoT in ``.mise.toml``).

    Usage
    -----
    >>> settings = Settings.get()
    >>> settings.population.default_threshold
    250
    >>> settings.monitoring.telegram_chat_id
    '90417581'

    Reload after env change:

    >>> Settings.reload()
    >>> settings = Settings.get()
    """

    population: PopulationConfig = field(default_factory=PopulationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def get(cls) -> Settings:
        """Return the cached singleton instance (thread-safe).

        Returns
        -------
        Settings
            The cached settings instance. Creates one on first call.
        """
        global _instance  # noqa: PLW0603
        if _instance is not None:
            return _instance
        with _lock:
            if _instance is None:
                _instance = cls()
            return _instance

    @classmethod
    def reload(cls) -> Settings:
        """Discard the cached instance and reload from environment.

        Returns
        -------
        Settings
            A freshly loaded settings instance.
        """
        global _instance  # noqa: PLW0603
        with _lock:
            _instance = cls()
            return _instance

    @classmethod
    def reload_and_clear(cls) -> Settings:
        """Reload settings and clear cached threshold data.  # Issue #126

        Combines ``reload()`` with threshold cache clearing. Designed for
        SIGHUP handlers in daemons to enable runtime config switching
        without service restart.

        Returns
        -------
        Settings
            A freshly loaded settings instance.
        """
        from rangebar.threshold import clear_threshold_cache

        settings = cls.reload()
        clear_threshold_cache()
        return settings

    @property
    def clickhouse(self) -> ClickHouseConfig:
        """Lazy access to ClickHouse config (avoids circular import)."""
        from .clickhouse import ClickHouseConfig

        return ClickHouseConfig.from_env()

    @property
    def sidecar(self) -> SidecarConfig:
        """Lazy access to Sidecar config (avoids circular import)."""
        from .sidecar import SidecarConfig

        return SidecarConfig.from_env()

    @property
    def algorithm(self) -> AlgorithmConfig:
        """Lazy access to Algorithm config (Rust tunables, Phase 7)."""
        from .algorithm import AlgorithmConfig

        return AlgorithmConfig()

    @property
    def streaming(self) -> StreamingConfig:
        """Lazy access to Streaming config (Rust tunables, Phase 7)."""
        from .streaming import StreamingConfig

        return StreamingConfig()
