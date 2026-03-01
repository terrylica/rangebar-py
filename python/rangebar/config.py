"""Unified configuration for rangebar-py.

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/110

Consolidates scattered env var access into a single Settings dataclass
with nested sections. All values come from environment variables (mise SSoT).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rangebar.clickhouse.config import ClickHouseConfig
    from rangebar.sidecar import SidecarConfig


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes")


def _env_list(key: str, default: str) -> list[str]:
    raw = os.environ.get(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationConfig:
    """Configuration for cache population jobs."""

    default_threshold: int = field(
        default_factory=lambda: _env_int("RANGEBAR_CRYPTO_MIN_THRESHOLD", 250),
    )
    ouroboros_mode: str = field(
        default_factory=lambda: _env("RANGEBAR_OUROBOROS_MODE", "month"),
    )
    # Issue #126: Guard behavior for ouroboros mode consistency checks.
    # "strict": raise CacheWriteError on mode mismatch or connection failure
    # "warn" (default): log warning, allow write (safe during migration)
    # "off": skip check entirely
    ouroboros_guard: str = field(
        default_factory=lambda: _env("RANGEBAR_OUROBOROS_GUARD", "warn"),
    )
    inter_bar_lookback_count: int = field(
        default_factory=lambda: _env_int("RANGEBAR_INTER_BAR_LOOKBACK_COUNT", 200),
    )
    include_intra_bar_features: bool = field(
        default_factory=lambda: _env_bool("RANGEBAR_INCLUDE_INTRA_BAR_FEATURES", True),
    )
    symbol_gate: str = field(
        default_factory=lambda: _env("RANGEBAR_SYMBOL_GATE", "strict"),
    )
    continuity_tolerance: float = field(
        default_factory=lambda: _env_float("RANGEBAR_CONTINUITY_TOLERANCE", 0.001),
    )

    def __post_init__(self) -> None:  # Issue #126: Fail-fast on invalid config
        from rangebar.ouroboros import validate_ouroboros_mode

        validate_ouroboros_mode(self.ouroboros_mode)
        valid_guards = {"strict", "warn", "off"}
        if self.ouroboros_guard not in valid_guards:
            msg = (
                f"Invalid RANGEBAR_OUROBOROS_GUARD: {self.ouroboros_guard!r}. "
                f"Must be one of: {valid_guards}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for monitoring, notifications, and recency checks."""

    telegram_token: str | None = field(
        default_factory=lambda: os.environ.get("RANGEBAR_TELEGRAM_TOKEN"),
    )
    telegram_chat_id: str = field(
        default_factory=lambda: _env("RANGEBAR_TELEGRAM_CHAT_ID", ""),
    )
    recency_fresh_threshold_min: int = field(
        default_factory=lambda: _env_int(
            "RANGEBAR_RECENCY_FRESH_THRESHOLD_MIN", 30,
        ),
    )
    recency_stale_threshold_min: int = field(
        default_factory=lambda: _env_int(
            "RANGEBAR_RECENCY_STALE_THRESHOLD_MIN", 120,
        ),
    )
    recency_critical_threshold_min: int = field(
        default_factory=lambda: _env_int(
            "RANGEBAR_RECENCY_CRITICAL_THRESHOLD_MIN", 1440,
        ),
    )
    environment: str = field(
        default_factory=lambda: _env("RANGEBAR_ENV", "development"),
    )
    git_sha: str = field(
        default_factory=lambda: _env("RANGEBAR_GIT_SHA", "unknown"),
    )


# ---------------------------------------------------------------------------
# Unified Settings (singleton)
# ---------------------------------------------------------------------------

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
        from rangebar.clickhouse.config import ClickHouseConfig

        return ClickHouseConfig.from_env()

    @property
    def sidecar(self) -> SidecarConfig:
        """Lazy access to Sidecar config (avoids circular import)."""
        from rangebar.sidecar import SidecarConfig

        return SidecarConfig.from_env()


# Re-exports are lazy via __getattr__ to avoid triggering ClickHouse
# import-time checks when only PopulationConfig/MonitoringConfig are needed.

def __getattr__(name: str) -> type:
    if name == "ClickHouseConfig":
        from rangebar.clickhouse.config import ClickHouseConfig
        return ClickHouseConfig
    if name == "SidecarConfig":
        from rangebar.sidecar import SidecarConfig
        return SidecarConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
