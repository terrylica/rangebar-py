"""Unified configuration for rangebar-py (pydantic-settings).

Issue #110: All configuration models use BaseSettings with automatic overlay:
CLI flags > env vars > rangebar.toml > defaults.

Usage
-----
>>> from rangebar.config import Settings
>>> s = Settings.get()
>>> s.population.compute_tier2
True
"""

from __future__ import annotations

from .monitoring import MonitoringConfig
from .population import PopulationConfig
from .settings import Settings

__all__ = [
    "MonitoringConfig",
    "PopulationConfig",
    "Settings",
]


# Re-exports are lazy via __getattr__ to avoid triggering ClickHouse
# import-time checks when only PopulationConfig/MonitoringConfig are needed.


def __getattr__(name: str) -> type:
    if name == "ClickHouseConfig":
        from .clickhouse import ClickHouseConfig

        return ClickHouseConfig
    if name == "SidecarConfig":
        from .sidecar import SidecarConfig

        return SidecarConfig
    if name in ("ClickHouseConfigError", "ConnectionMode"):
        from .clickhouse import ClickHouseConfigError, ConnectionMode

        mapping = {
            "ClickHouseConfigError": ClickHouseConfigError,
            "ConnectionMode": ConnectionMode,
        }
        return mapping[name]
    if name == "AlgorithmConfig":
        from .algorithm import AlgorithmConfig

        return AlgorithmConfig
    if name == "StreamingConfig":
        from .streaming import StreamingConfig

        return StreamingConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
