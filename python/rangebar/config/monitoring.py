"""Monitoring configuration for notifications and recency checks.

Issue #110: Unified configuration via pydantic-settings.

All values load from environment variables (RANGEBAR_ prefix) with automatic
overlay: CLI > env > TOML > defaults via pydantic-settings.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring, notifications, and recency checks.

    Environment Variables
    ---------------------
    RANGEBAR_TELEGRAM_TOKEN : str | None
        Telegram bot token (default: None)
    RANGEBAR_TELEGRAM_CHAT_ID : str
        Telegram chat ID (default: "")
    RANGEBAR_RECENCY_FRESH_THRESHOLD_MIN : int
        Minutes before data is considered stale (default: 30)
    RANGEBAR_RECENCY_STALE_THRESHOLD_MIN : int
        Minutes before data is considered stale (default: 120)
    RANGEBAR_RECENCY_CRITICAL_THRESHOLD_MIN : int
        Minutes before data is critical (default: 1440)
    RANGEBAR_ENV : str
        Environment name (default: "development")
    RANGEBAR_GIT_SHA : str
        Git commit SHA (default: "unknown")
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_",
        case_sensitive=False,
    )

    telegram_token: str | None = None
    telegram_chat_id: str = ""
    recency_fresh_threshold_min: int = 30
    recency_stale_threshold_min: int = 120
    recency_critical_threshold_min: int = 1440
    # Env var is RANGEBAR_ENV, not RANGEBAR_ENVIRONMENT (historical)
    environment: str = Field(
        "development",
        validation_alias=AliasChoices("RANGEBAR_ENV", "RANGEBAR_ENVIRONMENT"),
    )
    git_sha: str = "unknown"
