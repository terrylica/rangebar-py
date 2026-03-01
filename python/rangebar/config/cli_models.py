"""Pydantic CliApp models for the rangebar populate CLI.

Issue #110: Unified CLI via pydantic-settings CliApp.

Provides structured subcommands for cache population with full feature
toggle support (Issue #128). All fields accept CLI flags, env vars,
and rangebar.toml overrides via pydantic-settings automatic cascade.

Usage
-----
    rangebar populate range --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
    rangebar populate month --symbol BTCUSDT --month 2024-03
    rangebar populate year --symbol ETHUSDT --year 2024 --tier3 --hurst
    rangebar populate phase --phase 2 --parallel 2
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, CliSubCommand, SettingsConfigDict


class PopulateBase(BaseSettings):
    """Shared options inherited by all populate subcommands."""

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_",
        cli_parse_args=False,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        case_sensitive=False,
    )

    # Core
    threshold: int = Field(250, description="Threshold in dbps")
    ouroboros: Literal["year", "month", "week"] = Field(
        "month", description="Reset boundary",
    )
    microstructure: bool = Field(True, description="Include microstructure features")

    # Feature toggles (Issue #128 — all CLI-controllable)
    tier2: bool = Field(True, description="Tier 2 inter-bar features")
    tier3: bool = Field(False, description="Tier 3 inter-bar features")
    hurst: bool = Field(False, description="Hurst DFA computation")
    permutation_entropy: bool = Field(False, description="Permutation entropy")

    # Lookback
    lookback_count: int = Field(200, description="Inter-bar lookback trade count")
    lookback_bars: int | None = Field(None, description="Inter-bar lookback bar count")

    # Execution
    force_refresh: bool = Field(False, description="Wipe cache + checkpoint")
    notify: bool = Field(True, description="Telegram notifications")
    memory_limit: float | None = Field(None, description="Memory limit in GB")


class PopulateRange(PopulateBase):
    """Populate cache for a date range."""

    symbol: str = Field(description="Trading symbol (e.g. BTCUSDT)")
    start: str = Field(description="Start date (YYYY-MM-DD)")
    end: str = Field(description="End date (YYYY-MM-DD)")

    def cli_cmd(self) -> None:
        """Execute the populate range command."""
        from rangebar.population import run_populate_range

        run_populate_range(self)


class PopulateMonth(PopulateBase):
    """Populate cache for a calendar month."""

    symbol: str = Field(description="Trading symbol")
    month: str = Field(description="Month (YYYY-MM)")

    def cli_cmd(self) -> None:
        """Execute the populate month command."""
        from rangebar.population import run_populate_month

        run_populate_month(self)


class PopulateYear(PopulateBase):
    """Populate cache for a calendar year."""

    symbol: str = Field(description="Trading symbol")
    year: int = Field(description="Year (e.g. 2024)")

    def cli_cmd(self) -> None:
        """Execute the populate year command."""
        from rangebar.population import run_populate_year

        run_populate_year(self)


class PopulatePhase(PopulateBase):
    """Run batch population phase (all registered symbols)."""

    phase: int = Field(ge=1, le=4, description="Phase 1-4")
    parallel: int = Field(1, description="Parallel job count")

    def cli_cmd(self) -> None:
        """Execute the populate phase command."""
        from rangebar.population import run_populate_phase

        run_populate_phase(self)


class Populate(BaseSettings):
    """Populate ClickHouse cache with range bars."""

    model_config = SettingsConfigDict(cli_parse_args=True)

    range: CliSubCommand[PopulateRange]
    month: CliSubCommand[PopulateMonth]
    year: CliSubCommand[PopulateYear]
    phase: CliSubCommand[PopulatePhase]
