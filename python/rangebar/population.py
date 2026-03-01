"""Shared population logic for the rangebar CLI and scripts.

Issue #110: Extracted from scripts/populate_full_cache.py to be importable
from both the `rangebar populate` CLI and the standalone script.

Provides phase definitions, date helpers, and run_* functions that bridge
CLI models to ``populate_cache_resumable()``.
"""

from __future__ import annotations

import logging
import sys
from calendar import monthrange
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rangebar.config.cli_models import (
        PopulateMonth,
        PopulatePhase,
        PopulateRange,
        PopulateYear,
    )

logger = logging.getLogger(__name__)

# Phase definitions (resource-aware grouping)
PHASES: dict[int, dict] = {
    1: {
        "name": "Quick wins (1000 dbps)",
        "thresholds": [1000],
        "max_parallel": 4,
        "estimated_hours": 8,
    },
    2: {
        "name": "Day trading (250 dbps)",
        "thresholds": [250],
        "max_parallel": 2,
        "estimated_hours": 24,
    },
    3: {
        "name": "Mid thresholds (500, 750 dbps)",
        "thresholds": [500, 750],
        "max_parallel": 3,
        "estimated_hours": 16,
    },
    4: {
        "name": "Scalping (100 dbps) - RESOURCE INTENSIVE",
        "thresholds": [100],
        "max_parallel": 1,
        "estimated_hours": 80,
    },
}


def _enable_notifications(notify: bool) -> None:
    """Enable Telegram notifications if requested."""
    if notify:
        try:
            from rangebar.notify.telegram import enable_telegram_notifications

            enable_telegram_notifications()
        except ImportError:
            logger.warning("Telegram notifications not available")


def _populate_one(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    threshold: int = 250,
    ouroboros: str = "month",
    microstructure: bool = True,
    tier2: bool = True,
    tier3: bool = False,
    hurst: bool = False,
    permutation_entropy: bool = False,
    lookback_count: int = 200,  # noqa: ARG001
    lookback_bars: int | None = None,  # noqa: ARG001
    force_refresh: bool = False,
    notify: bool = True,
) -> int:
    """Run a single populate job. Returns bar count."""
    from rangebar import populate_cache_resumable

    _enable_notifications(notify)

    start_time = datetime.now(tz=UTC)
    logger.info(
        "Starting: %s @ %d dbps from %s to %s",
        symbol, threshold, start_date, end_date,
    )

    bars = populate_cache_resumable(
        symbol,
        start_date,
        end_date,
        threshold_decimal_bps=threshold,
        include_microstructure=microstructure,
        force_refresh=force_refresh,
        ouroboros_mode=ouroboros,
        notify=notify,
        compute_tier2=tier2,
        compute_tier3=tier3,
        compute_hurst=hurst,
        compute_permutation_entropy=permutation_entropy,
    )

    elapsed = (datetime.now(tz=UTC) - start_time).total_seconds() / 60
    logger.info(
        "Completed: %s @ %d - %d bars in %.1f min",
        symbol, threshold, bars, elapsed,
    )
    return bars


def _settings_from_base(settings: PopulateRange | PopulateMonth | PopulateYear) -> dict:
    """Extract common kwargs from a PopulateBase subclass."""
    return {
        "threshold": settings.threshold,
        "ouroboros": settings.ouroboros,
        "microstructure": settings.microstructure,
        "tier2": settings.tier2,
        "tier3": settings.tier3,
        "hurst": settings.hurst,
        "permutation_entropy": settings.permutation_entropy,
        "lookback_count": settings.lookback_count,
        "lookback_bars": settings.lookback_bars,
        "force_refresh": settings.force_refresh,
        "notify": settings.notify,
    }


def run_populate_range(settings: PopulateRange) -> int:
    """Execute populate for a date range."""
    return _populate_one(
        settings.symbol,
        settings.start,
        settings.end,
        **_settings_from_base(settings),
    )


def run_populate_month(settings: PopulateMonth) -> int:
    """Execute populate for a calendar month."""
    try:
        dt = datetime.strptime(settings.month, "%Y-%m")  # noqa: DTZ007
    except ValueError:
        logger.exception("Invalid month format: %s. Use YYYY-MM.", settings.month)
        sys.exit(1)

    year, month = dt.year, dt.month
    _, last_day = monthrange(year, month)
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    return _populate_one(
        settings.symbol,
        start_date,
        end_date,
        **_settings_from_base(settings),
    )


def run_populate_year(settings: PopulateYear) -> int:
    """Execute populate for a calendar year."""
    start_date = f"{settings.year}-01-01"
    end_date = f"{settings.year}-12-31"

    return _populate_one(
        settings.symbol,
        start_date,
        end_date,
        **_settings_from_base(settings),
    )


def run_populate_phase(settings: PopulatePhase) -> int:
    """Execute batch population for a phase (all registered symbols)."""
    from rangebar.binance_vision import probe_latest_available_date
    from rangebar.symbol_registry import (
        get_effective_start_date,
        get_registered_symbols,
    )

    if settings.phase not in PHASES:
        logger.error("Invalid phase: %d. Use 1-4.", settings.phase)
        sys.exit(1)

    phase = PHASES[settings.phase]
    parallel = min(settings.parallel, phase["max_parallel"])
    if parallel < settings.parallel:
        logger.warning(
            "Phase %d max parallel is %d, reducing from %d",
            settings.phase, phase["max_parallel"], settings.parallel,
        )

    _enable_notifications(settings.notify)

    # Load symbols from registry
    registered = get_registered_symbols(asset_class="crypto")
    end_date = probe_latest_available_date()

    logger.info("=" * 70)
    logger.info("PHASE %d: %s", settings.phase, phase["name"])
    logger.info("Parallelism: %d | Symbols: %d", parallel, len(registered))
    logger.info("=" * 70)

    total_bars = 0
    for symbol in registered:
        start = get_effective_start_date(symbol)
        if not start:
            logger.warning("Skipping %s (no effective_start)", symbol)
            continue

        for threshold in phase["thresholds"]:
            try:
                bars = _populate_one(
                    symbol,
                    start,
                    end_date,
                    threshold=threshold,
                    ouroboros=settings.ouroboros,
                    microstructure=settings.microstructure,
                    tier2=settings.tier2,
                    tier3=settings.tier3,
                    hurst=settings.hurst,
                    permutation_entropy=settings.permutation_entropy,
                    force_refresh=settings.force_refresh,
                    notify=settings.notify,
                )
                total_bars += bars
            except Exception:
                logger.exception("Failed: %s @ %d", symbol, threshold)

    logger.info("Phase %d complete: %d total bars", settings.phase, total_bars)
    return total_bars
