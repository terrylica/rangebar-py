# Issue #98: Entry-point discovery and enrichment orchestration.
"""Plugin discovery via entry points and bar enrichment orchestration.

Discovers installed FeatureProvider plugins via the
``rangebar.feature_providers`` entry-point group. Providers are loaded
once per process and cached for zero-overhead subsequent calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .protocol import FeatureProvider

logger = logging.getLogger(__name__)

# Entry-point group name for plugin discovery
_ENTRY_POINT_GROUP = "rangebar.feature_providers"

# Lazy one-time discovery cache (R4: zero overhead after first call)
_providers_cache: list[FeatureProvider] | None = None


def discover_providers() -> list[FeatureProvider]:
    """Auto-discover installed FeatureProvider plugins via entry points.

    Scans the ``rangebar.feature_providers`` entry-point group, loads each
    provider class, validates it implements the FeatureProvider protocol,
    and registers its columns in the dynamic column registry.

    Returns
    -------
    list[FeatureProvider]
        List of validated provider instances. Empty if none installed.
    """
    from importlib.metadata import entry_points

    from rangebar.constants import register_plugin_columns

    eps = entry_points(group=_ENTRY_POINT_GROUP)
    providers: list[FeatureProvider] = []

    for ep in eps:
        try:
            cls = ep.load()
            instance = cls()
            if not isinstance(instance, FeatureProvider):
                logger.warning(
                    "Plugin %s does not implement FeatureProvider protocol, skipping",
                    ep.name,
                )
                continue
            register_plugin_columns(instance.columns)
            providers.append(instance)
            logger.info(
                "Loaded feature provider: %s v%s (%d columns)",
                instance.name,
                instance.version,
                len(instance.columns),
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning("Failed to load feature provider %s: %s", ep.name, e)

    return providers


def _get_providers() -> list[FeatureProvider]:
    """Get cached providers, discovering on first call."""
    global _providers_cache  # noqa: PLW0603
    if _providers_cache is None:
        _providers_cache = discover_providers()
    return _providers_cache


def enrich_bars(
    bars_df: pd.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
) -> pd.DataFrame:
    """Apply all discovered FeatureProvider plugins to a bar DataFrame.

    Discovers providers on first call (cached). If no providers are
    installed or the DataFrame is empty, returns immediately with
    zero overhead.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Range bar DataFrame with OHLCV columns.
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold used for bar generation.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with plugin columns added in-place.
    """
    providers = _get_providers()
    if not providers or bars_df.empty:
        return bars_df

    from rangebar.hooks import HookEvent, emit_hook

    for provider in providers:
        try:
            bars_df = provider.enrich(bars_df, symbol, threshold_decimal_bps)
            logger.debug(
                "Provider %s enriched %d bars with %d columns",
                provider.name,
                len(bars_df),
                len(provider.columns),
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:
            logger.warning(
                "Provider %s failed to enrich bars: %s", provider.name, e,
            )
            emit_hook(
                HookEvent.PLUGIN_ENRICH_FAILED,
                symbol=symbol,
                provider_name=provider.name,
                threshold_decimal_bps=threshold_decimal_bps,
            )

    emit_hook(
        HookEvent.PLUGIN_ENRICH_COMPLETE,
        symbol=symbol,
        provider_count=len(providers),
        threshold_decimal_bps=threshold_decimal_bps,
    )

    return bars_df


def reset_provider_cache() -> None:
    """Reset the provider cache. Useful for testing."""
    global _providers_cache  # noqa: PLW0603
    _providers_cache = None
