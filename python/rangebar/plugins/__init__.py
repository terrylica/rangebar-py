# Issue #98: FeatureProvider plugin system for external feature enrichment.
"""Plugin system for enriching range bars with external feature columns.

Provides entry-point based discovery of FeatureProvider plugins that add
columns (e.g., Laguerre RSI) to range bars before ClickHouse INSERT.

Usage
-----
>>> from rangebar.plugins import discover_providers, enrich_bars
>>> providers = discover_providers()  # auto-discovers installed plugins
>>> enriched_df = enrich_bars(bars_df, "BTCUSDT", 500)
"""

from __future__ import annotations

from .loader import discover_providers, enrich_bars
from .protocol import FeatureProvider

__all__ = [
    "FeatureProvider",
    "discover_providers",
    "enrich_bars",
]
