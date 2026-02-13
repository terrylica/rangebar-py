"""Alpha-forge compatibility layer for rangebar-py (Issue #95).

Provides feature metadata, panel format conversion, and cache availability
APIs to simplify future integration with alpha-forge.
"""

from __future__ import annotations

from .availability import get_available_symbols, get_cache_coverage
from .manifest import (
    FeatureGroup,
    FeatureMetadata,
    FeatureRegistry,
    get_feature_groups,
    get_feature_manifest,
)
from .panel import get_range_bars_panel, to_panel_format

__all__ = [
    "FeatureGroup",
    "FeatureMetadata",
    "FeatureRegistry",
    "get_available_symbols",
    "get_cache_coverage",
    "get_feature_groups",
    "get_feature_manifest",
    "get_range_bars_panel",
    "to_panel_format",
]
