# Issue #98: Schema migration helper for plugin feature columns.
"""ClickHouse schema migration for FeatureProvider plugin columns.

Adds plugin-defined columns to the ``range_bars`` table using
``ALTER TABLE ADD COLUMN IF NOT EXISTS`` (idempotent).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import FeatureProvider

logger = logging.getLogger(__name__)


def migrate_plugin_columns(
    client: object,
    providers: list[FeatureProvider],
) -> None:
    """Add plugin feature columns to ClickHouse range_bars table.

    Uses ``ADD COLUMN IF NOT EXISTS`` for idempotent execution.
    All plugin columns are ``Nullable(Float64)`` — same type as
    inter-bar features.

    Parameters
    ----------
    client : clickhouse_connect.driver.Client
        Active ClickHouse client connection.
    providers : list[FeatureProvider]
        Discovered providers whose columns need to exist in the schema.
    """
    for provider in providers:
        for col in provider.columns:
            try:
                client.command(
                    "ALTER TABLE rangebar_cache.range_bars "
                    f"ADD COLUMN IF NOT EXISTS `{col}` Nullable(Float64) DEFAULT NULL"
                )
                logger.debug(
                    "Ensured column %s exists (provider: %s)", col, provider.name,
                )
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "Failed to add column %s for provider %s: %s",
                    col,
                    provider.name,
                    e,
                )
