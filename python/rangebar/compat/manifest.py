"""Feature manifest registry — parses Rust-embedded TOML SSoT (Issue #95).

The feature manifest lives in `crates/rangebar-core/data/feature_manifest.toml`,
is compiled into the Rust binary via `include_str!()`, and exposed to Python
through the `_core.get_feature_manifest_raw()` PyO3 function.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class FeatureGroup(Enum):
    """Feature group classification."""

    MICROSTRUCTURE = "microstructure"
    INTER_BAR = "inter_bar"
    INTRA_BAR = "intra_bar"


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata for a single microstructure feature."""

    name: str
    group: FeatureGroup
    description: str
    units: str
    nullable: bool
    category: str
    tier: int
    formula: str = ""
    value_range: tuple[float | None, float | None] = (None, None)
    min_trades: int | None = None
    interpretation: str = ""
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["group"] = self.group.value
        return d


def _parse_range(raw: list[Any]) -> tuple[float | None, float | None]:
    """Parse TOML range like [0.0, "inf"] to (0.0, None)."""
    def _val(v: Any) -> float | None:
        if isinstance(v, str):
            if v == "inf":
                return None
            if v == "-inf":
                return None
        return float(v)

    if len(raw) != 2:
        return (None, None)
    return (_val(raw[0]), _val(raw[1]))


def _parse_feature(name: str, data: dict[str, Any]) -> FeatureMetadata:
    """Parse a single [features.X] TOML section into FeatureMetadata."""
    raw_range = data.get("range", [])
    return FeatureMetadata(
        name=name,
        group=FeatureGroup(data["group"]),
        description=data["description"],
        units=data["units"],
        nullable=data["nullable"],
        category=data["category"],
        tier=data["tier"],
        formula=data.get("formula", ""),
        value_range=_parse_range(raw_range) if raw_range else (None, None),
        min_trades=data.get("min_trades"),
        interpretation=data.get("interpretation", ""),
        reference=data.get("reference"),
    )


class FeatureRegistry:
    """Registry of all microstructure features, parsed from Rust-embedded TOML.

    Lazy singleton — instantiated on first call to `get_feature_manifest()`.
    """

    def __init__(self, features: dict[str, FeatureMetadata], schema_version: int) -> None:
        self._features = features
        self._schema_version = schema_version

    @property
    def schema_version(self) -> int:
        """TOML schema version."""
        return self._schema_version

    def get(self, name: str) -> FeatureMetadata | None:
        """Get feature metadata by name."""
        return self._features.get(name)

    def all_features(self) -> list[FeatureMetadata]:
        """All features in definition order."""
        return list(self._features.values())

    def by_group(self, group: FeatureGroup) -> list[FeatureMetadata]:
        """Filter features by group."""
        return [f for f in self._features.values() if f.group == group]

    def by_category(self, category: str) -> list[FeatureMetadata]:
        """Filter features by category."""
        return [f for f in self._features.values() if f.category == category]

    def column_names(self, group: FeatureGroup | None = None) -> list[str]:
        """Get feature column names, optionally filtered by group."""
        if group is None:
            return list(self._features.keys())
        return [f.name for f in self._features.values() if f.group == group]

    def to_dict(self) -> dict[str, Any]:
        """Convert entire registry to JSON-serializable dict."""
        return {
            "schema_version": self._schema_version,
            "features": {name: f.to_dict() for name, f in self._features.items()},
        }

    def to_json(self) -> str:
        """Serialize registry to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# Module-level singleton
_registry: FeatureRegistry | None = None


def get_feature_manifest() -> FeatureRegistry:
    """Return global feature registry (lazy singleton, parsed from Rust-embedded TOML)."""
    global _registry
    if _registry is None:
        from rangebar._core import get_feature_manifest_raw

        raw = get_feature_manifest_raw()
        data = tomllib.loads(raw)

        features: dict[str, FeatureMetadata] = {}
        for name, fdata in data["features"].items():
            features[name] = _parse_feature(name, fdata)

        _registry = FeatureRegistry(
            features=features,
            schema_version=data["meta"]["schema_version"],
        )
    return _registry


def get_feature_groups() -> dict[str, list[str]]:
    """Feature names grouped by FeatureGroup."""
    registry = get_feature_manifest()
    result: dict[str, list[str]] = {}
    for group in FeatureGroup:
        result[group.value] = registry.column_names(group)
    return result
