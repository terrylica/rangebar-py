"""Validate feature_manifest.toml stays in sync with constants.py (Issue #95).

Usage: mise run validate:metadata
"""

import sys
import tomllib
from pathlib import Path

MANIFEST_PATH = Path("crates/rangebar-core/data/feature_manifest.toml")

REQUIRED_FIELDS = {"group", "description", "units", "nullable", "category", "tier"}


def main():
    # Load manifest
    raw = MANIFEST_PATH.read_text()
    data = tomllib.loads(raw)
    features = data["features"]

    # Load constants.py column tuples
    from rangebar.constants import (
        INTER_BAR_FEATURE_COLUMNS,
        INTRA_BAR_FEATURE_COLUMNS,
        MICROSTRUCTURE_COLUMNS,
    )

    manifest_names = set(features.keys())
    constants_names = set(MICROSTRUCTURE_COLUMNS) | set(INTER_BAR_FEATURE_COLUMNS) | set(INTRA_BAR_FEATURE_COLUMNS)

    # Check manifest covers constants
    missing_from_manifest = constants_names - manifest_names
    if missing_from_manifest:
        print(f"FAIL: Constants columns missing from manifest: {sorted(missing_from_manifest)}")
        sys.exit(1)

    # Check constants cover manifest
    extra_in_manifest = manifest_names - constants_names
    if extra_in_manifest:
        print(f"FAIL: Manifest has features not in constants: {sorted(extra_in_manifest)}")
        sys.exit(1)

    # Validate required fields
    errors = []
    for name, fdata in features.items():
        missing = REQUIRED_FIELDS - set(fdata.keys())
        if missing:
            errors.append(f"  {name}: missing {sorted(missing)}")

    if errors:
        print("FAIL: Features missing required fields:\n" + "\n".join(errors))
        sys.exit(1)

    # Group counts
    groups = {}
    for fdata in features.values():
        g = fdata["group"]
        groups[g] = groups.get(g, 0) + 1

    print(f"PASS: {len(features)} features validated")
    for g, count in sorted(groups.items()):
        print(f"  {g}: {count}")


if __name__ == "__main__":
    main()
