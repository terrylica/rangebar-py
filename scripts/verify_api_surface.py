#!/usr/bin/env python3
"""
Verify PyO3 API surface hasn't changed.

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/94
Part of src/lib.rs refactoring: splitting monolithic PyO3 bindings into domain modules.

Usage:
  python scripts/verify_api_surface.py --snapshot  # Create baseline
  python scripts/verify_api_surface.py --verify    # Compare against baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def get_api_surface() -> dict[str, Any]:
    """Extract API surface from _core module."""
    try:
        from rangebar._core import (
            PositionVerification,
            PyRangeBarProcessor,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import rangebar._core: {e}")
        print("Did you run 'maturin develop'?")
        sys.exit(1)

    surface = {}

    # PyRangeBarProcessor
    surface["PyRangeBarProcessor"] = {
        "type": "class",
        "methods": [m for m in dir(PyRangeBarProcessor) if not m.startswith("_")],
        "properties": {
            m: type(getattr(PyRangeBarProcessor, m)).__name__
            for m in dir(PyRangeBarProcessor)
            if not m.startswith("_") and not callable(getattr(PyRangeBarProcessor, m))
        },
    }

    # PositionVerification
    surface["PositionVerification"] = {
        "type": "class",
        "methods": [m for m in dir(PositionVerification) if not m.startswith("_")],
        "properties": {
            m: type(getattr(PositionVerification, m)).__name__
            for m in dir(PositionVerification)
            if not m.startswith("_") and not callable(getattr(PositionVerification, m))
        },
    }

    return surface


def main():
    parser = argparse.ArgumentParser(
        description="Verify PyO3 API surface"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create baseline snapshot"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against baseline"
    )

    args = parser.parse_args()

    snapshot_file = Path(__file__).parent / ".api_surface_baseline.json"

    surface = get_api_surface()

    if args.snapshot:
        with snapshot_file.open("w") as f:
            json.dump(surface, f, indent=2, sort_keys=True)
        print(f"✓ API surface snapshot saved to {snapshot_file}")
        print(f"  Classes: {len(surface)}")
        for cls, info in surface.items():
            methods = len(info.get("methods", []))
            fields = len(info.get("fields", []))
            print(f"    {cls}: {methods} methods, {fields} fields")
        return 0

    if args.verify:
        if not snapshot_file.exists():
            print(f"ERROR: No baseline found at {snapshot_file}")
            print("Run with --snapshot first to create baseline")
            return 1

        with snapshot_file.open() as f:
            baseline = json.load(f)

        # Check for regressions
        errors = []

        for cls, baseline_info in baseline.items():
            if cls not in surface:
                errors.append(f"Missing class: {cls}")
                continue

            current_info = surface[cls]

            # Check methods
            baseline_methods = set(baseline_info.get("methods", []))
            current_methods = set(current_info.get("methods", []))

            missing_methods = baseline_methods - current_methods
            if missing_methods:
                errors.append(
                    f"{cls}: Missing methods: {missing_methods}"
                )

            # Check fields
            baseline_fields = set(baseline_info.get("fields", []))
            current_fields = set(current_info.get("fields", []))

            missing_fields = baseline_fields - current_fields
            if missing_fields:
                errors.append(
                    f"{cls}: Missing fields: {missing_fields}"
                )

        if errors:
            print("ERROR: API surface regressions detected:")
            for error in errors:
                print(f"  ✗ {error}")
            return 1
        print("✓ API surface matches baseline")
        return 0

    print("ERROR: Specify --snapshot or --verify")
    return 1


if __name__ == "__main__":
    sys.exit(main())
