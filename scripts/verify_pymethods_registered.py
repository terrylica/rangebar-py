#!/usr/bin/env python3
"""
Detect silent #[pymethods] failures in PyO3 bindings.

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/94

After refactoring src/lib.rs into modules, verify all #[pymethods] are properly
registered. PyO3 can silently fail to register methods if there are compilation issues
in separate modules.

Usage:
  python scripts/verify_pymethods_registered.py

This script checks that:
1. All expected classes are importable
2. All methods are callable (not silently unregistered)
3. Expected method signatures work
"""

import sys


def check_class_methods(cls: type, expected_methods: list[str]) -> list[str]:
    """Check that all expected methods/properties are registered on the class."""
    errors = []
    actual_attrs = {m for m in dir(cls) if not m.startswith("_")}

    for method in expected_methods:
        if method not in actual_attrs:
            errors.append(f"  ✗ {cls.__name__}.{method} not registered")
        else:
            # Verify it exists and is accessible
            try:
                getattr(cls, method)
            except (AttributeError, TypeError) as e:
                errors.append(f"  ✗ {cls.__name__}.{method} error: {e}")

    return errors


def main():
    try:
        from rangebar._core import (
            PositionVerification,
            PyRangeBarProcessor,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import rangebar._core: {e}")
        print("Did you run 'maturin develop'?")
        return 1

    errors = []

    # Expected methods for PyRangeBarProcessor
    processor_methods = [
        "process_trades",
        "create_checkpoint",
        "from_checkpoint",
        "enable_microstructure",
        "verify_position",
    ]

    # Expected methods for PositionVerification
    verification_methods = [
        "gap_details",
        "timestamp_gap_ms",
    ]

    print("Checking PyRangeBarProcessor methods...")
    errors.extend(check_class_methods(PyRangeBarProcessor, processor_methods))

    print("Checking PositionVerification methods...")
    errors.extend(check_class_methods(PositionVerification, verification_methods))

    # Try instantiating PyRangeBarProcessor to verify __init__ works
    print("Checking PyRangeBarProcessor instantiation...")
    try:
        processor = PyRangeBarProcessor(250, symbol="BTCUSDT")
        print("  ✓ PyRangeBarProcessor instantiated successfully")
        print(f"    threshold_decimal_bps: {processor.threshold_decimal_bps}")
        print(f"    symbol: {processor.symbol}")
    except (TypeError, ValueError, AttributeError) as e:
        errors.append(f"  ✗ Failed to instantiate PyRangeBarProcessor: {e}")

    if errors:
        print("\nERROR: #[pymethods] registration issues detected:")
        for error in errors:
            print(error)
        return 1
    print("\n✓ All #[pymethods] properly registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
