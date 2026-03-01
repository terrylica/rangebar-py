#!/usr/bin/env python3
"""Test that all examples run without errors.

ADR-003: Testing Strategy with Real Binance Data
Automated verification that examples execute successfully.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

EXAMPLES = [
    "basic_usage.py",
    "validate_output.py",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    """Test that example runs without errors."""
    example_path = EXAMPLES_DIR / example
    assert example_path.exists(), f"Example not found: {example}"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert (
        result.returncode == 0
    ), f"Example failed: {example}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
