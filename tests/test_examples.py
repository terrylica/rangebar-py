#!/usr/bin/env python3
"""Test that all examples run without errors.

ADR-003: Testing Strategy with Real Binance Data
Automated verification that examples execute successfully.
"""

import pytest
import subprocess
import sys
from pathlib import Path

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

    # Run example
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Should not crash
    assert result.returncode == 0, f"Example failed: {example}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import backtesting"],
                   capture_output=True).returncode != 0,
    reason="backtesting.py not installed"
)
def test_backtesting_integration_example():
    """Test backtesting integration example (requires backtesting.py)."""
    example_path = EXAMPLES_DIR / "backtesting_integration.py"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Backtesting example failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_binance_csv_example_with_sample():
    """Test Binance CSV example with generated sample data."""
    example_path = EXAMPLES_DIR / "binance_csv_example.py"

    # Run without arguments (creates sample data)
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Binance CSV example failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
