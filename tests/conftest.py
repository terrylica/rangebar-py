"""Pytest configuration and fixtures for rangebar tests.

MEM-005: Memory management fixtures to prevent OOM in full test suite.
"""

from __future__ import annotations

import gc
import os

import pytest


@pytest.fixture(autouse=True)
def gc_after_test():
    """Force garbage collection after each test to free memory.

    This fixture is automatically applied to all tests (autouse=True).
    Prevents memory accumulation across tests that can cause OOM when
    running the full test suite.

    MEM-005: Single test can use 52 GB peak - GC between tests prevents
    accumulation that would exceed available memory.
    """
    yield
    gc.collect()


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Modify test collection to skip heavy tests when not in CI.

    Heavy E2E tests that use real market data are only run in CI to avoid
    excessive memory/network usage during local development.
    """
    if os.environ.get("CI") == "true":
        # In CI: run all tests
        return

    # Locally: skip tests marked as "e2e" or in e2e test files
    skip_heavy = pytest.mark.skip(
        reason="Heavy E2E tests only run in CI (set CI=true to run locally)"
    )

    for item in items:
        # Skip tests in e2e test files
        if (
            "test_e2e" in item.fspath.basename
            or "test_get_range_bars_e2e" in item.fspath.basename
        ) or "e2e" in [marker.name for marker in item.iter_markers()]:
            item.add_marker(skip_heavy)
