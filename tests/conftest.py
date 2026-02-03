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


@pytest.fixture(autouse=True, scope="session")
def setup_threshold_env_vars():
    """Set up threshold environment variables for tests (Issue #62).

    This fixture ensures that the SSoT threshold configuration is set
    for all tests, avoiding fallback warnings and ensuring consistent
    behavior across test runs.

    Note: For tests, we use lower thresholds than production to allow
    testing with common presets (micro, tight, standard, medium, wide).
    The minimum is set to 1 dbps for all asset classes during testing.
    """
    # Store original values
    original_crypto = os.environ.get("RANGEBAR_CRYPTO_MIN_THRESHOLD")
    original_forex = os.environ.get("RANGEBAR_FOREX_MIN_THRESHOLD")
    original_equities = os.environ.get("RANGEBAR_EQUITIES_MIN_THRESHOLD")
    original_unknown = os.environ.get("RANGEBAR_UNKNOWN_MIN_THRESHOLD")

    # Set test values: allow all threshold presets for testing
    # Production uses 1000/50/100/1, tests use 1/1/1/1 for flexibility
    os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "1"
    os.environ["RANGEBAR_FOREX_MIN_THRESHOLD"] = "1"
    os.environ["RANGEBAR_EQUITIES_MIN_THRESHOLD"] = "1"
    os.environ["RANGEBAR_UNKNOWN_MIN_THRESHOLD"] = "1"

    # Clear the threshold cache to pick up new values
    try:
        from rangebar.threshold import clear_threshold_cache

        clear_threshold_cache()
    except ImportError:
        pass

    yield

    # Restore original values
    if original_crypto is not None:
        os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = original_crypto
    elif "RANGEBAR_CRYPTO_MIN_THRESHOLD" in os.environ:
        del os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"]

    if original_forex is not None:
        os.environ["RANGEBAR_FOREX_MIN_THRESHOLD"] = original_forex
    elif "RANGEBAR_FOREX_MIN_THRESHOLD" in os.environ:
        del os.environ["RANGEBAR_FOREX_MIN_THRESHOLD"]

    if original_equities is not None:
        os.environ["RANGEBAR_EQUITIES_MIN_THRESHOLD"] = original_equities
    elif "RANGEBAR_EQUITIES_MIN_THRESHOLD" in os.environ:
        del os.environ["RANGEBAR_EQUITIES_MIN_THRESHOLD"]

    if original_unknown is not None:
        os.environ["RANGEBAR_UNKNOWN_MIN_THRESHOLD"] = original_unknown
    elif "RANGEBAR_UNKNOWN_MIN_THRESHOLD" in os.environ:
        del os.environ["RANGEBAR_UNKNOWN_MIN_THRESHOLD"]

    # Clear cache again to restore original behavior
    try:
        from rangebar.threshold import clear_threshold_cache

        clear_threshold_cache()
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
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
