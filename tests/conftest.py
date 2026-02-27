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


def _clickhouse_has_migrated_schema() -> bool:
    """Check if ClickHouse has post-migration schema (close_time_ms)."""
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            result = cache.client.query(
                "SELECT name FROM system.columns "
                "WHERE database='rangebar_cache' AND table='range_bars' "
                "AND name='close_time_ms'"
            )
            return bool(result.result_rows)
    except (ImportError, OSError, RuntimeError):
        return False


# Cache the result at session level (avoid repeated ClickHouse queries)
_MIGRATED_SCHEMA = _clickhouse_has_migrated_schema()


def pytest_addoption(parser):
    """Add --run-slow CLI option for explicitly running slow tests."""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run tests marked @pytest.mark.slow (network-fetching E2E tests)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip ClickHouse tests (pre-migration) and slow tests (unless --run-slow)."""
    # Skip ClickHouse tests when schema not migrated
    if not _MIGRATED_SCHEMA:
        ch_skip = pytest.mark.skip(
            reason="ClickHouse schema not yet migrated: timestamp_ms → close_time_ms"
        )
        for item in items:
            if "clickhouse" in item.keywords:
                item.add_marker(ch_skip)

    # Skip slow tests unless --run-slow or -m "slow" explicitly requested
    if not config.getoption("--run-slow"):
        marker_expr = config.getoption("-m", default="")
        if "slow" not in marker_expr:
            slow_skip = pytest.mark.skip(reason="Slow test: use --run-slow to include")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(slow_skip)
