"""Pytest configuration and fixtures for rangebar tests.

Fast-only test suite: all tests must complete in < 30s total.
No ClickHouse, no network I/O, no slow markers.
"""

from __future__ import annotations

import gc
import os

import pytest


@pytest.fixture(autouse=True)
def gc_after_test():
    """Force GC after each test to prevent memory accumulation."""
    yield
    gc.collect()


@pytest.fixture(autouse=True, scope="session")
def setup_threshold_env_vars():
    """Set threshold env vars for tests (Issue #62).

    Production uses 1000/50/100/1, tests use 1/1/1/1 for flexibility.
    """
    originals = {}
    keys = [
        "RANGEBAR_CRYPTO_MIN_THRESHOLD",
        "RANGEBAR_FOREX_MIN_THRESHOLD",
        "RANGEBAR_EQUITIES_MIN_THRESHOLD",
        "RANGEBAR_UNKNOWN_MIN_THRESHOLD",
    ]
    for key in keys:
        originals[key] = os.environ.get(key)
        os.environ[key] = "1"

    try:
        from rangebar.threshold import clear_threshold_cache
        clear_threshold_cache()
    except ImportError:
        pass

    yield

    for key in keys:
        if originals[key] is not None:
            os.environ[key] = originals[key]
        elif key in os.environ:
            del os.environ[key]

    try:
        from rangebar.threshold import clear_threshold_cache
        clear_threshold_cache()
    except ImportError:
        pass
