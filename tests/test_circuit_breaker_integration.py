"""Tests for Issue #108: Circuit breaker integration.

Validates that the circuit breaker protects fatal_cache_write() and
transitions through CLOSED → OPEN → HALF_OPEN → CLOSED states.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rangebar.resilience import CircuitBreaker, CircuitOpenError, CircuitState


def test_circuit_opens_after_threshold_failures():
    """After failure_threshold consecutive failures, circuit opens."""
    cb = CircuitBreaker(
        name="test_write",
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=ValueError,
    )

    def failing_func():
        msg = "write failed"
        raise ValueError(msg)

    # First 3 calls should raise ValueError (circuit still closed)
    for i in range(3):
        with pytest.raises(ValueError):
            cb.call(failing_func)

    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 3

    # 4th call should raise CircuitOpenError (not ValueError)
    with pytest.raises(CircuitOpenError, match="test_write"):
        cb.call(failing_func)


def test_circuit_half_open_recovery():
    """After recovery_timeout, circuit transitions to HALF_OPEN and recovers."""
    from datetime import UTC, datetime, timedelta

    cb = CircuitBreaker(
        name="test_recovery",
        failure_threshold=2,
        recovery_timeout=1,  # 1 second
        expected_exception=ValueError,
    )

    def failing_func():
        msg = "fail"
        raise ValueError(msg)

    # Open the circuit
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(failing_func)
    assert cb.state == CircuitState.OPEN

    # Simulate time passing beyond recovery_timeout
    cb.last_failure_time = datetime.now(tz=UTC) - timedelta(seconds=2)

    # Next call should attempt (HALF_OPEN) — if it succeeds, circuit closes
    result = cb.call(lambda: 42)
    assert result == 42
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_callback_on_circuit_open():
    """Registered callback fires when circuit opens."""
    cb = CircuitBreaker(
        name="test_callback",
        failure_threshold=2,
        recovery_timeout=30,
        expected_exception=ValueError,
    )

    callback_states: list[CircuitState] = []
    cb.add_callback(lambda state: callback_states.append(state))

    def failing():
        msg = "fail"
        raise ValueError(msg)

    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(failing)

    assert CircuitState.OPEN in callback_states


def test_fatal_write_breaker_exists():
    """Module-level circuit breaker is configured for fatal_cache_write."""
    from rangebar.orchestration.range_bars_cache import get_fatal_write_breaker

    breaker = get_fatal_write_breaker()
    assert breaker.name == "fatal_cache_write"
    assert breaker.failure_threshold == 3
    assert breaker.recovery_timeout == 60


def test_fatal_cache_write_uses_circuit_breaker():
    """fatal_cache_write() routes through the circuit breaker."""
    from rangebar.orchestration.range_bars_cache import (
        _fatal_write_breaker,
        fatal_cache_write,
    )

    # Reset breaker state for test isolation
    _fatal_write_breaker.state = CircuitState.CLOSED
    _fatal_write_breaker.failure_count = 0
    _fatal_write_breaker.last_failure_time = None

    # Mock the inner write to simulate CacheWriteError
    import pandas as pd
    from rangebar.exceptions import CacheWriteError

    dummy_df = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10.0]})

    with patch(
        "rangebar.orchestration.range_bars_cache._do_fatal_cache_write",
        side_effect=CacheWriteError("connection dropped", symbol="TEST", operation="test"),
    ):
        # 3 failures should open the circuit
        for _ in range(3):
            with pytest.raises(CacheWriteError):
                fatal_cache_write(dummy_df, "TEST", 250, "year")

        assert _fatal_write_breaker.state == CircuitState.OPEN

        # Next call should raise CircuitOpenError, not CacheWriteError
        with pytest.raises(CircuitOpenError):
            fatal_cache_write(dummy_df, "TEST", 250, "year")

    # Clean up
    _fatal_write_breaker.state = CircuitState.CLOSED
    _fatal_write_breaker.failure_count = 0
    _fatal_write_breaker.last_failure_time = None
