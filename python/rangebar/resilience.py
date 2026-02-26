"""Circuit breaker pattern for cache write resilience.

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/109
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""


class CircuitBreaker:
    """Circuit breaker for resilient cache writes."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[CircuitState], None]] = []

    def add_callback(
        self, callback: Callable[[CircuitState], None],
    ) -> None:
        """Register a state-change callback."""
        self._callbacks.append(callback)

    def call(
        self, func: Callable[..., object],
        *args: object, **kwargs: object,
    ) -> object:
        """Execute func through the circuit breaker."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    msg = f"Circuit {self.name} is OPEN"
                    raise CircuitOpenError(msg)

        try:
            result = func(*args, **kwargs)
        except self.expected_exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now(tz=UTC) - self.last_failure_time
        return elapsed >= timedelta(seconds=self.recovery_timeout)

    def _on_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                for cb in self._callbacks:
                    cb(CircuitState.CLOSED)

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(tz=UTC)
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                for cb in self._callbacks:
                    cb(CircuitState.OPEN)
