"""Event hook system for rangebar-py operations.

This module provides a publish-subscribe event system for monitoring
long-running cache operations, validation results, and population progress.

Usage
-----
>>> from rangebar.hooks import register_hook, HookEvent
>>>
>>> def my_callback(payload):
...     print(f"Event: {payload.event.value}, Symbol: {payload.symbol}")
...
>>> register_hook(HookEvent.CACHE_WRITE_COMPLETE, my_callback)
>>> # Later, when bars are cached:
>>> # Event: cache_write_complete, Symbol: BTCUSDT

Events
------
- CACHE_WRITE_START: Cache write operation started
- CACHE_WRITE_COMPLETE: Cache write operation completed successfully
- CACHE_WRITE_FAILED: Cache write operation failed
- VALIDATION_COMPLETE: Post-storage validation passed
- VALIDATION_FAILED: Post-storage validation failed
- CHECKPOINT_SAVED: Resumable population checkpoint saved
- POPULATION_COMPLETE: Cache population completed successfully
- POPULATION_FAILED: Cache population failed
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Events that can trigger hook callbacks.

    Attributes
    ----------
    CACHE_WRITE_START : str
        Emitted when cache write operation begins.
    CACHE_WRITE_COMPLETE : str
        Emitted when cache write completes successfully.
    CACHE_WRITE_FAILED : str
        Emitted when cache write fails.
    VALIDATION_COMPLETE : str
        Emitted when post-storage validation passes.
    VALIDATION_FAILED : str
        Emitted when post-storage validation fails.
    CHECKPOINT_SAVED : str
        Emitted when a population checkpoint is saved.
    POPULATION_COMPLETE : str
        Emitted when cache population completes.
    POPULATION_FAILED : str
        Emitted when cache population fails.
    """

    CACHE_WRITE_START = "cache_write_start"
    CACHE_WRITE_COMPLETE = "cache_write_complete"
    CACHE_WRITE_FAILED = "cache_write_failed"
    VALIDATION_COMPLETE = "validation_complete"
    VALIDATION_FAILED = "validation_failed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    POPULATION_COMPLETE = "population_complete"
    POPULATION_FAILED = "population_failed"


@dataclass
class HookPayload:
    """Payload delivered to hook callbacks.

    Attributes
    ----------
    event : HookEvent
        The event type that triggered this callback.
    symbol : str
        Trading symbol involved (e.g., "BTCUSDT").
    timestamp : datetime
        When the event occurred (UTC).
    details : dict
        Additional event-specific information.
    is_failure : bool
        True if this event represents a failure.
    """

    event: HookEvent
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)
    is_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary for serialization."""
        d = asdict(self)
        d["event"] = self.event.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        """Convert payload to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# Global hook registry
_hooks: dict[HookEvent, list[Callable[[HookPayload], None]]] = defaultdict(list)


def register_hook(
    event: HookEvent,
    callback: Callable[[HookPayload], None],
) -> None:
    """Register a callback for a specific event.

    Parameters
    ----------
    event : HookEvent
        The event to subscribe to.
    callback : Callable[[HookPayload], None]
        Function to call when the event occurs. Receives a HookPayload.

    Examples
    --------
    >>> def log_completion(payload):
    ...     print(f"Completed: {payload.symbol} at {payload.timestamp}")
    ...
    >>> register_hook(HookEvent.CACHE_WRITE_COMPLETE, log_completion)
    """
    _hooks[event].append(callback)
    logger.debug("Registered hook for %s: %s", event.value, callback.__name__)


def unregister_hook(
    event: HookEvent,
    callback: Callable[[HookPayload], None],
) -> bool:
    """Unregister a callback for a specific event.

    Parameters
    ----------
    event : HookEvent
        The event to unsubscribe from.
    callback : Callable[[HookPayload], None]
        The callback to remove.

    Returns
    -------
    bool
        True if the callback was found and removed, False otherwise.
    """
    if callback in _hooks[event]:
        _hooks[event].remove(callback)
        logger.debug("Unregistered hook for %s: %s", event.value, callback.__name__)
        return True
    return False


def clear_hooks(event: HookEvent | None = None) -> None:
    """Clear all hooks for a specific event or all events.

    Parameters
    ----------
    event : HookEvent | None
        Event to clear hooks for. If None, clears all hooks.
    """
    if event is None:
        _hooks.clear()
        logger.debug("Cleared all hooks")
    else:
        _hooks[event].clear()
        logger.debug("Cleared hooks for %s", event.value)


def emit_hook(
    event: HookEvent,
    symbol: str,
    **details: Any,
) -> None:
    """Emit an event to all registered callbacks.

    Parameters
    ----------
    event : HookEvent
        The event type to emit.
    symbol : str
        Trading symbol involved.
    **details
        Additional event-specific information.

    Notes
    -----
    Callback exceptions are caught and logged but do not propagate.
    This ensures one failing callback doesn't break the main operation.

    Examples
    --------
    >>> emit_hook(
    ...     HookEvent.CACHE_WRITE_COMPLETE,
    ...     symbol="BTCUSDT",
    ...     bars_written=1500,
    ...     threshold_bps=250,
    ... )
    """
    # Add memory snapshot to all hook payloads (Issue #49 T2.3)
    try:
        from rangebar.resource_guard import get_memory_info

        mem = get_memory_info()
        details["memory_rss_mb"] = mem.process_rss_mb
        details["memory_pct"] = round(mem.usage_pct, 3)
    except Exception:
        logger.debug("Memory snapshot unavailable for hook payload", exc_info=True)

    is_failure = "FAILED" in event.name
    payload = HookPayload(
        event=event,
        symbol=symbol,
        timestamp=datetime.now(UTC),
        details=details,
        is_failure=is_failure,
    )

    callbacks = _hooks.get(event, [])
    if not callbacks:
        logger.debug("No hooks registered for %s", event.value)
        return

    logger.debug(
        "Emitting %s for %s to %d callback(s)",
        event.value,
        symbol,
        len(callbacks),
    )

    for callback in callbacks:
        try:
            callback(payload)
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            # Log but don't propagate - callbacks shouldn't break main flow
            logger.warning(
                "Hook callback %s failed for %s: %s",
                callback.__name__,
                event.value,
                e,
            )


def register_for_failures(
    callback: Callable[[HookPayload], None],
) -> None:
    """Register a callback for all failure events.

    Convenience function to subscribe to all *_FAILED events.

    Parameters
    ----------
    callback : Callable[[HookPayload], None]
        Function to call when any failure event occurs.

    Examples
    --------
    >>> def alert_on_failure(payload):
    ...     send_slack_alert(f"FAILURE: {payload.event.value}")
    ...
    >>> register_for_failures(alert_on_failure)
    """
    for event in HookEvent:
        if "FAILED" in event.name:
            register_hook(event, callback)


def register_for_all(
    callback: Callable[[HookPayload], None],
) -> None:
    """Register a callback for all events.

    Parameters
    ----------
    callback : Callable[[HookPayload], None]
        Function to call when any event occurs.

    Examples
    --------
    >>> def log_all(payload):
    ...     print(f"{payload.event.value}: {payload.symbol}")
    ...
    >>> register_for_all(log_all)
    """
    for event in HookEvent:
        register_hook(event, callback)


__all__ = [
    "HookEvent",
    "HookPayload",
    "clear_hooks",
    "emit_hook",
    "register_for_all",
    "register_for_failures",
    "register_hook",
    "unregister_hook",
]
