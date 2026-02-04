"""Progress tracking for long-running cache operations (Issue #70).

Implements the ResumableObserver pattern from Gemini 3 Pro Deep Research:
- Wrapped iterator with tqdm for console progress bars
- Loguru dual-sink for structured NDJSON + console logging
- Time-based throttling via tqdm's mininterval
- Resumability support via initial parameter

Usage
-----
>>> from rangebar.progress import ResumableObserver
>>> dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
>>> observer = ResumableObserver(
...     total=len(dates),
...     desc="BTCUSDT",
...     unit="days",
...     verbose=True,
... )
>>> for date in observer.wrap(dates):
...     process_date(date)
...     observer.log_event("day_complete", date=date)

References
----------
- Research: docs/research/2026-02-04-progress-logging-patterns-gemini-3-pro.md
- Issue: https://github.com/terrylica/rangebar-py/issues/70
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

T = TypeVar("T")


@dataclass
class ResumableObserver:
    """Progress observer with tqdm + loguru dual-sink.

    Wraps an iterator to provide:
    - Console progress bar (tqdm) with ETA estimation
    - Structured NDJSON logging (loguru) with throttling
    - Resumability support via initial parameter

    Parameters
    ----------
    total : int
        Total number of items to process.
    desc : str
        Description prefix for progress bar (e.g., symbol name).
    unit : str
        Unit name for items (default: "days").
    initial : int
        Starting count for resumed operations (default: 0).
    verbose : bool
        Enable progress output (default: True).
    log_interval_sec : float
        Minimum seconds between NDJSON log entries (default: 5.0).
    mininterval : float
        Minimum seconds between tqdm updates (default: 0.5).

    Attributes
    ----------
    count : int
        Current item count.
    start_time : datetime
        When processing started.
    pbar : tqdm | None
        The tqdm progress bar instance (None if verbose=False).

    Examples
    --------
    Basic usage:

    >>> observer = ResumableObserver(total=100, desc="BTCUSDT", verbose=True)
    >>> for item in observer.wrap(items):
    ...     process(item)

    Resumed operation:

    >>> observer = ResumableObserver(
    ...     total=100,
    ...     desc="BTCUSDT",
    ...     initial=45,  # Resume from item 45
    ...     verbose=True,
    ... )
    >>> for item in observer.wrap(remaining_items):
    ...     process(item)
    """

    total: int
    desc: str = ""
    unit: str = "days"
    initial: int = 0
    verbose: bool = True
    log_interval_sec: float = 5.0
    mininterval: float = 0.5

    # Runtime state (not constructor params)
    count: int = field(default=0, init=False)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    pbar: TqdmType | None = field(default=None, init=False)
    _last_log_time: float = field(default=0.0, init=False)
    _logger: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize tqdm and loguru after dataclass init."""
        self.count = self.initial
        self.start_time = datetime.now(UTC)

        if self.verbose:
            self._init_tqdm()
            self._init_logger()

    def _init_tqdm(self) -> None:
        """Initialize tqdm progress bar."""
        from tqdm import tqdm

        self.pbar = tqdm(
            total=self.total,
            initial=self.initial,
            desc=self.desc,
            unit=self.unit,
            mininterval=self.mininterval,
            file=sys.stderr,
            leave=True,
            dynamic_ncols=True,
        )

    def _init_logger(self) -> None:
        """Initialize loguru with tqdm-safe console sink."""
        try:
            from loguru import logger
            from tqdm import tqdm

            # Create a new logger instance bound with context
            self._logger = logger.bind(
                component="progress",
                symbol=self.desc,
                total=self.total,
            )

            # Add tqdm-safe console sink (routes through tqdm.write)
            # Only add if not already configured
            if not hasattr(self, "_sink_id"):
                self._sink_id = logger.add(
                    lambda msg: tqdm.write(msg, end="", file=sys.stderr),
                    format="{time:HH:mm:ss} | {level: <8} | {message}",
                    level="INFO",
                    filter=lambda record: record["extra"].get("component") == "progress",
                )
        except ImportError:
            self._logger = None

    def wrap(self, iterable: Iterator[T]) -> Iterator[T]:
        """Wrap an iterator to track progress.

        Parameters
        ----------
        iterable : Iterator[T]
            The iterator to wrap.

        Yields
        ------
        T
            Items from the original iterator.

        Examples
        --------
        >>> observer = ResumableObserver(total=10, desc="Processing")
        >>> for item in observer.wrap(items):
        ...     process(item)
        """
        for item in iterable:
            yield item
            self.update(1)

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Parameters
        ----------
        n : int
            Number of items completed (default: 1).
        """
        self.count += n

        if self.pbar is not None:
            self.pbar.update(n)

    def log_event(
        self,
        event_type: str,
        **details: Any,
    ) -> None:
        """Log a structured event with throttling.

        Events are logged to NDJSON at most once per log_interval_sec.
        This prevents log spam during high-frequency updates.

        Parameters
        ----------
        event_type : str
            Type of event (e.g., "day_complete", "checkpoint_saved").
        **details
            Additional event-specific fields.

        Examples
        --------
        >>> observer.log_event(
        ...     "day_complete",
        ...     date="2024-01-15",
        ...     bars_today=142,
        ... )
        """
        import time

        current_time = time.time()

        # Throttle logging (skip if within interval)
        if current_time - self._last_log_time < self.log_interval_sec:
            return

        self._last_log_time = current_time

        if self._logger is not None:
            elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
            progress_pct = (self.count / self.total * 100) if self.total > 0 else 0

            self._logger.bind(
                event_type=event_type,
                count=self.count,
                total=self.total,
                progress_pct=round(progress_pct, 1),
                elapsed_sec=round(elapsed, 1),
                **details,
            ).info(f"{event_type}: {self.desc} [{self.count}/{self.total}]")

    def set_postfix(self, **kwargs: Any) -> None:
        """Update tqdm postfix with current stats.

        Parameters
        ----------
        **kwargs
            Key-value pairs to display in postfix.

        Examples
        --------
        >>> observer.set_postfix(bars=1500, rate="2.3/s")
        """
        if self.pbar is not None:
            self.pbar.set_postfix(**kwargs)

    def close(self) -> None:
        """Close the progress bar and clean up resources."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        # Log final summary
        if self._logger is not None:
            elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
            rate = self.count / elapsed if elapsed > 0 else 0

            self._logger.bind(
                event_type="progress_complete",
                count=self.count,
                total=self.total,
                elapsed_sec=round(elapsed, 1),
                rate=round(rate, 2),
            ).info(f"Complete: {self.desc} [{self.count}/{self.total}] in {elapsed:.1f}s")

    def __enter__(self) -> ResumableObserver:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()


__all__ = ["ResumableObserver"]
