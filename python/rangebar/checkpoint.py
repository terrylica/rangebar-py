"""Checkpoint system for resumable cache population (Issue #40).

Enables long-running cache population jobs to be resumed after interruption.
Uses atomic file writes and daily granularity for reliable state persistence.

Usage
-----
>>> from rangebar.checkpoint import populate_cache_resumable
>>> bars = populate_cache_resumable(
...     symbol="BTCUSDT",
...     start_date="2024-01-01",
...     end_date="2024-06-30",
... )
>>> print(f"Populated {bars} bars")

If interrupted, simply run again - it will resume from the last completed day.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from platformdirs import user_cache_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default checkpoint directory
_CHECKPOINT_DIR = Path(user_cache_dir("rangebar", "terrylica")) / "checkpoints"


@dataclass
class PopulationCheckpoint:
    """Checkpoint for resumable cache population.

    Tracks progress of a multi-day cache population job, allowing
    resumption after interruption.

    Attributes
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_bps : int
        Threshold in decimal basis points.
    start_date : str
        Original start date (YYYY-MM-DD).
    end_date : str
        Target end date (YYYY-MM-DD).
    last_completed_date : str
        Most recent successfully completed date.
    bars_written : int
        Total bars written so far.
    created_at : str
        ISO timestamp of checkpoint creation.
    updated_at : str
        ISO timestamp of last update.
    """

    symbol: str
    threshold_bps: int
    start_date: str
    end_date: str
    last_completed_date: str
    bars_written: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopulationCheckpoint:
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save checkpoint to file with atomic write.

        Uses tempfile + fsync + rename pattern for crash safety.

        Parameters
        ----------
        path : Path
            Path to save checkpoint file.
        """
        self.updated_at = datetime.now(UTC).isoformat()
        data = json.dumps(self.to_dict(), indent=2)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, fsync, rename
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".checkpoint_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (POSIX guarantees this is atomic)
            os.replace(temp_path, path)
            logger.debug("Saved checkpoint to %s", path)
        except (OSError, RuntimeError):
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> PopulationCheckpoint | None:
        """Load checkpoint from file.

        Parameters
        ----------
        path : Path
            Path to checkpoint file.

        Returns
        -------
        PopulationCheckpoint | None
            Loaded checkpoint, or None if file doesn't exist.
        """
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load checkpoint from %s: %s", path, e)
            return None


def _get_checkpoint_path(
    symbol: str,
    start_date: str,
    end_date: str,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Get the checkpoint file path for a population job.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.

    Returns
    -------
    Path
        Path to the checkpoint file.
    """
    if checkpoint_dir is None:
        checkpoint_dir = _CHECKPOINT_DIR

    # Create unique filename from job parameters
    filename = f"{symbol}_{start_date}_{end_date}.json"
    return checkpoint_dir / filename


def _date_range(start_date: str, end_date: str) -> Iterator[str]:
    """Generate dates from start to end (inclusive).

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Yields
    ------
    str
        Dates in YYYY-MM-DD format.
    """
    from datetime import timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
    end = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007

    current = start
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def populate_cache_resumable(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    threshold_decimal_bps: int = 250,
    checkpoint_dir: Path | None = None,
    notify: bool = True,
) -> int:
    """Populate cache for a date range with automatic checkpointing.

    This function fetches tick data and computes range bars day-by-day,
    saving progress after each day. If interrupted, simply run again
    with the same parameters to resume from the last completed day.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    threshold_decimal_bps : int
        Threshold in decimal basis points (default: 250 = 0.25%).
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.
    notify : bool
        Whether to emit hook events for progress tracking.

    Returns
    -------
    int
        Total number of bars written.

    Examples
    --------
    >>> bars = populate_cache_resumable("BTCUSDT", "2024-01-01", "2024-06-30")
    >>> print(f"Populated {bars} bars")
    14523

    >>> # If interrupted, just run again:
    >>> bars = populate_cache_resumable("BTCUSDT", "2024-01-01", "2024-06-30")
    >>> # Will resume from last completed day
    """
    from rangebar import get_range_bars
    from rangebar.hooks import HookEvent, emit_hook

    checkpoint_path = _get_checkpoint_path(symbol, start_date, end_date, checkpoint_dir)

    # Check for existing checkpoint
    checkpoint = PopulationCheckpoint.load(checkpoint_path)
    resume_date = start_date
    total_bars = 0

    if checkpoint:
        # Validate checkpoint matches our parameters
        if (
            checkpoint.symbol == symbol
            and checkpoint.threshold_bps == threshold_decimal_bps
            and checkpoint.start_date == start_date
            and checkpoint.end_date == end_date
        ):
            # Resume from day after last completed
            from datetime import timedelta

            last_completed = datetime.strptime(  # noqa: DTZ007
                checkpoint.last_completed_date, "%Y-%m-%d"
            )
            resume_date = (last_completed + timedelta(days=1)).strftime("%Y-%m-%d")
            total_bars = checkpoint.bars_written

            logger.info(
                "Resuming %s population from %s (%d bars already written)",
                symbol,
                resume_date,
                total_bars,
            )

            if notify:
                emit_hook(
                    HookEvent.CHECKPOINT_SAVED,
                    symbol=symbol,
                    action="resumed",
                    last_completed_date=checkpoint.last_completed_date,
                    bars_written=total_bars,
                )
        else:
            logger.warning("Checkpoint parameters don't match, starting fresh")
            checkpoint = None

    # Process day by day
    dates = list(_date_range(resume_date, end_date))
    total_days = len(dates)

    for i, date in enumerate(dates, 1):
        logger.info(
            "Processing %s [%d/%d]: %s",
            symbol,
            i,
            total_days,
            date,
        )

        try:
            # Fetch and compute range bars for this day
            df = get_range_bars(
                symbol,
                date,
                date,
                threshold_decimal_bps=threshold_decimal_bps,
                use_cache=True,
                fetch_if_missing=True,
            )

            bars_today = len(df) if df is not None else 0
            total_bars += bars_today

            logger.debug(
                "%s %s: %d bars (total: %d)",
                symbol,
                date,
                bars_today,
                total_bars,
            )

            # Save checkpoint after each successful day
            checkpoint = PopulationCheckpoint(
                symbol=symbol,
                threshold_bps=threshold_decimal_bps,
                start_date=start_date,
                end_date=end_date,
                last_completed_date=date,
                bars_written=total_bars,
                created_at=(
                    checkpoint.created_at
                    if checkpoint
                    else datetime.now(UTC).isoformat()
                ),
            )
            checkpoint.save(checkpoint_path)

            if notify:
                emit_hook(
                    HookEvent.CHECKPOINT_SAVED,
                    symbol=symbol,
                    date=date,
                    bars_today=bars_today,
                    total_bars=total_bars,
                    progress_pct=round(i / total_days * 100, 1),
                )

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception(
                "Failed to process %s for %s",
                symbol,
                date,
            )
            if notify:
                emit_hook(
                    HookEvent.POPULATION_FAILED,
                    symbol=symbol,
                    date=date,
                    error=str(exc),
                    total_bars=total_bars,
                )
            raise

    # Clean up checkpoint on success
    try:
        checkpoint_path.unlink()
        logger.debug("Removed checkpoint file after successful completion")
    except OSError:
        pass

    logger.info(
        "Population complete for %s: %d bars from %s to %s",
        symbol,
        total_bars,
        start_date,
        end_date,
    )

    if notify:
        emit_hook(
            HookEvent.POPULATION_COMPLETE,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
        )

    return total_bars


def list_checkpoints(
    checkpoint_dir: Path | None = None,
) -> list[PopulationCheckpoint]:
    """List all existing checkpoints.

    Parameters
    ----------
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.

    Returns
    -------
    list[PopulationCheckpoint]
        List of checkpoints found.
    """
    if checkpoint_dir is None:
        checkpoint_dir = _CHECKPOINT_DIR

    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for path in checkpoint_dir.glob("*.json"):
        checkpoint = PopulationCheckpoint.load(path)
        if checkpoint:
            checkpoints.append(checkpoint)

    return checkpoints


def clear_checkpoint(
    symbol: str,
    start_date: str,
    end_date: str,
    checkpoint_dir: Path | None = None,
) -> bool:
    """Clear a specific checkpoint.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.

    Returns
    -------
    bool
        True if checkpoint was found and removed.
    """
    checkpoint_path = _get_checkpoint_path(symbol, start_date, end_date, checkpoint_dir)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Cleared checkpoint: %s", checkpoint_path)
        return True

    return False


__all__ = [
    "PopulationCheckpoint",
    "clear_checkpoint",
    "list_checkpoints",
    "populate_cache_resumable",
]
