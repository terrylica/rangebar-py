# FILE-SIZE-OK
"""Checkpoint system for resumable cache population (Issue #40, Issue #69).

Enables long-running cache population jobs to be resumed after interruption.
Uses atomic file writes with bar-level checkpointing for accurate resume.

Issue #69 Enhancements:
- Bar-level resumability: Preserves incomplete bar state via processor checkpoints
- Hybrid storage: Local + ClickHouse for cross-machine resume
- force_refresh parameter: Wipe cache and checkpoint to start fresh

Usage
-----
>>> from rangebar.checkpoint import populate_cache_resumable
>>> bars = populate_cache_resumable(
...     symbol="BTCUSDT",
...     start_date="2024-01-01",
...     end_date="2024-06-30",
... )
>>> print(f"Populated {bars} bars")

If interrupted, simply run again - it will resume from the last checkpoint.
Use force_refresh=True to wipe cache and start over.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
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
    resumption after interruption with bar-level accuracy.

    Issue #69 Enhancements:
    - processor_checkpoint: Preserves incomplete bar state
    - last_trade_timestamp_ms: For mid-day resume filtering
    - include_microstructure: Track feature configuration
    - ouroboros_mode: Track reset mode for consistency

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
    processor_checkpoint : dict | None
        Serialized processor state (incomplete bar, defer_open).
        Enables bar-level resumability (Issue #69).
    last_trade_timestamp_ms : int | None
        Timestamp of last processed trade for mid-day resume.
    include_microstructure : bool
        Whether microstructure features are enabled.
    ouroboros_mode : str
        Ouroboros reset mode for consistency.
    """

    symbol: str
    threshold_bps: int
    start_date: str
    end_date: str
    last_completed_date: str
    bars_written: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    # Issue #69: Bar-level resumability
    processor_checkpoint: dict | None = None
    last_trade_timestamp_ms: int | None = None
    include_microstructure: bool = False
    ouroboros_mode: str = "year"
    # Issue #72: Full Audit Trail - agg_trade_id range in incomplete bar
    first_agg_trade_id_in_bar: int | None = None
    last_agg_trade_id_in_bar: int | None = None
    # Issue #111 (Ariadne): High-water mark for deterministic resume via fromId
    last_processed_agg_trade_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopulationCheckpoint:
        """Create from dictionary, ignoring unknown keys (e.g., provenance)."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        """Save checkpoint to file with atomic write.

        Uses tempfile + fsync + rename pattern for crash safety.

        Parameters
        ----------
        path : Path
            Path to save checkpoint file.
        """
        self.updated_at = datetime.now(UTC).isoformat()
        d = self.to_dict()
        d["provenance"] = _checkpoint_provenance()
        data = json.dumps(d, indent=2)

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
    threshold_decimal_bps: int,
    start_date: str,
    end_date: str,
    checkpoint_dir: Path | None = None,
    ouroboros_mode: str = "year",
) -> Path:
    """Get the checkpoint file path for a population job.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    threshold_decimal_bps : int
        Threshold in decimal basis points. Included in filename to prevent
        collisions when multiple thresholds run concurrently (Issue #84).
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.
    ouroboros_mode : str
        Ouroboros mode included in filename to prevent cross-mode
        checkpoint pollution (Issue #126).

    Returns
    -------
    Path
        Path to the checkpoint file.
    """
    if checkpoint_dir is None:
        checkpoint_dir = _CHECKPOINT_DIR

    # Issue #126: Mode in filename prevents cross-mode resume contamination
    filename = (
        f"{symbol}_{threshold_decimal_bps}_{start_date}_{end_date}"
        f"_{ouroboros_mode}.json"
    )
    return checkpoint_dir / filename


def _load_checkpoint_from_clickhouse(
    symbol: str,
    threshold_decimal_bps: int,
    start_date: str,
    end_date: str,
    ouroboros_mode: str = "year",
) -> PopulationCheckpoint | None:
    """Load checkpoint from ClickHouse for cross-machine resume.

    Issue #69: Hybrid checkpoint storage enables resume on different machines.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    ouroboros_mode : str
        Ouroboros reset mode filter (default: "year").
        Prevents cross-mode checkpoint pollution.

    Returns
    -------
    PopulationCheckpoint | None
        Loaded checkpoint, or None if not found.
    """
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            data = cache.load_checkpoint(
                symbol, threshold_decimal_bps, start_date, end_date,
                ouroboros_mode=ouroboros_mode,
            )
            if data is None:
                return None

            # Convert ClickHouse data to PopulationCheckpoint
            return PopulationCheckpoint(
                symbol=symbol,
                threshold_bps=threshold_decimal_bps,
                start_date=start_date,
                end_date=end_date,
                last_completed_date=data["last_completed_date"],
                bars_written=data["bars_written"],
                processor_checkpoint=json.loads(data["processor_checkpoint"])
                if data.get("processor_checkpoint")
                else None,
                last_trade_timestamp_ms=data.get("last_trade_timestamp_ms"),
                include_microstructure=data.get("include_microstructure", False),
                ouroboros_mode=data.get("ouroboros_mode", "year"),
                # Issue #72: Full Audit Trail
                first_agg_trade_id_in_bar=data.get("first_agg_trade_id_in_bar"),
                last_agg_trade_id_in_bar=data.get("last_agg_trade_id_in_bar"),
                # Issue #111 (Ariadne): High-water mark
                last_processed_agg_trade_id=data.get("last_processed_agg_trade_id"),
            )
    except (ImportError, ConnectionError, OSError, RuntimeError) as e:
        logger.debug("ClickHouse checkpoint load failed: %s", e)
        return None


def _save_checkpoint_to_clickhouse(checkpoint: PopulationCheckpoint) -> None:
    """Save checkpoint to ClickHouse for cross-machine resume.

    Issue #69: Hybrid checkpoint storage enables resume on different machines.

    Parameters
    ----------
    checkpoint : PopulationCheckpoint
        Checkpoint to save.
    """
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            cache.save_checkpoint(
                symbol=checkpoint.symbol,
                threshold_decimal_bps=checkpoint.threshold_bps,
                start_date=checkpoint.start_date,
                end_date=checkpoint.end_date,
                last_completed_date=checkpoint.last_completed_date,
                last_trade_timestamp_ms=checkpoint.last_trade_timestamp_ms,
                processor_checkpoint=json.dumps(checkpoint.processor_checkpoint)
                if checkpoint.processor_checkpoint
                else "",
                bars_written=checkpoint.bars_written,
                include_microstructure=checkpoint.include_microstructure,
                ouroboros_mode=checkpoint.ouroboros_mode,
                # Issue #72: Full Audit Trail
                first_agg_trade_id_in_bar=checkpoint.first_agg_trade_id_in_bar,
                last_agg_trade_id_in_bar=checkpoint.last_agg_trade_id_in_bar,
                # Issue #111 (Ariadne): High-water mark
                last_processed_agg_trade_id=checkpoint.last_processed_agg_trade_id,
            )
    except (ImportError, ConnectionError, OSError, RuntimeError) as e:
        logger.debug("ClickHouse checkpoint save failed (non-fatal): %s", e)


def _checkpoint_provenance(trace_id: str | None = None) -> dict:
    """Build provenance metadata for checkpoint files (Issue #97)."""
    import rangebar

    return {
        "service": "rangebar-py",
        "version": getattr(rangebar, "__version__", "unknown"),
        "git_sha": os.environ.get("RANGEBAR_GIT_SHA", "unknown"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_utc": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
    }


def _is_ouroboros_boundary(date_str: str, ouroboros_mode: str) -> bool:
    """Check if a date falls on an ouroboros reset boundary.

    Issue #97: Used by populate_cache_resumable() to reset the threaded
    processor at year/month/week boundaries.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    if ouroboros_mode == "year":
        return d.month == 1 and d.day == 1
    if ouroboros_mode == "month":
        return d.day == 1
    _sunday = 6
    if ouroboros_mode == "week":
        return d.weekday() == _sunday  # Sunday (matches ouroboros.py SSoT)
    return False


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

    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    current = start
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def _ariadne_disabled() -> bool:
    """Check if Ariadne (trade-ID resume) is explicitly disabled.

    Ariadne is ON by default. Set ``RANGEBAR_ARIADNE_ENABLED=false`` to
    fall back to timestamp-based resume.
    """
    val = os.environ.get("RANGEBAR_ARIADNE_ENABLED", "true")
    return val.lower() in ("0", "false", "no")


def _ariadne_resume(
    symbol: str,
    last_processed_agg_trade_id: int,
    end_date: str,
    threshold_decimal_bps: int,
    processor: object,
    *,
    include_microstructure: bool = False,  # noqa: ARG001
    ouroboros_mode: str | None = None,
    notify: bool = True,  # noqa: ARG001
    trace_id: str | None = None,  # noqa: ARG001
) -> int:
    """Issue #111: Resume processing from last trade ID via Rust fromId fetch.

    Fetches trades starting from ``last_processed_agg_trade_id + 1`` in
    batches via the Rust ``fetch_aggtrades_by_id()`` PyO3 binding, feeds
    each batch to the Rust processor, and writes completed bars to ClickHouse.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    last_processed_agg_trade_id : int
        High-water mark: last fully processed agg_trade_id.
    end_date : str
        Stop processing when trade timestamps exceed this date.
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    processor : RangeBarProcessor
        Active processor with restored state.
    include_microstructure : bool
        Whether microstructure features are enabled.
    ouroboros_mode : str
        Ouroboros reset mode.
    notify : bool
        Whether to emit hook events.
    trace_id : str | None
        Trace ID for telemetry.

    Returns
    -------
    int
        Total bars written via Ariadne resume.
    """
    # Issue #126: Resolve ouroboros mode from config if not specified
    if ouroboros_mode is None:
        from rangebar.ouroboros import get_operational_ouroboros_mode

        ouroboros_mode = get_operational_ouroboros_mode()

    from rangebar._core import fetch_aggtrades_by_id
    from rangebar.orchestration.range_bars_cache import fatal_cache_write

    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000)
    batch_size = 1000
    from_id = last_processed_agg_trade_id + 1
    total_bars = 0

    logger.info(
        "Ariadne resume: %s from agg_trade_id=%d (end_date=%s)",
        symbol,
        from_id,
        end_date,
    )

    while True:
        try:
            trades = fetch_aggtrades_by_id(symbol, from_id, batch_size)
        except (RuntimeError, OSError) as e:
            logger.warning("Ariadne fetch failed at from_id=%d: %s", from_id, e)
            break

        if not trades:
            logger.info("Ariadne: no more trades after from_id=%d", from_id)
            break

        # Check if we've gone past end_date
        last_trade_ts = trades[-1]["timestamp"]
        if last_trade_ts > end_ms:
            # Filter trades to only include those within range
            trades = [t for t in trades if t["timestamp"] <= end_ms]
            if not trades:
                logger.info("Ariadne: all trades past end_date, stopping")
                break

        # Process through Rust processor
        bars_df = processor.process_trades(
            [
                {
                    "timestamp": t["timestamp"],
                    "price": t["price"],
                    "quantity": t["quantity"],
                    "is_buyer_maker": t["is_buyer_maker"],
                }
                for t in trades
            ],
        )

        if bars_df is not None and len(bars_df) > 0:
            import pandas as pd

            if not isinstance(bars_df, pd.DataFrame):
                bars_df = pd.DataFrame(bars_df)
            if not bars_df.empty:
                written = fatal_cache_write(
                    bars_df, symbol, threshold_decimal_bps, ouroboros_mode,
                )
                total_bars += written

        # Advance cursor to next batch
        from_id = trades[-1]["agg_trade_id"] + 1

        # Check if this was a partial batch (end of data)
        if last_trade_ts > end_ms:
            break

    logger.info(
        "Ariadne resume complete: %s — %d bars written (from_id reached %d)",
        symbol,
        total_bars,
        from_id,
    )

    return total_bars


def populate_cache_resumable(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    threshold_decimal_bps: int = 250,
    force_refresh: bool = False,
    include_microstructure: bool = False,
    ouroboros_mode: str | None = None,
    checkpoint_dir: Path | None = None,
    notify: bool = True,
    verbose: bool = True,
    inter_bar_lookback_bars: int | None = None,
    num_async_writers: int = 0,
) -> int:
    """Populate cache for a date range with automatic checkpointing.

    This function fetches tick data and computes range bars day-by-day,
    saving progress after each day. If interrupted, simply run again
    with the same parameters to resume from the last checkpoint.

    Issue #69 Enhancements:
    - Bar-level resumability: Preserves incomplete bar state via processor checkpoints
    - Hybrid storage: Checkpoints saved to both local and ClickHouse
    - force_refresh: Wipe cache and checkpoint to start fresh

    Issue #126: ouroboros defaults to None → resolved from RANGEBAR_OUROBOROS_MODE.

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
    force_refresh : bool
        If True, wipe existing cache and checkpoint, start fresh.
        If False (default), resume from last checkpoint.
    include_microstructure : bool
        Include microstructure features (default: False).
    ouroboros_mode : str
        Ouroboros reset mode: "year", "month", or "week" (default: "year").
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.
    notify : bool
        Whether to emit hook events for progress tracking.
    verbose : bool
        Show progress bar (tqdm) and structured logging (default: True).
        Set to False for batch/CI environments.
    num_async_writers : int
        Number of concurrent ClickHouse write workers (default: 0 = disabled).
        If > 0, enables async writes with connection pooling for 2-3x speedup.
        Issue #96 Phase 4: Async cache writes + connection pooling.

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
    >>> # Will resume from last checkpoint (bar-level accuracy)

    >>> # Start fresh (wipe cache and checkpoint):
    >>> bars = populate_cache_resumable(
    ...     "BTCUSDT", "2024-01-01", "2024-06-30",
    ...     force_refresh=True,
    ... )
    """
    from rangebar.hooks import HookEvent, emit_hook
    from rangebar.logging import generate_trace_id, log_checkpoint_event

    trace_id = generate_trace_id("pop")

    # Symbol registry gate + start date clamping (Issue #79)
    from rangebar.symbol_registry import (
        validate_and_clamp_start_date,
        validate_symbol_registered,
    )

    validate_symbol_registered(symbol, operation="populate_cache_resumable")
    start_date = validate_and_clamp_start_date(symbol, start_date)

    # Threshold validation: enforce per-symbol minimums (Issue #115)
    from rangebar.threshold import resolve_and_validate_threshold

    threshold_decimal_bps = resolve_and_validate_threshold(
        symbol, threshold_decimal_bps,
    )

    # Issue #126: Resolve ouroboros mode from config if not specified
    if ouroboros_mode is None:
        from rangebar.ouroboros import get_operational_ouroboros_mode

        ouroboros_mode = get_operational_ouroboros_mode()
    logger.info("populate_cache_resumable: ouroboros_mode=%s", ouroboros_mode)

    # T-1 guard: Binance Vision publishes with ~1-day lag.
    # Attempting to fetch today's data causes RuntimeError: "No data available"
    # which crashes the job after checkpoint is written but before ClickHouse
    # receives bars (gap regression scenario).
    today_utc = datetime.now(UTC).date()
    t1_cutoff = today_utc - timedelta(days=1)
    end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC).date()
    if end_date_parsed >= today_utc:
        logger.warning(
            "end_date %s is today or future; clamping to T-1 (%s) "
            "(Binance Vision publishes with ~1-day lag)",
            end_date_parsed,
            t1_cutoff,
        )
        end_date = t1_cutoff.strftime("%Y-%m-%d")

    checkpoint_path = _get_checkpoint_path(
        symbol, threshold_decimal_bps, start_date, end_date, checkpoint_dir,
        ouroboros_mode=ouroboros_mode,  # Issue #126: mode in filename
    )

    # Issue #69: Handle force_refresh - wipe cache and checkpoint
    if force_refresh:
        logger.info("Force refresh: clearing cache and checkpoint for %s", symbol)
        # Delete local checkpoint (missing_ok=True prevents race with concurrent jobs)
        checkpoint_path.unlink(missing_ok=True)
        logger.debug("Deleted local checkpoint: %s", checkpoint_path)

        # Delete cached bars and ClickHouse checkpoint
        try:
            from rangebar.clickhouse import RangeBarCache

            # Convert dates to timestamps for deletion
            from rangebar.orchestration.helpers import (
                _datetime_to_end_ms,
                _datetime_to_start_ms,
            )

            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
            start_ts = _datetime_to_start_ms(start_dt)
            end_ts = _datetime_to_end_ms(end_dt)

            with RangeBarCache() as cache:
                cache.delete_bars(
                    symbol, threshold_decimal_bps, start_ts, end_ts,
                    ouroboros_mode=ouroboros_mode,  # Issue #126
                )
                cache.delete_checkpoint(
                    symbol, threshold_decimal_bps, start_date, end_date,
                    ouroboros_mode=ouroboros_mode,
                )
                logger.debug("Deleted cached bars and ClickHouse checkpoint")
        except (ImportError, ConnectionError) as e:
            logger.debug("ClickHouse cleanup skipped: %s", e)

        checkpoint = None
    else:
        # Check for existing checkpoint (local first, then ClickHouse)
        checkpoint = PopulationCheckpoint.load(checkpoint_path)

        # Issue #69: Try ClickHouse checkpoint for cross-machine resume
        if checkpoint is None:
            checkpoint = _load_checkpoint_from_clickhouse(
                symbol, threshold_decimal_bps, start_date, end_date,
                ouroboros_mode=ouroboros_mode,
            )

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
            # Issue #126: Verify ouroboros_mode consistency
            if (
                hasattr(checkpoint, "ouroboros_mode")
                and checkpoint.ouroboros_mode
                and checkpoint.ouroboros_mode != ouroboros_mode
            ):
                msg = (
                    f"Checkpoint ouroboros_mode mismatch: checkpoint has "
                    f"{checkpoint.ouroboros_mode!r} but current mode is "
                    f"{ouroboros_mode!r}. Delete the checkpoint or use matching mode."
                )
                raise ValueError(msg)

            # Resume from day after last completed
            last_completed = datetime.strptime(
                checkpoint.last_completed_date, "%Y-%m-%d"
            ).replace(tzinfo=UTC)
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
                    trace_id=trace_id,
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

    # Issue #70: Progress tracking with ResumableObserver
    # Calculate initial count for resumed operations
    if checkpoint:
        # Count days already completed
        all_dates = list(_date_range(start_date, end_date))
        initial_days = len(all_dates) - total_days
    else:
        initial_days = 0

    # Initialize progress observer (lazy import to avoid circular deps)
    observer = None
    if verbose:
        from rangebar.progress import ResumableObserver

        observer = ResumableObserver(
            total=len(list(_date_range(start_date, end_date))),
            initial=initial_days,
            desc=symbol,
            unit="days",
            verbose=True,
        )

    # Issue #97: Thread processor across days for bar-level continuity.
    # Bypass get_range_bars() to maintain a single processor across days.
    # get_range_bars() creates a fresh processor per call, losing incomplete
    # bar state at day boundaries.

    from rangebar.exceptions import CacheWriteError
    from rangebar.orchestration.helpers import (
        _parse_microstructure_env_vars,
        _process_binance_trades,
    )
    from rangebar.orchestration.range_bars_cache import fatal_cache_write
    from rangebar.processors.core import RangeBarProcessor
    from rangebar.storage.parquet import TickStorage

    eff_count, eff_bars, enable_intra = _parse_microstructure_env_vars(
        include_microstructure, None, inter_bar_lookback_bars,
    )
    storage = TickStorage()
    active_processor: RangeBarProcessor | None = None

    # Restore processor from checkpoint if available
    if checkpoint and checkpoint.processor_checkpoint:
        try:
            active_processor = RangeBarProcessor.from_checkpoint(
                checkpoint.processor_checkpoint,
            )
            # Re-enable microstructure features after checkpoint restore
            if include_microstructure:
                active_processor.enable_microstructure(
                    inter_bar_lookback_count=eff_count,
                    inter_bar_lookback_bars=eff_bars,
                    include_intra_bar_features=enable_intra,
                )
            logger.info(
                "Restored processor from checkpoint "
                "(has_incomplete_bar=%s, defer_open=%s)",
                checkpoint.processor_checkpoint.get("has_incomplete_bar"),
                checkpoint.processor_checkpoint.get("defer_open"),
            )
            log_checkpoint_event(
                "checkpoint_restore",
                symbol,
                trace_id,
                threshold_dbps=threshold_decimal_bps,
                last_completed_date=checkpoint.last_completed_date,
                bars_written=checkpoint.bars_written,
                has_incomplete_bar=checkpoint.processor_checkpoint.get(
                    "has_incomplete_bar"
                ),
                last_trade_id=checkpoint.processor_checkpoint.get(
                    "last_trade_id"
                ),
            )
            if notify:
                emit_hook(
                    HookEvent.CHECKPOINT_RESTORED,
                    symbol=symbol,
                    trace_id=trace_id,
                    threshold_dbps=threshold_decimal_bps,
                    has_incomplete_bar=checkpoint.processor_checkpoint.get(
                        "has_incomplete_bar"
                    ),
                    last_trade_id=checkpoint.processor_checkpoint.get(
                        "last_trade_id"
                    ),
                    defer_open=checkpoint.processor_checkpoint.get(
                        "defer_open"
                    ),
                )
        except Exception:
            logger.exception(
                "Failed to restore processor from checkpoint, starting fresh",
            )
            log_checkpoint_event(
                "checkpoint_restore_failed",
                symbol,
                trace_id,
                threshold_dbps=threshold_decimal_bps,
                fallback="fresh_processor",
            )
            if notify:
                emit_hook(
                    HookEvent.CHECKPOINT_RESTORE_FAILED,
                    symbol=symbol,
                    trace_id=trace_id,
                    threshold_dbps=threshold_decimal_bps,
                )
            active_processor = None

    created_at = (
        checkpoint.created_at if checkpoint else datetime.now(UTC).isoformat()
    )

    # Issue #111 (Ariadne): Resume via trade ID by default.
    # Deterministic, gap-free — every boundary, every touchpoint.
    # Disable with RANGEBAR_ARIADNE_ENABLED=false to fall back to timestamps.
    if (
        not _ariadne_disabled()
        and checkpoint is not None
        and checkpoint.last_processed_agg_trade_id
        and active_processor is not None
    ):
        logger.info(
            "Ariadne: resuming %s via trade ID %d",
            symbol,
            checkpoint.last_processed_agg_trade_id,
        )
        ariadne_bars = _ariadne_resume(
            symbol,
            checkpoint.last_processed_agg_trade_id,
            end_date,
            threshold_decimal_bps,
            active_processor,
            include_microstructure=include_microstructure,
            ouroboros_mode=ouroboros_mode,
            notify=notify,
            trace_id=trace_id,
        )
        total_bars += ariadne_bars
        return total_bars

    # Issue #96 Phase 4: Initialize thread pool for concurrent ClickHouse writes
    thread_pool = None
    if num_async_writers > 0:
        try:
            from concurrent.futures import ThreadPoolExecutor

            from rangebar.clickhouse.connection_pool import get_connection_pool

            # Initialize connection pool (reuse across all writes)
            get_connection_pool(max_pool_size=num_async_writers * 2)
            logger.info(
                "Initialized ClickHouse connection pool: max=%d",
                num_async_writers * 2,
            )

            # Create thread pool for concurrent writes
            thread_pool = ThreadPoolExecutor(max_workers=num_async_writers)
            logger.info(
                "Concurrent cache writer initialized: %d workers",
                num_async_writers,
            )

        except (ImportError, ConnectionError, OSError) as e:
            logger.warning("Failed to initialize thread pool: %s", e)
            logger.info("Falling back to synchronous writes")
            thread_pool = None

    for i, date in enumerate(dates, 1):
        logger.info(
            "Processing %s [%d/%d]: %s",
            symbol,
            i,
            total_days,
            date,
        )

        try:
            # Issue #97: Handle ouroboros boundary — reset processor state
            if active_processor is not None and _is_ouroboros_boundary(
                date, ouroboros_mode
            ):
                logger.info(
                    "Ouroboros %s boundary at %s — resetting processor",
                    ouroboros_mode, date,
                )
                log_checkpoint_event(
                    "ouroboros_reset",
                    symbol,
                    trace_id,
                    date=date,
                    boundary_type=ouroboros_mode,
                    had_incomplete_bar=active_processor.get_incomplete_bar()
                    is not None,
                )
                active_processor.reset_at_ouroboros()

            # Load tick data for this day (cache or network)
            from rangebar.orchestration.helpers import (
                _datetime_to_end_ms,
                _datetime_to_start_ms,
                _fetch_binance,
            )

            day_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
            day_start_ms = _datetime_to_start_ms(day_dt)
            day_end_ms = _datetime_to_end_ms(day_dt)
            cache_symbol = f"BINANCE_SPOT_{symbol}".upper()

            has_cached = storage.has_ticks(cache_symbol, day_start_ms, day_end_ms)
            if has_cached:
                day_ticks = storage.read_ticks(
                    cache_symbol, day_start_ms, day_end_ms,
                )
            else:
                day_ticks = _fetch_binance(symbol, date, date, "spot")
                if not day_ticks.is_empty():
                    storage.write_ticks(cache_symbol, day_ticks)

            if day_ticks.is_empty():
                logger.debug("%s %s: no tick data", symbol, date)
                bars_today = 0
            else:
                # Issue #96 Task #13 Phase D: Arrow optimization
                # Arrow: include_microstructure=False (skip Pandas, 1.3-1.5x speedup)
                # Pandas: include_microstructure=True (plugins need Pandas)
                return_format = "arrow" if not include_microstructure else "pandas"

                # Process ticks with threaded processor
                bars_result, active_processor = _process_binance_trades(
                    day_ticks,
                    threshold_decimal_bps,
                    False,  # include_incomplete
                    include_microstructure,
                    processor=active_processor,
                    symbol=symbol,
                    inter_bar_lookback_count=eff_count,
                    include_intra_bar_features=enable_intra,
                    inter_bar_lookback_bars=eff_bars,
                    return_format=return_format,
                )

                # Normalize return type for consistent downstream handling
                import polars as pl

                bars_df = bars_result

                # Issue #98: Plugin feature enrichment (post-Rust, pre-cache)
                # Plugins operate on Pandas, so convert if needed for enrichment
                if include_microstructure and bars_df is not None:
                    if isinstance(bars_df, pl.DataFrame):
                        # Convert Polars → Pandas for plugin enrichment
                        bars_df = bars_df.to_pandas()

                    if not bars_df.empty:
                        from rangebar.plugins.loader import enrich_bars

                        bars_df = enrich_bars(bars_df, symbol, threshold_decimal_bps)

                # Write to ClickHouse cache (fatal — ClickHouse IS the destination)
                if bars_df is not None and not bars_df.empty:
                    # Issue #96 Phase 4: Use thread pool if available
                    if thread_pool is not None:
                        # Submit write to thread pool (waits at end with shutdown)
                        thread_pool.submit(
                            fatal_cache_write,
                            bars_df,
                            symbol,
                            threshold_decimal_bps,
                            ouroboros_mode,
                        )
                        bars_today = len(bars_df)
                    else:
                        bars_today = fatal_cache_write(
                            bars_df, symbol, threshold_decimal_bps, ouroboros_mode,
                        )
                else:
                    bars_today = 0

                del bars_df

            total_bars += bars_today

            logger.debug(
                "%s %s: %d bars (total: %d)",
                symbol,
                date,
                bars_today,
                total_bars,
            )

            # Issue #70: Update progress observer
            if observer is not None:
                observer.update(1)
                observer.set_postfix(bars=total_bars)
                observer.log_event(
                    "day_complete",
                    date=date,
                    bars_today=bars_today,
                    total_bars=total_bars,
                )

            # Issue #97: Save processor checkpoint for bar-level resumability
            proc_cp = None
            if active_processor is not None:
                proc_cp = active_processor.create_checkpoint(symbol)

            checkpoint = PopulationCheckpoint(
                symbol=symbol,
                threshold_bps=threshold_decimal_bps,
                start_date=start_date,
                end_date=end_date,
                last_completed_date=date,
                bars_written=total_bars,
                created_at=created_at,
                include_microstructure=include_microstructure,
                ouroboros_mode=ouroboros_mode,
                processor_checkpoint=proc_cp,
                last_trade_timestamp_ms=(
                    proc_cp.get("last_timestamp_ms") if proc_cp else None
                ),
                first_agg_trade_id_in_bar=(
                    (proc_cp.get("incomplete_bar_raw") or {}).get(
                        "first_agg_trade_id"
                    )
                    if proc_cp
                    else None
                ),
                last_agg_trade_id_in_bar=(
                    (proc_cp.get("incomplete_bar_raw") or {}).get(
                        "last_agg_trade_id"
                    )
                    if proc_cp
                    else None
                ),
                # Issue #111 (Ariadne): High-water mark from processor
                last_processed_agg_trade_id=(
                    proc_cp.get("last_trade_id") if proc_cp else None
                ),
            )
            # Save to local filesystem (fast)
            checkpoint.save(checkpoint_path)
            # Issue #69: Also save to ClickHouse (cross-machine resume)
            _save_checkpoint_to_clickhouse(checkpoint)

            log_checkpoint_event(
                "checkpoint_save",
                symbol,
                trace_id,
                threshold_dbps=threshold_decimal_bps,
                date=date,
                bars_today=bars_today,
                total_bars=total_bars,
                has_incomplete_bar=proc_cp is not None
                and proc_cp.get("has_incomplete_bar", False),
                defer_open=proc_cp.get("defer_open") if proc_cp else None,
            )

            if notify:
                emit_hook(
                    HookEvent.CHECKPOINT_SAVED,
                    symbol=symbol,
                    trace_id=trace_id,
                    date=date,
                    bars_today=bars_today,
                    total_bars=total_bars,
                    progress_pct=round(i / total_days * 100, 1),
                )

            # Issue #76: Explicit memory cleanup
            del day_ticks

        except (ValueError, RuntimeError, OSError, CacheWriteError) as exc:
            # Defense-in-depth: if the final day has no data (archive not yet
            # published on Binance Vision), complete gracefully instead of
            # crashing the entire multi-day job at 99.96% completion.
            if "No data available" in str(exc) and date == dates[-1]:
                logger.warning(
                    "No data for final day %s (archive not yet published?) "
                    "— completing gracefully with %d bars",
                    date,
                    total_bars,
                )
                break

            logger.exception(
                "Failed to process %s for %s",
                symbol,
                date,
            )
            if notify:
                emit_hook(
                    HookEvent.POPULATION_FAILED,
                    symbol=symbol,
                    trace_id=trace_id,
                    date=date,
                    error=str(exc),
                    total_bars=total_bars,
                )
            raise

    # Issue #70: Close progress observer
    if observer is not None:
        observer.close()

    # Issue #77: Deduplicate after population completes
    # ReplacingMergeTree only deduplicates during background merges,
    # so we force immediate deduplication to ensure clean data.
    # Issue #90: Non-fatal — dedup failure should not crash the job.
    # INSERT dedup tokens (Issue #90) prevent most duplicates at source,
    # read queries use FINAL for on-the-fly dedup, and the pipeline-level
    # OPTIMIZE TABLE FINAL job catches any stragglers.
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            logger.info(
                "Running post-population deduplication for %s @ %d dbps",
                symbol,
                threshold_decimal_bps,
            )
            cache.deduplicate_bars(symbol, threshold_decimal_bps)
            logger.debug("Deduplication complete")
    except (ImportError, ConnectionError) as e:
        logger.debug("Post-population deduplication skipped: %s", e)
    except (OSError, RuntimeError) as e:
        logger.warning(
            "Post-population deduplication did not complete for %s @ %d dbps: %s. "
            "Data is written. Run OPTIMIZE TABLE FINAL manually or via pipeline.",
            symbol,
            threshold_decimal_bps,
            e,
        )

    # Clean up checkpoint on success (missing_ok for concurrent jobs)
    checkpoint_path.unlink(missing_ok=True)
    logger.debug("Removed checkpoint file after successful completion")

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
            trace_id=trace_id,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
        )

    # Issue #96 Phase 4: Wait for pending async writes to complete
    if thread_pool is not None:
        logger.info("Waiting for pending ClickHouse writes to complete...")
        thread_pool.shutdown(wait=True)
        logger.info("All writes completed, thread pool shut down")

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
    threshold_decimal_bps: int,
    start_date: str,
    end_date: str,
    checkpoint_dir: Path | None = None,
    ouroboros_mode: str = "year",
) -> bool:
    """Clear a specific checkpoint.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    checkpoint_dir : Path | None
        Custom checkpoint directory. Uses default if None.
    ouroboros_mode : str
        Ouroboros mode for checkpoint filename (Issue #126).

    Returns
    -------
    bool
        True if checkpoint was found and removed.
    """
    checkpoint_path = _get_checkpoint_path(
        symbol, threshold_decimal_bps, start_date, end_date, checkpoint_dir,
        ouroboros_mode=ouroboros_mode,
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)
        logger.info("Cleared checkpoint: %s", checkpoint_path)
        return True

    return False


__all__ = [
    "PopulationCheckpoint",
    "clear_checkpoint",
    "list_checkpoints",
    "populate_cache_resumable",
]
