"""Resource guards for memory-safe range bar processing (Issue #49).

Provides cross-platform memory monitoring and process-level caps
using only stdlib modules (no psutil dependency).

MEM-009: Process-level RLIMIT_AS cap (MemoryError instead of OOM kill)
MEM-010: Pre-flight memory estimation before tick loading
MEM-011: Environment variable default (RANGEBAR_MAX_MEMORY_GB)

Environment Variables
---------------------
RANGEBAR_MAX_MEMORY_GB : float
    Default memory limit in GB. When set, all calls to
    ensure_memory_limit() will use this as the default cap.
    Example: RANGEBAR_MAX_MEMORY_GB=45 (for 45 GB limit)

RANGEBAR_MAX_MEMORY_PCT : float
    Default memory limit as fraction of total RAM (0.0-1.0).
    Example: RANGEBAR_MAX_MEMORY_PCT=0.7 (for 70% of RAM)

If both are set, the smaller limit is used.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rangebar.storage.parquet import TickStorage


@dataclass(frozen=True)
class MemoryInfo:
    """Current process and system memory snapshot."""

    process_rss_mb: int
    """Resident set size of current process in MB."""

    system_total_mb: int
    """Total physical RAM in MB."""

    system_available_mb: int
    """Available RAM in MB (approximate on macOS)."""

    @property
    def usage_pct(self) -> float:
        """Process RSS as fraction of total system RAM."""
        if self.system_total_mb == 0:
            return 0.0
        return self.process_rss_mb / self.system_total_mb


def get_memory_info() -> MemoryInfo:
    """Get current process and system memory info.

    Cross-platform: uses /proc/self/status on Linux,
    resource.getrusage() on macOS. No external dependencies.

    Returns
    -------
    MemoryInfo
        Snapshot of process and system memory.
    """
    process_rss_mb = _get_process_rss_mb()
    system_total_mb = _get_system_total_mb()
    system_available_mb = _get_system_available_mb()

    return MemoryInfo(
        process_rss_mb=process_rss_mb,
        system_total_mb=system_total_mb,
        system_available_mb=system_available_mb,
    )


def set_memory_limit(
    *,
    max_gb: float | None = None,
    max_pct: float | None = None,
) -> int:
    """Set process virtual memory limit (MEM-009).

    Causes MemoryError instead of OOM kill when limit exceeded.

    Parameters
    ----------
    max_gb : float | None
        Hard limit in gigabytes.
    max_pct : float | None
        Fraction of total system RAM (e.g., 0.8 for 80%).
        If both max_gb and max_pct are given, uses the smaller limit.

    Returns
    -------
    int
        The limit set in bytes, or -1 if no limit was set.

    Notes
    -----
    On macOS, RLIMIT_RSS is used (advisory, not enforced by kernel).
    On Linux, RLIMIT_AS is used (hard cap on virtual address space).
    """
    if max_gb is None and max_pct is None:
        return -1

    limits: list[int] = []

    if max_gb is not None:
        limits.append(int(max_gb * 1024 * 1024 * 1024))

    if max_pct is not None:
        total_bytes = _get_system_total_mb() * 1024 * 1024
        limits.append(int(total_bytes * max_pct))

    limit_bytes = min(limits)

    # MEM-009: Set process-level cap
    try:
        if sys.platform == "darwin":
            # macOS: RLIMIT_RSS is advisory (kernel doesn't enforce)
            # but Python's allocator may still raise MemoryError
            # Query current hard limit and cap to that if needed
            _, hard_limit = resource.getrlimit(resource.RLIMIT_RSS)
            if hard_limit != resource.RLIM_INFINITY and limit_bytes > hard_limit:
                limit_bytes = hard_limit
            resource.setrlimit(resource.RLIMIT_RSS, (limit_bytes, limit_bytes))
        else:
            # Linux: RLIMIT_AS is enforced by kernel
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit_bytes > hard_limit:
                limit_bytes = hard_limit
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        # Can't set limit (e.g., unprivileged user, macOS restrictions)
        return -1

    return limit_bytes


# ---------------------------------------------------------------------------
# MEM-011: Default memory limit from environment
# ---------------------------------------------------------------------------

# Mutable container to track if memory limit was applied (avoids global stmt)
_memory_limit_state: dict[str, bool] = {"applied": False}


def _parse_env_float(name: str) -> float | None:
    """Parse environment variable as float, returning None on failure."""
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_default_memory_limit() -> tuple[float | None, float | None]:
    """Get default memory limit from environment variables.

    Returns
    -------
    tuple[float | None, float | None]
        (max_gb, max_pct) from RANGEBAR_MAX_MEMORY_GB and
        RANGEBAR_MAX_MEMORY_PCT environment variables.
    """
    max_gb = _parse_env_float("RANGEBAR_MAX_MEMORY_GB")

    max_pct = _parse_env_float("RANGEBAR_MAX_MEMORY_PCT")
    if max_pct is not None and not 0.0 < max_pct <= 1.0:
        max_pct = None

    return max_gb, max_pct


def ensure_memory_limit(
    *,
    max_gb: float | None = None,
    max_pct: float | None = None,
    fallback_pct: float = 0.7,
) -> int:
    """Ensure a process memory limit is set (MEM-011).

    Idempotent: only sets limit on first call per process.
    Uses environment variables if no explicit limit provided.

    Priority order:
    1. Explicit max_gb/max_pct parameters
    2. RANGEBAR_MAX_MEMORY_GB / RANGEBAR_MAX_MEMORY_PCT environment vars
    3. fallback_pct of total system RAM (default 70%)

    Parameters
    ----------
    max_gb : float | None
        Explicit limit in GB (overrides environment).
    max_pct : float | None
        Explicit limit as fraction of RAM (overrides environment).
    fallback_pct : float
        Fallback percentage of RAM if no explicit limit or env var.
        Default 0.7 (70% of system RAM).

    Returns
    -------
    int
        The limit set in bytes, or -1 if limit was already applied.

    Examples
    --------
    >>> # Set default 70% limit
    >>> ensure_memory_limit()

    >>> # Override with explicit limit
    >>> ensure_memory_limit(max_gb=45)

    >>> # Respect environment variable
    >>> import os
    >>> os.environ["RANGEBAR_MAX_MEMORY_GB"] = "50"
    >>> ensure_memory_limit()  # Uses 50 GB from env
    """
    if _memory_limit_state["applied"]:
        return -1

    # Priority 1: Explicit parameters
    if max_gb is not None or max_pct is not None:
        limit = set_memory_limit(max_gb=max_gb, max_pct=max_pct)
        _memory_limit_state["applied"] = True
        return limit

    # Priority 2: Environment variables
    env_gb, env_pct = get_default_memory_limit()
    if env_gb is not None or env_pct is not None:
        limit = set_memory_limit(max_gb=env_gb, max_pct=env_pct)
        _memory_limit_state["applied"] = True
        return limit

    # Priority 3: Fallback percentage
    limit = set_memory_limit(max_pct=fallback_pct)
    _memory_limit_state["applied"] = True
    return limit


def auto_memory_guard() -> int:
    """Automatically enable memory guard at module import (MEM-011).

    This function is called when `import rangebar` is executed.
    It ensures a memory limit is always set unless explicitly disabled
    via RANGEBAR_NO_MEMORY_GUARD=1.

    Returns
    -------
    int
        The limit set in bytes, or -1 if disabled or already applied.

    Environment Variables
    ---------------------
    RANGEBAR_NO_MEMORY_GUARD : str
        Set to "1" to disable automatic memory guard.
    RANGEBAR_MAX_MEMORY_GB : float
        Override default limit (see ensure_memory_limit).
    RANGEBAR_MAX_MEMORY_PCT : float
        Override default limit as fraction of RAM.
    """
    if os.environ.get("RANGEBAR_NO_MEMORY_GUARD") == "1":
        return -1
    return ensure_memory_limit()


@dataclass(frozen=True)
class MemoryEstimate:
    """Pre-flight memory estimate for tick data loading (MEM-010)."""

    parquet_bytes: int
    """Total on-disk compressed size of Parquet files."""

    estimated_memory_mb: int
    """Estimated in-memory size after decompression."""

    file_count: int
    """Number of Parquet files that would be loaded."""

    system_available_mb: int
    """Available system RAM at estimation time."""

    # Thresholds for memory recommendation (fraction of available RAM)
    _SAFE_THRESHOLD: float = 0.5
    _STREAMING_THRESHOLD: float = 0.8

    @property
    def can_fit(self) -> bool:
        """True if estimated size fits within 80% of available RAM."""
        return self.estimated_memory_mb < (
            self.system_available_mb * self._STREAMING_THRESHOLD
        )

    @property
    def recommendation(self) -> str:
        """One of 'safe', 'streaming_recommended', 'will_oom'."""
        if self.estimated_memory_mb == 0:
            return "safe"
        ratio = self.estimated_memory_mb / max(self.system_available_mb, 1)
        if ratio < self._SAFE_THRESHOLD:
            return "safe"
        if ratio < self._STREAMING_THRESHOLD:
            return "streaming_recommended"
        return "will_oom"

    def check_or_raise(self, max_mb: int | None = None) -> None:
        """Raise MemoryError if estimate exceeds budget.

        Parameters
        ----------
        max_mb : int | None
            Explicit budget in MB. If None, uses 80% of available RAM.
        """
        budget = max_mb if max_mb is not None else int(
            self.system_available_mb * 0.8
        )
        if self.estimated_memory_mb > budget:
            msg = (
                f"Estimated {self.estimated_memory_mb} MB for "
                f"{self.file_count} Parquet files, exceeds budget "
                f"{budget} MB (available: {self.system_available_mb} MB). "
                f"Use precompute_range_bars() or read_ticks_streaming() "
                f"for memory-safe processing."
            )
            raise MemoryError(msg)


def estimate_tick_memory(
    storage: TickStorage,
    symbol: str,
    start_ts: int,
    end_ts: int,
    *,
    compression_ratio: float = 4.0,
) -> MemoryEstimate:
    """Estimate memory required to load tick data (MEM-010).

    Uses Parquet file sizes on disk as a proxy, without reading data.

    Parameters
    ----------
    storage : TickStorage
        Tick storage instance.
    symbol : str
        Cache symbol (e.g., "BINANCE_SPOT_BTCUSDT").
    start_ts : int
        Start timestamp in milliseconds.
    end_ts : int
        End timestamp in milliseconds.
    compression_ratio : float
        Expected decompression ratio (default 4.0, empirically measured
        for Binance aggTrades Parquet files).

    Returns
    -------
    MemoryEstimate
        Estimate with recommendation.
    """
    symbol_dir = storage._get_symbol_dir(symbol)

    if not symbol_dir.exists():
        return MemoryEstimate(
            parquet_bytes=0,
            estimated_memory_mb=0,
            file_count=0,
            system_available_mb=_get_system_available_mb(),
        )

    parquet_files = sorted(symbol_dir.glob("*.parquet"))

    # Filter to relevant months
    start_month = storage._timestamp_to_year_month(start_ts)
    end_month = storage._timestamp_to_year_month(end_ts)
    parquet_files = [
        f for f in parquet_files if start_month <= f.stem <= end_month
    ]

    if not parquet_files:
        return MemoryEstimate(
            parquet_bytes=0,
            estimated_memory_mb=0,
            file_count=0,
            system_available_mb=_get_system_available_mb(),
        )

    total_bytes = sum(f.stat().st_size for f in parquet_files)
    estimated_mb = int(total_bytes * compression_ratio / (1024 * 1024))

    return MemoryEstimate(
        parquet_bytes=total_bytes,
        estimated_memory_mb=estimated_mb,
        file_count=len(parquet_files),
        system_available_mb=_get_system_available_mb(),
    )


# ---------------------------------------------------------------------------
# Platform-specific helpers
# ---------------------------------------------------------------------------


def _get_process_rss_mb() -> int:
    """Get current process RSS in MB."""
    if sys.platform == "darwin":
        # macOS: getrusage returns bytes
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss / (1024 * 1024))
    # Linux: /proc/self/status has VmRSS in kB
    try:
        status = Path("/proc/self/status").read_text()
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb // 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass
    # Fallback
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return int(ru.ru_maxrss // 1024)


def _get_system_total_mb() -> int:
    """Get total system RAM in MB."""
    try:
        if sys.platform == "darwin":
            # macOS: sysctl hw.memsize
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip()) // (1024 * 1024)
        # Linux: /proc/meminfo
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb // 1024
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        pass
    # Fallback: os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) // (1024 * 1024)
    except (ValueError, OSError):
        return 0


def _get_system_available_mb() -> int:
    """Get available system RAM in MB (approximate)."""
    try:
        if sys.platform == "darwin":
            # macOS: vm_stat gives free + inactive pages
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )
            free = 0
            inactive = 0
            page_size = 16384  # Default Apple Silicon
            for line in result.stdout.splitlines():
                if "page size of" in line:
                    page_size = int(
                        line.split("page size of")[1].strip().rstrip(")")
                    )
                elif "Pages free:" in line:
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive:" in line:
                    inactive = int(
                        line.split(":")[1].strip().rstrip(".")
                    )
            return (free + inactive) * page_size // (1024 * 1024)
        # Linux: /proc/meminfo MemAvailable
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb // 1024
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        pass
    # Fallback: assume 50% of total is available
    return _get_system_total_mb() // 2
