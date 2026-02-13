"""Parquet I/O helpers: validation, atomic writes, and cache directory.

Extracted from parquet.py (Issue #73 helpers + constants) to reduce
module size. All functions are module-level utilities used by both
TickStorage and other consumers.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path

import polars as pl
from platformdirs import user_cache_dir

# Constants
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
APP_NAME = "rangebar"
APP_AUTHOR = "terrylica"

logger = logging.getLogger(__name__)


# =============================================================================
# Parquet Validation and Atomic Write Helpers (Issue #73)
# =============================================================================

# Minimum valid Parquet file size: 4 (header) + 4 (footer length) + 4 (footer magic)
_MIN_PARQUET_SIZE = 12
_PARQUET_MAGIC = b"PAR1"


def _is_valid_parquet(path: Path) -> tuple[bool, str]:
    """Check if file is valid Parquet by verifying PAR1 magic bytes.

    Parquet files MUST start and end with b"PAR1" (4 bytes each).
    Minimum valid size: 12 bytes (4 header + 4 footer length + 4 footer magic).

    Parameters
    ----------
    path : Path
        Path to the Parquet file to validate.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason) - reason is empty string if valid.

    Examples
    --------
    >>> is_valid, reason = _is_valid_parquet(Path("data.parquet"))
    >>> if not is_valid:
    ...     print(f"Corrupted: {reason}")
    """
    if not path.exists():
        return False, "file does not exist"

    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"stat error: {e}"

    if size < _MIN_PARQUET_SIZE:
        return False, f"file too small ({size} bytes, minimum {_MIN_PARQUET_SIZE})"

    try:
        with path.open("rb") as f:
            header = f.read(4)
            if header != _PARQUET_MAGIC:
                return False, f"invalid header magic: {header!r}"
            f.seek(-4, 2)  # Seek to 4 bytes before end
            footer = f.read(4)
            if footer != _PARQUET_MAGIC:
                return False, f"invalid footer magic: {footer!r}"
    except OSError as e:
        return False, f"read error: {e}"

    return True, ""


def _validate_and_recover_parquet(path: Path, *, auto_delete: bool = True) -> bool:
    """Validate Parquet file and auto-delete if corrupted.

    Returns True if valid, False if corrupted (and deleted if auto_delete=True).
    Raises ParquetCorruptionError if corrupted and auto_delete=False.

    Parameters
    ----------
    path : Path
        Path to the Parquet file to validate.
    auto_delete : bool
        If True (default), delete corrupted files for re-fetch.
        If False, raise ParquetCorruptionError instead.

    Returns
    -------
    bool
        True if file is valid, False if corrupted and deleted.

    Raises
    ------
    ParquetCorruptionError
        If file is corrupted and auto_delete=False.

    Examples
    --------
    >>> if not _validate_and_recover_parquet(path, auto_delete=True):
    ...     # File was corrupted and deleted, will be re-fetched
    ...     pass
    """
    is_valid, reason = _is_valid_parquet(path)
    if is_valid:
        return True

    logger.warning("Corrupted Parquet file detected: %s (%s)", path, reason)

    if auto_delete:
        try:
            path.unlink()
            logger.info("Auto-deleted corrupted file for re-fetch: %s", path)
        except OSError:
            logger.exception("Failed to delete corrupted file %s", path)
        return False

    from rangebar.exceptions import ParquetCorruptionError

    msg = f"Corrupted Parquet file: {path} ({reason})"
    raise ParquetCorruptionError(msg, path=path, reason=reason)


def _atomic_write_parquet(
    df: pl.DataFrame,
    target_path: Path,
    *,
    compression: str = COMPRESSION,
    compression_level: int = COMPRESSION_LEVEL,
) -> None:
    """Write Parquet file atomically using tempfile + fsync + rename.

    Guarantees: target_path is either complete valid Parquet or doesn't exist.
    Never leaves partial/corrupted files on crash or kill.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to write.
    target_path : Path
        Final destination path for the Parquet file.
    compression : str
        Compression codec (default: "zstd").
    compression_level : int
        Compression level (default: 3).

    Raises
    ------
    OSError
        If write fails (temp file is cleaned up automatically).
    RuntimeError
        If Polars write fails (temp file is cleaned up automatically).

    Notes
    -----
    This function uses the POSIX atomic rename guarantee: Path.replace() is
    atomic on the same filesystem. The temp file is created in the same
    directory to ensure it's on the same filesystem.

    Examples
    --------
    >>> _atomic_write_parquet(df, Path("data.parquet"))
    """
    # Create temp file in same directory (required for atomic rename on same filesystem)
    fd, temp_path_str = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=".parquet_",
        suffix=".tmp",
    )
    temp_path = Path(temp_path_str)

    try:
        os.close(fd)  # Close fd, Polars will open it itself
        df.write_parquet(
            temp_path,
            compression=compression,
            compression_level=compression_level,
        )
        # Sync to disk before rename to ensure durability
        with temp_path.open("rb") as f:
            os.fsync(f.fileno())
        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        temp_path.replace(target_path)
    except (OSError, RuntimeError):
        # Clean up temp file on any failure
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def get_cache_dir() -> Path:
    """Get the cross-platform cache directory for rangebar.

    Returns
    -------
    Path
        Platform-specific cache directory:
        - macOS:   ~/Library/Caches/rangebar/
        - Linux:   ~/.cache/rangebar/ (respects XDG_CACHE_HOME)
        - Windows: %USERPROFILE%\\AppData\\Local\\terrylica\\rangebar\\Cache\\

    Examples
    --------
    >>> from rangebar.storage import get_cache_dir
    >>> cache_dir = get_cache_dir()
    >>> print(cache_dir)
    /Users/username/Library/Caches/rangebar
    """
    # Allow override via environment variable
    env_override = os.getenv("RANGEBAR_CACHE_DIR")
    if env_override:
        return Path(env_override)

    return Path(user_cache_dir(APP_NAME, APP_AUTHOR))
