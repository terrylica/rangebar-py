#!/usr/bin/env python3
"""Deduplicate Tier 1 Parquet tick cache.

This script fixes Issue #78 by removing duplicate agg_trade_id entries
from existing Parquet files in ~/.cache/rangebar/ticks/.

Root cause: parquet.py:write_ticks() appended data without deduplicating,
causing duplicates to accumulate when the same date range was fetched
multiple times (retry, checkpoint resume, multiple calls).

Usage:
    # Dry run - count duplicates without modifying files
    uv run python scripts/deduplicate_parquet_cache.py --dry-run

    # Deduplicate all files
    uv run python scripts/deduplicate_parquet_cache.py

    # Deduplicate specific symbol
    uv run python scripts/deduplicate_parquet_cache.py --symbol BINANCE_SPOT_BTCUSDT

    # Custom cache directory
    uv run python scripts/deduplicate_parquet_cache.py --cache-dir /path/to/ticks
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import polars as pl
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_default_cache_dir() -> Path:
    """Get default tick cache directory."""
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("rangebar", "terrylica")) / "ticks"


def atomic_write_parquet(df: pl.DataFrame, target_path: Path) -> None:
    """Write Parquet file atomically using tempfile + fsync + rename."""
    fd, temp_path_str = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=".parquet_",
        suffix=".tmp",
    )
    temp_path = Path(temp_path_str)

    try:
        os.close(fd)
        df.write_parquet(temp_path, compression="zstd", compression_level=3)
        with temp_path.open("rb") as f:
            os.fsync(f.fileno())
        temp_path.replace(target_path)
    except (OSError, RuntimeError):
        temp_path.unlink(missing_ok=True)
        raise


def deduplicate_file(
    parquet_path: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Deduplicate a single Parquet file.

    Returns
    -------
    tuple[int, int, int]
        (original_count, unique_count, removed_count)
    """
    df = pl.read_parquet(parquet_path)
    original_count = len(df)

    if "agg_trade_id" not in df.columns:
        return original_count, original_count, 0

    unique_count = df["agg_trade_id"].n_unique()
    removed_count = original_count - unique_count

    if removed_count == 0:
        return original_count, unique_count, 0

    if not dry_run:
        # Deduplicate keeping first occurrence (chronological order)
        df_deduped = df.unique(subset=["agg_trade_id"], maintain_order=True)
        atomic_write_parquet(df_deduped, parquet_path)

    return original_count, unique_count, removed_count


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deduplicate Tier 1 Parquet tick cache (Issue #78)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count duplicates without modifying files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Deduplicate only this symbol (e.g., BINANCE_SPOT_BTCUSDT)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Custom tick cache directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file details",
    )

    args = parser.parse_args()

    cache_dir = args.cache_dir or get_default_cache_dir()

    if not cache_dir.exists():
        logger.error("Cache directory not found: %s", cache_dir)
        return 1

    # Find all Parquet files
    if args.symbol:
        symbol_dir = cache_dir / args.symbol
        if not symbol_dir.exists():
            logger.error("Symbol directory not found: %s", symbol_dir)
            return 1
        parquet_files = sorted(symbol_dir.glob("*.parquet"))
    else:
        parquet_files = sorted(cache_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.info("No Parquet files found in %s", cache_dir)
        return 0

    logger.info("Found %d Parquet files in %s", len(parquet_files), cache_dir)
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    total_original = 0
    total_removed = 0
    files_with_dupes = 0

    for parquet_path in tqdm(parquet_files, desc="Processing", unit="file"):
        try:
            original, unique, removed = deduplicate_file(
                parquet_path, dry_run=args.dry_run
            )
            total_original += original
            total_removed += removed

            if removed > 0:
                files_with_dupes += 1
                if args.verbose:
                    logger.info(
                        "%s: %d -> %d (removed %d)",
                        parquet_path.relative_to(cache_dir),
                        original,
                        unique,
                        removed,
                    )
        except Exception:
            logger.exception("Error processing %s", parquet_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:      {len(parquet_files):,}")
    print(f"Files with duplicates: {files_with_dupes:,}")
    print(f"Total rows (original): {total_original:,}")
    print(f"Duplicates removed:    {total_removed:,}")
    print(f"Total rows (after):    {total_original - total_removed:,}")
    print(f"Storage saved:         ~{total_removed * 50 / 1024 / 1024:.1f} MB (est.)")

    if args.dry_run and total_removed > 0:
        print("\nRun without --dry-run to apply deduplication.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
