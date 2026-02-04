"""Tests for Parquet corruption detection and atomic writes (Issue #73).

This module tests the PAR1 magic byte validation, atomic write pattern,
and auto-recovery from corrupted Parquet files in the TickStorage pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest


class TestPAR1Validation:
    """Test PAR1 magic byte validation."""

    def test_valid_parquet_passes(self, tmp_path: Path) -> None:
        """Valid Parquet file should pass validation."""
        from rangebar.storage.parquet import _is_valid_parquet

        path = tmp_path / "valid.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)

        is_valid, reason = _is_valid_parquet(path)
        assert is_valid
        assert reason == ""

    def test_truncated_file_fails(self, tmp_path: Path) -> None:
        """Truncated file (missing footer) should fail validation."""
        from rangebar.storage.parquet import _is_valid_parquet

        path = tmp_path / "truncated.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)

        # Truncate to simulate kill mid-write
        with path.open("r+b") as f:
            f.truncate(100)

        is_valid, reason = _is_valid_parquet(path)
        assert not is_valid
        assert "magic" in reason.lower()

    def test_garbage_file_fails(self, tmp_path: Path) -> None:
        """Random garbage should fail validation."""
        from rangebar.storage.parquet import _is_valid_parquet

        path = tmp_path / "garbage.parquet"
        path.write_bytes(os.urandom(1000))

        is_valid, reason = _is_valid_parquet(path)
        assert not is_valid
        assert "magic" in reason.lower()

    def test_empty_file_fails(self, tmp_path: Path) -> None:
        """Empty file should fail validation."""
        from rangebar.storage.parquet import _is_valid_parquet

        path = tmp_path / "empty.parquet"
        path.touch()

        is_valid, reason = _is_valid_parquet(path)
        assert not is_valid
        assert "small" in reason.lower()

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        """Non-existent file should fail validation."""
        from rangebar.storage.parquet import _is_valid_parquet

        path = tmp_path / "missing.parquet"

        is_valid, reason = _is_valid_parquet(path)
        assert not is_valid
        assert "exist" in reason.lower()


class TestAtomicWrite:
    """Test atomic write behavior."""

    def test_atomic_write_creates_valid_file(self, tmp_path: Path) -> None:
        """Atomic write should create valid Parquet file."""
        from rangebar.storage.parquet import _atomic_write_parquet, _is_valid_parquet

        path = tmp_path / "atomic.parquet"
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        _atomic_write_parquet(df, path)

        assert path.exists()
        is_valid, _ = _is_valid_parquet(path)
        assert is_valid

        # Verify content
        result = pl.read_parquet(path)
        assert result.equals(df)

    def test_atomic_write_no_temp_file_on_success(self, tmp_path: Path) -> None:
        """Temp files should be cleaned up on success."""
        from rangebar.storage.parquet import _atomic_write_parquet

        path = tmp_path / "atomic.parquet"
        df = pl.DataFrame({"a": [1, 2, 3]})

        _atomic_write_parquet(df, path)

        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Atomic write should overwrite existing file."""
        from rangebar.storage.parquet import _atomic_write_parquet

        path = tmp_path / "overwrite.parquet"

        # Write original
        _atomic_write_parquet(pl.DataFrame({"a": [1]}), path)

        # Overwrite
        _atomic_write_parquet(pl.DataFrame({"a": [2, 3]}), path)

        result = pl.read_parquet(path)
        assert result["a"].to_list() == [2, 3]


class TestAutoRecovery:
    """Test auto-recovery from corrupted files."""

    def test_auto_delete_corrupted_file(self, tmp_path: Path) -> None:
        """Corrupted file should be auto-deleted."""
        from rangebar.storage.parquet import _validate_and_recover_parquet

        path = tmp_path / "corrupted.parquet"
        path.write_bytes(b"not a parquet file")

        result = _validate_and_recover_parquet(path, auto_delete=True)

        assert not result  # Returns False for corrupted
        assert not path.exists()  # File deleted

    def test_raises_when_auto_delete_disabled(self, tmp_path: Path) -> None:
        """Should raise exception when auto_delete=False."""
        from rangebar.exceptions import ParquetCorruptionError
        from rangebar.storage.parquet import _validate_and_recover_parquet

        path = tmp_path / "corrupted.parquet"
        path.write_bytes(b"not a parquet file")

        with pytest.raises(ParquetCorruptionError) as exc_info:
            _validate_and_recover_parquet(path, auto_delete=False)

        assert exc_info.value.path == path
        assert "magic" in exc_info.value.reason.lower()

    def test_valid_file_returns_true(self, tmp_path: Path) -> None:
        """Valid file should return True without modification."""
        from rangebar.storage.parquet import _validate_and_recover_parquet

        path = tmp_path / "valid.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)

        result = _validate_and_recover_parquet(path, auto_delete=True)

        assert result
        assert path.exists()


class TestTickStorageIntegration:
    """Test full TickStorage pipeline with corruption handling."""

    def test_write_ticks_atomic(self, tmp_path: Path) -> None:
        """write_ticks() should use atomic writes."""
        from rangebar.storage.parquet import TickStorage, _is_valid_parquet

        storage = TickStorage(cache_dir=tmp_path)
        df = pl.DataFrame({
            "timestamp": [1704067200000, 1704067201000],  # 2024-01-01
            "price": [50000.0, 50100.0],
            "quantity": [1.0, 1.0],
        })

        storage.write_ticks("BTCUSDT", df)

        # Find the written file
        files = list((tmp_path / "ticks" / "BTCUSDT").glob("*.parquet"))
        assert len(files) == 1

        is_valid, _ = _is_valid_parquet(files[0])
        assert is_valid

    def test_read_ticks_detects_corruption(self, tmp_path: Path) -> None:
        """read_ticks() should detect and recover from corruption."""
        from rangebar.storage.parquet import TickStorage

        storage = TickStorage(cache_dir=tmp_path)

        # Create a corrupted file manually
        symbol_dir = tmp_path / "ticks" / "BTCUSDT"
        symbol_dir.mkdir(parents=True)
        corrupted_path = symbol_dir / "2024-01.parquet"
        corrupted_path.write_bytes(b"corrupted data")

        # Read should return empty (corrupted file deleted)
        result = storage.read_ticks("BTCUSDT", 1704067200000, 1704153600000)

        assert result.is_empty()
        assert not corrupted_path.exists()  # Corrupted file deleted

    def test_write_ticks_recovers_from_existing_corruption(
        self, tmp_path: Path
    ) -> None:
        """write_ticks() should recover from pre-existing corrupted file."""
        from rangebar.storage.parquet import TickStorage, _is_valid_parquet

        storage = TickStorage(cache_dir=tmp_path)

        # Create a corrupted file manually
        symbol_dir = tmp_path / "ticks" / "BTCUSDT"
        symbol_dir.mkdir(parents=True)
        corrupted_path = symbol_dir / "2024-01.parquet"
        corrupted_path.write_bytes(b"corrupted data")

        # Write new data - should detect corruption and replace
        df = pl.DataFrame({
            "timestamp": [1704067200000, 1704067201000],  # 2024-01-01
            "price": [50000.0, 50100.0],
            "quantity": [1.0, 1.0],
        })
        storage.write_ticks("BTCUSDT", df)

        # File should now be valid
        is_valid, _ = _is_valid_parquet(corrupted_path)
        assert is_valid

        # And contain the new data
        result = pl.read_parquet(corrupted_path)
        assert len(result) == 2

    def test_read_ticks_streaming_detects_corruption(self, tmp_path: Path) -> None:
        """read_ticks_streaming() should detect and recover from corruption."""
        from rangebar.storage.parquet import TickStorage

        storage = TickStorage(cache_dir=tmp_path)

        # Create a corrupted file manually
        symbol_dir = tmp_path / "ticks" / "BTCUSDT"
        symbol_dir.mkdir(parents=True)
        corrupted_path = symbol_dir / "2024-01.parquet"
        corrupted_path.write_bytes(b"corrupted data")

        # Read streaming should return no chunks (corrupted file deleted)
        chunks = list(
            storage.read_ticks_streaming("BTCUSDT", 1704067200000, 1704153600000)
        )

        assert len(chunks) == 0
        assert not corrupted_path.exists()  # Corrupted file deleted
