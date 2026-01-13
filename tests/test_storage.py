"""Tests for local Parquet tick storage module.

These tests validate the TickStorage class which provides local Parquet
storage for raw tick data (Tier 1 cache), replacing the previous ClickHouse
implementation for better portability.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from rangebar.storage import TickStorage, get_cache_dir


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_cache_dir returns a Path object."""
        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that RANGEBAR_CACHE_DIR env var overrides default."""
        custom_path = "/custom/cache/path"
        monkeypatch.setenv("RANGEBAR_CACHE_DIR", custom_path)

        result = get_cache_dir()
        assert result == Path(custom_path)

    def test_default_contains_rangebar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default path contains 'rangebar'."""
        monkeypatch.delenv("RANGEBAR_CACHE_DIR", raising=False)

        result = get_cache_dir()
        assert "rangebar" in str(result).lower()


class TestTickStorage:
    """Tests for TickStorage class."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def storage(self, temp_cache_dir: Path) -> TickStorage:
        """Create a TickStorage instance with temp directory."""
        return TickStorage(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_ticks_polars(self) -> pl.DataFrame:
        """Generate sample tick data as polars DataFrame."""
        import numpy as np

        np.random.seed(42)
        n_ticks = 100

        base_timestamp = 1704067200000  # 2024-01-01 00:00:00 UTC
        base_price = 42000.0

        returns = np.random.randn(n_ticks) * 0.001
        prices = base_price * np.cumprod(1 + returns)

        return pl.DataFrame(
            {
                "timestamp": [base_timestamp + i * 100 for i in range(n_ticks)],
                "price": prices.tolist(),
                "quantity": np.random.exponential(1.0, n_ticks).tolist(),
            }
        )

    @pytest.fixture
    def sample_ticks_pandas(self) -> pd.DataFrame:
        """Generate sample tick data as pandas DataFrame."""
        import numpy as np

        np.random.seed(42)
        n_ticks = 100

        base_timestamp = 1704067200000  # 2024-01-01 00:00:00 UTC
        base_price = 42000.0

        returns = np.random.randn(n_ticks) * 0.001
        prices = base_price * np.cumprod(1 + returns)

        return pd.DataFrame(
            {
                "timestamp": [base_timestamp + i * 100 for i in range(n_ticks)],
                "price": prices,
                "quantity": np.random.exponential(1.0, n_ticks),
            }
        )

    def test_init_default_cache_dir(self) -> None:
        """Test initialization with default cache directory."""
        storage = TickStorage()
        assert storage.cache_dir is not None
        assert isinstance(storage.cache_dir, Path)

    def test_init_custom_cache_dir(self, temp_cache_dir: Path) -> None:
        """Test initialization with custom cache directory."""
        storage = TickStorage(cache_dir=temp_cache_dir)
        assert storage.cache_dir == temp_cache_dir

    def test_init_string_cache_dir(self, tmp_path: Path) -> None:
        """Test initialization with string cache directory."""
        cache_str = str(tmp_path / "string_cache")
        storage = TickStorage(cache_dir=cache_str)
        assert storage.cache_dir == Path(cache_str)

    def test_write_ticks_empty(self, storage: TickStorage) -> None:
        """Test writing empty DataFrame."""
        empty_df = pl.DataFrame()
        count = storage.write_ticks("BTCUSDT", empty_df)
        assert count == 0

    def test_write_ticks_polars(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test writing polars DataFrame."""
        count = storage.write_ticks("BTCUSDT", sample_ticks_polars)
        assert count == len(sample_ticks_polars)

        # Verify files created
        symbol_dir = storage.ticks_dir / "BTCUSDT"
        assert symbol_dir.exists()
        parquet_files = list(symbol_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

    def test_write_ticks_pandas(
        self, storage: TickStorage, sample_ticks_pandas: pd.DataFrame
    ) -> None:
        """Test writing pandas DataFrame."""
        count = storage.write_ticks("BTCUSDT", sample_ticks_pandas)
        assert count == len(sample_ticks_pandas)

    def test_write_ticks_partitioned_by_month(self, storage: TickStorage) -> None:
        """Test that ticks are partitioned by month."""
        # Create ticks spanning multiple months
        ticks = pl.DataFrame(
            {
                "timestamp": [
                    1704067200000,  # 2024-01-01
                    1706745600000,  # 2024-02-01
                    1709251200000,  # 2024-03-01
                ],
                "price": [42000.0, 43000.0, 44000.0],
                "quantity": [1.0, 2.0, 3.0],
            }
        )

        storage.write_ticks("BTCUSDT", ticks)

        # Should have 3 separate parquet files
        symbol_dir = storage.ticks_dir / "BTCUSDT"
        parquet_files = list(symbol_dir.glob("*.parquet"))
        assert len(parquet_files) == 3

        # Check file names
        file_names = {f.stem for f in parquet_files}
        assert "2024-01" in file_names
        assert "2024-02" in file_names
        assert "2024-03" in file_names

    def test_read_ticks_empty_symbol(self, storage: TickStorage) -> None:
        """Test reading ticks for non-existent symbol."""
        df = storage.read_ticks("NONEXISTENT")
        assert df.is_empty()

    def test_read_ticks_full(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test reading all ticks for a symbol."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        df = storage.read_ticks("BTCUSDT")
        assert len(df) == len(sample_ticks_polars)
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "quantity" in df.columns

    def test_read_ticks_time_range(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test reading ticks within time range."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        start_ts = int(sample_ticks_polars["timestamp"].min()) + 1000
        end_ts = int(sample_ticks_polars["timestamp"].max()) - 1000

        df = storage.read_ticks("BTCUSDT", start_ts, end_ts)
        assert len(df) > 0
        assert len(df) < len(sample_ticks_polars)

        # All timestamps should be within range
        assert df["timestamp"].min() >= start_ts
        assert df["timestamp"].max() <= end_ts

    def test_read_ticks_sorted(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test that read ticks are sorted by timestamp."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        df = storage.read_ticks("BTCUSDT")
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_has_ticks_false(self, storage: TickStorage) -> None:
        """Test has_ticks returns False for non-existent data."""
        assert storage.has_ticks("BTCUSDT", 1704067200000, 1704153600000) is False

    def test_has_ticks_true(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test has_ticks returns True when data exists."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        start_ts = int(sample_ticks_polars["timestamp"].min())
        end_ts = int(sample_ticks_polars["timestamp"].max())

        assert storage.has_ticks("BTCUSDT", start_ts, end_ts) is True

    def test_has_ticks_insufficient_coverage(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test has_ticks with insufficient coverage."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        start_ts = int(sample_ticks_polars["timestamp"].min())
        # Request much larger range than available
        end_ts = start_ts + 1000000000  # Way beyond data

        assert (
            storage.has_ticks("BTCUSDT", start_ts, end_ts, min_coverage=0.95) is False
        )

    def test_list_symbols_empty(self, storage: TickStorage) -> None:
        """Test listing symbols when no data exists."""
        symbols = storage.list_symbols()
        assert symbols == []

    def test_list_symbols(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test listing symbols."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)
        storage.write_ticks("ETHUSDT", sample_ticks_polars)

        symbols = storage.list_symbols()
        assert sorted(symbols) == ["BTCUSDT", "ETHUSDT"]

    def test_list_months_empty(self, storage: TickStorage) -> None:
        """Test listing months for non-existent symbol."""
        months = storage.list_months("BTCUSDT")
        assert months == []

    def test_list_months(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test listing months for a symbol."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        months = storage.list_months("BTCUSDT")
        assert len(months) > 0
        assert "2024-01" in months

    def test_delete_ticks_specific_month(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test deleting ticks for a specific month."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)
        months_before = storage.list_months("BTCUSDT")
        assert len(months_before) > 0

        result = storage.delete_ticks("BTCUSDT", months_before[0])
        assert result is True

        months_after = storage.list_months("BTCUSDT")
        assert months_before[0] not in months_after

    def test_delete_ticks_all(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test deleting all ticks for a symbol."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)
        assert "BTCUSDT" in storage.list_symbols()

        result = storage.delete_ticks("BTCUSDT")
        assert result is True
        assert "BTCUSDT" not in storage.list_symbols()

    def test_delete_ticks_nonexistent(self, storage: TickStorage) -> None:
        """Test deleting ticks for non-existent symbol."""
        result = storage.delete_ticks("NONEXISTENT")
        assert result is False

    def test_get_stats_nonexistent(self, storage: TickStorage) -> None:
        """Test getting stats for non-existent symbol."""
        stats = storage.get_stats("NONEXISTENT")
        assert stats["exists"] is False
        assert stats["file_count"] == 0
        assert stats["total_rows"] == 0

    def test_get_stats(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test getting stats for a symbol."""
        storage.write_ticks("BTCUSDT", sample_ticks_polars)

        stats = storage.get_stats("BTCUSDT")
        assert stats["exists"] is True
        assert stats["symbol"] == "BTCUSDT"
        assert stats["file_count"] > 0
        assert stats["total_rows"] == len(sample_ticks_polars)
        assert stats["total_size_bytes"] > 0
        assert "zstd" in stats["compression"]

    def test_write_append_existing(
        self, storage: TickStorage, sample_ticks_polars: pl.DataFrame
    ) -> None:
        """Test that writing to existing month appends data."""
        # Write initial data
        storage.write_ticks("BTCUSDT", sample_ticks_polars)
        initial_count = len(storage.read_ticks("BTCUSDT"))

        # Write more data with same timestamps (overlapping month)
        storage.write_ticks("BTCUSDT", sample_ticks_polars)
        final_count = len(storage.read_ticks("BTCUSDT"))

        # Should have doubled (data appended)
        assert final_count == initial_count * 2


class TestVectorizedYearMonth:
    """Tests for MEM-001 fix: vectorized _timestamp_to_year_month."""

    def test_vectorized_year_month_extraction(self) -> None:
        """Verify vectorized Polars year-month extraction produces correct results."""
        # Test data spanning multiple months
        timestamps = [
            1704067200000,  # 2024-01-01 00:00:00 UTC
            1706745600000,  # 2024-02-01 00:00:00 UTC
            1709251200000,  # 2024-03-01 00:00:00 UTC
            1735689600000,  # 2025-01-01 00:00:00 UTC (year boundary)
        ]

        df = pl.DataFrame({"ts": timestamps})

        # This is the vectorized approach (MEM-001 fix)
        result = df.with_columns(
            pl.col("ts")
            .cast(pl.Datetime(time_unit="ms"))
            .dt.strftime("%Y-%m")
            .alias("ym")
        )

        expected = ["2024-01", "2024-02", "2024-03", "2025-01"]
        assert result["ym"].to_list() == expected

    def test_vectorized_handles_boundary_timestamps(self) -> None:
        """Verify boundary cases: midnight UTC transitions."""
        # Exact midnight UTC on month boundaries
        timestamps = [
            1704067199999,  # 2023-12-31 23:59:59.999 UTC
            1704067200000,  # 2024-01-01 00:00:00.000 UTC
            1704067200001,  # 2024-01-01 00:00:00.001 UTC
        ]

        df = pl.DataFrame({"ts": timestamps})
        result = df.with_columns(
            pl.col("ts")
            .cast(pl.Datetime(time_unit="ms"))
            .dt.strftime("%Y-%m")
            .alias("ym")
        )

        expected = ["2023-12", "2024-01", "2024-01"]
        assert result["ym"].to_list() == expected


class TestTickStorageCompression:
    """Tests for compression behavior."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> TickStorage:
        """Create a TickStorage instance with temp directory."""
        return TickStorage(cache_dir=tmp_path / "cache")

    def test_compression_applied(self, storage: TickStorage) -> None:
        """Test that ZSTD compression is applied to files."""
        import numpy as np

        # Create larger dataset to ensure compression is meaningful
        n_ticks = 10000
        base_timestamp = 1704067200000

        ticks = pl.DataFrame(
            {
                "timestamp": [base_timestamp + i * 100 for i in range(n_ticks)],
                "price": np.random.randn(n_ticks).tolist(),
                "quantity": np.random.exponential(1.0, n_ticks).tolist(),
            }
        )

        storage.write_ticks("BTCUSDT", ticks)

        # Get file size
        parquet_files = list((storage.ticks_dir / "BTCUSDT").glob("*.parquet"))
        assert len(parquet_files) == 1

        file_size = parquet_files[0].stat().st_size
        # Raw size would be roughly: 10000 * (8 + 8 + 8) = 240KB
        # Compressed should be much smaller
        assert (
            file_size < 240000
        ), f"File size {file_size} is too large, compression may not be working"
