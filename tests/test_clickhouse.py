"""Tests for ClickHouse cache functionality.

These tests use mocking to avoid requiring an actual ClickHouse connection.
For integration tests with a real ClickHouse server, see test_clickhouse_integration.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from rangebar.clickhouse import CacheKey, ClickHouseConfig
from rangebar.clickhouse.cache import RangeBarCache
from rangebar.clickhouse.mixin import ClickHouseClientMixin


class TestClickHouseConfig:
    """Tests for ClickHouseConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClickHouseConfig()
        assert config.host == "localhost"
        assert config.port == 8123
        assert config.database == "rangebar_cache"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClickHouseConfig(host="remotehost", port=9000, database="custom_db")
        assert config.host == "remotehost"
        assert config.port == 9000
        assert config.database == "custom_db"

    @patch.dict("os.environ", {"CLICKHOUSE_HOST": "envhost", "CLICKHOUSE_PORT": "9999"})
    def test_from_env(self) -> None:
        """Test loading configuration from environment."""
        config = ClickHouseConfig.from_env()
        assert config.host == "envhost"
        assert config.port == 9999

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_defaults(self) -> None:
        """Test environment defaults when not set."""
        config = ClickHouseConfig.from_env()
        assert config.host == "localhost"
        assert config.port == 8123


class TestCacheKey:
    """Tests for CacheKey dataclass."""

    def test_creation(self) -> None:
        """Test creating a cache key."""
        key = CacheKey(
            symbol="BTCUSDT",
            threshold_decimal_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        assert key.symbol == "BTCUSDT"
        assert key.threshold_decimal_bps == 250
        assert key.start_ts == 1704067200000
        assert key.end_ts == 1704153600000

    def test_hash_key(self) -> None:
        """Test hash key generation."""
        key = CacheKey(
            symbol="BTCUSDT",
            threshold_decimal_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        hash_key = key.hash_key
        assert isinstance(hash_key, str)
        assert len(hash_key) == 32  # MD5 hex digest length

    def test_hash_key_consistency(self) -> None:
        """Test that same inputs produce same hash."""
        key1 = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        key2 = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        assert key1.hash_key == key2.hash_key

    def test_hash_key_uniqueness(self) -> None:
        """Test that different inputs produce different hashes."""
        key1 = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        key2 = CacheKey("ETHUSDT", 250, 1704067200000, 1704153600000)
        key3 = CacheKey("BTCUSDT", 500, 1704067200000, 1704153600000)
        assert key1.hash_key != key2.hash_key
        assert key1.hash_key != key3.hash_key

    def test_frozen(self) -> None:
        """Test that CacheKey is immutable."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        with pytest.raises(AttributeError):
            key.symbol = "ETHUSDT"  # type: ignore[misc]


class TestClickHouseClientMixin:
    """Tests for ClickHouseClientMixin."""

    def test_init_with_external_client(self) -> None:
        """Test initialization with external client."""
        mock_client = MagicMock()

        class TestClass(ClickHouseClientMixin):
            def __init__(self, client: Any) -> None:
                self._init_client(client)

        obj = TestClass(mock_client)
        assert obj.client is mock_client
        assert obj._owns_client is False

    def test_init_without_client(self) -> None:
        """Test initialization without client (owns_client=True)."""

        class TestClass(ClickHouseClientMixin):
            def __init__(self) -> None:
                self._init_client(None)

        obj = TestClass()
        assert obj._owns_client is True
        assert obj._client is None

    def test_close_owned_client(self) -> None:
        """Test closing owned client."""
        mock_client = MagicMock()

        class TestClass(ClickHouseClientMixin):
            def __init__(self) -> None:
                self._init_client(None)
                self._client = mock_client
                self._owns_client = True

        obj = TestClass()
        obj.close()
        mock_client.close.assert_called_once()
        assert obj._client is None

    def test_close_external_client(self) -> None:
        """Test that external clients are not closed."""
        mock_client = MagicMock()

        class TestClass(ClickHouseClientMixin):
            def __init__(self, client: Any) -> None:
                self._init_client(client)

        obj = TestClass(mock_client)
        obj.close()
        mock_client.close.assert_not_called()

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        mock_client = MagicMock()

        class TestClass(ClickHouseClientMixin):
            def __init__(self) -> None:
                self._init_client(None)
                self._client = mock_client
                self._owns_client = True

        with TestClass() as obj:
            assert obj._client is mock_client

        mock_client.close.assert_called_once()


class TestRangeBarCache:
    """Tests for RangeBarCache class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClickHouse client."""
        client = MagicMock()
        client.command.return_value = None
        client.query.return_value = MagicMock(result_rows=[])
        client.query_df.return_value = pd.DataFrame()
        client.query_df_arrow.return_value = pd.DataFrame()  # Arrow-optimized
        # Configure insert_df to return QuerySummary-like object with written_rows
        insert_summary = MagicMock()
        insert_summary.written_rows = 1
        client.insert_df.return_value = insert_summary
        return client

    @pytest.fixture
    def cache(self, mock_client: MagicMock) -> RangeBarCache:
        """Create a RangeBarCache with mock client."""
        with patch.object(RangeBarCache, "_ensure_schema"):
            cache = RangeBarCache(client=mock_client)
        return cache

    def test_init_with_client(self, mock_client: MagicMock) -> None:
        """Test initialization with provided client."""
        with patch.object(RangeBarCache, "_ensure_schema"):
            cache = RangeBarCache(client=mock_client)
            assert cache.client is mock_client
            assert cache._owns_client is False

    @patch("rangebar.clickhouse.cache.get_available_clickhouse_host")
    @patch("rangebar.clickhouse.cache.get_client")
    def test_init_localhost(
        self, mock_get_client: MagicMock, mock_get_host: MagicMock
    ) -> None:
        """Test initialization with localhost connection."""
        from rangebar.clickhouse.preflight import HostConnection

        mock_get_host.return_value = HostConnection(
            host="localhost", method="local", port=8123
        )
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch.object(RangeBarCache, "_ensure_schema"):
            cache = RangeBarCache()

        mock_get_host.assert_called_once()
        mock_get_client.assert_called_once_with("localhost", 8123)

    # Note: Raw trades (Tier 1) tests removed - raw tick data is now stored
    # locally using Parquet files via rangebar.storage.TickStorage.
    # See tests/test_storage.py for tick storage tests.

    def test_store_range_bars_empty(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test storing empty range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        df = pd.DataFrame()
        result = cache.store_range_bars(key, df)
        assert result == 0
        mock_client.insert_df.assert_not_called()

    def test_store_range_bars(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test storing range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        df = pd.DataFrame(
            {
                "Open": [42000.0],
                "High": [42100.0],
                "Low": [41900.0],
                "Close": [42050.0],
                "Volume": [10.5],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00")]),
        )
        result = cache.store_range_bars(key, df)
        assert result == 1
        mock_client.insert_df.assert_called_once()

    def test_get_range_bars_not_found(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test getting non-existent range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        mock_client.query_df_arrow.return_value = pd.DataFrame()

        result = cache.get_range_bars(key)
        assert result is None

    def test_get_range_bars(self, cache: RangeBarCache, mock_client: MagicMock) -> None:
        """Test retrieving range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        mock_client.query_df_arrow.return_value = pd.DataFrame(
            {
                "timestamp_ms": [1704067200000],
                "Open": [42000.0],
                "High": [42100.0],
                "Low": [41900.0],
                "Close": [42050.0],
                "Volume": [10.5],
            }
        )

        result = cache.get_range_bars(key)
        assert result is not None
        assert isinstance(result.index, pd.DatetimeIndex)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_has_range_bars_false(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test checking for non-existent range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        mock_client.command.return_value = 0

        assert cache.has_range_bars(key) is False

    def test_has_range_bars_true(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test checking for existing range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        mock_client.command.return_value = 1

        assert cache.has_range_bars(key) is True

    def test_invalidate_range_bars(
        self, cache: RangeBarCache, mock_client: MagicMock
    ) -> None:
        """Test invalidating range bars."""
        key = CacheKey("BTCUSDT", 250, 1704067200000, 1704153600000)
        result = cache.invalidate_range_bars(key)
        # ClickHouse DELETE is async, returns 0
        assert result == 0
        mock_client.command.assert_called()

    def test_close_with_tunnel(self, mock_client: MagicMock) -> None:
        """Test closing cache with SSH tunnel."""
        with patch.object(RangeBarCache, "_ensure_schema"):
            cache = RangeBarCache(client=mock_client)

        mock_tunnel = MagicMock()
        cache._tunnel = mock_tunnel
        cache._owns_client = True
        cache._client = mock_client

        cache.close()

        mock_tunnel.stop.assert_called_once()
        mock_client.close.assert_called_once()


class TestProcessTradesDataframeCached:
    """Tests for process_trades_to_dataframe_cached function."""

    @patch("rangebar.clickhouse.RangeBarCache")
    def test_cache_hit(self, mock_cache_class: MagicMock) -> None:
        """Test when range bars are found in cache."""
        from rangebar import process_trades_to_dataframe_cached

        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.has_range_bars.return_value = True
        cached_df = pd.DataFrame(
            {
                "Open": [42000.0],
                "High": [42100.0],
                "Low": [41900.0],
                "Close": [42050.0],
                "Volume": [10.5],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        mock_cache.get_range_bars.return_value = cached_df
        mock_cache_class.return_value = mock_cache

        # Test data
        trades = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000],
                "price": [42000.0, 42105.0],
                "quantity": [1.5, 2.3],
            }
        )

        result = process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")

        assert result is cached_df
        mock_cache.has_range_bars.assert_called_once()
        mock_cache.get_range_bars.assert_called_once()
        mock_cache.close.assert_called_once()

    @patch("rangebar.process_trades_to_dataframe")
    @patch("rangebar.clickhouse.RangeBarCache")
    def test_cache_miss(
        self, mock_cache_class: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test when range bars are not in cache."""
        from rangebar import process_trades_to_dataframe_cached

        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.has_range_bars.return_value = False
        mock_cache_class.return_value = mock_cache

        # Setup mock processor
        computed_df = pd.DataFrame(
            {
                "Open": [42000.0],
                "High": [42105.0],
                "Low": [42000.0],
                "Close": [42105.0],
                "Volume": [3.8],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        mock_process.return_value = computed_df

        # Test data
        trades = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000],
                "price": [42000.0, 42105.0],
                "quantity": [1.5, 2.3],
            }
        )

        result = process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")

        assert len(result) == 1
        mock_cache.store_range_bars.assert_called_once()
        mock_cache.close.assert_called_once()

    @patch("rangebar.clickhouse.cache.RangeBarCache")
    def test_external_cache(self, mock_cache_class: MagicMock) -> None:
        """Test using external cache instance."""
        from rangebar import process_trades_to_dataframe_cached

        # Setup external mock cache
        external_cache = MagicMock()
        external_cache.has_range_bars.return_value = True
        cached_df = pd.DataFrame(
            {
                "Open": [42000.0],
                "High": [42100.0],
                "Low": [41900.0],
                "Close": [42050.0],
                "Volume": [10.5],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        external_cache.get_range_bars.return_value = cached_df

        # Test data
        trades = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000],
                "price": [42000.0, 42105.0],
                "quantity": [1.5, 2.3],
            }
        )

        result = process_trades_to_dataframe_cached(
            trades, symbol="BTCUSDT", cache=external_cache
        )

        assert result is cached_df
        # External cache should NOT be closed
        external_cache.close.assert_not_called()
        # RangeBarCache should NOT be instantiated
        mock_cache_class.assert_not_called()

    def test_missing_timestamp_column(self) -> None:
        """Test error when timestamp column is missing."""
        from rangebar import process_trades_to_dataframe_cached
        from rangebar.clickhouse.cache import RangeBarCache

        trades = pd.DataFrame(
            {
                "price": [42000.0, 42105.0],
                "quantity": [1.5, 2.3],
            }
        )

        with patch.object(RangeBarCache, "__init__", return_value=None):
            with pytest.raises(ValueError, match="timestamp"):
                process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")
