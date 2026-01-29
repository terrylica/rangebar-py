# polars-exception: ClickHouse cache returns Pandas for backtesting.py compatibility
"""Tests for cache schema evolution functionality (Issue #39).

Tests the three-layer defense against stale cached data:
1. Schema version filtering in SQL queries
2. Content-based staleness detection
3. Schema version validation utilities
"""

from __future__ import annotations

import pandas as pd
import pytest
from rangebar import (
    MIN_VERSION_FOR_MICROSTRUCTURE,
    SCHEMA_VERSION_MICROSTRUCTURE,
    SCHEMA_VERSION_OHLCV_ONLY,
    StalenessResult,
    detect_staleness,
)
from rangebar.validation.cache_staleness import validate_schema_version


class TestSchemaVersionConstants:
    """Tests for schema version constants."""

    def test_schema_version_ordering(self) -> None:
        """Test that schema versions are properly ordered."""
        # OHLCV-only < Microstructure
        assert SCHEMA_VERSION_OHLCV_ONLY < SCHEMA_VERSION_MICROSTRUCTURE
        assert SCHEMA_VERSION_OHLCV_ONLY == "6.0.0"
        assert SCHEMA_VERSION_MICROSTRUCTURE == "7.0.0"

    def test_min_version_for_microstructure(self) -> None:
        """Test minimum version for microstructure features."""
        assert MIN_VERSION_FOR_MICROSTRUCTURE == "7.0.0"
        assert MIN_VERSION_FOR_MICROSTRUCTURE == SCHEMA_VERSION_MICROSTRUCTURE


class TestSchemaVersionValidation:
    """Tests for validate_schema_version function."""

    def test_equal_versions(self) -> None:
        """Test that equal versions pass validation."""
        assert validate_schema_version("7.0.0", "7.0.0")
        assert validate_schema_version("11.0.0", "11.0.0")

    def test_greater_versions(self) -> None:
        """Test that greater versions pass validation."""
        assert validate_schema_version("11.0.0", "7.0.0")
        assert validate_schema_version("7.1.0", "7.0.0")
        assert validate_schema_version("7.0.1", "7.0.0")
        assert validate_schema_version("8.0.0", "7.0.0")

    def test_lesser_versions(self) -> None:
        """Test that lesser versions fail validation."""
        assert not validate_schema_version("6.0.0", "7.0.0")
        assert not validate_schema_version("6.9.9", "7.0.0")
        assert not validate_schema_version("1.0.0", "7.0.0")

    def test_none_version(self) -> None:
        """Test that None version fails validation."""
        assert not validate_schema_version(None, "7.0.0")

    def test_empty_version(self) -> None:
        """Test that empty string version fails validation."""
        assert not validate_schema_version("", "7.0.0")

    def test_invalid_version_format(self) -> None:
        """Test that invalid version formats fail gracefully."""
        # Should return False, not raise
        assert not validate_schema_version("invalid", "7.0.0")
        assert not validate_schema_version("7.x.0", "7.0.0")

    def test_partial_versions(self) -> None:
        """Test handling of partial version strings."""
        # Should be padded with zeros
        assert validate_schema_version("7", "7.0.0")
        assert validate_schema_version("7.0", "7.0.0")
        assert not validate_schema_version("6.9", "7.0.0")


class TestStalenessDetection:
    """Tests for detect_staleness function."""

    @pytest.fixture
    def fresh_microstructure_df(self) -> pd.DataFrame:
        """Create a DataFrame with valid microstructure features."""
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1000.0, 1100.0, 1200.0],
                "vwap": [100.5, 101.5, 102.5],  # Valid: within [Low, High]
                "buy_volume": [600.0, 650.0, 700.0],
                "sell_volume": [400.0, 450.0, 500.0],
                "ofi": [0.2, 0.18, 0.17],  # Valid: within [-1, 1]
                "turnover_imbalance": [0.1, 0.09, 0.08],  # Valid: within [-1, 1]
                "duration_us": [60000000, 55000000, 50000000],  # Valid: non-negative
                "aggregation_density": [2.5, 3.0, 2.8],  # Valid: >= 1
                "individual_trade_count": [50, 55, 60],  # Valid: >= 1
            }
        )

    @pytest.fixture
    def stale_vwap_zero_df(self) -> pd.DataFrame:
        """Create a DataFrame with all VWAP values as zero (pre-v7.0 cache)."""
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1000.0, 1100.0, 1200.0],
                "vwap": [0.0, 0.0, 0.0],  # STALE: all zeros
                "buy_volume": [600.0, 650.0, 700.0],
                "sell_volume": [400.0, 450.0, 500.0],
                "ofi": [0.2, 0.18, 0.17],
                "turnover_imbalance": [0.1, 0.09, 0.08],
                "duration_us": [60000000, 55000000, 50000000],
                "aggregation_density": [2.5, 3.0, 2.8],
                "individual_trade_count": [50, 55, 60],
            }
        )

    @pytest.fixture
    def stale_all_microstructure_zero_df(self) -> pd.DataFrame:
        """Create a DataFrame with all microstructure columns as zero."""
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000.0, 1100.0],
                "vwap": [0.0, 0.0],
                "buy_volume": [0.0, 0.0],
                "sell_volume": [0.0, 0.0],
                "ofi": [0.0, 0.0],
                "turnover_imbalance": [0.0, 0.0],
                "duration_us": [0, 0],
                "aggregation_density": [0.0, 0.0],
                "individual_trade_count": [0, 0],
            }
        )

    def test_fresh_data_not_stale(self, fresh_microstructure_df: pd.DataFrame) -> None:
        """Test that valid microstructure data is not detected as stale."""
        result = detect_staleness(fresh_microstructure_df, require_microstructure=True)
        assert not result.is_stale
        assert result.reason is None
        assert result.confidence == "high"
        assert len(result.recommendations) == 0

    def test_vwap_zero_detected_as_stale(
        self, stale_vwap_zero_df: pd.DataFrame
    ) -> None:
        """Test that all-zero VWAP is detected as stale."""
        result = detect_staleness(stale_vwap_zero_df, require_microstructure=True)
        assert result.is_stale
        assert "VWAP" in result.reason or "vwap" in result.reason.lower()
        assert result.confidence == "high"
        assert len(result.recommendations) > 0

    def test_all_microstructure_zero_detected(
        self, stale_all_microstructure_zero_df: pd.DataFrame
    ) -> None:
        """Test that all-zero microstructure columns are detected."""
        result = detect_staleness(
            stale_all_microstructure_zero_df, require_microstructure=True
        )
        assert result.is_stale
        assert (
            "microstructure" in result.reason.lower() or "zero" in result.reason.lower()
        )

    def test_ohlcv_only_not_checked_when_microstructure_not_required(self) -> None:
        """Test that OHLCV-only data passes when microstructure not required."""
        ohlcv_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000.0, 1100.0],
            }
        )
        result = detect_staleness(ohlcv_df, require_microstructure=False)
        assert not result.is_stale

    def test_staleness_result_structure(
        self, fresh_microstructure_df: pd.DataFrame
    ) -> None:
        """Test that StalenessResult has expected structure."""
        result = detect_staleness(fresh_microstructure_df, require_microstructure=True)
        assert isinstance(result, StalenessResult)
        assert isinstance(result.is_stale, bool)
        assert isinstance(result.checks_passed, dict)
        assert isinstance(result.recommendations, list)
        assert result.confidence in ("high", "medium", "low")


class TestStalenessDetectionEdgeCases:
    """Edge case tests for staleness detection."""

    def test_vwap_outside_high_low_bounds(self) -> None:
        """Test that VWAP outside [Low, High] is detected."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [150.0],  # INVALID: above High
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        assert result.is_stale
        assert "vwap" in result.reason.lower() or "VWAP" in result.reason

    def test_ofi_outside_bounds(self) -> None:
        """Test that OFI outside [-1, 1] is detected."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [1.5],  # INVALID: > 1
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        assert result.is_stale
        assert "ofi" in result.reason.lower() or "OFI" in result.reason

    def test_negative_duration_detected(self) -> None:
        """Test that negative duration is detected."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [-1000],  # INVALID: negative
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        assert result.is_stale
        assert "duration" in result.reason.lower()

    def test_aggregation_density_less_than_one(self) -> None:
        """Test that aggregation_density < 1 is detected."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [0.5],  # INVALID: < 1
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        assert result.is_stale
        assert (
            "aggregation" in result.reason.lower() or "density" in result.reason.lower()
        )

    def test_volume_consistency_check(self) -> None:
        """Test that buy_volume + sell_volume != Volume is logged but not stale.

        Volume consistency is a warning check, not a staleness indicator.
        This is because small floating-point differences are common and
        don't indicate pre-v7.0 cache data.
        """
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [100.0],  # Note: 100 + 200 != 1000
                "sell_volume": [200.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        # Volume consistency is a warning, not a staleness indicator
        assert not result.is_stale
        # But the check should still be recorded as failed
        assert "volume_consistency" in result.checks_passed
        assert not result.checks_passed["volume_consistency"]

    def test_invalid_trade_count(self) -> None:
        """Test that trade count < 1 is detected."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [0],  # INVALID: < 1
            }
        )
        result = detect_staleness(df, require_microstructure=True)
        assert result.is_stale
        assert "trade" in result.reason.lower() or "count" in result.reason.lower()


class TestStalenessResultRecommendations:
    """Tests for staleness detection recommendations."""

    def test_stale_data_has_recommendations(self) -> None:
        """Test that stale data includes recommendations."""
        stale_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [0.0],  # STALE
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(stale_df, require_microstructure=True)
        assert result.is_stale
        assert len(result.recommendations) > 0
        # Should recommend invalidating cache and recomputing
        recommendations_text = " ".join(result.recommendations).lower()
        assert (
            "invalidate" in recommendations_text or "recompute" in recommendations_text
        )

    def test_fresh_data_has_no_recommendations(self) -> None:
        """Test that fresh data has no recommendations."""
        fresh_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000.0],
                "vwap": [100.5],
                "buy_volume": [600.0],
                "sell_volume": [400.0],
                "ofi": [0.2],
                "turnover_imbalance": [0.1],
                "duration_us": [60000000],
                "aggregation_density": [2.5],
                "individual_trade_count": [50],
            }
        )
        result = detect_staleness(fresh_df, require_microstructure=True)
        assert not result.is_stale
        assert len(result.recommendations) == 0
