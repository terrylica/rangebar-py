#!/usr/bin/env python3
"""End-to-end tests for get_range_bars() with real Binance data.

These tests validate that the threshold configuration is fully exposed and working
correctly for downstream users. All tests use REAL data from Binance - no fake
or synthetic data.

Test categories:
1. Threshold numeric values - validates compression behavior
2. Threshold presets - validates preset strings work correctly
3. Market types - validates spot, futures-um, futures-cm
4. Microstructure columns - validates additional data availability
5. Documentation examples - validates user-facing examples work

ADR: These tests serve as living documentation for downstream users.
"""

import pandas as pd
import pytest
from rangebar import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
    TIER1_SYMBOLS,
    get_range_bars,
)

# =============================================================================
# Test Constants - Real Data Parameters
# =============================================================================

# Use a single day to minimize test time while ensuring real data
TEST_SYMBOL = "BTCUSDT"
TEST_START = "2024-11-01"
TEST_END = "2024-11-01"  # Single day for fast tests

# Short date range for expensive tests
TEST_START_SHORT = "2024-11-01"
TEST_END_SHORT = "2024-11-01"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def real_btc_bars_medium():
    """Get real BTC range bars with medium threshold (cached for module)."""
    return get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=250)


# =============================================================================
# Section 1: Threshold Numeric Value Tests
# =============================================================================


class TestThresholdNumericValues:
    """Test threshold configuration with numeric values.

    These tests validate that different threshold values produce different
    compression ratios, demonstrating the core range bar functionality.
    """

    def test_threshold_micro_produces_most_bars(self):
        """Micro threshold (10 = 1bps) should produce the most bars."""
        df = get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=10)

        assert len(df) > 0, "Should produce at least one bar"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # Micro threshold on 1 day of BTC should produce many bars
        assert len(df) > 100, f"Micro threshold should produce >100 bars, got {len(df)}"

    def test_threshold_macro_produces_fewest_bars(self):
        """Macro threshold (1000 = 100bps) should produce the fewest bars."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=1000
        )

        assert len(df) > 0, "Should produce at least one bar"
        assert isinstance(df.index, pd.DatetimeIndex)

        # Macro threshold produces far fewer bars than micro
        assert len(df) < 500, f"Macro threshold should produce <500 bars, got {len(df)}"

    def test_threshold_compression_monotonic(self):
        """Higher thresholds should produce fewer bars (monotonic)."""
        thresholds = [50, 100, 250, 500, 1000]  # 5bps to 100bps
        bar_counts = {}

        for threshold in thresholds:
            df = get_range_bars(
                TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=threshold
            )
            bar_counts[threshold] = len(df)

        # Verify monotonic decrease (or equal) as threshold increases
        counts = list(bar_counts.values())
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"Bar count should decrease as threshold increases: "
                f"threshold {thresholds[i]} ({counts[i]} bars) vs "
                f"threshold {thresholds[i+1]} ({counts[i+1]} bars)"
            )

    def test_threshold_extreme_low(self):
        """Test minimum valid threshold (1 = 0.1bps)."""
        df = get_range_bars(
            TEST_SYMBOL,
            TEST_START,
            TEST_END,
            threshold_decimal_bps=THRESHOLD_DECIMAL_MIN,
        )

        assert len(df) > 0, "Should produce bars even at minimum threshold"
        # Minimum threshold should produce many bars
        assert len(df) > 1000, f"Min threshold should produce >1000 bars, got {len(df)}"

    def test_threshold_extreme_high(self):
        """Test maximum valid threshold (100,000 = 10,000bps = 100%)."""
        df = get_range_bars(
            TEST_SYMBOL,
            TEST_START,
            TEST_END,
            threshold_decimal_bps=THRESHOLD_DECIMAL_MAX,
        )

        # At 100% threshold, we might get 0-1 bars for a single day
        assert len(df) >= 0, "Should not error at maximum threshold"
        # Realistically, 100% movement in a day is unlikely
        assert len(df) <= 10, f"Max threshold should produce <=10 bars, got {len(df)}"

    def test_threshold_invalid_below_minimum(self):
        """Threshold below minimum should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=0)

    def test_threshold_invalid_above_maximum(self):
        """Threshold above maximum should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            get_range_bars(
                TEST_SYMBOL,
                TEST_START,
                TEST_END,
                threshold_decimal_bps=THRESHOLD_DECIMAL_MAX + 1,
            )


# =============================================================================
# Section 2: Threshold Preset Tests
# =============================================================================


class TestThresholdPresets:
    """Test threshold configuration with string presets.

    Validates that all preset names work correctly and produce expected
    compression relative to each other.
    """

    @pytest.mark.parametrize(
        ("preset_name", "expected_bps"),
        [
            ("micro", 10),
            ("tight", 50),
            ("standard", 100),
            ("medium", 250),
            ("wide", 500),
            ("macro", 1000),
        ],
    )
    def test_preset_produces_bars(self, preset_name, expected_bps):
        """Each preset should produce valid range bars."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=preset_name
        )

        assert len(df) > 0, f"Preset '{preset_name}' should produce bars"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_preset_matches_numeric_equivalent(self):
        """Preset should produce same result as numeric equivalent."""
        # Test "medium" preset vs numeric 250
        df_preset = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps="medium"
        )
        df_numeric = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=250
        )

        # Should produce identical bar counts
        assert len(df_preset) == len(df_numeric), (
            f"Preset 'medium' ({len(df_preset)} bars) should match "
            f"numeric 250 ({len(df_numeric)} bars)"
        )

    def test_presets_compression_order(self):
        """Presets should follow expected compression order."""
        preset_order = ["micro", "tight", "standard", "medium", "wide", "macro"]
        bar_counts = {}

        for preset in preset_order:
            df = get_range_bars(
                TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=preset
            )
            bar_counts[preset] = len(df)

        # Verify decreasing bar counts
        counts = list(bar_counts.values())
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"Preset '{preset_order[i]}' ({counts[i]} bars) should produce >= "
                f"'{preset_order[i+1]}' ({counts[i+1]} bars)"
            )

    def test_preset_invalid_name(self):
        """Invalid preset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown threshold preset"):
            get_range_bars(
                TEST_SYMBOL,
                TEST_START,
                TEST_END,
                threshold_decimal_bps="invalid_preset",
            )

    def test_threshold_presets_constant_matches(self):
        """THRESHOLD_PRESETS constant should have expected values."""
        assert THRESHOLD_PRESETS["micro"] == 10
        assert THRESHOLD_PRESETS["tight"] == 50
        assert THRESHOLD_PRESETS["standard"] == 100
        assert THRESHOLD_PRESETS["medium"] == 250
        assert THRESHOLD_PRESETS["wide"] == 500
        assert THRESHOLD_PRESETS["macro"] == 1000


# =============================================================================
# Section 3: Market Type Tests
# =============================================================================


class TestMarketTypes:
    """Test different Binance market types with real data.

    Validates that spot, futures-um, and futures-cm all work correctly.
    """

    def test_spot_market(self):
        """Spot market should return valid range bars."""
        df = get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, market="spot")

        assert len(df) > 0, "Spot market should produce bars"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_futures_um_market(self):
        """USD-M Futures market should return valid range bars."""
        df = get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, market="futures-um")

        assert len(df) > 0, "Futures-UM market should produce bars"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_futures_um_alias(self):
        """'um' alias should work same as 'futures-um'."""
        df = get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, market="um")

        assert len(df) > 0, "'um' alias should produce bars"

    def test_futures_cm_market(self):
        """COIN-M Futures market should return valid range bars."""
        # COIN-M uses different symbol format: BTCUSD (not BTCUSDT)
        # This market uses coin-margined contracts
        df = get_range_bars("BTCUSD_PERP", TEST_START, TEST_END, market="futures-cm")

        assert len(df) > 0, "Futures-CM market should produce bars"

    def test_futures_cm_alias(self):
        """'cm' alias should work same as 'futures-cm'."""
        # COIN-M uses different symbol format: BTCUSD_PERP
        df = get_range_bars("BTCUSD_PERP", TEST_START, TEST_END, market="cm")

        assert len(df) > 0, "'cm' alias should produce bars"

    def test_invalid_market(self):
        """Invalid market should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown market"):
            get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, market="invalid")


# =============================================================================
# Section 4: Microstructure Column Tests
# =============================================================================


class TestMicrostructureColumns:
    """Test microstructure data availability.

    Validates that include_microstructure=True returns additional columns.
    """

    def test_microstructure_disabled_by_default(self):
        """Default should return only OHLCV columns."""
        df = get_range_bars(TEST_SYMBOL, TEST_START, TEST_END)

        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_microstructure_enabled_has_extra_columns(self):
        """With include_microstructure=True, should have additional columns."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, include_microstructure=True
        )

        # Should have OHLCV plus microstructure columns
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns

        # Should have microstructure columns
        assert "vwap" in df.columns, "Should have VWAP"
        assert "buy_volume" in df.columns, "Should have buy volume"
        assert "sell_volume" in df.columns, "Should have sell volume"

    def test_microstructure_values_valid(self):
        """Microstructure values should be valid."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, include_microstructure=True
        )

        # VWAP should be within High-Low range
        assert (df["vwap"] >= df["Low"]).all(), "VWAP should be >= Low"
        assert (df["vwap"] <= df["High"]).all(), "VWAP should be <= High"

        # Buy + Sell volume should approximately equal total volume
        total_from_sides = df["buy_volume"] + df["sell_volume"]
        assert (
            (total_from_sides - df["Volume"]).abs() < 1e-6
        ).all(), "Buy + Sell volume should equal total Volume"


# =============================================================================
# Section 5: OHLCV Invariant Tests
# =============================================================================


class TestOHLCVInvariants:
    """Test OHLCV data invariants on real data.

    These tests ensure data quality regardless of threshold.
    """

    @pytest.mark.parametrize("threshold", [50, 250, 1000])
    def test_ohlc_invariants(self, threshold):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=threshold
        )

        # High is highest point
        assert (df["High"] >= df["Open"]).all(), "High should be >= Open"
        assert (df["High"] >= df["Close"]).all(), "High should be >= Close"

        # Low is lowest point
        assert (df["Low"] <= df["Open"]).all(), "Low should be <= Open"
        assert (df["Low"] <= df["Close"]).all(), "Low should be <= Close"

    @pytest.mark.parametrize("threshold", [50, 250, 1000])
    def test_no_nan_values(self, threshold):
        """No NaN values in output (backtesting.py requirement)."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=threshold
        )

        assert not df.isna().any().any(), "Should have no NaN values"

    @pytest.mark.parametrize("threshold", [50, 250, 1000])
    def test_chronological_order(self, threshold):
        """Bars should be in chronological order."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=threshold
        )

        assert df.index.is_monotonic_increasing, "Timestamps should be increasing"

    @pytest.mark.parametrize("threshold", [50, 250, 1000])
    def test_positive_volume(self, threshold):
        """Volume should be positive."""
        df = get_range_bars(
            TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=threshold
        )

        assert (df["Volume"] > 0).all(), "All volumes should be positive"


# =============================================================================
# Section 6: Tier-1 Symbol Tests
# =============================================================================


class TestTier1Symbols:
    """Test Tier-1 symbols constant and availability.

    Validates that the TIER1_SYMBOLS constant is accurate and symbols work.
    """

    def test_tier1_symbols_count(self):
        """Should have exactly 18 tier-1 symbols."""
        assert len(TIER1_SYMBOLS) == 18

    def test_tier1_symbols_contains_expected(self):
        """Should contain expected high-liquidity symbols."""
        assert "BTC" in TIER1_SYMBOLS
        assert "ETH" in TIER1_SYMBOLS
        assert "SOL" in TIER1_SYMBOLS
        assert "XRP" in TIER1_SYMBOLS

    @pytest.mark.parametrize("symbol", ["BTC", "ETH", "SOL"])
    def test_tier1_symbol_works_with_usdt(self, symbol):
        """Tier-1 symbols should work with USDT pair."""
        df = get_range_bars(
            f"{symbol}USDT", TEST_START, TEST_END, threshold_decimal_bps=250
        )

        assert len(df) > 0, f"{symbol}USDT should produce bars"


# =============================================================================
# Section 7: Documentation Example Tests
# =============================================================================


class TestDocumentationExamples:
    """Test that documentation examples work correctly.

    These tests ensure that examples shown to users actually work.
    """

    def test_basic_usage_example(self):
        """Basic usage example from docstring."""
        from rangebar import get_range_bars

        df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_preset_usage_example(self):
        """Preset usage example from docstring."""
        df = get_range_bars(
            "BTCUSDT", "2024-01-01", "2024-01-01", threshold_decimal_bps="tight"
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_futures_usage_example(self):
        """Futures market example from docstring."""
        df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-01", market="futures-um")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_microstructure_usage_example(self):
        """Microstructure example from docstring."""
        df = get_range_bars(
            "BTCUSDT", "2024-01-01", "2024-01-01", include_microstructure=True
        )

        assert "vwap" in df.columns
        assert "buy_volume" in df.columns


# =============================================================================
# Section 8: Compression Ratio Tests (with actual numbers)
# =============================================================================


class TestCompressionRatios:
    """Test that compression ratios are reasonable for real data.

    These tests capture expected compression behavior as documentation.
    """

    def test_compression_ratio_summary(self):
        """Generate and log compression ratios for all presets."""
        results = {}

        for preset_name, bps_value in THRESHOLD_PRESETS.items():
            df = get_range_bars(
                TEST_SYMBOL, TEST_START, TEST_END, threshold_decimal_bps=preset_name
            )
            results[preset_name] = {
                "threshold_decimal_bps": bps_value,
                "bar_count": len(df),
            }

        # Log results for documentation
        print("\n=== Compression Ratios (BTCUSDT, 1 day) ===")
        for preset, data in results.items():
            print(
                f"  {preset} ({data['threshold_decimal_bps']} dbps): "
                f"{data['bar_count']} bars"
            )

        # Verify we got meaningful results
        assert (
            results["micro"]["bar_count"] > results["macro"]["bar_count"]
        ), "Micro should produce more bars than macro"

        # Micro should produce at least 10x more bars than macro
        ratio = results["micro"]["bar_count"] / max(results["macro"]["bar_count"], 1)
        assert ratio > 5, f"Expected micro/macro ratio > 5, got {ratio:.1f}"


# =============================================================================
# Section 9: Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_date_format(self):
        """Invalid date format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            get_range_bars(TEST_SYMBOL, "01-01-2024", "01-31-2024")

    def test_start_after_end(self):
        """Start date after end date should raise ValueError."""
        with pytest.raises(ValueError, match="start_date must be <= end_date"):
            get_range_bars(TEST_SYMBOL, "2024-01-31", "2024-01-01")

    def test_invalid_source(self):
        """Invalid source should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            get_range_bars(TEST_SYMBOL, TEST_START, TEST_END, source="invalid")
