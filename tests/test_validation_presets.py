"""Test tiered validation system (Issue #19 - v6.2.0+)."""

import warnings

import pandas as pd
import pytest
from rangebar import (
    ASSET_CLASS_MULTIPLIERS,
    VALIDATION_PRESETS,
    AssetClass,
    ContinuityError,
    ContinuityWarning,
    GapInfo,
    GapTier,
    TieredValidationResult,
    TierSummary,
    TierThresholds,
    ValidationPreset,
    detect_asset_class,
    validate_continuity_tiered,
)


class TestGapTier:
    """Test GapTier enum."""

    def test_gap_tier_ordering(self):
        """Test that gap tiers are ordered by severity."""
        assert GapTier.PRECISION < GapTier.NOISE
        assert GapTier.NOISE < GapTier.MARKET_MOVE
        assert GapTier.MARKET_MOVE < GapTier.MICROSTRUCTURE
        assert GapTier.MICROSTRUCTURE < GapTier.SESSION_BOUNDARY

    def test_gap_tier_values(self):
        """Test that gap tier values are as expected."""
        assert GapTier.PRECISION == 1
        assert GapTier.NOISE == 2
        assert GapTier.MARKET_MOVE == 3
        assert GapTier.MICROSTRUCTURE == 4
        assert GapTier.SESSION_BOUNDARY == 5


class TestAssetClass:
    """Test AssetClass enum."""

    def test_asset_class_values(self):
        """Test that asset class values are as expected."""
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.FOREX.value == "forex"
        assert AssetClass.EQUITIES.value == "equities"
        assert AssetClass.UNKNOWN.value == "unknown"

    def test_asset_class_multipliers_exist(self):
        """Test that all asset classes have multipliers."""
        for asset_class in AssetClass:
            assert asset_class in ASSET_CLASS_MULTIPLIERS


class TestDetectAssetClass:
    """Test detect_asset_class() function."""

    def test_crypto_btc_usdt(self):
        """Test detection of BTC pairs as crypto."""
        assert detect_asset_class("BTCUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("btcusdt") == AssetClass.CRYPTO  # Case insensitive

    def test_crypto_eth_usdt(self):
        """Test detection of ETH pairs as crypto."""
        assert detect_asset_class("ETHUSDT") == AssetClass.CRYPTO

    def test_crypto_other_bases(self):
        """Test detection of other crypto bases."""
        assert detect_asset_class("BNBUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("SOLUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("XRPUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("ADAUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("DOGEUSDT") == AssetClass.CRYPTO

    def test_crypto_stablecoin_suffixes(self):
        """Test detection of stablecoin-quoted pairs as crypto."""
        assert detect_asset_class("ANYUSDT") == AssetClass.CRYPTO
        assert detect_asset_class("ANYBUSD") == AssetClass.CRYPTO
        assert detect_asset_class("ANYUSDC") == AssetClass.CRYPTO
        assert detect_asset_class("ANYTUSD") == AssetClass.CRYPTO
        assert detect_asset_class("ANYFDUSD") == AssetClass.CRYPTO

    def test_forex_standard_pairs(self):
        """Test detection of standard forex pairs."""
        assert detect_asset_class("EURUSD") == AssetClass.FOREX
        assert detect_asset_class("GBPUSD") == AssetClass.FOREX
        assert detect_asset_class("USDJPY") == AssetClass.FOREX
        assert detect_asset_class("AUDUSD") == AssetClass.FOREX
        assert detect_asset_class("USDCHF") == AssetClass.FOREX
        assert detect_asset_class("USDCAD") == AssetClass.FOREX
        assert detect_asset_class("NZDUSD") == AssetClass.FOREX

    def test_forex_commodities(self):
        """Test detection of commodity symbols via forex brokers."""
        assert detect_asset_class("XAUUSD") == AssetClass.FOREX
        assert detect_asset_class("XAGUSD") == AssetClass.FOREX
        assert detect_asset_class("BRENT") == AssetClass.FOREX
        assert detect_asset_class("WTI") == AssetClass.FOREX

    def test_unknown_symbols(self):
        """Test detection of unknown symbols."""
        assert detect_asset_class("AAPL") == AssetClass.UNKNOWN
        assert detect_asset_class("MSFT") == AssetClass.UNKNOWN
        assert detect_asset_class("RANDOM") == AssetClass.UNKNOWN


class TestTierThresholds:
    """Test TierThresholds dataclass."""

    def test_default_values(self):
        """Test default tier threshold values."""
        thresholds = TierThresholds()
        assert thresholds.precision == 0.00001  # 0.001%
        assert thresholds.noise == 0.0001  # 0.01%
        assert thresholds.market_move == 0.001  # 0.1%
        assert thresholds.session_factor == 2.0

    def test_custom_values(self):
        """Test custom tier threshold values."""
        thresholds = TierThresholds(
            precision=0.00002,
            noise=0.0002,
            market_move=0.002,
            session_factor=3.0,
        )
        assert thresholds.precision == 0.00002
        assert thresholds.noise == 0.0002
        assert thresholds.market_move == 0.002
        assert thresholds.session_factor == 3.0

    def test_immutability(self):
        """Test that TierThresholds is immutable (frozen)."""
        thresholds = TierThresholds()
        with pytest.raises(AttributeError):
            thresholds.precision = 0.0001  # type: ignore


class TestValidationPreset:
    """Test ValidationPreset dataclass."""

    def test_preset_creation(self):
        """Test creating a validation preset."""
        preset = ValidationPreset(
            tolerance_pct=0.01,
            mode="warn",
            description="Test preset",
        )
        assert preset.tolerance_pct == 0.01
        assert preset.mode == "warn"
        assert preset.description == "Test preset"
        assert preset.asset_class is None

    def test_preset_with_asset_class(self):
        """Test preset with explicit asset class."""
        preset = ValidationPreset(
            tolerance_pct=0.02,
            mode="error",
            asset_class=AssetClass.CRYPTO,
        )
        assert preset.asset_class == AssetClass.CRYPTO

    def test_preset_immutability(self):
        """Test that ValidationPreset is immutable (frozen)."""
        preset = ValidationPreset(tolerance_pct=0.01, mode="warn")
        with pytest.raises(AttributeError):
            preset.tolerance_pct = 0.02  # type: ignore


class TestValidationPresets:
    """Test VALIDATION_PRESETS dictionary."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        expected_presets = [
            "permissive",
            "research",
            "standard",
            "strict",
            "paranoid",
            "crypto",
            "forex",
            "equities",
            "skip",
            "audit",
        ]
        for preset_name in expected_presets:
            assert preset_name in VALIDATION_PRESETS

    def test_preset_tolerances_ordering(self):
        """Test that preset tolerances are in logical order."""
        # General-purpose: permissive > research > standard > strict > paranoid
        assert (
            VALIDATION_PRESETS["permissive"].tolerance_pct
            > VALIDATION_PRESETS["research"].tolerance_pct
        )
        assert (
            VALIDATION_PRESETS["research"].tolerance_pct
            > VALIDATION_PRESETS["standard"].tolerance_pct
        )
        assert (
            VALIDATION_PRESETS["standard"].tolerance_pct
            > VALIDATION_PRESETS["strict"].tolerance_pct
        )
        assert (
            VALIDATION_PRESETS["strict"].tolerance_pct
            > VALIDATION_PRESETS["paranoid"].tolerance_pct
        )

    def test_asset_class_presets_have_correct_class(self):
        """Test that asset-class presets have correct asset_class set."""
        assert VALIDATION_PRESETS["crypto"].asset_class == AssetClass.CRYPTO
        assert VALIDATION_PRESETS["forex"].asset_class == AssetClass.FOREX
        assert VALIDATION_PRESETS["equities"].asset_class == AssetClass.EQUITIES

    def test_skip_preset_disables_validation(self):
        """Test that 'skip' preset has mode='skip'."""
        assert VALIDATION_PRESETS["skip"].mode == "skip"


class TestTieredValidationResult:
    """Test TieredValidationResult dataclass."""

    def test_result_properties(self):
        """Test result properties."""
        gaps_by_tier = {tier: TierSummary() for tier in GapTier}
        gaps_by_tier[GapTier.MICROSTRUCTURE] = TierSummary(count=2, max_gap_pct=0.005)

        result = TieredValidationResult(
            is_valid=True,
            bar_count=100,
            gaps_by_tier=gaps_by_tier,
            all_gaps=[],
            threshold_used_pct=0.025,
            asset_class_detected=AssetClass.CRYPTO,
            preset_used="research",
        )

        assert result.is_valid is True
        assert result.bar_count == 100
        assert result.has_microstructure_events is True
        assert result.has_session_breaks is False

    def test_summary_dict(self):
        """Test summary_dict method."""
        gaps_by_tier = {tier: TierSummary() for tier in GapTier}
        gaps_by_tier[GapTier.NOISE] = TierSummary(count=5)
        gaps_by_tier[GapTier.MARKET_MOVE] = TierSummary(count=10)

        result = TieredValidationResult(
            is_valid=True,
            bar_count=100,
            gaps_by_tier=gaps_by_tier,
            all_gaps=[],
            threshold_used_pct=0.025,
            asset_class_detected=AssetClass.CRYPTO,
            preset_used="research",
        )

        summary = result.summary_dict()
        assert summary["NOISE"] == 5
        assert summary["MARKET_MOVE"] == 10
        assert summary["PRECISION"] == 0
        assert summary["SESSION_BOUNDARY"] == 0


class TestValidateContinuityTiered:
    """Test validate_continuity_tiered() function."""

    def _create_continuous_df(self, n_bars: int = 10) -> pd.DataFrame:
        """Create a DataFrame with continuous bars (no gaps)."""
        data = []
        price = 100.0
        for _ in range(n_bars):
            data.append(
                {
                    "Open": price,
                    "High": price + 0.001,
                    "Low": price - 0.001,
                    "Close": price + 0.0001,  # Small move, no gap
                    "Volume": 1.0,
                }
            )
            price = price + 0.0001  # Next open == prev close
        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        return df

    def _create_gapped_df(self, gap_pcts: list) -> pd.DataFrame:
        """Create a DataFrame with specified gap percentages."""
        n_bars = len(gap_pcts) + 1
        data = []
        price = 100.0

        for i in range(n_bars):
            close = price + 0.0001
            data.append(
                {
                    "Open": price,
                    "High": max(price, close) + 0.001,
                    "Low": min(price, close) - 0.001,
                    "Close": close,
                    "Volume": 1.0,
                }
            )
            # Apply gap to next bar's open if gap specified, otherwise continue
            price = close * (1 + gap_pcts[i]) if i < len(gap_pcts) else close

        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        return df

    def test_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = validate_continuity_tiered(df, validation="research")
        assert result.is_valid is True
        assert result.bar_count == 0

    def test_single_bar(self):
        """Test with single bar (no gaps possible)."""
        df = self._create_continuous_df(1)
        result = validate_continuity_tiered(df, validation="research")
        assert result.is_valid is True
        assert result.bar_count == 1

    def test_continuous_bars(self):
        """Test with continuous bars (tiny gaps from floating-point)."""
        df = self._create_continuous_df(10)
        result = validate_continuity_tiered(df, validation="research")
        assert result.is_valid is True
        assert result.bar_count == 10
        # Tiny gaps are classified as PRECISION tier
        # PRECISION gaps are NOT recorded in all_gaps list
        assert len(result.all_gaps) == 0  # PRECISION gaps excluded from detailed list
        # But they are still counted in tier statistics
        assert result.gaps_by_tier[GapTier.SESSION_BOUNDARY].count == 0

    def test_skip_mode(self):
        """Test that skip mode skips validation."""
        df = self._create_gapped_df([0.1])  # 10% gap
        result = validate_continuity_tiered(df, validation="skip")
        assert result.is_valid is True
        assert len(result.all_gaps) == 0  # No gaps recorded in skip mode

    def test_auto_detection_crypto(self):
        """Test auto-detection for crypto symbols."""
        df = self._create_continuous_df(5)
        result = validate_continuity_tiered(df, validation="auto", symbol="BTCUSDT")
        assert result.asset_class_detected == AssetClass.CRYPTO
        assert result.preset_used == "crypto"

    def test_auto_detection_forex(self):
        """Test auto-detection for forex symbols."""
        df = self._create_continuous_df(5)
        result = validate_continuity_tiered(df, validation="auto", symbol="EURUSD")
        assert result.asset_class_detected == AssetClass.FOREX
        assert result.preset_used == "forex"

    def test_microstructure_gap_warning(self):
        """Test that microstructure gaps trigger warning in warn mode."""
        # Create df with 1.5% gap (exceeds "standard" preset tolerance of 1%)
        df = self._create_gapped_df([0.015])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_continuity_tiered(df, validation="standard")

        assert result.has_microstructure_events is True
        assert len(w) == 1
        assert issubclass(w[0].category, ContinuityWarning)

    def test_tolerance_exceeded_error(self):
        """Test that tolerance exceeded raises error in error mode."""
        # Create df with 5% gap (exceeds strict tolerance of 0.5%)
        df = self._create_gapped_df([0.05])

        with pytest.raises(ContinuityError) as exc_info:
            validate_continuity_tiered(df, validation="strict")

        assert len(exc_info.value.discontinuities) == 1

    def test_custom_dict_validation(self):
        """Test custom dict validation config."""
        df = self._create_gapped_df([0.005])  # 0.5% gap

        result = validate_continuity_tiered(
            df,
            validation={"tolerance_pct": 0.01, "mode": "warn"},
            symbol="BTCUSDT",
        )
        # Gap is below tolerance, should pass without warning
        assert result.is_valid is True
        assert result.asset_class_detected == AssetClass.CRYPTO
        assert result.preset_used is None  # Custom config

    def test_custom_preset_validation(self):
        """Test custom ValidationPreset instance."""
        df = self._create_gapped_df([0.005])

        custom_preset = ValidationPreset(
            tolerance_pct=0.01,
            mode="warn",
            description="Custom test preset",
        )

        result = validate_continuity_tiered(df, validation=custom_preset)
        assert result.is_valid is True
        assert result.preset_used is None

    def test_invalid_preset_name(self):
        """Test that invalid preset name raises ValueError."""
        df = self._create_continuous_df(5)

        with pytest.raises(ValueError, match="Unknown validation preset"):
            validate_continuity_tiered(df, validation="nonexistent")

    def test_tier_classification(self):
        """Test that gaps are classified into correct tiers."""
        # Create gaps at different levels
        gaps = [
            0.000005,  # PRECISION (below 0.001%)
            0.00005,  # NOISE (0.001% - 0.01%)
            0.0005,  # MARKET_MOVE (0.01% - 0.1%)
            0.005,  # MICROSTRUCTURE (> 0.1%)
        ]
        df = self._create_gapped_df(gaps)

        # Use permissive to avoid errors
        result = validate_continuity_tiered(df, validation="permissive")

        # Check tier breakdown (PRECISION gaps not recorded in all_gaps)
        tier_counts = result.summary_dict()
        assert tier_counts["NOISE"] >= 1
        assert tier_counts["MARKET_MOVE"] >= 1
        assert tier_counts["MICROSTRUCTURE"] >= 1

    def test_gap_info_details(self):
        """Test that GapInfo contains correct details."""
        df = self._create_gapped_df([0.01])  # 1% gap

        result = validate_continuity_tiered(df, validation="permissive")

        assert len(result.all_gaps) >= 1
        gap = result.all_gaps[0]
        assert isinstance(gap, GapInfo)
        assert gap.bar_index >= 1  # First bar can't have a gap
        assert gap.gap_pct > 0
        assert gap.tier in GapTier
        assert gap.timestamp is not None  # Since we use DatetimeIndex
