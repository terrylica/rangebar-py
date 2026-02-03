#!/usr/bin/env python3
"""Test crypto minimum threshold enforcement (Issue #62).

This test module verifies the hierarchical SSoT configuration for minimum
thresholds, ensuring crypto symbols cannot use thresholds below the configured
minimum (default 1000 dbps = 1%).

Test cases cover:
- Crypto symbol + low threshold -> ThresholdError
- Crypto symbol + valid threshold -> OK
- Forex symbol + low threshold -> OK (lower minimum)
- Checkpoint with low threshold -> ThresholdError
- Asset-class env var override works
- Per-symbol env var override works (higher priority)
- Cache clearing works
- All entry points covered
"""

from __future__ import annotations

import os

import pytest
from rangebar import (
    RangeBarProcessor,
    ThresholdError,
    clear_threshold_cache,
    get_min_threshold,
    get_min_threshold_for_symbol,
    resolve_and_validate_threshold,
)
from rangebar.threshold import validate_checkpoint_threshold
from rangebar.validation.gap_classification import AssetClass, detect_asset_class


class TestAssetClassDetection:
    """Test asset class detection for threshold validation."""

    def test_btcusdt_is_crypto(self):
        """BTCUSDT should be classified as CRYPTO."""
        assert detect_asset_class("BTCUSDT") == AssetClass.CRYPTO

    def test_ethusdt_is_crypto(self):
        """ETHUSDT should be classified as CRYPTO."""
        assert detect_asset_class("ETHUSDT") == AssetClass.CRYPTO

    def test_wbtcusdt_is_crypto(self):
        """WBTCUSDT (wrapped Bitcoin) should be classified as CRYPTO."""
        assert detect_asset_class("WBTCUSDT") == AssetClass.CRYPTO

    def test_wethusdt_is_crypto(self):
        """WETHUSDT (wrapped Ethereum) should be classified as CRYPTO."""
        assert detect_asset_class("WETHUSDT") == AssetClass.CRYPTO

    def test_ethusde_is_crypto(self):
        """ETHUSDE (USDE stablecoin pair) should be classified as CRYPTO."""
        assert detect_asset_class("ETHUSDE") == AssetClass.CRYPTO

    def test_btcusdm_is_crypto(self):
        """BTCUSDM (USDM stablecoin pair) should be classified as CRYPTO."""
        assert detect_asset_class("BTCUSDM") == AssetClass.CRYPTO

    def test_eurusd_is_forex(self):
        """EURUSD should be classified as FOREX."""
        assert detect_asset_class("EURUSD") == AssetClass.FOREX

    def test_xauusd_is_forex(self):
        """XAUUSD (gold) should be classified as FOREX."""
        assert detect_asset_class("XAUUSD") == AssetClass.FOREX


class TestMinThresholdLookup:
    """Test minimum threshold lookup functions."""

    def test_crypto_minimum_from_env(self):
        """Crypto minimum should come from env var."""
        # conftest.py sets this to 1 for testing
        min_threshold = get_min_threshold(AssetClass.CRYPTO)
        assert isinstance(min_threshold, int)
        assert min_threshold >= 1

    def test_forex_minimum_from_env(self):
        """Forex minimum should come from env var."""
        min_threshold = get_min_threshold(AssetClass.FOREX)
        assert isinstance(min_threshold, int)
        assert min_threshold >= 1

    def test_per_symbol_lookup(self):
        """Per-symbol lookup should work for any symbol."""
        btc_min = get_min_threshold_for_symbol("BTCUSDT")
        assert isinstance(btc_min, int)
        assert btc_min >= 1


class TestPerSymbolEnvVarOverride:
    """Test per-symbol environment variable override (highest priority)."""

    def test_per_symbol_override_takes_priority(self):
        """Per-symbol env var should override asset-class default."""
        # Set per-symbol override
        os.environ["RANGEBAR_MIN_THRESHOLD_TESTBTC"] = "999"
        clear_threshold_cache()

        try:
            min_threshold = get_min_threshold_for_symbol("TESTBTC")
            assert min_threshold == 999
        finally:
            del os.environ["RANGEBAR_MIN_THRESHOLD_TESTBTC"]
            clear_threshold_cache()


class TestAssetClassEnvVarOverride:
    """Test asset-class environment variable override."""

    def test_asset_class_override_works(self):
        """Asset-class env var should be used when no per-symbol override."""
        # Set asset-class override
        original = os.environ.get("RANGEBAR_CRYPTO_MIN_THRESHOLD")
        os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "888"
        clear_threshold_cache()

        try:
            min_threshold = get_min_threshold_for_symbol("NEWCRYPTOUSDT")
            assert min_threshold == 888
        finally:
            if original is not None:
                os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = original
            else:
                del os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"]
            clear_threshold_cache()


class TestResolveAndValidateThreshold:
    """Test the central validation function."""

    def test_valid_threshold_passes(self):
        """Valid threshold should pass validation."""
        result = resolve_and_validate_threshold("BTCUSDT", 1000)
        assert result == 1000

    def test_preset_name_resolution(self):
        """Preset names should be resolved to integers."""
        result = resolve_and_validate_threshold("BTCUSDT", "macro")
        assert result == 1000

    def test_unknown_preset_raises_error(self):
        """Unknown preset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown threshold preset"):
            resolve_and_validate_threshold("BTCUSDT", "nonexistent_preset")

    def test_threshold_below_global_minimum_raises_error(self):
        """Threshold below global minimum (1 dbps) should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            resolve_and_validate_threshold("BTCUSDT", 0)

    def test_threshold_above_global_maximum_raises_error(self):
        """Threshold above global maximum (100,000 dbps) should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            resolve_and_validate_threshold("BTCUSDT", 200_000)

    def test_no_symbol_skips_asset_class_validation(self):
        """When symbol is None, asset-class validation should be skipped."""
        # This should pass even if threshold is below typical crypto minimum
        result = resolve_and_validate_threshold(None, 50)
        assert result == 50


class TestThresholdErrorForCrypto:
    """Test ThresholdError is raised for crypto symbols below minimum."""

    def test_crypto_below_minimum_raises_error(self):
        """Crypto symbol with threshold below minimum should raise ThresholdError."""
        # Temporarily set crypto minimum to 1000
        original = os.environ.get("RANGEBAR_CRYPTO_MIN_THRESHOLD")
        os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "1000"
        clear_threshold_cache()

        try:
            with pytest.raises(ThresholdError) as exc_info:
                resolve_and_validate_threshold("BTCUSDT", 250)

            # Verify error message contains helpful information
            error_msg = str(exc_info.value)
            assert "250" in error_msg
            assert "1000" in error_msg
            assert "BTCUSDT" in error_msg
        finally:
            if original is not None:
                os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = original
            else:
                del os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"]
            clear_threshold_cache()


class TestThresholdErrorForForex:
    """Test forex symbols can use lower thresholds."""

    def test_forex_low_threshold_allowed(self):
        """Forex symbol should allow lower threshold than crypto."""
        # With conftest.py setting minimums to 1, this should pass
        result = resolve_and_validate_threshold("EURUSD", 50)
        assert result == 50


class TestCheckpointValidation:
    """Test checkpoint threshold validation."""

    def test_valid_checkpoint_passes(self):
        """Checkpoint with valid threshold should pass."""
        checkpoint = {
            "symbol": "BTCUSDT",
            "threshold_decimal_bps": 1000,
        }
        # Should not raise
        validate_checkpoint_threshold(checkpoint)

    def test_checkpoint_with_low_threshold_raises_error(self):
        """Checkpoint with threshold below minimum should raise ThresholdError."""
        # Temporarily set crypto minimum to 1000
        original = os.environ.get("RANGEBAR_CRYPTO_MIN_THRESHOLD")
        os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "1000"
        clear_threshold_cache()

        try:
            checkpoint = {
                "symbol": "BTCUSDT",
                "threshold_decimal_bps": 250,
            }
            with pytest.raises(ThresholdError):
                validate_checkpoint_threshold(checkpoint)
        finally:
            if original is not None:
                os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = original
            else:
                del os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"]
            clear_threshold_cache()


class TestRangeBarProcessorValidation:
    """Test RangeBarProcessor validates threshold for symbol."""

    def test_processor_with_symbol_validates(self):
        """RangeBarProcessor should validate threshold for symbol."""
        # With conftest.py setting minimums to 1, this should pass
        processor = RangeBarProcessor(threshold_decimal_bps=1000, symbol="BTCUSDT")
        assert processor.threshold_decimal_bps == 1000

    def test_processor_without_symbol_skips_validation(self):
        """RangeBarProcessor without symbol should skip asset-class validation."""
        # Should pass without symbol
        processor = RangeBarProcessor(threshold_decimal_bps=50)
        assert processor.threshold_decimal_bps == 50


class TestCacheClearing:
    """Test LRU cache clearing."""

    def test_cache_clear_works(self):
        """Cache clearing should allow new env var values to take effect."""
        # Set initial value
        os.environ["RANGEBAR_MIN_THRESHOLD_TESTCACHE"] = "100"
        clear_threshold_cache()

        first_result = get_min_threshold_for_symbol("TESTCACHE")
        assert first_result == 100

        # Change value
        os.environ["RANGEBAR_MIN_THRESHOLD_TESTCACHE"] = "200"
        clear_threshold_cache()

        second_result = get_min_threshold_for_symbol("TESTCACHE")
        assert second_result == 200

        # Cleanup
        del os.environ["RANGEBAR_MIN_THRESHOLD_TESTCACHE"]
        clear_threshold_cache()


class TestExportedSymbols:
    """Test that all threshold-related symbols are properly exported."""

    def test_threshold_error_exported(self):
        """ThresholdError should be importable from rangebar."""
        assert ThresholdError is not None

    def test_crypto_threshold_error_alias_exported(self):
        """CryptoThresholdError alias should be importable from rangebar."""
        from rangebar import CryptoThresholdError

        assert CryptoThresholdError is ThresholdError

    def test_get_min_threshold_exported(self):
        """get_min_threshold should be importable from rangebar."""
        assert callable(get_min_threshold)

    def test_get_min_threshold_for_symbol_exported(self):
        """get_min_threshold_for_symbol should be importable from rangebar."""
        assert callable(get_min_threshold_for_symbol)

    def test_resolve_and_validate_threshold_exported(self):
        """resolve_and_validate_threshold should be importable from rangebar."""
        assert callable(resolve_and_validate_threshold)

    def test_clear_threshold_cache_exported(self):
        """clear_threshold_cache should be importable from rangebar."""
        assert callable(clear_threshold_cache)
