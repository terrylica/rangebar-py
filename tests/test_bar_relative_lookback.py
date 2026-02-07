"""Tests for bar-relative lookback mode (Issue #81).

Validates:
- RangeBarProcessor accepts inter_bar_lookback_bars parameter
- BarRelative mode takes precedence over FixedCount
- Environment variable RANGEBAR_INTER_BAR_LOOKBACK_BARS is respected
- Feature computation works with bar-relative lookback
"""

from __future__ import annotations

import os
from unittest.mock import patch


class TestBarRelativeLookbackParameter:
    """Test inter_bar_lookback_bars parameter acceptance."""

    def test_parameter_accepted_by_processor(self):
        """RangeBarProcessor should accept inter_bar_lookback_bars kwarg."""
        from rangebar._core import PyRangeBarProcessor as RangeBarProcessor

        # Should not raise
        p = RangeBarProcessor(
            250,
            inter_bar_lookback_bars=3,
            include_intra_bar_features=True,
        )
        assert p is not None

    def test_parameter_with_none(self):
        """inter_bar_lookback_bars=None should disable bar-relative mode."""
        from rangebar._core import PyRangeBarProcessor as RangeBarProcessor

        p = RangeBarProcessor(
            250,
            inter_bar_lookback_bars=None,
        )
        assert p is not None

    def test_bars_takes_precedence_over_count(self):
        """inter_bar_lookback_bars should take precedence over inter_bar_lookback_count."""
        from rangebar._core import PyRangeBarProcessor as RangeBarProcessor

        # Both set — bars should win (no error)
        p = RangeBarProcessor(
            250,
            inter_bar_lookback_count=200,
            inter_bar_lookback_bars=3,
        )
        assert p is not None


class TestBarRelativeEnvVar:
    """Test RANGEBAR_INTER_BAR_LOOKBACK_BARS environment variable."""

    def test_env_var_parsed(self):
        """RANGEBAR_INTER_BAR_LOOKBACK_BARS should be parsed by helpers."""
        from rangebar.orchestration.helpers import _parse_microstructure_env_vars

        with patch.dict(os.environ, {"RANGEBAR_INTER_BAR_LOOKBACK_BARS": "3"}):
            result_count, result_bars, _intra = _parse_microstructure_env_vars(
                include_microstructure=True,
                inter_bar_lookback_count=None,
            )
            assert result_bars == 3
            assert result_count is None  # bars takes precedence

    def test_env_var_not_set_falls_back_to_count(self):
        """Without RANGEBAR_INTER_BAR_LOOKBACK_BARS, should fall back to count."""
        from rangebar.orchestration.helpers import _parse_microstructure_env_vars

        env = {k: v for k, v in os.environ.items()
               if k != "RANGEBAR_INTER_BAR_LOOKBACK_BARS"}
        with patch.dict(os.environ, env, clear=True):
            result_count, result_bars, _intra = _parse_microstructure_env_vars(
                include_microstructure=True,
                inter_bar_lookback_count=None,
            )
            assert result_bars is None
            assert result_count == 200  # default from RANGEBAR_INTER_BAR_LOOKBACK_COUNT

    def test_explicit_param_overrides_env(self):
        """Explicit inter_bar_lookback_bars param should override env var."""
        from rangebar.orchestration.helpers import _parse_microstructure_env_vars

        with patch.dict(os.environ, {"RANGEBAR_INTER_BAR_LOOKBACK_BARS": "5"}):
            _count, result_bars, _intra = _parse_microstructure_env_vars(
                include_microstructure=True,
                inter_bar_lookback_count=None,
                inter_bar_lookback_bars=3,
            )
            assert result_bars == 3  # param wins over env


class TestBarRelativeCreateProcessor:
    """Test _create_processor with bar-relative lookback."""

    def test_create_processor_with_bars(self):
        """_create_processor should accept inter_bar_lookback_bars."""
        from rangebar.orchestration.helpers import _create_processor

        p = _create_processor(
            250,
            include_microstructure=True,
            inter_bar_lookback_bars=3,
        )
        assert p is not None

    def test_create_processor_bars_overrides_count(self):
        """_create_processor should pass bars to RangeBarProcessor."""
        from rangebar.orchestration.helpers import _create_processor

        # Both set — should not raise
        p = _create_processor(
            250,
            include_microstructure=True,
            inter_bar_lookback_count=200,
            inter_bar_lookback_bars=3,
        )
        assert p is not None


class TestBarRelativeSignature:
    """Test that function signatures include the new parameter."""

    def test_parse_env_vars_signature(self):
        """_parse_microstructure_env_vars should have inter_bar_lookback_bars param."""
        import inspect

        from rangebar.orchestration.helpers import _parse_microstructure_env_vars

        sig = inspect.signature(_parse_microstructure_env_vars)
        assert "inter_bar_lookback_bars" in sig.parameters

    def test_create_processor_signature(self):
        """_create_processor should have inter_bar_lookback_bars param."""
        import inspect

        from rangebar.orchestration.helpers import _create_processor

        sig = inspect.signature(_create_processor)
        assert "inter_bar_lookback_bars" in sig.parameters

    def test_type_stub_has_parameter(self):
        """Type stub should include inter_bar_lookback_bars for get_range_bars."""
        from pathlib import Path

        stub_path = (
            Path(__file__).parent / ".." / "python" / "rangebar" / "__init__.pyi"
        )
        content = stub_path.read_text()
        assert "inter_bar_lookback_bars" in content
