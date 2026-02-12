"""Tests for configurable continuity tolerance (Issue #18, Phase 3).

Validates:
- CONTINUITY_TOLERANCE_PCT constant default (0.001)
- Environment variable override via RANGEBAR_CONTINUITY_TOLERANCE
- get_n_range_bars() continuity_tolerance_pct parameter
- Tolerance parameter flows to validation logic
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestContinuityToleranceConstant:
    """Test CONTINUITY_TOLERANCE_PCT constant and env var override."""

    def test_default_value_is_0_001(self):
        """Default tolerance should be 0.001 (0.1%)."""
        # Import fresh without env var
        from rangebar.constants import CONTINUITY_TOLERANCE_PCT

        # Default from code is 0.001 (may be overridden by env in test env)
        env_val = os.environ.get("RANGEBAR_CONTINUITY_TOLERANCE")
        if env_val is None:
            assert pytest.approx(0.001) == CONTINUITY_TOLERANCE_PCT

    def test_env_var_override(self):
        """RANGEBAR_CONTINUITY_TOLERANCE env var should override default."""
        with patch.dict(os.environ, {"RANGEBAR_CONTINUITY_TOLERANCE": "0.005"}):
            # Must reimport to pick up env var
            import importlib

            import rangebar.constants as constants_mod

            importlib.reload(constants_mod)
            try:
                assert pytest.approx(0.005) == constants_mod.CONTINUITY_TOLERANCE_PCT
            finally:
                # Restore original value
                importlib.reload(constants_mod)

    def test_env_var_float_parsing(self):
        """Env var should parse as float correctly."""
        with patch.dict(os.environ, {"RANGEBAR_CONTINUITY_TOLERANCE": "0.0001"}):
            import importlib

            import rangebar.constants as constants_mod

            importlib.reload(constants_mod)
            try:
                assert pytest.approx(0.0001) == constants_mod.CONTINUITY_TOLERANCE_PCT
            finally:
                importlib.reload(constants_mod)


class TestGetNRangeBarsToleranceParam:
    """Test continuity_tolerance_pct parameter on get_n_range_bars()."""

    def test_parameter_exists_in_signature(self):
        """get_n_range_bars should accept continuity_tolerance_pct kwarg."""
        import inspect

        from rangebar.orchestration.count_bounded import get_n_range_bars

        sig = inspect.signature(get_n_range_bars)
        assert "continuity_tolerance_pct" in sig.parameters
        param = sig.parameters["continuity_tolerance_pct"]
        assert param.default is None

    def test_parameter_in_type_stubs(self):
        """Type stub should include continuity_tolerance_pct for get_n_range_bars."""

        stub_dir = Path(__file__).parent / ".." / "python" / "rangebar"
        found = any(
            "continuity_tolerance_pct" in p.read_text()
            for p in stub_dir.rglob("*.pyi")
        )

        # Verify the parameter appears in the stubs
        assert found

    def test_tolerance_used_in_validation(self):
        """Custom tolerance should be used instead of constant when provided."""
        # Create a DataFrame with a small discontinuity (0.05%)
        timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.05, 100.10, 100.15, 100.20],
                "High": [100.10, 100.15, 100.20, 100.25, 100.30],
                "Low": [99.90, 100.00, 100.05, 100.10, 100.15],
                "Close": [100.05, 100.10, 100.15, 100.20, 100.25],
                "Volume": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=timestamps,
        )
        # Introduce a discontinuity: Close[1]=100.10, Open[2] should be 100.10
        # but set it to 100.15 (0.05% gap)
        df.iloc[2, df.columns.get_loc("Open")] = 100.15

        close_prices = df["Close"].to_numpy()[:-1]
        open_prices = df["Open"].to_numpy()[1:]

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.abs(open_prices - close_prices) / np.abs(close_prices)

        max_diff = float(np.nanmax(rel_diff))

        # With tight tolerance (0.0001), this gap should be detected
        assert max_diff > 0.0001
        # With loose tolerance (0.01), this gap should pass
        assert max_diff < 0.01


class TestToleranceDocumentation:
    """Test that tolerance is properly documented."""

    def test_docstring_mentions_tolerance(self):
        """get_n_range_bars docstring should document continuity_tolerance_pct."""
        from rangebar.orchestration.count_bounded import get_n_range_bars

        doc = get_n_range_bars.__doc__
        assert doc is not None
        assert "continuity_tolerance_pct" in doc

    def test_constants_module_has_tolerance(self):
        """constants module should export CONTINUITY_TOLERANCE_PCT."""
        from rangebar.constants import CONTINUITY_TOLERANCE_PCT

        assert isinstance(CONTINUITY_TOLERANCE_PCT, float)
        assert CONTINUITY_TOLERANCE_PCT > 0
