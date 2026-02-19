# Issue #98: FeatureProvider protocol for external feature enrichment.
"""FeatureProvider protocol — the contract for plugin feature providers.

Any external package that wants to enrich range bars with additional columns
must implement this protocol. Discovery is automatic via Python entry points.

Example
-------
A plugin package (e.g., atr-adaptive-laguerre) registers its provider::

    # In atr_adaptive_laguerre/rangebar_plugin.py
    class LaguerreFeatureProvider:
        @property
        def name(self) -> str:
            return "laguerre"

        @property
        def version(self) -> str:
            return "2.1.0"

        @property
        def columns(self) -> tuple[str, ...]:
            return ("laguerre_rsi", "laguerre_regime")

        @property
        def min_bars(self) -> int:
            return 50

        def enrich(self, bars, symbol, threshold_decimal_bps):
            bars["laguerre_rsi"] = compute_laguerre_rsi(bars["close"].values)
            bars["laguerre_regime"] = classify_regime(bars["laguerre_rsi"].values)
            return bars

    # In pyproject.toml:
    [project.entry-points."rangebar.feature_providers"]
    laguerre = "atr_adaptive_laguerre.rangebar_plugin:LaguerreFeatureProvider"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class FeatureProvider(Protocol):
    """Protocol for external feature providers that enrich range bars.

    Implementations MUST:
    - Add columns IN-PLACE to the input DataFrame (no full copy).
    - Return the same DataFrame object.
    - NOT modify existing columns or reorder rows.
    - Use NaN for warmup period (rows 0..min_bars-1), not 0.0.
    - Be idempotent: calling twice produces the same result.
    - Be non-anticipative: feature at bar[i] uses only data from bars[0..i].
    """

    @property
    def name(self) -> str:
        """Unique provider name (e.g., 'laguerre')."""
        ...

    @property
    def version(self) -> str:
        """Provider version for schema tracking (semver)."""
        ...

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names this provider adds.

        All columns must be prefixed with the provider name
        (e.g., 'laguerre_rsi', 'laguerre_regime').
        """
        ...

    @property
    def min_bars(self) -> int:
        """Minimum bars needed for lookback warmup.

        Bars before this index will have NaN in the feature columns.
        """
        ...

    def enrich(
        self,
        bars: pd.DataFrame,
        symbol: str,
        threshold_decimal_bps: int,
    ) -> pd.DataFrame:
        """Add feature columns to completed range bars.

        Input DataFrame has at minimum: open, high, low, close, volume
        columns (lowercase after Rust processing) plus DatetimeIndex or
        timestamp_ms. May also include microstructure and inter-bar columns.

        MUST add columns in-place and return the same DataFrame object.
        MUST NOT modify existing columns. MUST NOT reorder rows.
        Columns must match self.columns exactly.

        Parameters
        ----------
        bars : pd.DataFrame
            Range bar DataFrame, sorted chronologically.
        symbol : str
            Trading symbol (e.g., "BTCUSDT").
        threshold_decimal_bps : int
            Threshold used for bar generation.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with additional columns appended in-place.
        """
        ...
