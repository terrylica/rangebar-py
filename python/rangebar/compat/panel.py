"""Panel format converter for alpha-forge compatibility (Issue #95).

Converts rangebar's DatetimeIndex + capitalized OHLCV format to
alpha-forge's panel format: ts, symbol, price.*, feature.* columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Columns that are always dropped from panel output (internal metadata)
_INTERNAL_COLUMNS = frozenset({
    "ouroboros_mode",
    "is_orphan",
    "first_agg_trade_id",
    "last_agg_trade_id",
    "ouroboros_boundary",
    "reason",
})

# OHLCV columns: rangebar name → panel name
_PRICE_COLUMN_MAP = {
    "Open": "price.open",
    "High": "price.high",
    "Low": "price.low",
    "Close": "price.close",
    "Volume": "price.volume",
}


def to_panel_format(
    df: pd.DataFrame,
    symbol: str,
    *,
    feature_prefix: str = "feature",
) -> pd.DataFrame:
    """Convert rangebar DataFrame to alpha-forge panel format.

    Transforms DatetimeIndex with capitalized OHLCV columns into a flat
    DataFrame with `ts`, `symbol`, `price.*`, and `feature.*` columns.

    Parameters
    ----------
    df : pd.DataFrame
        rangebar output with DatetimeIndex and OHLCV columns.
    symbol : str
        Symbol name (e.g., "BTCUSDT").
    feature_prefix : str
        Prefix for microstructure feature columns (default: "feature").

    Returns
    -------
    pd.DataFrame
        Panel-format DataFrame with columns: ts, symbol, price.*, feature.*
    """
    import pandas as pd

    from rangebar.constants import ALL_OPTIONAL_COLUMNS

    optional_set = set(ALL_OPTIONAL_COLUMNS)

    # Reset index to get timestamp as column
    result = df.reset_index()

    # Rename index column to 'ts'
    index_col = result.columns[0]
    result = result.rename(columns={index_col: "ts"})

    # Ensure ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(result["ts"]):
        result["ts"] = pd.to_datetime(result["ts"])

    # Add symbol column
    result["symbol"] = symbol

    # Rename OHLCV columns
    result = result.rename(columns=_PRICE_COLUMN_MAP)

    # Rename feature columns with prefix
    feature_renames = {}
    for col in result.columns:
        if col in optional_set and col not in _INTERNAL_COLUMNS:
            feature_renames[col] = f"{feature_prefix}.{col}"
    result = result.rename(columns=feature_renames)

    # Drop internal columns
    drop_cols = [c for c in result.columns if c in _INTERNAL_COLUMNS]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    return result


def get_range_bars_panel(
    symbols: list[str] | str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    include_microstructure: bool = False,
    ouroboros_mode: str | None = None,
    use_cache: bool = True,
    feature_prefix: str = "feature",
) -> pd.DataFrame:
    """Fetch range bars for multiple symbols in panel format.

    Parameters
    ----------
    symbols : list[str] | str
        One or more trading symbols.
    start_date, end_date : str
        Date range (e.g., "2024-01-01").
    threshold_decimal_bps : int | str
        Threshold in decimal basis points or preset name.
    include_microstructure : bool
        Include microstructure feature columns.
    ouroboros_mode : str | None
        Ouroboros reset mode. If None, resolved from config.  # Issue #126
    use_cache : bool
        Use ClickHouse cache if available.
    feature_prefix : str
        Prefix for feature columns.

    Returns
    -------
    pd.DataFrame
        Panel-format DataFrame sorted by ["symbol", "ts"].
    """
    import pandas as pd

    from rangebar.orchestration.range_bars import get_range_bars

    if isinstance(symbols, str):
        symbols = [symbols]

    frames = []
    for sym in symbols:
        bars = get_range_bars(
            sym,
            start_date,
            end_date,
            threshold_decimal_bps=threshold_decimal_bps,
            include_microstructure=include_microstructure,
            ouroboros_mode=ouroboros_mode,
            use_cache=use_cache,
        )
        panel = to_panel_format(bars, sym, feature_prefix=feature_prefix)
        frames.append(panel)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["symbol", "ts"]).reset_index(drop=True)
