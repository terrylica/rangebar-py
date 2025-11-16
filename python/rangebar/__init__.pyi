"""Type stubs for rangebar package."""

from typing import Dict, List, Union

import pandas as pd

__version__: str

class RangeBarProcessor:
    """Process tick-level trade data into range bars."""

    threshold_bps: int

    def __init__(self, threshold_bps: int) -> None:
        """Initialize processor with given threshold.

        Parameters
        ----------
        threshold_bps : int
            Threshold in 0.1 basis point units (250 = 25bps = 0.25%)

        Raises
        ------
        ValueError
            If threshold_bps is out of valid range [1, 100_000]
        """
        ...

    def process_trades(
        self, trades: List[Dict[str, Union[int, float]]]
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Process trades into range bars.

        Parameters
        ----------
        trades : List[Dict]
            List of trade dictionaries with required keys:
            timestamp, price, quantity (or volume)

        Returns
        -------
        List[Dict]
            List of range bar dictionaries

        Raises
        ------
        KeyError
            If required trade fields are missing
        RuntimeError
            If trades are not sorted chronologically
        """
        ...

    def to_dataframe(
        self, bars: List[Dict[str, Union[str, float, int]]]
    ) -> pd.DataFrame:
        """Convert range bars to pandas DataFrame.

        Parameters
        ----------
        bars : List[Dict]
            List of range bar dictionaries from process_trades()

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and OHLCV columns
        """
        ...

def process_trades_to_dataframe(
    trades: Union[List[Dict[str, Union[int, float]]], pd.DataFrame],
    threshold_bps: int = 250,
) -> pd.DataFrame:
    """Convenience function to process trades directly to DataFrame.

    Parameters
    ----------
    trades : List[Dict] or pd.DataFrame
        Trade data with columns/keys: timestamp, price, quantity
    threshold_bps : int, default=250
        Threshold in 0.1bps units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    Raises
    ------
    ValueError
        If required columns are missing or threshold is invalid
    RuntimeError
        If trades are not sorted chronologically
    """
    ...
