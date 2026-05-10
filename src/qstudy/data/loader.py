from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class StudyData:
    """Container returned by :func:`download`.

    Pass directly to :class:`~qstudy.study.Study.Study` as ``universe``,
    ``benchmark``, or ``factors``.

    Attributes:
        tickers:     Tickers successfully downloaded (failed tickers are dropped).
        close:       Adjusted close prices (dates x tickers).
        volume:      Daily volume (dates x tickers).
        returns:     Daily pct-change returns, NaN filled with 0 (dates x tickers).
        log_returns: Log returns (dates x tickers).
    """

    tickers: list[str]
    close: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    log_returns: pd.DataFrame


def download(tickers: list[str] | str, start: str, end: str) -> StudyData:
    """Download OHLCV data in a single yfinance API call.

    Args:
        tickers: List of ticker symbols, or a single ticker string.
        start:   Start date string (ISO format, e.g. "2015-01-01").
        end:     End date string (ISO format, e.g. "2024-12-31").

    Returns:
        :class:`StudyData` with aligned close, volume, returns, and log_returns DataFrames.
        Tickers that fail to download are silently dropped.
    """
    # Normalize to list so we always know the ticker names
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
    )
    close_raw = data["Close"]
    # Single-ticker downloads return a Series; normalize to DataFrame
    if isinstance(close_raw, pd.Series):
        close_raw = close_raw.to_frame(name=tickers[0])

    close = close_raw.dropna(axis=1)
    volume_raw = data["Volume"]
    if isinstance(volume_raw, pd.Series):
        volume_raw = volume_raw.to_frame(name=tickers[0])

    volume = volume_raw[close.columns]
    returns = close.pct_change().fillna(0)
    log_returns = np.log(close / close.shift(1))

    return StudyData(
        tickers=close.columns.tolist(),
        close=close,
        volume=volume,
        returns=returns,
        log_returns=log_returns,
    )
