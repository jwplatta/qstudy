from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

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
        open:        Adjusted open prices (dates x tickers).
        high:        Adjusted high prices (dates x tickers).
        low:         Adjusted low prices (dates x tickers).
        close:       Adjusted close prices (dates x tickers).
        volume:      Daily volume (dates x tickers).
        returns:     Daily pct-change returns, NaN filled with 0 (dates x tickers).
        log_returns: Log returns (dates x tickers).
    """

    tickers: list[str]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    log_returns: pd.DataFrame
    interval: str = "1d"


def download(
    tickers: list[str] | str,
    start: str,
    end: str,
    interval: str = "1d",
) -> StudyData:
    """Download OHLCV data in a single yfinance API call.

    Args:
        tickers: List of ticker symbols, or a single ticker string.
        start:   Start date string (ISO format, e.g. "2015-01-01").
        end:     End date string (ISO format, e.g. "2024-12-31").
        interval: yfinance bar interval, e.g. ``"1d"``, ``"1h"``, ``"5m"``, ``"1m"``.

    Returns:
        :class:`StudyData` with aligned OHLCV, returns, and log_returns DataFrames.
        Tickers that fail to download are silently dropped.
    """
    # Normalize to list so we always know the ticker names
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    def normalize_frame(field: str) -> pd.DataFrame:
        raw = data[field]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        return raw

    close_raw = normalize_frame("Close").sort_index()
    close = close_raw.dropna(axis=1)
    open_ = normalize_frame("Open").reindex(close.index)[close.columns]
    high = normalize_frame("High").reindex(close.index)[close.columns]
    low = normalize_frame("Low").reindex(close.index)[close.columns]
    volume = normalize_frame("Volume").reindex(close.index)[close.columns]
    returns = close.pct_change().fillna(0)
    log_returns = np.log(close / close.shift(1))

    return StudyData(
        tickers=close.columns.tolist(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        returns=returns,
        log_returns=log_returns,
        interval=interval,
    )


def get_sector_map(
    tickers: list[str],
    cache_path: str | Path | None = None,
    max_age_days: int = 30,
) -> dict[str, str]:
    """Fetch and disk-cache GICS sector classifications for a list of tickers.

    Uses a JSON on-disk cache to avoid repeated yfinance HTTP calls. The cache
    is invalidated after ``max_age_days``. Unknown sectors (missing or empty
    ``info['sector']``) are stored as ``"Unknown"``.

    Args:
        tickers:      List of ticker symbols.
        cache_path:   Path to the JSON cache file.
                      Defaults to ``~/.qstudy/sector_map.json``.
        max_age_days: Days before the on-disk cache is considered stale.

    Returns:
        Dict mapping ticker -> GICS sector string, e.g.
        ``{"AAPL": "Technology", "JPM": "Financial Services", ...}``.
    """
    if cache_path is None:
        cache_path = Path.home() / ".qstudy" / "sector_map.json"
    else:
        cache_path = Path(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if fresh enough
    cached: dict[str, str] = {}
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            with open(cache_path) as f:
                cached = json.load(f)

    # Fetch any tickers missing from cache
    missing = [t for t in tickers if t not in cached]
    if missing:
        print(f"Fetching sector classifications for {len(missing)} tickers...")
        for ticker in missing:
            try:
                info = yf.Ticker(ticker).info
                cached[ticker] = info.get("sector") or "Unknown"
            except Exception:
                cached[ticker] = "Unknown"

        with open(cache_path, "w") as f:
            json.dump(cached, f)

    return {t: cached.get(t, "Unknown") for t in tickers}
