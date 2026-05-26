from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from qstudy.experiments import load_studies_config


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
    data_dir: str | Path | None = None,
) -> StudyData:
    """Download OHLCV data in a single yfinance API call.

    Args:
        tickers: List of ticker symbols, or a single ticker string.
        start:   Start date string (ISO format, e.g. "2015-01-01").
        end:     End date string (ISO format, e.g. "2024-12-31").
        interval: yfinance bar interval, e.g. ``"1d"``, ``"1h"``, ``"5m"``, ``"1m"``.
        data_dir: Optional cache directory. If omitted, qstudy uses ``data_dir`` from
                  ``.qstudy.toml`` when configured. Cached datasets are keyed by the
                  normalized download request.

    Returns:
        :class:`StudyData` with aligned OHLCV, returns, and log_returns DataFrames.
        Tickers that fail to download are silently dropped.
    """
    # Normalize to list so we always know the ticker names
    if isinstance(tickers, str):
        tickers = [tickers]

    cache_dir = _resolve_data_dir(data_dir)
    request = _build_download_request(tickers=tickers, start=start, end=end, interval=interval)
    cache_paths = _cache_paths(cache_dir, request) if cache_dir is not None else None
    if cache_paths is not None:
        cached = _load_cached_study_data(cache_paths["data"])
        if cached is not None:
            return cached

    chunk_size = 100
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    all_frames: dict[str, list[pd.DataFrame]] = {
        "Close": [],
        "Open": [],
        "High": [],
        "Low": [],
        "Volume": [],
    }

    for chunk in chunks:
        data = yf.download(
            chunk,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        for field in all_frames:
            raw = data[field]
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=chunk[0])
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            all_frames[field].append(raw)

    def combine(field: str) -> pd.DataFrame:
        return pd.concat(all_frames[field], axis=1)

    close_raw = combine("Close").sort_index()
    close = close_raw.dropna(axis=1)
    open_ = combine("Open").reindex(close.index)[close.columns]
    high = combine("High").reindex(close.index)[close.columns]
    low = combine("Low").reindex(close.index)[close.columns]
    volume = combine("Volume").reindex(close.index)[close.columns]
    returns = close.pct_change().fillna(0)
    log_returns = np.log(close / close.shift(1))

    study_data = StudyData(
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
    if cache_paths is not None:
        _save_cached_study_data(cache_paths, request, study_data)
    return study_data


def _resolve_data_dir(data_dir: str | Path | None) -> Path | None:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    config = load_studies_config()
    return config.data_root


def _build_download_request(
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "yfinance",
        "tickers": list(tickers),
        "start": start,
        "end": end,
        "interval": interval,
        "auto_adjust": True,
        "progress": False,
        "multi_level_index": False,
    }


def _cache_paths(
    cache_root: Path,
    request: dict[str, object],
) -> dict[str, Path]:
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    base_dir = cache_root / "yfinance"
    return {
        "dir": base_dir,
        "data": base_dir / f"{digest}.pkl",
        "meta": base_dir / f"{digest}.json",
    }


def _load_cached_study_data(cache_path: Path) -> StudyData | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, StudyData) else None


def _save_cached_study_data(
    cache_paths: dict[str, Path],
    request: dict[str, object],
    study_data: StudyData,
) -> None:
    cache_paths["dir"].mkdir(parents=True, exist_ok=True)
    with cache_paths["data"].open("wb") as handle:
        pickle.dump(study_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cache_paths["meta"].write_text(json.dumps(request, indent=2), encoding="utf-8")


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
