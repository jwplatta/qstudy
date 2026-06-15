from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qstudy.experiments import load_studies_config

SUPPORTED_INTERVALS = {"1d", "day"}
DEFAULT_PROVIDER = "tickrake"


@dataclass
class StudyData:
    """Container returned by :func:`download`.

    Pass directly to :class:`~qstudy.study.Study.Study` as ``universe``,
    ``benchmark``, or ``factors``.
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
    provider: str = DEFAULT_PROVIDER
    frequency: str = "day"
    index_code: str | None = None
    membership_mask: pd.DataFrame | None = None
    requested_tickers: list[str] | None = None

    def __post_init__(self) -> None:
        self.interval = _normalize_interval(self.interval)
        self.frequency = "day"
        self.tickers = list(self.tickers)
        self.requested_tickers = (
            list(self.tickers) if self.requested_tickers is None else list(self.requested_tickers)
        )

        base_index = pd.DatetimeIndex(self.close.index).sort_values()
        columns = list(self.tickers)
        for field_name in ("open", "high", "low", "close", "volume", "returns", "log_returns"):
            frame = getattr(self, field_name)
            normalized = pd.DataFrame(frame).copy()
            normalized.index = pd.DatetimeIndex(normalized.index)
            normalized = normalized.sort_index().reindex(index=base_index, columns=columns)
            setattr(self, field_name, normalized)

        if self.membership_mask is None:
            return

        mask = pd.DataFrame(self.membership_mask).copy()
        mask.index = pd.DatetimeIndex(mask.index)
        mask = (
            mask.sort_index().reindex(index=base_index, columns=columns).fillna(False).astype(bool)
        )
        self.membership_mask = mask

        for field_name in ("open", "high", "low", "close", "volume"):
            setattr(self, field_name, getattr(self, field_name).where(mask))

        self.returns = self.close.pct_change(fill_method=None).where(mask)
        self.log_returns = np.log(self.close / self.close.shift(1)).where(mask)


def download(
    tickers: list[str] | str | None = None,
    start: str = "",
    end: str = "",
    interval: str = "1d",
    data_dir: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    history_dirs: list[str | Path] | None = None,
    index_code: str | None = None,
) -> StudyData:
    """Load OHLCV data from local Tickrake storage.

    Args:
        tickers: List of ticker symbols, a single ticker string, or ``None`` when
            loading an index-backed universe via ``index_code``.
        start: Inclusive start date string (ISO format).
        end: Inclusive end date string (ISO format).
        interval: Bar interval. Only daily intervals (``"1d"`` or ``"day"``) are
            supported in the Tickrake-backed loader.
        data_dir: Optional qstudy cache directory. If omitted, qstudy uses
            ``data_dir`` from ``.qstudy.toml`` when configured.
        sqlite_path: Optional Tickrake SQLite path override.
        history_dirs: Optional ordered candle-directory overrides.
        index_code: Optional market index code such as ``"SP500"``. When set, qstudy
            resolves the universe and membership mask from Tickrake SQLite.

    Returns:
        :class:`StudyData` with aligned OHLCV, returns, and log-returns frames.
    """
    if not start or not end:
        raise ValueError("download() requires both start and end dates.")
    if (tickers is None) == (index_code is None):
        raise ValueError("Provide exactly one of tickers or index_code.")

    normalized_interval = _normalize_interval(interval)
    sqlite_db_path = _resolve_tickrake_sqlite_path(sqlite_path)
    resolved_history_dirs = _resolve_history_dirs(history_dirs)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be on or after start.")

    if index_code is not None:
        requested_tickers = index_range_tickers(index_code, start, end, sqlite_db_path)
    else:
        requested_tickers = _normalize_tickers(tickers)

    if not requested_tickers:
        raise ValueError("No tickers resolved for download().")

    cache_dir = _resolve_data_dir(data_dir)
    request = _build_download_request(
        tickers=requested_tickers,
        start=start,
        end=end,
        interval=normalized_interval,
        sqlite_path=sqlite_db_path,
        history_dirs=resolved_history_dirs,
        index_code=index_code,
    )
    cache_paths = _cache_paths(cache_dir, request) if cache_dir is not None else None
    if cache_paths is not None:
        cached = _load_cached_study_data(cache_paths["data"])
        if cached is not None:
            return cached

    frames_by_ticker: dict[str, pd.DataFrame] = {}
    missing_tickers: list[str] = []
    for ticker in requested_tickers:
        history_path = _find_history_path(ticker, resolved_history_dirs)
        if history_path is None:
            missing_tickers.append(ticker)
            continue
        ticker_frame = _read_daily_history(history_path, start_ts=start_ts, end_ts=end_ts)
        if ticker_frame.empty:
            missing_tickers.append(ticker)
            continue
        frames_by_ticker[ticker] = ticker_frame

    if not frames_by_ticker:
        raise FileNotFoundError("No Tickrake candle files were found for the requested universe.")

    if missing_tickers:
        preview = ", ".join(missing_tickers[:10])
        extra = "" if len(missing_tickers) <= 10 else f", +{len(missing_tickers) - 10} more"
        warnings.warn(
            f"Dropping {len(missing_tickers)} ticker(s) with missing Tickrake candle data: "
            f"{preview}{extra}",
            stacklevel=2,
        )

    date_index = pd.DatetimeIndex(
        sorted({timestamp for frame in frames_by_ticker.values() for timestamp in frame.index})
    )
    tickers_loaded = list(frames_by_ticker)
    ohlcv = {
        field: pd.DataFrame(index=date_index, columns=tickers_loaded, dtype=float)
        for field in ("open", "high", "low", "close", "volume")
    }
    for ticker, frame in frames_by_ticker.items():
        for field_name, column_name in (
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
        ):
            ohlcv[field_name].loc[frame.index, ticker] = frame[column_name].astype(float)

    close = ohlcv["close"].sort_index()
    returns = close.pct_change(fill_method=None).fillna(0.0)
    log_returns = np.log(close / close.shift(1))

    membership_mask = None
    if index_code is not None:
        membership_mask = index_membership_mask(
            tickers_loaded,
            index_code=index_code,
            start=start,
            end=end,
            sqlite_path=sqlite_db_path,
            date_index=date_index,
        )

    study_data = StudyData(
        tickers=tickers_loaded,
        open=ohlcv["open"].sort_index(),
        high=ohlcv["high"].sort_index(),
        low=ohlcv["low"].sort_index(),
        close=close,
        volume=ohlcv["volume"].sort_index(),
        returns=returns,
        log_returns=log_returns,
        interval=normalized_interval,
        provider=DEFAULT_PROVIDER,
        frequency="day",
        index_code=index_code,
        membership_mask=membership_mask,
        requested_tickers=requested_tickers,
    )
    if cache_paths is not None:
        _save_cached_study_data(cache_paths, request, study_data)
    return study_data


def index_range_tickers(
    index_code: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    sqlite_path: str | Path | None = None,
) -> list[str]:
    """Return the union of tickers that were index members during the date range."""
    start_date = pd.Timestamp(start).date().isoformat()
    end_date = pd.Timestamp(end).date().isoformat()
    db_path = _resolve_tickrake_sqlite_path(sqlite_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT tickers.ticker
            FROM market_index_memberships memberships
            INNER JOIN market_indexes indexes
              ON indexes.id = memberships.market_index_id
            INNER JOIN tickers
              ON tickers.id = memberships.ticker_id
            WHERE indexes.code = ?
              AND memberships.start_date <= ?
              AND (memberships.end_date IS NULL OR memberships.end_date >= ?)
            ORDER BY tickers.ticker ASC
            """,
            (index_code, end_date, start_date),
        ).fetchall()
    return [row[0] for row in rows]


def index_membership_mask(
    tickers: list[str],
    index_code: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    sqlite_path: str | Path | None = None,
    date_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a per-date boolean membership mask for the requested ticker set."""
    normalized_tickers = _normalize_tickers(tickers)
    if date_index is None:
        date_index = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    mask_index = pd.DatetimeIndex(date_index).sort_values()
    mask = pd.DataFrame(False, index=mask_index, columns=normalized_tickers, dtype=bool)
    if not normalized_tickers or mask.empty:
        return mask

    db_path = _resolve_tickrake_sqlite_path(sqlite_path)
    placeholders = ", ".join("?" for _ in normalized_tickers)
    params: list[Any] = [
        index_code,
        pd.Timestamp(end).date().isoformat(),
        pd.Timestamp(start).date().isoformat(),
        *normalized_tickers,
    ]
    query = f"""
        SELECT tickers.ticker, memberships.start_date, memberships.end_date
        FROM market_index_memberships memberships
        INNER JOIN market_indexes indexes
          ON indexes.id = memberships.market_index_id
        INNER JOIN tickers
          ON tickers.id = memberships.ticker_id
        WHERE indexes.code = ?
          AND memberships.start_date <= ?
          AND (memberships.end_date IS NULL OR memberships.end_date >= ?)
          AND tickers.ticker IN ({placeholders})
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    for ticker, membership_start, membership_end in rows:
        start_ts = pd.Timestamp(membership_start)
        end_ts = pd.Timestamp(membership_end) if membership_end else mask.index[-1]
        active = (mask.index >= start_ts) & (mask.index <= end_ts)
        mask.loc[active, ticker] = True
    return mask


def _resolve_data_dir(data_dir: str | Path | None) -> Path | None:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    config = load_studies_config()
    return config.data_root


def _resolve_tickrake_sqlite_path(sqlite_path: str | Path | None) -> Path:
    if sqlite_path is not None:
        return Path(sqlite_path).expanduser().resolve()
    return load_studies_config().tickrake_sqlite_path


def _resolve_history_dirs(history_dirs: list[str | Path] | None) -> tuple[Path, ...]:
    if history_dirs is not None:
        return tuple(Path(path).expanduser().resolve() for path in history_dirs)
    return load_studies_config().tickrake_history_dirs


def _build_download_request(
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    sqlite_path: Path,
    history_dirs: tuple[Path, ...],
    index_code: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "provider": DEFAULT_PROVIDER,
        "tickers": list(tickers),
        "start": start,
        "end": end,
        "interval": interval,
        "sqlite_path": str(sqlite_path),
        "history_dirs": [str(path) for path in history_dirs],
        "index_code": index_code,
    }


def _cache_paths(
    cache_root: Path,
    request: dict[str, object],
) -> dict[str, Path]:
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    base_dir = cache_root / "tickrake"
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


def _normalize_interval(interval: str) -> str:
    normalized = interval.strip().lower()
    if normalized not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval {interval!r}. Tickrake-backed qstudy currently supports only "
            "'1d' or 'day'."
        )
    return "1d"


def _normalize_tickers(tickers: list[str] | str | None) -> list[str]:
    if tickers is None:
        return []
    values = [tickers] if isinstance(tickers, str) else list(tickers)
    return list(dict.fromkeys(value.strip().upper() for value in values if value and value.strip()))


def _find_history_path(ticker: str, history_dirs: tuple[Path, ...]) -> Path | None:
    filename = f"{ticker}_day.csv"
    for history_dir in history_dirs:
        path = history_dir / filename
        if path.exists():
            return path
    return None


def _read_daily_history(
    history_path: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    frame = pd.read_csv(history_path)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    normalized = frame.rename(columns=str.lower)
    normalized["datetime"] = pd.to_datetime(normalized["datetime"], utc=True)
    normalized.index = normalized["datetime"].dt.tz_convert(None).dt.normalize()
    trimmed = normalized.loc[(normalized.index >= start_ts) & (normalized.index <= end_ts)]
    if trimmed.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    trimmed = trimmed.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    for column in ("open", "high", "low", "close", "volume"):
        trimmed[column] = pd.to_numeric(trimmed[column], errors="coerce")
    return trimmed.groupby(level=0).last().sort_index()


def get_sector_map(
    tickers: list[str],
    sqlite_path: str | Path | None = None,
) -> dict[str, str]:
    """Fetch sector classifications from Tickrake SQLite."""
    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        return {}

    placeholders = ", ".join("?" for _ in normalized_tickers)
    db_path = _resolve_tickrake_sqlite_path(sqlite_path)
    query = f"SELECT ticker, gics_sector FROM tickers WHERE ticker IN ({placeholders})"
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, normalized_tickers).fetchall()
    sectors = {ticker: sector or "Unknown" for ticker, sector in rows}
    return {ticker: sectors.get(ticker, "Unknown") for ticker in normalized_tickers}
