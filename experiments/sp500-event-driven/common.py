from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
CACHE_DIR = Path.home() / ".qstudy" / "sp500_event_driven"


@cache
def load_sp500_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


def emit_metrics(study: Study) -> None:
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))


def save_study(study: Study, name: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    study.save(CACHE_DIR / f"{name}.pkl")


def demean(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    return signal.sub(signal.mean(axis=1), axis=0)


def gap_return(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
    return open_prices.div(close_prices.shift(1)).sub(1.0)


def intraday_return(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
    safe_open = open_prices.replace(0.0, np.nan)
    return close_prices.div(safe_open).sub(1.0)


def relative_volume(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return volume.div(volume.rolling(window).mean().replace(0.0, np.nan))


def range_fraction(
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
) -> pd.DataFrame:
    safe_open = open_prices.replace(0.0, np.nan)
    return high_prices.sub(low_prices).div(safe_open)


def close_location_value(
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    denom = high_prices.sub(low_prices).replace(0.0, np.nan)
    return close_prices.sub(low_prices).div(denom)


def top_abs_quantile_mask(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    threshold = frame.abs().quantile(quantile, axis=1)
    return frame.abs().ge(threshold, axis=0)


def gap_fade_signal(
    gap_quantile: float = 0.9,
    rel_volume_window: int = 20,
    rel_volume_quantile: float = 0.55,
):
    def signal_fn(**cache):
        open_prices = cache["open"]
        close_prices = cache["close"]
        volume = cache["volume"]
        gap = gap_return(open_prices, close_prices)
        rel_volume = relative_volume(volume, window=rel_volume_window)

        signal = -gap
        mask = top_abs_quantile_mask(gap, gap_quantile)
        mask &= rel_volume.le(rel_volume.quantile(rel_volume_quantile, axis=1), axis=0)
        return signal.where(mask)

    signal_fn.__name__ = f"gap_fade_signal_{gap_quantile}_{rel_volume_window}_{rel_volume_quantile}"
    return signal_fn


def volume_shock_continuation_signal(
    event_window: int = 5,
    volume_window: int = 20,
    volume_quantile: float = 0.9,
    move_quantile: float = 0.75,
):
    def signal_fn(**cache):
        returns = cache["returns"]
        volume = cache["volume"]
        price_move = returns.rolling(event_window).sum()
        rel_volume = relative_volume(volume, window=volume_window)
        volume_shock = np.log(rel_volume.replace(0.0, np.nan))
        signal = price_move.mul(volume_shock)

        mask = rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)
        mask &= top_abs_quantile_mask(price_move, move_quantile)
        return signal.where(mask)

    signal_fn.__name__ = (
        "volume_shock_continuation_signal_"
        f"{event_window}_{volume_window}_{volume_quantile}_{move_quantile}"
    )
    return signal_fn


def intraday_exhaustion_reversal_signal(
    range_window: int = 20,
    range_quantile: float = 0.9,
    volume_window: int = 20,
    volume_quantile: float = 0.7,
    move_quantile: float = 0.8,
):
    def signal_fn(**cache):
        open_prices = cache["open"]
        high_prices = cache["high"]
        low_prices = cache["low"]
        close_prices = cache["close"]
        volume = cache["volume"]

        day_move = intraday_return(open_prices, close_prices)
        day_range = range_fraction(high_prices, low_prices, open_prices)
        close_location = close_location_value(high_prices, low_prices, close_prices)
        rel_volume = relative_volume(volume, window=volume_window)

        range_threshold = day_range.quantile(range_quantile, axis=1)
        move_threshold = day_move.abs().quantile(move_quantile, axis=1)
        exhaustion_side = (0.5 - close_location) * 2.0
        signal = exhaustion_side.mul(day_move.abs()).mul(day_range)

        mask = day_range.ge(range_threshold, axis=0)
        mask &= day_move.abs().ge(move_threshold, axis=0)
        mask &= rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)

        # A second range filter vs each stock's own recent history avoids ranking quiet names.
        rolling_range_median = day_range.rolling(range_window).median()
        mask &= day_range.ge(rolling_range_median)
        return signal.where(mask)

    signal_fn.__name__ = (
        "intraday_exhaustion_reversal_signal_"
        f"{range_window}_{range_quantile}_{volume_window}_{volume_quantile}_{move_quantile}"
    )
    return signal_fn


def build_gap_fade_study(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(gap_fade_signal())
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_long_short(n_long=25, n_short=25)
        .run()
    )


def build_volume_shock_continuation_study(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(volume_shock_continuation_signal())
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=25, n_short=25)
        .run()
    )


def build_intraday_exhaustion_study(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(intraday_exhaustion_reversal_signal())
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_long_short(n_long=25, n_short=25)
        .run()
    )
