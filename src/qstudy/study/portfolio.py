from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tradeable constraint factories
# ---------------------------------------------------------------------------


def liquidity(top_n: int = 250, window: int = 60) -> Callable:
    """Factory: keep top_n assets by rolling average dollar volume.

    Returns a constraint function suitable for :meth:`Study.add_tradeable_constraint`.

    Args:
        top_n:  Number of most-liquid assets to keep eligible.
        window: Lookback for rolling average dollar volume.

    Returns:
        Constraint function ``fn(close, volume, **cache) -> pd.DataFrame[bool]``.
    """

    def fn(close, volume, **cache):
        return liquidity_filter(close, volume, top_n=top_n, window=window)

    fn.__name__ = f"liquidity(top_n={top_n})"
    return fn


def min_price(threshold: float = 5.0) -> Callable:
    """Factory: exclude assets whose close price is below a threshold.

    Returns a constraint function suitable for :meth:`Study.add_tradeable_constraint`.

    Args:
        threshold: Minimum close price (using previous day's close, shift(1)).

    Returns:
        Constraint function ``fn(close, **cache) -> pd.DataFrame[bool]``.
    """

    def fn(close, **cache):
        return close.shift(1) >= threshold

    fn.__name__ = f"min_price(threshold={threshold})"
    return fn


def min_adv(threshold: float = 1_000_000.0) -> Callable:
    """Factory: exclude assets whose rolling average daily dollar volume is below a threshold.

    Returns a constraint function suitable for :meth:`Study.add_tradeable_constraint`.

    Args:
        threshold: Minimum 20-day average dollar volume.

    Returns:
        Constraint function ``fn(close, volume, **cache) -> pd.DataFrame[bool]``.
    """

    def fn(close, volume, **cache):
        adv = (close * volume).rolling(20).mean()
        return adv >= threshold

    fn.__name__ = f"min_adv(threshold={threshold:.0f})"
    return fn


def liquidity_filter(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    top_n: int = 250,
    window: int = 60,
) -> pd.DataFrame:
    """Boolean mask keeping the top_n assets by rolling average dollar volume on each date.

    Args:
        close:   Closing price DataFrame (dates x tickers).
        volume:  Daily volume DataFrame (dates x tickers).
        top_n:   Number of most-liquid assets to keep eligible.
        window:  Lookback for rolling average dollar volume.

    Returns:
        Boolean DataFrame (dates x tickers), True = eligible for trading.
    """
    dollar_vol = (close * volume).dropna(axis=1)
    avg_dollar_vol = dollar_vol.rolling(window).mean()
    rank = avg_dollar_vol.rank(axis=1, ascending=False)
    return rank <= top_n


def build_long_short_positions(
    signal: pd.DataFrame,
    n_long: int = 25,
    n_short: int = 25,
) -> pd.DataFrame:
    """Convert a signal DataFrame into dollar-neutral positions.

    Steps:
      1. Rank signal cross-sectionally each day (NaN ranked last / excluded).
      2. Select top n_long as +1 and bottom n_short as -1.
      3. Normalize so abs(weights).sum(axis=1) == 1.0.

    Args:
        signal:  Signal DataFrame (dates x tickers). NaN = ineligible.
        n_long:  Number of long positions per rebalance date.
        n_short: Number of short positions per rebalance date.

    Returns:
        Float DataFrame of weights (dates x tickers), dollar-neutral.
    """
    signal_rank = signal.rank(axis=1, ascending=False, na_option="bottom")
    # rank has no NaNs after na_option='bottom', so count() == total columns every row
    n_total = signal_rank.count(axis=1)

    long_mask = signal_rank <= n_long
    short_cutoff = n_total - (n_short - 1)
    short_mask = signal_rank.ge(short_cutoff.values[:, None])

    positions = long_mask.astype(float) - short_mask.astype(float)
    abs_sum = positions.abs().sum(axis=1).replace(0, float("nan"))
    return positions.div(abs_sum, axis=0)


def build_long_only(
    signal: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Convert a signal DataFrame into long-only equal-weighted positions.

    Selects the top n assets by signal each day. Weights sum to 1.0.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.
        n:      Number of long positions per rebalance date.

    Returns:
        Float DataFrame of weights (dates x tickers), long-only, weights sum to 1.0.
    """
    ranks = signal.rank(axis=1, ascending=False, na_option="bottom")
    mask = ranks <= n
    positions = mask.astype(float)
    count = positions.sum(axis=1).replace(0, float("nan"))
    return positions.div(count, axis=0)


def rebalance(
    positions: pd.DataFrame,
    every: int = 5,
) -> pd.DataFrame:
    """Apply a rebalance schedule: keep positions only every N rows, forward-fill between.

    NaN convention: position builders must return NaN for ineligible stocks (not 0.0).
    This function preserves NaN values — it does not fill them with 0.0. The engine
    treats NaN positions as zero when computing returns.

    Args:
        positions: Positions DataFrame (dates x tickers). NaN = ineligible, not held.
        every:     Rebalance every N trading days (e.g. 5 = weekly, 21 = monthly).
                   Use 1 for daily (no-op).

    Returns:
        Rebalanced positions DataFrame, same shape as input.
    """
    if every == 1:
        return positions

    mask = np.arange(len(positions)) % every == 0
    result = positions.copy()
    result[~mask] = np.nan
    return result.ffill()
