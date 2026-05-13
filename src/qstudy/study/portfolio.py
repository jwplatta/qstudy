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


def rebalance_on(
    positions: pd.DataFrame,
    trigger_fn: Callable[[pd.Series, pd.Series], bool],
    signal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply a threshold-triggered rebalance schedule.

    On each date, calls ``trigger_fn(current_row, proposed_row)`` to decide whether
    to adopt the new portfolio or carry forward the existing one. The trigger fires
    on the very first date unconditionally.

    When ``signal`` is provided, the trigger receives signal rows (raw alpha scores)
    rather than position weight rows. This is the preferred mode for
    :func:`rank_change_trigger` because neutralized weights change very little
    day-to-day even when the underlying signal ranking has shifted materially.

    NaN convention: same as :func:`rebalance` — NaN means ineligible/not held.

    Args:
        positions:   Full proposed positions DataFrame (dates x tickers), as if
                     rebalancing daily. NaN = ineligible.
        trigger_fn:  Callable ``(current: pd.Series, proposed: pd.Series) -> bool``.
                     Return ``True`` to rebalance (adopt ``proposed``), ``False``
                     to hold (carry ``current`` forward).
        signal:      Optional signal DataFrame (dates x tickers). When provided,
                     trigger receives signal rows instead of position rows.

    Returns:
        Rebalanced positions DataFrame, same shape as input.

    Built-in trigger factories (importable from ``qstudy``):
        - :func:`rank_change_trigger` — rebalance when signal rank-correlation drops below threshold
        - :func:`signal_zscore_trigger` — rebalance when signal z-score exceeds threshold
    """
    compare = signal if signal is not None else positions
    result = positions.copy()
    current_pos = positions.iloc[0]
    current_cmp = compare.iloc[0]
    for i, date in enumerate(positions.index):
        proposed_pos = positions.loc[date]
        proposed_cmp = compare.loc[date]
        if i == 0 or trigger_fn(current_cmp, proposed_cmp):
            current_pos = proposed_pos
            current_cmp = proposed_cmp
        else:
            result.loc[date] = current_pos
    return result


# ---------------------------------------------------------------------------
# Built-in trigger factories for rebalance_on
# ---------------------------------------------------------------------------


def rank_change_trigger(threshold: float = 0.7) -> Callable[[pd.Series, pd.Series], bool]:
    """Rebalance when rank-correlation between current and proposed *signal rows* drops below threshold.

    The trigger receives signal rows (not position weights) so it measures genuine
    re-ranking of the underlying alpha, not noise in the neutralized weight vector.
    A rank-corr near 1.0 means the ordering is stable; dropping below ``threshold``
    means enough names have re-ranked to justify the turnover cost of rebalancing.

    Args:
        threshold: Spearman rank-correlation cutoff in [-1, 1]. Default 0.7 means
                   rebalance only when the signal rank ordering agrees less than 70%
                   of the time between the current and proposed dates.
    """
    def trigger(current: pd.Series, proposed: pd.Series) -> bool:
        # Drop NaN (ineligible) tickers — only compare where both are ranked
        valid = current.notna() & proposed.notna()
        if valid.sum() < 4:
            return True
        c = current[valid]
        p = proposed[valid]
        rank_corr = float(c.rank().corr(p.rank()))
        return rank_corr < threshold

    trigger.__name__ = f"rank_change_trigger(threshold={threshold})"
    return trigger


def book_overlap_trigger(
    n: int = 20, min_overlap: float = 0.7
) -> Callable[[pd.Series, pd.Series], bool]:
    """Rebalance when the proposed top/bottom N names differ enough from the current book.

    Compares the *set* of top-N and bottom-N tickers between the current and proposed
    signal rows. If the Jaccard overlap (intersection / union) of the long set OR the
    short set falls below ``min_overlap``, the trigger fires.

    This is more robust than rank-correlation for concentrated books (e.g. 20L/20S out
    of 150) because it directly asks "how many names would actually change?" rather than
    measuring global rank shift across all eligible names.

    Args:
        n:           Number of names on each side of the book (must match build_long_short n_long/n_short).
        min_overlap: Jaccard similarity threshold. 0.7 means rebalance when less than
                     70% of the current long (or short) names would survive. Default 0.7.
    """
    def trigger(current: pd.Series, proposed: pd.Series) -> bool:
        valid = current.notna() & proposed.notna()
        if valid.sum() < n * 2:
            return True
        c = current[valid]
        p = proposed[valid]
        # Top-N = longs, Bottom-N = shorts
        c_longs = set(c.nlargest(n).index)
        c_shorts = set(c.nsmallest(n).index)
        p_longs = set(p.nlargest(n).index)
        p_shorts = set(p.nsmallest(n).index)
        long_overlap = len(c_longs & p_longs) / len(c_longs | p_longs)
        short_overlap = len(c_shorts & p_shorts) / len(c_shorts | p_shorts)
        return long_overlap < min_overlap or short_overlap < min_overlap

    trigger.__name__ = f"book_overlap_trigger(n={n}, min_overlap={min_overlap})"
    return trigger


def signal_zscore_trigger(
    signal: pd.DataFrame,
    threshold: float = 1.5,
    window: int = 20,
) -> Callable[[pd.Series, pd.Series], bool]:
    """Rebalance when the cross-sectional signal z-score magnitude is large enough.

    Uses a closure over the full signal DataFrame to compute a rolling cross-sectional
    mean z-score at each date. High z-score = signal is extreme = worth paying turnover.
    Low z-score = signal is weak = cheaper to hold current book.

    Args:
        signal:    The raw signal DataFrame (dates x tickers), same index as positions.
        threshold: Z-score magnitude cutoff. Default 1.5.
        window:    Rolling window for computing signal mean/std. Default 20.
    """
    rolling_mean = signal.rolling(window).mean()
    rolling_std = signal.rolling(window).std().replace(0.0, np.nan)
    zscores = (signal - rolling_mean) / rolling_std
    mean_abs_z = zscores.abs().mean(axis=1)

    def trigger(_current: pd.Series, proposed: pd.Series) -> bool:
        # proposed.name is the date index label when iterating via .loc[date]
        date = proposed.name
        if date not in mean_abs_z.index:
            return True
        return float(mean_abs_z.loc[date]) >= threshold

    trigger.__name__ = f"signal_zscore_trigger(threshold={threshold})"
    return trigger
