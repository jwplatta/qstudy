"""v6: Threshold-triggered rebalance — gate the full neutralization + rebalance together.

Key insight from earlier attempts: the beta neutralizer running daily is the primary
source of turnover, not the position builder. Any trigger placed *before* neutralization
still lets the neutralizer produce fresh weights every day (high turnover). Any trigger
placed *after* neutralization produces a massive catch-up trade on trigger days because
the forward-filled stale weights are completely replaced by a freshly neutralized book.

Solution: combine the book-change detection AND the neutralization into one scaler that
only neutralizes when the underlying book has actually changed enough. On hold days,
the previous post-neutralization weights are carried forward unchanged — zero turnover.

Trigger: Jaccard overlap of top-N / bottom-N signal names. Fire when < min_overlap of
the current long or short book would survive (i.e., >= 30% rotation in either leg).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

universe = qs.download(SP500, "2015-01-01", "2023-12-31")
benchmark = qs.download(["SPY"], "2015-01-01", "2023-12-31")

sector_map: dict[str, str] = qs.get_sector_map(SP500)


def short_horizon_sector_relative_signal(window: int = 20, skip: int = 0):
    def signal_fn(**cache):
        returns = cache["returns"]
        sector_map_s = pd.Series(sector_map).reindex(returns.columns).fillna("Unknown")
        stock_ret = returns.rolling(window).sum().shift(skip + 1)
        sector_ret = stock_ret.copy()
        for sector, tickers in sector_map_s.groupby(sector_map_s):
            cols = tickers.index.tolist()
            sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
        return stock_ret - sector_ret

    signal_fn.__name__ = f"short_horizon_sector_relative_signal_{window}_{skip}"
    return signal_fn


def sector_beta_neutralize_positions(window: int = 60, passes: int = 2):
    def scaler(positions, **cache):
        returns = cache["returns"]
        bench = cache["benchmark"].reindex(returns.index).fillna(0.0)
        mean_r = returns.rolling(window).mean()
        mean_b = bench.rolling(window).mean()
        cov = (
            returns.mul(bench, axis=0)
            .rolling(window)
            .mean()
            .sub(mean_r.mul(mean_b, axis=0))
        )
        var_b = bench.rolling(window).var().replace(0.0, np.nan)
        betas = cov.div(var_b, axis=0).shift(1)
        adjusted = positions.copy()
        for date in positions.index:
            active = positions.loc[date]
            active = active[active != 0.0]
            if active.empty:
                continue
            weights = active.copy()
            for _ in range(passes):
                sectors = pd.Series(
                    {t: sector_map.get(t, "Unknown") for t in weights.index}
                )
                for s, tickers in sectors.groupby(sectors):
                    cols = tickers.index.tolist()
                    sw = weights.loc[cols]
                    weights.loc[cols] = sw - sw.mean()
                beta_slice = betas.loc[date, weights.index].dropna()
                if len(beta_slice) >= 2:
                    w = weights.reindex(beta_slice.index).fillna(0.0)
                    beta_exp = float((w * beta_slice).sum())
                    beta_norm = float((beta_slice**2).sum())
                    if beta_norm > 0.0:
                        weights.loc[beta_slice.index] = (
                            w - (beta_exp / beta_norm) * beta_slice
                        )
            weights = weights - weights.mean()
            gross = weights.abs().sum()
            if gross > 0.0 and not pd.isna(gross):
                adjusted.loc[date, weights.index] = weights / gross
        return adjusted.fillna(0.0)

    scaler.__name__ = f"sector_beta_neutralize_positions_{window}_{passes}"
    return scaler


def neutralize_on_book_change(
    n: int = 20,
    min_overlap: float = 0.7,
    beta_window: int = 60,
    beta_passes: int = 2,
):
    """Combined scaler: gate neutralization + carry forward on book-stable days.

    On each date:
      1. Check if the top-N / bottom-N signal names have changed enough (Jaccard < min_overlap).
      2. If yes: run full sector + beta neutralization and store the result.
      3. If no:  carry the previous post-neutralization weights forward unchanged.

    This ensures zero turnover on hold days (no neutralizer drift) and full fresh
    neutralization only when the book has genuinely rotated.
    """
    state: dict = {"last_signal": None, "last_weights": None}

    def _neutralize_one_date(date, weights, betas):
        """Run sector + beta neutralization for a single date's weights."""
        for _ in range(beta_passes):
            sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
            for _s, tickers in sectors.groupby(sectors):
                cols = tickers.index.tolist()
                sw = weights.loc[cols]
                weights.loc[cols] = sw - sw.mean()
            beta_slice = betas.loc[date, weights.index].dropna()
            if len(beta_slice) >= 2:
                w = weights.reindex(beta_slice.index).fillna(0.0)
                beta_exp = float((w * beta_slice).sum())
                beta_norm = float((beta_slice**2).sum())
                if beta_norm > 0.0:
                    weights.loc[beta_slice.index] = w - (beta_exp / beta_norm) * beta_slice
        weights = weights - weights.mean()
        gross = weights.abs().sum()
        if gross > 0.0 and not pd.isna(gross):
            weights = weights / gross
        return weights

    def scaler(positions, **cache):
        returns = cache["returns"]
        bench = cache["benchmark"].reindex(returns.index).fillna(0.0)
        signal = cache.get("signal")

        # Pre-compute rolling betas (same as sector_beta_neutralize_positions)
        mean_r = returns.rolling(beta_window).mean()
        mean_b = bench.rolling(beta_window).mean()
        cov = (
            returns.mul(bench, axis=0)
            .rolling(beta_window)
            .mean()
            .sub(mean_r.mul(mean_b, axis=0))
        )
        var_b = bench.rolling(beta_window).var().replace(0.0, np.nan)
        betas = cov.div(var_b, axis=0).shift(1)

        adjusted = positions.copy()

        for date in positions.index:
            active = positions.loc[date]
            active = active[active != 0.0]
            if active.empty:
                adjusted.loc[date] = 0.0
                continue

            sig_row = signal.loc[date] if signal is not None else None

            # Determine whether to rebalance
            should_rebalance = state["last_signal"] is None
            if not should_rebalance and sig_row is not None:
                prev_sig = state["last_signal"]
                valid = prev_sig.notna() & sig_row.notna()
                if valid.sum() >= n * 2:
                    prev_longs = set(prev_sig[valid].nlargest(n).index)
                    prev_shorts = set(prev_sig[valid].nsmallest(n).index)
                    new_longs = set(sig_row[valid].nlargest(n).index)
                    new_shorts = set(sig_row[valid].nsmallest(n).index)
                    long_overlap = len(prev_longs & new_longs) / len(prev_longs | new_longs)
                    short_overlap = len(prev_shorts & new_shorts) / len(prev_shorts | new_shorts)
                    should_rebalance = long_overlap < min_overlap or short_overlap < min_overlap
                else:
                    should_rebalance = True

            if should_rebalance:
                weights = active.copy()
                weights = _neutralize_one_date(date, weights, betas)
                state["last_signal"] = sig_row
                state["last_weights"] = weights
                adjusted.loc[date, weights.index] = weights
                # Zero out any tickers not in the new book
                all_tickers = adjusted.columns
                not_in_book = all_tickers.difference(weights.index)
                adjusted.loc[date, not_in_book] = 0.0
            else:
                # Carry forward last post-neutralization weights unchanged
                adjusted.loc[date] = 0.0
                if state["last_weights"] is not None:
                    adjusted.loc[date, state["last_weights"].index] = state["last_weights"]

        return adjusted.fillna(0.0)

    scaler.__name__ = f"neutralize_on_book_change(n={n}, min_overlap={min_overlap})"
    return scaler


study = (
    Study(name="short_horizon_momentum_v6", universe=universe, benchmark=benchmark)
    .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
    .add_tradeable_constraint(qs.min_price(5.0))
    .add_tradeable_constraint(qs.min_adv(20_000_000.0))
    .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
    .build_long_short(n_long=20, n_short=20)
    .weight_equal()
    .scale_risk(neutralize_on_book_change(n=20, min_overlap=0.7, beta_window=60, beta_passes=2))
    .with_transaction_costs(10)
    .run()
)

if __name__ == "__main__":
    study.report()
