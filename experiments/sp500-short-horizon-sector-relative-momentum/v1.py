"""v1: Longer rebalance cadence (every=20) to cut turnover.

v0 baseline: daily turnover ~17%, cost drag ~4.3%/yr.
Hypothesis: doubling the rebalance period from 10 to 20 days roughly halves turnover,
reducing cost drag by ~2%/yr with minimal signal decay at a 20-day horizon.
All other parameters identical to v0.
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


study = (
    Study(name="short_horizon_momentum_rebal20", universe=universe, benchmark=benchmark)
    .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
    .add_tradeable_constraint(qs.min_price(5.0))
    .add_tradeable_constraint(qs.min_adv(20_000_000.0))
    .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
    .build_long_short(n_long=20, n_short=20)
    .weight_equal()
    .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
    .rebalance(every=20)
    .with_transaction_costs(10)
    .run()
)

if __name__ == "__main__":
    study.report()
