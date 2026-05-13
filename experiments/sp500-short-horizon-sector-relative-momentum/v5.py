"""v5: Combined best ideas — skip=5, rebalance=20, signal smoothing=3.

Hypothesis: combining the skip period (v2), longer rebalance (v1), and signal
smoothing (v3) compounds the turnover-reduction effects. Each operates on a
different channel:
  - skip=5:       removes short-term reversal noise from signal inputs
  - smooth=3:     stabilizes daily ranks, prevents marginal-position flips
  - rebalance=20: limits how often the neutralization scaler reshuffles weights
Together these should push daily turnover below 8% and cost drag below 2%/yr,
while the underlying 20-day sector-relative signal still holds predictive power.
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


def short_horizon_sector_relative_signal_smoothed(
    window: int = 20, skip: int = 0, smooth: int = 3
):
    def signal_fn(**cache):
        returns = cache["returns"]
        sector_map_s = pd.Series(sector_map).reindex(returns.columns).fillna("Unknown")
        stock_ret = returns.rolling(window).sum().shift(skip + 1)
        sector_ret = stock_ret.copy()
        for sector, tickers in sector_map_s.groupby(sector_map_s):
            cols = tickers.index.tolist()
            sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
        raw = stock_ret - sector_ret
        return raw.rolling(smooth).mean()

    signal_fn.__name__ = f"short_horizon_sector_relative_signal_smoothed_{window}_{skip}_{smooth}"
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
    Study(
        name="short_horizon_momentum_combined",
        universe=universe,
        benchmark=benchmark,
    )
    .base_signal(
        short_horizon_sector_relative_signal_smoothed(window=20, skip=5, smooth=3)
    )
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
