"""
Residual Mean Reversion (5-day) with Correlation Regime Filter — Study pipeline version.

Strategy:
- Signal: negative 5-day rolling mean of residual returns
- Filters: vol, volume z-score, momentum context, liquidity
- Scalers: equity-curve regime (0.25x below 20-day MA) * cross-sectional correlation regime
           (0.25x when avg correlation is above its 80th percentile rolling threshold)
"""

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

start_date = "2015-01-01"
end_date = "2023-12-31"

universe = qs.download(SP500, start_date, end_date)
benchmark = qs.download(["SPY"], start_date, end_date)
factors = qs.download(["SPY", "XLK"], start_date, end_date)


def demean_signal(signal, **cache):
    """Cross-sectionally demean the signal each day."""
    return signal.sub(signal.mean(axis=1), axis=0)


def equity_curve_regime_scale(positions, **cache):
    """Scale down to 25% exposure when equity curve is below its 20-day MA."""
    returns = cache["returns"]
    liq_mask = cache.get("_liquidity_mask")
    if liq_mask is not None:
        returns = returns.where(liq_mask)
    raw_ret = (positions.shift(1) * returns).sum(axis=1)
    equity = (1 + raw_ret).cumprod()
    equity_ma = equity.rolling(20).mean()
    scale = pd.Series(
        np.where(equity > equity_ma, 1.0, 0.25),
        index=equity.index,
    )
    return positions.mul(scale.shift(1), axis=0)


def corr_regime_scale(positions, **cache):
    """Scale down to 25% exposure when avg cross-sectional correlation exceeds its 80th pctile.

    Mean reversion tends to fail when correlations spike — stocks move together rather than
    reverting to idiosyncratic levels.
    """
    returns = cache["returns"]
    corr_window = 20
    avg_corr = (
        returns.rolling(corr_window)
        .corr()
        .groupby(level=0)
        .mean()
        .mean(axis=1)
    )
    corr_thresh = avg_corr.rolling(60).quantile(0.8)
    scale = pd.Series(
        np.where(avg_corr < corr_thresh, 1.0, 0.25),
        index=avg_corr.index,
    )
    return positions.mul(scale.shift(1), axis=0)


study = (
    Study(
        universe=universe,
        benchmark=benchmark,
        factors=factors,
        name="residual_mr_5d_corr_filter",
    )
    .residualize_returns()
    .mean_reversion(window=5)
    .add_filter(demean_signal)
    .add_vol_filter(vol_window=5, quantile=0.6)
    .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
    .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
    .add_liquidity_filter(top_n=250)
    .build_long_short(n_long=25, n_short=25)
    .scale_returns(equity_curve_regime_scale)
    .scale_returns(corr_regime_scale)
    .run()
)

study.report()
