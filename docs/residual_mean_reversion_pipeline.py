"""
Residual Mean Reversion — rewritten using the Study pipeline.

Original: manual steps using raw qstudy functions.
This version: same strategy expressed as a Study pipeline.

The equity-curve regime filter (scale to 0.25x when below 20-day MA)
is the one step that requires a custom function since it's not a
built-in filter. Everything else maps directly to pipeline methods.
"""

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study

start_date = "2015-01-01"
end_date = "2023-12-31"

# Download all data upfront — do this once, reuse across studies.
universe = qs.download(index_code="SP500", start=start_date, end=end_date)
benchmark = qs.download(["SPY"], start_date, end_date)
factors = qs.download(["SPY", "XLK"], start_date, end_date)


# ---------------------------------------------------------------------------
# Custom signal mutation
# ---------------------------------------------------------------------------
# The original code demeaned the signal after computing it:
#   signal = signal.sub(signal.mean(axis=1), axis=0)
# This is a signal mutation (not a filter), so we pass it as add_filter.
# We return the same shape — no cells are zeroed, just re-centered.


def demean_signal(signal, **cache):
    """Cross-sectionally demean the signal each day."""
    return signal.sub(signal.mean(axis=1), axis=0)


# ---------------------------------------------------------------------------
# Custom position scaler: equity-curve regime filter
# ---------------------------------------------------------------------------
# Scale positions to 0.25x when the cumulative equity curve is below its
# 20-day moving average, full exposure (1.0x) when above.


def equity_curve_regime_scale(positions, **cache):
    """Scale down to 25% exposure when equity curve is below its 20-day MA."""
    returns = cache["returns"]
    liq_mask = cache.get("_liquidity_mask")
    if liq_mask is not None:
        returns = returns.where(liq_mask)
    # Approximate equity curve from positions as they currently stand
    raw_ret = (positions.shift(1) * returns).sum(axis=1)
    equity = (1 + raw_ret).cumprod()
    equity_ma = equity.rolling(20).mean()
    scale = pd.Series(
        np.where(equity > equity_ma, 1.0, 0.25),
        index=equity.index,
    )
    return positions.mul(scale.shift(1), axis=0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
study = (
    Study(
        universe=universe,
        benchmark=benchmark,
        factors=factors,
        name="residual_mean_reversion",
    )
    .residualize_returns()  # produces residual_returns in cache
    .mean_reversion(window=60)  # signal = -residuals.rolling(60).mean().shift(1)
    .add_filter(demean_signal)  # signal = signal.sub(signal.mean(axis=1), axis=0)
    .add_vol_filter(
        vol_window=5, quantile=0.6
    )  # qs.vol_filter(signal, residuals, vol_window=5, quantile=0.6)
    .add_volume_zscore_filter(  # qs.volume_zscore_filter(signal, volume, ...)
        window=30,
        min_zscore_quantile=0.8,
    )
    .add_momentum_context_filter(  # signal.where(med_mom.abs() < med_mom.quantile(0.7))
        window=60,
        max_abs_quantile=0.7,
    )
    .add_liquidity_filter(top_n=250)  # qs.liquidity_filter(close, volume, top_n=250)
    .build_long_short(n_long=25, n_short=25)  # qs.build_long_short_positions(signal, 25, 25)
    .scale_returns(equity_curve_regime_scale)  # 0.25x exposure when below 20-day equity MA
    .run()
)

study.report()
