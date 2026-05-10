import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

start_date = "2015-01-01"
end_date = "2023-12-31"
universe = qs.download(SP500, start_date, end_date)
factors = qs.download(["SPY", "XLK"], start_date, end_date)
benchmark = qs.download("SPY", start_date, end_date)

returns_df = universe.returns
close_df = universe.close
volume_df = universe.volume
factor_returns = factors.returns
benchmark_returns = benchmark.returns

residuals_df, factor_params, factor_rsq = qs.residualize(returns_df, factor_returns)

# -------------------------------------------------------------------
# residual mean reversion base signal
# -------------------------------------------------------------------

signal = -residuals_df.rolling(60).mean().shift(1)
signal = signal.sub(signal.mean(axis=1), axis=0)

# -------------------------------------------------------------------
# conditioning filters
# -------------------------------------------------------------------

signal = qs.vol_filter(signal, residuals_df, vol_window=5, quantile=0.6)

signal = qs.volume_zscore_filter(signal, volume_df, window=30, min_zscore_quantile=0.8)

med_mom = residuals_df.rolling(60).mean()

signal = signal.where(med_mom.abs().lt(med_mom.quantile(0.7, axis=1), axis=0))

# -------------------------------------------------------------------
# liquidity universe
# -------------------------------------------------------------------

liq_mask = qs.liquidity_filter(close_df, volume_df, top_n=250)

signal = signal.where(liq_mask)
ret_filtered = returns_df.where(liq_mask)

# -------------------------------------------------------------------
# portfolio construction
# -------------------------------------------------------------------

positions = qs.build_long_short_positions(signal, n_long=25, n_short=25)

# -------------------------------------------------------------------
# raw strategy returns
# -------------------------------------------------------------------

raw_port_ret = qs.run(positions, ret_filtered)

# -------------------------------------------------------------------
# equity curve regime filter
# -------------------------------------------------------------------

equity_curve = (1 + raw_port_ret).cumprod()
equity_ma = equity_curve.rolling(20).mean()
exposure_scale = pd.Series(np.where(equity_curve > equity_ma, 1.0, 0.25), index=equity_curve.index)
scaled_positions = positions.mul(exposure_scale.shift(1), axis=0)

# -------------------------------------------------------------------
# final returns
# -------------------------------------------------------------------
port_ret = qs.run(scaled_positions, ret_filtered)

# -------------------------------------------------------------------
# diagnostics
# -------------------------------------------------------------------

print(qs.metrics.summary(port_ret, scaled_positions, benchmark=benchmark_returns))

ax = qs.summary_plot(port_ret, figsize=(8, 10))
