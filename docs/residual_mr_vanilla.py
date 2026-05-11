"""
Residual Mean Reversion — vanilla implementation (no qstudy library).

Implements the full pipeline in plain pandas/numpy so the logic is transparent
and independently verifiable. Uses the same seed-42 synthetic data as the
equivalence tests in tests/test_qstudy.py::TestStudyPipelineEquivalence.

Steps:
  1. Generate synthetic universe + factor returns
  2. Residualize returns against factors (per-ticker OLS)
  3. Build signal: negative 5-day rolling mean of residuals
  4. Demean signal cross-sectionally
  5. Vol filter: keep stocks below 60th pct of realized vol
  6. Volume z-score filter: keep stocks above 70th pct of volume z-score
  7. Liquidity filter: keep top 15 stocks by rolling dollar volume
  8. Build long/short positions (top 3 long, bottom 3 short)
  9. Equity curve regime filter: scale to 25% when below 10-day MA
 10. Run backtest with 1-day execution lag
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# 1. Synthetic data  (identical to make_factor_study_data(seed=42))
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
N_DATES, N_TICKERS = 200, 30
dates = pd.bdate_range("2020-01-01", periods=N_DATES)
tickers = [f"T{i:02d}" for i in range(N_TICKERS)]

mkt = rng.normal(0, 0.01, N_DATES)
returns_arr = (
    mkt[:, None] * rng.uniform(0.5, 1.5, N_TICKERS)[None, :]
    + rng.normal(0, 0.005, (N_DATES, N_TICKERS))
)
returns_df = pd.DataFrame(returns_arr, index=dates, columns=tickers)
close_df = (1 + returns_df).cumprod() * 100
volume_df = pd.DataFrame(
    rng.integers(100_000, 10_000_000, (N_DATES, N_TICKERS)).astype(float),
    index=dates,
    columns=tickers,
)

f1 = mkt + rng.normal(0, 0.003, N_DATES)
f2 = mkt + rng.normal(0, 0.003, N_DATES)
factor_ret = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)

# ---------------------------------------------------------------------------
# 2. Residualize returns against factors (per-ticker time-series OLS)
#    Mirrors qstudy.residualize() exactly.
# ---------------------------------------------------------------------------

common_index = returns_df.index.intersection(factor_ret.index)
r = returns_df.loc[common_index]
f = sm.add_constant(factor_ret.loc[common_index])

residuals = pd.DataFrame(index=common_index, columns=r.columns, dtype=float)
for ticker in r.columns:
    y = r[ticker].dropna()
    x = f.loc[y.index]
    model = sm.OLS(y, x).fit()
    residuals.loc[y.index, ticker] = model.resid

residuals_df = residuals

# ---------------------------------------------------------------------------
# 3. Base signal: negative 5-day rolling mean of residuals
#    No .shift(1) here — the backtest engine applies the 1-day execution lag.
# ---------------------------------------------------------------------------

signal = -residuals_df.rolling(5).mean()

# ---------------------------------------------------------------------------
# 4. Demean signal cross-sectionally each day
# ---------------------------------------------------------------------------

signal = signal.sub(signal.mean(axis=1), axis=0)

# ---------------------------------------------------------------------------
# 5. Vol filter: keep stocks below 60th percentile of realized vol
#    Mirrors vol_filter(signal, residuals_df, vol_window=5, quantile=0.6)
# ---------------------------------------------------------------------------

realized_vol = residuals_df.rolling(5).std()
vol_thresh = realized_vol.quantile(0.6, axis=1)
vol_mask = realized_vol.lt(vol_thresh, axis=0)
signal = signal.where(vol_mask)

# ---------------------------------------------------------------------------
# 6. Volume z-score filter: keep stocks above 70th pct of volume z-score
#    Mirrors volume_zscore_filter(signal, volume_df, window=20, min_zscore_quantile=0.7)
# ---------------------------------------------------------------------------

vol_mean = volume_df.rolling(20).mean()
vol_std = volume_df.rolling(20).std()
vol_z = (volume_df - vol_mean) / vol_std
zscore_thresh = vol_z.quantile(0.7, axis=1)
zscore_mask = vol_z.ge(zscore_thresh, axis=0)
signal = signal.where(zscore_mask)

# ---------------------------------------------------------------------------
# 7. Liquidity filter: keep top 15 stocks by rolling 30-day dollar volume
#    Mirrors liquidity_filter(close_df, volume_df, top_n=15, window=30)
# ---------------------------------------------------------------------------

dollar_vol = (close_df * volume_df).dropna(axis=1)
avg_dollar_vol = dollar_vol.rolling(30).mean()
rank = avg_dollar_vol.rank(axis=1, ascending=False)
liq_mask = rank <= 15

signal = signal.where(liq_mask)           # NaN ineligible stocks (not 0.0)
ret_filtered = returns_df.where(liq_mask) # mask returns for backtest

# ---------------------------------------------------------------------------
# 8. Build long/short positions
#    Mirrors build_long_short_positions(signal, n_long=3, n_short=3)
#    - rank descending, NaN → bottom (highest rank number)
#    - top 3 = long (+1), bottom 3 = short (-1)
#    - normalize so abs(weights).sum(axis=1) == 1.0
# ---------------------------------------------------------------------------

signal_rank = signal.rank(axis=1, ascending=False, na_option="bottom")
n_total = signal_rank.count(axis=1)

long_mask = signal_rank <= 3
short_cutoff = n_total - (3 - 1)
short_mask = signal_rank.ge(short_cutoff.values[:, None])

positions = long_mask.astype(float) - short_mask.astype(float)
abs_sum = positions.abs().sum(axis=1).replace(0, float("nan"))
positions = positions.div(abs_sum, axis=0).fillna(0.0)

# ---------------------------------------------------------------------------
# 9. Raw backtest (no regime filter yet)
#    Mirrors engine.run(positions, ret_filtered): positions.shift(1) * returns
# ---------------------------------------------------------------------------

raw_port_ret = (positions.shift(1) * ret_filtered).sum(axis=1)

# ---------------------------------------------------------------------------
# 10. Equity curve regime filter
#     Scale to 25% exposure when equity curve is below its 10-day MA.
#     scale.shift(1): use yesterday's regime to size today's positions.
# ---------------------------------------------------------------------------

equity = (1 + raw_port_ret).cumprod()
equity_ma = equity.rolling(10).mean()
scale = pd.Series(np.where(equity > equity_ma, 1.0, 0.25), index=equity.index)

scaled_positions = positions.mul(scale.shift(1), axis=0)

# ---------------------------------------------------------------------------
# 11. Final backtest with scaled positions
# ---------------------------------------------------------------------------

port_ret = (scaled_positions.shift(1) * ret_filtered).sum(axis=1)

# ---------------------------------------------------------------------------
# 12. Metrics (vanilla implementations)
# ---------------------------------------------------------------------------

trading_days = 252

ann_return = (1 + port_ret).prod() ** (trading_days / len(port_ret)) - 1
ann_vol = port_ret.std() * np.sqrt(trading_days)
sharpe = port_ret.mean() / port_ret.std() * np.sqrt(trading_days)  # arithmetic, matches library

cum_ret = (1 + port_ret).cumprod()
rolling_max = cum_ret.cummax()
drawdown = (cum_ret - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"sharpe:       {sharpe:.6f}")
print(f"ann_return:   {ann_return:.6f}")
print(f"ann_vol:      {ann_vol:.6f}")
print(f"max_drawdown: {max_drawdown:.6f}")
print()
print("first 5 daily returns:")
print(port_ret.head(5).tolist())
print("last 5 daily returns:")
print(port_ret.tail(5).tolist())
