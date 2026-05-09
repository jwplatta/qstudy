# qstudy

Lightweight backtesting library for cross-sectional equity research. Atomic, composable pieces — the signal formula is the only thing you write, everything else is one function call.

## Install

```python
import qstudy as qs
from qstudy.constants import SP500, SECTOR_ETFS
```

---

## Data

Single API call returns a dict with `close`, `volume`, `returns`, `log_returns`, and the raw `data`.

```python
d = qs.download(SP500, start="2015-01-01", end="2023-12-31")

close_df    = d["close"]       # (dates x tickers) adjusted close
volume_df   = d["volume"]      # (dates x tickers) daily volume
returns_df  = d["returns"]     # close.pct_change().fillna(0)
log_ret_df  = d["log_returns"] # log(close / close.shift(1))
```

For factor/benchmark tickers (used in `residualize`):

```python
d_factors = qs.download(["SPY", "XLK"], start="2015-01-01", end="2023-12-31")
factor_returns = d_factors["returns"]
```

---

## Signals

Signals are plain DataFrames `(dates x tickers)`. You write the formula inline — that's the research. Filters zero out ineligible assets by setting them to `NaN`.

### Signal formula (inline in notebook)

```python
# Mean reversion
signal = -returns_df.rolling(5).mean().shift(1)

# Momentum
signal = returns_df.shift(5).rolling(120).mean().shift(1)

# Residual mean reversion (after factor stripping)
residuals, params, rsq = qs.residualize(returns_df, factor_returns)
signal = -residuals.rolling(5).mean().shift(1)
```

### Filters

All filters take a signal DataFrame and return a filtered copy with `NaN` where the condition fails. They compose by chaining.

```python
# Keep assets with low realized vol (cross-sectional, below quantile on each date)
signal = qs.vol_filter(signal, returns_df, vol_window=40, quantile=0.75, keep="low")

# Keep assets with above-average volume activity
signal = qs.volume_zscore_filter(signal, volume_df, window=10, min_zscore_quantile=0.65)

# Keep assets with weak medium-term momentum (for mean reversion)
signal = qs.momentum_context_filter(signal, returns_df, window=15, max_abs_quantile=0.75)
```

### `residualize`

Strips market/sector factor returns via OLS, returns idiosyncratic residuals.

```python
residuals, params_df, rsq_s = qs.residualize(returns_df, factor_returns)
# residuals: same shape as returns_df
# params_df: (tickers x factors+const) regression coefficients
# rsq_s:     per-ticker R²
```

---

## Portfolio Construction

### Liquidity filter

Returns a boolean mask — `True` where the asset is in the top N by rolling dollar volume. Apply with `.where()`.

```python
liq_mask = qs.liquidity_filter(close_df, volume_df, top_n=250, window=60)
signal = signal.where(liq_mask)
ret_filtered = returns_df.where(liq_mask)
```

### Build positions

Ranks signal cross-sectionally, selects top `n_long` as `+1` and bottom `n_short` as `-1`, normalizes to dollar-neutral (abs weights sum to 1).

```python
positions = qs.build_long_short_positions(signal, n_long=25, n_short=25)
```

> **Note:** `NaN` signals are ranked last (`na_option='bottom'`), so assets outside the liquidity mask will land in the short bucket. Always apply `liquidity_filter` before `build_long_short_positions`.

### Rebalance

Hold positions for `every` trading days, forward-filling between rebalance dates.

```python
positions = qs.rebalance(positions, every=5)   # weekly
positions = qs.rebalance(positions, every=21)  # monthly
# every=1 is a no-op (daily rebalance)
```

---

## Backtest Engine

Applies a 1-day execution lag: position on day T earns the return on day T+1.

```python
port_ret = qs.run(positions, ret_filtered)
```

---

## Metrics

```python
qs.metrics.sharpe(port_ret)                  # annualized Sharpe
qs.metrics.annualized_return(port_ret)       # CAGR
qs.metrics.annualized_vol(port_ret)          # annualized volatility
qs.metrics.max_drawdown(port_ret)            # peak-to-trough (negative fraction)
qs.metrics.max_drawdown_duration(port_ret)   # (days, (start_date, end_date)) | (0, None)
qs.metrics.drawdown_series(port_ret)         # full drawdown time series
qs.metrics.rolling_sharpe(port_ret, window=90)
qs.metrics.turnover(positions)               # daily one-way turnover

# All at once
qs.metrics.summary(port_ret, positions)
# sharpe, ann_return, ann_vol, max_drawdown, max_drawdown_duration,
# max_drawdown_start, max_drawdown_end, avg_daily_turnover
```

---

## Charts

All chart functions accept an optional `ax` and return the `Axes` for further customization. No forced `plt.show()` except `summary_plot`.

```python
qs.equity_curve(port_ret)
qs.drawdown_plot(port_ret)
qs.rolling_sharpe_plot(port_ret, window=90)
qs.summary_plot(port_ret)                    # 3-panel: equity / drawdown / rolling Sharpe

qs.corr_heatmap(returns_matrix)              # correlation heatmap from a returns DataFrame
qs.param_heatmap(results_df, row_param="qt", col_param="window", metric="metric", figsize=(12, 8))
```

---

## Parameter Grid Search

```python
liq_mask = qs.liquidity_filter(close_df, volume_df, top_n=250)
ret_filtered = returns_df.where(liq_mask)

def run_backtest(params):
    signal = -returns_df.rolling(params["window"]).mean().shift(1)
    signal = qs.vol_filter(signal, returns_df, vol_window=params["vol_wind"], quantile=params["qt"])
    signal = signal.where(liq_mask)
    positions = qs.build_long_short_positions(signal, n_long=25, n_short=25)
    return qs.run(positions, ret_filtered)

results = qs.param_grid(
    {"window": [5, 10, 15], "vol_wind": [20, 40], "qt": [0.6, 0.75, 0.9]},
    run_backtest,
    metric_fn=qs.metrics.sharpe,   # or omit for full summary()
)

qs.param_heatmap(results, row_param="qt", col_param="window", metric="metric")
```

`param_grid` returns a DataFrame with one row per combination. `metric_fn=qs.metrics.sharpe` stores the result in a `"metric"` column; omitting `metric_fn` calls `qs.metrics.summary()` and expands all metrics as columns.

---

## Full Example: Residual Mean Reversion

```python
import qstudy as qs
from qstudy.constants import SP500

d = qs.download(SP500, start="2015-01-01", end="2023-12-31")
close_df, volume_df, returns_df = d["close"], d["volume"], d["returns"]

d_f = qs.download(["SPY", "XLK"], start="2015-01-01", end="2023-12-31")
residuals, _, _ = qs.residualize(returns_df, d_f["returns"])

signal = -residuals.rolling(5).mean().shift(1)
signal = signal.sub(signal.mean(axis=1), axis=0)        # cross-sectional demean
signal = qs.vol_filter(signal, residuals, vol_window=5, quantile=0.6)
signal = qs.volume_zscore_filter(signal, volume_df, window=30, min_zscore_quantile=0.8)

liq_mask = qs.liquidity_filter(close_df, volume_df, top_n=250)
signal = signal.where(liq_mask)

positions = qs.build_long_short_positions(signal, n_long=25, n_short=25)
port_ret = qs.run(positions, returns_df.where(liq_mask))

print(qs.metrics.summary(port_ret, positions))
qs.summary_plot(port_ret)
```

---

## Constants

```python
from qstudy.constants import SP500        # ~500 S&P 500 tickers
from qstudy.constants import SECTOR_ETFS  # 11 SPDR sector ETFs
from qstudy.constants import MAJOR_INDEXES  # SPY, QQQ, DIA, IWM
```
