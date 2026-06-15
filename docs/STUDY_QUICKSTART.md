# Study Pipeline Quickstart

The `Study` class provides a chainable pipeline for running cross-sectional equity backtests.
Every method returns `self`, so the full pipeline can be expressed as a single fluent expression.

## Setup: Download Data First

`qs.download()` returns a `StudyData` object. Download all the tickers you need once
and pass the `StudyData` objects to `Study` — no downloading happens inside the pipeline.

```python
import qstudy as qs
from qstudy import Study

universe_data  = qs.download(index_code="SP500", start="2018-01-01", end="2024-12-31")
benchmark_data = qs.download(["SPY"],      "2018-01-01", "2024-12-31")
factors_data   = qs.download(["SPY","QQQ"],"2018-01-01", "2024-12-31")
```

`StudyData` has fields: `tickers`, `close`, `volume`, `returns`, `log_returns`.

## Basic Pipeline Shape

```
Study(universe, benchmark=..., factors=...)
  [.residualize_returns()]          # optional
  .mean_reversion() | .momentum() | .base_signal(fn)   # required
  [.add_liquidity_filter()]         # optional, any number
  [.add_filter(fn)]                 # optional, any number
  .build_long_short() | .build_long_only()              # required
  [.scale_returns(fn)]              # optional, any number
  [.weight_equal_vol() | ...]       # optional
  .run()
  .report()
```

---

## Example 1: Mean Reversion with Built-in Filters

The simplest case — everything uses built-in methods, no custom functions needed.

```python
import qstudy as qs
from qstudy import Study

universe_data  = qs.download(index_code="SP500", start="2018-01-01", end="2024-12-31")
benchmark_data = qs.download(["SPY"], "2018-01-01", "2024-12-31")

study = (
    Study(universe=universe_data, benchmark=benchmark_data)
    .mean_reversion(window=5)
    .add_liquidity_filter(top_n=300)
    .add_vol_filter(vol_window=40, quantile=0.75, keep="low")
    .build_long_short(n_long=25, n_short=25, rebalance_every=5)
    .run()
)

study.report()
study.to_csv("mean_reversion_returns.csv")
study.save("mean_reversion.pkl")

# Reload later without re-running
study2 = Study.from_cache("mean_reversion.pkl")
study2.report()
```

### What each step does

| Method | Effect |
|--------|--------|
| `mean_reversion(window=5)` | Signal = negative 5-day rolling mean return. Recent losers score highest. |
| `add_liquidity_filter(top_n=300)` | Exclude tickers outside the 300 most liquid by rolling dollar volume by setting their signal to `NaN`. |
| `add_vol_filter(quantile=0.75, keep="low")` | Exclude tickers with realized vol above the 75th cross-sectional percentile by setting their signal to `NaN`. |
| `build_long_short(n_long=25, n_short=25)` | Top 25 signal → long (+1), bottom 25 → short (−1), dollar-neutral, rebalanced weekly. |

---

## Example 2: Residualized Momentum with Custom Functions

This example shows how to plug in custom signal filters and position scalers.

```python
import numpy as np
import qstudy as qs
from qstudy import Study

universe_data  = qs.download(index_code="SP500", start="2015-01-01", end="2024-12-31")
benchmark_data = qs.download(["SPY"],         "2015-01-01", "2024-12-31")
factors_data   = qs.download(["SPY", "QQQ"],  "2015-01-01", "2024-12-31")


# --- Custom signal filter ---
# Receives (signal, **cache) and must return a signal DataFrame of the same shape.
# Use cache to access any intermediate data — returns, residual_returns, close, volume, etc.
# Do NOT reassign cache keys or mutate DataFrames in place; the cache is a shallow copy.

def regime_filter(signal, **cache):
    """Exclude signal on days where cross-sectional return dispersion is low.

    Low dispersion means there is little spread between winners and losers,
    so a long/short strategy is unlikely to add value that day.
    """
    returns = cache["returns"]
    # Cross-sectional std of daily returns, rolling 10 days
    dispersion = returns.rolling(10).std().mean(axis=1)
    # Only trade on days where dispersion is above its own median
    high_dispersion_days = dispersion > dispersion.rolling(60).median()
    return signal.where(high_dispersion_days, other=float("nan"))


# --- Custom position scaler ---
# Receives (positions, **cache) and must return a positions DataFrame of the same shape.
# cache["returns"] always reflects the most recent state. cache["residual_returns"] is
# available if residualize_returns() was called earlier in the pipeline.

def vol_target_scale(positions, **cache):
    """Scale the portfolio to target 10% annualized volatility.

    Computes the rolling realized vol of the equal-weight portfolio and
    scales positions up or down to hit the vol target.
    """
    returns = cache["returns"]
    # Approximate portfolio return using current positions (lagged 1 day)
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    # 20-day rolling realized vol, annualized
    realized_vol = port_ret.rolling(20).std() * np.sqrt(252)
    target_vol = 0.10
    # Scale factor: shrink when vol is high, grow when vol is low, cap at 2x
    scale = (target_vol / realized_vol.clip(lower=0.01)).clip(upper=2.0)
    return positions.mul(scale.shift(1), axis=0).fillna(positions)


study = (
    Study(
        universe=universe_data,
        benchmark=benchmark_data,
        factors=factors_data,      # residualize against SPY + QQQ
        name="residualized_momentum",
    )
    .residualize_returns()         # strips out SPY + QQQ exposure before signal
    .momentum(window=90)           # 90-day rolling mean of residual returns
    .add_liquidity_filter(top_n=250)
    .add_filter(regime_filter)     # custom filter: only trade on high-dispersion days
    .build_long_short(n_long=30, n_short=30, rebalance_every=5)
    .scale_returns(vol_target_scale)  # custom scaler: target 10% vol
    .run()
)

study.report()
```

---

## Custom Function Reference

### Base signal: `fn(**cache) -> pd.DataFrame`

Called once to produce the initial signal matrix (dates × tickers).

```python
def my_signal(**cache):
    # cache keys always available:
    #   "returns"         pd.DataFrame  daily returns (dates x tickers)
    #   "close"           pd.DataFrame  closing prices
    #   "volume"          pd.DataFrame  daily volume
    #   "_active_returns" pd.DataFrame  residual_returns if residualize was called, else returns
    #
    # After residualize_returns() is called:
    #   "residual_returns" pd.DataFrame  OLS residuals vs factors/benchmark

    r = cache["_active_returns"]  # always use _active_returns for signal generation
    return r.rolling(20).mean()

study.base_signal(my_signal)
```

### Signal filter: `fn(signal, **cache) -> pd.DataFrame`

Called after the base signal, in the order they were added. Must return a signal
DataFrame of the same shape — set unwanted cells to `float("nan")` to exclude them.
Note: `signal` is passed as a positional argument, not via `**cache`.

```python
def my_filter(signal, **cache):
    returns = cache["returns"]
    # ... compute some mask ...
    mask = ...  # bool DataFrame, same shape as signal
    return signal.where(mask)   # NaN where mask is False

study.add_filter(my_filter)
```

### Position scaler: `fn(positions, **cache) -> pd.DataFrame`

Called after the position builder. Must return a positions DataFrame of the same shape.
Note: `positions` is passed as a positional argument, not via `**cache`.

```python
def my_scaler(positions, **cache):
    returns = cache["returns"]
    # ... compute some scale factor ...
    scale = ...  # pd.Series indexed by date
    return positions.mul(scale.shift(1), axis=0).fillna(positions)

study.scale_returns(my_scaler)
```

### Rules for custom functions

- **Read the cache, don't write to it.** The cache is a shallow copy — reassigning
  a key (e.g. `cache["signal"] = x`) only affects the local copy, not the Study.
- **Don't mutate DataFrames in place.** Since it's a shallow copy, the underlying
  DataFrame objects are shared. Use `.copy()` if you need to modify values.
- **Always return a DataFrame of the same shape** as the input `signal` or `positions`.
- Use `float("nan")` (not `0`) to mark ineligible cells in signal filters — zeros are
  valid signal values and will be ranked and traded.
- `signal` and `positions` are passed **positionally** — do not name them differently
  in your function signature (or use `**kwargs` to absorb everything).

---

## Cache Keys Reference

These are available in `**cache` for all custom functions:

| Key | Type | Description |
|-----|------|-------------|
| `"returns"` | `pd.DataFrame` | Daily returns (dates × tickers) |
| `"close"` | `pd.DataFrame` | Adjusted close prices |
| `"volume"` | `pd.DataFrame` | Daily volume |
| `"log_returns"` | `pd.DataFrame` | Log returns |
| `"benchmark"` | `pd.Series` | Benchmark daily returns (if provided) |
| `"factor_returns"` | `pd.DataFrame` | Factor returns (if factors provided) |
| `"_active_returns"` | `pd.DataFrame` | `residual_returns` if residualized, else `returns` |
| `"residual_returns"` | `pd.DataFrame` | OLS residuals vs factors/benchmark (after `residualize_returns()`) |
| `"base_signal"` | `pd.DataFrame` | The raw signal before any filters |
| `"signal"` | `pd.DataFrame` | Current signal state (updated after each filter) |
| `"positions"` | `pd.DataFrame` | Current positions (available in `scale_returns`) |
| `"_signal_history"` | `list` | `[(label, signal_df), ...]` snapshot after each filter step |
| `"_position_history"` | `list` | `[(label, positions_df), ...]` snapshot after each position step |

After `.run()`, these are also populated:

| Key | Type | Description |
|-----|------|-------------|
| `"portfolio_returns"` | `pd.Series` | Daily portfolio returns |
| `"metrics_summary"` | `pd.Series` | Output of `metrics.summary()` |

Access them directly via `study.cache["portfolio_returns"]` etc.

---

## Weighting Schemes

Apply after the position builder, before `.run()`:

```python
study.weight_equal()                        # default, no-op
study.weight_equal_vol(vol_window=60)       # inverse realized vol
study.weight_equal_sharpe(window=126)       # proportional to rolling Sharpe
study.weight_optimal(window=126, gamma=1.0) # mean-variance (slow for large universes)
```

## Save and Reload

```python
# Save the cache (returns, positions, metrics) to disk
study.save("my_study.pkl")

# Reload — no re-downloading or re-running needed
study2 = Study.from_cache("my_study.pkl")
study2.report()
study2.to_csv("returns.csv")
print(study2.cache["metrics_summary"])
```
