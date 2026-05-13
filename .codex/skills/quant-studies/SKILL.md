---
name: quant-studies
description: Design and iterate on cross-sectional equity backtests using the qstudy Study pipeline. Use when the user wants to build, explore, or improve a long/short or long-only factor study on equities.
---

# quant-studies

## Overview

`qstudy` is a library for doing unconstrainted backtesting. The `Study` class provides a chainable pipeline that handles data alignment, signal generation, filtering, position construction, PnL computation, and metrics.

The preferred workflow is now the first-party `qstudy` CLI. Use the CLI to create and manage experiment folders, then implement study logic inside the generated scaffold. Do not default to ad hoc `common.py` / `run_all.py` structures when the CLI already provides the correct experiment layout.

```python
import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500
```

---

## Workflow: Starting a New Study

When the user invokes this skill to start or iterate on a study, prefer the CLI workflow:

1. Resolve the studies root from `.qstudy.toml` or the current working directory.
2. Create the experiment with `uv run qstudy create <name>` if it does not exist yet.
3. Edit the generated `shared.py` and `v0.py` or add `v1.py`, `v2.py`, and so on.
4. Run the experiment with `python run.py` from inside the experiment directory.
5. Review results with `uv run qstudy show-results <name>`.

Ask the following questions before choosing parameters or writing study logic. Ask them all at once.

### Required questions

1. **Universe** — what assets to trade?
   - Common answers: `SP500`, `SECTOR_ETFS`, a custom list of tickers
   - Default if unspecified: `SP500`

2. **Benchmark** — what to compare against?
   - Common answers: `SPY`, `QQQ`, `None`
   - Used for benchmark metrics and (optionally) residualization
   - Default if unspecified: `SPY`

3. **Signal type** — what drives the trade?
   - Mean reversion, momentum, residual mean reversion (factor-stripped), or custom
   - If residual: ask which factors to strip (e.g. `["SPY", "XLK"]`)

4. **Date range** — start and end dates
   - Default if unspecified: `"2015-01-01"` to `"2023-12-31"`

5. **Iterations** — how many versions should the agent attempt to improve?
   - Each iteration should be a new `vN.py` module in the CLI-managed experiment directory
   - The agent improves signal, filters, or position sizing based on prior results
   - Suggested range: 2–5; more is fine but slower

6. **Experiment name** — what should the CLI create or reuse?
   - This becomes the folder name passed to `qstudy create <name>`
   - Use a short kebab-case name when possible

7. **Studies root** — should the default config be used, or should `.qstudy.toml` point somewhere specific?
   - If the user does not care, use the resolved default
   - If they want repo-local experiments, prefer a local `.qstudy.toml` with `studies_dir = "experiments"`

### CLI-first experiment layout

`qstudy create <name>` generates:

```
<studies_root>/<study-name>/
  shared.py
  v0.py
  run.py
  results.json
  results.csv
  log.md
  readme.md
```

Use this layout as the source of truth. Do not replace it with `common.py` or `run_all.py` unless the user explicitly asks for a nonstandard structure.

### Command sequence

```bash
uv run qstudy create <study-name>
cd <study-name>
python run.py
uv run qstudy show-results <study-name>
```

If the studies root is repo-local, the final command should be run from the directory containing the relevant `.qstudy.toml`.

---

## Study Pipeline Reference

### Basic shape

```python
def my_signal(**cache):
    # cache["residual_returns"] if residualize_returns() was called, else cache["returns"]
    return -cache["residual_returns"].rolling(5).mean().shift(1)

study = (
    Study(universe=universe_data, benchmark=benchmark_data, factors=factors_data)
    # [optional] strip factor exposure before signal
    .residualize_returns()
    # signal source — always .base_signal(fn); fn(**cache) -> pd.DataFrame
    .base_signal(my_signal)
    # filters — zero or more, applied in order
    .add_vol_filter(vol_window=40, quantile=0.75)
    .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
    .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
    .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
    # position builder — exactly one
    .build_positions(my_position_fn)
    .rebalance(every=1)
    # risk scalers — zero or more
    .scale_risk(my_regime_scaler)
    .run()
)

study.report()   # prints metrics + 3-panel chart
```

### Data loading

```python
# Returns a StudyData object with fields: tickers, close, volume, returns, log_returns
universe_data  = qs.download(SP500,           "2015-01-01", "2023-12-31")
benchmark_data = qs.download(["SPY"],         "2015-01-01", "2023-12-31")
factors_data   = qs.download(["SPY", "XLK"], "2015-01-01", "2023-12-31")
```

For CLI-managed experiments, put shared download helpers in `shared.py` and version-specific study logic in `vN.py`.

### Built-in filters

| Method | What it does |
|--------|-------------|
| `.add_liquidity_filter(top_n=250)` | Keep only top N by rolling dollar volume; also masks returns in the engine |
| `.add_vol_filter(vol_window=40, quantile=0.75, keep="low")` | Remove high-vol assets cross-sectionally |
| `.add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)` | Remove assets with below-average recent volume |
| `.add_momentum_context_filter(window=60, max_abs_quantile=0.7)` | Remove strongly trending assets (for mean reversion) |

When `residualize_returns()` was called, `add_vol_filter` and `add_momentum_context_filter` use `residual_returns` instead of raw returns.

### Custom signal filter

```python
def my_filter(signal, **cache):
    # cache keys: returns, close, volume, residual_returns, _active_returns, benchmark, ...
    returns = cache["returns"]
    mask = ...   # bool DataFrame, same shape as signal
    return signal.where(mask)   # NaN to exclude, never 0

study.add_filter(my_filter)
```

### Custom position scaler

```python
def my_scaler(positions, **cache):
    returns = cache["returns"]
    liq_mask = cache.get("_liquidity_mask")
    if liq_mask is not None:
        returns = returns.where(liq_mask)
    # compute scale: pd.Series indexed by date
    scale = ...
    return positions.mul(scale.shift(1), axis=0)

study.scale_returns(my_scaler)
```

### Weighting schemes

```python
study.weight_equal()                        # default, no-op
study.weight_equal_vol(vol_window=60)       # inverse realized vol
study.weight_equal_sharpe(window=126)       # proportional to rolling Sharpe
study.weight_optimal(window=126, gamma=1.0) # mean-variance (slow for large universes)
```

### Save and reload

```python
study.save("~/.qstudy/my_study/v1.pkl")
study2 = Study.from_cache("~/.qstudy/my_study/v1.pkl")
study2.report()
```

---

## Iteration Strategy

For each iteration after the baseline, improve along **one axis at a time** so the effect is interpretable. Suggested progression:

| Iteration | Focus |
|-----------|-------|
| v1 | Baseline: raw signal + liquidity filter only |
| v2 | Add conditioning filters (vol, volume z-score, momentum context) |
| v3 | Add residualization (strip market/sector beta) |
| v4 | Add a regime scaler (equity curve, correlation spike, or VIX) |
| v5 | Tune position sizing, rebalance frequency, or weighting scheme |

After each iteration, note changes in Sharpe, ann_return, max_drawdown, and information_ratio. Only carry forward improvements.

---

## CLI Experiment Structure

Use the CLI-generated structure whenever you are running a systematic, multi-version experiment where results should be aggregated and compared in a results table.

### Directory layout

```
<studies_root>/<study-name>/
  shared.py       # shared loaders, constants, and helper functions
  v0.py           # baseline study with run_study() -> dict
  v1.py           # first iteration
  ...
  vN.py           # Nth iteration
  run.py          # discovers all top-level v*.py files and writes results.json/results.csv
  log.md          # per-version rationale and outcome notes
  readme.md       # local experiment notes
  results.json    # generated by run.py — source of truth for results
  results.csv     # generated by run.py — flat table
```

### `shared.py`

`shared.py` should hold everything every version script reuses: data loaders, constants, and helper signal/filter/scaler functions.

```python
from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
N_LONG = 25
N_SHORT = 25


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)

@cache
def load_benchmark():
    return qs.download([BENCHMARK_TICKER], START_DATE, END_DATE)

@cache
def load_baseline_factors():
    return qs.download(["SPY", "XLK"], START_DATE, END_DATE)


def mean_reversion_signal(window=5):
    def signal(**cache):
        returns = cache["_active_returns"]
        return -returns.rolling(window).mean()

    signal.__name__ = f"mean_reversion_signal_{window}"
    return signal
```

### `v0.py`

`v0.py` should define `run_study() -> dict` and return `study.metrics_dict()`.

```python
import json

from qstudy import Study

from shared import N_LONG, N_SHORT, load_benchmark, load_universe, mean_reversion_signal


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name="v0")
        .base_signal(mean_reversion_signal(window=5))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
```

### `v1.py`–`vN.py`

Each iteration should define `run_study() -> dict`. Import shared helpers from `shared.py`, change one axis at a time, and return metrics for aggregation by `run.py`.

```python
import json

import qstudy as qs
from qstudy import Study

from shared import N_LONG, N_SHORT, load_benchmark, load_universe, mean_reversion_signal


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name="v1")
        .base_signal(mean_reversion_signal(window=10))
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
```

### `run.py`

`run.py` is generated by the CLI and should usually be left alone. It discovers top-level `v*.py` files, sorts them by version number, imports each module, calls `run_study()`, and writes `results.json` plus `results.csv`.

Run with:

```bash
python run.py
```

### `log.md`

Record the per-version rationale and outcome. A simple structure is:

```
### `vN`
- Change: <one-line description of what changed vs. prior version>
- Result: Sharpe `X.XX`, max drawdown `-X.XX%`, duration `NNN`; keep / reject
```

End with a `## Summary` section noting what worked, what didn't, and the final recommended version.

---

## Key Rules

- **NaN = excluded, 0 = signal.** In filters, use `signal.where(mask)` to NaN-out ineligible assets. Never set signal to 0 — zero is a valid signal value that will be ranked and potentially traded.
- **Use the CLI layout first.** Start with `qstudy create`, edit the generated files, run `python run.py`, inspect with `qstudy show-results`.
- **One version = one `run_study()` module.** Each `vN.py` should be independently runnable through the generated `run.py`.
- **1-day execution lag** is baked into `engine.run()` via `positions.shift(1)`. Do not shift signals manually.
- **Download once, reuse** — `qs.download()` hits yfinance. Put shared loaders in `shared.py` and reuse across `vN.py` files.
- **residualize_returns() requires** either `factors=` or `benchmark=` in the `Study` constructor.

---

## Cache Keys in Custom Functions

| Key | Type | Notes |
|-----|------|-------|
| `"returns"` | `pd.DataFrame` | always available |
| `"close"` | `pd.DataFrame` | always available |
| `"volume"` | `pd.DataFrame` | always available |
| `"benchmark"` | `pd.Series` | if benchmark provided |
| `"factor_returns"` | `pd.DataFrame` | if factors provided |
| `"_active_returns"` | `pd.DataFrame` | residuals if residualized, else returns |
| `"residual_returns"` | `pd.DataFrame` | after `residualize_returns()` |
| `"signal"` | `pd.DataFrame` | current signal state (in filters) |
| `"positions"` | `pd.DataFrame` | current positions (in scalers) |
| `"_liquidity_mask"` | `pd.DataFrame` | after `add_liquidity_filter()` runs |

---

## Full API Reference

**Constants**
```python
from qstudy.constants import SP500         # ~500 S&P 500 tickers
from qstudy.constants import SECTOR_ETFS  # XLC XLY XLP XLE XLF XLV XLI XLK XLB XLRE XLU
from qstudy.constants import MAJOR_INDEXES # SPY QQQ DIA IWM
```

**Data**
- `qs.download(tickers, start, end) -> StudyData`

**Filters (functional API — use inside custom functions)**
- `qs.vol_filter(signal, returns, vol_window, quantile, keep) -> pd.DataFrame`
- `qs.volume_zscore_filter(signal, volume, window, min_zscore_quantile) -> pd.DataFrame`
- `qs.momentum_context_filter(signal, returns, window, max_abs_quantile) -> pd.DataFrame`
- `qs.vix_contango_filter(signal, vix_close, window) -> pd.DataFrame`
- `qs.residualize(returns, factor_returns) -> (residuals_df, params_df, rsquared_s)`

**Portfolio Construction**
- `qs.liquidity_filter(close, volume, top_n=250, window=60) -> pd.DataFrame`
- `qs.build_long_short_positions(signal, n_long=25, n_short=25) -> pd.DataFrame`
- `qs.build_long_only(signal, n=10) -> pd.DataFrame`
- `qs.rebalance(positions, every=5) -> pd.DataFrame`

**Backtest Engine**
- `qs.run(positions, returns) -> pd.Series` — 1-day lag applied

**Metrics**
- `qs.metrics.summary(returns, positions=None, benchmark=None) -> pd.Series`
- `qs.metrics.sharpe(returns) -> float`
- `qs.metrics.annualized_return(returns) -> float`
- `qs.metrics.max_drawdown(returns) -> float`
- `qs.metrics.information_ratio(returns, benchmark) -> float`

**Charts**
- `qs.summary_plot(returns, figsize=(14, 8))` — equity / drawdown / rolling Sharpe
- `qs.param_heatmap(results_df, row_param, col_param, metric="metric")`

**Grid Search**
- `qs.param_grid(param_dict, backtest_fn, metric_fn=None) -> pd.DataFrame`
