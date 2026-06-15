# SP500 Membership Debugging Summary

## Overview

After the Tickrake-backed SP500 membership work landed, `tmp/multi-sleeve-us-equities-stat-arb.py`
exposed three separate problems:

1. a hard crash in time-series residualization
2. a benchmark-shape bug that killed the factor-model sleeve
3. two signal-construction assumptions in the script that failed under sparse,
   membership-aware history

All of these were triggered by the same broader change in data semantics:

- `qs.download(index_code="SP500", ...)` now loads a historical constituent universe
- prices and returns are masked outside each ticker's valid membership dates
- the data is therefore sparse in ways the old survivorship-biased workflow never had to handle

## Scope of This PR

This pull request includes the reusable library fixes and regression tests for:

- empty OLS samples in time-series residualization
- single-column benchmark frame handling in the Barra-lite factor model
- the debugging summary in this document

The distance-sleeve and residual-dispersion filter fixes were applied locally to the test script during debugging so the full workflow could be validated, but that script is not part of the PR.

## Relevant Commits

- `ef9b7f6` `Replace yfinance loader with Tickrake-backed data access (#10)`
- `8f69961` `Remove stale SP500 constant references (#11)`

Interpretation:

- `#10` switched the loader to Tickrake-backed historical index membership data
- `#11` removed the stale hardcoded `SP500` constant and updated docs/examples to use
  `index_code="SP500"`

## README State

`src/qstudy/README.md` was already directionally correct:

- use `qs.download(index_code="SP500", start=..., end=...)`
- expect membership-aware `NaN` masking outside valid constituent dates
- treat `NaN` as excluded data, not zero returns

The bugs were not in the README itself. They were in downstream code that still implicitly assumed a dense, static universe.

## Bug 1: Empty OLS Sample in `residualize()`

### Symptom

The script crashed in:

1. `tmp/multi-sleeve-us-equities-stat-arb.py`
2. `Study._run_residualize()`
3. `qstudy.signals.factors.residualize()`
4. `statsmodels.OLS(...)`

### Root Cause

Historical SP500 membership plus local candle availability produced tickers that were valid index members on paper but had no usable price history inside those membership windows.

The concrete tickers found during debugging were:

- `ADT`
- `DNR`
- `DO`

For those tickers:

- `membership_mask` was `True` for part of the date range
- `close` and `returns` were still all `NaN` after masking
- `residualize()` did `y = r[ticker].dropna()`
- `y` became empty
- `statsmodels.OLS(y, x)` received a zero-row design matrix and raised

### Local Validation Fix

`src/qstudy/signals/factors.py` now joins returns and factors first, drops missing rows, and skips tickers with no usable regression sample.

Old logic:

```python
y = r[ticker].dropna()
x = f.loc[y.index]
model = sm.OLS(y, x).fit()
```

New logic:

```python
y = r[ticker].dropna()
regression_frame = pd.concat([y.rename("returns"), f.loc[y.index]], axis=1).dropna()
if regression_frame.empty:
    continue
y = regression_frame["returns"]
x = regression_frame.drop(columns="returns")
model = sm.OLS(y, x).fit()
```

### Result from Local Validation

- no crash
- unusable tickers stay all-`NaN` in residual output
- the script proceeds normally

## Bug 2: Benchmark Shape Bug in `BarraLiteFactorModel.fit()`

### Symptom

After fixing the empty-OLS crash, the factor-model residual sleeve still produced:

- all-`NaN` residuals
- zero active position days
- `NaN` Sharpe

### Root Cause

`BarraLiteFactorModel.fit()` assumed `benchmark_returns` was a `Series`, but the stat-arb script passed `load_benchmark().returns`, which is a single-column `(dates x 1)` `DataFrame`.

That broke the market-beta exposure construction:

- the benchmark reindexing still returned a frame
- the rolling covariance / variance math misaligned
- `self.factor_exposures_["market"]` became all `NaN`
- cross-sectional residualization never had valid continuous factor exposures

### Local Validation Fix

`src/qstudy/signals/factors.py` now accepts a single-column benchmark frame by squeezing it to a `Series` first:

```python
bench = benchmark_returns.squeeze().reindex(returns.index).fillna(0.0)
```

### Result from Local Validation

After the fix:

- market exposures became populated
- factor-model residuals became populated
- the factor-model sleeve advanced past residualization

## Bug 3: Distance Sleeves Normalized from an All-`NaN` First Row

### Symptom

All distance-pair sleeves were dead:

- zero signals
- zero positions
- zero returns
- `NaN` Sharpe

### Root Cause

The script normalized pair prices with:

```python
price = (1 + r).cumprod()
norm = price / price.iloc[0]
```

Under the membership-aware universe, the first return row was all `NaN`, so:

- `price.iloc[0]` was all `NaN`
- `norm` became all `NaN`
- pair spreads became all `NaN`
- the distance sleeves never produced a tradable signal

### Fix

The script now normalizes each ticker by its first valid price instead of the first calendar row:

```python
def normalize_from_first_valid(frame: pd.DataFrame) -> pd.DataFrame:
    first_valid = frame.bfill().iloc[0].replace(0.0, np.nan)
    return frame.div(first_valid, axis=1)
```

and:

```python
norm = normalize_from_first_valid(price)
```

### Result

Distance sleeves now produce live signals, live positions, and finite Sharpe values.

## Bug 4: Residual-Dispersion Regime Filter Assumed Dense Calendar History

### Symptom

After fixing the benchmark-shape bug, the factor-model residual sleeve still stayed dead because the conditioning filter removed every signal row:

- base signal populated
- `residual_dispersion_high_20_q75` produced zero eligible rows
- zero positions
- zero return variance
- `NaN` Sharpe

### Root Cause

The script used:

```python
disp = resid.std(axis=1).rolling(20).mean()
thresh = disp.rolling(252).quantile(0.75)
```

That logic implicitly assumes dense daily residual history.

Under the membership-aware factor-model output:

- residuals only existed on a sparse subset of the calendar
- `disp` had many gaps
- `rolling(252)` on the full calendar almost never had 252 usable observations
- the threshold was almost always `NaN`
- the gate never turned on

Observed during debugging:

- `resid_dates_with_any = 766`
- `disp_non_na = 625`
- `thresh_non_na = 9`
- `mask_true = 0`

### Fix

The filter now computes the regime series on valid dispersion observations only, while keeping the original 20-observation smoothing and 252-observation threshold intent:

```python
disp = resid.std(axis=1).dropna()
disp = disp.rolling(20, min_periods=20).mean()
thresh = disp.rolling(252, min_periods=252).quantile(0.75)
mask = disp.gt(thresh).reindex(signal.index).fillna(False)
```

### Result

The factor-model residual sleeve now trades and produces finite performance metrics.

Example from the repaired sleeve:

- Sharpe: about `0.42`
- active position days: `1794`

## Test Coverage

Added or updated regression coverage in `tests/test_qstudy.py`:

- `test_residualize_skips_tickers_with_no_usable_factor_overlap`
- `test_fit_accepts_single_column_benchmark_dataframe`

These tests cover:

- skipping empty time-series OLS samples safely
- accepting a single-column benchmark frame in the Barra-lite factor model

## Validation

Focused tests:

```bash
uv run pytest tests/test_qstudy.py -k 'residualize_skips_tickers_with_no_usable_factor_overlap or pipeline_matches_manual_portfolio_returns'
uv run pytest tests/test_qstudy.py -k 'single_column_benchmark_dataframe or residualize_skips_tickers_with_no_usable_factor_overlap'
```

Script validation:

```bash
uv run python tmp/multi-sleeve-us-equities-stat-arb.py
```

Observed outcomes after the fixes:

- the script no longer crashes
- the factor-model residual sleeve reports finite Sharpe
- the distance sleeves report finite Sharpe
- the portfolio buildout no longer starts with a dead first sleeve

## Final Takeaway

The SP500 membership migration was correct, but it surfaced several hidden assumptions:

- time-series residualization assumed every ticker had at least one usable sample
- the Barra-lite factor model assumed the benchmark was already a `Series`
- pair normalization assumed the first calendar row was valid
- a regime filter assumed dense calendar history instead of sparse valid observations

All of those assumptions were safe in the old static-universe workflow. They were not safe once qstudy started operating on a historically accurate, membership-aware SP500 universe.
