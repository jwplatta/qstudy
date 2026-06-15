# SP500 Membership Debugging Notes

## Summary

After the Tickrake-backed SP500 membership work landed, the script at `tmp/multi-sleeve-us-equities-stat-arb.py` failed on the current `main` branch.

The failure was not caused by the removal of the old `SP500` constant itself. The hard failure came from the new membership-aware universe producing fully masked return columns for some historical constituents, which then broke time-series residualization.

## Relevant Commits

- `ef9b7f6` `Replace yfinance loader with Tickrake-backed data access (#10)`
- `8f69961` `Remove stale SP500 constant references (#11)`

Interpretation:

- `#10` changed `qs.download()` so `index_code="SP500"` resolves a historical universe and applies a membership mask.
- `#11` removed the stale hardcoded `SP500` constant and updated docs/examples to use `index_code="SP500"`.

## README State

`src/qstudy/README.md` already reflects the intended new behavior:

- use `qs.download(index_code="SP500", start=..., end=...)`
- expect `StudyData.returns` to contain `NaN` outside each ticker's valid membership dates
- treat `NaN` as excluded rather than forcing zeroes

That documentation direction was correct. The runtime bug was that residualization still assumed every ticker would have at least one usable regression row.

## Reproduction

Run:

```bash
uv run python tmp/multi-sleeve-us-equities-stat-arb.py
```

Observed failure on `main`:

```text
ValueError: zero-size array to reduction operation maximum which has no identity
```

Failure path:

1. `tmp/multi-sleeve-us-equities-stat-arb.py`
2. `Study._run_residualize()`
3. `qstudy.signals.factors.residualize()`
4. `statsmodels.OLS(...)`

The crash happened while running the second sleeve:

- `Active-Return MR (z/60, r10; resid-disp gate)`

## Root Cause

The new historical SP500 universe included tickers whose membership dates existed in the index table, but whose locally available Tickrake candle history did not overlap those membership dates.

The concrete tickers found during debugging were:

- `ADT`
- `DNR`
- `DO`

What this meant in practice:

- `membership_mask` was `True` for part of the requested date range
- `close` and `returns` were completely `NaN` after the loader applied the mask
- `residualize()` did `y = r[ticker].dropna()`
- for those tickers, `y` was empty
- `statsmodels.OLS(y, x)` was called with a zero-row design matrix
- statsmodels raised before fitting

This is an edge case introduced by the shift from a survivorship-biased static ticker list to a historical membership-aware universe.

## Data Observations

The failing run used:

- universe shape: `2301 x 681`
- factor shape: `2264 x 11`
- common index length: `2264`

There were:

- `3` tickers with no non-`NaN` returns on the factor-overlap index
- `37` universe dates not present in the factor dataset

The empty-regression tickers were exactly:

- `ADT`
- `DNR`
- `DO`

All three had ticker history files on disk, but the available candle history did not overlap the historical membership windows being requested. That strongly suggests a historical identity / ticker-reuse mismatch rather than a simple missing-file problem.

## Fix

The fix was implemented in `src/qstudy/signals/factors.py`.

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

Behavior after the fix:

- rows with missing factor inputs are dropped before regression
- tickers with zero usable rows are skipped
- their residual column remains all `NaN`
- no exception is raised

## Test Coverage

Added regression coverage in `tests/test_qstudy.py`:

- `test_residualize_skips_tickers_with_no_usable_factor_overlap`

The test verifies:

- normal tickers still produce residuals
- a fully unusable ticker stays all `NaN`
- skipped tickers are absent from parameter and `rsquared` outputs

## Validation

Focused test run:

```bash
uv run pytest tests/test_qstudy.py -k 'residualize_skips_tickers_with_no_usable_factor_overlap or pipeline_matches_manual_portfolio_returns'
```

Result:

- passed

Script rerun after the fix:

```bash
uv run python /Users/jplatta/repos/qstudy/tmp/multi-sleeve-us-equities-stat-arb.py
```

Result:

- completed successfully
- previous empty-OLS crash no longer occurred

## Remaining Issue

Several sleeves still produce `NaN` Sharpe after the crash fix.

Examples observed during the successful rerun:

- `Residual MR (factor model 5d, r10; resid-disp gate)`
- `Dist Pairs MR (k=3, z60, r10; always-on)`
- several DPMR sleeves in the portfolio buildout

This appears to be a separate data/signal sparsity issue, not another hard failure in the loader or residualizer.

Most likely next debugging targets:

1. inspect whether those sleeves generate all-zero or all-`NaN` positions
2. inspect whether their upstream signals are eliminated by tradeability or conditioning filters under the membership-aware universe
3. inspect whether distance-pair partner construction is too sparse once historical membership masking is applied
4. inspect whether the cross-sectional factor-model sleeve is failing minimum-stock thresholds often enough to suppress residuals

## Practical Takeaway

The SP500 membership migration exposed a real assumption gap:

- historical membership masking can legitimately create tickers with zero usable sample after alignment
- time-series residualization must tolerate that case

The loader change was directionally correct. The residualization path needed to be made robust to sparse historical universes.
