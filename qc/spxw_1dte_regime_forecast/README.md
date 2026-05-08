# SPXW 1DTE Regime Forecast

## Purpose

This algorithm starts from the baseline 1DTE SPXW iron condor strategy and adds one new trade filter:

- if the model estimates that the next session has a high probability of being a high-range day,
  the strategy skips the entry.

For this project, a "high-range" day means an SPX daily range of `51.698` points or more.

## What changed vs baseline

The core baseline logic is unchanged:

- enter around `3:50pm ET`
- skip entries when the next expiry lands on a known macro event date
- manage the trade with the same profit target, max loss, and 0DTE exit rules

The added logic is:

1. capture the current session's SPX cash open after the market opens
2. compute four regime features during the entry check
3. score those features with a fixed logistic model
4. skip the trade when `probability >= 0.50`

## Regime features used

The filter uses the feature set selected in `qc/spxw_1dte_baseline/research.ipynb`:

- `prior_slope`
- `5d_avg_range`
- `prior_abs_ret`
- `gap_mag`

At entry time on trading day `t`, the features are defined as:

- `prior_slope = VIX9D_close[t-1] - VIX_close[t-1]`
- `5d_avg_range = mean(SPX_range[t-5], ..., SPX_range[t-1])`
- `prior_abs_ret = abs(log(SPX_close[t-1] / SPX_close[t-2]))`
- `gap_mag = abs((SPX_open[t] - SPX_close[t-1]) / SPX_close[t-1])`

These are all observable before the `3:50pm ET` entry decision on day `t`.

## What "offline fit" means

The comment in `main.py` refers to how the logistic model coefficients were produced.

It does **not** mean the algorithm fits a model during the backtest or live run.

It means:

- the feature set came from the research in `qc/spxw_1dte_baseline/research.ipynb`
- I then re-fit a logistic regression outside the algorithm using the local daily CSV data in `research/data/`
- the target was shifted forward by one trading day so the model predicts the regime for the session the trade will expire in
- the resulting coefficients were hardcoded into `main.py`

That last step is important. The algo only does inference at runtime. It does not retrain.

## Why the target was shifted forward

The baseline research notebook evaluates whether the selected features explain the **same day's** regime.

That is useful for feature selection, but this strategy enters at the end of day `t` for exposure on day `t+1`.
So the implementation needs a forecast of the **next trading day's** regime.

To match the trade timing, the fitted target becomes:

- `high_regime_target[t] = 1 if SPX_range[t+1] >= 51.698 else 0`

The live algorithm then uses data known on day `t` to decide whether to carry 1DTE risk into day `t+1`.

## Model in the algorithm

The algo uses a fixed logistic score:

```text
logit = intercept
      + w1 * prior_slope
      + w2 * 5d_avg_range
      + w3 * prior_abs_ret
      + w4 * gap_mag

probability = 1 / (1 + exp(-logit))
```

Current hardcoded values:

- `intercept = -3.9225339071705148`
- `prior_slope = 0.15572728210666714`
- `5d_avg_range = 0.06168979627510342`
- `prior_abs_ret = 36.499736265948194`
- `gap_mag = 40.380107779126185`
- skip threshold = `0.50`

## Look-ahead bias check

I checked the implementation against the usual leak points.

### No look-ahead in live feature construction

The runtime features only use:

- yesterday and earlier completed daily bars
- today's SPX open, captured after the market opens

The helper `daily_history()` explicitly drops any row whose date equals `self.time.date()`. That prevents the algorithm from accidentally using the current session's unfinished daily high, low, or close.

### No look-ahead from the target definition

The next-day target was only used during the offline fitting step.

At runtime, the algorithm never accesses tomorrow's range or tomorrow's prices. It only evaluates today's known features and applies fixed coefficients.

### No rolling refit during the backtest

The coefficients are frozen in code. That avoids accidental training on future data during the backtest.

### Important backtest caveat

There is no per-trade look-ahead in the runtime feature calculation, but there is still an
evaluation caveat:

- the hardcoded coefficients were fit on the `2022-04-01` to `2023-12-31` sample
- the algorithm currently starts trading on `2022-04-01`

So if you run the full `2022-2025` backtest, the `2022-2023` portion is an in-sample period for
the regime model. That is not look-ahead bias, but it is not a clean out-of-sample evaluation of
the filter either.

If you want a cleaner validation, either:

- start the regime-filtered backtest on `2024-01-01`, or
- do a proper walk-forward / expanding-window retraining procedure outside the algo and export the
  coefficients for each evaluation window

## Remaining caveats

This implementation avoids direct look-ahead bias, but a few research caveats remain:

- the coefficients were fit once on the local historical sample and then frozen
- the `0.50` skip threshold is a design choice, not a separately optimized walk-forward parameter
- the feature set came from same-day regime research, then was re-aligned to next-day forecasting for the strategy

That is reasonable for a first implementation, but the stronger validation path is:

1. re-create the next-day target explicitly in the notebook
2. fit the model there
3. document the exact train/test split and metrics
4. then keep the exported coefficients in the algo

## Files

- `qc/spxw_1dte_regime_forecast/main.py` - baseline strategy plus regime filter
- `qc/spxw_1dte_regime_forecast/IronCondorFinder.py` - copied from baseline
- `qc/spxw_1dte_regime_forecast/event_dates.py` - copied from baseline
- `qc/spxw_1dte_regime_forecast/event_dates.yml` - copied from baseline

## Validation done

I validated the code path with:

```bash
python -m py_compile qc/spxw_1dte_regime_forecast/main.py \
  qc/spxw_1dte_regime_forecast/IronCondorFinder.py \
  qc/spxw_1dte_regime_forecast/event_dates.py
```

I have not yet run a full Lean backtest for this regime-filtered project.
