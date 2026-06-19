# S&P 500 Membership Mask: How It Works in qstudy

## Overview

When you load an S&P 500 universe with `qs.download(index_code="SP500", ...)`,
qstudy builds a survivorship-bias-free universe. It fetches every ticker that was
ever a member during the requested date range, and then applies a per-day boolean
mask so that on any given day only current index members are visible to the pipeline.
Tickers are present in the DataFrame but masked to NaN when they are not members.

---

## Step 1: Universe construction — `index_range_tickers`

`loader.py:223` — `index_range_tickers(index_code, start, end, sqlite_path)`

This queries the Tickrake SQLite database for the **union** of all tickers that
were members at any point during `[start, end]`:

```sql
SELECT DISTINCT tickers.ticker
FROM market_index_memberships memberships
...
WHERE indexes.code = ?
  AND memberships.start_date <= ?          -- membership started before end
  AND (memberships.end_date IS NULL        -- still a member, OR
       OR memberships.end_date >= ?)       -- membership ended after start
```

For a 2015–2023 window this typically returns ~680 tickers — far more than the
~500 in the index on any single day, because the index turns over ~20–30 names
per year. OHLCV data is then fetched for all of them.

---

## Step 2: Building the per-day membership mask — `index_membership_mask`

`loader.py:252` — `index_membership_mask(tickers, index_code, start, end, ...)`

This queries the same membership table again, but now returns the exact
`(ticker, start_date, end_date)` intervals for each ticker. It builds a
`(dates × tickers)` boolean DataFrame, initialised to `False`, and sets each
ticker's rows to `True` only for the dates it was actually in the index:

```python
for ticker, membership_start, membership_end in rows:
    start_ts = pd.Timestamp(membership_start)
    end_ts = pd.Timestamp(membership_end) if membership_end else mask.index[-1]
    active = (mask.index >= start_ts) & (mask.index <= end_ts)
    mask.loc[active, ticker] = True         # loader.py:296
```

Result: a DataFrame like

```
            AAPL   GOOG   XYZ (dropped 2017)   ABC (joined 2018)
2015-01-02  True   True   True                 False
...
2017-06-01  True   True   False                False
2018-03-01  True   True   False                True
```

---

## Step 3: Applying the mask to OHLCV — `StudyData.__post_init__`

`loader.py:71` — inside `StudyData.__post_init__`:

```python
for field_name in ("open", "high", "low", "close", "volume"):
    setattr(self, field_name, getattr(self, field_name).where(mask))

self.returns = self.close.pct_change(fill_method=None).where(mask)
self.log_returns = np.log(self.close / self.close.shift(1)).where(mask)
```

`DataFrame.where(mask)` keeps values where `mask` is `True` and replaces
them with `NaN` where `mask` is `False`. After this step:

- `universe.close["XYZ"]` is NaN on every date XYZ was not in the index
- `universe.returns["XYZ"]` is NaN on those same dates
- `universe.volume["XYZ"]` is NaN on those same dates

The column **exists** in every DataFrame — it is never dropped — but it carries
NaN wherever the ticker was not a member.

---

## Step 4: How the liquidity filter sees this — `liquidity_filter`

`study/portfolio.py:89`:

```python
dollar_vol = close * volume
avg_dollar_vol = dollar_vol.rolling(window).mean()
rank = avg_dollar_vol.rank(axis=1, ascending=False, na_option="keep")
return rank <= top_n
```

Because `close` and `volume` are already membership-masked (NaN for non-members),
`close * volume` is also NaN for any ticker not in the index on that day.
`rolling().mean()` propagates NaN — a ticker accumulates a valid rolling mean
only on days where it has had `window` consecutive days of membership data.
`rank(..., na_option="keep")` leaves NaN-valued tickers unranked, so they can
never satisfy `rank <= top_n`. The liquidity mask is therefore always restricted
to current S&P 500 members with sufficient recent trading history.

### Why the old `dropna(axis=1)` was wrong

The previous implementation was:

```python
dollar_vol = (close * volume).dropna(axis=1)   # BUG
```

`dropna(axis=1)` drops any column with *any* NaN across the **entire** date
range of the loaded DataFrame. Since every ticker is outside the index for some
portion of a multi-year window, every column has at least one NaN somewhere —
so `dropna(axis=1)` silently discarded all ~680 columns and returned an empty
DataFrame. The subsequent `.reindex(columns=signal.columns).fillna(False)` in
`Study.apply_constraint` (Study.py:559) then marked all tickers as non-tradeable.

In practice the Study pipeline passed a shorter date-intersection window to
`liquidity_filter` (trimmed by the benchmark/factors date index), so roughly
327 of 680 tickers survived `dropna` — but those were specifically the tickers
with no gaps at all in the intersection window, which is a biased subset. On a
typical day in 2019, **164 tickers differed in tradeable status** between the
old and corrected implementations.

---

## How the mask flows through the rest of the Study pipeline

Because `cache["returns"]` carries NaN for non-members, every downstream
computation inherits the membership constraint automatically:

- **Residualization** (`_run_residualize`): OLS is fit using `y = r[ticker].dropna()`,
  so non-member dates are excluded from beta estimation per ticker.
- **Signal generation** (e.g. `mean_reversion`): operates on `_active_returns`
  which is `residual_returns` or `returns` — NaN rows produce NaN signals,
  which the position builder treats as ineligible.
- **Tradeable constraint** (`apply_constraint`, Study.py:553): liquidity mask is
  ANDed into `_tradeable_mask`; signal is set to NaN for excluded tickers via
  `signal.where(combined)`.
- **Backtest engine** (`engine.run`): `returns_for_engine = returns.where(combined_mask)`
  ensures non-member returns are excluded from PnL even if a position somehow
  leaked through.

The result is a fully point-in-time pipeline: a ticker can only contribute to
signal, positions, and PnL on days it is actually in the S&P 500.
