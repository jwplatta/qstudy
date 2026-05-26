# Future Development

This document captures ideas from removed legacy code that may still be worth
rewriting into `qstudy` later. These are not active parts of the current
library, but they contain concepts that could become real features if they are
rebuilt against the current package architecture.

## High-Value Candidates

### Regression-test fixtures from the long/short bug demo

Source removed:

- `src/scripts/long_short_bug_demo.py`

Why it is useful:

- It is a compact, readable reproduction of the historical NaN-universe bug in
  `build_long_short_positions`.
- The hard-coded price and signal setup is a good template for regression
  tests around liquidity masks, rank cutoffs, and short-book construction.

Recommended follow-up:

- Convert the core scenario into a focused pytest case in `tests/`.
- Keep it as test data, not as a standalone script.

Exact bug-demo code worth preserving:

```python
import numpy as np
import pandas as pd

dates = pd.date_range("2024-01-01", periods=6, freq="B")

prices = pd.DataFrame(
    {
        "AAPL": [100, 103, 107, 112, 118, 125],
        "MSFT": [100, 101, 102, 103, 104, 105],
        "GOOG": [100, 99, 98, 97, 96, 95],
        "META": [100, 96, 91, 85, 78, 70],
        "PENNY1": [1.00, 1.01, 0.99, 1.02, 0.98, 1.00],
        "PENNY2": [0.50, 0.51, 0.49, 0.52, 0.48, 0.50],
        "PENNY3": [2.00, 2.01, 1.99, 2.02, 1.98, 2.00],
        "TINY1": [3.00, 3.01, 2.99, 3.02, 2.98, 3.00],
        "TINY2": [0.10, 0.11, 0.09, 0.12, 0.08, 0.10],
        "TINY3": [0.25, 0.26, 0.24, 0.27, 0.23, 0.25],
    },
    index=dates,
)

returns_1d = prices.pct_change(periods=1)

tradeable = pd.Series(
    {t: True for t in ["AAPL", "MSFT", "GOOG", "META"]}
    | {t: False for t in ["PENNY1", "PENNY2", "PENNY3", "TINY1", "TINY2", "TINY3"]}
)

signal_df = -returns_1d
signal_df.loc[:, ~tradeable] = np.nan

n_long = 2
n_short = 2

signal_rank = signal_df.rank(axis=1, ascending=False, na_option="bottom")

# Buggy logic: count() runs on the ranked frame after NaNs have been pushed to the bottom.
# That makes n_total equal to the full column count, so the short cutoff lands on the
# non-tradeable NaN names instead of the true short candidates.
n_total_buggy = signal_rank.count(axis=1)
short_cutoff_buggy = n_total_buggy - (n_short - 1)

long_mask = signal_rank <= n_long
short_mask_buggy = signal_rank.ge(short_cutoff_buggy.values[:, None])
positions_buggy = long_mask.astype(float) - short_mask_buggy.astype(float)

# Fixed logic: count tradeable names from the original signal, then explicitly require
# signal.notna() so bottom-ranked NaN assets cannot leak into the short book.
n_tradeable_count = signal_df.count(axis=1)
short_cutoff_fixed = n_tradeable_count - (n_short - 1)

short_mask_fixed = signal_rank.ge(short_cutoff_fixed.values[:, None]) & signal_df.notna()
positions_fixed = long_mask.astype(float) - short_mask_fixed.astype(float)
abs_sum = positions_fixed.abs().sum(axis=1).replace(0, float("nan"))
positions_fixed = positions_fixed.div(abs_sum, axis=0)
```

Behavior to preserve in tests:

- The buggy version can produce an empty short book because the short cutoff is
  anchored to the full ranked column count rather than the tradeable universe.
- The fixed version must select shorts only from `signal.notna()` assets and
  must produce both longs and shorts when enough tradeable names exist.

### Gamma regime diagnostics utilities

Source removed:

- `src/utils/gex.py`

Why it is useful:

- Contains pure, coherent functions such as:
  - `calculate_flip_distance`
  - `calculate_gamma_influence`
  - `classify_regime`
- These are easier to reuse than the surrounding chart code and could support a
  future diagnostics layer.

Recommended follow-up:

- If `qstudy` expands into options microstructure research, rebuild these under
  a dedicated namespace such as `qstudy.options` or `qstudy.microstructure`.
- Add tests and detach the logic from legacy CSV and chart assumptions.

### Hedge Flow Score indicator

Source removed:

- `src/indicators/hedge_flow_score.py`

Why it is useful:

- The implementation is self-contained and conceptually clean.
- It represents a plausible reusable indicator for dealer-flow and
  gamma-regime analysis.

Recommended follow-up:

- Reintroduce only if `qstudy` intentionally expands beyond cross-sectional
  equity backtests into options-derived signals.
- Keep the indicator as pure library logic with tests, not as a chart-side
  helper.

### Intraday option-chain wrangling

Source removed:

- `src/utils/intraday.py`

Why it is useful:

- Contains practical routines for loading grouped option-chain snapshots,
  locating expirations, estimating ATM IV, and calculating zero-gamma levels.

Why it is not ready for `qstudy` as-is:

- It is tightly coupled to one specific CSV naming convention and data layout.
- It assumes an options data workflow that is outside the current `qstudy`
  package scope.

Recommended follow-up:

- If you want intraday options research later, rebuild the useful parts around
  explicit data adapters and normalized schemas.

## Not Worth Restoring Directly

These removed areas are better treated as historical baggage than as candidates
for reintroduction:

- `src/charts/`
  - mostly presentation wrappers around the removed options/gamma utilities
- most of `src/scripts/`
  - one-off analysis files, often tied to missing packages or local data
- `tests/test_regime_detection.py`
  - targeted another package namespace entirely

## Scope Guard

Right now `qstudy` is strongest as a cross-sectional equity research library.
If future options or intraday work is added, it should be introduced as a
deliberate product decision rather than by reviving legacy files wholesale.
