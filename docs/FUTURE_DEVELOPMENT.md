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
