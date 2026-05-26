# Old Code Index

This document tracks code that was inherited from earlier experiments or adjacent projects and
is not part of the active `qstudy` library surface.

## Removed from `src/`

These paths were removed because they are not imported by `src/qstudy/`, are not covered by the
current `qstudy` tests, and in several cases still reference packages that do not exist in this
repository.

### Legacy charting stack

- `src/charts/`
  - Older options and gamma-exposure chart modules.
  - Depends on top-level `src/utils/` helpers instead of `qstudy.charts`.
  - Not packaged or re-exported by `qstudy`.

### Legacy helpers and indicators

- `src/utils/`
  - Black-Scholes, GEX, intraday, and volume helpers used only by the removed legacy charts.
- `src/indicators/`
  - Standalone indicator code not used by `qstudy`.
- `src/config.py`
  - Legacy top-level config module outside the package namespace.
- `src/__init__.py`
  - Empty top-level package marker for the old layout.

### Legacy scripts

- `src/scripts/`
  - One-off analysis and debugging scripts.
  - Some files still referenced missing modules such as `src.trade_lab.*` and `src.qc_utils.*`.
  - Not part of the public library API or CLI package.

### Stale tests

- `tests/test_regime_detection.py`
  - Referenced `src.trade_lab.utils.gex`, which is not part of this repository or package.

## Follow-up Notes

- Salvageable ideas from the removed code are tracked in [FUTURE_DEVELOPMENT.md](FUTURE_DEVELOPMENT.md).
- Legacy docs and old scripts that were only retained as historical baggage
  should be deleted rather than preserved indefinitely.
