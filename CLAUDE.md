# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills

Use the `skillex` skill to find, install, and update skills before starting tasks. If a skill exists for the work at hand, install and invoke it rather than implementing from scratch.

```bash
skillex list                                  # list available skills
skillex pull <skill-name> --agent claude      # install a skill
skillex update <skill-name> --agent claude    # update an installed skill
```

## Contributing workflow

**Never commit directly to main.** All changes go through a worktree + PR:

```bash
git fetch origin
git worktree add ../<worktree-name> -b <branch-name> origin/main
cd ../<worktree-name>
uv sync
# ... make changes ...
uv run ruff format . && uv run ruff check --fix . && uv run pytest
git add <files>
git commit -m "verb: short description"
git push -u origin HEAD
gh pr create --base main --fill
```

Branch prefixes: `feature/*`, `fix/*`, `chore/*`

Commit style: conventional, no co-author attribution, subject line only (e.g. `fix(Study): correct __repr__`).

After PR is merged:
```bash
git worktree remove ../<worktree-name>
git branch -d <branch-name>
```

## Commands

```bash
# Install/sync dependencies
uv sync

# Run tests
uv run pytest                          # all tests
uv run pytest tests/test_metrics.py   # single file
uv run pytest -k test_sharpe           # single test by name

# Code quality
uv run ruff format .                   # format
uv run ruff check --fix .              # lint + auto-fix
uv run mypy src/                       # type check

# CLI (requires .qstudy.toml to be configured)
uv run qstudy list
uv run qstudy create <name>
uv run qstudy iterate <name> <version-suffix>
uv run qstudy run <name> [--version v1_foo]
uv run qstudy show-results <name>
uv run qstudy append <name> --version v1_foo --hypothesis "..." --analysis "..." --results '{...}'
```

## Configuration

The CLI reads `.qstudy.toml` from CWD or `~/.qstudy.toml`. Required key:

```toml
studies_dir = "./experiments"
data_dir = "./.qstudy-data"   # optional cache dir
```

## Architecture: `src/qstudy`

This is a library for quickly iterating on cross-sectional equity backtests. The public API is fully re-exported from `qstudy/__init__.py`.

### Study pipeline

`Study` (`study/Study.py`) is the core abstraction — a chainable pipeline executed in strict order by `.run()`:

1. **Optional: factor model** — `.add_factor_model()` fits `BarraLiteFactorModel` (market, sector, momentum, volatility, size)
2. **Optional: residualize** — `.residualize_returns()` runs OLS against benchmark/factors; stores `cache["residual_returns"]`
3. **Base signal** *(required)* — `.base_signal(fn)` where `fn(**cache) -> pd.DataFrame`
4. **Signal filters/transforms** — `.add_filter(fn)` / `.filter_signal(fn)` / `.transform_signal(fn)` / `.neutralize_signal()` and built-ins (`add_vol_filter`, `add_volume_zscore_filter`, `add_momentum_context_filter`, `add_vix_contango_filter`)
5. **Tradeable constraints** — `.add_tradeable_constraint(fn)` with factories `qs.liquidity()`, `qs.min_price()`, `qs.min_adv()`; ANDed into `cache["_tradeable_mask"]`; ineligible signals become NaN
6. **Position builder** *(required)* — `.build_long_short(n_long, n_short)` / `.build_long_only(n)` / `.build_positions(fn)`
7. **Position scalers** — `.rebalance(every=N)` / `.rebalance_on(trigger_fn)` / `.neutralize_positions()` / `.scale_risk()`; canonical order: `weight → scale_risk → neutralize → rebalance` (enforced at run time)
8. **Weighting** — `.weight_equal()` (default) / `.weight_equal_vol()` / `.weight_equal_sharpe()` / `.weight_optimal()`
9. **Engine** — `engine.run(positions, returns)` applies 1-day execution lag: `pnl = positions.shift(1) * returns`
10. **Metrics** — `metrics.summary()` → `StudyMetrics` (Sharpe, ann_return, ann_vol, max_drawdown, IR, turnover, etc.)

**Critical conventions:**
- Signal filters use **NaN** (not 0.0) for ineligible assets; position builders rank over NaN columns
- Cache is passed as a **shallow copy** to all callables — mutations don't persist; reassigning keys is a no-op
- `save()` / `from_cache()` persists the data cache only (DataFrames + scalars), not the pipeline definition

### Data layer

`download(tickers, start, end)` (`data/loader.py`) fetches from yfinance in 100-ticker chunks, disk-caches by SHA256 hash of the request, and returns `StudyData` (aligned DataFrames for `open/high/low/close/volume/returns/log_returns`). Multiple `StudyData` objects share date intersection via `Study._inject_data()`.

### Multi-strategy composition

`PortfolioStudy` (`study/PortfolioStudy.py`) accepts a list of `Study` objects with `universe=None`; it injects shared universe/benchmark before running each, then combines positions weighted by portfolio-level weighting.

### Parameter sweeps

`param_grid(param_dict, backtest_fn)` (`study/grid.py`) sweeps all combinations, returns a DataFrame of params + metrics.

### Experiment management (CLI + `experiments/`)

Each experiment lives under `studies_dir/<name>/` with version files `v0.py`, `v1_<suffix>.py`, etc. Every version file must define `run_study() -> dict` returning metrics. The CLI automates scaffolding (`create`), branching (`iterate --parent`), running (`run`), and annotated logging (`append` writes to `log.json` with hypothesis, analysis, metrics).

### Diagnostics

`study.audit()` — after `.run()`, returns a DataFrame showing candidate count and weight normalization at each pipeline step. Useful for debugging filter logic or verifying dollar-neutrality through scalers.
