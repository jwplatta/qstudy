# qstudy

`qstudy` is a Python library for fast iteration on cross-sectional equity research.
It gives you a compact pipeline for:

- downloading and caching market data
- building signals and eligibility filters
- constructing long/short or long-only portfolios
- combining multiple strategies into a portfolio-level backtest
- running lag-aware backtests
- summarizing results and parameter sweeps

Created by Joseph Platta (`jwplatta@gmail.com`).

## Requirements

- Python `3.10+`
- `uv` for environment and dependency management
- internet access for first-time market data downloads via `yfinance`

## Installation

### From source for development

```bash
git clone https://github.com/jwplatta/qstudy.git
cd qstudy
uv sync
```

### Install from GitHub

```bash
uv pip install "git+https://github.com/jwplatta/qstudy.git"
```

## Quickstart

```python
import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

universe = qs.download(SP500, start="2018-01-01", end="2024-12-31")
benchmark = qs.download(["SPY"], start="2018-01-01", end="2024-12-31")

study = (
    Study(universe=universe, benchmark=benchmark, name="mr_5d")
    .mean_reversion(window=5)
    .add_liquidity_filter(top_n=250)
    .add_vol_filter(vol_window=40, quantile=0.75, keep="low")
    .build_long_short(n_long=25, n_short=25, rebalance_every=5)
    .run()
)

print(study.metrics_dict())
study.report()
```

## Core Workflow

The typical `qstudy` flow is:

1. Download `StudyData` for your universe, benchmark, and optional factors.
2. Define a base signal with built-ins like `.mean_reversion()` or `.momentum()`, or supply a custom function.
3. Apply filters that mark ineligible assets with `NaN`.
4. Build positions with long/short or long-only portfolio rules.
5. Run the engine, which applies a 1-day execution lag to returns.
6. Review metrics, charts, and parameter sweep results.

For multi-strategy research, `PortfolioStudy` combines several configured `Study`
pipelines into one shared portfolio run with strategy-level weighting and optional
portfolio-level leverage.

## Usage Illustration

### Functional API

```python
import qstudy as qs
from qstudy.constants import SP500

data = qs.download(SP500, start="2015-01-01", end="2024-12-31")
signal = -data.returns.rolling(5).mean().shift(1)
signal = qs.vol_filter(signal, data.returns, vol_window=20, quantile=0.7, keep="low")

liq_mask = qs.liquidity_filter(data.close, data.volume, top_n=250, window=60)
positions = qs.build_long_short_positions(signal.where(liq_mask), n_long=25, n_short=25)
positions = qs.rebalance(positions, every=5)

portfolio_returns = qs.run(positions, data.returns.where(liq_mask))
print(qs.metrics.summary(portfolio_returns, positions))
```

### Study API

```python
import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

universe = qs.download(SP500, start="2015-01-01", end="2024-12-31")
benchmark = qs.download(["SPY"], start="2015-01-01", end="2024-12-31")
factors = qs.download(["SPY", "XLK", "XLF"], start="2015-01-01", end="2024-12-31")

study = (
    Study(universe=universe, benchmark=benchmark, factors=factors, name="residual_mr")
    .residualize_returns()
    .mean_reversion(window=5)
    .add_liquidity_filter(top_n=250)
    .build_long_short(n_long=25, n_short=25, rebalance_every=5)
    .run()
)

study.report()
```

### PortfolioStudy

```python
import qstudy as qs
from qstudy import PortfolioStudy, Study
from qstudy.constants import SP500

universe = qs.download(SP500, start="2015-01-01", end="2024-12-31")
benchmark = qs.download(["SPY"], start="2015-01-01", end="2024-12-31")

mr = (
    Study(name="mean_reversion")
    .mean_reversion(window=5)
    .add_liquidity_filter(top_n=250)
    .build_long_short(n_long=25, n_short=25, rebalance_every=5)
)

mom = (
    Study(name="momentum")
    .momentum(window=90)
    .add_liquidity_filter(top_n=250)
    .build_long_only(n=20, rebalance_every=5)
)

portfolio = (
    PortfolioStudy(
        strategies=[mr, mom],
        universe=universe,
        benchmark=benchmark,
        name="mr_plus_momentum",
    )
    .weight_equal()
    .fully_invest()
    .leverage(vol_target=0.12)
    .run()
)

print(portfolio.metrics_dict())
portfolio.report()
```

`PortfolioStudy` is the right abstraction when you want to:

- run multiple `Study` sleeves against the same shared universe and benchmark
- weight sleeves with `weight_equal()`, `weight_equal_vol()`, `weight_equal_sharpe()`, or `weight_optimal()`
- fully invest or rescale the combined book with `fully_invest()` and `leverage()`
- inspect cross-strategy relationships via `portfolio.strategy_returns` and `portfolio.strategy_corr`

## CLI Experiment Workflow

`qstudy` also includes an experiment scaffolding CLI:

```bash
uv run qstudy list
uv run qstudy create residual-mr
uv run qstudy iterate residual-mr volume-confirmed
uv run qstudy run residual-mr
uv run qstudy show-results residual-mr
```

Project-level configuration lives in `.qstudy.toml`:

```toml
studies_dir = "./experiments"
data_dir = "./.qstudy-data"
```

## Package Layout

- `src/qstudy/`: library package
- `src/qstudy/study/`: pipeline classes, engine, portfolio construction, metrics
- `src/qstudy/data/`: dataset loading and cache-aware downloads
- `src/qstudy/signals/`: factors, filters, and transforms
- `src/qstudy/experiments/`: CLI experiment scaffolding and results tooling
- `tests/`: pytest suite
- `docs/`: examples, workflow notes, and cleanup indexes

## Documentation

- [Study quickstart](docs/STUDY_QUICKSTART.md)
- [Combined portfolio example](docs/combined_portfolio_study.py)

## Development

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run mypy src/
uv run pytest
```

Contribution workflow is documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
