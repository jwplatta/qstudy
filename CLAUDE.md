# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install/sync dependencies
uv sync

# Run the dashboard
uv run streamlit run src/app.py

# Run tests
uv run pytest                    # all tests
uv run pytest tests/test_app.py  # single file
uv run pytest -k test_bs_gamma   # single test by name

# Code quality
uv run ruff format .             # format code
uv run ruff check .              # lint
uv run ruff check --fix .        # auto-fix lint issues
uv run mypy src/                 # type checking
```

## Architecture

This is a Streamlit-based options trading dashboard for visualizing gamma exposure (GEX) and related analytics. Data is read from a directory specified by the `DATA_DEV` environment variable.

### Source Layout

- `src/app.py` - Streamlit entry point and UI logic
- `src/config.py` - Configuration constants (DATA_DIR, DEFAULT_SYMBOL, trading session times, GEX parameters)
- `src/utils/` - Calculation utilities
  - `black_scholes.py` - Black-Scholes gamma calculation
  - `gex.py` - Gamma exposure (GEX) calculations with dealer sign conventions
- `src/charts/` - Chart modules (to be populated from trade_lab)

### Key Concepts

- **GEX (Gamma Exposure)**: Measures dealer hedging pressure. Uses `row_gross_gex()` for unsigned values and `apply_dealer_sign()` for directional exposure.
- **Default symbol**: SPXW (S&P 500 weekly options)
- **Trading parameters**: Defined in `config.py` (STRIKE_WIDTH=50, MULTIPLIER=100, GAMMA_SCALE=0.01)

## Code Style

- Python 3.9+, ruff for formatting/linting (line length 100, double quotes)
- snake_case for functions/modules, PascalCase for classes
- Tests in `tests/` following `test_*.py` naming, use pytest

## Multi-Agent Workflow

When working in parallel with other agents, use git worktrees:
- One agent = one worktree = one branch = one PR
- Branch prefixes: `feature/*`, `fix/*`, `chore/*`
- Always run `uv run ruff format . && uv run ruff check --fix . && uv run pytest` before committing
