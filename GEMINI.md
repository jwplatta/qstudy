# GEMINI.md

This file provides instructional context about the `trade_lab` project for Gemini.

## Project Overview

`trade_lab` is a Python-based trading analysis toolkit for developing and testing financial indicators, analyzing trades, and visualizing market data. It uses Ruby for fetching data from the Schwab API. The project is structured with a `src` directory for the main Python package, `notebooks` for prototyping, and `tests` for unit testing.

The core Python libraries used are `numpy`, `pandas`, `matplotlib`, and `jupyter`. Development tools include `pytest` for testing, `ruff` for linting and formatting, and `mypy` for type checking.

## Building and Running

The project uses both Python and Ruby.

### Python Environment (uv)

- **Sync dependencies:** `uv sync`
- **Activate virtualenv:** `source .venv/bin/activate`
- **Run tests:** `uv run pytest`
- **Format code:** `uv run ruff format .`
- **Lint code:** `uv run ruff check .`
- **Type check:** `uv run mypy src/`

### Ruby Environment (bundler)

- **Install dependencies:** `bundle install`
- **Fetch SPX option chains:** `bundle exec ruby bin/fetch_spx_option_chains`

## Development Conventions

- **Prototyping:** New analysis and charts are first prototyped in Jupyter notebooks in the `notebooks/` directory.
- **Code Structure:**
    - Reusable chart components are encapsulated as classes in `src/trade_lab/charts/`.
    - Trading indicator logic is placed in `src/trade_lab/indicators/`.
- **Testing:** Production code in `src` should be accompanied by tests in the `tests/` directory.
- **Linting & Formatting:** The project uses `ruff` for code formatting and linting to maintain a consistent style.
- **Type Checking:** `mypy` is used for static type analysis to ensure code quality.
