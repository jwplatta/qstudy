# Trade Lab

Trading analysis tools for developing indicators like GEX, analyzing trades, and visualizing data.

## Overview

Trade Lab is a Python project focused on:
- Developing custom trading indicators (GEX, etc.)
- Analyzing trade data and performance
- Creating reusable chart classes for common visualizations
- Prototyping in Jupyter notebooks

## Project Structure

```
trade-lab/
├── src/trade_lab/          # Main package
│   ├── charts/             # Reusable chart classes
│   └── indicators/         # Trading indicators (GEX, etc.)
├── notebooks/              # Jupyter notebooks for prototyping
├── tests/                  # Test files
└── docs/                   # Documentation
```

## Setup

This project uses `uv` for fast Python package management.

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create virtual environment and install dependencies

```bash
uv sync
```

This will create a virtual environment in `.venv` and install all dependencies.

## Usage

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Run Jupyter Lab

```bash
jupyter lab
```

Notebooks should be created in the `notebooks/` directory.

### Run tests

```bash
uv run pytest
```

### Format and lint code

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

### Type checking

```bash
uv run mypy src/
```

## Development Workflow

1. Prototype analysis and charts in Jupyter notebooks (`notebooks/`)
2. Convert frequently-used charts into reusable classes in `src/trade_lab/charts/`
3. Implement trading indicators in `src/trade_lab/indicators/`
4. Write tests in `tests/` for production code

## Dependencies

Core libraries:
- `numpy` - Numerical computing
- `pandas` - Data analysis and manipulation
- `matplotlib` - Plotting and visualization
- `jupyter` - Interactive notebooks

Development tools:
- `pytest` - Testing framework
- `ruff` - Fast linting and formatting
- `mypy` - Static type checking

## Adding New Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```
