# Trade Lab

Trading analysis tools for developing indicators like GEX, analyzing trades, and visualizing data.

## Overview

Trade Lab is a Python project focused on:
- Developing trading indicators (GEX, etc.)
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

This project uses both Python (via `uv`) and Ruby (via `bundler`) for different tasks:
- Python for analysis, indicators, and visualization
- Ruby for fetching data from Schwab API

### Python Setup

Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install Python dependencies:

```bash
uv sync
```

### Ruby Setup

Install Ruby dependencies:

```bash
bundle install
```

### Schwab API Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Schwab API credentials:
   - Get API credentials from https://developer.schwab.com/
   - Set `SCHWAB_API_KEY`, `SCHWAB_APP_SECRET`, and `SCHWAB_APP_CALLBACK_URL`

3. On first run, the script will open a browser for OAuth authentication

## Usage

### Fetching Data from Schwab

Fetch SPX option chains for the next 7 days:

```bash
bundle exec ruby bin/fetch_spx_option_chains
```

This will create CSV files in the `data/` directory with option chain data for each expiration date.

### Python Analysis

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Run Jupyter Lab for interactive analysis:

```bash
jupyter lab
```

Notebooks should be created in the `notebooks/` directory. Load the CSV data from `data/` for analysis.

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

### Python Dependencies

Core libraries:
- `numpy` - Numerical computing
- `pandas` - Data analysis and manipulation
- `matplotlib` - Plotting and visualization
- `jupyter` - Interactive notebooks

Development tools:
- `pytest` - Testing framework
- `ruff` - Fast linting and formatting
- `mypy` - Static type checking

### Ruby Dependencies

Data fetching:
- `schwab_rb` - Schwab API client
- `dotenv` - Environment variable management
- `pry` - Debugging (optional)

## Adding New Dependencies

### Python Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

### Ruby Dependencies

```bash
# Add to Gemfile, then run:
bundle install
```
