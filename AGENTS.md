# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/trade_lab/dashboard`, with the Streamlit entry point at `src/trade_labl/dashboard.py`.
- Tests are in `tests/` and follow `test_*.py` naming (see `tests/test_app.py`).
- Documentation and supporting notes live in `docs/`.
- Project configuration is defined in `pyproject.toml`.

## Build, Test, and Development Commands
- `uv sync`: create/update the virtual environment and install dependencies.
- `uv run streamlit run src/trade_lab/dashboard.py`: run the Streamlit dashboard locally.
- `uv run pytest`: run the test suite.
- `uv run ruff format .`: format code with Ruff.
- `uv run ruff check .`: lint the codebase.
- `uv run ruff check --fix .`: auto-fix lint issues where possible.
- `uv run mypy src/`: run type checks.

## Coding Style & Naming Conventions
- Python 3.9+ target. Use 4 spaces for indentation.
- Ruff is the formatter and linter; line length is 100 and quote style is double.
- Use snake_case for modules and functions, PascalCase for classes.
- Keep Streamlit UI logic in `dashboard.py`; add new modules under `src/trade_lab/dashboard` for reusable logic.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Test discovery uses `tests/` with `test_*.py` files and `test_*` functions.
- Add focused unit tests for new logic; avoid UI-heavy tests unless needed.
- Run tests with `uv run pytest` before opening a pull request.

## Commit & Pull Request Guidelines
- Git history is minimal; current convention is short, imperative messages (e.g., “start project”).
- Prefer messages like “add”, “fix”, “update”, “refactor”, and keep them concise.
- PRs should include a clear description, testing notes, and screenshots for UI changes.
- Link related issues or tasks if applicable.

## Configuration & Security Tips
- Do not commit secrets or local data files. Use environment variables for credentials.
- Keep dependencies updated via `uv` and validate changes with tests and linting.
