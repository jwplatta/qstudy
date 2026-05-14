# AGENTS.md

This repository is centered on `qstudy`: a Python library for quickly iterating on trading ideas with unconstrained backtests. Prefer work in `src/qstudy/` unless a task explicitly targets older top-level `src/` modules, charts, or docs utilities.

## Required Workflow

Agents must read and follow `CONTRIBUTING.md` before making substantial changes. Treat that file as the source of truth for contribution process and Git workflow.

Key requirements from `CONTRIBUTING.md`:

- Use a Git worktree for feature or fix work instead of developing directly in the main clone.
- Never commit directly to `main`.
- Use branch names prefixed with `feature/`, `fix/`, or `chore/`.
- Run formatting, linting, and tests before committing.
- Keep changes focused and covered by tests where practical.

## Skills

Before starting substantial work, check whether a shared skill applies. Use `skillex` first.

```bash
skillex list
skillex pull --agent codex
skillex pull python-code-quality --agent codex
skillex pull python-project-setup --agent codex
skillex pull python-testing --agent codex
skillex pull quant-studies --agent codex
```

For this repo, the baseline Codex skill set should include:

- `skillex`
- `python-code-quality`
- `python-project-setup`
- `python-testing`
- `quant-studies`

## Project Layout

- `src/qstudy/`: main library package
- `src/qstudy/study/`: core pipeline classes, engine, metrics, weighting, portfolio composition
- `src/qstudy/data/`: dataset loading and cache-aware download helpers
- `src/qstudy/signals/`: factor and filter helpers
- `src/qstudy/experiments/`: experiment scaffolding, config, logging, and runner support for the CLI
- `src/qstudy/charts/`: result visualization helpers
- `tests/`: pytest suite
- `docs/`: research notes, examples, and workflow docs

## Commands

```bash
uv sync
uv run pytest
uv run pytest tests/test_qstudy.py
uv run pytest -k sharpe
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .
uv run mypy src/
uv run qstudy list
uv run qstudy create <name>
uv run qstudy iterate <name> <suffix>
uv run qstudy run <name> [--version v1_example]
uv run qstudy show-results <name>
```

## Configuration

The CLI expects `.qstudy.toml` in the current directory or `~/.qstudy.toml`.

```toml
studies_dir = "./experiments"
data_dir = "./.qstudy-data"
```

## Architecture Notes

- `Study` in `src/qstudy/study/Study.py` is the primary pipeline abstraction.
- The normal flow is: load data -> optional residualization/factor prep -> base signal -> filters/transforms -> position construction -> weighting/scaling/rebalancing -> engine run -> metrics.
- Signal filters should mark ineligible assets with `NaN`, not `0.0`.
- The backtest engine applies a 1-day execution lag when mapping positions to returns.
- `PortfolioStudy` combines multiple `Study` instances into a shared portfolio-level run.
- `param_grid` in `src/qstudy/study/grid.py` is the parameter sweep entry point.
- Public-facing library imports should remain cleanly re-exported from `src/qstudy/__init__.py` when appropriate.

## Working Conventions

- Keep reusable library logic in `src/qstudy/`; avoid pushing core behavior into ad hoc scripts.
- Add focused tests for new library behavior, especially around metrics, weighting, pipeline ordering, and CLI experiment workflows.
- Treat `docs/` as supporting material, not the source of truth for runtime behavior.
- The worktree may be dirty. Do not revert unrelated user changes.
