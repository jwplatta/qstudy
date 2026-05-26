# Contributing to qstudy

This repository is centered on the `qstudy` package in `src/qstudy/`. Keep core logic in the
library package unless a change explicitly belongs in `docs/` or test fixtures.

## Requirements

- Python `3.10+`
- `git`
- `uv`
- `pytest`
- `ruff`
- `mypy`

## Working Rules

1. Read `AGENTS.md` before making substantial changes.
2. Use a Git worktree for each task.
3. Never commit directly to `main`.
4. Use a branch prefixed with `feature/`, `fix/`, or `chore/`.
5. Keep PRs focused and covered by tests when practical.

## Setup

```bash
git clone https://github.com/jwplatta/qstudy.git
cd qstudy
uv sync
```

## Create a Worktree

From the main clone:

```bash
git fetch origin
git worktree add ../wt-my-task -b chore/my-task origin/main
cd ../wt-my-task
uv sync
```

## Development Checklist

1. Make changes in `src/qstudy/` unless the task clearly targets docs or tests.
2. Add or update tests in `tests/` for behavioral changes.
3. Run formatting, linting, and tests before committing.

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
uv run pytest
```

## Project Conventions

- Prefer reusable library code over one-off scripts.
- Signal filters should exclude assets with `NaN`, not `0.0`.
- The engine applies a 1-day execution lag from positions to returns.
- Public APIs should be re-exported from `src/qstudy/__init__.py` when appropriate.

## Commits and Pull Requests

- Use concise, action-oriented commit messages.
- Push your branch with `git push -u origin HEAD`.
- Open a PR against `main`.
- Rebase onto `origin/main` if the branch falls behind.

## Cleanup After Merge

From the main clone:

```bash
git worktree remove ../wt-my-task
git branch -d chore/my-task
```
