MULTI-AGENT PYTHON CODING WORKFLOW (PLAINTEXT)

Purpose:
This document defines a simple, repeatable workflow for multiple coding agents working in parallel on the same GitHub repository using git worktrees, uv, pytest, and ruff.

⸻

REQUIREMENTS
	•	Python 3.12+
	•	git
	•	uv package manager
	•	pytest
	•	ruff
	•	GitHub push access
	•	(Optional) GitHub CLI (gh) for creating pull requests

⸻

PROJECT SETUP (ONE TIME, MAIN WORKTREE)
	1.	Initialize project with uv
uv init –python 3.12
uv add –dev pytest ruff
	2.	Project layout
src/myapp/init.py
src/myapp/core.py
tests/test_core.py
	3.	Example core functionality (src/myapp/core.py)
def add(a: int, b: int) -> int:
return a + b
	4.	Minimal unit tests (tests/test_core.py)
from myapp.core import add
def test_add_basic():
assert add(2, 3) == 5
def test_add_negative():
assert add(-1, -2) == -3
	5.	pyproject.toml essentials
[project]
name = “myapp”
version = “0.1.0”
requires-python = “>=3.12”
[tool.ruff]
line-length = 88
target-version = “py312”
[tool.ruff.lint]
select = [“E”, “F”, “I”, “B”, “UP”]
fixable = [“ALL”]
[tool.ruff.format]
quote-style = “double”
[tool.pytest.ini_options]
testpaths = [“tests”]

⸻

BRANCH NAMING CONVENTIONS

All branches MUST start with one of:
	•	feature/*
	•	fix/*
	•	chore/*

Examples:
	•	feature/add-export
	•	fix/null-pointer-crash
	•	chore/update-deps

⸻

CREATING A WORKTREE (ONE PER AGENT)

From the main worktree:
	1.	Fetch latest changes
git fetch origin
	2.	Create a new worktree and branch
git worktree add ../wt-agent-a -b feature/agent-a-task origin/main

Each agent must:
	•	Use exactly one worktree
	•	Use exactly one branch
	•	Never share a branch with another agent

⸻

PER-AGENT DAILY WORKFLOW
	1.	Enter your worktree
cd ../wt-agent-a
	2.	Sync dependencies
uv sync
	3.	Implement the core change
	•	Modify code in src/
	•	Add minimal tests in tests/
	•	Tests must directly cover the main behavior of the PR
	4.	Run formatter and linter BEFORE committing
uv run ruff format .
uv run ruff check . –fix
	5.	Run unit tests
uv run pytest

⸻

COMMIT RULES

Commit messages must:
	•	Be short (one line)
	•	Start with a verb/action
	•	Describe the change clearly

Good examples:
	•	Add CSV export
	•	Fix crash on empty input
	•	Refactor parsing logic
	•	Update dependencies

Commit:
git add -A
git commit -m “Add CSV export”

⸻

PUSHING AND OPENING A PULL REQUEST
	1.	Push your branch
git push -u origin HEAD
	2.	Open a pull request
Using GitHub CLI:
gh pr create –base main –fill
Or:
	•	Push branch
	•	Open PR via GitHub web UI

⸻

UPDATING YOUR BRANCH

If main has moved:
	1.	Fetch and rebase
git fetch origin
git rebase origin/main
	2.	Re-run checks
uv run ruff format .
uv run ruff check . –fix
uv run pytest

⸻

CLEANUP AFTER MERGE

From the main worktree:
	1.	Remove the worktree
git worktree remove ../wt-agent-a
	2.	Delete local branch
git branch -d feature/agent-a-task

⸻

RULES OF ENGAGEMENT (IMPORTANT)
	•	One agent = one worktree = one branch = one PR
	•	Never share a branch between agents
	•	Always run ruff and pytest before committing
	•	Keep PRs small and focused
	•	Tests should cover the core behavior only
	•	Do not push broken tests or failing lint

⸻

END OF FILE

If you want, I can also:
	•	Convert this into an executable shell script
	•	Add a pre-commit hook version
	•	Add a GitHub Actions CI file that enforces all of this