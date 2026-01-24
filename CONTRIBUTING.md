# Contributing to This Project

This document outlines the workflow for contributing to this repository. Following these guidelines helps maintain code quality and a smooth development process for everyone.

## Requirements
- Python 3.12+
- `git`
- `uv` package manager
- `pytest`
- `ruff`
- A GitHub account with push access to a fork or the main repository.
- (Optional) GitHub CLI (`gh`) for creating pull requests.

## Project Setup (One-Time)

If you are setting up the project for the first time on your main worktree:

1.  **Initialize the project with `uv`**:
    ```bash
    uv init --python 3.12
    uv add --dev pytest ruff
    ```
2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

## Branch Naming Conventions

All branches MUST start with one of the following prefixes:
- `feature/*`
- `fix/*`
- `chore/*`

**Examples:**
- `feature/add-export-feature`
- `fix/resolve-null-pointer-issue`
- `chore/update-dependencies`

## Development Workflow

To ensure that work is isolated and does not conflict with others, we recommend using git worktrees. Each feature or fix should be developed in its own worktree and branch.

### Creating a Worktree

From your main repository clone:

1.  **Fetch the latest changes from the remote repository**:
    ```bash
    git fetch origin
    ```

2.  **Create a new worktree and branch for your task**:
    ```bash
    git worktree add ../<worktree-name> -b <branch-name> origin/main
    ```
    For example:
    ```bash
    git worktree add ../wt-new-feature -b feature/add-new-feature origin/main
    ```

### Daily Workflow Steps

1.  **Navigate to your worktree directory**:
    ```bash
    cd ../<worktree-name>
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

3.  **Implement your changes**:
    - Modify the source code in the `src/` directory.
    - Add or update tests in the `tests/` directory to cover your changes.

4.  **Format and lint your code before committing**:
    ```bash
    uv run ruff format .
    uv run ruff check . --fix
    ```

5.  **Run all unit tests to ensure nothing has broken**:
    ```bash
    uv run pytest
    ```

## Commit Rules

Commit messages should be concise and clear.

- Keep the subject line short (one line).
- Start the message with a verb that describes the action taken.
- Clearly describe the purpose of the change.

**Good Examples:**
- `Add CSV export functionality`
- `Fix crash when input is empty`
- `Refactor data parsing logic`
- `Update project dependencies`

**Committing your work:**
```bash
git add -A
git commit -m "Your descriptive commit message"
```

## Pushing and Opening a Pull Request

1.  **Push your branch to the remote repository**:
    ```bash
    git push -u origin HEAD
    ```

2.  **Open a pull request**:
    - You can use the GitHub CLI:
      ```bash
      gh pr create --base main --fill
      ```
    - Alternatively, you can open a pull request through the GitHub web interface.

## Keeping Your Branch Updated

If the `main` branch has been updated while you were working, you should update your branch:

1.  **Fetch the latest changes and rebase your branch**:
    ```bash
    git fetch origin
    git rebase origin/main
    ```

2.  **Re-run all checks to ensure compatibility**:
    ```bash
    uv run ruff format .
    uv run ruff check . --fix
    uv run pytest
    ```

## Cleaning Up After Your PR is Merged

Once your pull request has been merged, you can clean up your local environment.

From your main worktree:

1.  **Remove the worktree**:
    ```bash
    git worktree remove ../<worktree-name>
    ```

2.  **Delete the local feature branch**:
    ```bash
    git branch -d <branch-name>
    ```

## Core Principles

- **Isolate Your Work**: One feature/fix = one worktree = one branch = one PR.
- **Don't Share Branches**: Never share a feature branch with another contributor.
- **Check Your Work**: Always run `ruff` and `pytest` before committing.
- **Small, Focused PRs**: Keep pull requests small and focused on a single issue.
- **Test Your Code**: Tests should cover the core behavior of your changes.
- **Don't Push Broken Code**: Ensure tests and linting pass before pushing.
