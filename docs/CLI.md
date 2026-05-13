# qstudy CLI

The `qstudy` CLI manages experiment folders for multi-version studies.

Current commands:

- `qstudy create <name>`
- `qstudy iterate <study> <version-name>`
- `qstudy run <name>`
- `qstudy list`
- `qstudy show-results <name>`

---

## Install

From the repository root:

```bash
uv sync
```

That installs the project into the local `uv` environment and exposes the console script defined in [pyproject.toml](../pyproject.toml).

Run the CLI with `uv run`:

```bash
uv run qstudy list
uv run qstudy create my-experiment
uv run qstudy iterate my-experiment volume-confirmed
uv run qstudy run my-experiment
uv run qstudy run my-experiment --version v1_volume_confirmed
uv run qstudy show-results my-experiment
```

---

## Config

The CLI looks for `.qstudy.toml` in this order:

1. `.qstudy.toml` in the current working directory
2. `~/.qstudy.toml`
3. fallback to the current working directory as the studies root

Supported config:

```toml
studies_dir = "experiments"
```

Rules:

- `studies_dir` may be absolute or relative
- if loaded from a local `.qstudy.toml`, relative paths resolve from that file's directory
- if loaded from `~/.qstudy.toml`, relative paths resolve from your home directory

Example local config:

```toml
studies_dir = "experiments"
```

With that file in the repo root, `uv run qstudy create alpha` creates:

```text
experiments/alpha/
```

---

## Commands

## `qstudy create <name>`

Creates a runnable experiment scaffold under the configured studies root.

Example:

```bash
uv run qstudy create residual-mr
```

Generated files:

- `v0.py`
- `run.py`
- `shared.py`
- `iteration_index.json`
- `results.json`
- `results.csv`
- `log.md`
- `readme.md`

Behavior:

- fails if the experiment directory already exists
- initializes `iteration_index.json` with the baseline `v0.py` entry
- initializes `results.json` as an empty JSON array
- initializes `results.csv` as an empty but valid CSV artifact

## `qstudy iterate <study> <version-name>`

Creates the next top-level version file by copying the highest existing version file.

Example:

```bash
uv run qstudy iterate residual-mr volume-confirmed
```

If the highest existing version is `v0.py`, the command creates:

```text
v1_volume_confirmed.py
```

Behavior:

- resolves `<study>` under the configured studies root
- picks the highest existing numeric version file; if several files share that number, lexicographic filename order breaks the tie
- normalizes `<version-name>` into a safe suffix like `volume_confirmed`
- copies the source file and updates obvious embedded version labels when they directly reference the prior stem
- appends metadata to `iteration_index.json`

## `qstudy list`

Lists top-level experiment directories and counts top-level `v*.py` files in each one.

Example:

```bash
uv run qstudy list
```

Example output:

```text
experiment     study_count
-------------  -----------
residual-mr    3
momentum-test  1
```

Only top-level version files count. Nested files are ignored.

## `qstudy run <name>`

Runs all top-level `v*.py` files in the named experiment and writes `results.json` plus `results.csv`.

Example:

```bash
uv run qstudy run residual-mr
uv run qstudy run residual-mr --version v1_volume_confirmed
```

Behavior:

- resolves `<name>` under the configured studies root
- uses the same execution path as the generated `run.py`
- accepts `--version <stem-or-filename>` to run exactly one version
- prints how many study versions were run
- overwrites `results.json` and `results.csv` with exactly the versions that were run

## `qstudy show-results <name>`

Reads `<experiment>/results.json` and prints a terminal summary table.

Example:

```bash
uv run qstudy show-results residual-mr
```

Default displayed columns:

- `version`
- `sharpe`
- `ann_return`
- `ann_vol`
- `max_drawdown`
- `information_ratio`
- `avg_daily_turnover`

If a metric is absent in every row, that column is omitted from the table.

If `results.json` is empty, the CLI prints a clear message instead of failing.

---

## Experiment Workflow

Typical flow:

```bash
uv run qstudy create residual-mr
uv run qstudy iterate residual-mr volume-confirmed
uv run qstudy run residual-mr
uv run qstudy show-results residual-mr
```

What to edit:

- `shared.py`: shared dates, benchmark, universe loader, and starter signal helpers
- `v0.py`: baseline study with `run_study() -> dict`
- future `v*.py` files created by `qstudy iterate`
- `iteration_index.json`: append-only lineage metadata for iterations

What `run.py` does:

- discovers top-level `v*.py` files
- sorts them by numeric version suffix
- imports each module dynamically
- requires a `run_study()` function
- collects one metrics dict per version
- writes `results.json`
- writes `results.csv`

`qstudy run <name>` calls that same execution path without requiring you to `cd` into the experiment directory first.
`qstudy run <name> --version <version>` uses that same execution path but filters to one exact version stem or filename.

The generated `v0.py` uses a minimal runnable baseline:

- `SP500` universe
- `SPY` benchmark
- starter date range in `shared.py`
- a simple mean-reversion signal
- long/short portfolio construction

The scaffold is intentionally runnable first and customizable second. The `TODO` markers in `shared.py` and `v0.py` show the intended edit points.

---

## Results Format

`results.json`:

- JSON array of objects
- one object per version
- each row includes `version` plus flattened metrics

Example:

```json
[
  {
    "version": "v0",
    "sharpe": 1.12,
    "ann_return": 0.18
  },
  {
    "version": "v1",
    "sharpe": 1.34,
    "ann_return": 0.21
  }
]
```

`results.csv`:

- one row per version
- same columns as `results.json`
- overwritten on every run, including filtered single-version runs

`iteration_index.json`:

- JSON array of objects
- one object per version file
- stores the numeric version, filename, source filename, and normalized label
- metadata only; `run.py` still uses `v*.py` discovery as the source of truth

---

## Errors

The CLI returns a non-zero exit code for:

- invalid config files
- missing experiments
- missing or malformed `results.json`
- duplicate scaffold targets

The generated `run.py` also fails if a version module does not define `run_study()`.

---

## Notes

- The CLI only reads config in v1. There is no config-writing command.
- `qstudy show-results` reads `results.json` as the source of truth.
- The scaffold is meant for local research iteration, not as a deployment system.
