# qstudy Experiments CLI

The `qstudy` CLI is the experiment-management layer for the library. It scaffolds study folders, creates versioned iterations, runs one or more `run_study()` modules, and stores two different kinds of output:

- raw run artifacts in `out/`
- researcher notes plus selected metrics in `log.json`

The implementation lives in [`src/qstudy/experiments/`](/Users/jplatta/repos/qstudy/src/qstudy/experiments) and the command entrypoint is [`src/qstudy/cli.py`](/Users/jplatta/repos/qstudy/src/qstudy/cli.py).

## Feature Review

The current experiments feature is intentionally small and file-based:

- `qstudy create` generates a runnable experiment scaffold.
- `qstudy iterate` copies an existing `v*.py` file into the next numeric version.
- `qstudy run` executes one or more version files and writes timestamped artifacts to `out/`.
- `qstudy log-study` appends annotated results to `log.json`.
- `qstudy show-results` renders a summary table from `log.json`.
- `qstudy query` sorts logged results by a supported metric alias.

The key design choice is that `run` and `log-study` are separate steps. Running a study does not update `log.json` for you. If you want a version to appear in `show-results` or `query`, you need to log it explicitly.

## Config

The CLI looks for `.qstudy.toml` in this order:

1. `.qstudy.toml` in the current working directory
2. `~/.qstudy.toml`
3. fallback to the current working directory as `studies_root`

Supported keys:

```toml
studies_dir = "experiments"
data_dir = ".qstudy-data"
tickrake_sqlite_path = "~/.tickrake/tickrake.sqlite3"
tickrake_history_dirs = [
  "~/.tickrake/data/history/ibkr-paper",
  "~/.tickrake/data/history/tickrake",
]
```

Rules:

- `studies_dir` is required if a config file exists
- `data_dir` is optional
- `tickrake_sqlite_path` is optional and defaults to `~/.tickrake/tickrake.sqlite3`
- `tickrake_history_dirs` is optional and defaults to the ordered Tickrake day-history roots
- relative paths resolve relative to the config file that defined them
- if no config file exists, studies are created directly under the current working directory

Example:

```toml
studies_dir = "experiments"
data_dir = ".qstudy-data"
tickrake_sqlite_path = "~/.tickrake/tickrake.sqlite3"
tickrake_history_dirs = [
  "~/.tickrake/data/history/ibkr-paper",
  "~/.tickrake/data/history/tickrake",
]
```

With that file in the repo root:

```bash
uv run qstudy create residual-mr
```

creates:

```text
experiments/residual-mr/
```

## Commands

### `qstudy create <name>`

Creates a new experiment directory under `studies_root`.

Example:

```bash
uv run qstudy create residual-mr
```

Generated scaffold:

```text
residual-mr/
  shared.py
  v0.py
  run.py
  iteration_index.json
  log.json
  readme.md
```

Behavior:

- rejects duplicate experiment directories
- validates that `<name>` is a single path segment
- initializes `iteration_index.json` with the baseline `v0.py` entry
- initializes `log.json` as `[]`

### `qstudy iterate <study> <version-name> [--parent <stem>]`

Creates the next top-level version file by copying an existing version file.

Example:

```bash
uv run qstudy iterate residual-mr volume-confirmed
uv run qstudy iterate residual-mr vol-filter --parent v1_volume_confirmed
```

Behavior:

- finds the experiment under `studies_root`
- defaults to branching from the highest existing `v*.py` file
- supports branching from an explicit parent with `--parent`
- always assigns the next numeric version: if the highest version is `v10_*`, the next file is `v11_*`
- normalizes `<version-name>` into a lowercase underscore suffix
- rewrites obvious embedded version labels such as docstrings, `STUDY_NAME`, and `Study(name=...)`
- appends lineage metadata to `iteration_index.json`

Notes:

- execution discovers actual `v*.py` files on disk; `iteration_index.json` is metadata, not the execution source of truth
- if `iteration_index.json` is missing, it is rebuilt from the discovered version files

### `qstudy run <name> [--version <stem-or-filename>]`

Runs one or all version files in an experiment.

Examples:

```bash
uv run qstudy run residual-mr
uv run qstudy run residual-mr --version v1_volume_confirmed
uv run qstudy run residual-mr --version v1_volume_confirmed.py
```

Behavior:

- discovers top-level `v*.py` files and sorts them numerically
- prepends the experiment directory to `sys.path`, so version files can import `shared.py`
- imports each selected module dynamically
- requires each module to define `run_study()`
- requires `run_study()` to return a `dict`
- prints one JSON object per version to stdout
- writes one timestamped raw artifact per version into `out/`

Example stdout for one version:

```json
{
  "version": "v0",
  "run_at": "2026-05-29T12:34:56Z",
  "metrics": {
    "sharpe": 1.1,
    "ann_return": 0.2
  }
}
```

Important:

- `qstudy run` does not write `log.json`
- `qstudy run` does not aggregate results into `results.json` or `results.csv`
- `run.py` is just a local wrapper around the same `run_experiment(...)` path

### `qstudy log-study <name> --version ... --hypothesis ... --analysis ... --results ... [--parent ...]`

Appends an annotated entry to `log.json`.

Example:

```bash
uv run qstudy log-study residual-mr \
  --version v1_volume_confirmed \
  --hypothesis "Add a volume confirmation filter to reduce weak reversals" \
  --analysis "Sharpe improved while turnover fell; drawdown stayed similar." \
  --results '{"net_sharpe": 0.81, "ann_return": 0.11}'
```

Behavior:

- requires a JSON object for `--results`
- stores `version`, `ancestor`, `hypothesis`, `metrics`, `analysis`, and `run_at`
- infers `ancestor` from `iteration_index.json` when `--parent` is omitted and the version exists there

Use this command after `qstudy run` when you want a durable research log that can be rendered and queried later.

### `qstudy show-results <name>`

Reads `log.json` and renders a summary table.

Example:

```bash
uv run qstudy show-results residual-mr
```

Default columns are drawn from logged metrics and may include:

- `version`
- `ancestor`
- `metrics.net_sharpe`
- `metrics.ann_return`
- `metrics.ann_vol`
- `metrics.max_drawdown`
- `metrics.information_ratio`
- `metrics.avg_daily_turnover`

If no entries have been logged yet, the CLI prints `No results have been recorded yet.`.

### `qstudy query <name> --metric <alias> [--sort asc|desc|--min|--max]`

Sorts `log.json` entries by a supported metric alias.

Example:

```bash
uv run qstudy query residual-mr --metric net-sharpe --max
uv run qstudy query residual-mr --metric turnover --min
```

Supported metric aliases:

- `sharpe`
- `net-sharpe`
- `gross-sharpe`
- `return`
- `vol`
- `drawdown`
- `turnover`
- `bench-corr`
- `ir`
- `benchmark-sharpe`

Rows missing the requested metric are pushed to the end.

### `qstudy list`

Lists experiment directories and counts top-level `v*.py` files.

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

## Experiment Layout

The scaffold is small by design:

- `shared.py`: cached data loaders, constants, and helper signal functions reused across versions
- `v0.py`: baseline implementation with `run_study() -> dict`
- `vN_<label>.py`: later iterations created by `qstudy iterate`
- `run.py`: wrapper around `qstudy.experiments.run_experiment`
- `iteration_index.json`: lineage metadata for iterated versions
- `log.json`: researcher log for annotated results
- `out/`: timestamped raw metrics from `qstudy run`

The generated starter baseline uses:

- `index_code="SP500"` as the default universe
- `SPY` as the default benchmark
- a simple mean-reversion starter signal
- `Study(...).base_signal(...).build_long_short(...).run()`

## Typical Workflow

```bash
uv run qstudy create residual-mr
uv run qstudy run residual-mr --version v0
uv run qstudy log-study residual-mr \
  --version v0 \
  --hypothesis "Baseline mean reversion" \
  --analysis "Runnable baseline; establishes control metrics." \
  --results '{"net_sharpe": 0.68, "ann_return": 0.07}'
uv run qstudy iterate residual-mr volume-confirmed
uv run qstudy run residual-mr --version v1_volume_confirmed
uv run qstudy show-results residual-mr
uv run qstudy query residual-mr --metric net-sharpe --max
```

## Artifacts

### `out/<timestamp>_<version>.json`

Written by `qstudy run`.

Shape:

```json
{
  "version": "v1_volume_confirmed",
  "run_at": "2026-05-29T12:34:56Z",
  "metrics": {
    "net_sharpe": 0.81,
    "ann_return": 0.11
  }
}
```

This is the raw execution artifact. Each run creates a new file.

### `log.json`

Written by `qstudy log-study`.

Shape:

```json
[
  {
    "version": "v1_volume_confirmed",
    "ancestor": "v0",
    "hypothesis": "Add a volume confirmation filter",
    "metrics": {
      "net_sharpe": 0.81,
      "ann_return": 0.11
    },
    "analysis": "Sharpe improved while turnover fell.",
    "run_at": "2026-05-29T12:40:00Z"
  }
]
```

This is the source of truth for `show-results` and `query`.

### `iteration_index.json`

Written by `qstudy create` and `qstudy iterate`.

Shape:

```json
[
  {
    "version": 0,
    "file": "v0.py",
    "source_file": null,
    "label": null
  },
  {
    "version": 1,
    "file": "v1_volume_confirmed.py",
    "source_file": "v0.py",
    "parent": null,
    "label": "volume_confirmed"
  }
]
```

This file tracks lineage. It does not control which version files are run.

## Error Cases

The CLI returns a non-zero exit code for common failures, including:

- malformed `.qstudy.toml`
- missing experiments
- missing selected version files
- missing `run_study()`
- `run_study()` returning a non-dict value
- malformed `log.json`
- malformed `iteration_index.json`
- invalid metric aliases in `query`

## Current Limitations

These are worth knowing when using the feature:

- `qstudy run` and `qstudy log-study` are intentionally separate, so there is no single command that both executes and records a result
- `show-results` and `query` only see what has been logged to `log.json`
- the execution runner only discovers top-level `v*.py` files
- lineage in `iteration_index.json` can drift from files on disk, although execution still follows the files on disk
