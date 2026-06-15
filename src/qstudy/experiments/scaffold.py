from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from qstudy.experiments.errors import QStudyCliError

ITERATION_INDEX_FILENAME = "iteration_index.json"
LOG_FILENAME = "log.json"

_STUDY_FILE_RE = re.compile(r"^v(\d+)(?:[^/]*)\.py$")
_VERSION_STEM_RE = re.compile(r"^v(\d+)(?:_(.+))?$")


def create_experiment(studies_root: Path, name: str) -> Path:
    _validate_experiment_name(name)
    experiment_dir = studies_root / name
    if experiment_dir.exists():
        raise QStudyCliError(f"Experiment already exists: {experiment_dir}")

    studies_root.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir()
    files = scaffold_files(name)
    for filename, content in files.items():
        (experiment_dir / filename).write_text(content, encoding="utf-8")

    return experiment_dir


def scaffold_files(name: str) -> dict[str, str]:
    title = name.replace("-", " ").replace("_", " ").title()
    iteration_index = _build_iteration_index_rows(["v0.py"])
    return {
        "shared.py": _shared_template(),
        "v0.py": _v0_template(name),
        "run.py": _run_template(),
        ITERATION_INDEX_FILENAME: f"{json.dumps(iteration_index, indent=2)}\n",
        LOG_FILENAME: "[]\n",
        "readme.md": _readme_template(name, title),
    }


def list_experiments(studies_root: Path) -> list[tuple[str, int]]:
    if not studies_root.exists():
        return []

    rows: list[tuple[str, int]] = []
    for child in sorted(studies_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        rows.append((child.name, len(discover_version_files(child))))
    return rows


def discover_version_files(experiment_dir: Path) -> list[Path]:
    versions: list[tuple[int, str, Path]] = []
    for child in experiment_dir.iterdir():
        if not child.is_file():
            continue
        match = _STUDY_FILE_RE.match(child.name)
        if match is None:
            continue
        versions.append((int(match.group(1)), child.name, child))
    return [path for _, _, path in sorted(versions, key=lambda item: (item[0], item[1]))]


def iterate_experiment(
    studies_root: Path, study: str, version_name: str, parent: str | None = None
) -> Path:
    experiment_dir = studies_root / study
    if not experiment_dir.exists():
        raise QStudyCliError(f"Experiment not found: {experiment_dir}")

    version_files = discover_version_files(experiment_dir)
    if not version_files:
        raise QStudyCliError(f"No version files found in {experiment_dir}")

    if parent is not None:
        parent_matches = [f for f in version_files if f.stem == parent]
        if not parent_matches:
            raise QStudyCliError(f"Parent version not found: {parent}")
        source_file = parent_matches[0]
    else:
        source_file = version_files[-1]

    index_rows = read_iteration_index_rows(experiment_dir)
    source_version, _ = _parse_version_stem(source_file.stem)
    suffix = sanitize_version_name(version_name)

    # Next version number is always one past the highest existing version.
    max_version = max(_parse_version_stem(f.stem)[0] for f in version_files)
    next_version = max_version + 1

    destination_name = f"v{next_version}_{suffix}.py"
    destination_path = experiment_dir / destination_name
    if destination_path.exists():
        raise QStudyCliError(f"Iteration already exists: {destination_path}")

    shutil.copyfile(source_file, destination_path)
    new_text = _rewrite_iteration_text(
        destination_path.read_text(encoding="utf-8"),
        old_stem=source_file.stem,
        new_stem=destination_path.stem,
    )
    destination_path.write_text(new_text, encoding="utf-8")

    index_rows.append(
        {
            "version": next_version,
            "file": destination_name,
            "source_file": source_file.name,
            "parent": parent,
            "label": suffix,
        }
    )
    write_iteration_index_rows(experiment_dir, index_rows)
    return destination_path


def read_iteration_index_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    index_path = experiment_dir / ITERATION_INDEX_FILENAME
    if not index_path.exists():
        version_filenames = [path.name for path in discover_version_files(experiment_dir)]
        rows = _build_iteration_index_rows(version_filenames)
        write_iteration_index_rows(experiment_dir, rows)
        return rows

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QStudyCliError(f"Malformed iteration index JSON in {index_path}: {exc.msg}") from exc

    if payload == []:
        return []
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise QStudyCliError(
            f"Malformed iteration index JSON in {index_path}: expected a list of objects"
        )
    return [dict(row) for row in payload]


def write_iteration_index_rows(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    index_path = experiment_dir / ITERATION_INDEX_FILENAME
    index_path.write_text(f"{json.dumps(rows, indent=2)}\n", encoding="utf-8")


def lookup_parent(experiment_dir: Path, version_stem: str) -> str | None:
    """Return the parent version stem recorded for ``version_stem`` in the index, or None."""
    for row in read_iteration_index_rows(experiment_dir):
        stem = Path(row.get("file", "")).stem
        if stem == version_stem:
            return row.get("parent") or None
    return None


def sanitize_version_name(version_name: str) -> str:
    normalized = version_name.strip().lower()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        raise QStudyCliError("Version name must include at least one letter or number.")
    return normalized


def _validate_experiment_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise QStudyCliError("Experiment name must not be empty.")
    if Path(name).name != name:
        raise QStudyCliError("Experiment name must be a single path segment.")


def _parse_version_stem(stem: str) -> tuple[int, str | None]:
    match = _VERSION_STEM_RE.match(stem)
    if match is None:
        raise QStudyCliError(f"Invalid study version filename stem: {stem}")
    return int(match.group(1)), match.group(2)


def _build_iteration_index_rows(version_filenames: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_file: str | None = None
    for filename in version_filenames:
        version, label = _parse_version_stem(Path(filename).stem)
        rows.append(
            {
                "version": version,
                "file": filename,
                "source_file": previous_file,
                "label": label,
            }
        )
        previous_file = filename
    return rows


def _rewrite_iteration_text(text: str, old_stem: str, new_stem: str) -> str:
    old_version, _ = _parse_version_stem(old_stem)
    new_version, _ = _parse_version_stem(new_stem)
    text = text.replace(old_stem, new_stem)
    text = text.replace(old_stem.replace("_", "-"), new_stem.replace("_", "-"))
    text = text.replace(f"v{old_version}", f"v{new_version}", 1)
    text = re.sub(
        r'^(STUDY_NAME\s*=\s*["\'])([^"\']*)(["\'])',
        lambda m: f"{m.group(1)}{m.group(2).replace(f'v{old_version}', new_stem, 1)}{m.group(3)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'(\bname\s*=\s*["\'])([^"\']*)(["\'])',
        lambda m: _replace_name_argument(m, old_stem, new_stem, old_version),
        text,
        flags=re.MULTILINE,
    )

    # Adjust leading docstring/version labels without broad free-text replacement.
    text = re.sub(
        rf'(^[ruRUfF]*"""?)v{old_version}(\b)',
        lambda m: m.group(0).replace(f"v{old_version}", f"v{new_version}", 1),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        rf"(^[ruRUfF]*'''?)v{old_version}(\b)",
        lambda m: m.group(0).replace(f"v{old_version}", f"v{new_version}", 1),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def _replace_name_argument(
    match: re.Match[str],
    old_stem: str,
    new_stem: str,
    old_version: int,
) -> str:
    value = match.group(2).replace(old_stem, new_stem, 1)
    value = value.replace(f"v{old_version}", new_stem, 1)
    return f"{match.group(1)}{value}{match.group(3)}"


def _shared_template() -> str:
    return """from __future__ import annotations

from functools import cache

import qstudy as qs
START_DATE = "2018-01-01"  # TODO: adjust the study start date.
END_DATE = "2024-12-31"  # TODO: adjust the study end date.
UNIVERSE_INDEX = "SP500"  # TODO: replace with a different index-backed universe if needed.
BENCHMARK_TICKER = "SPY"  # TODO: change the benchmark if needed.
N_LONG = 25
N_SHORT = 25


@cache
def load_universe():
    \"\"\"Download the default universe for this experiment.

    TODO: replace the default index-backed universe or switch to explicit tickers if needed.
    \"\"\"

    return qs.download(index_code=UNIVERSE_INDEX, start=START_DATE, end=END_DATE)


@cache
def load_benchmark():
    return qs.download([BENCHMARK_TICKER], START_DATE, END_DATE)


def mean_reversion_signal(window: int = 5):
    \"\"\"Starter signal: recent losers score highest.

    TODO: replace this with your own signal logic once the scaffold is in place.
    \"\"\"

    def signal(**cache):
        returns = cache["_active_returns"]
        return -returns.rolling(window).mean()

    signal.__name__ = f"mean_reversion_signal_{window}"
    return signal
"""


def _v0_template(name: str) -> str:
    return f"""from __future__ import annotations

import json

from qstudy import Study

from shared import N_LONG, N_SHORT, load_benchmark, load_universe, mean_reversion_signal


def run_study() -> dict:
    \"\"\"Run the baseline study for {name}.\"\"\"

    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name="{name}:v0")
        .base_signal(mean_reversion_signal(window=5))  # TODO: replace the starter signal.
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
"""


def _run_template() -> str:
    return """from __future__ import annotations

from pathlib import Path

from qstudy.experiments import run_experiment


def main() -> int:
    rows = run_experiment(Path(__file__).resolve().parent)
    print(f"Ran {len(rows)} study version(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _readme_template(name: str, title: str) -> str:
    return f"""# {title}

Created with `qstudy create {name}`.

Files:
- `v0.py`: baseline study entrypoint with `run_study() -> dict`
- `shared.py`: shared universe, benchmark, and signal helpers
- `iteration_index.json`: append-only metadata for CLI-created study iterations
- `log.json`: structured experiment log (version, hypothesis, metrics, analysis)
- `out/`: timestamped raw metrics from each run (written automatically by `qstudy run`)

Workflow:
1. Edit `shared.py` and `v0.py` with your signal logic.
2. Run `qstudy run {name} --version v0` to execute the study and write `out/<timestamp>_v0.json`.
3. Review metrics, then log the result:
   ```
   qstudy append {name} \\
     --version v0 \\
     --hypothesis "..." \\
     --analysis "..." \\
     --results '{{"net_sharpe": 0.68, ...}}'
   ```
4. Create the next iteration: `qstudy iterate {name} <version-name>`.
5. Inspect the log: `qstudy show-results {name}`.
"""
