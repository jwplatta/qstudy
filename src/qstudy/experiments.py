from __future__ import annotations

import importlib.util
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_FILENAME = ".qstudy.toml"
RESULTS_FILENAME = "results.json"
RESULTS_CSV_FILENAME = "results.csv"
DEFAULT_RESULTS_COLUMNS = [
    "version",
    "sharpe",
    "ann_return",
    "ann_vol",
    "max_drawdown",
    "information_ratio",
    "avg_daily_turnover",
]

_STUDY_FILE_RE = re.compile(r"^v(\d+)(?:[^/]*)\.py$")


class QStudyCliError(Exception):
    """Base error for qstudy CLI operations."""


class ConfigError(QStudyCliError):
    """Raised for invalid qstudy config files."""


@dataclass(frozen=True)
class StudiesConfig:
    studies_root: Path
    source: Path | None


def load_studies_config(
    cwd: Path | None = None,
    home: Path | None = None,
) -> StudiesConfig:
    cwd = Path.cwd() if cwd is None else Path(cwd).resolve()
    home = Path.home() if home is None else Path(home).resolve()

    local_config = cwd / CONFIG_FILENAME
    if local_config.exists():
        studies_root = _read_studies_dir(local_config)
        return StudiesConfig(studies_root=studies_root, source=local_config)

    global_config = home / CONFIG_FILENAME
    if global_config.exists():
        studies_root = _read_studies_dir(global_config)
        return StudiesConfig(studies_root=studies_root, source=global_config)

    return StudiesConfig(studies_root=cwd, source=None)


def _read_studies_dir(config_path: Path) -> Path:
    raw = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    studies_dir = Path(raw)
    if not studies_dir.is_absolute():
        studies_dir = (config_path.parent / studies_dir).resolve()
    return studies_dir


def _parse_config_text(text: str, config_path: Path) -> str:
    studies_dir: str | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key != "studies_dir":
            raise ConfigError(
                f"Invalid config in {config_path}:{lineno}: unsupported key {key!r}"
            )
        if studies_dir is not None:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: duplicate studies_dir")
        studies_dir = _parse_toml_string(value, config_path, lineno)

    if studies_dir is None:
        raise ConfigError(f"Invalid config in {config_path}: missing studies_dir")
    return studies_dir


def _strip_toml_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    out: list[str] = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\" and in_quote is not None:
            out.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            if in_quote is None:
                in_quote = char
            elif in_quote == char:
                in_quote = None
            out.append(char)
            continue
        if char == "#" and in_quote is None:
            break
        out.append(char)
    return "".join(out)


def _parse_toml_string(value: str, config_path: Path, lineno: int) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    raise ConfigError(
        f"Invalid config in {config_path}:{lineno}: studies_dir must be a quoted string"
    )


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
    return {
        "shared.py": _shared_template(),
        "v0.py": _v0_template(name),
        "run.py": _run_template(),
        RESULTS_FILENAME: "[]\n",
        RESULTS_CSV_FILENAME: "version\n",
        "log.md": f"# {title} Log\n\n- Created with `qstudy create {name}`.\n",
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


def run_experiment(experiment_dir: Path) -> list[dict[str, Any]]:
    if not experiment_dir.exists():
        raise QStudyCliError(f"Experiment not found: {experiment_dir}")

    version_files = discover_version_files(experiment_dir)
    rows: list[dict[str, Any]] = []
    with _prepend_sys_path(experiment_dir):
        for version_file in version_files:
            version = version_file.stem
            module_name = f"qstudy_experiment_{experiment_dir.name}_{version}"
            module = _load_module(version_file, module_name)
            run_study = getattr(module, "run_study", None)
            if not callable(run_study):
                raise QStudyCliError(f"Missing run_study() in {version_file}")

            metrics = run_study()
            if not isinstance(metrics, dict):
                raise QStudyCliError(f"run_study() in {version_file} must return a dict")

            row = {"version": version}
            row.update(flatten_metrics(metrics))
            rows.append(row)

    write_results_artifacts(experiment_dir, rows)
    return rows


def write_results_artifacts(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = experiment_dir / RESULTS_FILENAME
    csv_path = experiment_dir / RESULTS_CSV_FILENAME

    json_path.write_text(f"{json.dumps(rows, indent=2, default=str)}\n", encoding="utf-8")

    all_columns = union_columns(rows, leading_columns=["version"])
    frame = pd.DataFrame(rows)
    if all_columns:
        frame = frame.reindex(columns=all_columns)
    frame.to_csv(csv_path, index=False)


def read_results_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    results_path = experiment_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise QStudyCliError(f"Results file not found: {results_path}")

    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QStudyCliError(f"Malformed results JSON in {results_path}: {exc.msg}") from exc

    if payload == []:
        return []
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise QStudyCliError(
            f"Malformed results JSON in {results_path}: expected a list of objects"
        )

    return [dict(row) for row in payload]


def render_results_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No results have been recorded yet."

    columns = [
        column
        for column in DEFAULT_RESULTS_COLUMNS
        if any(row.get(column) not in {None, ""} for row in rows)
    ]
    if "version" not in columns:
        columns.insert(0, "version")
    return render_table(rows, columns)


def render_experiment_list(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return "No experiments found."

    records = [{"experiment": name, "study_count": count} for name, count in rows]
    return render_table(records, ["experiment", "study_count"])


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    formatted_rows: list[list[str]] = []
    widths = [len(column) for column in columns]

    for row in rows:
        formatted_row = [format_cell(row.get(column)) for column in columns]
        formatted_rows.append(formatted_row)
        widths = [max(width, len(cell)) for width, cell in zip(widths, formatted_row)]

    def render_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    header = render_row(columns)
    separator = "  ".join("-" * width for width in widths)
    body = [render_row(cells) for cells in formatted_rows]
    return "\n".join([header, separator, *body])


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.4f}"
    return str(value)


def union_columns(
    rows: list[dict[str, Any]],
    leading_columns: list[str] | None = None,
) -> list[str]:
    leading_columns = [] if leading_columns is None else list(leading_columns)
    seen = set(leading_columns)
    columns = list(leading_columns)
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    return columns


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        _flatten_value(flattened, key, value)
    return flattened


def _flatten_value(flattened: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _flatten_value(flattened, f"{prefix}.{child_key}", child_value)
        return
    flattened[prefix] = value


def _validate_experiment_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise QStudyCliError("Experiment name must not be empty.")
    if Path(name).name != name:
        raise QStudyCliError("Experiment name must be a single path segment.")


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise QStudyCliError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _shared_template() -> str:
    return """from __future__ import annotations

import qstudy as qs
from qstudy.constants import SP500

START_DATE = "2018-01-01"  # TODO: adjust the study start date.
END_DATE = "2024-12-31"  # TODO: adjust the study end date.
BENCHMARK_TICKER = "SPY"  # TODO: change the benchmark if needed.
N_LONG = 25
N_SHORT = 25


def load_universe():
    \"\"\"Download the default universe for this experiment.

    TODO: replace SP500 with a different universe or a cached dataset if needed.
    \"\"\"

    return qs.download(SP500, START_DATE, END_DATE)


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
- `run.py`: execute all top-level `v*.py` files and write `results.json` and `results.csv`
- `log.md`: experiment notes

Workflow:
1. Edit `shared.py` and `v0.py`.
2. Add `v1.py`, `v2.py`, and so on as you iterate.
3. Run `python run.py` inside this directory.
4. Inspect `results.json` or use `qstudy show-results {name}`.
"""
