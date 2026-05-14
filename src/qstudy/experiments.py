from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_FILENAME = ".qstudy.toml"
LOG_FILENAME = "log.json"
OUT_DIRNAME = "out"
ITERATION_INDEX_FILENAME = "iteration_index.json"
DEFAULT_LOG_COLUMNS = [
    "version",
    "ancestor",
    "metrics.net_sharpe",
    "metrics.ann_return",
    "metrics.ann_vol",
    "metrics.max_drawdown",
    "metrics.information_ratio",
    "metrics.avg_daily_turnover",
]

_STUDY_FILE_RE = re.compile(r"^v(\d+)(?:[^/]*)\.py$")
_VERSION_STEM_RE = re.compile(r"^v(\d+)(?:_(.+))?$")


class QStudyCliError(Exception):
    """Base error for qstudy CLI operations."""


class ConfigError(QStudyCliError):
    """Raised for invalid qstudy config files."""


@dataclass(frozen=True)
class StudiesConfig:
    studies_root: Path
    data_root: Path | None
    source: Path | None


@dataclass
class ExperimentEntry:
    """A single annotated log entry combining metrics with researcher notes."""

    version: str
    ancestor: str | None
    hypothesis: str
    metrics: dict[str, Any]
    analysis: str
    run_at: str  # ISO 8601 UTC

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_studies_config(
    cwd: Path | None = None,
    home: Path | None = None,
) -> StudiesConfig:
    cwd = Path.cwd() if cwd is None else Path(cwd).resolve()
    home = Path.home() if home is None else Path(home).resolve()

    local_config = cwd / CONFIG_FILENAME
    if local_config.exists():
        config_values = _read_config(local_config)
        return StudiesConfig(
            studies_root=config_values["studies_dir"],
            data_root=config_values["data_dir"],
            source=local_config,
        )

    global_config = home / CONFIG_FILENAME
    if global_config.exists():
        config_values = _read_config(global_config)
        return StudiesConfig(
            studies_root=config_values["studies_dir"],
            data_root=config_values["data_dir"],
            source=global_config,
        )

    return StudiesConfig(studies_root=cwd, data_root=None, source=None)


def _read_config(config_path: Path) -> dict[str, Path | None]:
    raw = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    return {
        "studies_dir": _resolve_config_path(raw["studies_dir"], config_path),
        "data_dir": _resolve_config_path(raw["data_dir"], config_path),
    }


def _resolve_config_path(raw: str | None, config_path: Path) -> Path | None:
    if raw is None:
        return None
    resolved = Path(raw)
    if not resolved.is_absolute():
        resolved = (config_path.parent / resolved).resolve()
    return resolved


def _parse_config_text(text: str, config_path: Path) -> dict[str, str | None]:
    studies_dir: str | None = None
    data_dir: str | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in {"studies_dir", "data_dir"}:
            raise ConfigError(
                f"Invalid config in {config_path}:{lineno}: unsupported key {key!r}"
            )
        if key == "studies_dir":
            if studies_dir is not None:
                raise ConfigError(
                    f"Invalid config in {config_path}:{lineno}: duplicate studies_dir"
                )
            studies_dir = _parse_toml_string(value, config_path, lineno, key)
            continue

        if data_dir is not None:
            raise ConfigError(f"Invalid config in {config_path}:{lineno}: duplicate data_dir")
        data_dir = _parse_toml_string(value, config_path, lineno, key)

    if studies_dir is None:
        raise ConfigError(f"Invalid config in {config_path}: missing studies_dir")
    return {"studies_dir": studies_dir, "data_dir": data_dir}


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


def _parse_toml_string(value: str, config_path: Path, lineno: int, key: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    raise ConfigError(
        f"Invalid config in {config_path}:{lineno}: {key} must be a quoted string"
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


def iterate_experiment(studies_root: Path, study: str, version_name: str) -> Path:
    experiment_dir = studies_root / study
    if not experiment_dir.exists():
        raise QStudyCliError(f"Experiment not found: {experiment_dir}")

    version_files = discover_version_files(experiment_dir)
    if not version_files:
        raise QStudyCliError(f"No version files found in {experiment_dir}")

    index_rows = read_iteration_index_rows(experiment_dir)
    source_file = version_files[-1]
    source_version, _ = _parse_version_stem(source_file.stem)
    suffix = sanitize_version_name(version_name)
    next_version = source_version + 1
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
            "label": suffix,
        }
    )
    write_iteration_index_rows(experiment_dir, index_rows)
    return destination_path


def run_experiment(experiment_dir: Path, version: str | None = None) -> list[dict[str, Any]]:
    if not experiment_dir.exists():
        raise QStudyCliError(f"Experiment not found: {experiment_dir}")

    version_files = discover_version_files(experiment_dir)
    version_files = _select_version_files(version_files, version)
    rows: list[dict[str, Any]] = []
    with _prepend_sys_path(experiment_dir):
        for version_file in version_files:
            version_stem = version_file.stem
            module_name = f"qstudy_experiment_{experiment_dir.name}_{version_stem}"
            module = _load_module(version_file, module_name)
            run_study = getattr(module, "run_study", None)
            if not callable(run_study):
                raise QStudyCliError(f"Missing run_study() in {version_file}")

            metrics = run_study()
            if not isinstance(metrics, dict):
                raise QStudyCliError(f"run_study() in {version_file} must return a dict")

            run_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            write_out_artifact(experiment_dir, version_stem, metrics, run_at)

            row = {"version": version_stem, "run_at": run_at, "metrics": metrics}
            rows.append(row)

    return rows


def write_out_artifact(
    experiment_dir: Path,
    version_stem: str,
    metrics: dict[str, Any],
    run_at: str,
) -> None:
    """Write raw metrics to out/<timestamp>_<version>.json. Never overwrites."""
    out_dir = experiment_dir / OUT_DIRNAME
    out_dir.mkdir(exist_ok=True)
    ts = run_at.replace("-", "").replace(":", "").replace("T", "T").replace("Z", "")
    filename = f"{ts}_{version_stem}.json"
    payload = {"version": version_stem, "run_at": run_at, "metrics": metrics}
    (out_dir / filename).write_text(
        f"{json.dumps(payload, indent=2, default=str)}\n", encoding="utf-8"
    )


def append_log_entry(
    experiment_dir: Path,
    version: str,
    ancestor: str | None,
    hypothesis: str,
    analysis: str,
    metrics: dict[str, Any],
    run_at: str | None = None,
) -> ExperimentEntry:
    """Append a fully-annotated entry to log.json and return it."""
    if run_at is None:
        run_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = ExperimentEntry(
        version=version,
        ancestor=ancestor,
        hypothesis=hypothesis,
        metrics=metrics,
        analysis=analysis,
        run_at=run_at,
    )

    entries = read_log_entries(experiment_dir)
    entries.append(entry.to_dict())

    log_path = experiment_dir / LOG_FILENAME
    log_path.write_text(f"{json.dumps(entries, indent=2, default=str)}\n", encoding="utf-8")
    return entry


def read_log_entries(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read log.json and return the list of entries (empty list if file missing)."""
    log_path = experiment_dir / LOG_FILENAME
    if not log_path.exists():
        return []

    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QStudyCliError(f"Malformed log JSON in {log_path}: {exc.msg}") from exc

    if payload == []:
        return []
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise QStudyCliError(
            f"Malformed log JSON in {log_path}: expected a list of objects"
        )
    return [dict(row) for row in payload]


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


def render_results_table(entries: list[dict[str, Any]]) -> str:
    """Render a summary table from log.json entries.

    Each entry has a nested ``metrics`` dict. Columns are drawn from
    DEFAULT_LOG_COLUMNS using dotted paths (e.g. ``metrics.net_sharpe``).
    """
    if not entries:
        return "No results have been recorded yet."

    # Flatten nested metrics for display only
    flat_rows = []
    for entry in entries:
        flat: dict[str, Any] = {"version": entry.get("version"), "ancestor": entry.get("ancestor")}
        for k, v in (entry.get("metrics") or {}).items():
            flat[f"metrics.{k}"] = v
        flat_rows.append(flat)

    columns = [
        col
        for col in DEFAULT_LOG_COLUMNS
        if any(row.get(col) not in {None, ""} for row in flat_rows)
    ]
    if "version" not in columns:
        columns.insert(0, "version")
    return render_table(flat_rows, columns)


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


def sanitize_version_name(version_name: str) -> str:
    normalized = version_name.strip().lower()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        raise QStudyCliError("Version name must include at least one letter or number.")
    return normalized


def _select_version_files(version_files: list[Path], version: str | None) -> list[Path]:
    if version is None:
        return version_files

    matches = [path for path in version_files if path.stem == version or path.name == version]
    if not matches:
        raise QStudyCliError(f"Study version not found: {version}")
    if len(matches) > 1:
        raise QStudyCliError(f"Study version is ambiguous: {version}")
    return matches


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

from functools import cache

import qstudy as qs
from qstudy.constants import SP500

START_DATE = "2018-01-01"  # TODO: adjust the study start date.
END_DATE = "2024-12-31"  # TODO: adjust the study end date.
BENCHMARK_TICKER = "SPY"  # TODO: change the benchmark if needed.
N_LONG = 25
N_SHORT = 25


@cache
def load_universe():
    \"\"\"Download the default universe for this experiment.

    TODO: replace SP500 with a different universe or a cached dataset if needed.
    \"\"\"

    return qs.download(SP500, START_DATE, END_DATE)


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
- `run.py`: execute all top-level `v*.py` files and write `results.json` and `results.csv`
- `iteration_index.json`: append-only metadata for CLI-created study iterations
- `log.md`: experiment notes

Workflow:
1. Edit `shared.py` and `v0.py`.
2. Run `uv run qstudy iterate {name} <version-name>` to create the next version file.
3. Run `python run.py` inside this directory.
4. Inspect `results.json` or use `qstudy show-results {name}`.
"""
