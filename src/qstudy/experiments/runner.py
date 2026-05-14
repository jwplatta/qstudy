from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qstudy.experiments.errors import QStudyCliError
from qstudy.experiments.log import write_out_artifact
from qstudy.experiments.scaffold import discover_version_files


def run_experiment(experiment_dir: Path, version: str | None = None) -> list[dict[str, Any]]:
    """Run one or all version files in an experiment directory.

    Returns a list of rows, each with ``version``, ``run_at``, and ``metrics``.
    Also writes a timestamped artifact to ``out/`` for each version run.
    """
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

            rows.append({"version": version_stem, "run_at": run_at, "metrics": metrics})

    return rows


def _select_version_files(version_files: list[Path], version: str | None) -> list[Path]:
    if version is None:
        return version_files

    matches = [path for path in version_files if path.stem == version or path.name == version]
    if not matches:
        raise QStudyCliError(f"Study version not found: {version}")
    if len(matches) > 1:
        raise QStudyCliError(f"Study version is ambiguous: {version}")
    return matches


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
