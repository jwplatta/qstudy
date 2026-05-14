from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qstudy.experiments.errors import QStudyCliError

LOG_FILENAME = "log.json"
OUT_DIRNAME = "out"


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
