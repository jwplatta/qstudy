from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qstudy.cli import main
from qstudy.experiments import (
    CONFIG_FILENAME,
    QStudyCliError,
    create_experiment,
    discover_version_files,
    list_experiments,
    load_studies_config,
    read_results_rows,
    render_results_table,
    run_experiment,
)


def test_load_studies_config_prefers_local_config(tmp_path: Path) -> None:
    local_root = tmp_path / "local-studies"
    home_root = tmp_path / "home-studies"
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "local-studies"\n', encoding="utf-8")
    (tmp_path / "home" / CONFIG_FILENAME).parent.mkdir()
    (tmp_path / "home" / CONFIG_FILENAME).write_text(
        'studies_dir = "home-studies"\n',
        encoding="utf-8",
    )

    config = load_studies_config(cwd=tmp_path, home=tmp_path / "home")

    assert config.source == tmp_path / CONFIG_FILENAME
    assert config.studies_root == local_root.resolve()
    assert config.studies_root != home_root.resolve()


def test_load_studies_config_uses_global_when_local_missing(tmp_path: Path) -> None:
    cwd = tmp_path / "workspace"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    (home / CONFIG_FILENAME).write_text('studies_dir = "experiments"\n', encoding="utf-8")

    config = load_studies_config(cwd=cwd, home=home)

    assert config.source == home / CONFIG_FILENAME
    assert config.studies_root == (home / "experiments").resolve()


def test_load_studies_config_falls_back_to_cwd(tmp_path: Path) -> None:
    config = load_studies_config(cwd=tmp_path, home=tmp_path / "home")

    assert config.source is None
    assert config.studies_root == tmp_path.resolve()


def test_create_generates_expected_scaffold_and_empty_results(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")

    expected = {
        "v0.py",
        "run.py",
        "shared.py",
        "results.json",
        "results.csv",
        "log.md",
        "readme.md",
    }
    assert expected == {path.name for path in experiment_dir.iterdir()}
    assert json.loads((experiment_dir / "results.json").read_text(encoding="utf-8")) == []

    with (experiment_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["version"]]


def test_load_studies_config_rejects_invalid_config(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text('bad_key = "studies"\n', encoding="utf-8")

    with pytest.raises(QStudyCliError, match="unsupported key"):
        load_studies_config(cwd=tmp_path, home=tmp_path / "home")


def test_create_refuses_to_overwrite_existing_experiment(tmp_path: Path) -> None:
    create_experiment(tmp_path, "alpha-study")

    with pytest.raises(QStudyCliError, match="already exists"):
        create_experiment(tmp_path, "alpha-study")


def test_list_reports_top_level_versions_only(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    (alpha / "v0.py").write_text("", encoding="utf-8")
    (alpha / "v1.py").write_text("", encoding="utf-8")
    (alpha / "v2_equal_sharpe.py").write_text("", encoding="utf-8")
    (alpha / "notes.txt").write_text("", encoding="utf-8")
    (alpha / "nested").mkdir()
    (alpha / "nested" / "v2.py").write_text("", encoding="utf-8")

    beta = tmp_path / "beta"
    beta.mkdir()
    (beta / "v10.py").write_text("", encoding="utf-8")

    assert list_experiments(tmp_path) == [("alpha", 3), ("beta", 1)]


def test_discover_version_files_orders_by_numeric_suffix(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    for name in ("v10.py", "v2_equal_sharpe.py", "v1.py", "v0.py"):
        (experiment_dir / name).write_text("def run_study():\n    return {}\n", encoding="utf-8")

    assert [path.name for path in discover_version_files(experiment_dir)] == [
        "v0.py",
        "v1.py",
        "v2_equal_sharpe.py",
        "v10.py",
    ]


def test_run_experiment_writes_json_and_csv(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    (experiment_dir / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.0, 'ann_return': 0.12}\n",
        encoding="utf-8",
    )
    (experiment_dir / "v1_growth_tilt.py").write_text(
        "from shared import VALUE\n\n"
        "def run_study():\n"
        "    return {'sharpe': 2.0, 'ann_return': 0.34, 'nested': {'x': VALUE}}\n",
        encoding="utf-8",
    )

    rows = run_experiment(experiment_dir)

    assert rows == [
        {"version": "v0", "sharpe": 1.0, "ann_return": 0.12},
        {"version": "v1_growth_tilt", "sharpe": 2.0, "ann_return": 0.34, "nested.x": 1},
    ]
    assert read_results_rows(experiment_dir) == rows

    csv_text = (experiment_dir / "results.csv").read_text(encoding="utf-8")
    assert "version,sharpe,ann_return,nested.x" in csv_text
    assert "v1_growth_tilt,2.0,0.34,1.0" in csv_text


def test_generated_run_py_executes_versions(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "generated-study")
    (experiment_dir / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.23, 'ann_return': 0.45}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [sys.executable, str(experiment_dir / "run.py")],
        cwd=experiment_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "Ran 1 study version(s)." in result.stdout
    assert read_results_rows(experiment_dir) == [
        {"version": "v0", "sharpe": 1.23, "ann_return": 0.45}
    ]


def test_render_results_table_omits_missing_columns() -> None:
    table = render_results_table([{"version": "v0", "sharpe": 1.0}])

    assert "version" in table
    assert "sharpe" in table
    assert "ann_return" not in table


def test_cli_show_results_prints_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (studies_root / "alpha" / "results.json").write_text(
        json.dumps([{"version": "v0", "sharpe": 1.1, "ann_return": 0.2}]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")

    code = main(["show-results", "alpha"])
    out = capsys.readouterr()

    assert code == 0
    assert "version" in out.out
    assert "v0" in out.out
    assert out.err == ""


def test_cli_reports_missing_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["show-results", "missing"])
    out = capsys.readouterr()

    assert code == 1
    assert "Experiment not found" in out.err


def test_cli_reports_missing_results_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (studies_root / "alpha" / "results.json").unlink()
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["show-results", "alpha"])
    out = capsys.readouterr()

    assert code == 1
    assert "Results file not found" in out.err


def test_cli_reports_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (studies_root / "alpha" / "results.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["show-results", "alpha"])
    out = capsys.readouterr()

    assert code == 1
    assert "Malformed results JSON" in out.err


def test_run_experiment_reports_missing_run_study(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "v0.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(QStudyCliError, match="Missing run_study"):
        run_experiment(experiment_dir)
