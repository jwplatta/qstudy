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
    ITERATION_INDEX_FILENAME,
    QStudyCliError,
    create_experiment,
    discover_version_files,
    iterate_experiment,
    list_experiments,
    load_studies_config,
    read_iteration_index_rows,
    read_results_rows,
    render_results_table,
    run_experiment,
    sanitize_version_name,
)


def test_load_studies_config_prefers_local_config(tmp_path: Path) -> None:
    local_root = tmp_path / "local-studies"
    local_data = tmp_path / "local-data"
    home_root = tmp_path / "home-studies"
    (tmp_path / CONFIG_FILENAME).write_text(
        'studies_dir = "local-studies"\n'
        'data_dir = "local-data"\n',
        encoding="utf-8",
    )
    (tmp_path / "home" / CONFIG_FILENAME).parent.mkdir()
    (tmp_path / "home" / CONFIG_FILENAME).write_text(
        'studies_dir = "home-studies"\n'
        'data_dir = "home-data"\n',
        encoding="utf-8",
    )

    config = load_studies_config(cwd=tmp_path, home=tmp_path / "home")

    assert config.source == tmp_path / CONFIG_FILENAME
    assert config.studies_root == local_root.resolve()
    assert config.data_root == local_data.resolve()
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
    assert config.data_root is None


def test_load_studies_config_falls_back_to_cwd(tmp_path: Path) -> None:
    config = load_studies_config(cwd=tmp_path, home=tmp_path / "home")

    assert config.source is None
    assert config.studies_root == tmp_path.resolve()
    assert config.data_root is None


def test_create_generates_expected_scaffold_and_empty_results(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")

    expected = {
        "iteration_index.json",
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
    assert json.loads((experiment_dir / ITERATION_INDEX_FILENAME).read_text(encoding="utf-8")) == [
        {"version": 0, "file": "v0.py", "source_file": None, "label": None}
    ]

    with (experiment_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["version"]]


def test_load_studies_config_rejects_invalid_config(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text('bad_key = "studies"\n', encoding="utf-8")

    with pytest.raises(QStudyCliError, match="unsupported key"):
        load_studies_config(cwd=tmp_path, home=tmp_path / "home")


def test_load_studies_config_rejects_duplicate_data_dir(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text(
        'studies_dir = "studies"\n'
        'data_dir = "one"\n'
        'data_dir = "two"\n',
        encoding="utf-8",
    )

    with pytest.raises(QStudyCliError, match="duplicate data_dir"):
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


def test_run_experiment_can_filter_to_single_version_and_overwrite_results(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    (experiment_dir / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.0, 'ann_return': 0.12}\n",
        encoding="utf-8",
    )
    (experiment_dir / "v1_growth_tilt.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 2.0, 'ann_return': 0.34}\n",
        encoding="utf-8",
    )

    rows = run_experiment(experiment_dir, version="v1_growth_tilt")

    assert rows == [{"version": "v1_growth_tilt", "sharpe": 2.0, "ann_return": 0.34}]
    assert read_results_rows(experiment_dir) == rows
    csv_text = (experiment_dir / "results.csv").read_text(encoding="utf-8")
    assert "v1_growth_tilt,2.0,0.34" in csv_text
    assert "v0,1.0,0.12" not in csv_text


def test_run_experiment_reports_missing_selected_version(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "v0.py").write_text("def run_study():\n    return {}\n", encoding="utf-8")

    with pytest.raises(QStudyCliError, match="Study version not found"):
        run_experiment(experiment_dir, version="v9")


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


def test_sanitize_version_name_normalizes_common_inputs() -> None:
    assert sanitize_version_name("Volume Confirmed") == "volume_confirmed"
    assert sanitize_version_name("volume-confirmed.py") == "volume_confirmed"


def test_iterate_creates_next_version_and_appends_index(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / "v0.py").write_text(
        '"""v0 - baseline."""\n'
        'STUDY_NAME = "alpha_study_v0"\n'
        "from qstudy import Study\n\n"
        "study = Study(name=\"alpha-study:v0\")\n",
        encoding="utf-8",
    )

    new_file = iterate_experiment(tmp_path, "alpha-study", "Volume Confirmed")

    assert new_file.name == "v1_volume_confirmed.py"
    text = new_file.read_text(encoding="utf-8")
    assert '"""v1_volume_confirmed - baseline."""' in text
    assert 'STUDY_NAME = "alpha_study_v1_volume_confirmed"' in text
    assert 'Study(name="alpha-study:v1_volume_confirmed")' in text
    assert read_iteration_index_rows(experiment_dir) == [
        {"version": 0, "file": "v0.py", "source_file": None, "label": None},
        {
            "version": 1,
            "file": "v1_volume_confirmed.py",
            "source_file": "v0.py",
            "label": "volume_confirmed",
        },
    ]


def test_iterate_uses_highest_existing_version_and_lexicographic_tiebreak(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / "v10_alpha.py").write_text("VALUE = 'alpha'\n", encoding="utf-8")
    (experiment_dir / "v10_beta.py").write_text(
        'STUDY_NAME = "alpha_beta_v10_beta"\n',
        encoding="utf-8",
    )

    new_file = iterate_experiment(tmp_path, "alpha-study", "Next Step")

    assert new_file.name == "v11_next_step.py"
    assert new_file.read_text(encoding="utf-8") == 'STUDY_NAME = "alpha_beta_v11_next_step"\n'


def test_iterate_bootstraps_missing_index_from_existing_versions(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / ITERATION_INDEX_FILENAME).unlink()
    (experiment_dir / "v1_growth.py").write_text("VALUE = 1\n", encoding="utf-8")

    iterate_experiment(tmp_path, "alpha-study", "quality")

    assert read_iteration_index_rows(experiment_dir) == [
        {"version": 0, "file": "v0.py", "source_file": None, "label": None},
        {"version": 1, "file": "v1_growth.py", "source_file": "v0.py", "label": "growth"},
        {
            "version": 2,
            "file": "v2_quality.py",
            "source_file": "v1_growth.py",
            "label": "quality",
        },
    ]


def test_iterate_rejects_invalid_name(tmp_path: Path) -> None:
    create_experiment(tmp_path, "alpha-study")

    with pytest.raises(QStudyCliError, match="Version name must include"):
        iterate_experiment(tmp_path, "alpha-study", "!!!")


def test_iterate_reports_malformed_index(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / ITERATION_INDEX_FILENAME).write_text("{bad json", encoding="utf-8")

    with pytest.raises(QStudyCliError, match="Malformed iteration index JSON"):
        iterate_experiment(tmp_path, "alpha-study", "quality")


def test_cli_iterate_creates_version_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["iterate", "alpha", "quality tilt"])
    out = capsys.readouterr()

    assert code == 0
    assert "Created iteration at" in out.out
    assert (studies_root / "alpha" / "v1_quality_tilt.py").exists()


def test_cli_run_executes_named_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (studies_root / "alpha" / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.1, 'ann_return': 0.2}\n",
        encoding="utf-8",
    )
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["run", "alpha"])
    out = capsys.readouterr()

    assert code == 0
    assert "Ran 1 study version(s)." in out.out
    assert read_results_rows(studies_root / "alpha") == [
        {"version": "v0", "sharpe": 1.1, "ann_return": 0.2}
    ]


def test_cli_run_can_execute_single_selected_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (studies_root / "alpha" / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.1, 'ann_return': 0.2}\n",
        encoding="utf-8",
    )
    (studies_root / "alpha" / "v1_quality.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 2.2, 'ann_return': 0.3}\n",
        encoding="utf-8",
    )
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["run", "alpha", "--version", "v1_quality"])
    out = capsys.readouterr()

    assert code == 0
    assert "Ran 1 study version(s)." in out.out
    assert read_results_rows(studies_root / "alpha") == [
        {"version": "v1_quality", "sharpe": 2.2, "ann_return": 0.3}
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


def test_run_experiment_ignores_iteration_index_file(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.0}\n",
        encoding="utf-8",
    )
    (experiment_dir / ITERATION_INDEX_FILENAME).write_text(
        json.dumps([{"version": 999, "file": "nope.py", "source_file": None, "label": "bad"}]),
        encoding="utf-8",
    )

    assert run_experiment(experiment_dir) == [{"version": "v0", "sharpe": 1.0}]
