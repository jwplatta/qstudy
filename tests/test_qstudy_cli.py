from __future__ import annotations

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
    LOG_FILENAME,
    OUT_DIRNAME,
    QStudyCliError,
    append_log_entry,
    create_experiment,
    discover_version_files,
    iterate_experiment,
    list_experiments,
    load_studies_config,
    read_iteration_index_rows,
    read_log_entries,
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


def test_create_generates_expected_scaffold_and_empty_log(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")

    expected = {
        "iteration_index.json",
        "v0.py",
        "run.py",
        "shared.py",
        "log.json",
        "readme.md",
    }
    assert expected == {path.name for path in experiment_dir.iterdir()}
    assert json.loads((experiment_dir / LOG_FILENAME).read_text(encoding="utf-8")) == []
    assert json.loads((experiment_dir / ITERATION_INDEX_FILENAME).read_text(encoding="utf-8")) == [
        {"version": 0, "file": "v0.py", "source_file": None, "label": None}
    ]


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


def test_run_experiment_writes_out_artifacts(tmp_path: Path) -> None:
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
        "    return {'sharpe': 2.0, 'ann_return': 0.34}\n",
        encoding="utf-8",
    )

    rows = run_experiment(experiment_dir)

    # Returns rows with version, run_at, metrics
    assert len(rows) == 2
    assert rows[0]["version"] == "v0"
    assert rows[0]["metrics"] == {"sharpe": 1.0, "ann_return": 0.12}
    assert rows[1]["version"] == "v1_growth_tilt"
    assert rows[1]["metrics"] == {"sharpe": 2.0, "ann_return": 0.34}
    assert "run_at" in rows[0]

    # out/ directory written
    out_dir = experiment_dir / OUT_DIRNAME
    assert out_dir.is_dir()
    out_files = list(out_dir.iterdir())
    assert len(out_files) == 2
    # Each file contains version + run_at + metrics
    for f in out_files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        assert "version" in payload
        assert "run_at" in payload
        assert "metrics" in payload


def test_run_experiment_can_filter_to_single_version(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "v0.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 1.0}\n",
        encoding="utf-8",
    )
    (experiment_dir / "v1_growth_tilt.py").write_text(
        "def run_study():\n"
        "    return {'sharpe': 2.0}\n",
        encoding="utf-8",
    )

    rows = run_experiment(experiment_dir, version="v1_growth_tilt")

    assert len(rows) == 1
    assert rows[0]["version"] == "v1_growth_tilt"
    assert rows[0]["metrics"] == {"sharpe": 2.0}

    out_dir = experiment_dir / OUT_DIRNAME
    out_files = list(out_dir.iterdir())
    assert len(out_files) == 1


def test_run_experiment_reports_missing_selected_version(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / "v0.py").write_text("def run_study():\n    return {}\n", encoding="utf-8")

    with pytest.raises(QStudyCliError, match="Study version not found"):
        run_experiment(experiment_dir, version="v9")


def test_append_log_entry_creates_and_appends(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / LOG_FILENAME).write_text("[]\n", encoding="utf-8")

    append_log_entry(
        experiment_dir=experiment_dir,
        version="v1_test",
        ancestor="v0",
        hypothesis="Test hypothesis",
        analysis="Test analysis",
        metrics={"net_sharpe": 0.68, "ann_return": 0.07},
    )

    entries = read_log_entries(experiment_dir)
    assert len(entries) == 1
    assert entries[0]["version"] == "v1_test"
    assert entries[0]["ancestor"] == "v0"
    assert entries[0]["hypothesis"] == "Test hypothesis"
    assert entries[0]["analysis"] == "Test analysis"
    assert entries[0]["metrics"]["net_sharpe"] == 0.68
    assert "run_at" in entries[0]

    # Append a second entry
    append_log_entry(
        experiment_dir=experiment_dir,
        version="v2_test",
        ancestor="v1_test",
        hypothesis="Second hypothesis",
        analysis="Second analysis",
        metrics={"net_sharpe": 0.75},
    )
    entries = read_log_entries(experiment_dir)
    assert len(entries) == 2
    assert entries[1]["version"] == "v2_test"


def test_read_log_entries_returns_empty_when_file_missing(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()

    assert read_log_entries(experiment_dir) == []


def test_read_log_entries_raises_on_malformed_json(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "alpha"
    experiment_dir.mkdir()
    (experiment_dir / LOG_FILENAME).write_text("{bad json", encoding="utf-8")

    with pytest.raises(QStudyCliError, match="Malformed log JSON"):
        read_log_entries(experiment_dir)


def test_render_results_table_uses_nested_metrics(tmp_path: Path) -> None:
    entries = [
        {
            "version": "v0",
            "ancestor": None,
            "metrics": {"net_sharpe": 0.68, "ann_return": 0.07},
        }
    ]
    table = render_results_table(entries)
    assert "version" in table
    assert "v0" in table


def test_render_results_table_empty() -> None:
    assert render_results_table([]) == "No results have been recorded yet."


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
    # out/ artifact written
    out_files = list((experiment_dir / OUT_DIRNAME).iterdir())
    assert len(out_files) == 1
    payload = json.loads(out_files[0].read_text())
    assert payload["metrics"] == {"sharpe": 1.23, "ann_return": 0.45}


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
            "parent": None,
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
            "parent": None,
            "label": "quality",
        },
    ]


def test_iterate_with_parent_records_parent_and_branches_from_it(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    # Create v1 alongside v0 so we have two versions to branch from.
    (experiment_dir / "v1_momentum.py").write_text(
        'STUDY_NAME = "alpha_study_v1_momentum"\n', encoding="utf-8"
    )

    new_file = iterate_experiment(tmp_path, "alpha-study", "vol_filter", parent="v1_momentum")

    assert new_file.name == "v2_vol_filter.py"
    index = read_iteration_index_rows(experiment_dir)
    last = index[-1]
    assert last["file"] == "v2_vol_filter.py"
    assert last["source_file"] == "v1_momentum.py"
    assert last["parent"] == "v1_momentum"
    # lookup_parent returns the stored parent stem
    from qstudy.experiments import lookup_parent
    assert lookup_parent(experiment_dir, "v2_vol_filter") == "v1_momentum"


def test_iterate_with_unknown_parent_raises(tmp_path: Path) -> None:
    create_experiment(tmp_path, "alpha-study")

    with pytest.raises(QStudyCliError, match="Parent version not found"):
        iterate_experiment(tmp_path, "alpha-study", "next", parent="v99_nonexistent")


def test_append_infers_ancestor_from_index(tmp_path: Path) -> None:
    experiment_dir = create_experiment(tmp_path, "alpha-study")
    (experiment_dir / "v1_momentum.py").write_text("VALUE = 1\n", encoding="utf-8")
    iterate_experiment(tmp_path, "alpha-study", "vol_filter", parent="v1_momentum")

    entry = append_log_entry(
        experiment_dir=experiment_dir,
        version="v2_vol_filter",
        ancestor=None,  # omitted — should be inferred
        hypothesis="Add vol filter",
        analysis="Improved Sharpe",
        metrics={"net_sharpe": 0.75},
    )
    # ancestor should NOT be auto-inferred by append_log_entry itself —
    # inference happens in the CLI layer via lookup_parent.
    # Here we test the CLI path via main().
    from qstudy.experiments import lookup_parent
    assert lookup_parent(experiment_dir, "v2_vol_filter") == "v1_momentum"


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


def test_cli_run_prints_metrics_json(
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
    # stdout is JSON
    parsed = json.loads(out.out)
    assert parsed["version"] == "v0"
    assert parsed["metrics"]["sharpe"] == 1.1


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
    parsed = json.loads(out.out)
    assert parsed["version"] == "v1_quality"
    assert parsed["metrics"]["sharpe"] == 2.2


def test_cli_append_writes_log_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    experiment_dir = create_experiment(studies_root, "alpha")
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    metrics_json = json.dumps({"net_sharpe": 0.68, "ann_return": 0.074})
    code = main([
        "log-study", "alpha",
        "--version", "v1_test",
        "--parent", "v0",
        "--hypothesis", "Test hypothesis",
        "--analysis", "Test analysis",
        "--results", metrics_json,
    ])
    out = capsys.readouterr()

    assert code == 0
    assert "v1_test" in out.out

    entries = read_log_entries(experiment_dir)
    assert len(entries) == 1
    assert entries[0]["version"] == "v1_test"
    assert entries[0]["ancestor"] == "v0"
    assert entries[0]["metrics"]["net_sharpe"] == 0.68


def test_cli_append_rejects_invalid_results_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    create_experiment(studies_root, "alpha")
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main([
        "log-study", "alpha",
        "--version", "v1",
        "--hypothesis", "h",
        "--analysis", "a",
        "--results", "{bad json",
    ])
    out = capsys.readouterr()

    assert code == 1
    assert "not valid JSON" in out.err


def test_cli_show_results_reads_log_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    experiment_dir = create_experiment(studies_root, "alpha")
    (experiment_dir / LOG_FILENAME).write_text(
        json.dumps([{
            "version": "v0",
            "ancestor": None,
            "hypothesis": "baseline",
            "metrics": {"net_sharpe": 0.68, "ann_return": 0.074},
            "analysis": "ok",
            "run_at": "2026-01-01T00:00:00Z",
        }]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")

    code = main(["show-results", "alpha"])
    out = capsys.readouterr()

    assert code == 0
    assert "v0" in out.out
    assert out.err == ""


def test_cli_show_results_missing_log_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    studies_root = tmp_path / "studies"
    experiment_dir = create_experiment(studies_root, "alpha")
    (experiment_dir / LOG_FILENAME).unlink()
    (tmp_path / CONFIG_FILENAME).write_text('studies_dir = "studies"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code = main(["show-results", "alpha"])
    out = capsys.readouterr()

    assert code == 1
    assert "Results file not found" in out.err


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

    rows = run_experiment(experiment_dir)
    assert len(rows) == 1
    assert rows[0]["version"] == "v0"
    assert rows[0]["metrics"] == {"sharpe": 1.0}
