"""qstudy experiments package.

Public API — import from here or directly from submodules.
"""

from __future__ import annotations

from qstudy.experiments.config import (
    CONFIG_FILENAME,
    StudiesConfig,
    load_studies_config,
)
from qstudy.experiments.display import (
    DEFAULT_LOG_COLUMNS,
    flatten_metrics,
    format_cell,
    render_experiment_list,
    render_results_table,
    render_table,
    union_columns,
)
from qstudy.experiments.errors import ConfigError, QStudyCliError
from qstudy.experiments.log import (
    LOG_FILENAME,
    OUT_DIRNAME,
    ExperimentEntry,
    append_log_entry,
    read_log_entries,
    write_out_artifact,
)
from qstudy.experiments.query import (
    METRIC_NAMES,
    render_query_result,
    resolve_metric,
    run_query,
)
from qstudy.experiments.runner import run_experiment
from qstudy.experiments.scaffold import (
    ITERATION_INDEX_FILENAME,
    create_experiment,
    discover_version_files,
    iterate_experiment,
    list_experiments,
    lookup_parent,
    read_iteration_index_rows,
    sanitize_version_name,
    scaffold_files,
    write_iteration_index_rows,
)

__all__ = [
    # config
    "CONFIG_FILENAME",
    "StudiesConfig",
    "load_studies_config",
    # display
    "DEFAULT_LOG_COLUMNS",
    "flatten_metrics",
    "format_cell",
    "render_experiment_list",
    "render_results_table",
    "render_table",
    "union_columns",
    # errors
    "ConfigError",
    "QStudyCliError",
    # log
    "LOG_FILENAME",
    "OUT_DIRNAME",
    "ExperimentEntry",
    "append_log_entry",
    "read_log_entries",
    "write_out_artifact",
    # query
    "METRIC_NAMES",
    "render_query_result",
    "resolve_metric",
    "run_query",
    # runner
    "run_experiment",
    # scaffold
    "ITERATION_INDEX_FILENAME",
    "create_experiment",
    "discover_version_files",
    "iterate_experiment",
    "list_experiments",
    "lookup_parent",
    "read_iteration_index_rows",
    "sanitize_version_name",
    "scaffold_files",
    "write_iteration_index_rows",
]
