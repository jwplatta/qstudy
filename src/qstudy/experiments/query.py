from __future__ import annotations

from typing import Any

from qstudy.experiments.display import render_table
from qstudy.experiments.errors import QStudyCliError

METRIC_NAMES: dict[str, str] = {
    "sharpe": "sharpe",
    "net-sharpe": "net_sharpe",
    "gross-sharpe": "gross_sharpe",
    "return": "ann_return",
    "vol": "ann_vol",
    "drawdown": "max_drawdown",
    "turnover": "avg_daily_turnover",
    "bench-corr": "benchmark_corr",
    "ir": "information_ratio",
    "benchmark-sharpe": "benchmark_sharpe",
}


def resolve_metric(name: str) -> str:
    """Map a CLI metric name to the corresponding log.json field name.

    Raises QStudyCliError if the name is not recognized.
    """
    field = METRIC_NAMES.get(name)
    if field is None:
        valid = ", ".join(sorted(METRIC_NAMES))
        raise QStudyCliError(f"Unknown metric {name!r}. Valid metrics: {valid}")
    return field


def run_query(
    entries: list[dict[str, Any]],
    field: str,
    ascending: bool,
) -> list[dict[str, Any]]:
    """Sort log entries by a metric field and return flat row dicts.

    Entries that do not contain the field are placed last.
    """
    rows: list[dict[str, Any]] = []
    for entry in entries:
        value = (entry.get("metrics") or {}).get(field)
        rows.append(
            {
                "version": entry.get("version"),
                "ancestor": entry.get("ancestor"),
                f"metrics.{field}": value,
            }
        )

    present = [r for r in rows if r[f"metrics.{field}"] is not None]
    missing = [r for r in rows if r[f"metrics.{field}"] is None]
    present.sort(key=lambda r: r[f"metrics.{field}"], reverse=not ascending)
    return present + missing


def render_query_result(rows: list[dict[str, Any]], field: str, ascending: bool) -> str:
    """Render sorted query results as a table with a summary header.

    Returns 'No results have been recorded yet.' if rows is empty.
    """
    if not rows:
        return "No results have been recorded yet."

    direction = "ascending" if ascending else "descending"
    header = f"Sorted by metrics.{field} ({direction})\n"
    columns = ["version", "ancestor", f"metrics.{field}"]
    table = render_table(rows, columns)
    return header + table
