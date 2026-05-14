from __future__ import annotations

from typing import Any

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


def render_results_table(entries: list[dict[str, Any]]) -> str:
    """Render a summary table from log.json entries.

    Each entry has a nested ``metrics`` dict. Columns are drawn from
    DEFAULT_LOG_COLUMNS using dotted paths (e.g. ``metrics.net_sharpe``).
    """
    if not entries:
        return "No results have been recorded yet."

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
