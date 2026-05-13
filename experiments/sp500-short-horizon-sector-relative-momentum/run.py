"""Run all v*.py studies and write metrics to results.csv."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"


def discover_versions() -> list[str]:
    paths = sorted(HERE.glob("v*.py"))
    return [p.stem for p in paths]


def load_study(version: str):
    path = HERE / f"{version}.py"
    spec = importlib.util.spec_from_file_location(version, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.study


if __name__ == "__main__":
    versions = discover_versions()
    print(f"Found versions: {versions}")

    rows = []
    for version in versions:
        print(f"\n--- {version} ---")
        try:
            study = load_study(version)
            metrics = study.metrics_dict()
            metrics["version"] = version
            rows.append(metrics)
            print(json.dumps(metrics, default=str, sort_keys=True))
        except Exception as exc:
            print(f"ERROR in {version}: {exc}")

    if not rows:
        print("No results to write.")
    else:
        df = pd.DataFrame(rows).set_index("version")
        df.transpose().to_csv(CSV_PATH)

        preferred_cols = [
            "sharpe",
            "ann_return",
            "ann_vol",
            "max_drawdown",
            "max_drawdown_duration",
            "avg_daily_turnover",
            "benchmark_corr",
            "information_ratio",
        ]
        existing_cols = [col for col in preferred_cols if col in df.columns]
        print("\n=== Summary ===")
        print(df[existing_cols].sort_index().round(4).to_string())
        print(f"\nWrote CSV: {CSV_PATH}")
