from __future__ import annotations

import argparse
import sys

from qstudy.experiments import (
    ConfigError,
    QStudyCliError,
    create_experiment,
    list_experiments,
    load_studies_config,
    read_results_rows,
    render_experiment_list,
    render_results_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qstudy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new experiment scaffold")
    create_parser.add_argument("name", help="Experiment name")

    subparsers.add_parser("list", help="List experiments")

    results_parser = subparsers.add_parser(
        "show-results", help="Show a summary table from an experiment's results.json"
    )
    results_parser.add_argument("name", help="Experiment name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_studies_config()
        if args.command == "create":
            experiment_dir = create_experiment(config.studies_root, args.name)
            print(f"Created experiment at {experiment_dir}")
            return 0

        if args.command == "list":
            print(render_experiment_list(list_experiments(config.studies_root)))
            return 0

        if args.command == "show-results":
            experiment_dir = config.studies_root / args.name
            if not experiment_dir.exists():
                raise QStudyCliError(f"Experiment not found: {experiment_dir}")
            print(render_results_table(read_results_rows(experiment_dir)))
            return 0

        parser.error(f"Unknown command: {args.command}")
    except (ConfigError, QStudyCliError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
