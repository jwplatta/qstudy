from __future__ import annotations

import argparse
import json
import sys

from qstudy.experiments import (
    CONFIG_FILENAME,
    QStudyCliError,
    ConfigError,
    ExperimentEntry,
    append_log_entry,
    create_experiment,
    iterate_experiment,
    list_experiments,
    load_studies_config,
    read_log_entries,
    render_experiment_list,
    render_results_table,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qstudy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new experiment scaffold")
    create_parser.add_argument("name", help="Experiment name")

    iterate_parser = subparsers.add_parser("iterate", help="Create the next study version file")
    iterate_parser.add_argument("study", help="Experiment name")
    iterate_parser.add_argument("version_name", help="Name for the new version suffix")

    run_parser = subparsers.add_parser("run", help="Run study versions in an experiment")
    run_parser.add_argument("name", help="Experiment name")
    run_parser.add_argument(
        "--version",
        help="Run only a single study version by exact stem or filename",
    )

    subparsers.add_parser("list", help="List experiments")

    results_parser = subparsers.add_parser(
        "show-results", help="Show a summary table from an experiment's log.json"
    )
    results_parser.add_argument("name", help="Experiment name")

    append_parser = subparsers.add_parser(
        "append",
        help="Append an annotated entry (metrics + hypothesis + analysis) to log.json",
    )
    append_parser.add_argument("name", help="Experiment name")
    append_parser.add_argument("--version", required=True, help="Study version stem")
    append_parser.add_argument("--hypothesis", required=True, help="What this version tests")
    append_parser.add_argument(
        "--analysis", required=True, help="1-2 sentence interpretation of results"
    )
    append_parser.add_argument(
        "--results",
        required=True,
        help="Metrics JSON string (output of run_study())",
    )
    append_parser.add_argument(
        "--ancestor",
        default=None,
        help="Parent version stem this iteration branched from (optional)",
    )

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

        if args.command == "iterate":
            version_file = iterate_experiment(config.studies_root, args.study, args.version_name)
            print(f"Created iteration at {version_file}")
            return 0

        if args.command == "run":
            experiment_dir = config.studies_root / args.name
            rows = run_experiment(experiment_dir, version=args.version)
            for row in rows:
                print(json.dumps(row, indent=2, default=str))
            return 0

        if args.command == "list":
            print(render_experiment_list(list_experiments(config.studies_root)))
            return 0

        if args.command == "show-results":
            experiment_dir = config.studies_root / args.name
            if not experiment_dir.exists():
                raise QStudyCliError(f"Experiment not found: {experiment_dir}")
            entries = read_log_entries(experiment_dir)
            if not entries:
                log_path = experiment_dir / "log.json"
                if not log_path.exists():
                    raise QStudyCliError(f"Results file not found: {log_path}")
            print(render_results_table(entries))
            return 0

        if args.command == "append":
            experiment_dir = config.studies_root / args.name
            if not experiment_dir.exists():
                raise QStudyCliError(f"Experiment not found: {experiment_dir}")

            try:
                metrics = json.loads(args.results)
            except json.JSONDecodeError as exc:
                raise QStudyCliError(f"--results is not valid JSON: {exc}") from exc

            if not isinstance(metrics, dict):
                raise QStudyCliError("--results must be a JSON object")

            entry = append_log_entry(
                experiment_dir=experiment_dir,
                version=args.version,
                ancestor=args.ancestor,
                hypothesis=args.hypothesis,
                analysis=args.analysis,
                metrics=metrics,
            )
            print(f"Appended entry for {entry.version} to log.json")
            return 0

        parser.error(f"Unknown command: {args.command}")
    except (ConfigError, QStudyCliError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
