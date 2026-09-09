from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import analyze_runs
from .config import load_config
from .datasets import prepare_all_data, prepare_code_data, prepare_math_data
from .io import read_json
from .planning import PlannedRun, plan_files
from .preflight import run_runtime_preflight
from .runner import run_planned
from .schedules import minimum_budget_for_target, optimize_fixed_budget
from .theory_experiments import run_theory_suite


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _config_list(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.config]
    if args.manifest:
        manifest = Path(args.manifest)
        paths.extend(
            (manifest.parent / line.strip()).resolve()
            if not Path(line.strip()).is_absolute()
            else Path(line.strip())
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not paths:
        raise ValueError("Pass at least one --config or --manifest")
    return paths


def command_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.domain == "code":
        result = prepare_code_data(config)
    elif args.domain == "math":
        result = prepare_math_data(config)
    else:
        result = prepare_all_data(config)
    _print(result)


def command_plan(args: argparse.Namespace) -> None:
    result = plan_files(_config_list(args), args.output)
    _print({"output": str(Path(args.output).resolve()), "runs": len(result["runs"])})


def _selected_plan_runs(args: argparse.Namespace) -> list[PlannedRun]:
    payload = read_json(args.plan)
    runs = [PlannedRun.from_dict(row) for row in payload["runs"]]
    if args.index is not None:
        if args.index < 0 or args.index >= len(runs):
            raise IndexError(f"Run index {args.index} outside [0, {len(runs)})")
        runs = [runs[args.index]]
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
    if args.condition:
        runs = [run for run in runs if run.condition == args.condition]
    if args.seed is not None:
        runs = [run for run in runs if run.seed == args.seed]
    if not runs:
        raise ValueError("No plan entries match the selection")
    if len(runs) > 1 and not args.all:
        raise ValueError("Selection matches multiple runs; pass --all or narrow the selection")
    return runs


def command_run(args: argparse.Namespace) -> None:
    summaries = []
    for run in _selected_plan_runs(args):
        state = run_planned(run, args.output_root, resume=not args.no_resume)
        summaries.append(
            {
                "run_id": run.run_id,
                "status": state["status"],
                "rounds": len(state.get("actual_real", [])),
            }
        )
    _print(summaries)


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _print({"valid": True, "path": config["_meta"]["config_path"]})


def command_controller(args: argparse.Namespace) -> None:
    synthetic = [args.synthetic_per_round] * args.rounds
    if args.target_risk is None:
        result = optimize_fixed_budget(
            synthetic,
            args.budget,
            bias_scale=args.bias_scale,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )
    else:
        result = minimum_budget_for_target(
            synthetic,
            args.target_risk,
            args.budget,
            bias_scale=args.bias_scale,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )
    _print(
        {
            "real_counts": result.real_counts,
            "total_real": sum(result.real_counts),
            "risk": result.risk,
            "success": result.success,
            "message": result.message,
        }
    )


def command_preflight(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _print(run_runtime_preflight(config, args.output, check_model=not args.skip_model))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounding-mle",
        description="Experiments for external grounding in recursive self-training",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Download, clean, split, and fingerprint datasets")
    prepare.add_argument("--config", default="configs/base_llm.yaml")
    prepare.add_argument("--domain", choices=["code", "math", "all"], default="all")
    prepare.set_defaults(func=command_prepare)

    plan = subparsers.add_parser("plan", help="Materialize all conditions, schedules, and seeds")
    plan.add_argument("--config", action="append", default=[])
    plan.add_argument("--manifest")
    plan.add_argument("--output", default="runs/full_plan.json")
    plan.set_defaults(func=command_plan)

    run = subparsers.add_parser("run", help="Execute one or more entries from a materialized plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--output-root", default="runs")
    run.add_argument("--index", type=int)
    run.add_argument("--run-id")
    run.add_argument("--condition")
    run.add_argument("--seed", type=int)
    run.add_argument("--all", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=command_run)

    theory = subparsers.add_parser("theory", help="Run the paper-aligned Monte Carlo suite")
    theory.add_argument("--profile", choices=["quick", "full"], default="quick")
    theory.add_argument("--output", default="results/theory")
    theory.set_defaults(func=lambda args: _print(run_theory_suite(args.output, args.profile)))

    analyze = subparsers.add_parser("analyze", help="Aggregate runs, test the schedule law, and make figures")
    analyze.add_argument("--runs", default="runs")
    analyze.add_argument("--output", default="results/llm")
    analyze.set_defaults(func=lambda args: _print(analyze_runs(args.runs, args.output)))

    validate = subparsers.add_parser("validate-config", help="Validate a study configuration")
    validate.add_argument("config")
    validate.set_defaults(func=command_validate)

    controller = subparsers.add_parser("controller", help="Optimize a grounding schedule")
    controller.add_argument("--rounds", type=int, default=10)
    controller.add_argument("--synthetic-per-round", type=int, default=512)
    controller.add_argument("--budget", type=int, default=1024, help="Budget or maximum budget")
    controller.add_argument("--target-risk", type=float)
    controller.add_argument("--bias-scale", type=float, default=1.0)
    controller.add_argument("--noise-scale", type=float, default=1.0)
    controller.add_argument("--seed", type=int, default=20260906)
    controller.set_defaults(func=command_controller)

    preflight = subparsers.add_parser(
        "preflight", help="Exercise CUDA, model training, Docker, and offline EvalPlus integration"
    )
    preflight.add_argument("--config", default="configs/base_llm.yaml")
    preflight.add_argument("--output", default="results/preflight.json")
    preflight.add_argument("--skip-model", action="store_true")
    preflight.set_defaults(func=command_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError, IndexError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
