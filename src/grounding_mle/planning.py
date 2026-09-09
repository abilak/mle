from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .io import atomic_write_json, read_json, runtime_manifest, stable_hash
from .schedules import (
    fixed_schedule,
    optimize_fixed_budget,
    random_schedule_bank,
    synthetic_schedule,
    vanishing_external_schedule,
)
from .theory import schedule_statistics


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    experiment: str
    condition: str
    seed: int
    config: dict[str, Any]
    real_counts: tuple[int, ...]
    synthetic_counts: tuple[int, ...]
    law_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "condition": self.condition,
            "seed": self.seed,
            "config": self.config,
            "real_counts": list(self.real_counts),
            "synthetic_counts": list(self.synthetic_counts),
            "law_split": self.law_split,
            "theory": schedule_statistics(self.real_counts, self.synthetic_counts).to_dict(),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PlannedRun":
        return cls(
            run_id=str(row["run_id"]),
            experiment=str(row["experiment"]),
            condition=str(row["condition"]),
            seed=int(row["seed"]),
            config=copy.deepcopy(row["config"]),
            real_counts=tuple(int(x) for x in row["real_counts"]),
            synthetic_counts=tuple(int(x) for x in row["synthetic_counts"]),
            law_split=row.get("law_split"),
        )


def _condition_schedule(
    condition: dict[str, Any], rounds: int, study: dict[str, Any]
) -> tuple[list[int], list[int]]:
    schedule = condition.get("schedule", {})
    kind = str(schedule.get("kind", "uniform"))
    if kind == "explicit":
        real = [int(x) for x in schedule["real"]]
        synthetic = [int(x) for x in schedule["synthetic"]]
    elif kind in {"selected_worst", "selected_schedule"}:
        selection_path = Path(str(schedule.get("file", "results/llm/selected_confirmation_schedule.json")))
        if selection_path.exists():
            selection = read_json(selection_path)
            real = [int(x) for x in selection["real_counts"]]
            synthetic = [int(x) for x in selection["synthetic_counts"]]
        else:
            fallback = str(schedule.get("fallback_kind", "front_loaded"))
            synthetic = synthetic_schedule(
                rounds, schedule.get("synthetic_per_round", study["synthetic_per_round"])
            )
            real = fixed_schedule(
                fallback,
                rounds,
                int(schedule.get("total_real", study.get("total_real", 0))),
                decay=float(schedule.get("decay", 0.8)),
            )
    elif kind == "all_real":
        batch = int(schedule.get("batch_per_round", study["synthetic_per_round"]))
        real, synthetic = [batch] * rounds, [0] * rounds
    elif kind == "all_synthetic":
        batch = int(schedule.get("batch_per_round", study["synthetic_per_round"]))
        real, synthetic = [0] * rounds, [batch] * rounds
    elif kind == "vanishing":
        real, synthetic = vanishing_external_schedule(
            rounds=rounds,
            synthetic_scale=float(schedule.get("synthetic_scale", 32)),
            real_scale=float(schedule.get("real_scale", 16)),
            log_power=float(schedule["log_power"]),
        )
    else:
        synthetic = synthetic_schedule(
            rounds, schedule.get("synthetic_per_round", study["synthetic_per_round"])
        )
        total_real = int(schedule.get("total_real", study.get("total_real", 0)))
        if kind == "theory_optimized":
            bias_scale = float(schedule.get("bias_scale", 1.0))
            noise_scale = float(schedule.get("noise_scale", 1.0))
            calibration_file = schedule.get("calibration_file", study.get("calibration_file"))
            if calibration_file and Path(str(calibration_file)).exists():
                calibration = read_json(str(calibration_file))
                coefficients = calibration.get("theory_coefficients", {})
                if coefficients:
                    bias_scale = float(coefficients["Q_T_squared"])
                    noise_scale = float(coefficients["Q_T_squared_A_T"])
            optimized = optimize_fixed_budget(
                synthetic,
                total_real,
                bias_scale=bias_scale,
                noise_scale=noise_scale,
                seed=int(schedule.get("optimizer_seed", 0)),
            )
            real = list(optimized.real_counts)
        else:
            options = {key: value for key, value in schedule.items() if key not in {
                "kind", "synthetic_per_round", "total_real"
            }}
            real = fixed_schedule(kind, rounds, total_real, **options)
    if len(real) != rounds or len(synthetic) != rounds:
        raise ValueError(f"Condition {condition.get('name')} has the wrong number of rounds")
    return real, synthetic


def plan_config(config: dict[str, Any]) -> list[PlannedRun]:
    study = config["study"]
    rounds = int(study["rounds"])
    experiment = str(study["name"])
    conditions: list[dict[str, Any]] = []
    generator = study.get("condition_generator")
    if generator:
        generator_kind = generator.get("kind")
        if generator_kind == "schedule_bank":
            synthetic = synthetic_schedule(rounds, study["synthetic_per_round"])
            for item in random_schedule_bank(
                rounds=rounds,
                total_real=int(study["total_real"]),
                synthetic_counts=synthetic,
                count=int(generator.get("count", 40)),
                seed=int(generator.get("seed", 20260906)),
            ):
                conditions.append(
                    {
                        "name": item["name"],
                        "schedule": {
                            "kind": "explicit",
                            "real": item["real"],
                            "synthetic": item["synthetic"],
                        },
                        "law_split": item["law_split"],
                    }
                )
        elif generator_kind == "verification_grid":
            for mode in generator["verification_modes"]:
                for budget in generator["budgets"]:
                    conditions.append(
                        {
                            "name": f"{mode}_b{int(budget)}",
                            "schedule": {
                                "kind": "uniform",
                                "total_real": int(budget),
                            },
                            "overrides": {"verification": {"mode": str(mode)}},
                        }
                    )
            for budget in generator["budgets"]:
                conditions.append(
                    {
                        "name": f"theory_controlled_b{int(budget)}",
                        "schedule": {
                            "kind": "theory_optimized",
                            "total_real": int(budget),
                        },
                        "overrides": {
                            "verification": {
                                "mode": str(generator.get("theory_verification", "external_tests"))
                            }
                        },
                    }
                )
        else:
            raise ValueError(f"Unknown condition generator: {generator_kind}")
    else:
        conditions = copy.deepcopy(study.get("conditions", []))
    if not conditions:
        raise ValueError("The study must define conditions or a condition_generator")

    planned: list[PlannedRun] = []
    for condition in conditions:
        name = str(condition["name"])
        real, synthetic = _condition_schedule(condition, rounds, study)
        for seed in (int(x) for x in study["seeds"]):
            run_config = copy.deepcopy(config)
            overrides = condition.get("overrides", {})
            for section, values in overrides.items():
                if not isinstance(values, dict):
                    run_config[section] = values
                else:
                    run_config.setdefault(section, {}).update(values)
            identity = {
                "experiment": experiment,
                "condition": name,
                "seed": seed,
                "real": real,
                "synthetic": synthetic,
                "model": run_config["model"],
                "training": run_config["training"],
            }
            planned.append(
                PlannedRun(
                    run_id=f"{experiment}-{name}-s{seed}-{stable_hash(identity, 10)}",
                    experiment=experiment,
                    condition=name,
                    seed=seed,
                    config=run_config,
                    real_counts=tuple(real),
                    synthetic_counts=tuple(synthetic),
                    law_split=condition.get("law_split"),
                )
            )
    return planned


def plan_files(config_paths: Iterable[str | Path], output: str | Path) -> dict[str, Any]:
    runs: list[PlannedRun] = []
    for path in config_paths:
        runs.extend(plan_config(load_config(path)))
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Generated duplicate run IDs")
    plan = {"runtime": runtime_manifest(), "runs": [run.to_dict() for run in runs]}
    atomic_write_json(output, plan)
    return plan


def select_runs(
    runs: Iterable[PlannedRun],
    condition: str | None = None,
    seed: int | None = None,
) -> list[PlannedRun]:
    selected = [
        run
        for run in runs
        if (condition is None or run.condition == condition)
        and (seed is None or run.seed == seed)
    ]
    if not selected:
        raise ValueError("No planned runs matched the selection")
    return selected
