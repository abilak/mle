from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import deep_merge
from .generative_schedules import materialize_generative_schedule
from .io import atomic_write_json, runtime_manifest, stable_hash
from .theory import schedule_statistics


SUPPORTED_GENERATIVE_BACKENDS = {
    "teacher_lm",
    "real_corpus_lm",
    "flow_teacher",
    "flow_mode_recovery",
    "diffusion_mode_recovery",
    "analytic_markov",
}


@dataclass(frozen=True)
class GenerativePlannedRun:
    run_id: str
    experiment: str
    condition: str
    seed: int
    backend: str
    config: dict[str, Any]
    real_counts: tuple[int, ...]
    synthetic_counts: tuple[int, ...]
    schedule_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "condition": self.condition,
            "seed": self.seed,
            "backend": self.backend,
            "config": self.config,
            "real_counts": list(self.real_counts),
            "synthetic_counts": list(self.synthetic_counts),
            "schedule_metadata": self.schedule_metadata,
            "theory": schedule_statistics(
                self.real_counts, self.synthetic_counts
            ).to_dict(),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "GenerativePlannedRun":
        return cls(
            run_id=str(row["run_id"]),
            experiment=str(row["experiment"]),
            condition=str(row["condition"]),
            seed=int(row["seed"]),
            backend=str(row["backend"]),
            config=copy.deepcopy(dict(row["config"])),
            real_counts=tuple(int(value) for value in row["real_counts"]),
            synthetic_counts=tuple(int(value) for value in row["synthetic_counts"]),
            schedule_metadata=copy.deepcopy(dict(row.get("schedule_metadata", {}))),
        )


def load_generative_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Generative configuration must be a YAML mapping")
    config.setdefault("_meta", {})["config_path"] = str(source)
    validate_generative_config(config)
    return config


def validate_generative_config(config: Mapping[str, Any]) -> None:
    for section in ("project", "data", "defaults", "studies"):
        if section not in config:
            raise ValueError(f"Missing generative configuration section: {section}")
    if not isinstance(config["studies"], list) or not config["studies"]:
        raise ValueError("studies must be a non-empty list")
    names: set[str] = set()
    for study in config["studies"]:
        name = str(study.get("name", ""))
        if not name or name in names:
            raise ValueError("Every generative study needs a unique non-empty name")
        names.add(name)
        experiment = str(study.get("experiment", name))
        if not experiment:
            raise ValueError(f"Study {name} must have a non-empty experiment label")
        backend = str(study.get("backend", config["defaults"].get("backend", "")))
        if backend not in SUPPORTED_GENERATIVE_BACKENDS:
            raise ValueError(f"Unsupported generative backend for {name}: {backend}")
        rounds = int(study.get("rounds", config["defaults"].get("rounds", 0)))
        if rounds <= 0:
            raise ValueError(f"Study {name} must have a positive round count")
        seeds = study.get("seeds", config["defaults"].get("seeds", []))
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError(f"Study {name} must contain distinct integer seeds")
        conditions = study.get("conditions", [])
        if not conditions:
            raise ValueError(f"Study {name} has no conditions")
        condition_names = [str(condition.get("name", "")) for condition in conditions]
        if any(not value for value in condition_names) or len(condition_names) != len(
            set(condition_names)
        ):
            raise ValueError(f"Study {name} must have unique non-empty condition names")
        for condition in conditions:
            if "schedule" not in condition:
                raise ValueError(f"Condition {name}/{condition['name']} has no schedule")
        resolved = _study_defaults(config, study)
        if backend in {"teacher_lm", "real_corpus_lm"}:
            if "lm" not in resolved or "evaluation" not in resolved:
                raise ValueError(f"Study {name} requires lm and evaluation sections")
            for model_name in ("teacher_model", "student_model"):
                model = resolved["lm"].get(model_name)
                if model is None:
                    raise ValueError(f"Study {name} is missing lm.{model_name}")
                if int(model.get("n_embd", 0)) % int(model.get("n_head", 1)):
                    raise ValueError(f"Study {name} has n_embd not divisible by n_head")
        if backend in {"flow_teacher", "flow_mode_recovery", "diffusion_mode_recovery"}:
            if "vision" not in resolved or "evaluation" not in resolved:
                raise ValueError(f"Study {name} requires vision and evaluation sections")


def _study_defaults(config: Mapping[str, Any], study: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deep_merge(config["defaults"], study.get("overrides", {}))
    resolved["data"] = deep_merge(config["data"], study.get("data", {}))
    return resolved


def plan_generative_config(config: Mapping[str, Any]) -> list[GenerativePlannedRun]:
    planned: list[GenerativePlannedRun] = []
    for study in config["studies"]:
        experiment = str(study.get("experiment", study["name"]))
        resolved_study = _study_defaults(config, study)
        rounds = int(study.get("rounds", resolved_study["rounds"]))
        batch_size = int(study.get("batch_size", resolved_study["batch_size"]))
        backend = str(study.get("backend", resolved_study["backend"]))
        seeds = [int(seed) for seed in study.get("seeds", resolved_study["seeds"])]
        for condition in study["conditions"]:
            condition_name = str(condition["name"])
            condition_config = deep_merge(resolved_study, condition.get("overrides", {}))
            condition_config.update(
                {
                    "backend": backend,
                    "rounds": rounds,
                    "batch_size": batch_size,
                    "project": copy.deepcopy(config["project"]),
                }
            )
            schedule = materialize_generative_schedule(
                condition_name,
                condition["schedule"],
                rounds,
                batch_size,
            )
            for seed in seeds:
                identity = {
                    "experiment": experiment,
                    "condition": condition_name,
                    "seed": seed,
                    "backend": backend,
                    "config": condition_config,
                    "real": schedule.real_counts,
                    "synthetic": schedule.synthetic_counts,
                }
                planned.append(
                    GenerativePlannedRun(
                        run_id=(
                            f"gen-{experiment}-{condition_name}-s{seed}-"
                            f"{stable_hash(identity, 10)}"
                        ),
                        experiment=experiment,
                        condition=condition_name,
                        seed=seed,
                        backend=backend,
                        config=copy.deepcopy(condition_config),
                        real_counts=schedule.real_counts,
                        synthetic_counts=schedule.synthetic_counts,
                        schedule_metadata=schedule.metadata,
                    )
                )
    return planned


def plan_generative_file(path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_generative_config(path)
    runs = plan_generative_config(config)
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("The generative plan contains duplicate run IDs")
    payload = {
        "kind": "grounding-mle-generative-plan-v1",
        "source": str(Path(path).resolve()),
        "runtime": runtime_manifest(),
        "runs": [run.to_dict() for run in runs],
    }
    atomic_write_json(output, payload)
    return payload
