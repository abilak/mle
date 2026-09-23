from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOWER_IS_BETTER = (
    "primary_value",
    "class_kl_to_uniform",
    "feature_frechet_distance",
)
HIGHER_IS_BETTER = ("class_entropy",)


def _finite_mean(table: pd.DataFrame, column: str) -> float:
    if column not in table:
        raise ValueError(f"Missing required metric column: {column}")
    values = pd.to_numeric(table[column], errors="coerce")
    if not values.map(math.isfinite).all():
        raise ValueError(f"Metric {column} contains a missing or non-finite value")
    return float(values.mean())


def load_repair_specification(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    specification = config.get("repair_analysis")
    if not isinstance(specification, dict):
        raise ValueError(f"No repair_analysis mapping found in {source}")
    if not specification.get("posthoc_exploratory"):
        raise ValueError("The vision repair must be labeled post-hoc exploratory")
    return specification


def evaluate_repair(
    results: str | Path, specification: dict[str, Any]
) -> dict[str, Any]:
    results_path = Path(results)
    table = pd.read_csv(results_path / "final_metrics.csv")
    planned_seeds = {int(value) for value in specification["planned_seeds"]}
    positive = str(specification["positive_condition"])
    negative = str(specification["negative_condition"])
    experiments = ("flow_mode_recovery", "diffusion_mode_recovery")
    summaries: list[dict[str, Any]] = []
    overall_pass = True

    for experiment in experiments:
        experiment_table = table[table["experiment"] == experiment]
        groups = {
            condition: experiment_table[
                (experiment_table["condition"] == condition)
                & experiment_table["seed"].isin(planned_seeds)
            ].copy()
            for condition in (positive, negative)
        }
        for condition, group in groups.items():
            observed = set(int(value) for value in group["seed"])
            if observed != planned_seeds or group["seed"].duplicated().any():
                raise ValueError(
                    f"{experiment}/{condition} does not contain exactly the planned seeds"
                )

        means: dict[str, dict[str, float]] = {}
        relative_checks: dict[str, bool] = {}
        for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
            means[metric] = {
                condition: _finite_mean(groups[condition], metric)
                for condition in (positive, negative)
            }
            if metric in LOWER_IS_BETTER:
                relative_checks[metric] = bool(
                    means[metric][positive] < means[metric][negative]
                )
            else:
                relative_checks[metric] = bool(
                    means[metric][positive] > means[metric][negative]
                )

        gates = specification["absolute_gates"]
        absolute_checks = {
            "all_real_primary_mean": bool(
                means["primary_value"][positive]
                <= float(gates["all_real_primary_mean_max"])
            ),
            "all_real_primary_each_seed": bool(
                pd.to_numeric(groups[positive]["primary_value"]).max()
                <= float(gates["all_real_primary_seed_max"])
            ),
            "all_real_class_kl_mean": bool(
                means["class_kl_to_uniform"][positive]
                <= float(gates["all_real_class_kl_mean_max"])
            ),
            "all_real_class_entropy_mean": bool(
                means["class_entropy"][positive]
                >= float(gates["all_real_class_entropy_mean_min"])
            ),
        }
        paired_primary = groups[positive].set_index("seed")["primary_value"].astype(float)
        paired_negative = groups[negative].set_index("seed")["primary_value"].astype(float)
        each_seed_improves = bool((paired_primary < paired_negative).all())
        if specification.get("require_each_seed_primary_improvement", False):
            relative_checks["primary_value_each_seed"] = each_seed_improves

        quantitative_pass = bool(
            all(relative_checks.values()) and all(absolute_checks.values())
        )
        overall_pass = overall_pass and quantitative_pass
        summaries.append(
            {
                "experiment": experiment,
                "quantitative_pass": quantitative_pass,
                "means": means,
                "relative_checks": relative_checks,
                "absolute_checks": absolute_checks,
            }
        )

    return {
        "kind": "grounding-mle-vision-repair-check-v1",
        "posthoc_exploratory": True,
        "quantitative_pass": overall_pass,
        "visual_review_required": True,
        "status": (
            "quantitative_pass_visual_review_required"
            if overall_pass
            else "failed_quality_gate"
        ),
        "experiments": summaries,
    }
