from __future__ import annotations

import json
import os
from itertools import combinations, product
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grounding-mle-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .analysis import bootstrap_mean_ci, paired_effect
from .io import atomic_write_json, read_json, runtime_manifest, sha256_file
from .theory import schedule_statistics


def _primary_metrics(
    experiment: str, backend: str, metrics: dict[str, Any]
) -> tuple[str, float, bool]:
    if experiment == "gpt_misspecified":
        return "excess_cross_entropy", float(metrics["excess_cross_entropy"]), False
    if backend == "real_corpus_lm":
        return "real_cross_entropy", float(metrics["real_cross_entropy"]), False
    if backend == "flow_mode_recovery" or backend == "diffusion_mode_recovery":
        probability = float(metrics["missing_class_probability"])
        target = float(metrics.get("missing_class_target_probability", 0.1))
        return "missing_class_absolute_error", abs(probability - target), False
    if backend == "flow_teacher":
        return "teacher_student_kl", float(metrics["teacher_student_kl"]), False
    return (
        "teacher_student_kl_per_token",
        float(metrics["teacher_student_kl_per_token"]),
        False,
    )


def collect_generative_results(
    runs_root: str | Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for state_path in sorted(Path(runs_root).glob("*/state.json")):
        state = read_json(state_path)
        if state.get("kind") != "grounding-mle-generative-state-v1":
            continue
        resolved = read_json(state_path.parent / "resolved_run.json")
        if state.get("status") != "complete":
            continue
        completed.append(
            {"state": state, "resolved": resolved, "run_dir": str(state_path.parent)}
        )
        actual_real = [int(value) for value in state.get("actual_real", [])]
        actual_synthetic = [int(value) for value in state.get("actual_synthetic", [])]
        for point in state["trajectory"]:
            round_number = int(point["round"])
            name, primary, higher_is_better = _primary_metrics(
                resolved["experiment"], resolved["backend"], point["metrics"]
            )
            row: dict[str, Any] = {
                "run_id": state["run_id"],
                "experiment": resolved["experiment"],
                "condition": resolved["condition"],
                "seed": int(resolved["seed"]),
                "backend": resolved["backend"],
                "round": round_number,
                "primary_metric": name,
                "primary_value": primary,
                "higher_is_better": higher_is_better,
                "cumulative_real": sum(actual_real[:round_number]),
                "cumulative_synthetic": sum(actual_synthetic[:round_number]),
            }
            row.update(point["metrics"])
            if name == "missing_class_absolute_error":
                row.setdefault("missing_class_target_probability", 0.1)
                row["missing_class_absolute_error"] = primary
            if round_number:
                row.update(
                    schedule_statistics(
                        actual_real[:round_number], actual_synthetic[:round_number]
                    ).final
                )
            rows.append(row)
    return pd.DataFrame(rows), completed


def _condition_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, condition, metric), group in final.groupby(
        ["experiment", "condition", "primary_metric"]
    ):
        directions = group["higher_is_better"].dropna().unique()
        if len(directions) != 1:
            raise ValueError(
                f"Inconsistent optimization direction for {experiment}/{condition}/{metric}"
            )
        mean, low, high = bootstrap_mean_ci(
            group["primary_value"], draws=10_000, seed=20260919
        )
        rows.append(
            {
                "experiment": experiment,
                "condition": condition,
                "primary_metric": metric,
                "higher_is_better": bool(directions[0]),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "seeds": int(group["seed"].nunique()),
                "G_T": float(group["G_T"].mean()) if "G_T" in group else float("nan"),
                "total_real": (
                    float(group["total_real"].mean())
                    if "total_real" in group
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _exact_paired_sign_flip_pvalue(
    differences: np.ndarray, *, maximum_exact_pairs: int = 20
) -> float:
    """Return the exact two-sided randomization p-value for paired differences."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("At least one finite paired difference is required")
    if len(values) > maximum_exact_pairs:
        return float("nan")
    observed = abs(float(values.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    extreme = 0
    permutations = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += statistic >= observed - tolerance
        permutations += 1
    return float(extreme / permutations)


def _paired_effects(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, table in final.groupby("experiment"):
        conditions = sorted(table["condition"].unique())
        for left, right in combinations(conditions, 2):
            left_table = table[table["condition"] == left].set_index("seed")
            right_table = table[table["condition"] == right].set_index("seed")
            common = sorted(set(left_table.index) & set(right_table.index))
            if not common:
                continue
            primary_metrics = set(left_table.loc[common, "primary_metric"]) | set(
                right_table.loc[common, "primary_metric"]
            )
            if len(primary_metrics) != 1:
                raise ValueError(
                    f"Cannot compare different primary metrics for {experiment}: "
                    f"{left} versus {right}"
                )
            directions = set(left_table.loc[common, "higher_is_better"]) | set(
                right_table.loc[common, "higher_is_better"]
            )
            if len(directions) != 1:
                raise ValueError(
                    f"Cannot compare inconsistent optimization directions for {experiment}: "
                    f"{left} versus {right}"
                )
            left_values = left_table.loc[common, "primary_value"].to_numpy(dtype=float)
            right_values = right_table.loc[common, "primary_value"].to_numpy(dtype=float)
            differences = left_values - right_values
            effect = paired_effect(
                left_values,
                right_values,
            )
            higher_is_better = bool(next(iter(directions)))
            mean_difference = effect["paired_mean_difference"]
            if np.isclose(mean_difference, 0.0):
                favored = "tie"
            elif (mean_difference > 0) == higher_is_better:
                favored = left
            else:
                favored = right
            nonzero = differences[~np.isclose(differences, 0.0)]
            rows.append(
                {
                    "experiment": experiment,
                    "left": left,
                    "right": right,
                    "seeds": len(common),
                    "primary_metric": next(iter(primary_metrics)),
                    "higher_is_better": higher_is_better,
                    "paired_difference_definition": "left_minus_right",
                    "favored_condition": favored,
                    "exact_sign_flip_pvalue": _exact_paired_sign_flip_pvalue(differences),
                    "sign_flip_inference": (
                        "exact_two_sided"
                        if len(differences) <= 20
                        else "not_computed_more_than_20_pairs"
                    ),
                    "all_nonzero_differences_same_direction": bool(
                        len(nonzero)
                        and (np.all(nonzero > 0) or np.all(nonzero < 0))
                    ),
                }
                | effect
            )
    return pd.DataFrame(rows)


def _holm_adjusted_pvalues(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Return Holm step-down family-wise adjusted p-values in original order."""

    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires a non-empty finite p-value vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running_maximum = max(running_maximum, (count - rank) * values[index])
        adjusted[index] = min(1.0, running_maximum)
    return adjusted


def _observed_favored_condition(
    left: str,
    right: str,
    mean_difference: float,
    higher_is_better: bool,
) -> str:
    if np.isclose(mean_difference, 0.0):
        return "tie"
    if (mean_difference > 0) == higher_is_better:
        return left
    return right


def _json_safe(value: Any) -> Any:
    """Replace non-finite scalar values with JSON null recursively."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _confirmatory_contrast_table(
    final: pd.DataFrame, specification: dict[str, Any]
) -> pd.DataFrame:
    planned_seeds = tuple(int(value) for value in specification["planned_seeds"])
    if len(planned_seeds) != len(set(planned_seeds)):
        raise ValueError("confirmatory_analysis.planned_seeds must be distinct")
    contrasts = specification.get("contrasts", [])
    identifiers = [str(row["id"]) for row in contrasts]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Confirmatory contrasts require unique non-empty ids")
    rows: list[dict[str, Any]] = []
    planned_set = set(planned_seeds)
    for contrast in contrasts:
        experiment = str(contrast["experiment"])
        left = str(contrast["left"])
        right = str(contrast["right"])
        expected_favored = str(contrast["expected_favored"])
        if expected_favored not in {left, right}:
            raise ValueError(
                f"Expected favored condition for {contrast['id']} must be left or right"
            )
        table = final[final["experiment"] == experiment]
        left_table = table[table["condition"] == left].set_index("seed")
        right_table = table[table["condition"] == right].set_index("seed")
        if left_table.index.has_duplicates or right_table.index.has_duplicates:
            raise ValueError(f"Duplicate seed rows found for contrast {contrast['id']}")
        left_available = set(int(value) for value in left_table.index) & planned_set
        right_available = set(int(value) for value in right_table.index) & planned_set
        common = sorted(left_available & right_available)
        complete = left_available == planned_set and right_available == planned_set
        row: dict[str, Any] = {
            "contrast_id": str(contrast["id"]),
            "family": str(contrast["family"]),
            "experiment": experiment,
            "left": left,
            "right": right,
            "expected_favored": expected_favored,
            "planned_seed_count": len(planned_seeds),
            "paired_seed_count": len(common),
            "planned_seeds": ";".join(str(value) for value in planned_seeds),
            "missing_left_seeds": ";".join(
                str(value) for value in sorted(planned_set - left_available)
            ),
            "missing_right_seeds": ";".join(
                str(value) for value in sorted(planned_set - right_available)
            ),
            "complete": complete,
        }
        if common:
            primary_metrics = set(left_table.loc[common, "primary_metric"]) | set(
                right_table.loc[common, "primary_metric"]
            )
            directions = set(left_table.loc[common, "higher_is_better"]) | set(
                right_table.loc[common, "higher_is_better"]
            )
            if len(primary_metrics) != 1 or len(directions) != 1:
                raise ValueError(f"Inconsistent endpoint for contrast {contrast['id']}")
            left_values = left_table.loc[common, "primary_value"].to_numpy(dtype=float)
            right_values = right_table.loc[common, "primary_value"].to_numpy(dtype=float)
            differences = left_values - right_values
            effect = paired_effect(left_values, right_values)
            higher_is_better = bool(next(iter(directions)))
            observed_favored = _observed_favored_condition(
                left,
                right,
                effect["paired_mean_difference"],
                higher_is_better,
            )
            nonzero = differences[~np.isclose(differences, 0.0)]
            row.update(
                {
                    "primary_metric": next(iter(primary_metrics)),
                    "higher_is_better": higher_is_better,
                    "left_mean": float(left_values.mean()),
                    "right_mean": float(right_values.mean()),
                    "observed_favored": observed_favored,
                    "supports_expected_direction": observed_favored == expected_favored,
                    "all_nonzero_differences_same_direction": bool(
                        len(nonzero)
                        and (np.all(nonzero > 0) or np.all(nonzero < 0))
                    ),
                }
                | effect
            )
            row["exact_sign_flip_pvalue"] = (
                _exact_paired_sign_flip_pvalue(differences)
                if complete
                else float("nan")
            )
        else:
            row.update(
                {
                    "primary_metric": "",
                    "higher_is_better": None,
                    "left_mean": float("nan"),
                    "right_mean": float("nan"),
                    "observed_favored": "pending",
                    "supports_expected_direction": False,
                    "all_nonzero_differences_same_direction": False,
                    "paired_mean_difference": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "paired_standardized_effect": float("nan"),
                    "exact_sign_flip_pvalue": float("nan"),
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    result["holm_adjusted_pvalue"] = float("nan")
    result["family_complete"] = False
    alpha = float(specification.get("alpha", 0.05))
    for family, indices in result.groupby("family").groups.items():
        family_indices = list(indices)
        family_complete = bool(result.loc[family_indices, "complete"].all())
        result.loc[family_indices, "family_complete"] = family_complete
        if family_complete:
            result.loc[family_indices, "holm_adjusted_pvalue"] = _holm_adjusted_pvalues(
                result.loc[family_indices, "exact_sign_flip_pvalue"].to_numpy(dtype=float)
            )
        else:
            result.loc[family_indices, "exact_sign_flip_pvalue"] = float("nan")
    result["decision"] = "pending"
    ready = result["family_complete"]
    supported = result["supports_expected_direction"]
    rejected = result["holm_adjusted_pvalue"] < alpha
    result.loc[ready & supported & rejected, "decision"] = "supports_expected_direction"
    result.loc[ready & ~(supported & rejected), "decision"] = "not_confirmed"
    return result


def _confirmatory_timing_analysis(
    final: pd.DataFrame, specification: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    experiment = str(specification["experiment"])
    conditions = [str(value) for value in specification["conditions"]]
    planned_seeds = [int(value) for value in specification["planned_seeds"]]
    expected_total = float(specification["total_real"])
    predictor = str(specification.get("predictor", "G_T"))
    outcome = str(specification.get("outcome", "primary_value"))
    rows: list[dict[str, Any]] = []
    table = final[final["experiment"] == experiment]
    for seed in planned_seeds:
        seed_table = table[table["seed"] == seed].set_index("condition")
        if seed_table.index.has_duplicates:
            raise ValueError(f"Duplicate timing-bank conditions for seed {seed}")
        missing = [condition for condition in conditions if condition not in seed_table.index]
        row: dict[str, Any] = {
            "seed": seed,
            "complete": not missing,
            "missing_conditions": ";".join(missing),
            "condition_count": len(conditions) - len(missing),
        }
        if not missing:
            selected = seed_table.loc[conditions]
            totals = selected["total_real"].to_numpy(dtype=float)
            if not np.allclose(totals, expected_total):
                raise ValueError(
                    f"Timing bank seed {seed} does not preserve total_real={expected_total:g}"
                )
            x_values = selected[predictor].to_numpy(dtype=float)
            y_values = selected[outcome].to_numpy(dtype=float)
            if len(np.unique(x_values)) < 2:
                raise ValueError("Timing-bank predictor must vary across conditions")
            correlation = spearmanr(x_values, y_values)
            row.update(
                {
                    "slope": float(np.polyfit(x_values, y_values, 1)[0]),
                    "spearman_rho": float(correlation.statistic),
                    "G_T_min": float(x_values.min()),
                    "G_T_max": float(x_values.max()),
                    "total_real": expected_total,
                }
            )
        else:
            row.update(
                {
                    "slope": float("nan"),
                    "spearman_rho": float("nan"),
                    "G_T_min": float("nan"),
                    "G_T_max": float("nan"),
                    "total_real": expected_total,
                }
            )
        rows.append(row)
    by_seed = pd.DataFrame(rows)
    complete = bool(by_seed["complete"].all())
    available_slopes = by_seed.loc[by_seed["complete"], "slope"].to_numpy(dtype=float)
    available_correlations = by_seed.loc[
        by_seed["complete"], "spearman_rho"
    ].to_numpy(dtype=float)
    if len(available_slopes):
        mean, low, high = bootstrap_mean_ci(
            available_slopes, draws=10_000, seed=20260920
        )
    else:
        mean = low = high = float("nan")
    expected_sign = str(specification.get("expected_slope_sign", "negative"))
    if expected_sign not in {"negative", "positive"}:
        raise ValueError("fixed_budget_timing.expected_slope_sign must be negative or positive")
    supports_direction = bool(
        np.isfinite(mean)
        and (
            (mean < 0 and expected_sign == "negative")
            or (mean > 0 and expected_sign == "positive")
        )
    )
    exact_pvalue = (
        _exact_paired_sign_flip_pvalue(available_slopes)
        if complete
        else float("nan")
    )
    alpha = float(specification.get("alpha", 0.05))
    if not complete:
        decision = "pending"
    elif supports_direction and exact_pvalue < alpha:
        decision = "supports_expected_direction"
    else:
        decision = "not_confirmed"
    summary = {
        "experiment": experiment,
        "conditions": conditions,
        "planned_seed_count": len(planned_seeds),
        "complete_seed_count": int(by_seed["complete"].sum()),
        "complete": complete,
        "predictor": predictor,
        "outcome": outcome,
        "expected_total_real": expected_total,
        "expected_slope_sign": expected_sign,
        "mean_within_seed_slope": mean,
        "slope_ci_low": low,
        "slope_ci_high": high,
        "mean_within_seed_spearman_rho": (
            float(available_correlations.mean())
            if len(available_correlations)
            else float("nan")
        ),
        "exact_sign_flip_pvalue": exact_pvalue,
        "decision": decision,
    }
    return by_seed, summary


def _confirmatory_vision_controls(
    final: pd.DataFrame, specifications: list[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "primary_value",
        "class_kl_to_uniform",
        "feature_frechet_distance",
        "class_entropy",
    ]
    for specification in specifications:
        experiment = str(specification["experiment"])
        planned_seeds = {int(value) for value in specification["planned_seeds"]}
        positive = str(specification["positive_condition"])
        negative = str(specification["negative_condition"])
        table = final[final["experiment"] == experiment]
        groups = {
            condition: table[
                (table["condition"] == condition) & table["seed"].isin(planned_seeds)
            ]
            for condition in (positive, negative)
        }
        complete = all(set(group["seed"]) == planned_seeds for group in groups.values())
        row: dict[str, Any] = {
            "experiment": experiment,
            "positive_condition": positive,
            "negative_condition": negative,
            "planned_seed_count": len(planned_seeds),
            "positive_seed_count": int(groups[positive]["seed"].nunique()),
            "negative_seed_count": int(groups[negative]["seed"].nunique()),
            "complete": complete,
            "visual_review_required": True,
        }
        comparisons = []
        for metric in metrics:
            positive_mean = (
                float(groups[positive][metric].mean())
                if metric in groups[positive] and not groups[positive].empty
                else float("nan")
            )
            negative_mean = (
                float(groups[negative][metric].mean())
                if metric in groups[negative] and not groups[negative].empty
                else float("nan")
            )
            row[f"all_real_{metric}"] = positive_mean
            row[f"no_real_{metric}"] = negative_mean
            if metric != "class_entropy" and np.isfinite(positive_mean + negative_mean):
                comparisons.append(positive_mean < negative_mean)
        quantitative_pass = bool(complete and comparisons and all(comparisons))
        row["quantitative_pass"] = quantitative_pass
        if not complete:
            row["status"] = "pending"
        elif quantitative_pass:
            row["status"] = "quantitative_pass_requires_visual_review"
        else:
            row["status"] = "failed_quantitative_control"
        rows.append(row)
    return pd.DataFrame(rows)


def _analyze_confirmatory(
    final: pd.DataFrame,
    config_path: str | Path,
    output: Path,
) -> dict[str, Any]:
    source = Path(config_path).resolve()
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    specification = config.get("confirmatory_analysis")
    if not isinstance(specification, dict):
        raise ValueError(f"No confirmatory_analysis mapping found in {source}")
    if not specification.get("frozen_before_new_runs"):
        raise ValueError("Confirmatory analysis must declare frozen_before_new_runs: true")
    contrasts = _confirmatory_contrast_table(final, specification)
    timing_by_seed, timing_summary = _confirmatory_timing_analysis(
        final, specification["fixed_budget_timing"]
    )
    controls = _confirmatory_vision_controls(
        final, list(specification.get("vision_controls", []))
    )
    contrasts.to_csv(output / "confirmatory_contrasts.csv", index=False)
    timing_by_seed.to_csv(output / "confirmatory_timing_by_seed.csv", index=False)
    controls.to_csv(output / "confirmatory_vision_controls.csv", index=False)
    families = {}
    for family, table in contrasts.groupby("family"):
        families[str(family)] = {
            "tests": int(len(table)),
            "complete": bool(table["family_complete"].all()),
            "confirmed": int((table["decision"] == "supports_expected_direction").sum()),
            "pending": int((table["decision"] == "pending").sum()),
        }
    manifest = {
        "kind": "grounding-mle-confirmatory-analysis-v1",
        "config": str(source),
        "config_sha256": sha256_file(source),
        "protocol_version": int(specification.get("version", 1)),
        "alpha": float(specification.get("alpha", 0.05)),
        "planned_seeds": [int(value) for value in specification["planned_seeds"]],
        "families": families,
        "timing_bank": timing_summary,
        "vision_controls": json.loads(controls.to_json(orient="records")),
        "inference_note": (
            "Confirmatory p-values are withheld until every preregistered seed in a "
            "contrast family is complete. Holm adjustment is applied within each family."
        ),
    }
    safe_manifest = _json_safe(manifest)
    atomic_write_json(output / "confirmatory_analysis.json", safe_manifest)
    return safe_manifest


def _schedule_law(final: pd.DataFrame) -> dict[str, Any]:
    table = final[final["experiment"] == "gpt_schedule_sweep"].copy()
    if table["condition"].nunique() < 3:
        return {"skipped": True, "reason": "fewer than three completed sweep conditions"}
    aggregated = table.groupby("condition", as_index=False).agg(
        final_kl=("primary_value", "mean"),
        G_T=("G_T", "first"),
        Q_T=("Q_T", "first"),
        total_real=("total_real", "first"),
        mean_batch_real_fraction=("mean_batch_real_fraction", "first"),
    )
    correlation = spearmanr(aggregated["G_T"], aggregated["final_kl"])
    budget_correlation = spearmanr(aggregated["total_real"], aggregated["final_kl"])
    return {
        "skipped": False,
        "spearman_G_T_vs_final_kl": float(correlation.statistic),
        "pvalue": float(correlation.pvalue),
        "nominal_pvalue": float(correlation.pvalue),
        "spearman_total_real_vs_final_kl": float(budget_correlation.statistic),
        "total_real_nominal_pvalue": float(budget_correlation.pvalue),
        "inference_note": (
            "Condition-level Spearman p-values are descriptive because conditions share "
            "seeds and total-real budgets vary. Use matched-budget paired contrasts for "
            "timing claims."
        ),
        "conditions": aggregated.to_dict(orient="records"),
    }


def _trajectory_figure(metrics: pd.DataFrame, output: Path) -> None:
    studies = ["gpt_boundary", "flow_exact_likelihood"]
    selected = metrics[metrics["experiment"].isin(studies)]
    if selected.empty:
        return
    experiments = [value for value in studies if value in set(selected["experiment"])]
    fig, axes = plt.subplots(
        1, len(experiments), figsize=(7 * len(experiments), 4.8), squeeze=False
    )
    for axis, experiment in zip(axes[0], experiments, strict=True):
        table = selected[selected["experiment"] == experiment]
        for condition, group in table.groupby("condition"):
            summary = (
                group.groupby("round")["primary_value"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            axis.plot(summary["round"], summary["mean"], marker="o", label=condition)
            lower = np.maximum(0, summary["mean"] - 1.96 * summary["sem"].fillna(0))
            upper = summary["mean"] + 1.96 * summary["sem"].fillna(0)
            axis.fill_between(summary["round"], lower, upper, alpha=0.14)
        axis.set(
            xlabel="recursive round",
            ylabel="teacher-to-student KL",
            title=experiment.replace("_", " "),
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(
        output / "figure_G1_exact_likelihood_trajectories.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(output / "figure_G1_exact_likelihood_trajectories.svg", bbox_inches="tight")
    plt.close(fig)


def _schedule_scatter(law: dict[str, Any], output: Path) -> None:
    if law.get("skipped"):
        return
    table = pd.DataFrame(law["conditions"])
    fig, axis = plt.subplots(figsize=(7.2, 5.1))
    axis.scatter(table["G_T"], table["final_kl"], s=55)
    for row in table.itertuples():
        axis.annotate(
            row.condition,
            (row.G_T, row.final_kl),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set(
        xlabel=r"cumulative restoring mass $G_T$",
        ylabel="final teacher-to-student KL",
        title="Does the theoretical schedule statistic organize neural recovery?",
    )
    axis.grid(alpha=0.25)
    fig.savefig(output / "figure_G2_neural_schedule_law.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_G2_neural_schedule_law.svg", bbox_inches="tight")
    plt.close(fig)


def _timing_figure(final: pd.DataFrame, output: Path) -> None:
    table = final[final["experiment"] == "gpt_same_budget_timing"]
    if table.empty:
        return
    summary = _condition_summary(table).sort_values("G_T")
    fig, axis = plt.subplots(figsize=(7.2, 5))
    errors = np.vstack(
        [summary["mean"] - summary["ci_low"], summary["ci_high"] - summary["mean"]]
    )
    axis.bar(summary["condition"], summary["mean"], yerr=errors, capsize=4)
    axis.set(
        ylabel="final teacher-to-student KL",
        title="Same real-data budget, different timing",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "figure_G3_same_budget_timing.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_G3_same_budget_timing.svg", bbox_inches="tight")
    plt.close(fig)


def _robustness_figure(final: pd.DataFrame, output: Path) -> None:
    experiments = ["gpt_misspecified", "gpt_real_corpus", "gpt_model_scale"]
    available = [value for value in experiments if value in set(final["experiment"])]
    if not available:
        return
    fig, axes = plt.subplots(
        1, len(available), figsize=(6.2 * len(available), 4.8), squeeze=False
    )
    for axis, experiment in zip(axes[0], available, strict=True):
        table = final[final["experiment"] == experiment]
        summary = _condition_summary(table)
        axis.bar(summary["condition"], summary["mean"])
        axis.set(title=experiment.replace("_", " "), ylabel=summary.iloc[0]["primary_metric"])
        axis.tick_params(axis="x", rotation=30, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "figure_G4_robustness.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_G4_robustness.svg", bbox_inches="tight")
    plt.close(fig)


def _mode_figure(metrics: pd.DataFrame, output: Path) -> None:
    experiments = ["flow_mode_recovery", "diffusion_mode_recovery"]
    available = [value for value in experiments if value in set(metrics["experiment"])]
    if not available:
        return
    fig, axes = plt.subplots(
        1, len(available), figsize=(7 * len(available), 4.8), squeeze=False
    )
    for axis, experiment in zip(axes[0], available, strict=True):
        table = metrics[metrics["experiment"] == experiment]
        for condition, group in table.groupby("condition"):
            summary = (
                group.groupby("round")["missing_class_probability"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            axis.plot(summary["round"], summary["mean"], marker="o", label=condition)
        target = 0.1
        finite_probabilities = pd.to_numeric(
            table["missing_class_probability"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        upper_limit = 0.16
        if not finite_probabilities.empty:
            upper_limit = min(
                1.0, max(upper_limit, float(finite_probabilities.max()) * 1.08)
            )
        axis.axhline(
            target,
            linestyle="--",
            color="black",
            linewidth=1,
            label="balanced target",
        )
        axis.set(
            xlabel="recursive round",
            ylabel="generated missing-class probability",
            title=f"{experiment.replace('_', ' ')}\n(primary: absolute distance from 0.10)",
            ylim=(0, upper_limit),
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output / "figure_G5_mode_recovery.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_G5_mode_recovery.svg", bbox_inches="tight")
    plt.close(fig)


def _sample_grids(completed: list[dict[str, Any]], output: Path) -> None:
    selected: dict[tuple[str, str], Path] = {}
    for item in completed:
        resolved, state = item["resolved"], item["state"]
        if resolved["backend"] not in {"flow_mode_recovery", "diffusion_mode_recovery"}:
            continue
        key = (resolved["experiment"], resolved["condition"])
        if key in selected:
            continue
        selected[key] = (
            Path(item["run_dir"])
            / "rounds"
            / f"round_{int(state['completed_rounds']):02d}"
            / "evaluation"
            / "sample_images.npy"
        )
    for (experiment, condition), path in selected.items():
        if not path.exists():
            continue
        images = np.load(path, allow_pickle=False)[:64]
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for axis, image in zip(axes.ravel(), images, strict=False):
            axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
        for axis in axes.ravel()[len(images) :]:
            axis.axis("off")
        fig.suptitle(f"{experiment}: {condition}")
        fig.savefig(output / f"samples_{experiment}_{condition}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def analyze_generative_runs(
    runs_root: str | Path = "runs/generative",
    output_dir: str | Path = "results/generative",
    confirmatory_config: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, completed = collect_generative_results(runs_root)
    if metrics.empty:
        raise ValueError(f"No completed generative runs found under {runs_root}")
    final_round = metrics.groupby("run_id")["round"].transform("max")
    final = metrics[metrics["round"] == final_round].copy()
    summary = _condition_summary(final)
    effects = _paired_effects(final)
    law = _schedule_law(final)
    metrics.to_csv(output / "trajectory_metrics.csv", index=False)
    final.to_csv(output / "final_metrics.csv", index=False)
    summary.to_csv(output / "condition_summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    atomic_write_json(output / "neural_schedule_law.json", law)
    _trajectory_figure(metrics, output)
    _schedule_scatter(law, output)
    _timing_figure(final, output)
    _robustness_figure(final, output)
    _mode_figure(metrics, output)
    _sample_grids(completed, output)
    confirmatory = (
        _analyze_confirmatory(final, confirmatory_config, output)
        if confirmatory_config is not None
        else None
    )
    manifest = {
        "kind": "grounding-mle-generative-analysis-v2",
        "completed_runs": int(final["run_id"].nunique()),
        "experiments": sorted(final["experiment"].unique().tolist()),
        "rows": int(len(metrics)),
        "primary_endpoints": {
            "vision_mode_recovery": (
                "absolute error between generated missing-class probability and the "
                "balanced target probability (0.10); lower is better"
            )
        },
        "analysis_revision": {
            "version": 2,
            "post_run_correction": True,
            "reason": (
                "Version 1 treated missing-class probability as monotonically better, "
                "which incorrectly rewarded overshoot beyond the balanced 0.10 target."
            ),
            "training_rerun_required": False,
        },
        "paired_inference": {
            "effect": "left-minus-right paired mean with percentile bootstrap interval",
            "test": "exact two-sided paired sign-flip randomization test",
            "small_sample_note": (
                "For n paired seeds, the smallest attainable nonzero two-sided sign-flip "
                "p-value is 2/(2**n). Confirmatory inference is reported separately and "
                "withheld until its frozen seed set is complete."
            ),
        },
        "confirmatory": confirmatory,
        "schedule_law": law,
        "runtime": runtime_manifest(),
    }
    atomic_write_json(output / "analysis_manifest.json", manifest)
    return manifest
