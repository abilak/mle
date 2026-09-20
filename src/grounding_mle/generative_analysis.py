from __future__ import annotations

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
from scipy.stats import spearmanr

from .analysis import bootstrap_mean_ci, paired_effect
from .io import atomic_write_json, read_json, runtime_manifest
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
                "With three paired seeds, the smallest attainable nonzero two-sided "
                "sign-flip p-value is 0.25."
            ),
        },
        "schedule_law": law,
        "runtime": runtime_manifest(),
    }
    atomic_write_json(output / "analysis_manifest.json", manifest)
    return manifest
