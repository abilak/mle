from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grounding-mle-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .capabilities import common_and_rare, difficulty_bin
from .io import atomic_write_json, read_json
from .theory import schedule_statistics


def bootstrap_mean_ci(
    values: Iterable[float], confidence: float = 0.95, draws: int = 10_000, seed: int = 20260906
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    alpha = 1 - confidence
    return (
        float(array.mean()),
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def paired_effect(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    left, right = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Paired samples must have the same non-zero shape")
    differences = left - right
    standard_deviation = differences.std(ddof=1) if len(differences) > 1 else 0.0
    standardized = float(differences.mean() / standard_deviation) if standard_deviation > 0 else float("inf")
    mean, low, high = bootstrap_mean_ci(differences)
    return {
        "paired_mean_difference": mean,
        "ci_low": low,
        "ci_high": high,
        "paired_standardized_effect": standardized,
    }


def _primary_metric(evaluation: dict[str, Any] | None) -> float | None:
    if not evaluation:
        return None
    if "analytic" in evaluation:
        return float(evaluation["analytic"]["pass_at_1"])
    if "humaneval" in evaluation and not evaluation["humaneval"].get("skipped"):
        return float(evaluation["humaneval"]["pass_at_1"])
    if "mbpp" in evaluation and not evaluation["mbpp"].get("skipped"):
        return float(evaluation["mbpp"]["pass_at_1"])
    if "gsm8k" in evaluation:
        return float(evaluation["gsm8k"]["accuracy"])
    return None


def collect_results(runs_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for state_path in sorted(Path(runs_dir).glob("*/state.json")):
        state = read_json(state_path)
        resolved = read_json(state_path.parent / "resolved_run.json")
        if state.get("status") != "complete":
            continue
        states.append({"state": state, "resolved": resolved, "run_dir": str(state_path.parent)})
        cumulative_real, cumulative_synthetic = 0, 0
        actual_real = [int(x) for x in state.get("actual_real", [])]
        actual_synthetic = [int(x) for x in state.get("actual_synthetic", [])]
        for point in state["trajectory"]:
            round_index = int(point["round"])
            if round_index > 0:
                cumulative_real += int(point.get("real", 0))
                cumulative_synthetic += int(point.get("synthetic", 0))
            evaluation = point.get("evaluation")
            quality = _primary_metric(evaluation)
            row = {
                "run_id": state["run_id"],
                "experiment": resolved["experiment"],
                "condition": resolved["condition"],
                "seed": int(resolved["seed"]),
                "round": round_index,
                "quality": quality,
                "error": None if quality is None else 1 - quality,
                "cumulative_real": cumulative_real,
                "cumulative_synthetic": cumulative_synthetic,
                "law_split": resolved.get("law_split"),
                "model": resolved["config"]["model"]["name"],
                "verification": resolved["config"].get("verification", {}).get("mode", "none"),
                "synthetic_strategy": resolved["config"].get("synthetic", {}).get("strategy", "passive"),
                "training_mode": resolved["config"]["training"]["mode"],
            }
            if round_index > 0:
                prefix = schedule_statistics(
                    actual_real[:round_index], actual_synthetic[:round_index]
                ).final
                row.update(prefix)
            rows.append(row)
            if evaluation:
                for benchmark, metrics in evaluation.items():
                    for task in metrics.get("tasks", []):
                        task_rows.append(
                            row
                            | {
                                "benchmark": benchmark,
                                "task_id": task["task_id"],
                                "passed": bool(task["passed"]),
                                "skill": task.get("skill", "math_reasoning"),
                                "algorithm_type": task.get("algorithm_type", task.get("skill", "unknown")),
                                "api_family": task.get("api_family", "unknown"),
                                "prompt_complexity": task.get("prompt_complexity", "unknown"),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(task_rows), states


def fit_schedule_law(final_rows: pd.DataFrame) -> dict[str, Any]:
    table = final_rows.dropna(subset=["error", "law_split"]).copy()
    if table.empty:
        return {"skipped": True, "reason": "no completed schedule-law runs"}
    aggregated = (
        table.groupby(["condition", "law_split"], as_index=False)
        .agg(
            error=("error", "mean"),
            total_real=("total_real", "first"),
            mean_batch_real_fraction=("mean_batch_real_fraction", "first"),
            cumulative_real_fraction=("cumulative_real_fraction", "first"),
            G_T=("G_T", "first"),
            Q_T_squared=("Q_T_squared", "first"),
            Q_T_squared_A_T=("Q_T_squared_A_T", "first"),
        )
    )
    train = aggregated[aggregated["law_split"] == "train"]
    test = aggregated[aggregated["law_split"] == "test"]
    if len(train) < 3 or len(test) < 2:
        return {"skipped": True, "reason": "insufficient train/test schedules"}
    theory_features = ["Q_T_squared", "Q_T_squared_A_T"]
    coefficients, _ = nnls(train[theory_features].to_numpy(), train["error"].to_numpy())
    theory_prediction = test[theory_features].to_numpy() @ coefficients
    baseline_features = [
        "total_real",
        "mean_batch_real_fraction",
        "cumulative_real_fraction",
    ]
    scaler = StandardScaler().fit(train[baseline_features])
    baseline = RidgeCV(alphas=np.logspace(-4, 4, 40)).fit(
        scaler.transform(train[baseline_features]), train["error"]
    )
    baseline_prediction = baseline.predict(scaler.transform(test[baseline_features]))

    def scores(prediction: np.ndarray) -> dict[str, float]:
        observed = test["error"].to_numpy()
        return {
            "r2": float(r2_score(observed, prediction)),
            "mae": float(mean_absolute_error(observed, prediction)),
            "spearman": float(spearmanr(observed, prediction).statistic),
        }

    prediction_rows = test[["condition", "error"]].copy()
    prediction_rows["theory_prediction"] = theory_prediction
    prediction_rows["baseline_prediction"] = baseline_prediction
    return {
        "theory_coefficients": dict(zip(theory_features, coefficients.tolist(), strict=True)),
        "theory_scores": scores(theory_prediction),
        "percentage_baseline_scores": scores(baseline_prediction),
        "baseline_alpha": float(baseline.alpha_),
        "predictions": prediction_rows.to_dict(orient="records"),
    }


def _line_figure(metrics: pd.DataFrame, output: Path) -> None:
    preferred = metrics[metrics["experiment"] == "exp01_same_budget"]
    subset = (preferred if not preferred.empty else metrics).dropna(subset=["quality"])
    headline = {"uniform", "front_loaded", "back_loaded", "bursty", "theory_optimized"}
    if headline.issubset(set(subset["condition"])):
        subset = subset[subset["condition"].isin(headline)]
    if subset.empty:
        return
    summary_rows = []
    for (condition, round_index), group in subset.groupby(["condition", "round"]):
        mean, low, high = bootstrap_mean_ci(group["quality"])
        summary_rows.append(
            {
                "condition": condition,
                "round": round_index,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for condition, group in summary.groupby("condition"):
        group = group.sort_values("round")
        axis.plot(group["round"], group["mean"], marker="o", label=condition)
        axis.fill_between(
            group["round"],
            group["ci_low"],
            group["ci_high"],
            alpha=0.15,
        )
    axis.set(xlabel="recursive training round", ylabel="held-out quality", title="Same budget, different fate")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(output / "figure_A_schedule_trajectories.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_A_schedule_trajectories.svg", bbox_inches="tight")
    plt.close(fig)


def _law_figure(law: dict[str, Any], output: Path) -> None:
    if law.get("skipped"):
        return
    data = pd.DataFrame(law["predictions"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    limits = [0, max(data[["error", "theory_prediction", "baseline_prediction"]].max()) * 1.05]
    for axis, column, title in [
        (axes[0], "baseline_prediction", "Percentage/count baseline"),
        (axes[1], "theory_prediction", "MLE schedule law"),
    ]:
        axis.scatter(data[column], data["error"])
        axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        axis.set(title=title, xlabel="predicted final error")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("observed final error")
    fig.savefig(output / "figure_B_schedule_law.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_B_schedule_law.svg", bbox_inches="tight")
    plt.close(fig)


def _controller_figure(final_rows: pd.DataFrame, output: Path) -> None:
    preferred = final_rows[final_rows["experiment"] == "exp03_grounding_controller"]
    subset = preferred if not preferred.empty else final_rows
    if subset.empty:
        return
    summary = (
        subset.dropna(subset=["quality"])
        .groupby(["condition", "cumulative_real"], as_index=False)["quality"]
        .mean()
    )
    if summary.empty:
        return
    fig, axis = plt.subplots(figsize=(7.5, 5))
    for condition, group in summary.groupby("condition"):
        group = group.sort_values("cumulative_real")
        axis.plot(group["cumulative_real"], group["quality"], marker="o", label=condition)
    axis.set(
        xlabel="external examples consumed",
        ylabel="final held-out quality",
        title="Grounding-cost Pareto frontier",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / "figure_C_controller_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_C_controller_frontier.svg", bbox_inches="tight")
    plt.close(fig)


def _verification_figure(final_rows: pd.DataFrame, output: Path) -> None:
    preferred = final_rows[final_rows["experiment"] == "exp06_verification_phase_diagram"]
    subset = (preferred if not preferred.empty else final_rows).dropna(subset=["quality"]).copy()
    if subset.empty or subset["verification"].nunique() < 2:
        return
    subset["phase_row"] = np.where(
        subset["condition"].str.startswith("theory_controlled"),
        "external_tests + theory controller",
        subset["verification"],
    )
    table = subset.pivot_table(
        index="phase_row", columns="cumulative_real", values="quality", aggfunc="mean"
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(table.columns)), [str(int(x)) for x in table.columns])
    axis.set_yticks(np.arange(len(table.index)), table.index)
    axis.set(xlabel="grounding budget", ylabel="verification", title="Grounding x verification")
    fig.colorbar(image, ax=axis, label="final quality")
    fig.savefig(output / "figure_D_verification_phase_diagram.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figure_D_verification_phase_diagram.svg", bbox_inches="tight")
    plt.close(fig)


def _capability_figure(
    tasks: pd.DataFrame, final_rows: pd.DataFrame, output: Path
) -> None:
    if tasks.empty:
        return
    preferred = tasks[tasks["experiment"] == "exp05_capability_erosion"]
    subset = (preferred if not preferred.empty else tasks).copy()
    if subset.empty:
        return
    enriched = _capability_groups(subset)
    final_rounds = enriched.groupby("run_id")["round"].transform("max")
    final = enriched[enriched["round"] == final_rounds].copy()
    initial_table = final.pivot_table(
        index="condition", columns="initial_group", values="passed", aggfunc="mean"
    )
    rarity_table = final.pivot_table(
        index="condition", columns="skill_rarity", values="passed", aggfunc="mean"
    )
    if initial_table.empty and rarity_table.empty:
        return
    transfer = final_rows[
        (final_rows["experiment"] == "exp08_model_scale_architecture")
        & final_rows["quality"].notna()
        & ~final_rows["condition"].str.endswith("_verified")
    ].copy()
    transfer_table = pd.DataFrame()
    if not transfer.empty:
        parsed = transfer["condition"].str.extract(
            r"^(qwen_05b|qwen_15b|deepseek_13b)_(uniform|selected_bad|selected_good|theory)(?:_verified)?$"
        )
        transfer["model_group"] = parsed[0].map(
            {"qwen_05b": "Qwen 0.5B", "qwen_15b": "Qwen 1.5B", "deepseek_13b": "DeepSeek 1.3B"}
        )
        transfer["schedule_group"] = parsed[1]
        transfer = transfer.dropna(subset=["model_group", "schedule_group"])
        transfer_table = transfer.pivot_table(
            index="schedule_group", columns="model_group", values="quality", aggfunc="mean"
        )
    panels = 3 if not transfer_table.empty else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.4 * panels, 5), sharey=True)
    initial_table.plot(kind="bar", ax=axes[0])
    rarity_table.plot(kind="bar", ax=axes[1])
    axes[0].set(title="Retention by initial capability", ylabel="final pass rate", ylim=(0, 1))
    axes[1].set(title="Retention by skill rarity", ylim=(0, 1))
    if panels == 3:
        transfer_table.plot(kind="bar", ax=axes[2])
        axes[2].set(title="Schedule transfer across models", ylim=(0, 1))
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output / "figure_E_capability_survival.png", dpi=220, bbox_inches="tight")
    plt.savefig(output / "figure_E_capability_survival.svg", bbox_inches="tight")
    plt.close()


def _capability_groups(tasks: pd.DataFrame) -> pd.DataFrame:
    enriched = tasks.copy()
    key = ["experiment", "model", "task_id"]
    baseline = (
        enriched[enriched["round"] == 0]
        .groupby(key, as_index=False)["passed"]
        .mean()
        .rename(columns={"passed": "initial_success"})
    )
    baseline["initial_group"] = baseline["initial_success"].map(difficulty_bin)
    baseline["initial_tail"] = "other"
    for _, group in baseline.groupby(["experiment", "model"]):
        count = max(1, int(np.ceil(len(group) * 0.1)))
        weakest = group.sort_values(["initial_success", "task_id"]).head(count).index
        baseline.loc[weakest, "initial_tail"] = "bottom_decile"
    enriched = enriched.merge(baseline, on=key, how="left")
    enriched["initial_group"] = enriched["initial_group"].fillna("unknown")
    unique_skills = enriched[["task_id", "skill"]].drop_duplicates("task_id")
    rarity = common_and_rare(unique_skills["skill"])
    enriched["skill_rarity"] = enriched["skill"].map(rarity).fillna("unknown")
    return enriched


def _paired_effect_table(final_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    usable = final_rows.dropna(subset=["quality"])
    for experiment, group in usable.groupby("experiment"):
        pivot = group.pivot_table(
            index="seed", columns="condition", values="quality", aggfunc="mean"
        )
        for condition_a, condition_b in combinations(sorted(pivot.columns), 2):
            paired = pivot[[condition_a, condition_b]].dropna()
            if paired.empty:
                continue
            effect = paired_effect(paired[condition_a], paired[condition_b])
            output.append(
                {
                    "experiment": experiment,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "n_pairs": int(len(paired)),
                }
                | effect
            )
    return pd.DataFrame(output)


def analyze_runs(runs_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, tasks, states = collect_results(runs_dir)
    if metrics.empty:
        raise RuntimeError("No completed runs found")
    metrics.to_csv(output / "trajectory_metrics.csv", index=False)
    tasks.to_csv(output / "task_metrics.csv", index=False)
    final_rows = metrics.sort_values("round").groupby("run_id", as_index=False).tail(1)
    paired = _paired_effect_table(final_rows)
    paired.to_csv(output / "paired_effects.csv", index=False)
    law = fit_schedule_law(final_rows)
    atomic_write_json(output / "schedule_law.json", law)
    eligible = {
        "uniform",
        "front_loaded",
        "back_loaded",
        "periodic",
        "decaying",
        "growing",
        "bursty",
    }
    discovery = final_rows[
        (final_rows["experiment"] == "exp01_same_budget")
        & final_rows["condition"].isin(eligible)
        & final_rows["quality"].notna()
    ]
    selected_confirmation = None
    if not discovery.empty:
        condition_means = discovery.groupby("condition")["quality"].mean()
        worst_condition = str(condition_means.idxmin())
        selected_run_id = str(
            discovery[discovery["condition"] == worst_condition].iloc[0]["run_id"]
        )
        selected_state = next(
            item for item in states if item["state"]["run_id"] == selected_run_id
        )
        selected_confirmation = {
            "selection_rule": "lowest mean final primary endpoint among preregistered matched-budget fixed schedules",
            "condition": worst_condition,
            "mean_final_quality": float(condition_means[worst_condition]),
            "real_counts": selected_state["state"]["actual_real"],
            "synthetic_counts": selected_state["state"]["actual_synthetic"],
        }
        atomic_write_json(
            output / "selected_confirmation_schedule.json", selected_confirmation
        )
        best_condition = str(condition_means.idxmax())
        best_run_id = str(
            discovery[discovery["condition"] == best_condition].iloc[0]["run_id"]
        )
        best_state = next(
            item for item in states if item["state"]["run_id"] == best_run_id
        )
        selected_success = {
            "selection_rule": "highest mean final primary endpoint among preregistered matched-budget fixed schedules",
            "condition": best_condition,
            "mean_final_quality": float(condition_means[best_condition]),
            "real_counts": best_state["state"]["actual_real"],
            "synthetic_counts": best_state["state"]["actual_synthetic"],
        }
        atomic_write_json(output / "selected_success_schedule.json", selected_success)
    else:
        selected_success = None
    summary_rows: list[dict[str, Any]] = []
    for (experiment, condition), group in final_rows.groupby(["experiment", "condition"]):
        mean, low, high = bootstrap_mean_ci(group["quality"].dropna())
        summary_rows.append(
            {
                "experiment": experiment,
                "condition": condition,
                "n_seeds": int(group["quality"].notna().sum()),
                "mean_final_quality": mean,
                "ci_low": low,
                "ci_high": high,
                "mean_external_examples": float(group["cumulative_real"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "condition_summary.csv", index=False)
    if not tasks.empty:
        enriched_tasks = _capability_groups(tasks)
        final_task_round = enriched_tasks.groupby("run_id")["round"].transform("max")
        final_tasks = enriched_tasks[enriched_tasks["round"] == final_task_round]
        summaries = []
        for dimension in [
            "initial_group",
            "initial_tail",
            "skill_rarity",
            "algorithm_type",
            "api_family",
            "prompt_complexity",
        ]:
            grouped = (
                final_tasks.groupby(["experiment", "condition", dimension], as_index=False)
                .agg(mean_pass=("passed", "mean"), task_count=("passed", "count"))
                .rename(columns={dimension: "group"})
            )
            grouped["dimension"] = dimension
            summaries.append(grouped)
        pd.concat(summaries, ignore_index=True).to_csv(
            output / "capability_summary.csv", index=False
        )
    _line_figure(metrics, output)
    _law_figure(law, output)
    _controller_figure(final_rows, output)
    _verification_figure(final_rows, output)
    _capability_figure(tasks, final_rows, output)
    manifest = {
        "completed_runs": len(states),
        "experiments": sorted(metrics["experiment"].unique().tolist()),
        "trajectory_rows": len(metrics),
        "task_rows": len(tasks),
        "paired_effect_rows": len(paired),
        "schedule_law": law,
        "selected_confirmation_schedule": selected_confirmation,
        "selected_success_schedule": selected_success,
        "figures": sorted(path.name for path in output.glob("figure_*.png")),
    }
    atomic_write_json(output / "analysis_manifest.json", manifest)
    return manifest
