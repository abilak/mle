from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .capabilities import capability_metadata, infer_skill
from .io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from .modeling import ModelBackend
from .records import PromptRecord


def _evalplus_problems(dataset_name: str) -> list[PromptRecord]:
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError("EvalPlus 0.3.1 is required for code evaluation") from exc
    if dataset_name == "humaneval":
        source = get_human_eval_plus()
    elif dataset_name == "mbpp":
        source = get_mbpp_plus()
    else:
        raise ValueError("EvalPlus dataset must be humaneval or mbpp")
    return [
        PromptRecord(
            task_id=str(task_id),
            prompt=str(problem["prompt"]),
            human_responses=[],
            skill=infer_skill(str(problem["prompt"])),
            metadata={"dataset": dataset_name} | capability_metadata(str(problem["prompt"])),
        )
        for task_id, problem in source.items()
    ]


def _parse_evalplus_result(path: Path, prompts: Sequence[PromptRecord]) -> dict[str, Any]:
    payload = read_json(path)
    evaluations = payload.get("eval", {})
    prompt_by_id = {prompt.task_id: prompt for prompt in prompts}
    task_rows: list[dict[str, Any]] = []
    for task_id, attempts in evaluations.items():
        attempt = attempts[0] if isinstance(attempts, list) and attempts else attempts
        if not isinstance(attempt, dict):
            continue
        base_status = str(attempt.get("base_status", attempt.get("status", "unknown"))).lower()
        plus_status = str(attempt.get("plus_status", base_status)).lower()
        passed = base_status == "pass" and plus_status == "pass"
        prompt = prompt_by_id.get(str(task_id))
        task_rows.append(
            {
                "task_id": str(task_id),
                "passed": passed,
                "base_passed": base_status == "pass",
                "skill": prompt.skill if prompt else "unknown",
                "algorithm_type": prompt.metadata.get("algorithm_type", "unknown") if prompt else "unknown",
                "api_family": prompt.metadata.get("api_family", "unknown") if prompt else "unknown",
                "prompt_complexity": prompt.metadata.get("prompt_complexity", "unknown") if prompt else "unknown",
            }
        )
    if not task_rows:
        raise RuntimeError(f"No task-level results found in {path}")
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in task_rows:
        grouped[row["skill"]].append(bool(row["passed"]))
    return {
        "pass_at_1": float(np.mean([row["passed"] for row in task_rows])),
        "base_pass_at_1": float(np.mean([row["base_passed"] for row in task_rows])),
        "worst_skill_pass_at_1": float(min(np.mean(values) for values in grouped.values())),
        "by_skill": {name: float(np.mean(values)) for name, values in sorted(grouped.items())},
        "tasks": task_rows,
        "raw_result": str(path),
    }


@lru_cache(maxsize=1)
def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for safe EvalPlus execution")
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot connect to the Docker daemon: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Cannot access the Docker daemon. Confirm that Docker is running and "
            f"the current user can access its socket: {detail[-1000:]}"
        )


def _cached_completions(
    samples_path: Path, prompts: Sequence[PromptRecord]
) -> list[str] | None:
    if not samples_path.exists():
        return None
    try:
        rows = list(read_jsonl(samples_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if [row.get("task_id") for row in rows] != [prompt.task_id for prompt in prompts]:
        return None
    solutions = [row.get("solution") for row in rows]
    if not all(isinstance(solution, str) for solution in solutions):
        return None
    return [str(solution) for solution in solutions]


def evaluate_evalplus(
    backend: ModelBackend,
    model_ref: str,
    base_model: str,
    generation_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    seed: int,
    output_dir: Path,
    dataset_name: str,
) -> dict[str, Any]:
    if evaluation_config.get("code_execution", "docker") == "disabled":
        return {"skipped": True, "reason": "code execution disabled"}
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{dataset_name}_metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)

    execution = evaluation_config.get("code_execution", "docker")
    if execution == "docker":
        _require_docker()
    elif execution != "native_unsafe":
        raise ValueError(f"Unsupported execution backend: {execution}")

    prompts = _evalplus_problems(dataset_name)
    max_tasks = evaluation_config.get("max_tasks")
    if max_tasks is not None and int(max_tasks) < len(prompts):
        raise ValueError(
            "EvalPlus requires a completion for every benchmark problem; "
            "partial max_tasks evaluation is intentionally unsupported"
        )
    samples_path = output_dir / f"{dataset_name}_samples.jsonl"
    completions = _cached_completions(samples_path, prompts)
    if completions is None:
        completions = backend.generate(
            model_ref, prompts, generation_config, seed, base_model, "code"
        )
        atomic_write_jsonl(
            samples_path,
            (
                {"task_id": prompt.task_id, "solution": completion}
                for prompt, completion in zip(prompts, completions, strict=True)
            ),
        )
    if execution == "docker":
        image = str(evaluation_config.get("evalplus_image", "ganler/evalplus:v0.3.1"))
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            str(evaluation_config.get("evalplus_memory", "4g")),
            "--cpus",
            str(evaluation_config.get("evalplus_cpus", 4)),
            "-v",
            f"{output_dir.resolve()}:/app",
            image,
            "evalplus.evaluate",
            "--dataset",
            dataset_name,
            "--samples",
            f"/app/{samples_path.name}",
            "--parallel",
            str(evaluation_config.get("parallel", 2)),
        ]
        if bool(evaluation_config.get("mini", False)):
            command.append("--mini")
    elif execution == "native_unsafe":
        command = [
            "python",
            "-m",
            "evalplus.evaluate",
            "--dataset",
            dataset_name,
            "--samples",
            str(samples_path),
            "--parallel",
            str(evaluation_config.get("parallel", 2)),
        ]
    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=int(evaluation_config.get("timeout_seconds", 7200)),
        check=False,
    )
    (output_dir / f"{dataset_name}_eval_stdout.txt").write_text(
        completed.stdout + "\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"EvalPlus failed for {dataset_name} with code {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    result_candidates = sorted(
        output_dir.glob(f"{samples_path.stem}*eval_results.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not result_candidates:
        result_candidates = sorted(
            output_dir.glob("*eval_results.json"), key=lambda path: path.stat().st_mtime
        )
    if not result_candidates:
        raise RuntimeError("EvalPlus completed without writing an eval_results JSON file")
    metrics = _parse_evalplus_result(result_candidates[-1], prompts)
    atomic_write_json(metrics_path, metrics)
    return metrics


FINAL_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")


def _normalize_math_answer(text: str) -> str:
    matches = FINAL_ANSWER_PATTERN.findall(text)
    value = matches[-1] if matches else text.strip().splitlines()[-1]
    return value.replace(",", "").replace("$", "").strip()


def _math_equal(prediction: str, reference: str) -> bool:
    left, right = _normalize_math_answer(prediction), _normalize_math_answer(reference)
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        return left == right


def evaluate_gsm8k(
    backend: ModelBackend,
    model_ref: str,
    base_model: str,
    generation_config: dict[str, Any],
    seed: int,
    output_dir: Path,
    test_records: Sequence[PromptRecord],
    max_tasks: int | None = None,
) -> dict[str, Any]:
    prompts = list(test_records[:max_tasks] if max_tasks else test_records)
    completions = backend.generate(
        model_ref, prompts, generation_config, seed, base_model, "math"
    )
    task_rows = [
        {
            "task_id": prompt.task_id,
            "passed": _math_equal(completion, prompt.human_responses[0]),
            "prediction": _normalize_math_answer(completion),
            "reference": _normalize_math_answer(prompt.human_responses[0]),
        }
        for prompt, completion in zip(prompts, completions, strict=True)
    ]
    metrics = {
        "accuracy": float(np.mean([row["passed"] for row in task_rows])),
        "tasks": task_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "gsm8k_metrics.json", metrics)
    return metrics


def evaluate_analytic(model_ref: str) -> dict[str, Any]:
    state = read_json(Path(model_ref) / "analytic_state.json")
    quality = 1.0 - float(state["error"])
    return {
        "pass_at_1": quality,
        "base_pass_at_1": min(1.0, quality + 0.03),
        "worst_skill_pass_at_1": max(0.0, quality - 0.12),
        "by_skill": {"common": quality, "rare": max(0.0, quality - 0.12)},
        "tasks": [],
        "analytic_smoke_only": True,
    }
