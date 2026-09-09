from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluation import (
    evaluate_analytic,
    evaluate_evalplus,
    evaluate_gsm8k,
)
from .io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    runtime_manifest,
    stable_hash,
)
from .modeling import ModelBackend, make_backend
from .planning import PlannedRun
from .records import PromptRecord, TrainingExample
from .theory import schedule_statistics
from .verification import verify_candidates


def _derived_seed(seed: int, stream: str, round_index: int, attempt: int = 0) -> int:
    payload = f"{seed}:{stream}:{round_index}:{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _ordered_pool(pool: Sequence[PromptRecord], seed: int, stream: str) -> list[PromptRecord]:
    rng = np.random.default_rng(_derived_seed(seed, stream, 0))
    return [pool[int(index)] for index in rng.permutation(len(pool))]


def _cyclic_slice(pool: Sequence[PromptRecord], start: int, count: int) -> list[PromptRecord]:
    if not pool and count:
        raise ValueError("Cannot draw from an empty prompt pool")
    return [pool[(start + offset) % len(pool)] for offset in range(count)]


def _human_examples(
    prompts: Sequence[PromptRecord], round_index: int, seed: int
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for index, prompt in enumerate(prompts):
        if not prompt.human_responses:
            raise ValueError(f"No human solution for {prompt.task_id}")
        response_index = _derived_seed(seed, "human-response", round_index, index) % len(
            prompt.human_responses
        )
        examples.append(
            TrainingExample(
                example_id=f"real-r{round_index}-{prompt.task_id}-{index}",
                prompt=prompt.prompt,
                response=prompt.human_responses[response_index],
                source="real",
                round_index=round_index,
                task_id=prompt.task_id,
                tests=prompt.tests,
                skill=prompt.skill,
                metadata=prompt.metadata,
            )
        )
    return examples


def _synthetic_examples(
    prompts: Sequence[PromptRecord],
    responses: Sequence[str],
    verification_reasons: Sequence[str],
    round_index: int,
) -> list[TrainingExample]:
    return [
        TrainingExample(
            example_id=f"synthetic-r{round_index}-{prompt.task_id}-{index}",
            prompt=prompt.prompt,
            response=response,
            source="synthetic",
            round_index=round_index,
            task_id=prompt.task_id,
            tests=prompt.tests,
            skill=prompt.skill,
            metadata=prompt.metadata | {"verification": reason},
        )
        for index, (prompt, response, reason) in enumerate(
            zip(prompts, responses, verification_reasons, strict=True)
        )
    ]


def _load_prompt_records(path: str | Path) -> list[PromptRecord]:
    return [PromptRecord.from_dict(row) for row in read_jsonl(path)]


def _generate_accepted(
    *,
    backend: ModelBackend,
    model_ref: str,
    base_model: str,
    prompts: Sequence[PromptRecord],
    count: int,
    round_index: int,
    seed: int,
    generation_config: dict[str, Any],
    verification_config: dict[str, Any],
    domain: str,
) -> tuple[list[PromptRecord], list[str], list[str], int, list[dict[str, Any]]]:
    if count == 0:
        return [], [], [], 0, []
    accepted_prompts: list[PromptRecord] = []
    accepted_outputs: list[str] = []
    reasons: list[str] = []
    attempts = 0
    audit: list[dict[str, Any]] = []
    maximum_attempts = max(count, int(count * float(verification_config.get("max_attempt_multiplier", 8))))
    cursor = 0
    while len(accepted_outputs) < count and attempts < maximum_attempts:
        remaining = count - len(accepted_outputs)
        batch_size = min(
            remaining,
            int(generation_config.get("batch_size", 4)),
            maximum_attempts - attempts,
        )
        batch_prompts = [prompts[(cursor + offset) % len(prompts)] for offset in range(batch_size)]
        attempt_seed = _derived_seed(seed, "synthetic-generate", round_index, attempts)
        outputs = backend.generate(
            model_ref,
            batch_prompts,
            generation_config,
            attempt_seed,
            base_model,
            domain,
        )
        if domain == "code":
            decisions = verify_candidates(
                str(verification_config.get("mode", "none")),
                batch_prompts,
                outputs,
                backend=backend,
                model_ref=model_ref,
                base_model=base_model,
                generation_config=generation_config,
                seed=attempt_seed,
                docker_image=str(verification_config.get("docker_image", "python:3.11-slim")),
            )
        else:
            decisions = [type("Decision", (), {"accepted": True, "reason": "unfiltered"})()] * len(outputs)
        for prompt, output, decision in zip(batch_prompts, outputs, decisions, strict=True):
            attempts += 1
            retained = bool(decision.accepted and len(accepted_outputs) < count)
            audit.append(
                {
                    "attempt": attempts,
                    "round": round_index,
                    "task_id": prompt.task_id,
                    "accepted": retained,
                    "verification_reason": decision.reason,
                    "generation_seed": attempt_seed,
                    "response": output,
                }
            )
            if retained:
                accepted_prompts.append(prompt)
                accepted_outputs.append(output)
                reasons.append(decision.reason)
        cursor += batch_size
    if len(accepted_outputs) != count:
        raise RuntimeError(
            f"Verification retained {len(accepted_outputs)}/{count} required synthetic examples "
            f"after {attempts} attempts"
        )
    return accepted_prompts, accepted_outputs, reasons, attempts, audit


def _monitor_metrics(
    backend: ModelBackend,
    model_ref: str,
    base_model: str,
    monitor: Sequence[PromptRecord],
    config: dict[str, Any],
    seed: int,
    round_index: int,
    domain: str,
) -> dict[str, Any]:
    maximum = int(config["evaluation"].get("monitor_tasks", 64))
    prompts = list(monitor[:maximum])
    outputs = backend.generate(
        model_ref,
        prompts,
        config["generation"],
        _derived_seed(seed, "monitor", round_index),
        base_model,
        domain,
    )
    if domain == "code":
        decisions = verify_candidates(
            str(config["evaluation"].get("monitor_verification", "external_tests")),
            prompts,
            outputs,
            backend=backend,
            model_ref=model_ref,
            base_model=base_model,
            generation_config=config["generation"],
            seed=_derived_seed(seed, "monitor-verify", round_index),
            docker_image=str(config["verification"].get("docker_image", "python:3.11-slim")),
        )
        passed = [decision.accepted for decision in decisions]
    else:
        from .evaluation import _math_equal

        passed = [
            _math_equal(output, prompt.human_responses[0])
            for prompt, output in zip(prompts, outputs, strict=True)
        ]
    by_skill: dict[str, list[bool]] = defaultdict(list)
    for prompt, value in zip(prompts, passed, strict=True):
        by_skill[prompt.skill].append(value)
    return {
        "accuracy": float(np.mean(passed)) if passed else float("nan"),
        "by_skill": {
            skill: float(np.mean(values)) for skill, values in sorted(by_skill.items())
        },
        "tasks": len(prompts),
        "source": "training-disjoint monitor split",
    }


def _evaluate_checkpoint(
    backend: ModelBackend,
    model_ref: str,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    domain: str,
    math_test: Sequence[PromptRecord] | None,
) -> dict[str, Any]:
    base_model = str(config["model"]["name"])
    if config["model"]["backend"] == "analytic":
        metrics = {"analytic": evaluate_analytic(model_ref)}
    elif domain == "code":
        metrics = {}
        for dataset in config["evaluation"].get("benchmarks", ["humaneval", "mbpp"]):
            metrics[dataset] = evaluate_evalplus(
                backend,
                model_ref,
                base_model,
                config["generation"] | {"do_sample": False},
                config["evaluation"],
                seed,
                output_dir / dataset,
                str(dataset),
            )
    else:
        if math_test is None:
            raise ValueError("Math evaluation requires a test dataset")
        metrics = {
            "gsm8k": evaluate_gsm8k(
                backend,
                model_ref,
                base_model,
                config["generation"] | {"do_sample": False},
                seed,
                output_dir,
                math_test,
                config["evaluation"].get("max_tasks"),
            )
        }
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_planned(planned: PlannedRun, output_root: str | Path, resume: bool = True) -> dict[str, Any]:
    config = planned.config
    run_dir = Path(output_root) / planned.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    corpus_path = run_dir / "corpus.jsonl"
    config_path = run_dir / "resolved_run.json"
    if not config_path.exists():
        atomic_write_json(config_path, planned.to_dict())
        atomic_write_json(run_dir / "runtime.json", runtime_manifest())
    else:
        if canonical_json(read_json(config_path)) != canonical_json(planned.to_dict()):
            raise RuntimeError(
                f"Existing run directory does not match plan entry {planned.run_id}"
            )
        if not resume:
            raise FileExistsError(f"Run directory already exists: {run_dir}")

    domain = str(config["training"].get("domain", "code"))
    processed = Path(config["data"]["processed_dir"])
    train_name = "math_train.jsonl" if domain == "math" else "code_train.jsonl"
    monitor_name = "math_monitor.jsonl" if domain == "math" else "code_monitor.jsonl"
    test_name = "math_test.jsonl"
    train_pool = _load_prompt_records(processed / train_name)
    monitor_pool = _ordered_pool(
        _load_prompt_records(processed / monitor_name), planned.seed, "monitor-pool"
    )
    math_test = _load_prompt_records(processed / test_name) if domain == "math" else None
    real_pool = _ordered_pool(train_pool, planned.seed, "real-pool")
    synthetic_pool = _ordered_pool(train_pool, planned.seed, "synthetic-pool")
    total_real_planned = sum(planned.real_counts)
    if total_real_planned > len(real_pool) and not bool(config["data"].get("allow_real_reuse", False)):
        raise ValueError(
            f"Schedule requires {total_real_planned} fresh real examples, but only {len(real_pool)} are available"
        )

    backend = make_backend(str(config["model"]["backend"]))
    base_model = str(config["model"]["name"])
    current_model = base_model
    start_round = 0
    real_cursor = 0
    synthetic_cursor = 0
    actual_real: list[int] = []
    actual_synthetic: list[int] = []
    trajectory: list[dict[str, Any]] = []
    corpus = []
    previous_monitor: dict[str, Any] | None = None
    remaining_reactive_budget = total_real_planned
    if state_path.exists():
        state = read_json(state_path)
        if state.get("status") == "complete":
            return state
        start_round = int(state["next_round"])
        current_model = str(state["current_model"])
        real_cursor = int(state["real_cursor"])
        synthetic_cursor = int(state["synthetic_cursor"])
        actual_real = [int(x) for x in state["actual_real"]]
        actual_synthetic = [int(x) for x in state["actual_synthetic"]]
        trajectory = list(state["trajectory"])
        previous_monitor = state.get("previous_monitor")
        remaining_reactive_budget = int(state.get("remaining_reactive_budget", 0))
        corpus_rows = list(read_jsonl(corpus_path))
        committed_examples = sum(actual_real) + sum(actual_synthetic)
        if len(corpus_rows) < committed_examples:
            raise RuntimeError(
                "The corpus is shorter than the last committed round; the run cannot be resumed safely"
            )
        if len(corpus_rows) > committed_examples:
            corpus_rows = corpus_rows[:committed_examples]
            atomic_write_jsonl(corpus_path, corpus_rows)
        corpus = [TrainingExample(**row) for row in corpus_rows]
    elif corpus_path.exists():
        # A crash before the first state commit can leave an uncommitted tail.
        atomic_write_jsonl(corpus_path, [])

    if not trajectory:
        baseline_dir = run_dir / "evaluation" / "round_00"
        if config["model"]["backend"] == "analytic":
            baseline_model_dir = run_dir / "checkpoints" / "round_00"
            baseline_model_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(baseline_model_dir / "analytic_state.json", {"error": 0.25})
            current_model = str(baseline_model_dir)
        baseline = _evaluate_checkpoint(
            backend,
            current_model,
            config,
            planned.seed,
            baseline_dir,
            domain,
            math_test,
        )
        trajectory.append({"round": 0, "evaluation": baseline, "phase": "baseline"})
        atomic_write_json(
            state_path,
            {
                "status": "running",
                "run_id": planned.run_id,
                "next_round": 0,
                "current_model": current_model,
                "real_cursor": 0,
                "synthetic_cursor": 0,
                "actual_real": [],
                "actual_synthetic": [],
                "remaining_reactive_budget": remaining_reactive_budget,
                "previous_monitor": None,
                "trajectory": trajectory,
            },
        )

    strategy = str(config.get("synthetic", {}).get("strategy", "passive"))
    static_cache: list[TrainingExample] | None = None
    static_path = run_dir / "static_synthetic.jsonl"
    if strategy == "static" and static_path.exists():
        static_cache = [TrainingExample(**row) for row in read_jsonl(static_path)]
    elif strategy == "static" and start_round == 0:
        total_synthetic = sum(planned.synthetic_counts)
        prompts = _cyclic_slice(synthetic_pool, 0, total_synthetic)
        kept_prompts, responses, reasons, attempts, audit = _generate_accepted(
            backend=backend,
            model_ref=current_model,
            base_model=base_model,
            prompts=prompts,
            count=total_synthetic,
            round_index=-1,
            seed=planned.seed,
            generation_config=config["generation"],
            verification_config=config["verification"],
            domain=domain,
        )
        static_cache = _synthetic_examples(kept_prompts, responses, reasons, -1)
        atomic_write_jsonl(static_path, (row.to_dict() for row in static_cache))
        atomic_write_jsonl(run_dir / "static_generation_attempts.jsonl", audit)
        atomic_write_json(run_dir / "static_synthetic_summary.json", {"attempts": attempts, "retained": len(static_cache)})
    elif strategy == "static":
        raise RuntimeError(f"Static synthetic cache is missing for resumed run {planned.run_id}")

    policy = config.get("policy", {})
    for round_index in range(start_round, len(planned.real_counts)):
        real_count = int(planned.real_counts[round_index])
        if policy.get("kind") == "reactive":
            threshold = float(policy.get("threshold", 0.55))
            chunk = int(policy.get("chunk", max(1, total_real_planned // len(planned.real_counts))))
            should_inject = previous_monitor is None or float(previous_monitor["accuracy"]) < threshold
            real_count = min(chunk, remaining_reactive_budget) if should_inject else 0
            remaining_reactive_budget -= real_count
        synthetic_count = int(planned.synthetic_counts[round_index])
        real_prompts = _cyclic_slice(real_pool, real_cursor, real_count)
        real_examples = _human_examples(real_prompts, round_index, planned.seed)
        real_cursor += real_count

        if strategy == "static":
            assert static_cache is not None
            selected = static_cache[synthetic_cursor : synthetic_cursor + synthetic_count]
            synthetic_examples = [
                TrainingExample(**(item.to_dict() | {"round_index": round_index})) for item in selected
            ]
            attempts = synthetic_count
            audit = []
        else:
            if strategy == "failure_targeted" and previous_monitor and previous_monitor.get("by_skill"):
                weakest = min(previous_monitor["by_skill"], key=previous_monitor["by_skill"].get)
                targeted = [row for row in synthetic_pool if row.skill == weakest]
                fallback = [row for row in synthetic_pool if row.skill != weakest]
                ordered_synthetic = targeted + fallback
            else:
                ordered_synthetic = synthetic_pool
            prompt_candidates = _cyclic_slice(
                ordered_synthetic,
                synthetic_cursor,
                max(synthetic_count, 1)
                * int(config["verification"].get("max_attempt_multiplier", 8)),
            )
            kept_prompts, responses, reasons, attempts, audit = _generate_accepted(
                backend=backend,
                model_ref=current_model,
                base_model=base_model,
                prompts=prompt_candidates,
                count=synthetic_count,
                round_index=round_index,
                seed=planned.seed,
                generation_config=config["generation"],
                verification_config=config["verification"],
                domain=domain,
            )
            synthetic_examples = _synthetic_examples(
                kept_prompts, responses, reasons, round_index
            )
            atomic_write_jsonl(
                run_dir / "generation" / f"round_{round_index + 1:02d}_attempts.jsonl",
                audit,
            )
        synthetic_cursor += synthetic_count
        new_batch = real_examples + synthetic_examples
        corpus.extend(new_batch)
        append_jsonl(corpus_path, (example.to_dict() for example in new_batch))

        training_mode = str(config["training"]["mode"])
        if training_mode == "refit_full":
            source_model, training_examples = base_model, corpus
        elif training_mode == "continual_new":
            source_model, training_examples = current_model, new_batch
        elif training_mode == "continual_full":
            source_model, training_examples = current_model, corpus
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        checkpoint = run_dir / "checkpoints" / f"round_{round_index + 1:02d}"
        current_model = backend.train(
            source_model,
            training_examples,
            checkpoint,
            config["training"],
            _derived_seed(planned.seed, "train", round_index),
            base_model,
        )

        monitor_required = (
            policy.get("kind") == "reactive"
            or strategy == "failure_targeted"
            or bool(config["evaluation"].get("monitor_every_round", False))
        )
        if monitor_required and config["model"]["backend"] == "analytic":
            analytic = evaluate_analytic(current_model)
            previous_monitor = {
                "accuracy": analytic["pass_at_1"],
                "by_skill": analytic["by_skill"],
                "tasks": 0,
                "source": "analytic smoke backend",
            }
        elif monitor_required:
            previous_monitor = _monitor_metrics(
                backend,
                current_model,
                base_model,
                monitor_pool,
                config,
                planned.seed,
                round_index,
                domain,
            )
        else:
            previous_monitor = None
        evaluate_every = int(config["evaluation"].get("every_rounds", 1))
        is_final = round_index + 1 == len(planned.real_counts)
        evaluation = None
        if is_final or (round_index + 1) % evaluate_every == 0:
            evaluation = _evaluate_checkpoint(
                backend,
                current_model,
                config,
                planned.seed,
                run_dir / "evaluation" / f"round_{round_index + 1:02d}",
                domain,
                math_test,
            )
        actual_real.append(real_count)
        actual_synthetic.append(synthetic_count)
        trajectory.append(
            {
                "round": round_index + 1,
                "real": real_count,
                "synthetic": synthetic_count,
                "synthetic_attempts": attempts,
                "checkpoint": current_model,
                "monitor": previous_monitor,
                "evaluation": evaluation,
            }
        )
        state = {
            "status": "running",
            "run_id": planned.run_id,
            "next_round": round_index + 1,
            "current_model": current_model,
            "real_cursor": real_cursor,
            "synthetic_cursor": synthetic_cursor,
            "actual_real": actual_real,
            "actual_synthetic": actual_synthetic,
            "remaining_reactive_budget": remaining_reactive_budget,
            "previous_monitor": previous_monitor,
            "trajectory": trajectory,
        }
        atomic_write_json(state_path, state)

    final_state = read_json(state_path)
    final_state["status"] = "complete"
    final_state["theory"] = schedule_statistics(actual_real, actual_synthetic).to_dict()
    final_state["corpus_examples"] = len(corpus)
    final_state["corpus_hash"] = stable_hash([example.to_dict() for example in corpus], 32)
    final_state["law_split"] = planned.law_split
    atomic_write_json(state_path, final_state)
    return final_state
