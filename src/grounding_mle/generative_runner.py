from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .generative_analytic import (
    fit_markov,
    initial_transition,
    load_transition,
    markov_metrics,
    sample_markov,
    save_transition,
    teacher_transition,
)
from .generative_data import atomic_save_numpy, exclusive_file_lock, load_token_arrays
from .generative_lm import (
    architecture_with_tokens,
    decode_examples,
    ensure_teacher_artifact,
    ensure_teacher_test_set,
    language_distribution_metrics,
    non_padding_token_counts,
    prepared_pad_token_id,
    sample_lm,
    sequence_log_probabilities,
    train_lm,
)
from .generative_planning import GenerativePlannedRun
from .io import atomic_write_json, canonical_json, read_json, runtime_manifest, stable_hash
from .theory import schedule_statistics


def derived_seed(seed: int, stream: str, round_index: int = 0) -> int:
    payload = f"{seed}:{stream}:{round_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _artifact_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["project"].get("artifact_root", "artifacts/generative")))


def _atomic_round_directory(run_dir: Path, round_number: int) -> tuple[Path, Path]:
    target = run_dir / "rounds" / f"round_{round_number:02d}"
    staging = run_dir / "rounds" / f".round_{round_number:02d}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging, target


def _commit_round(staging: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def _clean_uncommitted(run_dir: Path, completed_rounds: int, total_rounds: int) -> None:
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return
    for path in rounds_dir.iterdir():
        if path.name.startswith(".round_") and path.name.endswith(".tmp"):
            shutil.rmtree(path)
            continue
        if path.name.startswith("round_"):
            try:
                number = int(path.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if number > completed_rounds and number <= total_rounds:
                shutil.rmtree(path)


def _load_accumulated(run_dir: Path, completed_rounds: int) -> np.ndarray:
    arrays = []
    for number in range(1, completed_rounds + 1):
        directory = run_dir / "rounds" / f"round_{number:02d}"
        real = np.load(directory / "real.npy", allow_pickle=False)
        synthetic = np.load(directory / "synthetic.npy", allow_pickle=False)
        if len(real):
            arrays.append(real)
        if len(synthetic):
            arrays.append(synthetic)
    if not arrays:
        raise RuntimeError("No committed examples are available for a completed round")
    return np.concatenate(arrays, axis=0)


def _state_template(planned: GenerativePlannedRun) -> dict[str, Any]:
    return {
        "kind": "grounding-mle-generative-state-v1",
        "run_id": planned.run_id,
        "status": "running",
        "completed_rounds": 0,
        "current_checkpoint": None,
        "initial_checkpoint": None,
        "trajectory": [],
        "actual_real": [],
        "actual_synthetic": [],
    }


def _ensure_lm_initial(
    planned: GenerativePlannedRun,
    teacher: Path | None,
    tokenizer_path: Path,
    train_corpus: np.ndarray,
) -> Path:
    config = planned.config
    lm = config["lm"]
    architecture, vocab_size = architecture_with_tokens(lm["student_model"], tokenizer_path)
    identity = {
        "backend": planned.backend,
        "teacher": str(teacher.resolve()) if teacher else None,
        "architecture": architecture,
        "training": lm["initial_training"],
        "examples": int(lm.get("initial_examples", 512)),
        "seed": planned.seed,
        "data": str(config["data"].get("output_dir")),
    }
    manifest_path = Path(str(config["data"].get("output_dir", "data/generative"))) / "manifest.json"
    if teacher is None and manifest_path.exists():
        identity["data_fingerprint"] = read_json(manifest_path)["fingerprint"]
    target = _artifact_root(config) / "initial_students" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "config.json").exists():
            return target
        count = identity["examples"]
        if teacher is not None:
            sequences = sample_lm(
                teacher,
                count=count,
                sequence_length=int(config["data"]["sequence_length"]),
                batch_size=int(config["evaluation"].get("generation_batch_size", 32)),
                seed=derived_seed(planned.seed, "initial-teacher-samples"),
            )
        else:
            rng = np.random.default_rng(derived_seed(planned.seed, "initial-real-samples"))
            indices = rng.choice(len(train_corpus), size=count, replace=count > len(train_corpus))
            sequences = np.asarray(train_corpus[indices], dtype=np.int32)
        train_lm(
            sequences=sequences,
            output_dir=target,
            architecture=architecture,
            training=lm["initial_training"],
            seed=derived_seed(planned.seed, "initial-student-training"),
            vocab_size=vocab_size,
        )
        atomic_write_json(target / "artifact_identity.json", identity)
    return target


def _ensure_reference_student(
    planned: GenerativePlannedRun,
    teacher: Path,
    tokenizer_path: Path,
) -> Path | None:
    config = planned.config
    misspecified = config.get("misspecified", {})
    if not bool(misspecified.get("enabled", False)):
        return None
    lm = config["lm"]
    architecture, vocab_size = architecture_with_tokens(lm["student_model"], tokenizer_path)
    identity = {
        "teacher": str(teacher.resolve()),
        "architecture": architecture,
        "training": misspecified["reference_training"],
        "examples": int(misspecified.get("reference_examples", 20_000)),
        "seed": int(misspecified.get("reference_seed", 20260922)),
    }
    target = _artifact_root(config) / "reference_students" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "config.json").exists():
            return target
        sequences = sample_lm(
            teacher,
            count=identity["examples"],
            sequence_length=int(config["data"]["sequence_length"]),
            batch_size=int(config["evaluation"].get("generation_batch_size", 32)),
            seed=identity["seed"],
        )
        train_lm(
            sequences=sequences,
            output_dir=target,
            architecture=architecture,
            training=misspecified["reference_training"],
            seed=identity["seed"],
            vocab_size=vocab_size,
        )
        atomic_write_json(target / "artifact_identity.json", identity)
    return target


def _evaluate_teacher_lm(
    *,
    checkpoint: Path,
    teacher_samples: np.ndarray,
    teacher_log_probs: np.ndarray,
    reference_log_probs: np.ndarray | None,
    tokenizer_path: Path,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    batch_size = int(evaluation.get("likelihood_batch_size", 32))
    student_log_probs = sequence_log_probabilities(checkpoint, teacher_samples, batch_size=batch_size)
    pad_token_id = prepared_pad_token_id(tokenizer_path)
    token_counts = non_padding_token_counts(teacher_samples, pad_token_id)
    if np.any(token_counts == 0):
        raise RuntimeError("Teacher evaluation samples contain no predicted tokens")
    difference = (teacher_log_probs - student_log_probs) / token_counts
    total_tokens = int(token_counts.sum())
    teacher_ce = float(-teacher_log_probs.sum() / total_tokens)
    student_ce = float(-student_log_probs.sum() / total_tokens)
    diagnostic_count = int(evaluation.get("diagnostic_samples", 512))
    student_samples = sample_lm(
        checkpoint,
        count=diagnostic_count,
        sequence_length=int(config["data"]["sequence_length"]),
        batch_size=int(evaluation.get("generation_batch_size", 32)),
        seed=seed,
    )
    reference_samples = np.asarray(teacher_samples[:diagnostic_count], dtype=np.int32)
    metrics: dict[str, Any] = {
        "teacher_student_kl_per_token": float(np.mean(difference)),
        "teacher_student_kl_se": float(np.std(difference, ddof=1) / math.sqrt(len(difference))),
        "teacher_cross_entropy": teacher_ce,
        "student_cross_entropy": student_ce,
        "student_perplexity": float(math.exp(min(50, student_ce))),
        "test_sequences": int(len(teacher_samples)),
    }
    metrics.update(
        language_distribution_metrics(
            reference_samples, student_samples, pad_token_id=pad_token_id
        )
    )
    if reference_log_probs is not None:
        reference_ce = float(-reference_log_probs.sum() / total_tokens)
        metrics["reference_student_cross_entropy"] = reference_ce
        metrics["excess_cross_entropy"] = student_ce - reference_ce
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output_dir / "diagnostic_samples.npy", student_samples)
    atomic_write_json(
        output_dir / "sample_text.json",
        {
            "teacher": decode_examples(tokenizer_path, reference_samples),
            "student": decode_examples(tokenizer_path, student_samples),
        },
    )
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def _evaluate_real_corpus_lm(
    *,
    checkpoint: Path,
    validation: np.ndarray,
    tokenizer_path: Path,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    count = min(int(evaluation.get("test_sequences", 5000)), len(validation))
    test = np.asarray(validation[:count], dtype=np.int32)
    log_probs = sequence_log_probabilities(
        checkpoint, test, batch_size=int(evaluation.get("likelihood_batch_size", 32))
    )
    pad_token_id = prepared_pad_token_id(tokenizer_path)
    token_counts = non_padding_token_counts(test, pad_token_id)
    total_tokens = int(token_counts.sum())
    if total_tokens == 0:
        raise RuntimeError("Real-corpus evaluation samples contain no predicted tokens")
    cross_entropy = float(-log_probs.sum() / total_tokens)
    diagnostic_count = min(int(evaluation.get("diagnostic_samples", 512)), len(test))
    student_samples = sample_lm(
        checkpoint,
        count=diagnostic_count,
        sequence_length=int(config["data"]["sequence_length"]),
        batch_size=int(evaluation.get("generation_batch_size", 32)),
        seed=seed,
    )
    metrics: dict[str, Any] = {
        "real_cross_entropy": cross_entropy,
        "real_perplexity": float(math.exp(min(50, cross_entropy))),
        "test_sequences": count,
    }
    metrics.update(
        language_distribution_metrics(
            test[:diagnostic_count], student_samples, pad_token_id=pad_token_id
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output_dir / "diagnostic_samples.npy", student_samples)
    atomic_write_json(output_dir / "sample_text.json", {"student": decode_examples(tokenizer_path, student_samples)})
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def _run_lm(planned: GenerativePlannedRun, run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    config = planned.config
    train_corpus, validation, tokenizer_path = load_token_arrays(config["data"])
    teacher: Path | None = None
    teacher_samples: np.ndarray | None = None
    teacher_log_probs: np.ndarray | None = None
    reference_student: Path | None = None
    reference_log_probs: np.ndarray | None = None
    if planned.backend == "teacher_lm":
        teacher = ensure_teacher_artifact(config, _artifact_root(config))
        teacher_samples_raw, teacher_log_probs_raw, _ = ensure_teacher_test_set(
            teacher, config, _artifact_root(config)
        )
        teacher_samples = np.asarray(teacher_samples_raw)
        teacher_log_probs = np.asarray(teacher_log_probs_raw)
        reference_student = _ensure_reference_student(planned, teacher, tokenizer_path)
        if reference_student is not None:
            reference_log_probs = sequence_log_probabilities(
                reference_student,
                teacher_samples,
                batch_size=int(config["evaluation"].get("likelihood_batch_size", 32)),
            )
    initial = _ensure_lm_initial(planned, teacher, tokenizer_path, train_corpus)
    if not state["trajectory"]:
        output = run_dir / "baseline"
        if planned.backend == "teacher_lm":
            assert teacher is not None and teacher_samples is not None and teacher_log_probs is not None
            metrics = _evaluate_teacher_lm(
                checkpoint=initial,
                teacher_samples=teacher_samples,
                teacher_log_probs=teacher_log_probs,
                reference_log_probs=reference_log_probs,
                tokenizer_path=tokenizer_path,
                config=config,
                seed=derived_seed(planned.seed, "baseline-evaluation"),
                output_dir=output,
            )
        else:
            metrics = _evaluate_real_corpus_lm(
                checkpoint=initial,
                validation=validation,
                tokenizer_path=tokenizer_path,
                config=config,
                seed=derived_seed(planned.seed, "baseline-evaluation"),
                output_dir=output,
            )
        state.update(
            {
                "initial_checkpoint": str(initial),
                "current_checkpoint": str(initial),
                "trajectory": [{"round": 0, "metrics": metrics}],
            }
        )
        atomic_write_json(run_dir / "state.json", state)
    completed = int(state["completed_rounds"])
    current = Path(state["current_checkpoint"])
    real_order = np.random.default_rng(derived_seed(planned.seed, "real-corpus-order")).permutation(
        len(train_corpus)
    )
    real_cursor = sum(int(value) for value in state["actual_real"])
    for round_number in range(completed + 1, len(planned.real_counts) + 1):
        real_count = int(planned.real_counts[round_number - 1])
        synthetic_count = int(planned.synthetic_counts[round_number - 1])
        staging, target = _atomic_round_directory(run_dir, round_number)
        if teacher is not None:
            real = sample_lm(
                teacher,
                count=real_count,
                sequence_length=int(config["data"]["sequence_length"]),
                batch_size=int(config["evaluation"].get("generation_batch_size", 32)),
                seed=derived_seed(planned.seed, "round-real", round_number),
            )
        else:
            indices = [real_order[(real_cursor + index) % len(real_order)] for index in range(real_count)]
            real = np.asarray(train_corpus[indices], dtype=np.int32)
            real_cursor += real_count
        synthetic = sample_lm(
            current,
            count=synthetic_count,
            sequence_length=int(config["data"]["sequence_length"]),
            batch_size=int(config["evaluation"].get("generation_batch_size", 32)),
            seed=derived_seed(planned.seed, "round-synthetic", round_number),
        )
        atomic_save_numpy(staging / "real.npy", real)
        atomic_save_numpy(staging / "synthetic.npy", synthetic)
        previous = _load_accumulated(run_dir, completed) if completed else None
        new_values = [array for array in (real, synthetic) if len(array)]
        if previous is not None:
            new_values.insert(0, previous)
        corpus = np.concatenate(new_values, axis=0)
        architecture, vocab_size = architecture_with_tokens(config["lm"]["student_model"], tokenizer_path)
        train_lm(
            sequences=corpus,
            output_dir=staging / "checkpoint",
            architecture=architecture,
            training=config["lm"]["round_training"],
            seed=derived_seed(planned.seed, "round-training", round_number),
            vocab_size=vocab_size,
            source_checkpoint=initial,
        )
        checkpoint = staging / "checkpoint"
        if planned.backend == "teacher_lm":
            assert teacher is not None and teacher_samples is not None and teacher_log_probs is not None
            metrics = _evaluate_teacher_lm(
                checkpoint=checkpoint,
                teacher_samples=teacher_samples,
                teacher_log_probs=teacher_log_probs,
                reference_log_probs=reference_log_probs,
                tokenizer_path=tokenizer_path,
                config=config,
                seed=derived_seed(planned.seed, "round-evaluation", round_number),
                output_dir=staging / "evaluation",
            )
        else:
            metrics = _evaluate_real_corpus_lm(
                checkpoint=checkpoint,
                validation=validation,
                tokenizer_path=tokenizer_path,
                config=config,
                seed=derived_seed(planned.seed, "round-evaluation", round_number),
                output_dir=staging / "evaluation",
            )
        atomic_write_json(
            staging / "round_manifest.json",
            {"round": round_number, "real": real_count, "synthetic": synthetic_count},
        )
        _commit_round(staging, target)
        current = target / "checkpoint"
        state["completed_rounds"] = round_number
        state["current_checkpoint"] = str(current)
        state["actual_real"].append(real_count)
        state["actual_synthetic"].append(synthetic_count)
        state["trajectory"].append(
            {"round": round_number, "real": real_count, "synthetic": synthetic_count, "metrics": metrics}
        )
        atomic_write_json(run_dir / "state.json", state)
        completed = round_number
    return state


def _run_analytic(planned: GenerativePlannedRun, run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    config = planned.config
    analytic = config.get("analytic", {})
    vocabulary = int(analytic.get("vocabulary_size", 12))
    length = int(config["data"].get("sequence_length", 32))
    teacher = teacher_transition(vocabulary, int(analytic.get("teacher_seed", 17)))
    initial = initial_transition(teacher, float(analytic.get("initial_severity", 0.65)))
    initial_path = run_dir / "initial_transition.npy"
    if not initial_path.exists():
        save_transition(initial_path, initial)
    if not state["trajectory"]:
        metrics = markov_metrics(teacher, initial)
        state.update(
            {
                "initial_checkpoint": str(initial_path),
                "current_checkpoint": str(initial_path),
                "trajectory": [{"round": 0, "metrics": metrics}],
            }
        )
        atomic_write_json(run_dir / "state.json", state)
    completed = int(state["completed_rounds"])
    current = load_transition(state["current_checkpoint"])
    for round_number in range(completed + 1, len(planned.real_counts) + 1):
        real_count = int(planned.real_counts[round_number - 1])
        synthetic_count = int(planned.synthetic_counts[round_number - 1])
        staging, target = _atomic_round_directory(run_dir, round_number)
        real = sample_markov(
            teacher, real_count, length, derived_seed(planned.seed, "analytic-real", round_number)
        )
        synthetic = sample_markov(
            current,
            synthetic_count,
            length,
            derived_seed(planned.seed, "analytic-synthetic", round_number),
        )
        atomic_save_numpy(staging / "real.npy", real)
        atomic_save_numpy(staging / "synthetic.npy", synthetic)
        previous = _load_accumulated(run_dir, completed) if completed else None
        arrays = [value for value in (previous, real, synthetic) if value is not None and len(value)]
        corpus = np.concatenate(arrays, axis=0)
        current = fit_markov(corpus, float(analytic.get("smoothing", 0.25)))
        save_transition(staging / "checkpoint.npy", current)
        metrics = markov_metrics(teacher, current)
        atomic_write_json(staging / "evaluation.json", metrics)
        _commit_round(staging, target)
        state["completed_rounds"] = round_number
        state["current_checkpoint"] = str(target / "checkpoint.npy")
        state["actual_real"].append(real_count)
        state["actual_synthetic"].append(synthetic_count)
        state["trajectory"].append(
            {"round": round_number, "real": real_count, "synthetic": synthetic_count, "metrics": metrics}
        )
        atomic_write_json(run_dir / "state.json", state)
        completed = round_number
    return state


def _ensure_flow_initial(
    planned: GenerativePlannedRun,
    *,
    teacher: Path | None,
    images: np.ndarray,
    labels: np.ndarray,
) -> Path:
    from .generative_vision import balanced_indices, logit_transform, sample_flow, train_flow

    config = planned.config
    vision = config["vision"]
    identity = {
        "backend": planned.backend,
        "teacher": str(teacher.resolve()) if teacher else None,
        "model": vision["flow_model"],
        "training": vision["initial_training"],
        "examples": int(vision.get("initial_examples", 2000)),
        "seed": planned.seed,
        "missing_class": int(vision.get("missing_class", 8)),
    }
    target = _artifact_root(config) / "initial_flows" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "model.pt").exists():
            return target
        count = identity["examples"]
        if teacher is not None:
            transformed = sample_flow(
                teacher, count, derived_seed(planned.seed, "initial-flow-teacher")
            )
        else:
            allowed = labels != identity["missing_class"]
            allowed_images, allowed_labels = images[allowed], labels[allowed]
            indices = balanced_indices(
                allowed_labels, count, derived_seed(planned.seed, "initial-flow-real")
            )
            transformed = logit_transform(allowed_images[indices])
        train_flow(
            transformed_images=transformed,
            output_dir=target,
            specification=vision["flow_model"],
            training=vision["initial_training"],
            seed=derived_seed(planned.seed, "initial-flow-training"),
        )
        atomic_write_json(target / "artifact_identity.json", identity)
    return target


def _ensure_flow_test(teacher: Path, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from .generative_vision import flow_log_probabilities, sample_flow

    evaluation = config["evaluation"]
    identity = {
        "teacher": str(teacher.resolve()),
        "count": int(evaluation.get("vision_test_samples", 5000)),
        "seed": int(evaluation.get("test_seed", 20260920)),
    }
    target = _artifact_root(config) / "flow_test_sets" / stable_hash(identity, 16)
    samples_path, log_path = target / "samples.npy", target / "teacher_log_probs.npy"
    with exclusive_file_lock(target.with_suffix(".lock")):
        if not samples_path.exists() or not log_path.exists():
            target.mkdir(parents=True, exist_ok=True)
            samples = sample_flow(teacher, identity["count"], identity["seed"])
            values = flow_log_probabilities(teacher, samples)
            atomic_save_numpy(samples_path, samples)
            atomic_save_numpy(log_path, values)
    return np.load(samples_path, mmap_mode="r"), np.load(log_path, mmap_mode="r")


def _evaluate_flow_teacher(
    checkpoint: Path,
    test_samples: np.ndarray,
    teacher_log_probs: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    from .generative_vision import flow_log_probabilities, inverse_logit, sample_flow

    student = flow_log_probabilities(checkpoint, np.asarray(test_samples))
    difference = np.asarray(teacher_log_probs) - student
    diagnostic = sample_flow(checkpoint, 100, 1907)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output_dir / "sample_images.npy", inverse_logit(diagnostic))
    metrics = {
        "teacher_student_kl": float(np.mean(difference)),
        "teacher_student_kl_se": float(np.std(difference, ddof=1) / math.sqrt(len(difference))),
        "student_nll": float(-np.mean(student)),
        "teacher_nll": float(-np.mean(teacher_log_probs)),
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def _evaluate_mode_model(
    checkpoint: Path,
    *,
    kind: str,
    classifier: Path,
    reference_features: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    from .generative_vision import (
        class_distribution_metrics,
        classify_images,
        frechet_feature_distance,
        inverse_logit,
        sample_diffusion,
        sample_flow,
    )

    count = int(config["evaluation"].get("vision_diagnostic_samples", 2000))
    if kind == "flow":
        images = inverse_logit(sample_flow(checkpoint, count, seed))
    else:
        images = sample_diffusion(checkpoint, count, seed)
    labels, features = classify_images(classifier, images)
    metrics = class_distribution_metrics(
        labels, missing_class=int(config["vision"].get("missing_class", 8))
    )
    metrics["feature_frechet_distance"] = frechet_feature_distance(reference_features, features)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(output_dir / "sample_images.npy", images[:100])
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def _run_vision(planned: GenerativePlannedRun, run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    from .generative_vision import (
        balanced_indices,
        classify_images,
        ensure_flow_teacher,
        ensure_mnist_classifier,
        flatten_image_batch,
        load_mnist_arrays,
        logit_transform,
        sample_diffusion,
        sample_flow,
        train_diffusion,
        train_flow,
    )

    config = planned.config
    images, labels, test_images, test_labels = load_mnist_arrays(config["data"])
    classifier: Path | None = None
    reference_features: np.ndarray | None = None
    teacher: Path | None = None
    flow_test: np.ndarray | None = None
    flow_teacher_log_probs: np.ndarray | None = None
    if planned.backend == "flow_teacher":
        teacher = ensure_flow_teacher(config, _artifact_root(config))
        samples, values = _ensure_flow_test(teacher, config)
        flow_test = np.asarray(samples)
        flow_teacher_log_probs = np.asarray(values)
    else:
        classifier = ensure_mnist_classifier(config, _artifact_root(config))
        classifier_summary = read_json(classifier / "summary.json")
        minimum_accuracy = float(config["vision"].get("classifier_min_accuracy", 0.98))
        if float(classifier_summary["test_accuracy"]) < minimum_accuracy:
            raise RuntimeError(
                "The fixed vision classifier is below the preregistered accuracy threshold: "
                f"{classifier_summary['test_accuracy']:.4f} < {minimum_accuracy:.4f}"
            )
        reference_count = min(
            int(config["evaluation"].get("vision_diagnostic_samples", 2000)), len(test_images)
        )
        reference_indices = balanced_indices(
            test_labels,
            reference_count,
            int(config["evaluation"].get("test_seed", 20260920)),
        )
        _, reference_features = classify_images(classifier, test_images[reference_indices])
    if planned.backend in {"flow_teacher", "flow_mode_recovery"}:
        initial = _ensure_flow_initial(
            planned, teacher=teacher, images=images, labels=labels
        )
        model_kind = "flow"
    else:
        vision = config["vision"]
        missing = int(vision.get("missing_class", 8))
        identity = {
            "model": vision["diffusion_model"],
            "training": vision["initial_training"],
            "examples": int(vision.get("initial_examples", 5000)),
            "missing": missing,
            "seed": planned.seed,
        }
        initial = _artifact_root(config) / "initial_diffusions" / stable_hash(identity, 16)
        with exclusive_file_lock(initial.with_suffix(".lock")):
            if not (initial / "model.pt").exists():
                allowed = labels != missing
                indices = balanced_indices(
                    labels[allowed], identity["examples"], derived_seed(planned.seed, "diffusion-initial")
                )
                train_diffusion(
                    images=images[allowed][indices],
                    output_dir=initial,
                    specification=vision["diffusion_model"],
                    training=vision["initial_training"],
                    seed=derived_seed(planned.seed, "diffusion-initial-training"),
                )
        model_kind = "diffusion"
    if not state["trajectory"]:
        if planned.backend == "flow_teacher":
            assert flow_test is not None and flow_teacher_log_probs is not None
            metrics = _evaluate_flow_teacher(initial, flow_test, flow_teacher_log_probs, run_dir / "baseline")
        else:
            assert classifier is not None and reference_features is not None
            metrics = _evaluate_mode_model(
                initial,
                kind=model_kind,
                classifier=classifier,
                reference_features=reference_features,
                config=config,
                seed=derived_seed(planned.seed, "vision-baseline"),
                output_dir=run_dir / "baseline",
            )
        state.update(
            {
                "initial_checkpoint": str(initial),
                "current_checkpoint": str(initial),
                "trajectory": [{"round": 0, "metrics": metrics}],
            }
        )
        atomic_write_json(run_dir / "state.json", state)
    completed = int(state["completed_rounds"])
    current = Path(state["current_checkpoint"])
    for round_number in range(completed + 1, len(planned.real_counts) + 1):
        real_count = int(planned.real_counts[round_number - 1])
        synthetic_count = int(planned.synthetic_counts[round_number - 1])
        staging, target = _atomic_round_directory(run_dir, round_number)
        if planned.backend == "flow_teacher":
            assert teacher is not None
            real = sample_flow(teacher, real_count, derived_seed(planned.seed, "flow-real", round_number))
            synthetic = sample_flow(
                current, synthetic_count, derived_seed(planned.seed, "flow-synthetic", round_number)
            )
        else:
            indices = balanced_indices(
                labels, real_count, derived_seed(planned.seed, "vision-real", round_number)
            )
            real_images = images[indices]
            if model_kind == "flow":
                real = flatten_image_batch(logit_transform(real_images))
                synthetic = sample_flow(
                    current, synthetic_count, derived_seed(planned.seed, "flow-mode-synthetic", round_number)
                )
            else:
                real = real_images
                synthetic = sample_diffusion(
                    current,
                    synthetic_count,
                    derived_seed(planned.seed, "diffusion-synthetic", round_number),
                )
        atomic_save_numpy(staging / "real.npy", real)
        atomic_save_numpy(staging / "synthetic.npy", synthetic)
        previous = _load_accumulated(run_dir, completed) if completed else None
        arrays = [value for value in (previous, real, synthetic) if value is not None and len(value)]
        corpus = np.concatenate(arrays, axis=0)
        if model_kind == "flow":
            train_flow(
                transformed_images=corpus,
                output_dir=staging / "checkpoint",
                specification=config["vision"]["flow_model"],
                training=config["vision"]["round_training"],
                seed=derived_seed(planned.seed, "flow-round-training", round_number),
                source_checkpoint=initial,
            )
        else:
            train_diffusion(
                images=corpus,
                output_dir=staging / "checkpoint",
                specification=config["vision"]["diffusion_model"],
                training=config["vision"]["round_training"],
                seed=derived_seed(planned.seed, "diffusion-round-training", round_number),
                source_checkpoint=initial,
            )
        checkpoint = staging / "checkpoint"
        if planned.backend == "flow_teacher":
            assert flow_test is not None and flow_teacher_log_probs is not None
            metrics = _evaluate_flow_teacher(
                checkpoint, flow_test, flow_teacher_log_probs, staging / "evaluation"
            )
        else:
            assert classifier is not None and reference_features is not None
            metrics = _evaluate_mode_model(
                checkpoint,
                kind=model_kind,
                classifier=classifier,
                reference_features=reference_features,
                config=config,
                seed=derived_seed(planned.seed, "vision-round-evaluation", round_number),
                output_dir=staging / "evaluation",
            )
        _commit_round(staging, target)
        current = target / "checkpoint"
        state["completed_rounds"] = round_number
        state["current_checkpoint"] = str(current)
        state["actual_real"].append(real_count)
        state["actual_synthetic"].append(synthetic_count)
        state["trajectory"].append(
            {"round": round_number, "real": real_count, "synthetic": synthetic_count, "metrics": metrics}
        )
        atomic_write_json(run_dir / "state.json", state)
        completed = round_number
    return state


def run_generative_planned(
    planned: GenerativePlannedRun,
    output_root: str | Path = "runs/generative",
    *,
    resume: bool = True,
) -> dict[str, Any]:
    run_dir = Path(output_root) / planned.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    resolved_path = run_dir / "resolved_run.json"
    resolved = planned.to_dict()
    if resolved_path.exists():
        if canonical_json(read_json(resolved_path)) != canonical_json(resolved):
            raise RuntimeError(f"Run identity changed for existing directory {run_dir}")
    else:
        atomic_write_json(resolved_path, resolved)
        atomic_write_json(run_dir / "runtime.json", runtime_manifest())
    if state_path.exists() and not resume:
        raise RuntimeError(f"Run already exists: {run_dir}; omit --no-resume to continue")
    state = read_json(state_path) if state_path.exists() else _state_template(planned)
    if state.get("status") == "complete":
        return state
    completed = int(state.get("completed_rounds", 0))
    _clean_uncommitted(run_dir, completed, len(planned.real_counts))
    if planned.backend == "analytic_markov":
        state = _run_analytic(planned, run_dir, state)
    elif planned.backend in {"teacher_lm", "real_corpus_lm"}:
        state = _run_lm(planned, run_dir, state)
    elif planned.backend in {"flow_teacher", "flow_mode_recovery", "diffusion_mode_recovery"}:
        state = _run_vision(planned, run_dir, state)
    else:
        raise ValueError(f"Unsupported generative backend: {planned.backend}")
    state["status"] = "complete"
    state["theory"] = schedule_statistics(state["actual_real"], state["actual_synthetic"]).to_dict()
    atomic_write_json(state_path, state)
    return read_json(state_path)
