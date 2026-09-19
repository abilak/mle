from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pytest

from grounding_mle.generative_analysis import analyze_generative_runs
from grounding_mle.generative_analytic import (
    initial_transition,
    markov_metrics,
    teacher_transition,
)
from grounding_mle.generative_data import _tokenize_records
from grounding_mle.generative_lm import non_padding_token_counts
from grounding_mle.generative_planning import (
    GenerativePlannedRun,
    load_generative_config,
    plan_generative_config,
)
from grounding_mle.generative_runner import run_generative_planned
from grounding_mle.generative_schedules import materialize_generative_schedule
from grounding_mle.generative_vision import _read_idx_images, _read_idx_labels


def test_log_schedule_starts_at_anchor_and_decays() -> None:
    schedule = materialize_generative_schedule(
        "log",
        {"kind": "log_fraction", "power": 1.0, "initial_fraction": 0.25},
        rounds=10,
        default_batch_size=100,
    )
    assert schedule.real_counts[0] == 25
    assert all(
        left >= right
        for left, right in zip(schedule.real_counts, schedule.real_counts[1:])
    )
    assert all(
        real + synthetic == 100
        for real, synthetic in zip(
            schedule.real_counts, schedule.synthetic_counts, strict=True
        )
    )


def test_matched_timing_schedules_have_identical_budgets() -> None:
    schedules = [
        materialize_generative_schedule(
            name,
            {"kind": name, "total_real": 80, "block": True},
            rounds=10,
            default_batch_size=40,
        )
        for name in ("front_loaded", "back_loaded")
    ]
    schedules.append(
        materialize_generative_schedule(
            "uniform",
            {"kind": "uniform_budget", "total_real": 80},
            rounds=10,
            default_batch_size=40,
        )
    )
    assert {sum(schedule.real_counts) for schedule in schedules} == {80}
    assert {sum(schedule.synthetic_counts) for schedule in schedules} == {320}
    assert schedules[0].real_counts[:2] == (40, 40)
    assert schedules[1].real_counts[-2:] == (40, 40)


def test_full_generative_plan_covers_every_requested_family() -> None:
    config = load_generative_config("configs/generative/full.yaml")
    runs = plan_generative_config(config)
    assert len(runs) == 120
    assert {run.experiment for run in runs} == {
        "gpt_boundary",
        "gpt_schedule_sweep",
        "gpt_same_budget_timing",
        "gpt_misspecified",
        "gpt_real_corpus",
        "gpt_model_scale",
        "flow_exact_likelihood",
        "flow_mode_recovery",
        "diffusion_mode_recovery",
    }
    assert len({run.run_id for run in runs}) == len(runs)


def test_markov_metric_detects_initial_error() -> None:
    teacher = teacher_transition(8, 7)
    exact = markov_metrics(teacher, teacher)
    biased = markov_metrics(teacher, initial_transition(teacher, 0.8))
    assert abs(exact["teacher_student_kl_per_token"]) < 1e-12
    assert biased["teacher_student_kl_per_token"] > 0


def test_analytic_run_is_complete_and_idempotent(tmp_path) -> None:
    schedule = materialize_generative_schedule(
        "uniform",
        {"kind": "constant_fraction", "fraction": 0.25},
        rounds=3,
        default_batch_size=16,
    )
    run = GenerativePlannedRun(
        run_id="analytic-test",
        experiment="gpt_boundary",
        condition="uniform",
        seed=11,
        backend="analytic_markov",
        config={
            "project": {"artifact_root": str(tmp_path / "artifacts")},
            "data": {"sequence_length": 12},
            "analytic": {"vocabulary_size": 7},
            "evaluation": {},
        },
        real_counts=schedule.real_counts,
        synthetic_counts=schedule.synthetic_counts,
        schedule_metadata=schedule.metadata,
    )
    state = run_generative_planned(run, tmp_path / "runs")
    repeated = run_generative_planned(run, tmp_path / "runs")
    assert state == repeated
    assert state["status"] == "complete"
    assert state["completed_rounds"] == 3
    assert len(state["trajectory"]) == 4
    assert (tmp_path / "runs" / "analytic-test" / "rounds" / "round_03").exists()


def test_analytic_results_feed_analysis(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    for index, fraction in enumerate((0.1, 0.2, 0.4)):
        schedule = materialize_generative_schedule(
            f"c{index}",
            {"kind": "constant_fraction", "fraction": fraction},
            rounds=3,
            default_batch_size=20,
        )
        run = GenerativePlannedRun(
            run_id=f"analysis-{index}",
            experiment="gpt_schedule_sweep",
            condition=f"c{index}",
            seed=11,
            backend="analytic_markov",
            config={
                "project": {"artifact_root": str(tmp_path / "artifacts")},
                "data": {"sequence_length": 10},
                "analytic": {"vocabulary_size": 6},
                "evaluation": {},
            },
            real_counts=schedule.real_counts,
            synthetic_counts=schedule.synthetic_counts,
            schedule_metadata=schedule.metadata,
        )
        run_generative_planned(run, runs_root)
    manifest = analyze_generative_runs(runs_root, tmp_path / "results")
    assert manifest["completed_runs"] == 3
    assert not manifest["schedule_law"]["skipped"]
    assert (tmp_path / "results" / "trajectory_metrics.csv").exists()
    saved = json.loads((tmp_path / "results" / "analysis_manifest.json").read_text())
    assert saved["kind"] == "grounding-mle-generative-analysis-v1"


def test_idx_vision_loader_parses_images_and_labels(tmp_path) -> None:
    images_path = tmp_path / "images.gz"
    labels_path = tmp_path / "labels.gz"
    images = np.arange(12, dtype=np.uint8).reshape(2, 1, 2, 3)
    labels = np.asarray([4, 9], dtype=np.uint8)
    with gzip.open(images_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, 2, 2, 3))
        handle.write(images.tobytes())
    with gzip.open(labels_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, 2))
        handle.write(labels.tobytes())

    assert np.array_equal(_read_idx_images(images_path), images)
    assert np.array_equal(_read_idx_labels(labels_path), labels)


def test_text_preparation_skips_empty_dataset_rows() -> None:
    class TinyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            assert text
            return {"input_ids": [3, 4]}

    result = _tokenize_records(
        [{"text": ""}, {"other": "missing"}, {"text": "first"}, {"story": "second"}],
        TinyTokenizer(),
        sequence_length=5,
        maximum=2,
        text_fields=["text", "story"],
    )

    assert result.tolist() == [[1, 3, 4, 2, 0], [1, 3, 4, 2, 0]]


def test_non_padding_token_counts_exclude_bos_and_padding() -> None:
    sequences = np.asarray(
        [[1, 3, 2, 0, 0], [1, 4, 5, 2, 0]], dtype=np.int32
    )

    assert non_padding_token_counts(sequences, pad_token_id=0).tolist() == [2, 3]


def test_tiny_neural_lm_train_sample_and_score(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from grounding_mle.generative_lm import (
        sample_lm,
        sequence_log_probabilities,
        train_lm,
    )

    rng = np.random.default_rng(7)
    sequences = rng.integers(3, 32, size=(6, 8), dtype="int32")
    sequences[:, 0] = 1
    sequences[:, -2:] = 0
    checkpoint = tmp_path / "lm"
    summary = train_lm(
        sequences=sequences,
        output_dir=checkpoint,
        architecture={
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 1,
            "dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        training={"max_steps": 1, "batch_size": 2, "mixed_precision": False},
        seed=9,
        vocab_size=32,
    )
    samples = sample_lm(checkpoint, count=2, sequence_length=8, batch_size=2, seed=10)
    scores = sequence_log_probabilities(checkpoint, samples, batch_size=2)
    assert summary["parameters"] > 0
    assert samples.shape == (2, 8)
    assert np.isfinite(scores).all()


def test_tiny_flow_train_sample_and_score(tmp_path) -> None:
    pytest.importorskip("torch")
    from grounding_mle.generative_vision import (
        flow_log_probabilities,
        sample_flow,
        train_flow,
    )

    values = np.random.default_rng(7).normal(size=(12, 4)).astype("float32")
    checkpoint = tmp_path / "flow"
    train_flow(
        transformed_images=values,
        output_dir=checkpoint,
        specification={"dimension": 4, "hidden": 8, "layers": 2},
        training={"max_steps": 1, "batch_size": 4},
        seed=11,
    )
    samples = sample_flow(checkpoint, 3, seed=12)
    scores = flow_log_probabilities(checkpoint, samples)
    assert samples.shape == (3, 4)
    assert np.isfinite(scores).all()


def test_tiny_diffusion_train_and_sample(tmp_path) -> None:
    pytest.importorskip("torch")
    from grounding_mle.generative_vision import sample_diffusion, train_diffusion

    images = np.random.default_rng(13).random((4, 1, 28, 28), dtype=np.float32)
    checkpoint = tmp_path / "diffusion"
    train_diffusion(
        images=images,
        output_dir=checkpoint,
        specification={"hidden": 16, "time_dim": 4, "diffusion_steps": 3},
        training={"max_steps": 1, "batch_size": 2},
        seed=14,
    )
    samples = sample_diffusion(checkpoint, count=2, seed=15, batch_size=2)
    assert samples.shape == (2, 1, 28, 28)
    assert np.isfinite(samples).all()
    assert samples.min() >= 0 and samples.max() <= 1
