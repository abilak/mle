from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pandas as pd
import pytest

from grounding_mle.generative_analysis import (
    _confirmatory_contrast_table,
    _confirmatory_timing_analysis,
    _confirmatory_vision_controls,
    _exact_paired_sign_flip_pvalue,
    _holm_adjusted_pvalues,
    _paired_effects,
    _primary_metrics,
    analyze_generative_runs,
)
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
from grounding_mle.generative_vision import (
    _read_idx_images,
    _read_idx_labels,
    class_distribution_metrics,
    diffusion_beta_values,
    flatten_image_batch,
    frechet_feature_distance,
    inverse_logit,
    preprocess_flow_images,
)
from grounding_mle.generative_vision_repair import evaluate_repair


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


def test_vision_repair_plan_is_separate_and_uses_fixed_backends() -> None:
    config = load_generative_config("configs/generative/vision_repair.yaml")
    runs = plan_generative_config(config)

    assert len(runs) == 12
    assert {run.experiment for run in runs} == {
        "flow_mode_recovery",
        "diffusion_mode_recovery",
    }
    assert {run.condition for run in runs} == {"all_real", "no_real"}
    assert {run.seed for run in runs} == {211, 223, 227}
    assert {len(run.real_counts) for run in runs} == {1}
    flow = next(run for run in runs if run.backend == "flow_mode_recovery")
    diffusion = next(
        run for run in runs if run.backend == "diffusion_mode_recovery"
    )
    assert flow.config["vision"]["flow_preprocessing"]["dequantize"]
    assert flow.config["vision"]["flow_model"]["architecture"] == "conv"
    assert diffusion.config["vision"]["diffusion_model"]["noise_schedule"] == "cosine"
    assert diffusion.config["vision"]["diffusion_model"]["architecture"] == "conv"


def test_confirmatory_plan_reuses_original_runs_and_fixes_timing_budget() -> None:
    full_runs = plan_generative_config(
        load_generative_config("configs/generative/full.yaml")
    )
    confirmatory_config = load_generative_config(
        "configs/generative/confirmatory.yaml"
    )
    confirmatory_runs = plan_generative_config(confirmatory_config)
    full_ids = {
        (run.experiment, run.condition, run.seed): run.run_id for run in full_runs
    }
    reused = [
        run
        for run in confirmatory_runs
        if full_ids.get((run.experiment, run.condition, run.seed)) == run.run_id
    ]

    assert len(confirmatory_runs) == 257
    assert len(reused) == 45
    assert len(confirmatory_runs) - len(reused) == 212
    confirmatory_seeds = set(
        confirmatory_config["confirmatory_analysis"]["planned_seeds"]
    )
    assert {run.seed for run in reused} == {11, 23, 37}
    assert not ({run.seed for run in reused} & confirmatory_seeds)
    timing_runs = [
        run
        for run in confirmatory_runs
        if run.experiment == "gpt_same_budget_timing"
    ]
    assert len(timing_runs) == 89
    assert {sum(run.real_counts) for run in timing_runs} == {512}
    assert {sum(run.synthetic_counts) for run in timing_runs} == {2048}


def test_holm_adjustment_and_complete_confirmatory_contrasts() -> None:
    assert _holm_adjusted_pvalues([0.01, 0.03, 0.02]).tolist() == pytest.approx(
        [0.03, 0.04, 0.04]
    )
    seeds = [41, 53, 67, 79, 83, 97, 109, 127, 139, 151]
    rows = []
    for experiment in ("first", "second"):
        for seed in seeds:
            for condition, value in (("better", 1.0), ("worse", 2.0)):
                rows.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "seed": seed,
                        "primary_metric": "loss",
                        "primary_value": value,
                        "higher_is_better": False,
                    }
                )
    specification = {
        "alpha": 0.05,
        "planned_seeds": seeds,
        "contrasts": [
            {
                "id": experiment,
                "family": "primary",
                "experiment": experiment,
                "left": "better",
                "right": "worse",
                "expected_favored": "better",
            }
            for experiment in ("first", "second")
        ],
    }

    result = _confirmatory_contrast_table(pd.DataFrame(rows), specification)

    assert result["complete"].all()
    assert result["family_complete"].all()
    assert result["exact_sign_flip_pvalue"].tolist() == pytest.approx(
        [2 / 2**10, 2 / 2**10]
    )
    assert result["holm_adjusted_pvalue"].tolist() == pytest.approx(
        [4 / 2**10, 4 / 2**10]
    )
    assert set(result["decision"]) == {"supports_expected_direction"}

    incomplete = pd.DataFrame(rows)
    incomplete = incomplete[
        ~(
            (incomplete["experiment"] == "second")
            & (incomplete["condition"] == "worse")
            & (incomplete["seed"] == 109)
        )
    ]
    pending = _confirmatory_contrast_table(incomplete, specification)
    assert not pending["family_complete"].any()
    assert pending["exact_sign_flip_pvalue"].isna().all()
    assert pending["holm_adjusted_pvalue"].isna().all()
    assert set(pending["decision"]) == {"pending"}


def test_confirmatory_timing_uses_within_seed_fixed_budget_slopes() -> None:
    seeds = [41, 53, 67, 79, 83, 97, 109, 127, 139, 151]
    conditions = ["back", "uniform", "front"]
    rows = []
    for seed in seeds:
        for condition, restoring_mass in zip(
            conditions, (0.2, 0.7, 1.2), strict=True
        ):
            rows.append(
                {
                    "experiment": "timing",
                    "condition": condition,
                    "seed": seed,
                    "G_T": restoring_mass,
                    "primary_value": 3.0 - restoring_mass + seed * 1e-6,
                    "total_real": 512,
                }
            )
    specification = {
        "experiment": "timing",
        "conditions": conditions,
        "planned_seeds": seeds,
        "total_real": 512,
        "predictor": "G_T",
        "outcome": "primary_value",
        "expected_slope_sign": "negative",
    }

    by_seed, summary = _confirmatory_timing_analysis(
        pd.DataFrame(rows), specification
    )

    assert by_seed["complete"].all()
    assert by_seed["slope"].tolist() == pytest.approx([-1.0] * 10)
    assert summary["complete"]
    assert summary["exact_sign_flip_pvalue"] == pytest.approx(2 / 2**10)
    assert summary["decision"] == "supports_expected_direction"


def test_confirmatory_vision_control_requires_complete_quantitative_recovery() -> None:
    rows = []
    for seed in (11, 23, 37):
        for condition, values in {
            "all_real": (0.01, 0.02, 3.0, 2.2),
            "no_real": (0.08, 0.20, 12.0, 1.1),
        }.items():
            rows.append(
                {
                    "experiment": "vision_control",
                    "condition": condition,
                    "seed": seed,
                    "primary_value": values[0],
                    "class_kl_to_uniform": values[1],
                    "feature_frechet_distance": values[2],
                    "class_entropy": values[3],
                }
            )
    specification = [
        {
            "experiment": "vision_control",
            "planned_seeds": [11, 23, 37],
            "positive_condition": "all_real",
            "negative_condition": "no_real",
        }
    ]

    result = _confirmatory_vision_controls(pd.DataFrame(rows), specification)

    assert result.loc[0, "complete"]
    assert result.loc[0, "quantitative_pass"]
    assert result.loc[0, "visual_review_required"]
    assert result.loc[0, "status"] == "quantitative_pass_requires_visual_review"

    incomplete = pd.DataFrame(rows[:-1])
    pending = _confirmatory_vision_controls(incomplete, specification)
    assert not pending.loc[0, "complete"]
    assert not pending.loc[0, "quantitative_pass"]
    assert pending.loc[0, "status"] == "pending"


def test_vision_repair_gate_requires_absolute_and_relative_validity(tmp_path) -> None:
    rows = []
    for experiment in ("flow_mode_recovery", "diffusion_mode_recovery"):
        for seed in (211, 223, 227):
            for condition, values in {
                "all_real": (0.02, 0.08, 4.0, 2.15),
                "no_real": (0.09, 0.70, 20.0, 1.50),
            }.items():
                rows.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "seed": seed,
                        "primary_value": values[0],
                        "class_kl_to_uniform": values[1],
                        "feature_frechet_distance": values[2],
                        "class_entropy": values[3],
                    }
                )
    results = tmp_path / "results"
    results.mkdir()
    table = pd.DataFrame(rows)
    table.to_csv(results / "final_metrics.csv", index=False)
    specification = {
        "posthoc_exploratory": True,
        "planned_seeds": [211, 223, 227],
        "positive_condition": "all_real",
        "negative_condition": "no_real",
        "absolute_gates": {
            "all_real_primary_mean_max": 0.05,
            "all_real_primary_seed_max": 0.10,
            "all_real_class_kl_mean_max": 0.35,
            "all_real_class_entropy_mean_min": 1.90,
        },
        "require_each_seed_primary_improvement": True,
    }

    passing = evaluate_repair(results, specification)
    assert passing["quantitative_pass"]
    assert passing["visual_review_required"]
    assert json.loads(json.dumps(passing)) == passing

    table.loc[
        (table["experiment"] == "flow_mode_recovery")
        & (table["condition"] == "all_real")
        & (table["seed"] == 211),
        "primary_value",
    ] = 0.2
    table.to_csv(results / "final_metrics.csv", index=False)
    failing = evaluate_repair(results, specification)
    assert not failing["quantitative_pass"]
    assert failing["status"] == "failed_quality_gate"


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
    assert saved["kind"] == "grounding-mle-generative-analysis-v2"
    assert saved["analysis_revision"]["post_run_correction"]
    assert not saved["analysis_revision"]["training_rerun_required"]


def test_mode_recovery_primary_is_distance_from_balanced_target() -> None:
    name, value, higher_is_better = _primary_metrics(
        "flow_mode_recovery",
        "flow_mode_recovery",
        {"missing_class_probability": 0.82},
    )

    assert name == "missing_class_absolute_error"
    assert value == pytest.approx(0.72)
    assert not higher_is_better


def test_exact_paired_inference_and_direction_are_reported() -> None:
    rows = []
    for seed, left, right in zip(
        (11, 23, 37), (1.0, 1.2, 1.4), (2.0, 2.2, 2.4), strict=True
    ):
        for condition, value in (("left", left), ("right", right)):
            rows.append(
                {
                    "experiment": "paired",
                    "condition": condition,
                    "seed": seed,
                    "primary_metric": "loss",
                    "primary_value": value,
                    "higher_is_better": False,
                }
            )

    effects = _paired_effects(pd.DataFrame(rows))

    assert _exact_paired_sign_flip_pvalue(np.asarray([-1.0, -1.0, -1.0])) == 0.25
    assert effects.loc[0, "exact_sign_flip_pvalue"] == 0.25
    assert effects.loc[0, "favored_condition"] == "left"
    assert bool(effects.loc[0, "all_nonzero_differences_same_direction"])


def test_class_distribution_metrics_reports_target_distance() -> None:
    labels = np.repeat(np.arange(10), 2)
    metrics = class_distribution_metrics(labels, missing_class=8)

    assert metrics["missing_class_probability"] == pytest.approx(0.1)
    assert metrics["missing_class_target_probability"] == pytest.approx(0.1)
    assert metrics["missing_class_absolute_error"] == pytest.approx(0.0)


def test_empty_vision_batches_keep_their_feature_dimensions() -> None:
    images = np.empty((0, 1, 28, 28), dtype=np.float32)

    flattened = flatten_image_batch(images)

    assert flattened.shape == (0, 784)
    assert inverse_logit(flattened).shape == (0, 1, 28, 28)


def test_flow_preprocessing_is_seeded_and_invertible() -> None:
    images = np.zeros((2, 1, 28, 28), dtype=np.float32)
    specification = {
        "flow_preprocessing": {"dequantize": True, "logit_alpha": 0.01}
    }

    first = preprocess_flow_images(images, specification, seed=17)
    repeated = preprocess_flow_images(images, specification, seed=17)
    different = preprocess_flow_images(images, specification, seed=18)
    restored = inverse_logit(first, alpha=0.01)
    empty = preprocess_flow_images(images[:0], specification, seed=17)

    assert first.shape == (2, 784)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert restored.shape == images.shape
    assert restored.min() >= 0 and restored.max() <= 1 / 255
    assert empty.shape == (0, 784)


def test_cosine_diffusion_schedule_reaches_the_sampling_prior() -> None:
    linear = diffusion_beta_values(50, schedule="linear")
    cosine = diffusion_beta_values(50, schedule="cosine")

    assert np.prod(1 - linear) == pytest.approx(0.6029516, rel=1e-5)
    assert np.prod(1 - cosine) < 1e-3


def test_frechet_feature_distance_handles_singular_covariance() -> None:
    reference = np.arange(24, dtype=np.float64).reshape(6, 4)
    collapsed = np.ones((6, 4), dtype=np.float64)

    assert frechet_feature_distance(reference, reference) == pytest.approx(
        0.0, abs=1e-8
    )
    assert np.isfinite(frechet_feature_distance(reference, collapsed))


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


def test_tiny_convolutional_flow_train_sample_and_score(tmp_path) -> None:
    pytest.importorskip("torch")
    from grounding_mle.generative_vision import (
        flow_log_probabilities,
        sample_flow,
        train_flow,
    )

    values = np.random.default_rng(19).normal(size=(4, 784)).astype("float32")
    checkpoint = tmp_path / "convolutional-flow"
    train_flow(
        transformed_images=values,
        output_dir=checkpoint,
        specification={
            "architecture": "conv",
            "dimension": 784,
            "image_shape": [1, 28, 28],
            "coupling_channels": 4,
            "layers": 2,
        },
        training={"max_steps": 1, "batch_size": 2},
        seed=20,
    )
    samples = sample_flow(checkpoint, 2, seed=21)
    scores = flow_log_probabilities(checkpoint, samples)

    assert samples.shape == (2, 784)
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


def test_tiny_convolutional_diffusion_uses_cosine_schedule_and_ema(tmp_path) -> None:
    pytest.importorskip("torch")
    from grounding_mle.generative_vision import sample_diffusion, train_diffusion

    images = np.random.default_rng(16).random((4, 1, 28, 28), dtype=np.float32)
    checkpoint = tmp_path / "convolutional-diffusion"
    summary = train_diffusion(
        images=images,
        output_dir=checkpoint,
        specification={
            "architecture": "conv",
            "channels": 4,
            "depth": 1,
            "time_dim": 4,
            "diffusion_steps": 3,
            "noise_schedule": "cosine",
            "require_near_pure_noise": True,
        },
        training={"max_steps": 1, "batch_size": 2, "ema_decay": 0.5},
        seed=17,
    )
    samples = sample_diffusion(checkpoint, count=2, seed=18, batch_size=2)

    assert summary["model"]["architecture"] == "conv"
    assert summary["model"]["terminal_signal"] < 1e-3
    assert summary["ema_decay"] == 0.5
    assert samples.shape == (2, 1, 28, 28)
    assert np.isfinite(samples).all()
