from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .generative_data import load_token_arrays
from .generative_lm import (
    accelerator_device,
    architecture_with_tokens,
    sample_lm,
    sequence_log_probabilities,
    train_lm,
)
from .generative_planning import load_generative_config
from .io import atomic_write_json, runtime_manifest


def run_generative_preflight(
    config_path: str | Path,
    output_path: str | Path,
    *,
    check_vision: bool = True,
) -> dict[str, Any]:
    config = load_generative_config(config_path)
    train, validation, tokenizer_path = load_token_arrays(config["data"])
    result: dict[str, Any] = {
        "status": "running",
        "device": accelerator_device(),
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "runtime": runtime_manifest(),
        "checks": {},
    }
    with tempfile.TemporaryDirectory(prefix="grounding-mle-generative-preflight-") as temporary:
        target = Path(temporary) / "tiny-lm"
        architecture, vocabulary = architecture_with_tokens(
            {"n_embd": 32, "n_layer": 1, "n_head": 1, "dropout": 0.0},
            tokenizer_path,
        )
        sequences = np.asarray(train[: min(4, len(train))], dtype=np.int32)
        train_lm(
            sequences=sequences,
            output_dir=target,
            architecture=architecture,
            training={"max_steps": 1, "batch_size": 2, "mixed_precision": False},
            seed=19,
            vocab_size=vocabulary,
        )
        samples = sample_lm(
            target,
            count=2,
            sequence_length=int(config["data"]["sequence_length"]),
            batch_size=2,
            seed=23,
        )
        log_probs = sequence_log_probabilities(target, samples, batch_size=2)
        if samples.shape != (2, int(config["data"]["sequence_length"])) or not np.isfinite(
            log_probs
        ).all():
            raise RuntimeError("LM preflight produced invalid samples or likelihoods")
        result["checks"]["causal_lm"] = "passed"
    if check_vision:
        from .generative_vision import load_mnist_arrays

        train_images, train_labels, test_images, test_labels = load_mnist_arrays(config["data"])
        if train_images.shape[1:] != (1, 28, 28) or len(train_images) != len(train_labels):
            raise RuntimeError("Vision preflight found an invalid training dataset")
        if test_images.shape[1:] != (1, 28, 28) or len(test_images) != len(test_labels):
            raise RuntimeError("Vision preflight found an invalid test dataset")
        result["checks"]["vision_data"] = "passed"
    result["status"] = "passed"
    atomic_write_json(output_path, result)
    return result

