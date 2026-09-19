from __future__ import annotations

import gc
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .generative_data import atomic_save_numpy, exclusive_file_lock, load_token_arrays
from .io import atomic_write_json, read_json, seed_everything, stable_hash


def accelerator_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def clear_accelerator_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if (
        mps is not None
        and mps.is_available()
        and getattr(torch, "mps", None) is not None
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def _require_lm_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install the generative extra to run neural LM experiments") from exc
    return torch, transformers


def _architecture_payload(specification: Mapping[str, Any], vocab_size: int, length: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "n_positions": length,
        "n_ctx": length,
        "n_embd": int(specification.get("n_embd", 128)),
        "n_layer": int(specification.get("n_layer", 4)),
        "n_head": int(specification.get("n_head", 4)),
        "n_inner": int(specification.get("n_inner", 0)) or None,
        "activation_function": str(specification.get("activation", "gelu_new")),
        "resid_pdrop": float(specification.get("dropout", 0.0)),
        "embd_pdrop": float(specification.get("dropout", 0.0)),
        "attn_pdrop": float(specification.get("dropout", 0.0)),
        "bos_token_id": int(specification["bos_token_id"]),
        "eos_token_id": int(specification["eos_token_id"]),
        "pad_token_id": int(specification["pad_token_id"]),
    }


def create_lm(
    specification: Mapping[str, Any],
    *,
    vocab_size: int,
    sequence_length: int,
) -> Any:
    _, transformers = _require_lm_dependencies()
    config = transformers.GPT2Config(
        **_architecture_payload(specification, vocab_size, sequence_length)
    )
    model = transformers.GPT2LMHeadModel(config)
    model.loss_type = "ForCausalLM"
    return model


def load_lm(checkpoint: str | Path, *, trainable: bool = False) -> Any:
    torch, transformers = _require_lm_dependencies()
    model = transformers.GPT2LMHeadModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.loss_type = "ForCausalLM"
    model.config.use_cache = not trainable
    return model


def _require_dedicated_padding(model: Any) -> int:
    pad = model.config.pad_token_id
    if pad is None or pad in {model.config.bos_token_id, model.config.eos_token_id}:
        raise RuntimeError(
            "The LM tokenizer/checkpoint does not have a dedicated padding token. "
            "Re-run `grounding-mle generative-prepare --config CONFIG --force` and "
            "restart this experiment."
        )
    return int(pad)


def _autocast_context(device: str, enabled: bool) -> Any:
    import contextlib
    import torch

    if enabled and device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_lm(
    *,
    sequences: np.ndarray,
    output_dir: str | Path,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    seed: int,
    vocab_size: int,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train a small causal Transformer with a deterministic step budget."""

    if len(sequences) == 0:
        raise ValueError("Cannot train a language model on an empty sequence array")
    torch, _ = _require_lm_dependencies()
    seed_everything(seed)
    device = accelerator_device()
    length = int(sequences.shape[1])
    model = (
        load_lm(source_checkpoint, trainable=True)
        if source_checkpoint is not None
        else create_lm(architecture, vocab_size=vocab_size, sequence_length=length)
    )
    model.to(device)
    model.train()
    model.config.use_cache = False
    pad_token_id = _require_dedicated_padding(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
    )
    steps = int(training.get("max_steps", 200))
    batch_size = int(training.get("batch_size", 16))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    clip = float(training.get("max_grad_norm", 1.0))
    if steps <= 0 or batch_size <= 0 or accumulation <= 0:
        raise ValueError("LM training steps, batch size, and accumulation must be positive")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    mixed_precision = bool(training.get("mixed_precision", True))
    for step in range(steps):
        accumulated_loss = 0.0
        for _ in range(accumulation):
            indices = rng.integers(0, len(sequences), size=batch_size)
            batch = torch.as_tensor(
                np.asarray(sequences[indices], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            attention_mask = batch.ne(pad_token_id)
            labels = batch.masked_fill(~attention_mask, -100)
            with _autocast_context(device, mixed_precision):
                result = model(
                    input_ids=batch,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = result.loss / accumulation
            loss.backward()
            accumulated_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accumulated_loss)
    target = Path(output_dir)
    temporary = target.with_name(f".{target.name}.tmp-{stable_hash({'seed': seed, 'steps': steps}, 8)}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    summary = {
        "seed": seed,
        "examples": int(len(sequences)),
        "steps": steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "final_loss": losses[-1],
        "mean_tail_loss": float(np.mean(losses[-min(20, len(losses)) :])),
        "device": device,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
    }
    atomic_write_json(temporary / "training_summary.json", summary)
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)
    del optimizer, model
    clear_accelerator_cache()
    return summary


def sample_lm(
    checkpoint: str | Path,
    *,
    count: int,
    sequence_length: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Draw untruncated temperature-one samples from the model distribution."""

    if count == 0:
        return np.empty((0, sequence_length), dtype=np.int32)
    torch, _ = _require_lm_dependencies()
    seed_everything(seed)
    device = accelerator_device()
    model = load_lm(checkpoint)
    model.to(device)
    model.eval()
    bos = model.config.bos_token_id
    if bos is None:
        bos = model.config.eos_token_id
    if bos is None:
        raise ValueError("The LM configuration has no BOS/EOS token")
    pad_token_id = _require_dedicated_padding(model)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            current_size = min(batch_size, count - start)
            tokens = torch.full((current_size, 1), int(bos), dtype=torch.long, device=device)
            past = None
            current = tokens
            for _ in range(sequence_length - 1):
                result = model(input_ids=current, past_key_values=past, use_cache=True)
                logits = result.logits[:, -1, :]
                logits[:, pad_token_id] = -torch.inf
                probabilities = torch.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                current = next_token
                past = result.past_key_values
            outputs.append(tokens.cpu().numpy().astype(np.int32, copy=False))
    del model
    clear_accelerator_cache()
    return np.concatenate(outputs, axis=0)


def sequence_log_probabilities(
    checkpoint: str | Path,
    sequences: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if len(sequences) == 0:
        return np.empty(0, dtype=np.float64)
    torch, _ = _require_lm_dependencies()
    device = accelerator_device()
    model = load_lm(checkpoint)
    model.to(device)
    model.eval()
    pad_token_id = _require_dedicated_padding(model)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(
                np.asarray(sequences[start : start + batch_size], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            attention_mask = batch.ne(pad_token_id)
            logits = model(
                input_ids=batch,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1, :]
            targets = batch[:, 1:]
            token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            )
            target_mask = attention_mask[:, 1:]
            values.append(
                (token_log_probs.squeeze(-1) * target_mask).sum(dim=1).cpu().numpy()
            )
    del model
    clear_accelerator_cache()
    return np.concatenate(values).astype(np.float64, copy=False)


def _empirical_kl(reference: Counter[Any], candidate: Counter[Any], smoothing: float = 0.5) -> float:
    support = set(reference) | set(candidate)
    if not support:
        return 0.0
    reference_total = sum(reference.values()) + smoothing * len(support)
    candidate_total = sum(candidate.values()) + smoothing * len(support)
    value = 0.0
    for item in support:
        p = (reference[item] + smoothing) / reference_total
        q = (candidate[item] + smoothing) / candidate_total
        value += p * math.log(p / q)
    return float(value)


def non_padding_token_counts(sequences: np.ndarray, pad_token_id: int) -> np.ndarray:
    if sequences.ndim != 2:
        raise ValueError("Token sequences must be a two-dimensional array")
    return np.count_nonzero(sequences[:, 1:] != pad_token_id, axis=1).astype(
        np.int64, copy=False
    )


def _ngram_counter(
    sequences: np.ndarray, order: int, pad_token_id: int | None = None
) -> Counter[Any]:
    counter: Counter[Any] = Counter()
    for row in sequences:
        values = [int(value) for value in row]
        if pad_token_id is not None and pad_token_id in values[1:]:
            values = values[: values.index(pad_token_id, 1)]
        if order == 1:
            counter.update(values[1:])
        else:
            counter.update(
                tuple(values[index : index + order])
                for index in range(1, len(values) - order + 1)
            )
    return counter


def language_distribution_metrics(
    reference_samples: np.ndarray,
    student_samples: np.ndarray,
    *,
    pad_token_id: int | None = None,
) -> dict[str, float]:
    reference_unigrams = _ngram_counter(reference_samples, 1, pad_token_id)
    student_unigrams = _ngram_counter(student_samples, 1, pad_token_id)
    reference_bigrams = _ngram_counter(reference_samples, 2, pad_token_id)
    student_bigrams = _ngram_counter(student_samples, 2, pad_token_id)
    bigram_total = sum(student_bigrams.values())
    unique_fraction = len(student_bigrams) / max(1, bigram_total)
    repetitions = sum(max(0, count - 1) for count in student_bigrams.values())
    return {
        "unigram_kl": _empirical_kl(reference_unigrams, student_unigrams),
        "bigram_kl": _empirical_kl(reference_bigrams, student_bigrams),
        "unique_bigram_fraction": float(unique_fraction),
        "bigram_repetition_rate": float(repetitions / max(1, bigram_total)),
    }


def decode_examples(tokenizer_path: str | Path, sequences: np.ndarray, limit: int = 8) -> list[str]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the generative extra to decode samples") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return [
        tokenizer.decode([int(value) for value in row], skip_special_tokens=True)
        for row in sequences[:limit]
    ]


def _token_ids(tokenizer_path: Path) -> tuple[int, int, int, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if bos is None or eos is None or pad is None:
        raise ValueError("Prepared tokenizer lacks required special token IDs")
    if pad in {bos, eos}:
        raise ValueError(
            "Prepared tokenizer uses BOS/EOS as padding. Re-run "
            "`grounding-mle generative-prepare --config CONFIG --force`."
        )
    return int(bos), int(eos), int(pad), int(len(tokenizer))


def architecture_with_tokens(
    specification: Mapping[str, Any], tokenizer_path: Path
) -> tuple[dict[str, Any], int]:
    bos, eos, pad, vocab_size = _token_ids(tokenizer_path)
    payload = dict(specification)
    payload.update({"bos_token_id": bos, "eos_token_id": eos, "pad_token_id": pad})
    return payload, vocab_size


def prepared_pad_token_id(tokenizer_path: str | Path) -> int:
    return _token_ids(Path(tokenizer_path))[2]


def ensure_teacher_artifact(config: Mapping[str, Any], artifact_root: str | Path) -> Path:
    train, _, tokenizer_path = load_token_arrays(config["data"])
    lm = config["lm"]
    architecture, vocab_size = architecture_with_tokens(lm["teacher_model"], tokenizer_path)
    identity = {
        "data": read_json(Path(config["data"]["output_dir"]) / "manifest.json")["fingerprint"],
        "architecture": architecture,
        "training": lm["teacher_training"],
        "seed": int(lm.get("teacher_seed", 20260919)),
    }
    target = Path(artifact_root) / "teachers" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "config.json").exists() and (target / "training_summary.json").exists():
            return target
        rng = np.random.default_rng(int(lm.get("teacher_seed", 20260919)))
        count = min(int(lm.get("teacher_pretrain_sequences", len(train))), len(train))
        indices = rng.choice(len(train), size=count, replace=False)
        sequences = np.asarray(train[indices], dtype=np.int32)
        train_lm(
            sequences=sequences,
            output_dir=target,
            architecture=architecture,
            training=lm["teacher_training"],
            seed=int(lm.get("teacher_seed", 20260919)),
            vocab_size=vocab_size,
        )
        atomic_write_json(target / "artifact_identity.json", identity)
    return target


def ensure_teacher_test_set(
    teacher: str | Path,
    config: Mapping[str, Any],
    artifact_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    lm = config["lm"]
    evaluation = config["evaluation"]
    _, _, tokenizer_path = load_token_arrays(config["data"])
    identity = {
        "teacher": str(Path(teacher).resolve()),
        "count": int(evaluation.get("test_sequences", 5000)),
        "length": int(config["data"].get("sequence_length", 128)),
        "seed": int(evaluation.get("test_seed", 20260920)),
    }
    directory = Path(artifact_root) / "test_sets" / stable_hash(identity, 16)
    samples_path = directory / "samples.npy"
    log_probs_path = directory / "teacher_log_probs.npy"
    with exclusive_file_lock(directory.with_suffix(".lock")):
        if not samples_path.exists() or not log_probs_path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            samples = sample_lm(
                teacher,
                count=identity["count"],
                sequence_length=identity["length"],
                batch_size=int(evaluation.get("generation_batch_size", 32)),
                seed=identity["seed"],
            )
            teacher_log_probs = sequence_log_probabilities(
                teacher,
                samples,
                batch_size=int(evaluation.get("likelihood_batch_size", 32)),
            )
            atomic_save_numpy(samples_path, samples)
            atomic_save_numpy(log_probs_path, teacher_log_probs)
            atomic_write_json(directory / "manifest.json", identity)
    return (
        np.load(samples_path, mmap_mode="r", allow_pickle=False),
        np.load(log_probs_path, mmap_mode="r", allow_pickle=False),
        tokenizer_path,
    )
