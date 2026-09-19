from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .generative_planning import load_generative_config
from .io import atomic_write_json, runtime_manifest, sha256_file, stable_hash


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Serialize creation of shared artifacts across array workers."""

    import fcntl

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_save_numpy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npy", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _text_value(record: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    for field in candidates:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tokenize_records(
    records: Any,
    tokenizer: Any,
    *,
    sequence_length: int,
    maximum: int,
    text_fields: Sequence[str],
    split_name: str = "dataset",
) -> np.ndarray:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("The tokenizer needs a BOS or EOS token")
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("The tokenizer needs an EOS or padding token")
    rows: list[list[int]] = []
    skipped = 0
    progress_interval = min(10_000, maximum)
    for record in records:
        if not isinstance(record, Mapping):
            skipped += 1
            continue
        text = _text_value(record, text_fields)
        if text is None:
            skipped += 1
            continue
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length - 1,
        )["input_ids"]
        sequence = [int(bos_id)] + [int(value) for value in tokens]
        sequence.extend([int(pad_id)] * (sequence_length - len(sequence)))
        rows.append(sequence[:sequence_length])
        if progress_interval and len(rows) % progress_interval == 0:
            print(
                f"{split_name}: tokenized {len(rows):,}/{maximum:,} usable sequences",
                flush=True,
            )
        if len(rows) >= maximum:
            break
    if not rows:
        raise ValueError(
            f"No usable text records were found; skipped {skipped} rows without "
            f"non-empty fields among {list(text_fields)}"
        )
    if skipped:
        print(f"Skipped {skipped} dataset rows without usable text", flush=True)
    return np.asarray(rows, dtype=np.int32)


def prepare_generative_data(
    config_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize deterministic fixed-length TinyStories token arrays.

    Vision datasets are intentionally downloaded by the checksum-verified IDX
    loader on first use; this command prepares the shared language-model corpus
    and tokenizer.
    """

    config = load_generative_config(config_path)
    data = config["data"]
    output = Path(str(data.get("output_dir", "data/generative")))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        from .io import read_json

        existing = read_json(manifest_path)
        expected = {
            "dataset": str(data.get("dataset", "roneneldan/TinyStories")),
            "dataset_config": data.get("dataset_config"),
            "dataset_revision": data.get("dataset_revision"),
            "tokenizer": str(data.get("tokenizer", "gpt2")),
            "sequence_length": int(data.get("sequence_length", 128)),
        }
        mismatches = {
            key: {"existing": existing.get(key), "requested": value}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Prepared data under {output} do not match this config: {mismatches}. "
                "Use --force to rebuild them."
            )
        return existing

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the generative extra before preparing text data") from exc

    tokenizer_name = str(data.get("tokenizer", "gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sequence_length = int(data.get("sequence_length", 128))
    fields = [str(value) for value in data.get("text_fields", ["text", "story"])]
    dataset_name = str(data.get("dataset", "roneneldan/TinyStories"))
    dataset_config = data.get("dataset_config")
    train_split = str(data.get("train_split", "train"))
    validation_split = str(data.get("validation_split", "validation"))
    load_options: dict[str, Any] = {}
    if data.get("data_files"):
        load_options["data_files"] = data["data_files"]
    if data.get("dataset_revision"):
        load_options["revision"] = str(data["dataset_revision"])
    load_options["streaming"] = bool(data.get("streaming", True))
    train_records = load_dataset(dataset_name, dataset_config, split=train_split, **load_options)
    validation_records = load_dataset(
        dataset_name, dataset_config, split=validation_split, **load_options
    )
    train = _tokenize_records(
        train_records,
        tokenizer,
        sequence_length=sequence_length,
        maximum=int(data.get("train_sequences", 100_000)),
        text_fields=fields,
        split_name=train_split,
    )
    validation = _tokenize_records(
        validation_records,
        tokenizer,
        sequence_length=sequence_length,
        maximum=int(data.get("validation_sequences", 10_000)),
        text_fields=fields,
        split_name=validation_split,
    )
    train_path = output / "train.npy"
    validation_path = output / "validation.npy"
    atomic_save_numpy(train_path, train)
    atomic_save_numpy(validation_path, validation)
    tokenizer_dir = output / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    manifest = {
        "kind": "grounding-mle-generative-data-v1",
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": data.get("dataset_revision"),
        "streaming": bool(data.get("streaming", True)),
        "tokenizer": tokenizer_name,
        "sequence_length": sequence_length,
        "train_sequences": int(len(train)),
        "validation_sequences": int(len(validation)),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "fingerprint": stable_hash(
            {
                "train": sha256_file(train_path),
                "validation": sha256_file(validation_path),
                "tokenizer": tokenizer_name,
                "sequence_length": sequence_length,
            }
        ),
        "runtime": runtime_manifest(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_token_arrays(data_config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Path]:
    output = Path(str(data_config.get("output_dir", "data/generative")))
    train_path = output / "train.npy"
    validation_path = output / "validation.npy"
    tokenizer_dir = output / "tokenizer"
    if not train_path.exists() or not validation_path.exists() or not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Prepared language data are missing under {output}; run generative-prepare first"
        )
    return (
        np.load(train_path, mmap_mode="r", allow_pickle=False),
        np.load(validation_path, mmap_mode="r", allow_pickle=False),
        tokenizer_dir,
    )
