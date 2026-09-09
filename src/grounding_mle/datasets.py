from __future__ import annotations

import ast
import hashlib
import json
import re
import urllib.parse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .capabilities import infer_skill
from .io import atomic_write_json, atomic_write_jsonl, runtime_manifest
from .records import PromptRecord


def normalize_prompt(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = normalize_prompt(text).split()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _python_structure_fingerprint(code: str) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if hasattr(node, "lineno"):
            node.lineno = 0
        if hasattr(node, "col_offset"):
            node.col_offset = 0
        if hasattr(node, "end_lineno"):
            node.end_lineno = 0
        if hasattr(node, "end_col_offset"):
            node.end_col_offset = 0
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def contamination_matches(
    candidates: Iterable[PromptRecord],
    evaluation_prompts: Iterable[PromptRecord],
    threshold: float = 0.8,
) -> tuple[list[PromptRecord], list[dict[str, Any]]]:
    eval_records = list(evaluation_prompts)
    eval_normalized = {normalize_prompt(row.prompt): row.task_id for row in eval_records}
    eval_grams = [(row.task_id, _ngrams(row.prompt)) for row in eval_records]
    eval_solution_hashes = {
        fingerprint: row.task_id
        for row in eval_records
        for response in row.human_responses
        if (fingerprint := _python_structure_fingerprint(response)) is not None
    }
    clean: list[PromptRecord] = []
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        solution_match = next(
            (
                eval_solution_hashes[fingerprint]
                for response in candidate.human_responses
                if (fingerprint := _python_structure_fingerprint(response)) in eval_solution_hashes
            ),
            None,
        )
        if solution_match is not None:
            matches.append(
                {
                    "training_task_id": candidate.task_id,
                    "evaluation_task_id": solution_match,
                    "similarity": 1.0,
                    "kind": "exact_python_ast",
                }
            )
            continue
        normalized = normalize_prompt(candidate.prompt)
        if normalized in eval_normalized:
            matches.append(
                {
                    "training_task_id": candidate.task_id,
                    "evaluation_task_id": eval_normalized[normalized],
                    "similarity": 1.0,
                    "kind": "exact",
                }
            )
            continue
        grams = _ngrams(candidate.prompt)
        best_id, best = None, 0.0
        for task_id, target in eval_grams:
            union = grams | target
            similarity = len(grams & target) / len(union) if union else 0.0
            if similarity > best:
                best_id, best = task_id, similarity
        if best >= threshold:
            matches.append(
                {
                    "training_task_id": candidate.task_id,
                    "evaluation_task_id": best_id,
                    "similarity": best,
                    "kind": "fuzzy_5gram_jaccard",
                }
            )
        else:
            clean.append(candidate)
    return clean, matches


def _load_dataset_robust(
    repo_id: str,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the llm extra to prepare datasets") from exc
    kwargs: dict[str, Any] = {"split": split, "cache_dir": cache_dir}
    if revision:
        kwargs["revision"] = revision
    try:
        return load_dataset(repo_id, config_name, **kwargs)
    except RuntimeError as exc:
        if "scripts are no longer supported" not in str(exc).lower():
            raise
    query = urllib.parse.urlencode({"dataset": repo_id, "config": config_name or "default"})
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for the Parquet fallback") from exc
    response = requests.get(
        f"https://datasets-server.huggingface.co/parquet?{query}", timeout=60
    )
    response.raise_for_status()
    payload = response.json()
    urls = [item["url"] for item in payload["parquet_files"] if item["split"] == split]
    if not urls:
        raise RuntimeError(f"No Parquet files resolved for {repo_id}/{split}") from exc
    return load_dataset("parquet", data_files={split: urls}, split=split, cache_dir=cache_dir)


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value


def _valid_python_solutions(value: Any) -> list[str]:
    solutions = _parse_json(value, [])
    if not isinstance(solutions, list):
        return []
    valid: list[str] = []
    for solution in solutions:
        if not isinstance(solution, str):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(solution)
        except (SyntaxError, ValueError):
            continue
        valid.append(solution.strip())
    return valid


def _apps_records(dataset: Any) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for row in dataset:
        solutions = _valid_python_solutions(row.get("solutions"))
        io_tests = _parse_json(row.get("input_output"), {})
        if not solutions or not isinstance(io_tests, dict) or not io_tests.get("inputs"):
            continue
        question = str(row["question"]).strip()
        starter = str(row.get("starter_code") or "").strip()
        prompt = question if not starter else f"{question}\n\nStarter code:\n```python\n{starter}\n```"
        difficulty = str(row.get("difficulty", "unknown"))
        records.append(
            PromptRecord(
                task_id=f"apps/{row['problem_id']}",
                prompt=prompt,
                human_responses=solutions,
                tests=[],
                skill=infer_skill(question),
                metadata={
                    "difficulty": difficulty,
                    "url": row.get("url"),
                    "starter_code": starter,
                    "io_tests": io_tests,
                    "license": "MIT",
                    "dataset": "codeparrot/apps",
                },
            )
        )
    return records


def _evalplus_records(dataset: Any, name: str) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for row in dataset:
        prompt = str(row["prompt"])
        solution = str(row.get("canonical_solution", row.get("code", "")))
        tests = [str(x) for x in row.get("test_list", [])]
        records.append(
            PromptRecord(
                task_id=f"{name}/{row['task_id']}",
                prompt=prompt,
                human_responses=[solution] if solution else [],
                tests=tests,
                skill=infer_skill(prompt),
                metadata={"dataset": name, "license": "Apache-2.0"},
            )
        )
    return records


def _stable_split(records: list[PromptRecord], monitor_fraction: float) -> tuple[list[PromptRecord], list[PromptRecord]]:
    grouped: dict[str, list[PromptRecord]] = defaultdict(list)
    for row in records:
        grouped[str(row.metadata.get("difficulty", "unknown"))].append(row)
    train: list[PromptRecord] = []
    monitor: list[PromptRecord] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(row.task_id.encode("utf-8")).hexdigest(),
        )
        monitor_count = max(1, int(round(len(ordered) * monitor_fraction)))
        monitor.extend(ordered[:monitor_count])
        train.extend(ordered[monitor_count:])
    return sorted(train, key=lambda row: row.task_id), sorted(monitor, key=lambda row: row.task_id)


def prepare_code_data(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    processed = Path(data["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(data.get("cache_dir", ".cache/huggingface")))
    apps = _load_dataset_robust(
        data.get("code_dataset", "codeparrot/apps"),
        "train",
        config_name=data.get("code_config", "all"),
        revision=data.get("code_revision"),
        cache_dir=cache_dir,
    )
    humaneval = _load_dataset_robust(
        "evalplus/humanevalplus", "test", cache_dir=cache_dir
    )
    mbpp = _load_dataset_robust("evalplus/mbppplus", "test", cache_dir=cache_dir)
    apps_records = _apps_records(apps)
    eval_records = _evalplus_records(humaneval, "humanevalplus") + _evalplus_records(
        mbpp, "mbppplus"
    )
    clean, contamination = contamination_matches(
        apps_records, eval_records, float(data.get("contamination_threshold", 0.8))
    )
    train, monitor = _stable_split(clean, float(data.get("monitor_fraction", 0.1)))
    atomic_write_jsonl(processed / "code_train.jsonl", (row.to_dict() for row in train))
    atomic_write_jsonl(processed / "code_monitor.jsonl", (row.to_dict() for row in monitor))
    atomic_write_jsonl(processed / "evalplus_prompts.jsonl", (row.to_dict() for row in eval_records))
    atomic_write_jsonl(processed / "contamination_matches.jsonl", contamination)
    manifest = {
        "runtime": runtime_manifest(),
        "sources": {
            "code_train": "codeparrot/apps all/train split",
            "evaluation": ["evalplus/humanevalplus", "evalplus/mbppplus"],
        },
        "counts": {
            "apps_usable_before_decontamination": len(apps_records),
            "removed_as_contamination": len(contamination),
            "code_train": len(train),
            "code_monitor": len(monitor),
            "evalplus": len(eval_records),
        },
        "fingerprints": {
            "apps": getattr(apps, "_fingerprint", None),
            "humanevalplus": getattr(humaneval, "_fingerprint", None),
            "mbppplus": getattr(mbpp, "_fingerprint", None),
        },
    }
    atomic_write_json(processed / "code_manifest.json", manifest)
    return manifest


def prepare_math_data(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    processed = Path(data["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(data.get("cache_dir", ".cache/huggingface")))
    train_dataset = _load_dataset_robust(
        data.get("math_dataset", "openai/gsm8k"),
        "train",
        config_name=data.get("math_config", "main"),
        revision=data.get("math_revision"),
        cache_dir=cache_dir,
    )
    test_dataset = _load_dataset_robust(
        data.get("math_dataset", "openai/gsm8k"),
        "test",
        config_name=data.get("math_config", "main"),
        revision=data.get("math_revision"),
        cache_dir=cache_dir,
    )
    records = [
        PromptRecord(
            task_id=f"gsm8k/train/{index}",
            prompt=str(row["question"]),
            human_responses=[str(row["answer"])],
            skill="math_reasoning",
            metadata={"dataset": "openai/gsm8k", "license": "MIT"},
        )
        for index, row in enumerate(train_dataset)
    ]
    train, monitor = _stable_split(records, float(data.get("monitor_fraction", 0.1)))
    test = [
        PromptRecord(
            task_id=f"gsm8k/test/{index}",
            prompt=str(row["question"]),
            human_responses=[str(row["answer"])],
            skill="math_reasoning",
            metadata={"dataset": "openai/gsm8k", "license": "MIT"},
        )
        for index, row in enumerate(test_dataset)
    ]
    atomic_write_jsonl(processed / "math_train.jsonl", (row.to_dict() for row in train))
    atomic_write_jsonl(processed / "math_monitor.jsonl", (row.to_dict() for row in monitor))
    atomic_write_jsonl(processed / "math_test.jsonl", (row.to_dict() for row in test))
    manifest = {
        "runtime": runtime_manifest(),
        "source": "openai/gsm8k main",
        "counts": {"math_train": len(train), "math_monitor": len(monitor), "math_test": len(test)},
        "fingerprints": {
            "train": getattr(train_dataset, "_fingerprint", None),
            "test": getattr(test_dataset, "_fingerprint", None),
        },
    }
    atomic_write_json(processed / "math_manifest.json", manifest)
    return manifest


def prepare_all_data(config: dict[str, Any]) -> dict[str, Any]:
    return {"code": prepare_code_data(config), "math": prepare_math_data(config)}
