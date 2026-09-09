from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {source}")
    parent = raw.get("extends")
    if parent:
        parent_path = (source.parent / str(parent)).resolve()
        merged = deep_merge(load_config(parent_path), raw)
    else:
        merged = copy.deepcopy(raw)
    merged.setdefault("_meta", {})["config_path"] = str(source)
    validate_config(merged)
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    required = ["project", "model", "data", "training", "generation", "evaluation", "study"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    rounds = int(config["study"].get("rounds", 0))
    if rounds <= 0:
        raise ValueError("study.rounds must be positive")
    seeds = config["study"].get("seeds", [])
    if not seeds or len({int(x) for x in seeds}) != len(seeds):
        raise ValueError("study.seeds must contain distinct integer seeds")
    mode = config["training"].get("mode")
    if mode not in {"refit_full", "continual_new", "continual_full"}:
        raise ValueError(
            "training.mode must be refit_full, continual_new, or continual_full"
        )
    max_length = int(config["training"].get("max_length", 0))
    max_prompt_length = int(config["training"].get("max_prompt_length", max_length - 1))
    if max_length < 2 or not 0 < max_prompt_length < max_length:
        raise ValueError("training.max_prompt_length must be between 1 and max_length - 1")
    backend = config["model"].get("backend")
    if backend not in {"hf", "analytic"}:
        raise ValueError("model.backend must be hf or analytic")
    eval_backend = config["evaluation"].get("code_execution", "docker")
    if eval_backend not in {"docker", "disabled", "native_unsafe"}:
        raise ValueError("evaluation.code_execution has an unsupported value")
