from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .docker_support import docker_user_spec, require_docker
from .evaluation import _evalplus_dataset_path, _evalplus_problems
from .io import atomic_write_json
from .modeling import make_backend
from .records import PromptRecord, TrainingExample
from .verification import DockerCodeVerifier


def _run_checked(command: list[str], label: str, timeout: int = 300) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} timed out after {exc.timeout} seconds") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start {label}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{label} failed: {detail[-1500:]}")
    return completed.stdout.strip()


def _require_image(image: str) -> None:
    _run_checked(["docker", "image", "inspect", image], f"Docker image {image}", 60)


def _probe_evalplus_dataset(
    image: str,
    dataset_name: str,
    dataset_path: Path,
    dataset_variable: str,
) -> int:
    function_name = "get_human_eval_plus" if dataset_name == "humaneval" else "get_mbpp_plus"
    with tempfile.TemporaryDirectory(prefix="grounding-evalplus-probe-") as directory:
        root = Path(directory)
        cache = root / "cache"
        output = root / "output"
        cache.mkdir()
        output.mkdir()
        container_dataset = f"/evalplus-data/{dataset_path.name}"
        code = (
            f"from evalplus.data import {function_name}; "
            f"items={function_name}(); "
            "open('/probe/count.txt','w').write(str(len(items))); "
            "print(len(items))"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            docker_user_spec(),
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "1g",
            "--cpus",
            "1",
            "--env",
            f"{dataset_variable}={container_dataset}",
            "--env",
            "XDG_CACHE_HOME=/evalplus-cache",
            "-v",
            f"{dataset_path}:{container_dataset}:ro",
            "-v",
            f"{cache.resolve()}:/evalplus-cache",
            "-v",
            f"{output.resolve()}:/probe",
            image,
            "python",
            "-c",
            code,
        ]
        _run_checked(command, f"offline EvalPlus {dataset_name} dataset probe")
        count_path = output / "count.txt"
        if not count_path.exists():
            raise RuntimeError(f"EvalPlus {dataset_name} probe could not write its bind mount")
        count = int(count_path.read_text(encoding="utf-8"))
        if count <= 0:
            raise RuntimeError(f"EvalPlus {dataset_name} probe returned no tasks")
        return count


def _probe_verifier(image: str) -> None:
    prompt = PromptRecord(
        task_id="preflight/stdin",
        prompt="Echo one input line.",
        human_responses=[],
        metadata={"io_tests": {"inputs": [["hello"]], "outputs": [["hello"]]}},
    )
    result = DockerCodeVerifier(image=image).verify(prompt, "print(input())")
    if not result.accepted:
        raise RuntimeError(f"External-test verification probe failed: {result.reason}")


def _probe_model(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    backend = make_backend(str(config["model"]["backend"]))
    if config["model"]["backend"] != "hf":
        raise RuntimeError("GPU preflight requires the Hugging Face backend")
    base_model = str(config["model"]["name"])
    training = config["training"] | {
        "max_steps": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 128,
        "max_prompt_length": 96,
        "logging_steps": 1,
    }
    generation = config["generation"] | {
        "batch_size": 1,
        "max_prompt_length": 96,
        "max_new_tokens": 2,
        "do_sample": False,
    }
    example = TrainingExample(
        example_id="preflight-real",
        prompt="Write a Python function named answer that returns 1.",
        response="def answer():\n    return 1",
        source="real",
        round_index=0,
        task_id="preflight/model",
    )
    prompt = PromptRecord(
        task_id="preflight/model",
        prompt="Write a Python function named answer that returns 1.",
        human_responses=[],
    )
    with tempfile.TemporaryDirectory(prefix="grounding-model-probe-") as directory:
        checkpoint = Path(directory) / "adapter"
        model_ref = backend.train(
            base_model,
            [example],
            checkpoint,
            training,
            20260906,
            base_model,
        )
        outputs = backend.generate(
            model_ref,
            [prompt],
            generation,
            20260906,
            base_model,
            "code",
        )
    if len(outputs) != 1:
        raise RuntimeError("Model probe did not produce exactly one completion")
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "vram_gib": round(properties.total_memory / 2**30, 2),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "one_step_lora": "passed",
        "adapter_generation": "passed",
    }


def run_runtime_preflight(
    config: dict[str, Any], output_path: str | Path, *, check_model: bool = True
) -> dict[str, Any]:
    processed = Path(config["data"]["processed_dir"])
    required_data = ["code_train.jsonl", "code_monitor.jsonl"]
    if (processed / "math_manifest.json").exists():
        required_data.extend(["math_train.jsonl", "math_monitor.jsonl", "math_test.jsonl"])
    missing = [name for name in required_data if not (processed / name).is_file()]
    if missing:
        raise RuntimeError(f"Prepared dataset files are missing: {', '.join(missing)}")

    require_docker()
    evalplus_image = str(config["evaluation"].get("evalplus_image", "ganler/evalplus:v0.3.1"))
    verifier_image = str(config["verification"].get("docker_image", "python:3.11-slim"))
    for image in sorted({evalplus_image, verifier_image}):
        _require_image(image)

    dataset_counts: dict[str, int] = {}
    for dataset_name in ("humaneval", "mbpp"):
        _evalplus_problems(dataset_name)
        dataset_path, variable = _evalplus_dataset_path(dataset_name, False)
        dataset_counts[dataset_name] = _probe_evalplus_dataset(
            evalplus_image, dataset_name, dataset_path, variable
        )
    _probe_verifier(verifier_image)

    report: dict[str, Any] = {
        "status": "passed",
        "docker": "passed",
        "docker_images": sorted({evalplus_image, verifier_image}),
        "offline_evalplus_datasets": dataset_counts,
        "external_test_verifier": "passed",
        "prepared_data": required_data,
    }
    if check_model:
        report["model"] = _probe_model(config)
    atomic_write_json(output_path, report)
    return report
