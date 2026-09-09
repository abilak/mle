import subprocess

import pytest

from grounding_mle.docker_support import docker_user_spec, require_docker
from grounding_mle.evaluation import (
    _cached_completions,
    _docker_eval_command,
    _parse_evalplus_result,
)
from grounding_mle.io import atomic_write_json, atomic_write_jsonl
from grounding_mle.records import PromptRecord


def _prompt(task_id: str) -> PromptRecord:
    return PromptRecord(
        task_id=task_id,
        prompt="prompt",
        human_responses=[],
        tests=[],
        skill="general",
        metadata={},
    )


def test_docker_preflight_reports_socket_permission_before_evaluation(monkeypatch):
    require_docker.cache_clear()
    monkeypatch.setattr("grounding_mle.docker_support.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "grounding_mle.docker_support.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="permission denied on /var/run/docker.sock"
        ),
    )

    with pytest.raises(RuntimeError, match="current user can access its socket"):
        require_docker()

    require_docker.cache_clear()


def test_cached_completions_are_reused_only_for_the_same_ordered_tasks(tmp_path):
    samples = tmp_path / "samples.jsonl"
    atomic_write_jsonl(
        samples,
        [
            {"task_id": "one", "solution": "answer one"},
            {"task_id": "two", "solution": "answer two"},
        ],
    )

    assert _cached_completions(samples, [_prompt("one"), _prompt("two")]) == [
        "answer one",
        "answer two",
    ]
    assert _cached_completions(samples, [_prompt("two"), _prompt("one")]) is None


def test_docker_uses_host_file_owner(monkeypatch):
    monkeypatch.setattr("grounding_mle.docker_support.os.getuid", lambda: 1234)
    monkeypatch.setattr("grounding_mle.docker_support.os.getgid", lambda: 5678)

    assert docker_user_spec() == "1234:5678"


def test_evalplus_container_is_offline_and_receives_local_dataset(tmp_path, monkeypatch):
    output = tmp_path / "evaluation" / "round_00" / "humaneval"
    output.mkdir(parents=True)
    samples = output / "humaneval_samples.jsonl"
    dataset = tmp_path / "HumanEvalPlus-v0.1.10.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("grounding_mle.evaluation.docker_user_spec", lambda: "1234:5678")

    command = _docker_eval_command(
        output_dir=output,
        samples_path=samples,
        dataset_path=dataset,
        dataset_variable="HUMANEVAL_OVERRIDE_PATH",
        dataset_name="humaneval",
        evaluation_config={},
    )

    assert command[command.index("--user") + 1] == "1234:5678"
    assert command[command.index("--network") + 1] == "none"
    assert "HUMANEVAL_OVERRIDE_PATH=/evalplus-data/HumanEvalPlus-v0.1.10.jsonl" in command
    assert f"{dataset.resolve()}:/evalplus-data/{dataset.name}:ro" in command
    assert "XDG_CACHE_HOME=/evalplus-cache" in command


def test_evalplus_v031_result_schema_is_parsed(tmp_path):
    result_path = tmp_path / "samples.eval_results.json"
    atomic_write_json(
        result_path,
        {
            "eval": {
                "HumanEval/0": [
                    {"base_status": "pass", "plus_status": "pass"}
                ],
                "HumanEval/1": [
                    {"base_status": "pass", "plus_status": "fail"}
                ],
            }
        },
    )
    prompts = [
        PromptRecord(
            task_id=f"HumanEval/{index}",
            prompt="prompt",
            human_responses=[],
            skill="general",
            metadata={
                "algorithm_type": "general",
                "api_family": "builtins",
                "prompt_complexity": "short",
            },
        )
        for index in range(2)
    ]

    metrics = _parse_evalplus_result(result_path, prompts)

    assert metrics["pass_at_1"] == 0.5
    assert metrics["base_pass_at_1"] == 1.0
    assert len(metrics["tasks"]) == 2
