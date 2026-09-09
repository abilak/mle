import subprocess

import pytest

from grounding_mle.evaluation import _cached_completions, _require_docker
from grounding_mle.io import atomic_write_jsonl
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
    _require_docker.cache_clear()
    monkeypatch.setattr("grounding_mle.evaluation.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "grounding_mle.evaluation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="permission denied on /var/run/docker.sock"
        ),
    )

    with pytest.raises(RuntimeError, match="current user can access its socket"):
        _require_docker()

    _require_docker.cache_clear()


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
