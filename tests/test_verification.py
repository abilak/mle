import json
import os
import subprocess
import sys

from grounding_mle.records import PromptRecord
from grounding_mle.verification import DockerCodeVerifier, SANDBOX_RUNNER


def _run_safe_fixture(tmp_path, payload):
    (tmp_path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text(SANDBOX_RUNNER, encoding="utf-8")
    environment = os.environ.copy()
    environment["GROUNDING_VERIFY_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(runner)],
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_apps_class_solution_and_json_encoded_arguments(tmp_path):
    result = _run_safe_fixture(
        tmp_path,
        {
            "candidate": (
                "class Solution:\n"
                "    def decodeString(self, value: str) -> str:\n"
                "        return value * 2\n"
            ),
            "tests": [],
            "io_tests": {
                "fn_name": "decodeString",
                "inputs": [['"ab"']],
                "outputs": ['"abab"'],
            },
        },
    )
    assert result["accepted"] is True


def test_apps_standard_input_line_arrays(tmp_path):
    result = _run_safe_fixture(
        tmp_path,
        {
            "candidate": "print(input())\nprint(input())\n",
            "tests": [],
            "io_tests": {"inputs": [["2", "3"]], "outputs": [["2", "3"]]},
        },
    )
    assert result["accepted"] is True


def test_docker_verifier_uses_host_identity(monkeypatch):
    commands = []
    monkeypatch.setattr("grounding_mle.verification.require_docker", lambda: None)
    monkeypatch.setattr("grounding_mle.verification.docker_user_spec", lambda: "1234:5678")

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"accepted": true, "reason": "passed"}\n',
            stderr="",
        )

    monkeypatch.setattr("grounding_mle.verification.subprocess.run", fake_run)
    verifier = DockerCodeVerifier()
    result = verifier.verify(
        PromptRecord(task_id="one", prompt="prompt", human_responses=[]),
        "def answer(): return 1",
    )

    assert result.accepted is True
    command = commands[0]
    assert command[command.index("--user") + 1] == "1234:5678"
    assert command[command.index("--network") + 1] == "none"
