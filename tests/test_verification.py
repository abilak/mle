import json
import os
import subprocess
import sys

from grounding_mle.verification import SANDBOX_RUNNER


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
