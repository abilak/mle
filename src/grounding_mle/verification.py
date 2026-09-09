from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .docker_support import docker_user_spec, require_docker
from .io import atomic_write_json
from .modeling import ModelBackend
from .records import PromptRecord


SANDBOX_RUNNER = r'''
import contextlib
import io
import json
import math
import os
import subprocess
import sys

work_root = os.environ.get("GROUNDING_VERIFY_ROOT", "/work")
payload = json.load(open(os.path.join(work_root, "payload.json"), encoding="utf-8"))
candidate = payload["candidate"]
tests = payload.get("tests", [])
io_tests = payload.get("io_tests", {})

def decode_json_value(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

def equivalent(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            equivalent(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected

try:
    compile(candidate, "candidate.py", "exec")
    if tests:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(candidate, "candidate.py", "exec"), namespace, namespace)
            for test in tests:
                exec(compile(test, "test.py", "exec"), namespace, namespace)
    elif io_tests:
        inputs = io_tests.get("inputs", [])
        outputs = io_tests.get("outputs", [])
        function_name = io_tests.get("fn_name")
        if len(inputs) != len(outputs):
            raise AssertionError("malformed tests")
        if function_name:
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                prelude = (
                    "from typing import *\nfrom collections import *\nfrom math import *\n"
                    "import bisect, collections, functools, heapq, itertools, math, random, re\n"
                )
                exec(compile(prelude + candidate, "candidate.py", "exec"), namespace, namespace)
                if callable(namespace.get(function_name)):
                    function = namespace[function_name]
                elif "Solution" in namespace:
                    function = getattr(namespace["Solution"](), function_name)
                else:
                    raise AttributeError(f"callable {function_name!r} not found")
                for arguments, expected in zip(inputs, outputs):
                    if not isinstance(arguments, list):
                        arguments = [arguments]
                    arguments = [decode_json_value(value) for value in arguments]
                    expected = decode_json_value(expected)
                    actual = function(*arguments)
                    if not equivalent(actual, expected):
                        raise AssertionError(f"expected {expected!r}, got {actual!r}")
        else:
            open("/tmp/candidate.py", "w", encoding="utf-8").write(candidate)
            for stdin, expected in zip(inputs, outputs):
                if isinstance(stdin, list):
                    stdin = "\n".join(str(item) for item in stdin)
                if isinstance(expected, list):
                    expected = "\n".join(str(item) for item in expected)
                proc = subprocess.run(
                    [sys.executable, "/tmp/candidate.py"],
                    input=str(stdin), text=True, capture_output=True, timeout=4,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr[-500:])
                if proc.stdout.strip().split() != str(expected).strip().split():
                    raise AssertionError(f"expected {expected!r}, got {proc.stdout!r}")
    print(json.dumps({"accepted": True, "reason": "passed"}))
except BaseException as error:
    print(json.dumps({"accepted": False, "reason": type(error).__name__ + ": " + str(error)[:400]}))
'''


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: str


class DockerCodeVerifier:
    def __init__(self, image: str = "python:3.11-slim", timeout_seconds: int = 12):
        self.image = image
        self.timeout_seconds = timeout_seconds
        require_docker()

    def verify(self, prompt: PromptRecord, candidate: str) -> VerificationResult:
        with tempfile.TemporaryDirectory(prefix="grounding-verify-") as directory:
            root = Path(directory)
            atomic_write_json(
                root / "payload.json",
                {
                    "candidate": candidate,
                    "tests": prompt.tests,
                    "io_tests": prompt.metadata.get("io_tests", {}),
                },
            )
            (root / "runner.py").write_text(SANDBOX_RUNNER, encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "512m",
                "--cpus",
                "1",
                "--user",
                docker_user_spec(),
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{root.resolve()}:/work:ro",
                self.image,
                "python",
                "/work/runner.py",
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(False, "sandbox_timeout")
            if result.returncode != 0:
                return VerificationResult(False, f"sandbox_error:{result.stderr[-300:]}")
            try:
                parsed = json.loads(result.stdout.strip().splitlines()[-1])
                return VerificationResult(bool(parsed["accepted"]), str(parsed["reason"]))
            except (IndexError, KeyError, json.JSONDecodeError):
                return VerificationResult(False, "invalid_sandbox_response")


def verify_candidates(
    mode: str,
    prompts: Sequence[PromptRecord],
    candidates: Sequence[str],
    *,
    backend: ModelBackend,
    model_ref: str,
    base_model: str,
    generation_config: dict[str, Any],
    seed: int,
    docker_image: str = "python:3.11-slim",
) -> list[VerificationResult]:
    if len(prompts) != len(candidates):
        raise ValueError("Prompt and candidate counts differ")
    if mode == "none":
        return [VerificationResult(True, "unfiltered") for _ in candidates]
    if mode == "compile":
        output: list[VerificationResult] = []
        for candidate in candidates:
            try:
                ast.parse(candidate)
                output.append(VerificationResult(True, "compiles"))
            except SyntaxError as exc:
                output.append(VerificationResult(False, f"syntax_error:{exc.msg}"))
        return output
    if mode == "external_tests":
        verifier = DockerCodeVerifier(docker_image)
        return [
            verifier.verify(prompt, candidate)
            for prompt, candidate in zip(prompts, candidates, strict=True)
        ]
    if mode == "self_score":
        judge_prompts = [
            PromptRecord(
                task_id=prompt.task_id,
                prompt=(
                    "Judge whether the candidate fully solves the problem. Reply with PASS or FAIL "
                    "on the first line.\n\nProblem:\n"
                    + prompt.prompt
                    + "\n\nCandidate:\n"
                    + candidate
                ),
                human_responses=[],
            )
            for prompt, candidate in zip(prompts, candidates, strict=True)
        ]
        judgments = backend.generate(
            model_ref,
            judge_prompts,
            generation_config | {"do_sample": False, "max_new_tokens": 16},
            seed,
            base_model,
            "raw",
        )
        return [
            VerificationResult(text.strip().upper().startswith("PASS"), f"self_score:{text[:80]}")
            for text in judgments
        ]
    raise ValueError(f"Unknown verification mode: {mode}")
