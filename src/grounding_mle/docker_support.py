from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def require_docker() -> None:
    """Fail early when Docker is absent, stopped, or inaccessible."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for safe code execution")
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot connect to the Docker daemon: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Cannot access the Docker daemon. Confirm that Docker is running and "
            f"the current user can access its socket: {detail[-1000:]}"
        )


def docker_user_spec() -> str:
    """Run bind-mounted work as the host owner, not container root."""
    return f"{os.getuid()}:{os.getgid()}"
