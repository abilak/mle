# Code-execution safety

Model outputs are untrusted programs.

- External-test filtering runs inside a fresh `python:3.11-slim` container with networking disabled, a read-only root filesystem, dropped Linux capabilities, no-new-privileges, process/memory/CPU limits, an unprivileged user, and a bounded wall-clock timeout.
- Final HumanEval+/MBPP+ evaluation uses the official EvalPlus 0.3.1 container and never native execution by default.
- `native_unsafe` exists only for controlled infrastructure where an independent sandbox is already present. It must not be used on a personal workstation.
- Compilation filtering uses Python parsing only and does not execute code.
- Self-scoring asks the current model for a PASS/FAIL judgment and is treated as an experimental regime, not a safety boundary.

Container isolation reduces risk but does not make arbitrary code harmless under every Docker or kernel vulnerability. Use dedicated workers without credentials or sensitive mounts for full experiments.

