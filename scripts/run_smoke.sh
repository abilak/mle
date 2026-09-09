#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m grounding_mle.cli theory --profile quick --output results/theory
PYTHONPATH=src python3 -m grounding_mle.cli plan --config configs/smoke.yaml --output runs/smoke_plan.json
PYTHONPATH=src python3 -m grounding_mle.cli run --plan runs/smoke_plan.json --output-root runs/smoke --all
PYTHONPATH=src python3 -m grounding_mle.cli analyze --runs runs/smoke --output results/smoke
PYTHONPATH=src pytest -q

