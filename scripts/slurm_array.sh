#!/usr/bin/env bash
#SBATCH --job-name=grounding-mle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%A_%a.out

set -euo pipefail

PLAN_PATH="${PLAN_PATH:-runs/full_plan.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 2
fi

grounding-mle run \
  --plan "$PLAN_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --index "$SLURM_ARRAY_TASK_ID"

