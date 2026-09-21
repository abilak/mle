#!/usr/bin/env bash
set -euo pipefail

phase="${1:-all}"
config="${VISION_REPAIR_CONFIG:-configs/generative/vision_repair.yaml}"
plan="${VISION_REPAIR_PLAN:-runs/generative_vision_repair_plan.json}"
output_root="${VISION_REPAIR_RUNS:-runs/generative_vision_repair}"
results_root="${VISION_REPAIR_RESULTS:-results/generative_vision_repair}"
log_root="${VISION_REPAIR_LOGS:-logs/generative_vision_repair}"

mkdir -p "$log_root"

ensure_plan() {
  if [[ ! -f "$plan" ]]; then
    grounding-mle generative-plan --config "$config" --output "$plan"
  fi
}

run_experiment() {
  local experiment="$1"
  echo "[start] $experiment"
  grounding-mle generative-run \
    --plan "$plan" \
    --output-root "$output_root" \
    --experiment "$experiment" \
    --all 2>&1 | tee -a "$log_root/$experiment.log"
  echo "[done] $experiment"
}

analyze() {
  grounding-mle generative-analyze \
    --runs "$output_root" \
    --output "$results_root" 2>&1 | tee -a "$log_root/analysis.log"
  python scripts/check_vision_repair.py \
    --config "$config" \
    --results "$results_root" 2>&1 | tee -a "$log_root/quality_gate.log"
  echo "Inspect the four sample grids in $results_root before accepting the repair."
}

ensure_plan

case "$phase" in
  flow)
    run_experiment flow_mode_recovery
    ;;
  diffusion)
    run_experiment diffusion_mode_recovery
    ;;
  analyze)
    analyze
    ;;
  status)
    python scripts/generative_plan_status.py --plan "$plan" --runs "$output_root"
    ;;
  all)
    run_experiment flow_mode_recovery
    run_experiment diffusion_mode_recovery
    analyze
    ;;
  *)
    echo "usage: $0 {flow|diffusion|analyze|status|all}" >&2
    exit 2
    ;;
esac
