#!/usr/bin/env bash
set -euo pipefail

phase="${1:-all}"
plan="${CONFIRMATORY_PLAN:-runs/generative_confirmatory_plan.json}"
output_root="${GEN_OUTPUT_ROOT:-runs/generative}"
log_root="${GEN_LOG_ROOT:-logs/generative_confirmatory}"

mkdir -p "$log_root"

run_selection() {
  local label="$1"
  shift
  echo "[start] $label"
  grounding-mle generative-run \
    --plan "$plan" \
    --output-root "$output_root" \
    "$@" \
    --all 2>&1 | tee -a "$log_root/$label.log"
  echo "[done] $label"
}

run_core() {
  run_selection gpt_boundary --experiment gpt_boundary
  run_selection timing_front --experiment gpt_same_budget_timing --condition front_loaded
  run_selection timing_uniform --experiment gpt_same_budget_timing --condition uniform
  run_selection timing_back --experiment gpt_same_budget_timing --condition back_loaded
  run_selection flow_exact_likelihood --experiment flow_exact_likelihood
}

run_robustness() {
  run_selection gpt_misspecified --experiment gpt_misspecified
  run_selection gpt_real_corpus --experiment gpt_real_corpus
  run_selection gpt_model_scale --experiment gpt_model_scale
}

run_timing_bank() {
  run_selection timing_gradual_front \
    --experiment gpt_same_budget_timing --condition gradual_front
  run_selection timing_alternating \
    --experiment gpt_same_budget_timing --condition alternating
  run_selection timing_bursty \
    --experiment gpt_same_budget_timing --condition bursty
  run_selection timing_middle \
    --experiment gpt_same_budget_timing --condition middle_loaded
  run_selection timing_gradual_back \
    --experiment gpt_same_budget_timing --condition gradual_back
}

run_vision_controls() {
  run_selection flow_mode_validity_control --experiment flow_mode_validity_control
  run_selection diffusion_mode_validity_control \
    --experiment diffusion_mode_validity_control
}

case "$phase" in
  core)
    run_core
    ;;
  robustness)
    run_robustness
    ;;
  timing)
    run_timing_bank
    ;;
  vision)
    run_vision_controls
    ;;
  all)
    run_core
    run_robustness
    run_timing_bank
    run_vision_controls
    ;;
  *)
    echo "usage: $0 {core|robustness|timing|vision|all}" >&2
    exit 2
    ;;
esac
