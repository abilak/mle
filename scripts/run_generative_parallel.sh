#!/usr/bin/env bash
set -euo pipefail

plan="${1:-runs/generative_plan.json}"
first="${2:-0}"
last="${3:-0}"
workers="${4:-1}"
output_root="${GEN_OUTPUT_ROOT:-runs/generative}"
log_root="${GEN_LOG_ROOT:-logs/generative}"

if (( first < 0 || last < first || workers < 1 )); then
  echo "usage: $0 PLAN FIRST_INDEX LAST_INDEX WORKERS" >&2
  exit 2
fi

mkdir -p "$log_root"
failure=0

run_index() {
  local index="$1"
  echo "[start] generative plan index $index"
  if grounding-mle generative-run \
      --plan "$plan" \
      --output-root "$output_root" \
      --index "$index" \
      >"$log_root/index_${index}.log" 2>&1; then
    echo "[done]  generative plan index $index"
  else
    echo "[fail]  generative plan index $index; see $log_root/index_${index}.log" >&2
    return 1
  fi
}

for index in $(seq "$first" "$last"); do
  while (( $(jobs -pr | wc -l) >= workers )); do
    wait -n || failure=1
  done
  run_index "$index" &
done

while (( $(jobs -pr | wc -l) > 0 )); do
  wait -n || failure=1
done

exit "$failure"
