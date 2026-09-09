# Compute plan

The full plan is a cluster-scale research program, not a single laptop job. It contains hundreds of independent seed/condition trajectories and thousands of recursive fine-tunes. The implementation therefore uses a funnel.

## Funnel

1. Run the analytic smoke test and full paper simulations on CPU.
2. Run one seed and three schedules on Qwen2.5-Coder-0.5B to validate learning rates, output format, acceptance rate, and checkpoint size.
3. Run Experiments 1 and 2 on the 0.5B discovery model.
4. Freeze the schedule-law coefficients and target reliability.
5. Run controller, verification, capability, and synthetic-strategy studies on 0.5B.
6. Run only the key schedules on Qwen2.5-Coder-1.5B and DeepSeek-Coder-1.3B.
7. Run the three-schedule GSM8K transfer and training-mode robustness studies.

## Local Apple Silicon

The analytic smoke test, data preparation, plotting, and paper simulations work locally. LoRA fine-tuning of 0.5B models can run on MPS but is slow; the 1-2B confirmation work is better assigned to CUDA hardware. Do not lower seeds or silently shorten horizons after seeing results. Instead create a clearly labeled pilot config.

## Array execution

Materialize a plan once, archive it, and assign each integer index to one worker:

```bash
mkdir -p runs/slurm
grounding-mle run --plan runs/full_plan.json --index "$ARRAY_INDEX"
```

The current comprehensive plan contains 485 entries, so after Phase 1 calibration and
re-planning it can be submitted with `sbatch --array=0-484 scripts/slurm_array.sh`.
Do not submit the pre-calibration dry plan: the theory-controller coefficients and the
frozen best/worst confirmation schedules must first exist under `results/llm/`.

Every run writes a resolved config, runtime manifest, accumulated corpus, per-round checkpoints, generated samples, evaluation details, and an atomic `state.json`. Re-running the same index resumes safely.

## Storage

Refit-full runs can create many adapters. Preserve every checkpoint through analysis, then archive or prune only under an explicit retention policy. Do not prune during a run because earlier checkpoints are required for trajectory auditing.
