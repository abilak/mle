# Compute plan

The full plan is a cluster-scale research program, not a single laptop job. It contains hundreds of independent seed/condition trajectories and thousands of recursive fine-tunes. The implementation therefore uses a funnel.

## Funnel

1. Run the analytic smoke test and full paper simulations on CPU.
2. Run `grounding-mle preflight` on the target worker, then run one seed and three schedules on Qwen2.5-Coder-0.5B to validate learning rates, output format, acceptance rate, and checkpoint size.
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

## Neural-generative extension

The generative extension is a separate 120-entry plan. Its shared frozen teachers and test
sets are expensive to create but are content-addressed and reused across conditions. Prepare
TinyStories and run `grounding-mle generative-preflight` before allocating the full sweep.

The 96 GB GPU configuration discussed for this project can run multiple plan entries at
once, but one entry does not need 96 GB. Begin with two concurrent workers and benchmark a
complete trajectory before increasing to four. Shared-GPU jobs compete for compute even when
memory usage looks low. The exact-likelihood flow and diffusion runs also consume CPU and
system RAM for MNIST classification and feature metrics.

The suggested scientific funnel is boundary, matched timing, schedule sweep, real-corpus,
misspecified/scale, exact flow, flow mode recovery, and diffusion last. Commands and the
claim boundary are in `docs/GENERATIVE_EXPERIMENTS.md`.
