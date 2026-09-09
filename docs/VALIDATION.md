# Validation record

This file separates completed implementation QA from experiments that require external
compute. It must not be cited as if the 485-run language-model study had already produced
scientific results.

## Completed on the implementation host

- All 23 automated tests pass.
- All ten experiment configurations plus the smoke configuration validate.
- The full materialized dry plan contains 485 unique run IDs. Every schedule has the
  required length, nonnegative integer counts, and no run asks for more than the 4,001
  fresh held-in APPS training records.
- The analytic recursive-training smoke workflow completed two three-round trajectories,
  restart/resume checks, paired-effect analysis, and figure generation.
- A real tiny Hugging Face causal model completed one masked-prompt training step and a
  chat-template-aware generation pass through the production backend.
- APPS verification fixtures cover both `class Solution` call-based tasks with JSON-encoded
  arguments and stdin tasks whose inputs/outputs are stored as line arrays.
- All four full paper-scale simulation groups completed with seed `20260821`, producing
  795 result rows plus PNG and SVG figures.

## Paper-scale numerical checks

At round 900, normalized MSE for the divergent restoring-mass schedule was:

| Family | MSE |
|---|---:|
| Bernoulli | 0.0005055 |
| Poisson | 0.0003825 |
| Gamma | 0.0003625 |
| Categorical extension | 0.0005117 |

The corresponding finite-restoring-mass schedules plateaued at 0.1469, 0.1451,
0.1400, and 0.1471. Across sampled checkpoints, maximum Monte Carlo relative error
against the exact Gaussian recursion was 5.06% in the constant-rate study and 3.66%
in the power-log study. Maximum critical-window absolute error was 0.0111.

Machine-readable values are in `results/theory_full/manifest.json` and
`results/theory_full/theory_results.csv`.

## Not run on this host

The full language-model sweep was not executed here. This machine has no CUDA or MPS
accelerator, Docker is not installed, and the optional PEFT/EvalPlus packages are not in
the active environment. Docker is an intentional prerequisite: generated programs must
not be run directly on the host. Install `.[llm,dev]`, provide Docker and GPU workers,
complete Phase 1, freeze its calibration, re-plan, and then submit the array as described
in `README.md` and `docs/COMPUTE.md`.

