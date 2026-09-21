# Validation record

This file separates completed implementation QA from experiments that require external
compute. It must not be cited as if the 485-run language-model study had already produced
scientific results.

## Completed on the implementation host

- The automated suite covers the analysis endpoint direction, exact paired sign-flip
  inference, and missing-class target-distance calculation in addition to the end-to-end
  experiment checks below.
- All ten experiment configurations plus the smoke configuration validate.
- The full materialized dry plan contains 485 unique run IDs. Every schedule has the
  required length, nonnegative integer counts, and no run asks for more than the 4,001
  fresh held-in APPS training records.
- The analytic recursive-training smoke workflow completed two three-round trajectories,
  restart/resume checks, paired-effect analysis, and figure generation.
- A real tiny Hugging Face causal model completed one masked-prompt training step and a
  chat-template-aware generation pass through the production backend.
- The neural-generative 120-run plan materializes with unique IDs across all nine requested
  study families. The 20-run analytic smoke plan completed end to end, including atomic
  rounds, idempotent resume, aggregation, paired effects, `G_T` rank analysis, and figures.
- A real randomly initialized tiny GPT completed an optimizer step, untruncated
  autoregressive sampling, and exact sequence likelihood scoring. A real RealNVP completed
  an optimizer step, sampling, and likelihood scoring. A real tiny DDPM completed an
  optimizer step and reverse-diffusion sampling. These were implementation checks, not
  scientific experiment results.
- The post-hoc vision repair has unit coverage for deterministic uniform dequantization,
  correct inverse-logit scaling, a cosine diffusion schedule that reaches the pure-noise
  sampling prior, and an actual convolutional-denoiser/EMA train-load-sample cycle. These
  checks validate execution; the GPU repair suite must still pass its quantitative gates and
  manual sample-grid review before its outputs are described as credible.
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

The full language-model and neural-generative sweeps were not executed here. This machine has no CUDA or MPS
accelerator, Docker is not installed, and the optional PEFT/EvalPlus packages are not in
the active environment. Docker is an intentional prerequisite: generated programs must
not be run directly on the host. Install `.[llm,dev]`, provide Docker and GPU workers,
complete Phase 1, freeze its calibration, re-plan, and then submit the array as described
in `README.md` and `docs/COMPUTE.md`.

For the neural-generative extension, install `.[generative,dev]`, prepare TinyStories, run
the generative preflight on the target CUDA worker, and materialize
`configs/generative/full.yaml`. The passing analytic and one-step neural checks establish
execution coverage only; they do not establish any reported neural schedule effect.

The prepared tokenizer must have a dedicated padding token distinct from BOS and EOS. Neural
LM training supplies an attention mask, ignores padded targets, and computes held-out
cross-entropy over non-padding target tokens only. If an older data manifest used EOS padding,
rebuild it with `generative-prepare --force` before resuming any LM run.
