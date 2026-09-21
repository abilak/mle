# Neural generative-model experiments

This suite is an out-of-theorem stress test of the paper's schedule statistic. The neural
models are nonconvex and finite optimization is used at every round, so these experiments
must not be described as proving that the exact moment-matching theorem applies to
Transformers, flows, or diffusion models.

The implementation deliberately lives beside, rather than inside, the coding-model suite.
It has its own plans, output root, checkpoints, analysis tables, and figures. Running it
cannot alter an existing `runs/discovery_plan.json` trajectory.

## Coverage

The full configuration materializes 120 trajectories (40 conditions/seed combinations
across three paired seeds):

| Study | Backend | Question | Main endpoint |
|---|---|---|---|
| `gpt_boundary` | Frozen GPT teacher, same-size GPT student | Constant, log, log-squared, and finite-separation schedules | Monte Carlo teacher-to-student KL per token |
| `gpt_schedule_sweep` | Frozen GPT teacher/student | Does final neural error organize by `G_T` across 12 schedules? | Final KL versus `G_T` |
| `gpt_same_budget_timing` | Frozen GPT teacher/student | Does timing matter when total real and synthetic counts are identical? | Final KL for front/uniform/back loading |
| `gpt_misspecified` | Larger teacher, smaller student | Does grounding approach the best available student approximation? | Excess teacher cross-entropy over a real-only reference student |
| `gpt_real_corpus` | TinyStories as the real source | Does the result survive without a synthetic in-family truth? | Held-out real cross-entropy/perplexity |
| `gpt_model_scale` | Approximately 7M and 16M GPTs | Is the comparison a one-size artifact? | Key-schedule KL at both sizes |
| `flow_exact_likelihood` | Frozen RealNVP teacher/student | Does the schedule effect replicate in a second exact-likelihood neural family? | Monte Carlo exact flow KL |
| `flow_mode_recovery` | RealNVP initialized without one MNIST class | Does a missing mode return? | Absolute error from the balanced missing-class target; raw mass, class KL, and feature distance are diagnostics |
| `diffusion_mode_recovery` | Tiny unconditional DDPM initialized without one class | Does the qualitative mode result survive beyond MLE? | The same target-error and diagnostic metrics plus sample grids |

Every Transformer condition also records teacher and student cross-entropy, perplexity,
unigram KL, bigram KL, unique-bigram fraction, repetition rate, standard errors, and decoded
samples. Vision conditions retain 100 generated images at every evaluated round.

## Protocol details

- The teacher is pretrained once on a fixed TinyStories token array and then frozen.
- Teacher samples use temperature one with no top-k or top-p truncation. Sampling is an
  explicit autoregressive multinomial draw, including after an EOS token, so every record
  has the same preregistered length.
- The initial student is deliberately undertrained on a small teacher or real sample.
- At round `t`, the current student produces fresh synthetic records, the frozen teacher
  (or corpus) supplies fresh real records, and both are appended to the accumulated corpus.
- The next student is refit from the same round-zero initialization on the entire accumulated
  corpus. This is the closest practical analogue of recomputing an accumulated-data MLE.
- Evaluation uses a fixed teacher-generated test set. Teacher log probabilities are cached
  once, while every student is scored on exactly the same sequences.
- The misspecified reference is a small student trained on a much larger teacher-only sample.
- Matched timing conditions contain exactly 512 real and 2,048 synthetic records over 20
  rounds. Only their arrival times differ.
- Shared teachers, test sets, reference students, initial students, classifiers, and vision
  teachers are content-addressed and protected by file locks. Parallel workers build each
  shared artifact at most once on a shared filesystem.
- A round becomes visible only after its data, checkpoint, evaluation, and manifest have
  been atomically moved into place. An interrupted run discards only the uncommitted round.

## Installation and preparation

On the GPU worker:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[generative,dev]'

grounding-mle generative-prepare \
  --config configs/generative/full.yaml

grounding-mle generative-preflight \
  --config configs/generative/full.yaml \
  --output results/generative_preflight.json
```

The preflight performs an actual causal-Transformer update, exact sampling and likelihood
calculation, and verifies the MNIST/Fashion-MNIST loader. It must finish with
`"status": "passed"`.

## CPU protocol smoke test

The analytic Markov backend exercises planning, schedule resolution, accumulation,
resume behavior, metrics, statistics, and figures without downloading data or using a GPU:

```bash
grounding-mle generative-plan \
  --config configs/generative/smoke.yaml \
  --output runs/generative_smoke_plan.json

grounding-mle generative-run \
  --plan runs/generative_smoke_plan.json \
  --output-root runs/generative_smoke \
  --all

grounding-mle generative-analyze \
  --runs runs/generative_smoke \
  --output results/generative_smoke
```

This validates the protocol, not the neural scientific claims.

## Full execution

Materialize and archive the plan once:

```bash
grounding-mle generative-plan \
  --config configs/generative/full.yaml \
  --output runs/generative_plan.json
```

Run one scheduler entry:

```bash
grounding-mle generative-run \
  --plan runs/generative_plan.json \
  --index 0
```

Or run a complete study sequentially:

```bash
grounding-mle generative-run \
  --plan runs/generative_plan.json \
  --experiment gpt_boundary \
  --all
```

The recommended order is:

1. `gpt_boundary`
2. `gpt_same_budget_timing`
3. `gpt_schedule_sweep`
4. `gpt_real_corpus`
5. `gpt_misspecified`
6. `gpt_model_scale`
7. `flow_exact_likelihood`
8. `flow_mode_recovery`
9. `diffusion_mode_recovery`

The flow and diffusion studies download MNIST only when first used. Diffusion is the
furthest-from-theorem qualitative stress test and should be presented after the
exact-likelihood results.

## Parallel execution

For a single 96 GB GPU with ample CPU/RAM, begin with two workers and then try four after
checking utilization:

```bash
bash scripts/run_generative_parallel.sh \
  runs/generative_plan.json 0 11 2
```

The arguments are plan, inclusive first index, inclusive last index, and worker count.
Different machines must receive disjoint index ranges. If they do not share the artifact
directory, each machine will otherwise pretrain its own copy of the teacher; copy the
prepared `data/generative/` and `artifacts/generative/` directories to avoid that waste.

Four simultaneous EvalPlus containers are not involved in this suite, but four neural
training processes can still saturate a GPU. Increase parallelism only after measuring a
complete trajectory, not from VRAM usage alone.

## Analysis

```bash
grounding-mle generative-analyze \
  --runs runs/generative \
  --output results/generative
```

The command writes long-form and final CSVs, bootstrap condition intervals, paired-seed
effects with exact two-sided sign-flip randomization p-values, the `G_T` rank-correlation
result, five publication-oriented figure families, qualitative image grids, and a runtime
manifest. With the preregistered three paired seeds, the smallest attainable nonzero
two-sided sign-flip p-value is 0.25; bootstrap intervals must not be presented as substitutes
for that exact small-sample test.

For mode recovery, the primary endpoint is the absolute distance between the generated
missing-class probability and its balanced target of 0.10, so both failure to recover and
overshoot are penalized. Raw missing-class probability, class KL, classifier feature
distance, entropy, and sample grids remain necessary diagnostics. Classifier probabilities
on visibly out-of-distribution samples should not be interpreted without those diagnostics.

This is a post-run correction recorded as analysis schema v2. The original hypothesis said
that probability should approach the balanced target, but the v1 implementation mistakenly
treated raw missing-class probability as monotonically better and therefore rewarded severe
overshoot. The correction reuses saved per-round metrics and requires no retraining. Exact
sign-flip p-values were also added in this audit; publications should disclose both changes.

## Frozen confirmatory extension

`configs/generative/confirmatory.yaml` freezes ten new paired seeds, three primary contrasts,
five secondary robustness contrasts, an eight-condition fixed-budget timing bank, and
all-real/no-real validity controls for both vision backends. Seeds 11, 23, and 37 were already
inspected, so they are excluded from confirmatory p-values. The original global defaults
remain unchanged so their 45 matching completed checkpoints retain their original run IDs
and can still appear in descriptive pooled figures. Of 257 plan entries, 212 are new.

The three primary contrasts are divergent-log versus log-squared GPT, block-front versus
block-back timing at exactly 512 real records, and divergent-log versus log-squared exact
flow. Exact two-sided sign-flip p-values receive Holm correction across this family. The
secondary robustness family is corrected separately. Confirmatory p-values are withheld
until every frozen seed in the corresponding family is complete; this prevents optional
stopping on interim results.

Create the frozen plan once:

```bash
grounding-mle generative-plan \
  --config configs/generative/confirmatory.yaml \
  --output runs/generative_confirmatory_plan.json
```

Run phases serially on a single GPU. Each phase is restart-safe and reuses completed runs:

```bash
bash scripts/run_generative_confirmatory.sh core
bash scripts/run_generative_confirmatory.sh robustness
bash scripts/run_generative_confirmatory.sh timing
bash scripts/run_generative_confirmatory.sh vision
```

The `core` phase adds 90 runs, `robustness` adds 60, the remaining timing-bank schedules add
50 after core, and vision validity controls add 12. `all` runs those phases in that order.
Do not use `--no-resume`, change the frozen seeds, or stop after inspecting an interim
p-value.

Check completion at any time without touching a run:

```bash
python scripts/generative_plan_status.py \
  --plan runs/generative_confirmatory_plan.json \
  --runs runs/generative
```

Analyze the accumulated original and confirmatory runs together:

```bash
grounding-mle generative-analyze \
  --runs runs/generative \
  --output results/generative_confirmatory \
  --confirmatory-config configs/generative/confirmatory.yaml
```

The additional outputs are `confirmatory_contrasts.csv`,
`confirmatory_timing_by_seed.csv`, `confirmatory_vision_controls.csv`, and
`confirmatory_analysis.json`. The timing-bank test estimates a KL-versus-`G_T` slope within
each seed while holding total real and synthetic counts fixed, then applies the exact
sign-flip test to the ten slopes. Vision controls are validity checks, not searches for a
positive recursive-training result: an all-real quantitative pass still requires visual
review of the generated sample grids.

Do not interpret the log/log-squared finite-horizon ordering as an asymptotic theorem for
neural networks. The corresponding plot is an empirical stress test; the exact theorem
continues to be claimed only for the regular likelihood classes established in the paper.

## Post-hoc vision backend repair

The original all-real/no-real vision controls failed model-validity review. The flow's
all-real condition sometimes improved relative metrics, but its samples were blurry and its
generated missing-class probability was far from the balanced target. The diffusion samples
were noise and the all-real condition did not beat no-real. Do not report either original
vision control as positive evidence.

The failures had identifiable implementation and modeling causes:

- The 50-step linear diffusion schedule ended with `alpha_bar_T` near 0.603. Training
  therefore never presented nearly pure noise, even though sampling started from pure
  Gaussian noise. The repair uses a cosine schedule whose terminal signal is below 0.001.
- The original denoiser was a fully connected MLP over 784 pixels. The repair uses a
  convolutional residual denoiser, exponential moving-average weights, and a larger training
  budget.
- The image flow applied an extreme deterministic logit transform directly to discrete,
  nearly binary pixels. The repair uniformly dequantizes pixels, uses a less extreme logit
  boundary, and replaces the flattened coupling networks with image-aware checkerboard
  convolutional couplings.
- A digit classifier can be confidently wrong on noise or other out-of-distribution images.
  The repair therefore requires absolute class-balance and entropy gates, relative
  all-real/no-real improvements, feature-distance improvement, and manual sample-grid review.

The repair is isolated in `configs/generative/vision_repair.yaml`, uses three fresh seeds,
and writes to separate run/result directories. It is a post-hoc exploratory backend check,
not a replacement confirmatory analysis. It uses one large all-real/no-real intervention
rather than another 20-round recursive sweep because the question is whether the generators
are valid at all.

Run it only if a vision appendix or cross-objective stress test is useful:

```bash
bash scripts/run_generative_vision_repair.sh all
```

The command is resumable. Its final checker exits unsuccessfully unless both backends pass
the predeclared relative and absolute quantitative gates. Even after a quantitative pass,
inspect these files before accepting the repair:

```text
results/generative_vision_repair/samples_flow_mode_recovery_all_real.png
results/generative_vision_repair/samples_flow_mode_recovery_no_real.png
results/generative_vision_repair/samples_diffusion_mode_recovery_all_real.png
results/generative_vision_repair/samples_diffusion_mode_recovery_no_real.png
```

Use `flow`, `diffusion`, `status`, or `analyze` instead of `all` to run or inspect one phase.
The exact-likelihood flow experiment already supplies the continuous regular-MLE extension
needed by the main argument. Diffusion is score matching rather than MLE, and raw-MNIST flow
mode recovery is only a qualitative model check. The main paper therefore does not require
either repaired vision result.
