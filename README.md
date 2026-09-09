# When Should AI Return to Reality?

This repository implements the complete experiment program for studying how external grounding should be scheduled during recursive self-training. It contains two deliberately separate layers:

1. **Paper-aligned MLE simulations.** These test the proved finite-sample and asymptotic predictions in moment-matching likelihood families and the narrow-spike counterexample.
2. **Recursive language-model experiments.** These test whether the paper's schedule variables remain empirically predictive for coding and reasoning models. They are empirical analogues, not consequences of the exact iff theorem.

The implementation covers all ten experiments in the project specification, the required controls, repeated-seed statistics, held-out schedule-law testing, safe code execution, model-scale replication, and the final five-figure analysis package. See [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) for exact coverage.

## What is implemented

- Exact schedule variables: batch real fraction, `gamma_t = m_t / N_t`, restoring mass `G_T`, survival product `Q_T`, transformed noise `A_T`, and the stable `Q_T^2 A_T` recursion.
- Monte Carlo experiments for Gaussian, Bernoulli, categorical, Poisson, and Gamma MLEs.
- The constant-batch `rho = 1/2` rate transition, complete power-log boundary, and Construction 1 asymptotic and finite-sample spike thresholds.
- Deterministic 40-schedule matched-budget bank with a frozen 20/20 train/test split.
- Theory-guided fixed-budget and target-risk grounding controllers.
- Recursive refit-on-all-data, continual-on-new-data, and continual-on-all-data training modes.
- Passive, static, failure-targeted, and verified synthetic-data strategies.
- None/compile/external-test/self-score verification regimes.
- Qwen2.5-Coder 0.5B and 1.5B studies, a DeepSeek-Coder architecture replication, and GSM8K domain transfer.
- APPS preparation, exact and fuzzy benchmark decontamination, disjoint monitoring splits, EvalPlus evaluation, task-group retention analysis, bootstrapped confidence intervals, paired effects, and board-ready figures.
- Atomic state checkpoints and restart-safe runs.

## Installation

Python 3.10 or newer and Docker are required for the LLM suite. Docker is mandatory because model-generated programs must not be executed directly on the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[llm,dev]'
```

Check the implementation:

```bash
pytest
grounding-mle validate-config configs/experiments/01_same_budget.yaml
```

## Reproduce the paper-aligned experiments

The quick profile is suitable for a laptop and has already been used for implementation QA. The full profile uses the paper-scale Monte Carlo counts.

```bash
grounding-mle theory --profile quick --output results/theory
grounding-mle theory --profile full --output results/theory_full
```

Both profiles write machine-readable CSV data plus PNG and SVG figures. The fixed seed is `20260821`, matching the paper.

## Prepare every dataset

```bash
grounding-mle prepare-data --config configs/base_llm.yaml --domain all
```

This command downloads only the APPS training split, EvalPlus test sets, and GSM8K; validates and normalizes records; removes exact prompt matches, fuzzy prompt matches, and exact Python-AST matches against HumanEval+/MBPP+; makes a stable training-disjoint monitor split; and records fingerprints and counts under `data/processed/`.

The held-out benchmarks are never used by a scheduling policy. The reactive and failure-targeted baselines may use only `code_monitor.jsonl` or `math_monitor.jsonl`.

## Run the LLM research program

The suite is intentionally phased. Do not plan the controller studies from their default unit coefficients and present them as empirically calibrated.

### Phase 1: establish the effect and fit the schedule law

```bash
grounding-mle plan \
  --config configs/experiments/01_same_budget.yaml \
  --config configs/experiments/02_schedule_law.yaml \
  --output runs/discovery_plan.json
```

Run one array entry:

```bash
grounding-mle run --plan runs/discovery_plan.json --index 0
```

After the discovery entries finish:

```bash
grounding-mle analyze --runs runs --output results/llm
```

The schedule-law analysis fits the two nuisance coefficients only on the schedules marked `train`, freezes them, and reports predictive R2, MAE, and Spearman correlation on the schedules marked `test`. Percentage/count baselines are fit on the same split.

### Phase 2: freeze the calibration, then plan the interventions

The later configs point to `results/llm/schedule_law.json`. Re-plan them only after Phase 1 analysis so their theory-controlled schedules use the frozen coefficients.

```bash
grounding-mle plan --manifest configs/full_study.txt --output runs/full_plan.json
```

Use `scripts/slurm_array.sh` on a GPU cluster or invoke each plan index through your scheduler. Runs are idempotent and resume from their last completed round.

### Analyze final runs

```bash
grounding-mle analyze --runs runs --output results/final
```

The analysis writes trajectory and task-level tables, all paired condition effects, condition summaries with 95% bootstrap intervals, the held-out schedule-law test, frozen best/worst confirmation schedules, capability summaries, and Figures A-E from the specification.

## Experimental safeguards

- All matched-budget schedule comparisons share the same model, seed, prompt order, total real examples, total synthetic examples, optimizer steps, and padded token budget. Only arrival timing changes.
- Main endpoints use deterministic greedy evaluation. Synthetic generation remains stochastic under a fixed seed.
- HumanEval+/MBPP+ are evaluated with EvalPlus 0.3.1 in Docker. Native execution requires the explicit `native_unsafe` setting and is never the default.
- The code records every accepted and rejected synthetic-generation attempt, dataset fingerprint, resolved schedule, checkpoint, runtime version, and task-level result.
- “Theory-aligned” means restarting from the same base checkpoint and refitting on the full accumulated corpus. It does not mean the neural network satisfies the moment-matching theorem.

## Repository map

- `configs/experiments/`: the ten complete studies.
- `src/grounding_mle/theory_experiments.py`: paper simulations.
- `src/grounding_mle/runner.py`: recursive training engine.
- `src/grounding_mle/datasets.py`: dataset preparation and decontamination.
- `src/grounding_mle/schedules.py`: schedule generation and grounding controller.
- `src/grounding_mle/evaluation.py`: EvalPlus/GSM8K evaluation.
- `src/grounding_mle/analysis.py`: statistics and final figures.
- `docs/PROTOCOL.md`: preregistered experimental and statistical protocol.
- `docs/DATASETS.md`: provenance, splits, licenses, and leakage controls.
- `docs/COMPUTE.md`: execution funnel and cluster guidance.
- `docs/VALIDATION.md`: completed QA, numerical checks, and the explicit boundary between implemented and executed work.
