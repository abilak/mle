# Preregistered protocol

## Primary hypotheses

1. Equal total grounding and synthetic-data budgets can yield different final reliability solely because their arrival schedules differ.
2. A two-term predictor using `Q_T^2` and `Q_T^2 A_T`, fitted on a prespecified training half of the schedule bank, predicts unseen schedule outcomes better than total grounding or percentage baselines.
3. A schedule chosen from the frozen predictor reaches a prespecified reliability threshold using fewer external examples than uniform, periodic, front-loaded, back-loaded, random, or monitor-reactive allocation.
4. The ordering of key schedules transfers across model scale, model family, verification regime, training mode, and a reasoning domain.

## Primary endpoint

For code, the primary endpoint is HumanEval+ pass@1 at the final recursive round under greedy decoding. MBPP+ pass@1 is the confirmatory endpoint. For mathematics, the endpoint is exact-answer GSM8K accuracy. The operational endpoint is external examples required to keep the primary quality at or above the threshold fixed before Phase 2.

## Reliability endpoints

- Worst algorithm-group pass@1.
- Bottom initial-capability-bin retention.
- Common-versus-rare skill retention.
- Compilation and external-test acceptance rates.
- Area under the quality-over-generations trajectory.

Initial capability groups are defined from the baseline checkpoint before recursive training. Group definitions are frozen for all later rounds.

## Randomization and pairing

Seeds are listed in each YAML before results are observed. For a fixed seed, competing schedules receive the same base checkpoint, deterministic prompt order, response-choice order, decoding configuration, and training seed stream. The schedule determines only where cuts in those common streams occur. Headline studies use five seeds; broader grids use three.

## Schedule-law split

The 40 matched-budget schedules and their 20/20 `train`/`test` labels are generated deterministically from seed `20260906` before any model run. Nuisance coefficients and percentage baselines are fit on `train` only. The test labels are never reallocated, and test results are not used to modify features or controller objectives.

Candidate predictors fixed in advance are:

- total external count `M`;
- mean and final batch external fraction;
- cumulative external fraction;
- restoring mass `G_T`;
- survival product `Q_T`;
- propagated noise `Q_T^2 A_T`.

The theory model is a nonnegative two-term regression with no data-driven feature search. The baseline is ridge regression on count/percentage features.

## Statistics

- Report means and 95% percentile bootstrap confidence intervals across independent seeds.
- Report paired differences between schedule conditions sharing a seed.
- Report paired standardized effects, not only p-values.
- For schedule prediction, report held-out R2, MAE, and Spearman rank correlation.
- Keep all task-level outcomes so hierarchical or mixed-effects analysis can be added without rerunning models.
- Do not select a “catastrophic” confirmation schedule until Experiment 1 is complete; select it by the preregistered lowest mean final primary endpoint, then freeze its identity.

## Training protocols

`refit_full` restarts every generation from the same base checkpoint and fits the complete accumulated corpus. This is closest to the paper's MLE protocol. `continual_new` starts from the current checkpoint and trains on the new batch. `continual_full` starts from the current checkpoint and trains again on the complete corpus; it is included to measure the effect of repeated exposure.

Each matched condition uses the same maximum optimizer steps, batch size, gradient accumulation, sequence length, and full-length padding. Consequently the prescribed token-processing budget is identical even when real and synthetic response lengths differ.

## Stopping and exclusions

- A run fails rather than silently reducing its retained synthetic count when verification cannot supply the requested batch within the fixed attempt multiplier.
- Dataset records without parseable human Python solutions or usable tests are excluded before randomization.
- Training records with exact prompt, high 5-gram Jaccard prompt, or exact AST overlap with EvalPlus are removed and logged.
- Infrastructure failures are retried from the atomic round checkpoint; failed model outcomes are not selectively discarded.

## Claim discipline

The exact recovery theorem applies to the moment-matching MLE class under its stated assumptions. LLM studies can support empirical predictiveness or robustness of schedule variables; they cannot establish a universal human-data threshold for arbitrary neural models. Negative transfer is a valid result and must be reported.

