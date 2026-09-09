# Experiment coverage matrix

| # | Required experiment | Implementation | Primary outputs |
|---:|---|---|---|
| 0 | Controlled theory validation | `theory_experiments.py`; `grounding-mle theory` | Cross-family iff curves, rate transition, power-log boundary, spike surface, CSV |
| 1 | Same grounding budget, different fate | `01_same_budget.yaml` | Paired trajectories for eight schedules plus four controls |
| 2 | Held-out test of the MLE schedule law | `02_schedule_law.yaml`; `fit_schedule_law` | 40 schedules, frozen 20/20 split, theory versus percentage baseline |
| 3 | Grounding Controller and cost frontier | `03_grounding_controller.yaml`; controller optimizer | Budget/quality Pareto frontier against uniform, periodic, front, back, random, and reactive policies |
| 4 | Vanishing external percentage | `04_vanishing_grounding.yaml` | `t/log(t)` versus `t/log(t)^2` trajectories and schedule statistics |
| 5 | Weak capability erosion | `05_capability_erosion.yaml`; task metadata analysis | Initial-strength, algorithm, API, rarity, and prompt-complexity retention |
| 6 | Verification versus fresh grounding | `06_verification_phase_diagram.yaml` | 4 x 4 grid plus theory-controlled row; phase diagram |
| 7 | Modern synthetic-data strategies | `07_synthetic_strategies.yaml` | Passive, static, failure-targeted, and externally verified comparison |
| 8 | Model scale and architecture | `08_model_scale_and_architecture.yaml` | Frozen good/bad schedules, 0.5B discovery, 1.5B confirmation, 1.3B family replication, verification checks |
| 9 | Code to reasoning transfer | `09_reasoning_transfer.yaml` | Uniform/bad/theory schedules on GSM8K |
| 10 | Theory-aligned versus deployment training | `10_training_mode_robustness.yaml` | Refit-full versus continual-new and continual-full |

## Mandatory controls

| Control | Where |
|---|---|
| All-real upper reference | Experiment 1 |
| All-synthetic reference | Experiments 1 and 5 |
| Fixed original real corpus plus synthetic accumulation | Experiments 1 and 5 |
| Same total real and same total synthetic | Matched schedule comparisons in Experiments 1, 2, 3, 5, 7, 8, 9, 10; deliberately degenerate all-real/all-synthetic references are reported separately |
| Same optimizer steps and padded token budget | Base training config and trainer |
| Static synthetic pool | Experiments 1 and 7 |
| Multiple paired seeds | Every LLM experiment |
| Held-out evaluation excluded from scheduling | Runner policy boundary and protocol |

## Paper claims versus empirical questions

The exact iff rule and finite-horizon risk identity are tested only in the barycentric/moment-matching simulation layer. LLM results are reported as tests of transfer: whether `G_T`, `Q_T`, and `Q_T^2 A_T` predict recursive neural training under model mismatch, finite optimization, and parameter sharing. The narrow-spike study is an explicit counterexample to universal count-only claims and is therefore part of the theory suite rather than an LLM claim.
