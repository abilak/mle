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

## Neural generative-model extension

| # | Requested extension | Implementation | Primary outputs |
|---:|---|---|---|
| G1 | Frozen tiny-GPT teacher/student boundary | `gpt_boundary`; `teacher_lm` backend | Exact Monte Carlo sequence KL, CE/PPL, n-gram diagnostics, text samples |
| G2 | Diverse schedule bank organized by `G_T` | `gpt_schedule_sweep` | 12 schedules and final-KL versus `G_T` rank test |
| G3 | Same total data, different timing | `gpt_same_budget_timing` | Exact matched totals for front/uniform/back loading |
| G4 | Misspecified student | `gpt_misspecified` | Excess CE over a large-real-sample reference student |
| G5 | Real-corpus replication | `gpt_real_corpus` | TinyStories held-out CE/PPL and degeneration diagnostics |
| G6 | Model-size replication | `gpt_model_scale` | Key schedules at approximately 7M and 16M parameters |
| G7 | Exact-likelihood normalizing flow | `flow_exact_likelihood`; RealNVP backend | Teacher/student KL and NLL with generated image grids |
| G8 | Visually explicit mode recovery | `flow_mode_recovery` | Missing-class mass, class KL, feature distance, grids |
| G9 | Beyond-MLE qualitative stress test | `diffusion_mode_recovery` | DDPM mode metrics and grids under the same schedules |

The complete configuration is `configs/generative/full.yaml`; the CPU protocol validation
is `configs/generative/smoke.yaml`. See `docs/GENERATIVE_EXPERIMENTS.md` for the claim boundary
and exact execution order.

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
