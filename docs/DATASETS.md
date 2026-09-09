# Dataset provenance and leakage controls

## Code training and monitoring

[APPS](https://huggingface.co/datasets/codeparrot/apps) is used only from its 5,000-problem `all/train` split. Its dataset card describes manually curated programming problems, Python solutions, and 131,777 tests and lists the MIT license. The preparation step keeps records with at least one parseable Python solution and usable tests, then makes a deterministic difficulty-stratified 90/10 training/monitor split.

The monitor split may drive the explicitly labeled reactive and failure-targeted baselines. It is never part of the training corpus and is not a final benchmark.

## Code evaluation

[HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus) and [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) are used only for evaluation. Their Hub datasets contain 164 and 378 tasks, respectively, and list Apache-2.0. Evaluation uses [EvalPlus 0.3.1](https://github.com/evalplus/evalplus/releases/tag/v0.3.1), whose extended tests are substantially stricter than the original suites.

Before splitting APPS, the pipeline removes and logs:

- normalized exact prompt matches;
- prompt pairs with 5-gram Jaccard similarity at or above 0.80;
- human solutions with identical parsed Python AST structure to an evaluation solution.

This controls downstream fine-tuning leakage. It cannot prove that a pretrained base model never saw benchmark material, so baseline scores and changes-from-baseline are both retained.

## Reasoning transfer

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) `main` is used with its official train and test splits. A deterministic 10% subset of train becomes the monitor set; test remains evaluation-only. Exact answers are parsed from the final `####` field and compared numerically when possible.

## Generated artifacts

`data/processed/code_manifest.json` and `math_manifest.json` record row counts, library runtime, and dataset fingerprints. Processed JSONL records include source dataset, task ID, license, prompt, available human solutions, tests, inferred skill group, and original metadata. Raw and processed data are intentionally ignored by Git; rerun preparation to reproduce them.
