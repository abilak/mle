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

## Neural language-model extension

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) supplies the corpus
used to pretrain the frozen neural teacher and the independent real-corpus replication. The
dataset card identifies English text generation, approximately 1M-10M records, and the
CDLA-Sharing-1.0 license. The implementation uses the official train and validation splits,
the GPT-2 tokenizer, and at most 100,000/10,000 records by default.

Each record becomes a fixed 128-token sequence: a BOS token, truncated story tokens, EOS, and
a dedicated padding token when needed. The padding token is distinct from BOS/EOS and is
excluded from attention, training labels, likelihood totals, and n-gram diagnostics. The
preparation command stores integer token arrays, tokenizer files, SHA-256 digests, row counts,
sequence length, source name, padding strategy, and a combined fingerprint under
`data/generative/`. The frozen teacher sees only the prepared training array. The real-corpus
endpoint is calculated only on the separately prepared validation array.

Teacher-generated train and test sequences are content-addressed by teacher checkpoint,
sample count, length, and seed. A student never trains on the fixed teacher test set.

## Neural vision extension

[MNIST](https://storage.googleapis.com/cvdf-datasets/mnist/) is the default vision dataset;
`FashionMNIST` can be selected without changing code. The project downloads the original IDX
files into the configured vision-data directory and verifies every file against its published
MD5 checksum. No `torchvision` binary extension is required. The official 60,000-example
training and 10,000-example test partitions are kept separate. Training images fit the
flow/diffusion models and their initial missing-mode checkpoints. Test images are used only to
validate the frozen classifier and construct balanced reference features.

The mode-recovery real-data stream is deterministically balanced across all ten labels. The
initial model excludes the preregistered class 8. A classifier trained once on the training
partition must exceed the configured test-accuracy threshold before mode results are
accepted. Generated samples, predicted class distributions, and classifier features are
stored; test labels never influence training or schedule selection.
