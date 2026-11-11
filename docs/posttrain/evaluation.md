# Evaluation

One report per checkpoint, from `posttrain/eval_suite.py`. If you pass no rows, the report says **not run**. A unit test builds a report from a toy bit-vector and labels it `toy demo, not a model result`. That demo is not a lab score.

## What the report contains

- Accuracy and a bootstrap percentile interval of the mean, over resampled items.
- pass@k: \(1 - \binom{n-c}{k} / \binom{n}{k}\). Any of the k samples correct.
- average@k: the mean of the first k samples. You pay for every sample on a bench, so the average is the cost-relevant number. pass@k hides misses.
- Per-task and per-subset accuracy when the rows carry those keys.
- Calibration bins when a confidence is present (PQA asks for one).
- Response-length mean, min, and max.
- The reward-ladder table. Until the ladder is run, this block says not run. See [reward_baselines.md](reward_baselines.md).
- The hacking-check summary. See [reward_hacking.md](reward_hacking.md).
- The contamination summary. See [contamination.md](contamination.md).
- Base vs SFT vs DPO vs PPO vs GRPO, as paired bit-vectors on the same items.
- A paired bootstrap interval on the difference, and an exact McNemar test on the discordant pairs.
- The held-out gold split the monitor watches. It is not a training reward.

## Why 30 questions is a noisy gap

The standard error of an accuracy \(p\) on \(n\) items is \(\sqrt{p(1-p)/n}\).

At \(p = 0.5\) and \(n = 30\), that is \(\sqrt{0.25/30} \approx 0.091\), about 9 points. A gap of a few points is inside the noise. Two methods can swap rank on a fresh 30-item draw without either having changed.

Shao et al. say this in the Figure 13 caption (Appendix D): AIME has 30 questions, and differences under about 2 points may not be significant. The standard error says the noise is larger than 2 points at a coin-flip accuracy. Treat "under 2 points" as a low bar for caution, not as the width of the interval. The report prints the standard error for the \(n\) you actually have.

ProtocolQA in the existing eval notes has 108 questions but only about 24 protocols. The effective \(n\) is the number of independent items, not the number of rows that share a protocol. A bootstrap that resamples rows understates the uncertainty when rows are dependent. Cluster by protocol before you rank models. That cluster bootstrap is not implemented. The row bootstrap is.

## Held-out gold

The monitor compares the proxy reward to a gold metric that never enters the loss. For this repo the natural split is BioProBench test versus train. Training on the test split and then calling it gold does not count. The report has a slot for that gold accuracy. It stays **not run** until a checkpoint is scored.
