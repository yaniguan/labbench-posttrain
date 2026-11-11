# Evaluation

`posttrain/eval_suite.py` builds one report for a checkpoint. Any section you do not pass in is the string `not run`. The demo report in the tests is computed from synthetic rows so the arithmetic runs. Those rows are not a model, and they are not a lab score.

No checkpoint of a real model has been scored. Task metrics are **not run**.

## What the report contains

When you pass the records:

- Accuracy with a bootstrap 95% interval and the seed. Also the binomial standard error.
- pass@k and average@k. pass@k is the unbiased estimator: \(1 - \binom{n-c}{k}/\binom{n}{k}\) when there are enough failures, else 1. A sample counts as a pass at 0.5. The report stores that threshold.
- Mean correct by task and by subset.
- Calibration: expected calibration error over confidence bins, if confidences were logged. PQA's confidence field is the natural input. If you did not log a confidence, the section stays `not run`.
- Response-length mean, median, p90, min, max.
- The reward-ladder table, if you pass one. Otherwise `not run`. Do not fill it from memory.
- The hacking-check summary.
- The contamination summary.
- Base versus SFT versus DPO versus PPO versus GRPO, as paired bootstrap intervals on the per-item difference. Lists must be aligned on the same items.
- McNemar on paired correct/incorrect bits, for the binary case. The chi-square tail is reported when scipy is importable. With few disagreements the test is weak; the counts are in the report so you can see that.
- A held-out gold block, marked `used_for_training: false`.

## Why 30 questions is a noisy gap

For a proportion \(p\) on \(n\) items the usual standard error is

$$
\sqrt{\frac{p(1-p)}{n}}
$$

At \(p = 0.5\) and \(n = 30\), that is \(\sqrt{0.25/30} \approx 0.091\). A 10 point gap is about one standard error. Two points is a fraction of one item (0.6 of a question). Shao et al. say differences under about 2 points on AIME may not be significant (Figure 13 caption). Report the interval. Do not rank two methods on a 30-item gap that the interval covers.

The bootstrap in the report resamples items. It does not invent precision the sample does not have. A wider interval is the result.

## Held-out gold

The monitor's gold split is also an eval split. Build it with `holdout_ids` before you build SFT or DPO pairs, and pass the same ids into the pair builders so those items never become training text. Score them with `env.score` only. If the report's held-out block was computed on items that were also in the loss, say so and do not call it gold. The empty report does not pretend this split was scored.

## How to read a partial report

`status: not run` means the whole report is empty. `status: partial` means some blocks were computed from records you passed, and the blocks still marked `not run` were not. A partial report from the unit-test demo is a software check. Quote the smoke losses as smoke losses. Do not quote them as ERR or PQA accuracy.
