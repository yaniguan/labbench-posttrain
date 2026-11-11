# Contamination and memorisation

A training gain on questions the model already memorised is not a new lab skill. These checks ask whether a model is reproducing text it has seen. They do not train.

Shao et al. use AIME 2025 as a control because it was released after every model in their study (Figure 13 caption). Weak and spurious rewards still move some models on older sets. A post-cutoff set is where memorisation of those particular items cannot explain the score. They also note that AIME has 30 questions, so gaps under about 2 points may not be significant. See [evaluation](evaluation.md) for the standard-error arithmetic.

`configs/posttrain/aime_holdout.yaml` is that idea as an input slot. The question list is empty. No AIME score has been computed. Status: **not run**.

## What each outcome means

- High score on a post-cutoff set *before* any of our training: the items are easy, or the model saw something very close. Shuffle the options and paraphrase before calling it contamination.
- Our training gain shows up on older items and not on the post-cutoff holdout: the gain may be memorisation of the older set.
- The gain shows up on the holdout as well: harder to explain as memorisation of those items. It can still be format, prior, or a hack. The holdout does not replace the reward ladder.

## The checks

**Prefix completion.** Give the model the first 10%, 25%, 50%, and 75% of the text. Greedy-decode the rest. For the continuation, record:

- exact match after stripping
- normalised token edit distance (0 is a copy)
- ROUGE-L, implemented as the F1 of the token longest common subsequence
- LCS length divided by the gold length
- token overlap, as a bag of tokens

Always compare to two controls built from the same text: a shuffle of the tokens, and a light perturbation (one swap, one deletion). A high score on the original and a low score on both controls is the memorisation pattern. A high score on all three is easy text. The metrics are pure functions and are unit-tested on strings.

**Answer given a partial question, and option order.** `option_order_effect` takes, for each item, whether the original and the shuffled form were correct, and whether the letter stayed while the content moved. A sharp accuracy drop with the letter stuck is a memorised letter. Accuracy that stays, with the content followed to its new letter, is not.

**Perplexity ratio / min-k%.** The simple version: mean of the lowest 20% of token log-probs on the original, minus the same number on a paraphrase (`membership_gap`). A less negative original is the weak "this looks memorised" direction (the min-k% idea). The weakness, which the code does not fix: a hard paraphrase is also more negative, and a short easy sentence is less negative, whether or not it was in training. A single gap is not a membership decision. You would need a threshold fit on text you know was not seen. We do not fit one.

**Canary in outputs.** Search generations for the LAB-Bench canary strings. A hit means benchmark text is leaking out of the model. The same search refuses a training corpus. LAB-Bench paths are refused as training inputs by `assert_not_labbench_training_path`.

## What was run

`toy_demo` builds a tiny random GPT-2, fits it to one short sentence that is not from a benchmark, and runs the prefix report against a second tiny model that was not fit. The test checks that the fitted model's continuation score on that sentence beats the cold model. That is a check of the metric code. It is not a statement about Qwen, Llama, or OLMo.

`scripts/run_contamination_eval.py --demo` prints that report. `--model NAME --labbench` is the later path: it loads a real model and scores public LAB-Bench text. That path is eval only. There is no training flag. It has **not run**. LAB-Bench is CC-BY-SA and stays out of every training list in this package.

Do not put LAB-Bench questions into SFT, DPO, or RL data, even with the canary field stripped. Stripping the canary does not make the questions trainable. The license and the benchmark rules are the reason. The canary is only a tripwire.
