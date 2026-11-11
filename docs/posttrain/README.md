# Post-training for an automated lab

This is the entry point for `posttrain/`. It is about how to think about a training run, not only about how to launch one.

No GPU and no LLM API key were available while this was written. Every trainer has a CPU smoke path on a tiny random model. Those losses are in the pull request. They are not lab scores. Task metrics are **not run**.

## The pipeline

```
data (BioProBench, CC BY-NC 4.0)
  -> hold out a gold split that the loss never sees
  -> SFT on checker-ranked gold completions
  -> preference pairs (chosen = gold, rejected = a wrong label)
  -> DPO
  -> RL with the verifiable reward (GRPO, then PPO if you need a value baseline)
  -> the report in posttrain/eval_suite.py
```

SFT is a warm start. It teaches the format the checker can parse: tags, a list of indices, a step per line. DPO then widens the gap between a completion the checker likes and one it does not, without sampling during the update. GRPO and PPO sample, score, and update. The reward is code in `bioenvs/`, not a learned reward model.

Do the stages in that order on a small model before you spend a 7B run. The Colab order is in [the reward-ladder note](reward_baselines.md) and in `scripts/colab/run_posttrain.py`.

## When to use which method

| situation | use | why |
|---|---|---|
| The checker is cheap and exact (PQA choice, ORD permutation, ERR verdict) | GRPO or PPO on `ground_truth` | You can afford many samples. The reward is a fact about the text, not a guess. |
| The checker is a similarity to one reference (GEN, ERR repair) | SFT and DPO first; RL only with the hacking monitor on | A valid protocol that is not the reference loses points. RL will chase the reference wording. |
| The reward is noisy or sparse (most groups tie) | bigger groups, or drop the prompt; do not divide by a tiny std | A near-tie divided by std becomes a huge advantage. `std_norm=False` and the zero-variance skip exist for this. |
| Feedback is a real experiment (hours, reagents, a robot) | offline methods: SFT, DPO, rejection sampling | You cannot roll out thousands of actions on hardware. Log the run, score it later, update offline. |
| The action can hit a real instrument | a KL penalty to a known policy, a held-out gold metric, and a refusal that scores *above* a confident wrong action | The checker does not see every unsafe action. Drift has a cost that MATH-500 does not. |

PPO is the one with a value head. Use it when the reward arrives at the end of a long completion and you want a baseline that is not just "the other samples of this prompt". GRPO is simpler and is the method in Shao et al. Start there. Add PPO when the GRPO advantage is too noisy and you can pay for the critic.

## Questions to ask before you trust a reward curve

Shao et al., *Spurious Rewards*, is the reason these questions are not optional. On Qwen2.5-Math-7B, GRPO with a random reward (1 about half the time) gained **+21.4** points on MATH-500, against **+29.1** for ground truth (abstract and §2). The same spurious rewards did not help Llama3.1-8B-Instruct or OLMo2-7B (Figure 1 caption; §3). A curve on one model family is not evidence.

1. **Is it the prior?** Run `random_p50`, `constant_one`, and `prior_only` (no update) on at least two families. If the random reward captures most of the ground-truth gain, the update is amplifying something the pretraining already liked. Their §4 account is the GRPO clip: a token with old probability 0.85 cannot reach the upper clip at all (the clip would sit above 1), so it keeps a non-negative gradient, while a token at 0.02 is clipped as soon as it moves a little. Removing the clip removed the consistent random-reward gain (Figure 4). `use_clip=False` is that ablation.
2. **Is it the format?** Run `format_only`. Their format reward is "does the answer contain a non-empty `\boxed{}`?" (§2.2). Ours is "can the env parse an answer?". If format-only matches ground truth, you taught the tags, not the protocol.
3. **Is it a hack?** Run `posttrain/hacking_checks.py` on the rollouts. Watch the monitor: if the training reward rises and the held-out gold score falls, stop. The gold split must never enter the loss. See [reward hacking](reward_hacking.md).
4. **Is it contamination?** A gain only on items the model could have seen is not a new skill. Their control is AIME 2025, released after the models (Figure 13 caption). The slot for that idea is `configs/posttrain/aime_holdout.yaml`. It is empty and **not run**. See [contamination](contamination.md).

Code reasoning is their concrete example of a prior that RL surfaces: Qwen2.5-Math-7B used code in 65% of MATH-500 answers before RL and in over 90% after, including under spurious rewards (abstract and §5.2). We do not train a "mention python" reward. We log nothing that would let us claim the same frequency on a lab model, because that run has not happened.

## Map

| path | what |
|---|---|
| `posttrain/sft.py` | masked cross-entropy, optional LoRA |
| `posttrain/dpo.py` | the DPO loss, plus the likelihood-displacement check |
| `posttrain/ppo.py` | actor-critic, GAE, clip, value clip, KL, entropy |
| `posttrain/grpo.py` | group advantage, clip on/off, std on/off, loss normalisation |
| `posttrain/rewards.py` | `reward(prompt, completion, meta) -> (float, diagnostics)` |
| `posttrain/reward_baselines.py` | the ladder |
| `posttrain/hacking_checks.py` | detectors |
| `posttrain/monitor.py` | stop or flag during training |
| `posttrain/contamination.py` | prefix, min-k, canary, option order |
| `posttrain/eval_suite.py` | one report; missing parts stay `not run` |
| `posttrain/logging_utils.py` | the per-step record |
| `configs/posttrain/` | names and switches; nothing downloaded in CI |
| `scripts/colab/run_posttrain.py` | the runner; not executed here |
| `notebooks/posttrain_colab.ipynb` | the same order, for a person to open |

Losses and gradients, with the tests that check them: [losses and gradients](losses_and_gradients.md).

## What is implemented, config only, or not implemented

**Implemented** (CPU-tested, not a claim about a lab model):

- SFT with the prompt masked, token-level or sequence-level mean, optional LoRA.
- DPO with beta, a frozen reference, optional length normalisation, optional SFT term, likelihood-displacement check.
- PPO with GAE, clipped surrogate, optional value clipping, per-token KL, entropy bonus, batch advantage normalisation.
- GRPO with group-relative advantages, clip on/off, std normalisation on/off, token / sequence / `dr_grpo` loss scaling, zero-variance groups skipped and counted, several updates per rollout, optional SFT term, optional length penalty, `prior_only`.
- The reward ladder, including majority vote, format, incorrect label, random, constants, length, no-repetition.
- Hacking detectors, trivial baselines, the monitor, canary refusal.
- Contamination metrics and a tiny-model demo. A script to point at LAB-Bench later, eval only.
- The eval report (bootstrap intervals, pass@k, average@k, calibration, paired bootstrap, McNemar).
- Rejection sampling and best-of-n selection, then SFT (`posttrain/rft.py`).
- A mean of several checkers (`rewards.ensemble`), with the disagreement kept in the diagnostic.
- Resume checkpoints: weights, optimizer, step, RNG.
- A type filter on the gold-pair builder. GEN's types are easy / standard / difficulty. The other envs' types are not a difficulty order.

**Config only** (the yaml is real, the run is not):

- The grid in `configs/posttrain/grid.yaml`: Qwen 0.5B, Llama 3.2 1B, OLMo 2 1B, times the ladder, times four envs. 120 jobs. Status `not run`.
- The paper's 7B comparison names, listed and not scheduled first.
- Clip, KL, group size, and loss-norm fields. A sweep of them is not launched.
- `configs/posttrain/aime_holdout.yaml`: the post-cutoff idea, with an empty question list.

**Not implemented:**

- Sequence packing. Padding is used instead. A packer without a block attention mask trains across example boundaries.
- A learned reward model, and a process reward (a score per step rather than per final answer). The env rewards are outcome scores, some of them with partial credit.
- A curriculum *schedule*. You can filter by `type`. Nothing moves that filter as training goes.
- A replay buffer. `updates_per_rollout > 1` reuses one group. It does not store old groups.
- A noise-robust loss. Random and incorrect rewards are controls, not a training recipe. Shao et al. say the same: spurious rewards are for analysis, not a recommended way to train (§2.2).
- Any run on a downloaded model. Do not fill a results table with guesses.

## Licenses and data rules

- LAB-Bench (`data/labbench/`) is CC-BY-SA-4.0. Eval only. The jsonl files contain the canary. `assert_training_corpus_clean` refuses a training list that contains it, and `assert_not_labbench_training_path` refuses that directory as a training path.
- BioProBench (`data/bioprobench/`) is CC BY-NC 4.0. Not MIT. The reward code reads it. Do not relicense it.
- No model weights, API keys, or secrets are added here.

## Run the smoke path

```bash
python -m posttrain.sft --smoke
python -m posttrain.dpo --smoke
python -m posttrain.ppo --smoke
python -m posttrain.grpo --smoke
python -m pytest -q
```

`--smoke` builds a tiny GPT-2 in memory. It does not download. A non-finite loss raises. Full training is `scripts/colab/run_posttrain.py` on a GPU, and that has **not run**.
