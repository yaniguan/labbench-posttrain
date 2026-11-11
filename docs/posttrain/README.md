# Post-training for an automated lab

This is the entry point for `posttrain/`. The code trains a policy on text that an automated lab can check. The docs say how to read a reward curve before you trust it.

Nothing in this folder was trained on a GPU. Smoke logs are CPU runs of a tiny random model. Every other number is **not run yet**.

## The pipeline

1. **Data.** Prompts and completions a scientist would accept. BioProBench (CC BY-NC 4.0) is the reward source for PQA, ORD, ERR, and GEN. LAB-Bench is CC-BY-SA and is evaluation only. Do not put it in a training file. The canary strings stay in `data/labbench/` and the checker searches for them.
2. **SFT warm start.** Train on the scientist's preferred format, content, and style. Mask the prompt. See [losses and gradients](losses_and_gradients.md).
3. **Preference data, then DPO.** Chosen vs rejected completions. The reference model is the SFT checkpoint. `beta` is the KL weight. See [KL anchor](kl_anchor.md).
4. **RL with a verifiable reward.** PPO (value head, GAE) or GRPO (group baseline, no value head). The KL term stays on and points at the same SFT checkpoint.
5. **Eval.** Accuracy with a bootstrap interval, pass@k, the reward ladder, the hacking report, and the contamination report. See [evaluation](evaluation.md).

## When to use which method

An automated lab does not have one kind of feedback.

- **Cheap checkers** (does the answer parse, is the permutation valid, did the numbers match) can score thousands of completions. GRPO fits. The group baseline needs several samples of the same prompt, which a checker can afford and a wet-lab run cannot.
- **Expensive experiment feedback** (a plate, a robot trajectory, a day of incubation) gives you one outcome per try. PPO's value head is there for that, because it can assign credit along a long trace. We still implement PPO on a single outcome at the last token. Process rewards are not implemented.
- **Noisy rewards.** A checker that is right 80% of the time will still move the policy. The spurious-reward ladder is how you tell a real signal from the prior. See [reward baselines](reward_baselines.md). If random reward captures most of the gain, you did not learn the task.
- **Long horizon.** A protocol is a sequence. Token-level loss gives long traces more weight. Sequence-level loss gives each protocol the same weight. Neither is free. The tradeoff is in the losses doc.
- **Safety on real equipment.** A wrong but confident liquid volume is worse than a refusal. The SFT model is the scientist's format and the cautious style. RL must not walk away from it. That is why the KL anchor defaults to the SFT checkpoint and why KL off is an ablation, not a setting. A held-out gold metric that is never in the loss is how you notice the walk. See [reward hacking](reward_hacking.md).

## Questions to ask before you trust a reward curve

Ask these in order. A rising curve does not answer them.

1. **Is it the prior?** Shao et al., Section 2 and Figure 1: on Qwen2.5-Math-7B, random reward gained +21.4 points on MATH-500 and ground truth gained +29.1. The same random reward did not help Llama3.1-8B-Instruct or OLMo2-7B (Section 3, Figure 3). Run at least two model families. The grid is in `configs/posttrain/reward_ladder.yaml`. It has not been run.
2. **Is it the format?** A format-only reward (a non-empty `\boxed{}`) can look like learning. Our lab prompts ask for `[ANSWER_START]` tags, which is a different format. Score both.
3. **Is it the clip?** Section 4 and Figure 4: with the clip removed, random reward stopped helping Qwen. High-probability tokens often cannot hit the upper clip, so they keep a non-negative push. Turn the clip off in a paired run. KL on limits the same drift. Run random reward with KL on and KL off (`configs/posttrain/kl_grid/spurious_kl_on_off.yaml`).
4. **Is it a hack?** Does the completion satisfy the regex and miss the task? Is one letter always chosen? Did the proxy rise while the held-out gold metric fell?
5. **Is it contamination?** A post-cutoff holdout (the paper uses AIME 2025, Appendix D) is the control. A high score there is not proof of memorising that set, because the set did not exist yet. A gap that exists only on older items can be contamination or a harder new set. See [contamination](contamination.md).
6. **Is the test big enough?** Thirty questions have a standard error of several points. Do not rank runs on a gap of two points. See [evaluation](evaluation.md).

## What is implemented

Each line is **implemented** (the trainer or the function runs), **config only** (a YAML for a later run, not launched), or **not implemented** (described, no code that pretends to do it).

| Option | Status | Where |
|---|---|---|
| SFT, prompt masked, token or sequence loss | implemented | `posttrain/sft.py` |
| LoRA | implemented | `posttrain/batching.py` |
| Sequence packing | not implemented | A causal pack without a block mask lets one example see the next. We pad instead. |
| DPO, beta, reference = SFT, length-normalised log-probs, optional SFT term, likelihood-displacement check | implemented | `posttrain/dpo.py` |
| PPO, GAE, clipped surrogate, value clip, entropy bonus, advantage norm | implemented | `posttrain/ppo.py` |
| GRPO mean or leave-one-out baseline, std on or off | implemented | `posttrain/advantages.py` |
| Clip on or off, asymmetric clip (clip-higher) | implemented | `policy_surrogate` |
| Token-level vs sequence-level policy loss | implemented | `reduce_token_values` |
| Dynamic sampling (drop zero-variance groups) | implemented | `filter_zero_variance` |
| Overlong filtering | implemented | `apply_overlong_filter` |
| Stop-properly truncation penalty | implemented | `apply_stop_properly` |
| Advantage clipping | implemented | `clip_advantages` |
| CISPO (detached clipped importance weight) | implemented | `use_cispo=True` |
| Reinforce++-baseline | implemented | `advantage_mode="reinforce_pp"` |
| GDPO-style per-component norm | implemented | `advantage_mode="gdpo"` |
| KL estimators k1, k2, k3 | implemented | `posttrain/kl.py` |
| KL in the reward (PPO) or in the loss (GRPO) | implemented | `kl_in_reward` |
| Adaptive KL controller | implemented | `AdaptiveKLController` |
| Reference refresh vs fixed SFT anchor | implemented | `refresh_reference_every`, default 0 |
| SFT loss mixed into RL | implemented | `sft_coef` |
| KL coefficient sweeps | config only | `configs/posttrain/kl_grid/` |
| Reward ladder across Qwen, Llama, OLMo | config only | `configs/posttrain/reward_ladder.yaml` |
| Group-relative length penalty for binary rewards | implemented | `group_length_penalty` |
| The other NeMo length penalties (shortest bonus, zMAD, profile bands, pass-rate scaling) | not implemented | described in `docs/guides/length-penalty.md` of yaniguan/RL |
| Simple importance-sampling weight | implemented | `use_is_correction`. With one framework the weight is 1. |
| ICE-POP, adaptive PPO clip, seq-mask-tis | not implemented | described in `docs/guides/grpo.md` and `docs/guides/prorlv2.md` |
| On-policy vs several epochs on one batch | implemented | `epochs` |
| Async lag, replay, in-flight weight updates | not implemented | `docs/guides/async-grpo.md` |
| Rejection-sampling fine-tuning | not implemented | |
| Best-of-n distillation | not implemented | |
| Process reward | not implemented | Outcome reward is what the envs return. |
| Partial credit inside PQA, ORD, ERR, GEN | implemented | in `bioenvs/`, not re-tuned here |
| DAPO overlong buffer penalty (linear in the overflow) | not implemented | We have the stop-properly scale and the hard filter. |
| Curriculum, difficulty filtering | not implemented | |
| Entropy bonus | implemented | `entropy_coef`, default 0 |
| Ensembles of checkers | not implemented | One code checker per env. |
| Noisy-label robust loss | not implemented | `incorrect_label` is a ladder rung, not a robust trainer. |
| On-policy distillation | not implemented | config only in `configs/posttrain/nemo_rl/distillation.yaml`, from `docs/about/algorithms/on-policy-distillation.md` |
| Reward-model training | not implemented | NeMo trains a Bradley-Terry RM (`docs/guides/rm.md`). We use code checkers. |
| NeMo RL launch YAML | config only, not run | `configs/posttrain/nemo_rl/` |

## How to run the CPU checks

```bash
python -m pytest -q
python -m posttrain.sft --smoke
python -m posttrain.dpo --smoke
python -m posttrain.ppo --smoke
python -m posttrain.grpo --smoke
```

Smoke builds a random GPT-2 in memory. It does not download weights. GRPO and PPO smoke use a random 50% reward so a group is not all zeros, and they log the toy string-match as the held-out gold number. That gold number stays near zero. The run is a code check.

A later GPU order, cheapest first, is in `scripts/colab/posttrain_colab.py` and `notebooks/colab_posttrain.ipynb`. Those were not run.

## Licenses

BioProBench is CC BY-NC 4.0. LAB-Bench is CC-BY-SA-4.0 and is not training data. Do not copy SES material into this tree. There is none here.
