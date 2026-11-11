# Reward baselines

The question is not "did the reward go up?". It is "did *this* reward go up more than a reward that does not know the answer?".

Shao et al., *Spurious Rewards* §2.2, replace the ground-truth check with a ladder. We keep their ladder and add a few controls that matter in a lab. One field, `reward_name`, swaps the function. The prompts and the trainer stay put. Each name is a separate run from the same starting weights. Do not chain them: a format-only run that starts from a ground-truth checkpoint is not a control.

## The ladder

| name | what it rewards | where it comes from |
|---|---|---|
| `ground_truth` | `env.score`, the checked reward in [0, 1] | the task |
| `majority_vote` | 1 if the extracted answer equals the most common answer among the rollouts | §2.2, 64 samples in the paper. Pass them in `meta["rollouts"]`. If they are missing the reward is 0 and the diagnostic says `no_rollouts`. It does not quietly use the true label. |
| `format_only` | 1 if the env can parse an answer. If `format_style=boxed`, 1 if there is a non-empty `\boxed{...}` | §2.2. Theirs is the box. Ours is the tag, unless you ask for the box. |
| `incorrect_label` | 1 if the extracted answer equals a **wrong** label | §2.2. They take a majority label that is already wrong, so the label is still something the model might say. We do that when rollouts are provided and the majority is wrong. Otherwise we use `env.corrupt`. |
| `random_p50` | 1 on about half of (prompt, completion) pairs | §2.2, \(\gamma = 0.5\). A hash of seed, prompt, and completion, so scoring twice gives the same bit. They also tried other \(\gamma\) in Appendix B; \(\gamma = 0\) is a constant 0 and the gradient is zero. |
| `constant_one` / `constant_zero` | always 1, or always 0 | a control they discuss at \(\gamma = 0\). Always-1 is "reward every sample the same", which has no ranking inside a group. |
| `length_reward` | more words, capped at 1 | not in their ladder. It is here because length is the usual hack, and you want to see a trainer chase it. |
| `no_repetition` | 0 if a line or a token of length \(\ge 4\) repeats more than 10 times, else 1 | Appendix G. They give 0 when any string repeats more than 10 times, else 1. They do not publish the matcher. Ours ignores short common words so "the" eleven times is not a hit. It helped Qwen2.5-Math-7B and 1.5B on MATH and AMC, and did little or worse on other models (Appendix G). |
| `prior_only` | the true score is *logged*, and the optimizer does not step | the base model. A gain that also appears here was not learned in this run. |

GEN's majority and incorrect rewards compare the normalised step text exactly. That is sparse on purpose. The dense score stays in `ground_truth`. Mixing a dense incorrect reward with a binary one makes the ladder hard to read.

## What a result would mean

These are interpretations, not results. Nothing in the grid has been trained.

- Ground truth up, and every control flat: the run is using the checker. Still look for hacks and contamination. A real checker can be gamed.
- Random or format captures most of the ground-truth gain: the gain is the prior or the format, not the task. This is what they saw on Qwen2.5-Math-7B: random **+21.4** MATH-500, ground truth **+29.1** (abstract, §2). Incorrect labels were **+24.1** on that model and benchmark (§2). On AMC, format **+13.8**, incorrect **+24.1**, random **+21.4**, against about 27 to 29 points for majority vote and ground truth (§2).
- The same rewards on Llama3.1-8B-Instruct and OLMo2-7B: Figure 1's caption says the spurious rewards do not work there. §3 is more specific. OLMo stays flat under spurious rewards and gains mainly from ground truth. Llama's gains, where they exist, come from informative signals and are smaller. One family's curve is not a method.
- `no_repetition` up on one family only: same story. Appendix G.
- `constant_one` matches a trained run inside a GRPO group: every sample got the same reward, advantages are ~0, and the step is KL and entropy. If the eval number still moved, look at the clip and at logging, not at the reward.
- `prior_only` already high: do not train. You are measuring the base model.

Code reasoning (65% of MATH-500 answers, then over 90% under RL, §5) is their example of a behaviour the clip can amplify. We do not claim a lab model does this. We do not have the run.

## Why two families, and why cheap first

§3's practical warning is to test a reward on more than one model family. The grid names three small ones so the first check is not a 7B:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `allenai/OLMo-2-0425-1B-Instruct`

The paper's own comparison points (Qwen2.5-Math-7B, Llama3.1-8B-Instruct, OLMo2-7B) are listed in the yaml and are not the first job. CI does not download any of them. `download_in_ci: false` is on every expanded run. `posttrain.configs.load_grid` only reads the yaml.

## Compute plan, cheapest first

All of this is **not run**.

1. CPU smoke (`--smoke`). Tiny model, toy strings, minutes, no download. Confirms the code. Does not confirm a method.
2. One env, ERR (short answers), Qwen2.5-0.5B-Instruct, LoRA. SFT on gold completions for a few hundred steps. T4: fp16, not bf16. Checkpoint to Drive, including optimizer and RNG.
3. DPO on that adapter, checker-ranked pairs (chosen = gold, rejected = a wrong label). These are not human preferences. Say so.
4. GRPO, group size 4, ground truth only, tens of steps, KL on.
5. The same GRPO from the **same** start with `format_only` and `random_p50`. If those two match ground truth, stop and read the curve as format or prior.
6. Only if ground truth beats both: the rest of the ladder. Majority vote needs 64 rollouts cached once. That cache is the expensive part. Do it once.
7. The second family (Llama 1B or OLMo 1B), same three rewards, before any 7B job. If the gap disappears, the 0.5B result was a family effect.
8. 7B on an A100 only after step 7. bf16 is fine there.

`scripts/colab/run_posttrain.py` implements this order. The notebook points at it. Neither has been executed here.
