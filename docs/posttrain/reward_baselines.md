# Reward baselines

Shao et al., Section 2, train the same model on rewards that carry less and less information about the answer. On Qwen2.5-Math-7B, MATH-500 moved by +29.1 with ground truth and by +21.4 with a random reward (abstract and Figure 1). Format reward, incorrect labels, and majority vote also moved that model. Section 3 and Figure 3: the same spurious rewards did not help Llama3.1-8B-Instruct or OLMo2-7B. The result depends on the model family. One Qwen run is not a result about "RL for labs".

We have not run this ladder. The files in `configs/posttrain/` name the models and the rungs. They do not download weights.

## The rungs

`posttrain/reward_baselines.py`. One flag. Same prompts, same trainer.

| Flag | What it rewards | What a gain would mean |
|---|---|---|
| `ground_truth` | The env's checked reward (PQA, ORD, ERR, GEN). | The checker and the prior both could be responsible. Compare with the rows below. |
| `majority_vote` | 1 if the extracted answer equals the most common answer in the group. The paper uses 64 samples. We use the group you pass and we record the count. We do not pretend it was 64. | The policy is being pulled toward what it already says most often. |
| `format_only` | 1 if there is a non-empty `\boxed{...}`. This is the paper's format reward. A lab completion that only has `[ANSWER_START]` scores 0 here. | The gain is the boxed habit, which some math models were trained to emit. |
| `incorrect_label` | 1 if the extracted answer equals one specific wrong label in `meta["incorrect_label"]`. The paper rewards a specific wrong answer, not "anything incorrect". | If this matches ground truth, the model is learning to emit an answer span, or a near-miss the prior already liked. Section 2's footnote says incorrect labels may sit close to the truth and may act as a format reward. |
| `random_p50` | 1 with probability 1/2, seeded from the prompt, the completion, and the seed. | If this matches most of the ground-truth gain, the gain is the prior plus the optimiser's bias (the clip), not the task. |
| `constant_1` | Always 1. | No ranking. With a mean baseline the advantage is 0. |
| `constant_0` | Always 0. | Same, advantage 0. A control for "any update at all". |
| `length_reward` | Depends only on word count. | A control for length bias. Not a length penalty on correct answers. |
| `no_repetition` | 0 if any word or any 8-character span repeats more than 10 times, else 1. Appendix G: the paper gives 0 when any string repeats more than 10 times, else 1, and reports a gain on Qwen2.5-Math and not on the other families. Our 8-character rule is the operational reading of "any string". | A gain means "do not repeat yourself" was a useful prior, not that the protocol is right. |
| `prior_only` | No parameter update. The logged reward is still the ground-truth score. | The score of the SFT model before RL. Every later run has to beat this, not beat zero. |

Code reasoning in the paper (Section 5, Table 1) is the example of a prior RL can surface: Qwen2.5-Math-7B already wrote Python in 65.0% of MATH-500 answers, and under spurious rewards that rate went to about 90% within 15 steps and as high as 95.6% for random rewards (Section 5.2). We do not reward the string "python". The ladder is the test of whether our checker is doing more than surfacing a prior.

## Model families

`configs/posttrain/reward_ladder.yaml` names three families and nothing else:

- `Qwen/Qwen2.5-0.5B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `allenai/OLMo-2-0425-1B`

Confirm the hub id before a launch. CI does not download them. The paper's split was Qwen versus Llama and OLMo. A lab result that appears in only one family is not a result yet.

## KL on and KL off

The ladder is crossed with KL on, KL off, clip on, and clip off in `configs/posttrain/kl_grid/spurious_kl_on_off.yaml`. KL off is marked as an ablation. Read [kl_anchor.md](kl_anchor.md). If random reward only works with the clip on and KL off, that matches Section 4 and Figure 4, and it is not a reason to train the lab policy that way.

## Compute order, cheapest first

Not run.

1. CPU smoke in this repo (tiny model, toy strings).
2. Qwen2.5-0.5B, LoRA, SFT then DPO then GRPO, short completions, group size 4. fp16 on a T4. A T4 has no bf16.
3. The same three stages for Llama-3.2-1B-Instruct.
4. The same three stages for the OLMo model.
5. The full ladder (ten rungs) on the family that finished, then the others.
6. The KL and clip cross on `random_p50` and `ground_truth` only, before you spend a run on every cell.

Majority vote at 64 samples is the expensive rung. Do it after a rung with group size 4 already looks stable.
