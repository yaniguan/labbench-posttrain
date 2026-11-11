# The KL anchor

An automated lab often has no absolute ground truth. The reward is a checker: a parser, a similarity, a unit test. Those checkers are wrong in ways the [hacking notes](reward_hacking.md) list. The SFT model was trained on the format, the content, and the style a scientist preferred. That checkpoint is the thing RL is allowed to move away from only a little.

So the reference policy in DPO, PPO, and GRPO is the **SFT checkpoint**, not the raw pretrained model. KL is **on** in every default. KL off exists only as an ablation, and the trainer prints a warning.

## Defaults

| Method | Coefficient | Where the KL sits | Why this number |
|---|---|---|---|
| DPO | beta 0.1 | Inside the sigmoid margin. beta is the KL weight. | 0.1 is the usual DPO default. It multiplies a difference of log-prob sums, so it is not "0.1 nats per token" in the PPO sense. |
| PPO | 0.04 | Subtracted from the per-token reward, then fed to GAE. Detached. | Rewards are in [0, 1]. 0.04 nats is a visible tax on a token that has wandered. InstructGPT used about 0.02. A lab action is harder to undo than a sentence, so the default is a bit tighter. |
| GRPO | 0.01 | Added to the loss. Gradient flows through k3. | NeMo's GRPO recipe uses 0.01 (`examples/configs/grpo_math_1B.yaml`, `loss_fn.reference_policy_kl_penalty`). Advantages are standardised, so a PPO-sized 0.04 can dominate the policy term. 0.01 leaves room for the checker and still pulls toward SFT. |

The same number is not the same strength in the three losses. The sweeps use 0.001, 0.01, 0.04, and 0.1 for each method so you can see that, not because those four values are interchangeable. Configs: `configs/posttrain/kl_grid/`. **Not run.**

Shao et al. disabled KL in their GRPO runs (Section 4: they omit the KL term). That is part of why a random reward could move Qwen. We do not copy that default. The ablation file is `kl_grid/kl_off_ablation.yaml`, marked `ablation: true` and `recommended: false`.

## Estimators

Implemented in `posttrain/kl.py`. Formulas and gradients are in [losses_and_gradients.md](losses_and_gradients.md).

- **k3** is the default. Low variance, always non-negative, and the gradient pulls \(\log \pi\) toward the reference. NeMo's on-policy KL section (`docs/guides/grpo.md`) uses this form.
- **k2** also pulls toward the reference. It is a squared log-ratio, not an unbiased KL.
- **k1** is unbiased in expectation and the gradient is +1 everywhere. Descent only pushes probability down. Do not use it as the anchor if you care about the direction.

## Reward penalty vs loss term

- **PPO style** (`kl_in_reward=True`, the PPO default): subtract \(\beta\) times the detached per-token KL from the reward before GAE. The advantage changes. The direct derivative of k3 is not in the graph. This is the classical PPO penalty.
- **GRPO style** (`kl_in_reward=False`, the GRPO default): add \(\beta\) times k3 to the minimised loss. The gradient of k3 is applied. This matches NeMo's `reference_policy_kl_penalty` on the clipped policy loss.

Both are implemented on both trainers. Do not turn both on at once. The code does not: if the KL is in the reward, it is not also added to the loss.

## Adaptive controller

`AdaptiveKLController` is the clipped proportional controller used by Ziegler et al. and InstructGPT. If the measured KL is above the target, the coefficient grows. The error is clipped to \(\pm 0.2\) so one batch cannot explode it. Implemented. The target values in `kl_grid/adaptive.yaml` are **not run**.

## Fixed anchor vs refresh

`refresh_reference_every=0` is the default. The SFT weights stay the reference for the whole run.

A positive number copies the current policy into the **training** reference. The KL penalty then allows slow drift, because the target moves. The logged `kl_to_sft` still uses a frozen copy of the original SFT checkpoint (`sft_anchor`), so the report can show drift the training penalty no longer sees. Refresh is implemented. It is not the lab default. On a large model the extra copy costs memory. The Colab note says so.

## SFT loss as a second anchor

`sft_coef` adds token NLL on the scientist's completions. Default 0. The KL term is the anchor. Mixing SFT back in is a second leash when the checker is especially easy to game. Implemented. The grid is `kl_grid/sft_mix.yaml`. **Not run.**

## What to log, and the question the report asks

For every KL setting, log:

- proxy reward (the number in the loss)
- held-out gold metric (never in the loss)
- KL to the SFT model
- format-compliance rate
- response length
- entropy
- distance between answer-style unigrams and the SFT outputs
- hacking-detector flags

`posttrain/kl_report.py` prints the question: does turning KL down raise the proxy while gold or style gets worse, and where is the knee? With no rows it says **not run**.

`scripts/posttrain/plot_reward_vs_kl.py` plots proxy reward (and gold, if the column is present) against KL to the SFT model. If the log is missing it prints `not run` and writes nothing. No curve from a real training run exists.

## Ladder × KL

Random reward, format reward, and the other rungs should be run with KL on and with KL off, and with the clip on and off. KL and the clip both limit how far a spurious reward can move the policy (Shao et al., Section 4, Figure 4, the no-clip panels). The cross is `configs/posttrain/kl_grid/spurious_kl_on_off.yaml`. **Config only, not run.**

The rows with `kl_coef: 0` are ablations. They are not a recommendation. A result where random reward only works with KL off and the clip on is evidence for the paper's clipping-bias story, not evidence that the checker taught the task.
