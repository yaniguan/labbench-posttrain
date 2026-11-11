# Losses and gradients

The loss functions live in the trainer files so you can read the sign and the mask in one place. This note is the derivative, and the test that checks it. The tests are in `tests/test_posttrain_gradients.py`. They run on CPU on tensors, not on a lab model.

Notation: \(x\) is the prompt, \(y\) is the completion, \(\pi_\theta\) is the policy, \(\pi_{\mathrm{ref}}\) is a frozen copy, \(\pi_{\mathrm{old}}\) is the policy that sampled the tokens.

## SFT

Completion tokens only. Let \(m_t = 1\) when token \(t\) is a completion token and \(0\) when it is a prompt token or padding.

$$
\mathcal{L}_{\mathrm{SFT}}
= -\frac{1}{Z}\sum_t m_t \log \pi_\theta(y_t \mid x, y_{<t})
$$

Token-level normalisation sets \(Z\) to the number of completion tokens in the batch. A long answer then has more terms, so it pulls harder. Sequence-level normalisation averages inside each sequence first, then across sequences, so each example has the same weight and a long answer's tokens each get a smaller gradient. Both are in `sft_nll`. The test checks that a long, confidently wrong example moves the token-level loss more than the sequence-level loss.

The gradient is the usual next-token gradient, restricted by the mask:

$$
\nabla_\theta \mathcal{L}_{\mathrm{SFT}}
= -\frac{1}{Z}\sum_t m_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})
$$

**Why mask the prompt.** The prompt is given. Training on it spends the loss on copying the question, and a long protocol dominates a short answer. The test `test_prompt_token_targets_have_zero_logit_grad` checks the precise claim: the logit that would have predicted a prompt token has gradient exactly 0. The last prompt position is *not* masked as a predictor, because that logit predicts the first answer token. Prompt embeddings can still receive gradient through attention. That is the answer using the question as context, which is what we want.

**Packing.** Not implemented. If you concatenate examples to avoid padding, attention has to stop at the boundary. Without that mask, example 2's answer is trained on example 1's tokens.

**Loss and length.** We always divide, either by token count or by sequence count. A raw sum would let the longest row set the step size by itself.

## DPO

For one pair, with \(y_w\) chosen and \(y_l\) rejected:

$$
\Delta =
\big(\log \pi_\theta(y_w\mid x) - \log \pi_{\mathrm{ref}}(y_w\mid x)\big)
-
\big(\log \pi_\theta(y_l\mid x) - \log \pi_{\mathrm{ref}}(y_l\mid x)\big)
$$

$$
\mathcal{L}_{\mathrm{DPO}} = -\log \sigma(\beta \Delta)
$$

The batch loss is the mean. This is the expression in `dpo_loss`. Sequence log-probs are sums of token log-probs unless `length_normalize` is on.

**Beta.** \(\beta\) scales the margin. The implicit reward is \(\beta \log(\pi/\pi_{\mathrm{ref}})\). Larger \(\beta\) saturates the sigmoid sooner, so the policy is not pushed as far from the reference. Smaller \(\beta\) lets a modest gap keep a large gradient. The usual start is 0.1. It is an argument, not a tuned result.

**The reference.** It is a frozen copy, normally the SFT model. Without it, \(\Delta\) is only "which answer is more likely now", and both can drift with nothing to compare to. `run_smoke` deep-copies the tiny model before the first step.

**Gradient.** Let \(\sigma\) be the logistic sigmoid. For a single pair,

$$
\frac{\partial \mathcal{L}}{\partial \Delta} = -\beta \, \sigma(-\beta \Delta)
$$

The weight \(\sigma(-\beta \Delta)\) is near 1 when the policy prefers the rejected answer, and near 0 when it already prefers the chosen answer by a wide margin. Pairs the model already gets right stop dominating the step. The test compares this formula to autograd, and checks one coordinate by finite difference. Because the implementation averages over the batch, the parameter gradient is the formula divided by the batch size. The test divides by the batch size too.

**Likelihood displacement.** The loss depends on the gap, not on the level. Both log-probs can fall while \(\Delta\) rises: the rejected answer fell more. The pair looks better and both answers got less likely. `likelihood_displacement` flags that pattern. The smoke log records it. It is not an error by itself; it is a warning. `sft_coef > 0` adds the chosen-answer NLL, which pulls the chosen log-prob up. It is off by default so the smoke loss is the plain DPO loss.

**Length.** A sum of token log-probs is more negative for a longer string. A long chosen answer is then harder to prefer than a short one, even at the same per-token quality. Dividing by length removes that penalty and can also hide a real difference that lives in the extra tokens. Default is the sum, which is the published loss. Turn normalisation on when you are comparing that bias, not because it is always better.

## PPO and GRPO policy term

Both minimise the negative of the clipped surrogate. For a token,

$$
\rho = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\mathrm{old}}(y_t \mid x, y_{<t})}
$$

$$
\mathcal{L}_{\mathrm{clip}}
= -\min\big(\rho A,\; \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)\, A\big)
$$

\(A\) is an advantage. PPO's \(A\) is a GAE residual around the value head, then standardised over the valid tokens in the batch. GRPO's \(A\) is constant across the tokens of one completion:

$$
A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}
\quad\text{or}\quad
A_i = r_i - \mathrm{mean}(r)
$$

when `std_norm` is off. The second form is what Shao et al. write in §4 equation (1). The first form is the usual GRPO default. Dividing by a tiny std turns a near-tie into a huge step. That is the Dr. GRPO objection to the std term. We still **skip** a group whose std is about 0, under either setting, and we count it in `zero_var_frac`. A tied group has nothing to say. The test checks that a non-tied group's advantages sum to 0, and that a tied group is marked and zeroed.

**When the gradient is zero.** Shao et al., Appendix B, case by case (they maximise the min; we minimise the negative, so the zero region is the same):

- \(A \ge 0\) and \(\rho > 1+\epsilon\): the clip is the smaller term and it does not depend on \(\rho\). Gradient 0.
- \(A < 0\) and \(\rho < 1-\epsilon\): same, on the low side. Gradient 0.
- \(A \ge 0\) and \(\rho < 1+\epsilon\): the gradient flows.
- \(A < 0\) and \(\rho > 1-\epsilon\): the gradient flows. In particular a *high* ratio with a *negative* advantage is **not** clipped. The unclipped term is the worse one, so it is the one that is used.

`test_clip_region_gradient_is_zero` checks those four situations, including that turning the clip off restores a gradient at a point where the clip had killed it.

**The paper's point about which tokens get clipped.** With \(\epsilon = 0.2\) (§4 example): a token at old probability 0.85 would need new probability above \(0.85 \times 1.2 = 1.02\) to hit the upper clip. Probability cannot exceed 1, so that token never takes the zero-gradient branch on the way up. It keeps a non-negative push. A token at 0.02 hits the upper clip above 0.024, so a small increase is clipped and then pushed back. Random rewards, which have no task signal, can still amplify tokens the pretrained model already liked. Figure 4: random rewards gave a consistent gain only when the clip was present. Three ways of removing the clip effect are in the paper: delete the clip, take one update per rollout (the ratio stays 1), or match the mini-batch to the rollout. `use_clip=False` is the first. `updates_per_rollout=1` is close to the third, and it is why a single on-policy step often logs a clip fraction of 0. The clip shows up on the second update of the same samples. Smoke uses two updates so the path runs; the fraction can still be 0 if the ratio stays inside \((1-\epsilon, 1+\epsilon)\). A null bucket in the log means no tokens fell in that probability bin. It is not a measured rate of 0.

**Loss normalisation (GRPO).**

- `token`: mean over completion tokens. Long answers weigh more.
- `sequence`: each rollout weighs the same. Long answers have a smaller per-token gradient.
- `dr_grpo`: sum of token losses divided by (number of rows \(\times\) a fixed max length), not by this answer's length. Length does not rescale the step. Pair it with `std_norm=False` if you want the Dr. GRPO setup. That ablation is **not run** on a real model.

## KL

The per-token penalty is the Schulman k3 estimator, used in PPO and GRPO:

$$
\widehat{\mathrm{KL}}
= \exp(\log \pi_{\mathrm{ref}} - \log \pi_\theta)
- (\log \pi_{\mathrm{ref}} - \log \pi_\theta) - 1
$$

It is 0 when the two log-probs match, and non-negative otherwise.

$$
\frac{\partial \widehat{\mathrm{KL}}}{\partial \log \pi_\theta}
= 1 - \frac{\pi_{\mathrm{ref}}}{\pi_\theta}
$$

Near the reference this is about \(\log \pi_\theta - \log \pi_{\mathrm{ref}}\), so the penalty pulls the token back. The test checks the derivative. Shao et al. turned KL off in their runs (footnote in §4). Our default `kl_coef` is 0 for that reason. The smoke path sets a small coefficient so the term is actually computed. For a policy that can touch lab hardware, leave KL on: the checker does not price every unsafe action, and the reference is the last policy you were willing to run.

## Value loss (PPO only)

$$
\mathcal{L}_V = \tfrac{1}{2}\max\big((V-R)^2,\; (V_{\mathrm{clip}}-R)^2\big)
$$

with \(V_{\mathrm{clip}}\) the value after clipping the move from the rollout-time value. The max stops the critic from chasing a return by jumping. Outside the clip, if the unclipped error is the smaller one, the gradient in \(V\) is 0. The test sets the unclipped error to 0 and the clipped error large, and checks the gradient is 0. `use_clip=False` is plain MSE.

GAE puts the scalar reward on the last completion token and backs it up. With \(\gamma = \lambda = 1\) and a zero value function, every earlier token's advantage equals that reward. The test checks this. Padding stops the backup because the next mask bit is 0.

## What the step log records

`StepLogger` prints one short line and keeps a dict. Fields, when the method has them:

- loss, and the parts (policy, value, KL, entropy)
- global gradient norm, and the norm by layer group
- fraction of tokens clipped, overall and by old-policy probability bucket
- approximate KL (mean k3) and policy entropy
- reward mean and std
- fraction and count of zero-variance groups
- mean response length
- advantage mean and std
- for DPO: chosen log-prob, rejected log-prob, margin, pair accuracy, and whether this step looked like likelihood displacement

A smoke line looks like `sft step=0 loss=... finite=True`. `finite=False` is a failed run, not a result to quote.
