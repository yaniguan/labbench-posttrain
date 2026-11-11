# Losses and gradients

The losses are in `posttrain/losses.py`. We minimise them. A positive advantage should raise the token's log-probability, except where the clip sets the gradient to zero.

## SFT

For a completion token the loss is the negative log-probability. Prompt tokens are not targets.

\[
L_{\text{SFT}} = - \sum_t m_t \log \pi_\theta(y_t \mid x, y_{<t})
\]

\(m_t = 1\) on completion tokens, including the end token, and \(0\) on the prompt and on padding. The gradient of this loss with respect to the logits that would predict a masked token is zero. The last prompt position still has a gradient, because that position predicts the first completion token. The unit test checks the masked positions.

Why mask the prompt: the model did not write it. A long protocol would dominate the loss, and the model would spend capacity copying the question.

**Token-level** averages over tokens. A 200-token protocol counts ten times a 20-token one. **Sequence-level** averages inside each completion, then averages completions. Each protocol counts once, so a short one has a larger per-token gradient. NeMo's write-up is `docs/guides/grpo.md`, section Loss Normalization (`token_level_loss`). DAPO argues for token-level loss on long chain-of-thought (`docs/guides/dapo.md`, Token-Level Policy Gradient Loss). The original GRPO paper averages per sequence. If lengths are similar, the two are close. Our SFT default is token-level, matching NeMo's `NLLLossFn` note in that same guide.

**Packing** is not implemented. Putting two examples in one causal sequence without a block mask lets the second example attend into the first.

**Loss vs length.** Under the sum (token-level with a fixed token budget, or a summed log-prob), a longer target has a larger loss even when every token is equally likely. That is not the same bug as DPO's length bias, but it is the same arithmetic.

## DPO

\[
L_{\text{DPO}} = - \log \sigma \Big( \beta \big[ (\log \pi(y_w) - \log \pi_{\text{ref}}(y_w)) - (\log \pi(y_l) - \log \pi_{\text{ref}}(y_l)) \big] \Big)
\]

Let \(\Delta\) be the quantity inside the sigmoid, including \(\beta\). Then

\[
\frac{\partial L}{\partial \log \pi(y_w)} = - \beta (1 - \sigma(\Delta)), \qquad
\frac{\partial L}{\partial \log \pi(y_l)} = + \beta (1 - \sigma(\Delta)).
\]

The weight \(1 - \sigma(\Delta)\) is large when the model currently prefers the rejected completion. A pair the model already gets right contributes little. \(\beta = 0\) makes \(\Delta = 0\), the loss the constant \(-\log 0.5\), and both gradients zero. The trainer prints a warning. \(\beta\) is the KL weight. The reference is the SFT checkpoint. See [kl_anchor.md](kl_anchor.md).

**Length bias.** We sum token log-probs by default (`average_logprobs=False`). A short rejected answer has fewer negative terms, so its sum is higher even when it is worse. Set `average_logprobs=True` to divide by length. That removes the length bias and also changes the objective. The smoke log shows the pattern: the short rejection starts with a higher (less negative) sum than the longer chosen answer.

**Likelihood displacement.** DPO can raise the margin by lowering both log-probs, with the rejected one falling faster. `likelihood_displacement` is true when both the chosen and the rejected log-prob fell across the run. The smoke run did not do this. That is one run of a tiny model, not a finding about DPO in general.

**Optional SFT term.** `sft_coef * NLL(chosen)` is a second pull toward the text the scientist wrote. Default 0. The KL term is the anchor. The mix is implemented and off unless you set it.

## PPO and GRPO

Per token, with ratio \(\rho = \pi_\theta / \pi_{\text{old}}\) and advantage \(A\):

\[
L = - \min\big(\rho A,\ \mathrm{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) A\big)
\]

The gradient with respect to \(\log \pi\) is exactly zero when the clip is the value we keep:

- \(A > 0\) and \(\rho > 1 + \epsilon_{\text{high}}\): the token is already more likely than the clip allows. Pushing it further does not change the loss.
- \(A < 0\) and \(\rho < 1 - \epsilon_{\text{low}}\): the token is already less likely than the clip allows. Pushing it down does not change the loss.

Inside the interval the gradient is the usual score-function gradient, \(-A \rho\) on the minimised loss (equivalently \(A\) times the score if you maximise).

Shao et al., Section 4, equation (2), write the expected gradient under a random reward as zero inside the clip, a positive push on \(\log \pi\) when the ratio is below the lower clip, and a negative push when the ratio is above the upper clip. Their example uses \(\epsilon = 0.2\). A token with old probability 0.85 would need new probability \(0.85 \times 1.2 = 1.02\) to hit the upper clip, which is impossible. A token with old probability 0.02 hits the upper clip at 0.024. High-probability tokens keep a non-negative bias. Low-probability tokens get clipped. Figure 4: removing the clip removed the random-reward gain on Qwen2.5-Math-7B.

`clip=False` drops the clip so that gradient is not zeroed. `eps_high > eps_low` is clip-higher (DAPO, `docs/guides/dapo.md`; ProRLv2, `docs/guides/prorlv2.md`). The logger records the fraction of tokens whose clip binds, split by the behavior probability bucket, so you can see the bias in a real run. Smoke runs are on-policy with one update, so the ratio stays near 1 and the clip rarely binds. The unit test builds a ratio outside the range on purpose.

**CISPO** (`docs/guides/cispo.md`):

\[
L_{\text{CISPO}} = - A \cdot \mathrm{sg}\big(\mathrm{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big) \cdot \log \pi
\]

The clipped ratio is detached. The gradient is \(-A\) times that constant, which is not zero just because \(\rho\) left the interval. The unit test uses the same tensors as the PPO clip test: PPO's gradient is 0 and CISPO's is not. CISPO forces token-level reduction. NeMo rejects the sequence-level setting. We do too.

## KL

Let \(u = \log \pi_{\text{ref}} - \log \pi\).

- **k1:** \(\log \pi - \log \pi_{\text{ref}}\). Expectation under \(\pi\) equals \(\mathrm{KL}(\pi \| \pi_{\text{ref}})\). A single sample can be negative. Gradient w.r.t. \(\log \pi\) is \(+1\). Descent always lowers \(\log \pi\). It does not pull a too-small probability back up.
- **k2:** \(\tfrac{1}{2}(\log \pi - \log \pi_{\text{ref}})^2\). Always \(\ge 0\). Gradient is \(\log \pi - \log \pi_{\text{ref}}\), so descent walks toward the reference. This is a local approximation, not an unbiased KL.
- **k3:** \(e^{u} - u - 1 = \pi_{\text{ref}}/\pi - \log(\pi_{\text{ref}}/\pi) - 1\). Always \(\ge 0\). Expectation under \(\pi\) equals the KL, because \(\mathbb{E}_\pi[\pi_{\text{ref}}/\pi] = 1\). Gradient is \(1 - \pi_{\text{ref}}/\pi\). Descent decreases \(\log \pi\) when \(\pi > \pi_{\text{ref}}\) and increases it when \(\pi < \pi_{\text{ref}}\).

k3 is the default, matching NeMo `reference_policy_kl_type: k3` and the on-policy KL section of `docs/guides/grpo.md`. The optional on-policy form multiplies the term by the detached ratio. It is off by default, as in that guide.

## GRPO advantages

Mean baseline, group rewards \(r_1,\ldots,r_n\), mean \(\bar r\):

\[
A_i = r_i - \bar r
\]

The sum of \(A_i\) in the group is 0. Dividing by one group standard deviation keeps the sum 0.

Leave-one-out baseline (NeMo `docs/guides/grpo.md`, Leave-one-out baseline):

\[
A_i = r_i - \frac{1}{n-1}\sum_{j \ne i} r_j = \frac{n r_i - \sum_k r_k}{n-1}
\]

Without a per-row standard deviation, these also sum to 0. NeMo divides each row by the standard deviation of the *other* rows. Those scales differ, so the sum is not 0. We do the same when `baseline="loo"` and `normalize_std=True`. The test states which case sums to zero.

A group whose rewards are identical has no relative signal. Dynamic sampling drops it and counts it (`docs/guides/dapo.md`). The drop is per group, not per row.

**Reinforce++-baseline** subtracts the per-prompt mean, then subtracts the batch mean and divides by the batch standard deviation (`docs/guides/prorlv2.md`). **GDPO** does that group normalisation once per named `reward/` component, then batch-normalises the sum (`docs/guides/environments.md`, `GDPOAdvantageEstimator` in `nemo_rl/algorithms/advantage_estimator.py`).

## PPO value and GAE

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
A_t = \delta_t + \gamma \lambda A_{t+1}
\]

\(\gamma = 1\) and \(\lambda = 0.95\) match NeMo's PPO config. The outcome reward sits on the last completion token. The recursion does not cross the prompt mask. The value loss is squared error, with an optional clip around the value that was used to form the return (`value_loss_fn.cliprange` in `examples/configs/ppo_math_1B.yaml`).

Advantages are normalised over completion tokens when `normalize_advantages` is on. A zero standard deviation skips the division so we do not invent a huge advantage from rounding.

## What the logger stores

Every step can store: loss parts, global gradient norm, gradient norm by layer group, clip-bind fraction by probability bucket, approximate KL, KL to the SFT anchor, entropy, reward mean and std, fraction of zero-variance groups, response length, advantage mean and std. DPO also stores chosen and rejected log-probs, the margin, preference accuracy, and the displacement flag. See `posttrain/metrics.py`.
