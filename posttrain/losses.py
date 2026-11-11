"""Scalar losses. The gradient is the point of this file.

Each function is the quantity we minimise. Docs in
docs/posttrain/losses_and_gradients.md write the same formulas.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reduce_token_values(values: torch.Tensor, mask: torch.Tensor, token_level: bool) -> torch.Tensor:
    """Average per-token values.

    token_level True: every token counts the same. A long completion dominates
    the step. NeMo calls this `token_level_loss: true` (docs/guides/grpo.md,
    Loss Normalization). DAPO argues for it on long chain-of-thought
    (docs/guides/dapo.md, Token-Level Policy Gradient Loss).

    token_level False: average inside each sequence, then average sequences.
    A short completion and a long one count the same. The original GRPO
    paper uses this. NeMo's write-up of the tradeoff is the right reading:
    if lengths are similar the two settings barely differ.
    """
    mask = mask.to(values.dtype)
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} vs mask {tuple(mask.shape)}")
    if token_level:
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    seq = (values * mask).sum(dim=-1) / lengths
    valid = (mask.sum(dim=-1) > 0).to(values.dtype)
    return (seq * valid).sum() / valid.sum().clamp(min=1.0)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Log-prob of input_ids[t+1] under logits[t]. Shape [B, T-1]."""
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, target).squeeze(-1)


def completion_mask_shifted(completion_mask: torch.Tensor) -> torch.Tensor:
    """Mask for the tokens that token_logprobs predicts. 1 = completion target."""
    return completion_mask[:, 1:].to(dtype=torch.float32)


def sft_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    token_level: bool = True,
) -> torch.Tensor:
    """Cross-entropy on completion tokens only.

    Prompt positions are masked because the model did not write them. Training
    on the prompt teaches it to copy the question, and a long protocol would
    swamp the completion. The logits that predict a masked token get no
    gradient from this loss. The last prompt position still has a gradient:
    it is what predicts the first completion token.
    """
    nll = -token_logprobs(logits, input_ids)
    mask = completion_mask_shifted(completion_mask)
    return reduce_token_values(nll, mask, token_level=token_level)


def dpo_loss(
    logp_chosen: torch.Tensor,
    logp_rejected: torch.Tensor,
    logp_ref_chosen: torch.Tensor,
    logp_ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Bradley-Terry DPO loss, one number per pair, then the mean.

    loss = -log sigmoid(beta * ((logp_pi(c) - logp_ref(c)) - (logp_pi(r) - logp_ref(r))))

    beta is the KL weight in the closed form. beta 0 makes the loss the
    constant -log(0.5) and the gradient 0. Callers must warn before that.
    """
    # Reference log-probs are constants. Detach so a shared tensor cannot leak.
    margin = (logp_chosen - logp_ref_chosen.detach()) - (logp_rejected - logp_ref_rejected.detach())
    scaled = beta * margin
    loss = -F.logsigmoid(scaled)
    with torch.no_grad():
        stats = {
            "margin": float(margin.mean()),
            "accuracy": float((margin > 0).float().mean()),
            "chosen_logprob": float(logp_chosen.mean()),
            "rejected_logprob": float(logp_rejected.mean()),
            "reward_margin": float(scaled.mean()),
        }
    return loss.mean(), stats


def likelihood_displacement(chosen_before, chosen_after, rejected_before, rejected_after) -> bool:
    """True when both the chosen and the rejected log-prob fell.

    DPO can raise the margin by lowering both log-probs, with the rejected
    one falling faster. The preferred answer got less likely. That is
    likelihood displacement. A falling margin is a different failure.
    """
    return float(chosen_after) < float(chosen_before) and float(rejected_after) < float(rejected_before)


def policy_surrogate(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    clip: bool = True,
    cispo: bool = False,
    dual_clip_c: float | None = None,
) -> torch.Tensor:
    """Per-token policy loss (the quantity we minimise).

    PPO/GRPO maximise min(ratio * A, clip(ratio, 1-eps_low, 1+eps_high) * A).
    We minimise the negative of that.

    The clip binds, and the gradient w.r.t. logp is exactly 0, when:
      * A > 0 and ratio > 1 + eps_high  (cannot push a high token higher)
      * A < 0 and ratio < 1 - eps_low    (cannot push a low token lower)

    Shao et al., Section 4, equation (2): inside the clip the random-reward
    gradient is 0; outside it, the sign reinforces tokens that are already
    likely and suppresses tokens that are not. High-probability tokens often
    cannot reach the upper clip, because ratio would need pi > 1.

    CISPO (NeMo `docs/guides/cispo.md`) detaches the clipped ratio and
    multiplies log pi. The gradient is -A * sg(clip(ratio)), which is not
    zero just because the ratio left the range.

    `eps_low != eps_high` is the asymmetric clip (DAPO clip-higher,
    `docs/guides/dapo.md`; ProRLv2 `docs/guides/prorlv2.md`).
    """
    logp_old = logp_old.detach()
    advantage = advantage.detach()
    ratio = torch.exp(logp - logp_old)
    if cispo:
        if not clip:
            weight = ratio.detach()
        else:
            weight = ratio.detach().clamp(1.0 - eps_low, 1.0 + eps_high)
        return -(weight * advantage * logp)

    if not clip:
        obj = ratio * advantage
    else:
        low = 1.0 - eps_low
        high = 1.0 + eps_high
        upper_bind = (ratio > high) & (advantage > 0)
        lower_bind = (ratio < low) & (advantage < 0)
        bind = upper_bind | lower_bind
        clipped_ratio = ratio.detach().clamp(low, high)
        used = torch.where(bind, clipped_ratio, ratio)
        obj = used * advantage
        if dual_clip_c is not None:
            # Ye et al. dual clip, as in NeMo loss_fn.ratio_clip_c.
            # When A < 0, do not let ratio * A run off to -infinity.
            # Maximisation form: max(min(ratio*A, clip*A), c*A) with c > 1.
            floor = dual_clip_c * advantage
            obj = torch.where(advantage < 0, torch.maximum(obj, floor), obj)
    return -obj


def clip_binds(ratio: torch.Tensor, advantage: torch.Tensor, eps_low: float, eps_high: float) -> torch.Tensor:
    """True where the PPO/GRPO clip makes the policy gradient exactly 0."""
    upper = (ratio > 1.0 + eps_high) & (advantage > 0)
    lower = (ratio < 1.0 - eps_low) & (advantage < 0)
    return upper | lower


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    values_old: torch.Tensor | None = None,
    clip_eps: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared error on the value head. Optional PPO value clip.

    The clip keeps the value from jumping further than `clip_eps` away from
    the value that was used to compute the return. NeMo's PPO config exposes
    this as `value_loss_fn.cliprange` (`examples/configs/ppo_math_1B.yaml`).
    """
    if mask is None:
        mask = torch.ones_like(values)
    if clip_eps is None or values_old is None:
        err = (values - returns.detach()) ** 2
    else:
        v_clip = values_old.detach() + (values - values_old.detach()).clamp(-clip_eps, clip_eps)
        err = torch.maximum((values - returns.detach()) ** 2, (v_clip - returns.detach()) ** 2)
    return 0.5 * reduce_token_values(err, mask, token_level=True)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy at each position. Shape [B, T]."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def prob_buckets(token_prob: torch.Tensor) -> dict[str, torch.Tensor]:
    """Masks for the clipping-bias plot. Buckets are on the behavior probability."""
    edges = (("p<0.05", 0.0, 0.05), ("p<0.2", 0.05, 0.2), ("p<0.5", 0.2, 0.5), ("p>=0.5", 0.5, 1.01))
    return {name: (token_prob >= lo) & (token_prob < hi) for name, lo, hi in edges}
