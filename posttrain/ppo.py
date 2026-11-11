"""PPO for a single-turn verifiable reward.

Actor: the language model. Critic: a linear value head on the hidden state.
There is no value head inside GRPO. PPO uses one so the advantage is a residual
around a learned baseline, which is the point of paying for the extra model.

What we minimise, per completion token:

    ratio = pi_theta(token) / pi_old(token)
    policy = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    value  = 0.5 * max( (V - R)^2, (V_clip - R)^2 )     # the max is optional
    loss   = policy + value_coef * value + kl_coef * KL - entropy_coef * entropy

A is a GAE advantage. The reward sits on the last completion token. Earlier tokens
get credit only through the backup. KL is the Schulman k3 estimator against a frozen
reference, added in the loss (not mixed into the reward), so the term is visible.

Clipping
    The gradient w.r.t. the ratio is zero when A >= 0 and ratio > 1+eps, and when
    A < 0 and ratio < 1-eps. See clipped_policy_loss. Shao et al. §4 show that this
    zero region is not symmetric in probability: a token that is already very likely
    cannot reach the upper clip. That discussion is about GRPO, and the same clip
    term is here. Turn it off with use_clip=False.

Advantage normalisation
    After GAE, advantages are centred and divided by their std over the valid tokens
    in the batch. One wild return then cannot set the step size for every token.
"""
from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from posttrain.batching import completion_logprob_mask, generate_texts, pad_right
from posttrain.logging_utils import StepLogger, clip_fraction_by_bucket, clipped_token_mask, grad_norms
from posttrain.lora import apply_lora, trainable_parameters
from posttrain.tiny import build_tiny
from posttrain.toy import RL_PROMPTS


class ValueHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, hidden_states):
        return self.fc(hidden_states).squeeze(-1)


def gae(rewards, values, mask, gamma: float = 1.0, lam: float = 0.95):
    """Generalised advantage estimation. rewards, values, mask are [batch, time].

    mask is 1 on real completion tokens. The backup stops at the last real token
    because the next mask bit is 0.
    """
    batch, time = rewards.shape
    adv = torch.zeros_like(rewards)
    next_value = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    last = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(time)):
        if t + 1 < time:
            next_valid = mask[:, t + 1]
        else:
            next_valid = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        delta = rewards[:, t] + gamma * next_value * next_valid - values[:, t]
        last = (delta + gamma * lam * next_valid * last) * mask[:, t]
        adv[:, t] = last
        next_value = values[:, t]
    return adv, adv + values


def normalize_advantages(adv, mask, eps: float = 1e-8):
    count = mask.sum().clamp(min=1.0)
    mean = (adv * mask).sum() / count
    var = ((adv - mean).pow(2) * mask).sum() / count
    return (adv - mean) / (var.sqrt() + eps)


def clipped_policy_loss(logp, old_logp, advantage, epsilon: float, use_clip: bool):
    """Negative of the PPO clipped surrogate, per token. advantage broadcasts over time.

    We minimise this. The paper maximises the min(), so the sign here is flipped and
    the zero-gradient region is the same.
    """
    ratio = torch.exp(logp - old_logp)
    adv = advantage
    while adv.ndim < ratio.ndim:
        adv = adv.unsqueeze(-1)
    unclipped = ratio * adv
    if use_clip:
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
        surrogate = torch.min(unclipped, clipped)
    else:
        surrogate = unclipped
    return -surrogate, ratio


def value_loss(values, old_values, returns, epsilon: float, use_clip: bool):
    """Squared error. With clipping, take the worse of the clipped and unclipped errors.

    That stops the critic from chasing a return by jumping far from the value it had
    at rollout time. use_clip=False is ordinary MSE.
    """
    unclipped = (values - returns).pow(2)
    if not use_clip:
        return 0.5 * unclipped
    value_clipped = old_values + torch.clamp(values - old_values, -epsilon, epsilon)
    clipped = (value_clipped - returns).pow(2)
    return 0.5 * torch.max(unclipped, clipped)


def schulman_kl(logp, ref_logp):
    """k3 estimator. Non-negative, and 0 when the two log-probs match.

    d/d(log pi) = 1 - pi_ref / pi. Near the reference this is about log pi - log pi_ref,
    so the penalty pushes the token back toward the reference.
    """
    return torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0


def entropy_from_logits(logits):
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def _reduce(per_token, mask):
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_std(values, mask) -> float:
    mean = _reduce(values, mask)
    var = _reduce((values - mean) ** 2, mask)
    return float(var.detach().sqrt().item())


def toy_reward(prompt: str, completion: str, meta: dict | None = None) -> tuple[float, dict]:
    """A reward that depends on the bytes, so two samples of one prompt usually differ.

    It is not a lab reward. Smoke uses it so the advantage is not identically zero.
    """
    raw = sum(completion.encode("utf-8", errors="ignore")) % 7
    return raw / 6.0, {"reward_name": "toy", "parsed": True}


def _hidden_size(model) -> int:
    return int(getattr(model.config, "n_embd", None) or model.config.hidden_size)


def _rollout_batch(rows, tok):
    seqs = [r["prompt_ids"] + r["completion_ids"] for r in rows]
    if any(len(s) < 2 for s in seqs):
        # A one-token sequence has no next-token log-prob. Repeat the last id so the step still runs.
        seqs = [s if len(s) >= 2 else s + [tok.eos_token_id] for s in seqs]
        for row, seq in zip(rows, seqs):
            if len(row["prompt_ids"]) + len(row["completion_ids"]) < 2:
                row["completion_ids"] = seq[len(row["prompt_ids"]):]
    ids, mask = pad_right(seqs, tok.pad_token_id)
    prompt_lens = [len(r["prompt_ids"]) for r in rows]
    comp_mask = completion_logprob_mask(prompt_lens, mask)
    return ids, mask, comp_mask


def _forward_logp(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    hidden = out.hidden_states[-1][:, :-1, :]
    ent = entropy_from_logits(logits)
    return logp, hidden, ent


def train_steps(model, ref, tok, prompts, reward_fn, steps: int, lr: float, logger: StepLogger, *,
                epsilon: float = 0.2, kl_coef: float = 0.0, entropy_coef: float = 0.0, value_coef: float = 0.5,
                use_clip: bool = True, gamma: float = 1.0, lam: float = 0.95, max_new_tokens: int = 4,
                use_lora: bool = False, lora_targets=None, seed: int = 0):
    if use_lora:
        model = apply_lora(model, target_modules=lora_targets)
    value = ValueHead(_hidden_size(model))
    model.train()
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(trainable_parameters(model) + list(value.parameters()), lr=lr)
    for step in range(steps):
        torch.manual_seed(seed + step)
        rows = generate_texts(model, tok, prompts, max_new_tokens=max_new_tokens, do_sample=True)
        rewards = []
        for row in rows:
            score, _ = reward_fn(row["prompt"], row["text"], {"step": step})
            rewards.append(float(score))
        ids, attn, comp_mask = _rollout_batch(rows, tok)
        with torch.no_grad():
            old_logp, old_hidden, _ = _forward_logp(model, ids, attn)
            ref_logp, _, _ = _forward_logp(ref, ids, attn)
            old_values = value(old_hidden)
        model.train()
        logp, hidden, ent = _forward_logp(model, ids, attn)
        values = value(hidden)
        token_rewards = torch.zeros_like(comp_mask)
        lengths = comp_mask.sum(dim=-1).to(dtype=torch.long) 
        for i, score in enumerate(rewards):
            last = int(lengths[i].item()) - 1
            if last >= 0:
                token_rewards[i, last] = score
        adv, returns = gae(token_rewards, old_values.detach(), comp_mask, gamma=gamma, lam=lam)
        adv = normalize_advantages(adv, comp_mask)
        policy_per, ratio = clipped_policy_loss(logp, old_logp.detach(), adv.detach(), epsilon, use_clip)
        v_per = value_loss(values, old_values.detach(), returns.detach(), epsilon, use_clip)
        kl = schulman_kl(logp, ref_logp.detach())
        per = policy_per + value_coef * v_per + kl_coef * kl - entropy_coef * ent
        loss = _reduce(per, comp_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm, groups = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model) + list(value.parameters()), 1.0)
        opt.step()
        was_clipped = clipped_token_mask(ratio.detach(), adv.detach(), epsilon)
        old_prob = old_logp.detach().exp()
        buckets = clip_fraction_by_bucket(old_prob, was_clipped, comp_mask > 0)
        value_loss_mean = float(_reduce(v_per, comp_mask).detach().item())
        number = float(loss.detach().item())
        if not math.isfinite(number):
            raise RuntimeError(f"PPO loss is not finite: {number}")
        logger.log({
            "step": step,
            "loss": number,
            "policy_loss": float(_reduce(policy_per, comp_mask).detach().item()),
            "value_loss": value_loss_mean,
            "approx_kl": float(_reduce(kl, comp_mask).detach().item()),
            "entropy": float(_reduce(ent, comp_mask).detach().item()),
            "grad_norm": gnorm,
            "grad_norm_groups": groups,
            "clip_frac": buckets["all"],
            "clip_frac_by_prob": buckets,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if len(rewards) > 1 else 0.0,
            "advantage_mean": float(_reduce(adv, comp_mask).detach().item()),
            "advantage_std": _masked_std(adv, comp_mask),
            "mean_response_tokens": float(comp_mask.sum(dim=-1).mean().item()),
            "use_clip": float(use_clip),
        })
    return model, value, opt


def run_smoke(steps: int = 2, seed: int = 0) -> list[dict]:
    torch.manual_seed(seed)
    model, tok = build_tiny(seed=seed)
    ref = copy.deepcopy(model)
    logger = StepLogger("ppo")
    train_steps(
        model, ref, tok, RL_PROMPTS[:2], toy_reward, steps=steps, lr=1e-3, logger=logger,
        kl_coef=0.01, entropy_coef=0.001, max_new_tokens=4, seed=seed,
    )
    print(f"ppo smoke done steps={steps}", flush=True)
    return logger.rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PPO with GAE and a value head.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Full PPO is for Colab. Pass --smoke, or use scripts/colab/run_posttrain.py. Not run.", flush=True)
        return 0
    rows = run_smoke(steps=args.steps)
    if not rows or not all(r["finite"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
