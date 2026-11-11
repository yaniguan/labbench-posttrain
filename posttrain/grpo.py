"""Group Relative Policy Optimization, with the switches the spurious-reward paper needs.

For each prompt we draw a group of completions. The advantage of completion i is

    A_i = (r_i - mean(r)) / (std(r) + eps)     # std division can be turned off

There is no value model. The baseline is the other samples of the same prompt.

Per token we minimise

    -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + kl_coef * KL - entropy_coef * H

with ratio = pi_theta / pi_old. KL is the Schulman k3 estimator. The paper's runs
turned KL off (Shao et al., footnote in §4). kl_coef defaults to 0 for that reason.
A lab policy should usually turn it on: the checker does not see every unsafe action.

Switches
    use_clip: off reproduces the "remove the clip term" ablation. Shao et al. Fig. 4:
    random rewards did not give a consistent gain once the clip was gone.
    std_norm: off matches the centred advantage written in their equation (1),
    A = r - E[r], with no division by the group std. Dividing by a tiny std blows up
    the advantage on a prompt where every sample scored almost the same. That is the
    Dr. GRPO criticism of the std term. We still *skip* a group whose std is ~0,
    because every advantage is 0 and the group has nothing to say. Those groups are
    counted in zero_var_frac.
    loss_norm:
        token    — mean over completion tokens. Long answers weigh more.
        sequence — mean of per-sequence means. Each rollout weighs the same, so a
                   long answer's tokens each get a smaller gradient.
        dr_grpo  — sum of token losses divided by (group size * max length), not by
                   the answer length. Length then does not rescale the step. Pair this
                   with std_norm=False to follow the Dr. GRPO idea. We have not run
                   that ablation on a real model.

Zero-variance groups are dropped from the whole loss, including KL, so a tied group
does not quietly become a KL-only update.

On the first update after a rollout the ratio is about 1, so the clip does not fire.
The clip starts to matter when updates_per_rollout > 1 (several steps on the same
samples). The paper also notes that one update per rollout removes the clip effect
even if the clip code is present, because the ratio stays 1.
"""
from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn.functional as F

from posttrain.batching import completion_logprob_mask, generate_texts, pad_right
from posttrain.logging_utils import StepLogger, clip_fraction_by_bucket, clipped_token_mask, grad_norms
from posttrain.lora import apply_lora, trainable_parameters
from posttrain.ppo import entropy_from_logits, schulman_kl, toy_reward
from posttrain.tiny import build_tiny
from posttrain.toy import RL_PROMPTS


def group_advantages(rewards, std_norm: bool = True, eps: float = 1e-6):
    """rewards: [prompts, group]. Returns advantages of the same shape, and a bool mask of zero-variance groups.

    Advantages of a kept group sum to 0. A zero-variance group is set to 0 and marked skipped.
    """
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    std = centered.pow(2).mean(dim=-1, keepdim=True).sqrt()
    zero = (std.squeeze(-1) < eps)
    if std_norm:
        adv = centered / (std + eps)
    else:
        adv = centered
    adv = adv.masked_fill(zero.unsqueeze(-1), 0.0)
    return adv, zero


def clipped_token_objective(logp, old_logp, advantage, epsilon: float, use_clip: bool):
    """Per-token policy term we minimise. advantage is [batch] or [batch, 1]."""
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


def reduce_loss(per_token, mask, mode: str, max_len: int):
    if mode == "token":
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    if mode == "sequence":
        per_seq = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return per_seq.mean()
    if mode == "dr_grpo":
        # Constant denominator. Do not divide by this answer's length.
        denom = per_token.shape[0] * float(max_len)
        return (per_token * mask).sum() / denom
    raise ValueError(mode)


def _pack(rows, tok):
    seqs = []
    for row in rows:
        seq = row["prompt_ids"] + row["completion_ids"]
        if len(seq) < 2:
            seq = seq + [tok.eos_token_id]
            row["completion_ids"] = seq[len(row["prompt_ids"]):]
        seqs.append(seq)
    ids, attn = pad_right(seqs, tok.pad_token_id)
    prompt_lens = [len(r["prompt_ids"]) for r in rows]
    return ids, attn, completion_logprob_mask(prompt_lens, attn)


def _logp_and_entropy(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp, entropy_from_logits(logits)


def train_steps(model, ref, tok, prompts, reward_fn, steps: int, lr: float, logger: StepLogger, *,
                group_size: int = 4, epsilon: float = 0.2, kl_coef: float = 0.0, entropy_coef: float = 0.0,
                sft_coef: float = 0.0, use_clip: bool = True, std_norm: bool = True, loss_norm: str = "token",
                updates_per_rollout: int = 1, max_new_tokens: int = 4, prior_only: bool = False,
                length_coef: float = 0.0, use_lora: bool = False, lora_targets=None, seed: int = 0,
                monitor=None, max_len: int = 64, optimizer=None, metas=None):
    if use_lora:
        model = apply_lora(model, target_modules=lora_targets)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = optimizer or torch.optim.AdamW(trainable_parameters(model), lr=lr)
    model.train()
    logged_steps = 0
    metas = list(metas) if metas is not None else [{} for _ in prompts]
    if len(metas) != len(prompts):
        raise ValueError("metas must align with prompts")
    for step in range(steps):
        torch.manual_seed(seed + step)
        repeated = [p for p in prompts for _ in range(group_size)]
        repeated_meta = [m for m in metas for _ in range(group_size)]
        rows = generate_texts(model, tok, repeated, max_new_tokens=max_new_tokens, do_sample=True)
        rewards = []
        rollouts = []
        for row, row_meta in zip(rows, repeated_meta):
            score, diag = reward_fn(row["prompt"], row["text"], {**row_meta, "step": step})
            if length_coef:
                score = score - length_coef * len(row["completion_ids"])
            rewards.append(float(score))
            rollouts.append({"prompt": row["prompt"], "completion": row["text"], "reward": float(score), **diag})
        reward_tensor = torch.tensor(rewards, dtype=torch.float).view(len(prompts), group_size)
        advantages, zero = group_advantages(reward_tensor, std_norm=std_norm)
        flat_adv = advantages.reshape(-1)
        keep = (~zero).repeat_interleave(group_size).to(dtype=torch.float)
        ids, attn, comp_mask = _pack(rows, tok)
        with torch.no_grad():
            old_logp, _ = _logp_and_entropy(model, ids, attn)
            ref_logp, _ = _logp_and_entropy(ref, ids, attn)
        if prior_only:
            # The base model is the result. Do not step.
            logger.log({
                "step": step,
                "loss": 0.0,
                "grad_norm": 0.0,
                "reward_mean": float(reward_tensor.mean().item()),
                "reward_std": float(reward_tensor.std(unbiased=False).item()),
                "zero_var_frac": float(zero.float().mean().item()),
                "mean_response_tokens": float(comp_mask.sum(-1).mean().item()),
                "prior_only": 1.0,
                "approx_kl": 0.0,
                "entropy": 0.0,
                "clip_frac": 0.0,
                "advantage_mean": float(flat_adv.mean().item()),
            })
            logged_steps += 1
            if monitor is not None:
                monitor.update(step=step, proxy=float(reward_tensor.mean().item()), gold=None, rollouts=rollouts)
                if monitor.stop:
                    break
            continue
        for update in range(updates_per_rollout):
            model.train()
            logp, ent = _logp_and_entropy(model, ids, attn)
            policy_per, ratio = clipped_token_objective(logp, old_logp.detach(), flat_adv.detach(), epsilon, use_clip)
            kl = schulman_kl(logp, ref_logp.detach())
            # Optional SFT term: mean completion NLL, mixed into the RL loss. Off by default.
            nll = -(logp * comp_mask).sum(-1) / comp_mask.sum(-1).clamp(min=1.0)
            per = policy_per + kl_coef * kl - entropy_coef * ent
            per = per + sft_coef * nll.unsqueeze(-1)
            # Drop zero-variance groups entirely.
            group_mask = comp_mask * keep.unsqueeze(-1)
            if float(group_mask.sum().item()) == 0.0:
                # Every group tied. Nothing to differentiate. Log it and move on.
                loss_value = 0.0
                gnorm, groups = 0.0, {}
                buckets = {"all": 0.0}
                policy_value = 0.0
                kl_value = 0.0
                ent_value = 0.0
            else:
                loss = reduce_loss(per, group_mask, loss_norm, max_len=max_len)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm, groups = grad_norms(model)
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                opt.step()
                loss_value = float(loss.detach().item())
                was_clipped = clipped_token_mask(ratio.detach(), flat_adv.detach(), epsilon)
                buckets = clip_fraction_by_bucket(old_logp.detach().exp(), was_clipped, group_mask > 0)
                policy_value = float(reduce_loss(policy_per, group_mask, "token", max_len).detach().item())
                kl_value = float(reduce_loss(kl, group_mask, "token", max_len).detach().item())
                ent_value = float(reduce_loss(ent, group_mask, "token", max_len).detach().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"GRPO loss is not finite: {loss_value}")
            kept_adv = flat_adv[keep.bool()] if int(keep.sum().item()) else flat_adv
            logger.log({
                "step": logged_steps,
                "outer_step": step,
                "update": update,
                "loss": loss_value,
                "policy_loss": policy_value,
                "approx_kl": kl_value,
                "entropy": ent_value,
                "grad_norm": gnorm,
                "grad_norm_groups": groups,
                "clip_frac": buckets.get("all"),
                "clip_frac_by_prob": buckets,
                "reward_mean": float(reward_tensor.mean().item()),
                "reward_std": float(reward_tensor.std(unbiased=False).item()),
                "zero_var_frac": float(zero.float().mean().item()),
                "zero_var_groups": float(zero.sum().item()),
                "advantage_mean": float(kept_adv.mean().item()),
                "advantage_std": float(kept_adv.std(unbiased=False).item()) if kept_adv.numel() > 1 else 0.0,
                "mean_response_tokens": float(comp_mask.sum(-1).mean().item()),
                "use_clip": float(use_clip),
                "std_norm": float(std_norm),
                "prior_only": 0.0,
            })
            logged_steps += 1
        if monitor is not None:
            monitor.update(step=step, proxy=float(reward_tensor.mean().item()), gold=None, rollouts=rollouts)
            if monitor.stop:
                break
    return model, opt


def run_smoke(steps: int = 2, seed: int = 0) -> list[dict]:
    torch.manual_seed(seed)
    model, tok = build_tiny(seed=seed)
    ref = copy.deepcopy(model)
    logger = StepLogger("grpo")
    train_steps(
        model, ref, tok, RL_PROMPTS[:2], toy_reward, steps=steps, lr=1e-3, logger=logger,
        group_size=4, kl_coef=0.01, entropy_coef=0.001, updates_per_rollout=2,
        use_clip=True, std_norm=True, loss_norm="token", max_new_tokens=4, seed=seed,
    )
    # One prior-only step on a fresh copy: weights must not move.
    fresh, tok2 = build_tiny(seed=seed)
    before = [p.detach().clone() for p in fresh.parameters()]
    ref2 = copy.deepcopy(fresh)
    prior_logger = StepLogger("grpo")
    train_steps(
        fresh, ref2, tok2, RL_PROMPTS[:2], toy_reward, steps=1, lr=1e-3, logger=prior_logger,
        group_size=2, max_new_tokens=2, prior_only=True, seed=seed,
    )
    unchanged = all(torch.equal(a, b) for a, b in zip(before, fresh.parameters()))
    print(f"grpo smoke prior_only unchanged={unchanged}", flush=True)
    print(f"grpo smoke done steps={steps}", flush=True)
    if not unchanged:
        raise RuntimeError("prior_only changed weights")
    return logger.rows + prior_logger.rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GRPO with clip, std, and loss-norm switches.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Full GRPO is for Colab. Pass --smoke, or use scripts/colab/run_posttrain.py. Not run.", flush=True)
        return 0
    rows = run_smoke(steps=args.steps)
    if not rows or not all(r["finite"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
