"""TROPD: Trust-Region Multi-Teacher On-Policy Distillation (`arXiv:2601.02780`).

Matches `OPDAdvantageEstimator` in `yaniguan/RL` (`nemo_rl/algorithms/advantage_estimator.py`):
- Student generates trajectories online (`generate_rollouts`).
- One or more frozen teachers score token logprobs on the student's trajectories.
- Proximal trust-region mixture in log space:
    log pi_prox = log(alpha * pi_teacher + (1 - alpha) * pi_student)
    A_TROPD,t   = sg[log pi_prox - log pi_student] >= log(1 - alpha)
- Fuses token-level distillation advantages with group-normalized verifiable task rewards
  and updates the student via REINFORCE (`force_on_policy_ratio=True`) + ICE-POP gating.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from posttrain.advantages import grpo_advantages
from posttrain.batching import clone_frozen, frozen_logprobs, model_device, snapshot_state, states_equal
from posttrain.kl import DEFAULT_KL_COEF, DEFAULT_KL_ESTIMATOR
from posttrain.losses import completion_mask_shifted, compute_is_weights, token_logprobs
from posttrain.metrics import StepLogger, StepRecord, grad_norm, grad_norms_by_group
from posttrain.rl_common import collate_rollouts, generate_rollouts, masked_entropy, policy_loss, score_rollouts


def tropd_token_advantages(
    student_lp: torch.Tensor,
    teacher_lps: list[torch.Tensor] | torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: list[float] | None = None,
    alpha: float = 0.5,
    subtract_baseline: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute bounded token-level TROPD advantages `sg[log pi_prox - log pi_student]`."""
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError(f"proximal_teacher_alpha must be in (0, 1], got {alpha}")
    if isinstance(teacher_lps, torch.Tensor):
        t_list = [teacher_lps]
    else:
        t_list = list(teacher_lps)
    if not t_list:
        raise ValueError("tropd_token_advantages requires at least one teacher logprob tensor")

    if len(t_list) == 1:
        teacher_mix_lp = t_list[0]
    else:
        w = weights or [1.0 / len(t_list)] * len(t_list)
        w_sum = sum(w)
        log_terms = [tlp + math.log(max(wk / w_sum, 1e-12)) for tlp, wk in zip(t_list, w)]
        teacher_mix_lp = torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)

    raw_gap = (teacher_mix_lp - student_lp).detach()
    if float(alpha) == 1.0:
        distill_adv = raw_gap
    else:
        prox_lp = torch.logaddexp(
            teacher_mix_lp + math.log(float(alpha)),
            student_lp + math.log1p(-float(alpha)),
        )
        distill_adv = (prox_lp - student_lp).detach()

    support = ~torch.isneginf(student_lp)
    valid_mask = mask * support.to(mask.dtype)
    raw_gap = torch.where(support, raw_gap, torch.zeros_like(raw_gap))
    distill_adv = torch.where(support, distill_adv, torch.zeros_like(distill_adv))

    if subtract_baseline:
        valid_vals = torch.masked_select(distill_adv, valid_mask.bool())
        if valid_vals.numel() > 0:
            distill_adv = distill_adv - valid_vals.mean()

    out = distill_adv * valid_mask
    valid_gap = torch.masked_select(raw_gap, valid_mask.bool())
    metrics = {
        "tropd_gap_mean": float(valid_gap.mean()) if valid_gap.numel() > 0 else 0.0,
        "tropd_adv_min": float(torch.masked_select(out, valid_mask.bool()).min()) if valid_mask.bool().any() else 0.0,
    }
    return out, metrics


def train_tropd(
    prompts: list[dict[str, Any]],
    reward_fn,
    *,
    teachers: list[Any] | None = None,
    teacher_weights: list[float] | None = None,
    proximal_alpha: float = 0.5,
    distill_weight: float = 0.6,
    task_reward_weight: float = 0.4,
    kl_coef: float = 0.005,
    kl_estimator: str = DEFAULT_KL_ESTIMATOR,
    use_is_correction: bool = True,
    is_mode: str = "icepop",
    is_clip_min: float = 0.5,
    is_clip: float = 2.0,
    group_size: int = 4,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    steps: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    model=None,
    tokenizer=None,
    reference=None,
    logger: StepLogger | None = None,
) -> dict[str, Any]:
    """Run online TROPD student training with multi-teacher logprobs + verifiable task rewards."""
    from posttrain.tiny import build_tiny_lm

    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    dev = model_device(model)
    if reference is None:
        reference = clone_frozen(model)
    if not teachers:
        t_default, _ = build_tiny_lm(seed=seed + 7)
        t_default = t_default.to(dev).eval()
        for p in t_default.parameters():
            p.requires_grad_(False)
        teachers = [t_default]

    ref_before = snapshot_state(reference)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    logger = logger or StepLogger()
    losses: list[float] = []
    texts = [p["prompt"] for p in prompts]

    for step in range(steps):
        rows = generate_rollouts(model, tokenizer, texts, prompts, group_size, max_new_tokens, temperature, seed + step)
        score_rollouts(rows, reward_fn, "ground_truth", seed + step)
        batch = collate_rollouts(tokenizer, rows, device=dev)
        shift_mask = completion_mask_shifted(batch["completion_mask"])

        model.eval()
        with torch.no_grad():
            old_out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            old_lp = token_logprobs(old_out.logits, batch["input_ids"]).detach()
            ref_lp, _ = frozen_logprobs(reference, batch)
            t_lps = [frozen_logprobs(t_mod, batch)[0] for t_mod in teachers]

        tropd_adv, tropd_metrics = tropd_token_advantages(
            old_lp, t_lps, shift_mask, weights=teacher_weights, alpha=proximal_alpha, subtract_baseline=True
        )
        rewards_t = torch.tensor([r.reward for r in rows], dtype=torch.float32, device=dev)
        groups_t = torch.tensor([r.group_id for r in rows], dtype=torch.long, device=dev)
        task_adv = grpo_advantages(rewards_t, groups_t, baseline="mean", normalize_std=True).unsqueeze(1).expand_as(old_lp)
        combined_adv = (distill_weight * tropd_adv + task_reward_weight * task_adv) * shift_mask

        is_weight = None
        is_metrics: dict[str, float] = {}
        if use_is_correction:
            gen_lp = batch.get("gen_logprobs", old_lp)
            is_weight, is_metrics = compute_is_weights(old_lp, gen_lp, shift_mask, mode=is_mode, clip_min=is_clip_min, clip_max=is_clip)

        model.train()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logp = token_logprobs(out.logits, batch["input_ids"])
        ent = masked_entropy(out.logits, batch["completion_mask"])
        loss, parts, approx = policy_loss(
            logp,
            old_lp,
            ref_lp,
            combined_adv,
            shift_mask,
            kl_coef=kl_coef,
            kl_estimator=kl_estimator,
            kl_in_loss=True,
            eps_low=0.2,
            eps_high=0.2,
            clip=False,
            cispo=False,
            token_level=True,
            entropy=ent,
            force_on_policy_ratio=True,
            is_weight=is_weight,
        )
        parts.update(tropd_metrics)
        parts.update(is_metrics)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = grad_norm(model.parameters())
        groups_g = grad_norms_by_group(model)
        opt.step()
        losses.append(float(loss.detach()))

        logger.add(StepRecord(
            step=step,
            method="tropd",
            loss=losses[-1],
            loss_parts=parts,
            grad_norm=g,
            grad_norm_by_group=groups_g,
            approx_kl=approx,
            reward_mean=float(rewards_t.mean()),
            kl_coef=kl_coef,
        ))

    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "model": model,
        "tokenizer": tokenizer,
        "reference_unchanged": states_equal(ref_before, snapshot_state(reference)),
        "logger": logger,
        "status": "ran",
    }
