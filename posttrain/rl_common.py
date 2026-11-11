"""Generation and the shared policy-loss step.

PPO and GRPO both sample completions, score them, and take a clipped
policy gradient. The advantage is different. That difference stays in
ppo.py and grpo.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from posttrain.advantages import apply_overlong_filter, apply_stop_properly, group_length_penalty
from posttrain.batching import collate, forward_logprobs
from posttrain.kl import kl_term, warn_if_kl_disabled
from posttrain.losses import clip_binds, entropy_from_logits, policy_surrogate, reduce_token_values
from posttrain.reward_baselines import score_baseline


@dataclass
class Rollout:
    prompt: str
    completion: str
    meta: dict
    group_id: int
    input_ids: list[int]
    completion_mask: list[int]
    truncated: bool
    reward: float = 0.0
    gold: float | None = None
    components: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, metas, group_size, max_new, temperature, seed) -> list[Rollout]:
    """Sample `group_size` completions for each prompt. One prompt at a time.

    Right-padding a batch and then calling generate() lets the model continue
    after pad tokens. A loop is slower and it is correct. Smoke uses two prompts.
    """
    model.eval()
    torch.manual_seed(seed)
    max_prompt = model.config.n_positions - max_new
    if max_prompt < 2:
        raise ValueError("n_positions is too small for the requested completion length")
    rows: list[Rollout] = []
    for p_index, prompt in enumerate(prompts):
        meta = metas[p_index] if metas else {}
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_prompt]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        for g in range(group_size):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_ids = out[0, len(prompt_ids):].tolist()
            new_ids = [t for t in new_ids if t != tokenizer.pad_token_id]
            hit_eos = tokenizer.eos_token_id in new_ids
            if not new_ids:
                new_ids = [tokenizer.eos_token_id]
                hit_eos = True
            truncated = (not hit_eos) and len(new_ids) >= max_new
            # Keep eos in the trained tokens so the model can learn to stop.
            full_ids = prompt_ids + new_ids
            comp_mask = [0] * len(prompt_ids) + [1] * len(new_ids)
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            rows.append(Rollout(
                prompt=prompt,
                completion=text,
                meta=dict(meta),
                group_id=p_index,
                input_ids=full_ids,
                completion_mask=comp_mask,
                truncated=truncated,
            ))
    return rows


def score_rollouts(rows: list[Rollout], reward_fn, mode: str, seed: int) -> None:
    """Fill reward, gold, and components. gold is never passed to the optimiser."""
    by_group: dict[int, list[Rollout]] = {}
    for row in rows:
        by_group.setdefault(row.group_id, []).append(row)
    for row in rows:
        gt = reward_fn(row.prompt, row.completion, row.meta)
        row.gold = float(gt.value)
        group_text = [r.completion for r in by_group[row.group_id]]
        scored = score_baseline(
            mode,
            prompt=row.prompt,
            completion=row.completion,
            meta=row.meta,
            group_completions=group_text,
            ground_truth=gt.value,
            seed=seed,
            index=row.group_id,
        )
        row.reward = float(scored.value)
        if mode == "ground_truth":
            row.components = dict(gt.components) or {"reward/score": float(gt.value)}
        else:
            row.components = dict(scored.components)
        row.diagnostics = dict(scored.diagnostics)
        row.diagnostics["parsed_env"] = bool(gt.diagnostics.get("parsed", False))


def shape_rewards(rows: list[Rollout], stop_coef: float | None, length_coef: float) -> int:
    """Stop-properly penalty, then the binary group length penalty.

    Returns how many groups the length penalty skipped because the reward
    was not binary. BioProBench rewards are graded, so a non-zero length
    coefficient skips them and the reward is unchanged. That skip is the
    point of the NeMo rule.
    """
    rewards = torch.tensor([r.reward for r in rows], dtype=torch.float32)
    truncated = torch.tensor([r.truncated for r in rows])
    rewards = apply_stop_properly(rewards, truncated, stop_coef)
    lengths = torch.tensor([sum(r.completion_mask) for r in rows], dtype=torch.float32)
    groups = torch.tensor([r.group_id for r in rows], dtype=torch.long)
    rewards, skipped = group_length_penalty(rewards, lengths, groups, length_coef)
    for row, value in zip(rows, rewards.tolist()):
        row.reward = float(value)
    return skipped


def filter_zero_variance(rows: list[Rollout], enabled: bool) -> tuple[list[Rollout], int, int]:
    """Drop groups whose rewards are all equal. Count groups, not rows."""
    groups: dict[int, list[Rollout]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    n_groups = len(groups)
    if not enabled:
        return rows, 0, n_groups
    kept = []
    dropped = 0
    for members in groups.values():
        values = [m.reward for m in members]
        if max(values) == min(values):
            dropped += 1
        else:
            kept.extend(members)
    return kept, dropped, n_groups


def collate_rollouts(tokenizer, rows: list[Rollout]):
    pairs = [(r.input_ids, r.completion_mask) for r in rows]
    return collate(tokenizer, pairs)


def policy_loss(
    logp,
    logp_old,
    logp_ref,
    advantage,
    mask,
    *,
    kl_coef: float,
    kl_estimator: str,
    kl_in_loss: bool,
    eps_low: float,
    eps_high: float,
    clip: bool,
    cispo: bool,
    token_level: bool,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    dual_clip_c: float | None = None,
    on_policy_kl: bool = False,
    is_weight: torch.Tensor | None = None,
):
    """Clipped (or CISPO) surrogate, plus the KL term when it is a loss.

    `kl_in_loss` False means the caller already folded a detached KL into the
    advantage (PPO style). Adding it again would punish twice.
    """
    per = policy_surrogate(
        logp, logp_old, advantage,
        eps_low=eps_low, eps_high=eps_high, clip=clip, cispo=cispo, dual_clip_c=dual_clip_c,
    )
    if is_weight is not None:
        # Detached actor-vs-sampler weight. NeMo `use_importance_sampling_correction`.
        # The simple clamp is what we implement. ICE-POP and adaptive-ppo-clip are not.
        per = per * is_weight.detach()
    pg = reduce_token_values(per, mask, token_level=token_level)
    parts = {"policy": float(pg.detach())}
    loss = pg
    approx = None
    if logp_ref is not None and mask.sum() > 0:
        raw_kl = kl_term(logp, logp_ref, kl_estimator)
        if on_policy_kl:
            # NeMo docs/guides/grpo.md, On-Policy KL Approximation:
            # multiply the k3-style term by the probability ratio so the
            # expectation is on-policy while the samples came from pi_old.
            ratio = torch.exp(logp.detach() - logp_old.detach())
            raw_kl = raw_kl * ratio
        approx = float(reduce_token_values(raw_kl.detach(), mask, token_level=True))
        if kl_in_loss and kl_coef:
            kl_loss = reduce_token_values(raw_kl, mask, token_level=token_level)
            loss = loss + kl_coef * kl_loss
            parts["kl"] = float(kl_loss.detach())
    if entropy is not None and entropy_coef:
        # Maximise entropy: subtract it from the minimised loss.
        loss = loss - entropy_coef * entropy
        parts["entropy"] = float(entropy.detach())
    parts["loss"] = float(loss.detach())
    return loss, parts, approx


def behavior_token_prob(logp_old: torch.Tensor) -> torch.Tensor:
    return logp_old.detach().exp()


def bind_mask(logp, logp_old, advantage, eps_low, eps_high, clip: bool):
    if not clip:
        return torch.zeros_like(logp, dtype=torch.bool)
    ratio = torch.exp(logp.detach() - logp_old.detach())
    return clip_binds(ratio, advantage.detach(), eps_low, eps_high)


def masked_entropy(logits, completion_mask) -> torch.Tensor:
    ent = entropy_from_logits(logits)[:, :-1]
    from posttrain.losses import completion_mask_shifted
    mask = completion_mask_shifted(completion_mask)
    return reduce_token_values(ent, mask, token_level=True)


def apply_loss_mask(mask, truncated: torch.Tensor, overlong: bool):
    return apply_overlong_filter(mask, truncated, overlong)


def require_kl(method: str, kl_coef: float) -> None:
    warn_if_kl_disabled(kl_coef, method)
