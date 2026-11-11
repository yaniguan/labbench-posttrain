"""Advantage estimators used by PPO and GRPO.

GRPO and GDPO follow `nemo_rl/algorithms/advantage_estimator.py` in the
yaniguan/RL repo (a fork of NeMo RL). The formulas below are the ones we
implement. We do not import that code.
"""
from __future__ import annotations

import re

import torch

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def _group_slices(group_ids: torch.Tensor):
    ids = group_ids.detach().cpu().tolist()
    order: dict[int, list[int]] = {}
    for i, g in enumerate(ids):
        order.setdefault(int(g), []).append(i)
    return order


def grpo_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    baseline: str = "mean",
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group-relative advantage, one number per completion.

    baseline "mean": subtract the group mean. With or without a single group
    std, the advantages in a group sum to 0.

    baseline "loo": leave-one-out mean, as in NeMo
    `use_leave_one_out_baseline` (`docs/guides/grpo.md`, Leave-one-out baseline).
    Without std normalisation the advantages still sum to 0, because
    A_i = (n * r_i - S) / (n - 1) and the r_i terms cancel.
    With std normalisation NeMo divides each row by the std of the *other*
    rows. Those stds differ, so the sum is not 0. We do the same when
    normalize_std is on. A group of size 1 gets advantage 0: there is no
    baseline to compare against.

    If the comparison set has no spread, we do not divide. Dividing by a
    rounding-error std would invent a huge advantage. NeMo skips that case.
    """
    if baseline not in ("mean", "loo"):
        raise ValueError("baseline must be 'mean' or 'loo'")
    adv = torch.zeros_like(rewards)
    for idxs in _group_slices(group_ids).values():
        ix = torch.tensor(idxs, device=rewards.device)
        r = rewards.index_select(0, ix)
        n = r.numel()
        if n == 1:
            continue
        if baseline == "mean":
            centered = r - r.mean()
            if normalize_std:
                std = r.std(unbiased=False)
                if float(std) > 0:
                    centered = centered / (std + eps)
            adv.index_copy_(0, ix, centered)
        else:
            total = r.sum()
            for j in range(n):
                others = (total - r[j]) / (n - 1)
                a = r[j] - others
                if normalize_std:
                    rest = torch.cat([r[:j], r[j + 1:]])
                    if float(rest.max() - rest.min()) > 0:
                        a = a / (rest.std(unbiased=False) + eps)
                adv[ix[j]] = a
    return adv


def zero_variance_mask(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """True on completions whose whole group has identical rewards.

    DAPO dynamic sampling drops these groups (`docs/guides/dapo.md`,
    Dynamic Sampling). The test is min == max over the group, not the
    leave-one-out std. A group is kept or dropped together.
    """
    drop = torch.zeros(rewards.shape[0], dtype=torch.bool, device=rewards.device)
    for idxs in _group_slices(group_ids).values():
        ix = torch.tensor(idxs, device=rewards.device)
        r = rewards.index_select(0, ix)
        if float(r.max() - r.min()) == 0.0:
            drop[ix] = True
    return drop


def clip_advantages(advantages: torch.Tensor, low: float | None, high: float | None) -> torch.Tensor:
    """Clamp after normalisation. NeMo `advantage_clip_low/high`.

    A tiny reward std makes a normalised advantage huge. The clamp stops one
    odd completion from dominating the step. None means that side is open.
    """
    if low is None and high is None:
        return advantages
    lo = -1e9 if low is None else low
    hi = 1e9 if high is None else high
    return advantages.clamp(lo, hi)


def reinforce_pp_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    minus_baseline: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reinforce++-Baseline: local baseline, then a global normalisation.

    NeMo `docs/guides/prorlv2.md` (Decoupled local/global advantage
    normalization) and `ReinforcePlusPlusAdvantageEstimator`.
    minus_baseline True subtracts the per-prompt mean first.
    Then we subtract the batch mean and divide by the batch std.
    minus_baseline False is plain Reinforce++ (global norm only).
    """
    if minus_baseline:
        local = grpo_advantages(rewards, group_ids, baseline="mean", normalize_std=False)
    else:
        local = rewards.clone()
    std = local.std(unbiased=False)
    if float(std) == 0.0:
        return local - local.mean()
    return (local - local.mean()) / (std + eps)


def gdpo_advantages(
    components: dict[str, torch.Tensor],
    group_ids: torch.Tensor,
    weights: dict[str, float] | None = None,
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """One GRPO advantage per reward component, then a weighted sum.

    NeMo `docs/guides/environments.md` (Multi-reward support) and
    `GDPOAdvantageEstimator`: normalise each named component inside the
    group so a 0/1 format bit cannot drown a small calibration term, then
    batch-normalise the sum. Weights default to 1. Keys should use the
    `reward/` prefix, matching that guide.
    """
    if len(components) < 1:
        raise ValueError("GDPO needs at least one reward component")
    weights = weights or {}
    total = None
    for name, rewards in components.items():
        part = grpo_advantages(rewards, group_ids, baseline="mean", normalize_std=normalize_std)
        w = float(weights.get(name, 1.0))
        total = part * w if total is None else total + part * w
    assert total is not None
    std = total.std(unbiased=False)
    if float(std) == 0.0:
        return total - total.mean()
    return (total - total.mean()) / (std + eps)


def vapo_length_adaptive_lambda(mask: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """VAPO length-adaptive GAE lambda_i = max(0, 1 - 1 / (alpha * L_i)) (arXiv:2504.05118)."""
    lengths = mask.to(torch.float32).sum(dim=-1).clamp(min=1.0)
    lam = 1.0 - 1.0 / (max(float(alpha), 1e-6) * lengths)
    return lam.clamp(min=0.0, max=1.0)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float | torch.Tensor = 0.95,
) -> torch.Tensor:
    """Generalized Advantage Estimation (Schulman et al. 2016).

    `rewards` and `values` are per token, shape [B, T]. `mask` is 1 on tokens
    that belong to the completion. The recursion does not cross a masked
    token, so the prompt does not pick up the outcome reward.
    gamma=1 is the NeMo PPO default (`gae_gamma: 1` in ppo_math_1B.yaml):
    a short lab answer should not be discounted just for being later.
    """
    b, t = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for step in range(t - 1, -1, -1):
        next_mask = mask[:, step + 1] if step + 1 < t else torch.zeros(b, device=mask.device)
        next_value = values[:, step + 1] if step + 1 < t else torch.zeros(b, device=values.device)
        delta = rewards[:, step] + gamma * next_value * next_mask - values[:, step]
        running = delta + gamma * lam * next_mask * running
        adv[:, step] = running
    return adv * mask


def apply_stop_properly(rewards: torch.Tensor, truncated: torch.Tensor, coef: float | None) -> torch.Tensor:
    """Scale the reward of a completion that hit the length limit.

    NeMo ProRLv2 `docs/guides/prorlv2.md` ("Stop properly" penalty):
    coef 0 zeros a truncated reward, coef 1 keeps it, values in between
    interpolate. None leaves the reward alone (the GRPO default there).
    """
    if coef is None:
        return rewards
    if not 0.0 <= float(coef) <= 1.0:
        raise ValueError("stop_properly coef must be in [0, 1] or None")
    scale = torch.where(truncated.bool(), torch.full_like(rewards, float(coef)), torch.ones_like(rewards))
    return rewards * scale


def apply_overlong_filter(loss_mask: torch.Tensor, truncated: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Drop truncated completions from the loss, not from the baseline.

    NeMo `docs/guides/grpo.md`, Overlong Filtering. The baseline still sees
    them when the caller computed advantages before this mask. That matches
    `masked_reward_policy: include`.
    """
    if not enabled:
        return loss_mask
    keep = (~truncated.bool()).to(loss_mask.dtype).unsqueeze(-1)
    return loss_mask * keep


def group_length_penalty(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    group_ids: torch.Tensor,
    coef: float,
) -> tuple[torch.Tensor, int]:
    """Shorter correct completions get a small bonus. Binary rewards only.

    NeMo `docs/guides/length-penalty.md` refuses length penalties on graded
    rewards, because a penalty on a partial score changes a different
    objective than "be brief when you are right". We skip a group that is
    not binary and count it. coef 0 returns the rewards unchanged.

    Inside a binary group, only completions with reward > 0 are adjusted.
    The longest correct one loses `coef`; the shortest loses nothing.
    An all-wrong group is unchanged, so it stays zero-variance.
    """
    if coef == 0.0:
        return rewards, 0
    out = rewards.clone()
    skipped = 0
    for idxs in _group_slices(group_ids).values():
        ix = torch.tensor(idxs, device=rewards.device)
        r = rewards.index_select(0, ix)
        binary = torch.all((r == 0) | (r == 1))
        if not bool(binary):
            skipped += 1
            continue
        positive = r > 0
        if int(positive.sum()) < 2:
            continue
        lens = lengths.index_select(0, ix)
        pos_len = lens[positive]
        span = pos_len.max() - pos_len.min()
        if float(span) == 0.0:
            continue
        # 0 at the shortest correct completion, coef at the longest.
        penalty = coef * (pos_len - pos_len.min()) / span
        adjusted = r.clone()
        adjusted[positive] = r[positive] - penalty.to(r.dtype)
        out.index_copy_(0, ix, adjusted)
    return out, skipped


def extract_reasoning_span(text: str) -> str:
    """Extract `<think>...</think>` reasoning content without touching protocol output.

    Handles three common reasoning formats:
    1. Explicit `<think>...</think>` span(s) in the completion.
    2. Prompt-prefilled `<think>` where completion starts directly and closes with `</think>`.
    3. Unclosed `<think>...` (truncated mid-thought): returns the entire runaway thought tail.
    """
    if not text:
        return ""
    closed = _THINK_BLOCK_RE.findall(text)
    if closed:
        return "\n".join(closed)
    lower = text.lower()
    if "</think>" in lower:
        idx = lower.index("</think>")
        return text[:idx]
    if "<think>" in lower:
        idx = lower.index("<think>") + len("<think>")
        return text[idx:]
    return ""


def reasoning_length_penalty(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    mode: str = "zmad",
    coef: float = 0.0,
    max_length: int | None = None,
    buffer_length: int | None = None,
    allow_graded: bool = True,
    min_reward: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Penalize bloated `<think>` reasoning traces on BOTH binary and graded lab rewards.

    Mirrors NeMo-RL ProRLv2.1 `length_penalty.py` (`zmad`) and DAPO soft clamped
    `overlong_buffer` (`reward_functions.py`), avoiding the 54% 12k-token truncation
    observed on LAB-Bench while leaving the `[ANSWER_START]` / protocol step output intact.
    """
    if coef == 0.0:
        return rewards, 0
    out = rewards.clone()
    skipped = 0
    if mode == "overlong_buffer":
        if max_length is None or buffer_length is None or buffer_length <= 0:
            raise ValueError("overlong_buffer mode requires max_length and positive buffer_length")
        expected_len = float(max_length - buffer_length)
        excess = (lengths.to(torch.float32) - expected_len) / float(buffer_length)
        # Clamped to [0, 1] * coef so multi-turn cumulative lengths never blow up advantages.
        penalty = excess.clamp(min=0.0, max=1.0) * float(coef)
        return (out - penalty.to(out.dtype)), 0
    if mode != "zmad":
        raise ValueError(f"unknown reasoning length penalty mode: {mode!r}")

    for idxs in _group_slices(group_ids).values():
        ix = torch.tensor(idxs, device=rewards.device)
        r = rewards.index_select(0, ix)
        if not allow_graded and not bool(torch.all((r == 0) | (r == 1))):
            skipped += 1
            continue
        positive = r > min_reward
        if int(positive.sum()) < 2:
            continue
        pos_len = lengths.index_select(0, ix)[positive].to(torch.float32)
        span = pos_len.max() - pos_len.min()
        if float(span) == 0.0:
            continue
        median = pos_len.median()
        mad = (pos_len - median).abs().median()
        if float(mad) > 0.0:
            # Robust ProRLv2.1 Z-MAD excess above median, capped to [0, 1] * coef
            z_excess = ((pos_len - median) / (1.4826 * mad + 1.0)).clamp(min=0.0, max=2.0) / 2.0
        else:
            z_excess = (pos_len - pos_len.min()) / span
        adjusted = r.clone()
        adjusted[positive] = r[positive] - (float(coef) * z_excess).to(r.dtype)
        out.index_copy_(0, ix, adjusted)
    return out, skipped
