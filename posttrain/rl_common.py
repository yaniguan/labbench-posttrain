"""Generation and the shared policy-loss step.

PPO and GRPO both sample completions, score them, and take a clipped
policy gradient. The advantage is different. That difference stays in
ppo.py and grpo.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from posttrain.advantages import (
    apply_overlong_filter,
    apply_stop_properly,
    extract_reasoning_span,
    group_length_penalty,
    reasoning_length_penalty,
)
from posttrain.batching import collate, forward_logprobs, model_device, model_max_length
from posttrain.kl import (
    kl_term,
    reverse_kl_loss_term,
    sequence_reverse_kl_loss_term,
    warn_if_kl_disabled,
)
from posttrain.losses import clip_binds, compute_is_weights, entropy_from_logits, policy_surrogate, reduce_token_values
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
    gen_logprobs: list[float] | None = None


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompts,
    metas,
    group_size,
    max_new,
    temperature,
    seed,
    *,
    batched_group: bool = False,
) -> list[Rollout]:
    """Sample `group_size` completions for each prompt.

    Right-padding a heterogeneous batch and calling generate() lets causal LMs
    continue after pad tokens. Within a single prompt, all `group_size` copies
    have identical length so `batched_group=True` can sample a group in one
    GPU call without padding artifacts; default `False` keeps exact sequential
    RNG parity for CPU unit tests.
    """
    model.eval()
    torch.manual_seed(seed)
    dev = model_device(model)
    max_ctx = model_max_length(model, tokenizer)
    max_prompt = max_ctx - max_new
    if max_prompt < 2:
        raise ValueError("context window is too small for the requested completion length")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows: list[Rollout] = []
    for p_index, prompt in enumerate(prompts):
        meta = metas[p_index] if metas else {}
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_prompt]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        is_greedy = not (temperature and float(temperature) > 0.0)
        gen_kw = (
            {"do_sample": True, "temperature": float(temperature), "top_k": 0, "top_p": 1.0}
            if not is_greedy
            else {"do_sample": False}
        )
        temp_scale = 1.0 if is_greedy else float(temperature)

        def _extract_seq_and_lp(gen_res, batch_idx: int) -> tuple[list[int], list[float] | None]:
            if hasattr(gen_res, "sequences"):
                raw_seq = gen_res.sequences[batch_idx, len(prompt_ids) :].tolist()
                scores = getattr(gen_res, "scores", None)
                if scores:
                    lps: list[float] = []
                    n_steps = min(len(raw_seq), len(scores))
                    for step_i in range(n_steps):
                        step_logits = scores[step_i][batch_idx].float() / temp_scale
                        step_lp = torch.log_softmax(step_logits, dim=-1)
                        lps.append(float(step_lp[int(raw_seq[step_i])].item()))
                    return raw_seq[:n_steps], lps
                return raw_seq, None
            return gen_res[batch_idx, len(prompt_ids) :].tolist(), None

        if batched_group and group_size > 1:
            input_ids = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=dev)
            attn = torch.ones_like(input_ids)
            outs = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kw,
            )
            seq_lp_list = [_extract_seq_and_lp(outs, g) for g in range(group_size)]
        else:
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
            attn = torch.ones_like(input_ids)
            seq_lp_list = []
            for _g in range(group_size):
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kw,
                )
                seq_lp_list.append(_extract_seq_and_lp(out, 0))

        for new_ids, new_lps in seq_lp_list:
            # Truncate cleanly after first eos if pad == eos or model kept emitting
            if tokenizer.eos_token_id in new_ids:
                eos_pos = new_ids.index(tokenizer.eos_token_id)
                new_ids = new_ids[: eos_pos + 1]
                if new_lps is not None:
                    new_lps = new_lps[: eos_pos + 1]
                hit_eos = True
            else:
                if pad_id != tokenizer.eos_token_id:
                    if new_lps is not None:
                        kept_pairs = [(t, lp) for t, lp in zip(new_ids, new_lps) if t != pad_id]
                        new_ids = [t for t, _ in kept_pairs]
                        new_lps = [lp for _, lp in kept_pairs]
                    else:
                        new_ids = [t for t in new_ids if t != pad_id]
                hit_eos = False
            if not new_ids:
                new_ids = [tokenizer.eos_token_id]
                new_lps = None
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
                gen_logprobs=new_lps,
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
            if hasattr(gt, "gdpo_channels"):
                row.components = gt.gdpo_channels()
            else:
                row.components = dict(gt.components) or {"reward/score": float(gt.value)}
        else:
            row.components = dict(scored.components)
        row.diagnostics = dict(scored.diagnostics)
        gt_diag = getattr(gt, "diagnostics", None)
        if isinstance(gt_diag, dict):
            row.diagnostics["parsed_env"] = bool(gt_diag.get("parsed", False))
        else:
            row.diagnostics["parsed_env"] = bool(getattr(gt, "parsed", False))


def shape_rewards(
    rows: list[Rollout],
    stop_coef: float | None,
    length_coef: float,
    *,
    reasoning_coef: float = 0.0,
    reasoning_mode: str = "zmad",
    max_new_tokens: int | None = None,
    overlong_buffer_length: int | None = None,
    overlong_buffer_penalty: float = 0.0,
    tokenizer=None,
) -> int:
    """Stop-properly penalty, binary group length penalty, plus `<think>` / DAPO overlong shaping.

    Returns how many groups the binary length penalty skipped because the reward
    was not binary. `reasoning_coef` (`zmad`) and `overlong_buffer_penalty` operate
    cleanly on graded BioProBench rewards without shrinking protocol step content.
    """
    if not rows:
        return 0
    base_rewards = torch.tensor([r.reward for r in rows], dtype=torch.float32)
    truncated = torch.tensor([r.truncated for r in rows])
    rewards = apply_stop_properly(base_rewards, truncated, stop_coef)
    lengths = torch.tensor([sum(r.completion_mask) for r in rows], dtype=torch.float32)
    groups = torch.tensor([r.group_id for r in rows], dtype=torch.long)
    rewards, skipped = group_length_penalty(rewards, lengths, groups, length_coef)
    if reasoning_coef > 0.0:
        think_spans = [extract_reasoning_span(r.completion) for r in rows]
        if tokenizer is not None:
            think_lens = torch.tensor(
                [len(tokenizer.encode(sp, add_special_tokens=False)) if sp else 0 for sp in think_spans],
                dtype=torch.float32,
            )
        else:
            think_lens = torch.tensor([len(sp.split()) for sp in think_spans], dtype=torch.float32)
        rewards, _ = reasoning_length_penalty(
            rewards, think_lens, groups, mode=reasoning_mode, coef=reasoning_coef, allow_graded=True
        )
    if overlong_buffer_penalty > 0.0 and max_new_tokens is not None and overlong_buffer_length:
        rewards, _ = reasoning_length_penalty(
            rewards,
            lengths,
            groups,
            mode="overlong_buffer",
            coef=overlong_buffer_penalty,
            max_length=max_new_tokens,
            buffer_length=overlong_buffer_length,
            allow_graded=True,
        )
    deltas = (rewards - base_rewards).tolist()
    for row, base_val, value, delta in zip(rows, base_rewards.tolist(), rewards.tolist(), deltas):
        row.diagnostics["pre_shaping_reward"] = float(base_val)
        row.diagnostics["shaping_delta"] = float(delta)
        row.reward = float(value)
    return skipped


def filter_zero_variance(rows: list[Rollout], enabled: bool) -> tuple[list[Rollout], int, int]:
    """Drop groups whose rewards are all equal. Count groups, not rows."""
    groups: dict[int, list[Rollout]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    n_groups = len(groups)
    kept = []
    dropped = 0
    for members in groups.values():
        values = [m.reward for m in members]
        pre_vals = [float((m.diagnostics or {}).get("pre_shaping_reward", m.reward)) for m in members]
        gate_all_zero = bool(pre_vals) and max(pre_vals) <= 0.0 and min(pre_vals) == max(pre_vals)
        is_zero_var = (max(values) == min(values)) or gate_all_zero
        for m in members:
            m.diagnostics["effective_group_mask"] = not is_zero_var
        if is_zero_var:
            dropped += 1
        else:
            kept.extend(members)
    if not enabled:
        return rows, 0, n_groups
    return kept, dropped, n_groups


def collate_rollouts(tokenizer, rows: list[Rollout], device="cpu"):
    pairs = [(r.input_ids, r.completion_mask) for r in rows]
    batch = collate(tokenizer, pairs, device=device)
    if any(r.gen_logprobs is not None for r in rows):
        t_minus_1 = batch["input_ids"].shape[1] - 1
        gen_lp = torch.zeros((len(rows), t_minus_1), dtype=torch.float32, device=device)
        for i, r in enumerate(rows):
            if r.gen_logprobs is not None:
                # Align completion token logprobs with shifted completion positions
                prompt_len = r.completion_mask.index(1) if 1 in r.completion_mask else len(r.input_ids)
                start = max(0, prompt_len - 1)
                vals = r.gen_logprobs[: max(0, t_minus_1 - start)]
                if vals:
                    gen_lp[i, start : start + len(vals)] = torch.tensor(vals, dtype=torch.float32, device=device)
        batch["gen_logprobs"] = gen_lp
    return batch


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
    sequence_level_ratios: bool = False,
    force_on_policy_ratio: bool = False,
    positive_nll_mask: torch.Tensor | None = None,
    positive_nll_coef: float = 0.0,
    kl_mode: str = "token",
):
    """Clipped (or CISPO / GSPO) surrogate, plus Reverse-KL loss term.

    `kl_mode="token"` (default) applies per-token `reverse_kl_loss_term`, matching
    NeMo-RL `reference_policy_kl_type: k3` (unbiased per token, partial at sequence level).
    `kl_mode="sequence_exact"` applies `sequence_reverse_kl_loss_term`, restoring the
    exact multi-token sequence gradient `grad KL(Pi_theta || Pi_ref)` (#54).
    `kl_in_loss` False means the caller already folded a detached KL into the
    advantage (PPO style). Adding it again would punish twice.
    """
    if kl_mode not in ("token", "sequence_exact"):
        raise ValueError(f"kl_mode must be 'token' or 'sequence_exact', got {kl_mode!r}")
    if sequence_level_ratios and token_level:
        raise ValueError("GSPO sequence_level_ratios requires token_level=False")
    per = policy_surrogate(
        logp, logp_old, advantage,
        eps_low=eps_low, eps_high=eps_high, clip=clip, cispo=cispo,
        dual_clip_c=dual_clip_c,
        sequence_level_ratios=sequence_level_ratios,
        mask=mask,
        force_on_policy_ratio=force_on_policy_ratio,
    )
    if is_weight is not None:
        # Detached actor-vs-sampler weight (TIS / ICE-POP / Seq-Mask-TIS).
        per = per * is_weight.detach()
    pg = reduce_token_values(per, mask, token_level=token_level)
    parts = {"policy": float(pg.detach())}
    loss = pg
    approx = None
    if logp_ref is not None and mask.sum() > 0:
        if kl_mode == "sequence_exact":
            kl_surrogate = sequence_reverse_kl_loss_term(
                logp,
                logp_ref,
                mask,
                estimator=kl_estimator,
                logp_behavior=logp_old if on_policy_kl else None,
            )
            approx = float(reduce_token_values(kl_surrogate.detach(), mask, token_level=True))
            if kl_in_loss and kl_coef:
                active_seqs = (mask.sum(dim=-1) > 0).to(kl_surrogate.dtype).sum().clamp(min=1.0)
                kl_loss = kl_surrogate.sum() / active_seqs
                loss = loss + kl_coef * kl_loss
                parts["kl"] = float(kl_loss.detach())
        else:
            kl_surrogate = reverse_kl_loss_term(
                logp,
                logp_ref,
                estimator=kl_estimator,
                logp_behavior=logp_old if on_policy_kl else None,
            )
            approx = float(reduce_token_values(kl_surrogate.detach(), mask, token_level=True))
            if kl_in_loss and kl_coef:
                kl_loss = reduce_token_values(kl_surrogate, mask, token_level=token_level)
                loss = loss + kl_coef * kl_loss
                parts["kl"] = float(kl_loss.detach())
    if positive_nll_mask is not None and positive_nll_coef > 0.0 and float(positive_nll_mask.sum()) > 0:
        # VAPO positive-example NLL anchor on high-reward completions (arXiv:2504.05118, Eq. 10)
        pos_nll = reduce_token_values(-logp, positive_nll_mask, token_level=True)
        loss = loss + positive_nll_coef * pos_nll
        parts["positive_nll"] = float(pos_nll.detach())
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
    if kl_coef is not None and float(kl_coef) < 0.0:
        raise ValueError(f"{method} kl_coef must be non-negative, got {kl_coef}")
    warn_if_kl_disabled(kl_coef, method)
