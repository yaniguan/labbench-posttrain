"""GRPO with the switches the lab recipes actually argue about.

The reference policy is the SFT checkpoint. KL is a loss term by default
(NeMo `docs/guides/grpo.md`), coefficient 0.01, estimator k3. KL in the
reward is the PPO style; pass kl_in_reward=True to use it here too.

Each switch is a keyword with a default. The unit tests in
tests/test_posttrain_losses.py cover the tensor each switch changes.
What the switch is for is written next to the argument below and in
docs/posttrain/losses_and_gradients.md.
"""
from __future__ import annotations

import argparse

import torch

from posttrain.advantages import (
    clip_advantages,
    gdpo_advantages,
    grpo_advantages,
    reinforce_pp_advantages,
)
from posttrain.batching import clone_frozen, model_device, snapshot_state, states_equal
from posttrain.checkpoint import save_checkpoint
from posttrain.kl import DEFAULT_KL_COEF, DEFAULT_KL_ESTIMATOR, AdaptiveKLController, kl_term
from posttrain.losses import completion_mask_shifted, compute_is_weights, reduce_token_values, sft_nll, token_logprobs
from posttrain.metrics import StepLogger, StepRecord, clip_fraction_by_bucket, grad_norm, grad_norms_by_group
from posttrain.rewards import toy_protocol_reward
from posttrain.rl_common import (
    apply_loss_mask,
    bind_mask,
    collate_rollouts,
    filter_zero_variance,
    generate_rollouts,
    masked_entropy,
    policy_loss,
    require_kl,
    score_rollouts,
    shape_rewards,
)
from posttrain.sft import TOY_SFT, train_sft
from posttrain.tiny import build_tiny_lm

TOY_PROMPTS = [
    {"prompt": "Heat the sample to", "key": "37"},
    {"prompt": "Spin the tube at", "key": "3000"},
]


def train_grpo(
    prompts: list[dict],
    reward_fn,
    *,
    # KL anchor. On by default. 0 warns and is an ablation.
    kl_coef: float = DEFAULT_KL_COEF["grpo"],
    kl_estimator: str = DEFAULT_KL_ESTIMATOR,
    kl_in_reward: bool = False,
    on_policy_kl: bool = False,
    adaptive_kl_target: float | None = None,
    refresh_reference_every: int = 0,
    # Clip. Off studies the Shao et al. Section 4 clipping bias.
    # eps_low != eps_high is clip-higher (DAPO / ProRLv2).
    clip: bool = True,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    dual_clip_c: float | None = None,
    use_cispo: bool = False,
    sequence_level_ratios: bool = False,
    # Token-level loss weights long completions more. Sequence-level does not.
    token_level: bool = True,
    # Group baseline. "loo" is leave-one-out. normalize_std divides by the spread.
    baseline: str = "mean",
    normalize_std: bool = True,
    advantage_mode: str = "grpo",
    advantage_clip_low: float | None = None,
    advantage_clip_high: float | None = None,
    # Drop groups with identical rewards. They have no relative signal.
    dynamic_sampling: bool = False,
    # Truncation: drop from the loss, and/or scale the reward.
    overlong_filtering: bool = False,
    stop_properly_coef: float | None = None,
    # Binary rewards only. Graded lab rewards are skipped and counted.
    length_penalty_coef: float = 0.0,
    # Second anchor: NLL on scientist-written completions.
    sft_coef: float = 0.0,
    sft_examples: list | None = None,
    # Importance sampling correction ('tis', 'icepop', 'seq-mask-tis').
    use_is_correction: bool = False,
    is_mode: str = "tis",
    is_clip_min: float = 0.5,
    is_clip: float = 2.0,
    reward_mode: str = "ground_truth",
    entropy_coef: float = 0.0,
    group_size: int = 4,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    epochs: int = 1,
    steps: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    model=None,
    tokenizer=None,
    reference=None,
    sft_anchor=None,
    style_reference: list[str] | None = None,
    monitor_every: int = 0,
    stop_on_hack: bool = False,
    checkpoint_path: str | None = None,
    logger: StepLogger | None = None,
) -> dict:
    require_kl("grpo", kl_coef)
    if use_cispo and not token_level:
        raise ValueError("CISPO is a per-token objective; token_level must stay True")
    if sequence_level_ratios and token_level:
        raise ValueError("GSPO sequence_level_ratios requires token_level=False")
    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    dev = model_device(model)
    if reference is None:
        reference = clone_frozen(model)
    if sft_anchor is None:
        sft_anchor = clone_frozen(reference)
    ref_before = snapshot_state(reference)
    anchor_before = snapshot_state(sft_anchor)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    controller = None
    if adaptive_kl_target is not None and kl_coef > 0:
        controller = AdaptiveKLController(kl_coef, adaptive_kl_target)
    logger = logger or StepLogger()
    losses = []
    texts = [p["prompt"] for p in prompts]
    for step in range(steps):
        if refresh_reference_every and step > 0 and step % refresh_reference_every == 0:
            reference.load_state_dict(model.state_dict())
            for p in reference.parameters():
                p.requires_grad_(False)
        rows = generate_rollouts(
            model, tokenizer, texts, prompts, group_size, max_new_tokens, temperature, seed + step,
        )
        score_rollouts(rows, reward_fn, reward_mode, seed)
        skipped_length = shape_rewards(rows, stop_properly_coef, length_penalty_coef)
        _, n_zero_groups, n_groups = filter_zero_variance(rows, enabled=True)
        zero_frac = n_zero_groups / n_groups if n_groups else 0.0
        if reward_mode == "prior_only":
            from posttrain.ppo import _log_prior
            _log_prior(logger, step, rows, "grpo", kl_coef)
            print(f"grpo step {step} prior_only no parameter update zero_var {zero_frac:.2f}", flush=True)
            continue
        rows, n_dropped, _ = filter_zero_variance(rows, enabled=dynamic_sampling)
        if not rows:
            losses.append(0.0)
            logger.add(StepRecord(
                step=step, method="grpo", loss=0.0, kl_coef=kl_coef,
                zero_variance_group_fraction=zero_frac,
                loss_parts={"dropped_groups": float(n_dropped), "length_groups_skipped": float(skipped_length)},
                status="no_gradient_group",
            ))
            print(f"grpo step {step} no group with reward variance; skipped update", flush=True)
            continue
        rewards_t = torch.tensor([r.reward for r in rows], dtype=torch.float32, device=dev)
        groups = torch.tensor([r.group_id for r in rows], dtype=torch.long, device=dev)
        adv = _advantages(rows, rewards_t, groups, advantage_mode, baseline, normalize_std)
        adv = clip_advantages(adv, advantage_clip_low, advantage_clip_high)
        batch = collate_rollouts(tokenizer, rows, device=dev)
        shift_mask = completion_mask_shifted(batch["completion_mask"])
        truncated = torch.tensor([r.truncated for r in rows], device=dev)
        shift_mask = apply_loss_mask(shift_mask, truncated, overlong_filtering)
        model.eval()
        with torch.no_grad():
            old_out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            old_lp = token_logprobs(old_out.logits, batch["input_ids"]).detach()
            ref_lp, _ = _ref_lp(reference, batch)
            anchor_lp, _ = _ref_lp(sft_anchor, batch)
        if kl_in_reward and kl_coef:
            # One scalar per completion: mean token KL, detached, subtracted
            # from the reward before the advantage. This is the PPO placement.
            token_kl = kl_term(old_lp, ref_lp, kl_estimator).detach()
            per_row = (token_kl * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1.0)
            rewards_t = rewards_t - kl_coef * per_row
            adv = _advantages(rows, rewards_t, groups, advantage_mode, baseline, normalize_std)
            adv = clip_advantages(adv, advantage_clip_low, advantage_clip_high)
        adv_tokens = adv.unsqueeze(1).expand_as(old_lp)
        is_weight = None
        is_metrics = {}
        if use_is_correction:
            gen_lp = batch.get("gen_logprobs", old_lp)
            is_weight, is_metrics = compute_is_weights(
                old_lp, gen_lp, shift_mask,
                mode=is_mode, clip_min=is_clip_min, clip_max=is_clip,
                sequence_level=sequence_level_ratios,
            )
        ran_loss = 0.0
        last_parts = {"loss": 0.0}
        approx = 0.0
        ent_value = 0.0
        g = 0.0
        groups_g = {}
        bind = torch.zeros_like(shift_mask, dtype=torch.bool)
        ran = False
        for _epoch in range(epochs):
            model.train()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logp = token_logprobs(out.logits, batch["input_ids"])
            ent = masked_entropy(out.logits, batch["completion_mask"])
            loss, last_parts, approx = policy_loss(
                logp, old_lp, ref_lp, adv_tokens, shift_mask,
                kl_coef=0.0 if kl_in_reward else kl_coef,
                kl_estimator=kl_estimator,
                kl_in_loss=not kl_in_reward,
                eps_low=eps_low, eps_high=eps_high,
                dual_clip_c=dual_clip_c,
                clip=clip, cispo=use_cispo, token_level=token_level,
                entropy=ent, entropy_coef=entropy_coef,
                on_policy_kl=on_policy_kl, is_weight=is_weight,
                sequence_level_ratios=sequence_level_ratios,
            )
            if is_metrics:
                last_parts.update(is_metrics)
            if sft_coef and sft_examples:
                extra = _sft_extra(model, tokenizer, sft_examples)
                loss = loss + sft_coef * extra
                last_parts["sft"] = float(extra.detach())
            if not torch.isfinite(loss):
                raise RuntimeError(f"GRPO loss is not finite at step {step}")
            if float(shift_mask.sum()) == 0:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = grad_norm(model.parameters())
            groups_g = grad_norms_by_group(model)
            opt.step()
            ran = True
            ran_loss = float(loss.detach())
            ent_value = float(ent.detach())
            bind = bind_mask(logp, old_lp, adv_tokens, eps_low, eps_high, clip and not use_cispo)
        losses.append(ran_loss if ran else 0.0)
        with torch.no_grad():
            anchor_kl = float(reduce_token_values(kl_term(old_lp, anchor_lp, "k3"), shift_mask, token_level=True)) if float(shift_mask.sum()) else 0.0
        if controller is not None and approx is not None:
            kl_coef = controller.update(float(approx))
        completions = [r.completion for r in rows]
        proxy = float(rewards_t.mean())
        gold = sum(r.gold for r in rows) / len(rows)
        from posttrain.ppo import _format_rate, _maybe_monitor, _style_distance, _std
        flags = _maybe_monitor(step, monitor_every, rows, proxy, gold, stop_on_hack)
        logger.add(StepRecord(
            step=step, method="grpo", loss=losses[-1], loss_parts=last_parts,
            grad_norm=g, grad_norm_by_group=groups_g,
            clip_fraction_by_prob_bucket=clip_fraction_by_bucket(old_lp.exp(), bind, shift_mask),
            approx_kl=None if approx is None else float(approx),
            kl_to_sft=anchor_kl, entropy=ent_value,
            reward_mean=proxy, reward_std=_std(rewards_t.tolist()),
            zero_variance_group_fraction=zero_frac,
            response_length_mean=sum(sum(r.completion_mask) for r in rows) / len(rows),
            advantage_mean=float(adv.mean()), advantage_std=float(adv.std(unbiased=False)),
            format_compliance=_format_rate(completions), gold_metric=gold,
            style_distance_to_sft=_style_distance(completions, style_reference),
            hacking_flags=flags, kl_coef=kl_coef,
        ))
        print(
            f"grpo step {step} loss {losses[-1]:.4f} reward {proxy:.3f} gold {gold:.3f} "
            f"kl_to_sft {anchor_kl:.4f} zero_var {zero_frac:.2f} grad_norm {g:.4f} finite True",
            flush=True,
        )
    final = losses[-1] if losses else 0.0
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, steps, extra={"method": "grpo", "kl_coef": kl_coef})
    return {
        "losses": losses,
        "final_loss": final,
        "model": model,
        "tokenizer": tokenizer,
        "reference": reference,
        "sft_anchor": sft_anchor,
        "reference_unchanged": states_equal(ref_before, snapshot_state(reference)),
        "sft_anchor_unchanged": states_equal(anchor_before, snapshot_state(sft_anchor)),
        "logger": logger,
        "kl_coef": kl_coef,
        "status": "ran",
    }


def _advantages(rows, rewards_t, groups, mode, baseline, normalize_std):
    if mode == "grpo":
        return grpo_advantages(rewards_t, groups, baseline=baseline, normalize_std=normalize_std)
    if mode == "reinforce_pp":
        return reinforce_pp_advantages(rewards_t, groups, minus_baseline=True)
    if mode == "gdpo":
        names = sorted({k for r in rows for k in r.components})
        if len(names) < 2:
            raise ValueError("GDPO needs at least two reward/ components; this batch has " + ", ".join(names))
        comps = {n: torch.tensor([float(r.components.get(n, 0.0)) for r in rows], device=rewards_t.device) for n in names}
        return gdpo_advantages(comps, groups, normalize_std=normalize_std)
    raise ValueError(f"unknown advantage_mode {mode}")


def _ref_lp(model, batch):
    from posttrain.batching import frozen_logprobs
    return frozen_logprobs(model, batch)


def _sft_extra(model, tokenizer, examples):
    from posttrain.batching import collate, encode_example
    pairs = [encode_example(tokenizer, r["prompt"], r["completion"], 48) for r in examples[:2]]
    batch = collate(tokenizer, pairs, device=model_device(model))
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return sft_nll(out.logits, batch["input_ids"], batch["completion_mask"])


def smoke() -> dict:
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    ref = clone_frozen(sft["model"])
    anchor = clone_frozen(sft["model"])
    out = train_grpo(
        TOY_PROMPTS, toy_protocol_reward,
        model=sft["model"], tokenizer=sft["tokenizer"],
        reference=ref, sft_anchor=anchor,
        steps=2, group_size=2, max_new_tokens=4, seed=0, monitor_every=1,
        reward_mode="random_p50",
        style_reference=[r["completion"] for r in TOY_SFT],
    )
    print(
        f"grpo smoke done steps 2 final_loss {out['final_loss']:.4f} "
        f"reference_unchanged {out['reference_unchanged']}",
        flush=True,
    )
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="GRPO against a frozen SFT reference.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--kl-coef", type=float, default=None)
    args = p.parse_args(argv)
    if args.kl_coef is None:
        smoke()
        return
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    out = train_grpo(
        TOY_PROMPTS, toy_protocol_reward,
        model=sft["model"], tokenizer=sft["tokenizer"],
        reference=clone_frozen(sft["model"]),
        sft_anchor=clone_frozen(sft["model"]),
        steps=1, group_size=2, max_new_tokens=4, kl_coef=args.kl_coef, seed=0,
    )
    print(f"grpo smoke done steps 1 final_loss {out['final_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
