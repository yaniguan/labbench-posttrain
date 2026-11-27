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
    effective_group_mask,
    gdpo_advantages,
    grpo_advantages,
    reinforce_pp_advantages,
)
from posttrain.batching import clone_frozen, model_device, model_max_length, snapshot_state, states_equal
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
    kl_mode: str = "token",
    kl_in_reward: bool = False,
    on_policy_kl: bool = False,
    adaptive_kl_target: float | None = None,
    max_kl_drift: float = 0.5,
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
    gdpo_weights: dict[str, float] | None = None,
    advantage_clip_low: float | None = None,
    advantage_clip_high: float | None = None,
    # Drop groups with identical rewards and optionally resample until target active groups fill (DAPO).
    dynamic_sampling: bool = False,
    dynamic_sampling_max_rounds: int = 3,
    # Truncation: drop from the loss, and/or scale the reward.
    overlong_filtering: bool = False,
    stop_properly_coef: float | None = None,
    # Binary rewards only. Graded lab rewards are skipped and counted.
    length_penalty_coef: float = 0.0,
    # Reasoning `<think>` length penalty (ProRLv2.1 Z-MAD) and DAPO soft clamped overlong buffer.
    reasoning_length_coef: float = 0.0,
    reasoning_length_mode: str = "zmad",
    overlong_buffer_length: int | None = None,
    overlong_buffer_penalty: float = 0.0,
    # Second anchor: NLL on scientist-written completions.
    sft_coef: float = 0.0,
    sft_examples: list | None = None,
    sft_batch_size: int = 2,
    sft_max_length: int | None = None,
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
        from posttrain.tiny import build_tiny_lm

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
    kl_anchor_violations: list[dict] = []
    texts = [p["prompt"] for p in prompts]
    for step in range(steps):
        if refresh_reference_every and step > 0 and step % refresh_reference_every == 0:
            reference.load_state_dict(model.state_dict())
            for p in reference.parameters():
                p.requires_grad_(False)
        target_groups = len(prompts)
        max_rounds = max(1, int(dynamic_sampling_max_rounds)) if dynamic_sampling else 1
        kept_rows: list = []
        all_first_round_rows: list = []
        n_dropped_total = 0
        n_sampled_groups_total = 0
        skipped_length = 0
        rounds_run = 0
        group_offset = 0
        for round_idx in range(max_rounds):
            rounds_run += 1
            round_seed = seed + step * 97 + round_idx * 13
            cand_rows = generate_rollouts(
                model, tokenizer, texts, prompts, group_size, max_new_tokens, temperature, round_seed,
            )
            for r_row in cand_rows:
                r_row.group_id += group_offset
            group_offset += len(texts)
            score_rollouts(cand_rows, reward_fn, reward_mode, round_seed)
            skipped_length += shape_rewards(
                cand_rows,
                stop_properly_coef,
                length_penalty_coef,
                reasoning_coef=reasoning_length_coef,
                reasoning_mode=reasoning_length_mode,
                max_new_tokens=max_new_tokens,
                overlong_buffer_length=overlong_buffer_length,
                overlong_buffer_penalty=overlong_buffer_penalty,
                tokenizer=tokenizer,
            )
            if round_idx == 0:
                all_first_round_rows = list(cand_rows)
            if reward_mode == "prior_only":
                break
            if not dynamic_sampling:
                kept_rows = cand_rows
                _, n_z, n_g = filter_zero_variance(cand_rows, enabled=True)
                n_dropped_total += n_z
                n_sampled_groups_total += n_g
                break
            non_zero_rows, n_dropped, n_g = filter_zero_variance(cand_rows, enabled=True)
            n_dropped_total += n_dropped
            n_sampled_groups_total += n_g
            kept_rows.extend(non_zero_rows)
            active_gids = sorted({r_row.group_id for r_row in kept_rows})
            if len(active_gids) >= target_groups:
                keep_set = set(active_gids[:target_groups])
                kept_rows = [r_row for r_row in kept_rows if r_row.group_id in keep_set]
                break

        if reward_mode == "prior_only":
            _, n_z0, n_g0 = filter_zero_variance(all_first_round_rows, enabled=True)
            zero_frac = n_z0 / n_g0 if n_g0 else 0.0
            from posttrain.ppo import _log_prior
            _log_prior(logger, step, all_first_round_rows, "grpo", kl_coef)
            print(f"grpo step {step} prior_only no parameter update zero_var {zero_frac:.2f}", flush=True)
            continue

        zero_frac = n_dropped_total / n_sampled_groups_total if n_sampled_groups_total else 0.0
        rows = kept_rows
        n_dropped = n_dropped_total if dynamic_sampling else 0
        if not rows:
            obs = all_first_round_rows
            obs_mean = sum(float(r.reward) for r in obs) / len(obs) if obs else 0.0
            obs_gold = sum(float(r.gold or 0.0) for r in obs) / len(obs) if obs else 0.0
            obs_kl = 0.0
            if obs:
                with torch.no_grad():
                    b0 = collate_rollouts(tokenizer, obs, device=dev)
                    sm0 = completion_mask_shifted(b0["completion_mask"])
                    if float(sm0.sum()) > 0:
                        lp0 = token_logprobs(model(input_ids=b0["input_ids"], attention_mask=b0["attention_mask"]).logits, b0["input_ids"]).detach()
                        alp0, _ = _ref_lp(sft_anchor, b0)
                        obs_kl = float(reduce_token_values(kl_term(lp0, alp0, "k3"), sm0, token_level=True))
            losses.append(0.0)
            step_kl_viol = _eval_kl_anchor_guard(
                kl_coef,
                obs_kl,
                max_kl_drift,
                states_equal(anchor_before, snapshot_state(sft_anchor)),
            )
            step_flags = [f"kl_anchor:{step_kl_viol}"] if step_kl_viol is not None else []
            if step_kl_viol is not None:
                kl_anchor_violations.append({"step": step, "rule": step_kl_viol, "kl_to_sft": obs_kl})
            logger.add(StepRecord(
                step=step, method="grpo", loss=0.0, grad_norm=0.0, kl_coef=kl_coef,
                kl_to_sft=obs_kl,
                reward_mean=obs_mean, reward_std=0.0, gold_metric=obs_gold,
                advantage_mean=0.0, advantage_std=0.0,
                response_length_mean=(sum(sum(r.completion_mask) for r in obs) / len(obs)) if obs else 0.0,
                zero_variance_group_fraction=zero_frac,
                hacking_flags=step_flags,
                loss_parts={
                    "dropped_groups": float(n_dropped),
                    "resample_rounds": float(rounds_run),
                    "length_groups_skipped": float(skipped_length),
                },
                status="no_gradient_group",
            ))
            print(f"grpo step {step} no group with reward variance; skipped update", flush=True)
            continue
        task_rewards_t = torch.tensor([r.reward for r in rows], dtype=torch.float32, device=dev)
        rewards_t = task_rewards_t
        groups = torch.tensor([r.group_id for r in rows], dtype=torch.long, device=dev)
        eff_mask = _effective_task_mask(rows, task_rewards_t, groups, advantage_mode, gdpo_weights)
        for idx_r, r_row in enumerate(rows):
            r_row.diagnostics["effective_group_mask"] = bool(eff_mask[idx_r].item())
        if not bool(eff_mask.any()) and not (sft_coef and sft_examples):
            from posttrain.ppo import _format_rate, _std

            obs_kl = 0.0
            with torch.no_grad():
                b0 = collate_rollouts(tokenizer, rows, device=dev)
                sm0 = completion_mask_shifted(b0["completion_mask"])
                if float(sm0.sum()) > 0:
                    lp0 = token_logprobs(model(input_ids=b0["input_ids"], attention_mask=b0["attention_mask"]).logits, b0["input_ids"]).detach()
                    alp0, _ = _ref_lp(sft_anchor, b0)
                    obs_kl = float(reduce_token_values(kl_term(lp0, alp0, "k3"), sm0, token_level=True))
            obs_mean = float(task_rewards_t.mean())
            obs_gold = sum(float(r.gold or 0.0) for r in rows) / len(rows)
            losses.append(0.0)
            step_kl_viol = _eval_kl_anchor_guard(
                kl_coef,
                obs_kl,
                max_kl_drift,
                states_equal(anchor_before, snapshot_state(sft_anchor)),
            )
            step_flags = [f"kl_anchor:{step_kl_viol}"] if step_kl_viol is not None else []
            if step_kl_viol is not None:
                kl_anchor_violations.append({"step": step, "rule": step_kl_viol, "kl_to_sft": obs_kl})
            logger.add(StepRecord(
                step=step, method="grpo", loss=0.0, grad_norm=0.0, kl_coef=kl_coef,
                kl_to_sft=obs_kl,
                reward_mean=obs_mean, reward_std=_std(task_rewards_t.tolist()), gold_metric=obs_gold,
                advantage_mean=0.0, advantage_std=0.0,
                response_length_mean=sum(sum(r.completion_mask) for r in rows) / len(rows),
                format_compliance=_format_rate([r.completion for r in rows]),
                zero_variance_group_fraction=1.0,
                hacking_flags=step_flags,
                loss_parts={
                    "dropped_groups": float(n_dropped),
                    "resample_rounds": float(rounds_run),
                    "length_groups_skipped": float(skipped_length),
                    "effective_groups": 0.0,
                },
                status="no_gradient_group",
            ))
            print(f"grpo step {step} no group with reward variance; skipped update", flush=True)
            continue
        adv = _advantages(rows, task_rewards_t, groups, advantage_mode, baseline, normalize_std, gdpo_weights)
        adv = torch.where(eff_mask, adv, torch.zeros_like(adv))
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
            # Apply detached token-KL penalty AFTER task-reward group-std normalization (#44)
            # and mask out zero-variance groups so KL noise never amplifies to unit variance.
            token_kl = kl_term(old_lp, ref_lp, kl_estimator).detach()
            per_row = (token_kl * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1.0)
            kl_delta = -kl_coef * per_row
            kl_centered = grpo_advantages(kl_delta, groups, baseline=baseline, normalize_std=False)
            adv = adv + torch.where(eff_mask, kl_centered, torch.zeros_like(kl_centered))
            rewards_t = task_rewards_t + torch.where(eff_mask, kl_delta, torch.zeros_like(kl_delta))
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
                kl_mode=kl_mode,
            )
            if is_metrics:
                last_parts.update(is_metrics)
            last_parts["resample_rounds"] = float(rounds_run)
            last_parts["dropped_groups"] = float(n_dropped)
            if sft_coef and sft_examples:
                extra = _sft_extra(
                    model,
                    tokenizer,
                    sft_examples,
                    step=step,
                    batch_size=sft_batch_size,
                    max_length=sft_max_length,
                )
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
        flags = list(_maybe_monitor(step, monitor_every, rows, proxy, gold, stop_on_hack))
        step_kl_viol = _eval_kl_anchor_guard(
            kl_coef,
            anchor_kl,
            max_kl_drift,
            states_equal(anchor_before, snapshot_state(sft_anchor)),
        )
        if step_kl_viol is not None:
            flags.append(f"kl_anchor:{step_kl_viol}")
            kl_anchor_violations.append({"step": step, "rule": step_kl_viol, "kl_to_sft": anchor_kl})
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
        "kl_anchor_violations": kl_anchor_violations,
        "logger": logger,
        "kl_coef": kl_coef,
        "status": "ran",
    }


def _eval_kl_anchor_guard(
    kl_coef: float,
    kl_to_sft: float,
    max_kl_drift: float,
    sft_anchor_frozen: bool,
) -> str | None:
    """Invoke `autolab_wire.check_kl_anchor` (`LabPostTrainLean.checkKLAnchor`, #55)."""
    from autolab_wire import check_kl_anchor

    coef_val = float(kl_coef or 0.0)
    coef_pm = 0 if coef_val <= 0.0 else max(1, int(round(coef_val * 1000.0)))
    kl_val = float(kl_to_sft or 0.0)
    kl_pm = -1 if kl_val < -1e-6 else max(0, int(round(kl_val * 1000.0)))
    cap_pm = max(0, int(round(float(max_kl_drift) * 1000.0)))
    return check_kl_anchor(
        kl_coef_permille=coef_pm,
        kl_to_sft_permille=kl_pm,
        max_kl_drift_permille=cap_pm,
        sft_anchor_frozen=bool(sft_anchor_frozen),
    )


def _effective_task_mask(rows, rewards_t, groups, mode, gdpo_weights=None) -> torch.Tensor:
    eff = effective_group_mask(rewards_t, groups)
    if mode in ("gdpo", "gspo"):
        names = sorted({k for r in rows for k in r.components if str(k).startswith("reward/")})
        if len(names) >= 2 and (mode == "gdpo" or gdpo_weights):
            comps = {
                n: torch.tensor([float(r.components.get(n, 0.0)) for r in rows], device=rewards_t.device)
                for n in names
            }
            eff = eff | effective_group_mask(comps, groups, weights=gdpo_weights).to(rewards_t.device)
    return eff


def _advantages(rows, rewards_t, groups, mode, baseline, normalize_std, gdpo_weights=None):
    if mode == "grpo":
        return grpo_advantages(rewards_t, groups, baseline=baseline, normalize_std=normalize_std)
    if mode == "reinforce_pp":
        return reinforce_pp_advantages(rewards_t, groups, minus_baseline=True)
    if mode == "gspo":
        names = sorted({k for r in rows for k in r.components if str(k).startswith("reward/")})
        if gdpo_weights and len(names) >= 2:
            mode = "gdpo"
        else:
            return grpo_advantages(rewards_t, groups, baseline=baseline, normalize_std=normalize_std)
    if mode == "gdpo":
        from autolab_wire import check_gdpo_pareto_weights

        names = sorted({k for r in rows for k in r.components if str(k).startswith("reward/")})
        if len(names) < 2:
            raise ValueError("GDPO needs at least two reward/ components; this batch has " + ", ".join(names))
        w_vec_pm = [
            int(round(float((gdpo_weights or {}).get(n, 1.0)) * 1000.0))
            for n in names
        ]
        ch_dummy_pm = [1000 for _ in names]
        gdpo_viol = check_gdpo_pareto_weights(ch_dummy_pm, w_vec_pm)
        if gdpo_viol is not None:
            raise ValueError(f"Invalid GDPO weights ({gdpo_viol}): weights={gdpo_weights!r} for channels {names}")
        comps = {n: torch.tensor([float(r.components.get(n, 0.0)) for r in rows], device=rewards_t.device) for n in names}
        comp_eff = effective_group_mask(comps, groups, weights=gdpo_weights).to(rewards_t.device)
        base_adv = gdpo_advantages(comps, groups, weights=gdpo_weights, normalize_std=normalize_std)
        # Apply additive shaping (reasoning length, stop_properly, overlong_buffer) and kl_in_reward
        # deltas AFTER per-channel standardization so small coefficients retain exact linear scale (#21),
        # gated by `comp_eff` so dead groups never pick up shaping or KL noise (#44).
        unshaped = torch.tensor(
            [
                float((r.diagnostics or {}).get("pre_shaping_reward", float(r.reward) - float((r.diagnostics or {}).get("shaping_delta", 0.0))))
                for r in rows
            ],
            dtype=rewards_t.dtype,
            device=rewards_t.device,
        )
        additive_delta = rewards_t - unshaped
        if float(additive_delta.abs().max()) > 0.0 and bool(comp_eff.any()):
            centered_delta = grpo_advantages(additive_delta, groups, baseline="mean", normalize_std=False)
            return base_adv + torch.where(comp_eff, centered_delta, torch.zeros_like(centered_delta))
        return base_adv
    raise ValueError(f"unknown advantage_mode {mode}")


def _ref_lp(model, batch):
    from posttrain.batching import frozen_logprobs
    return frozen_logprobs(model, batch)


def _sft_extra(
    model,
    tokenizer,
    examples,
    *,
    step: int = 0,
    batch_size: int = 2,
    max_length: int | None = None,
):
    from posttrain.batching import collate, encode_example

    bs = max(1, min(int(batch_size), len(examples)))
    eff_max_len = int(max_length) if max_length is not None else model_max_length(model, tokenizer)
    sub = [examples[(step * bs + i) % len(examples)] for i in range(bs)]
    pairs = [encode_example(tokenizer, r["prompt"], r["completion"], eff_max_len) for r in sub]
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
