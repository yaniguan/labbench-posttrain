"""PPO with a value head and GAE. The KL anchor is the SFT checkpoint.

Default: KL is on, and it is subtracted inside the reward (the PPO style).
Set kl_in_reward False to put the same estimator in the loss instead.
Do not turn the coefficient to 0 unless the run is marked as an ablation.
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from posttrain.advantages import gae
from posttrain.batching import clone_frozen, snapshot_state, states_equal
from posttrain.checkpoint import save_checkpoint
from posttrain.kl import DEFAULT_KL_COEF, DEFAULT_KL_ESTIMATOR, AdaptiveKLController, kl_term
from posttrain.losses import completion_mask_shifted, reduce_token_values, sft_nll, token_logprobs, value_loss
from posttrain.metrics import StepLogger, StepRecord, clip_fraction_by_bucket, grad_norm, grad_norms_by_group
from posttrain.rewards import toy_protocol_reward
from posttrain.rl_common import (
    Rollout,
    bind_mask,
    collate_rollouts,
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


def _style_distance(current: list[str], reference_texts: list[str] | None) -> float | None:
    if not reference_texts:
        return None
    from collections import Counter

    def dist(texts):
        counts: Counter[str] = Counter()
        total = 0
        for text in texts:
            for word in (text or "").lower().split():
                counts[word] += 1
                total += 1
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    a, b = dist(current), dist(reference_texts)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _format_rate(completions: list[str]) -> float:
    from posttrain.reward_baselines import ANSWER_RE, format_only_reward
    if not completions:
        return 0.0
    hits = 0
    for text in completions:
        if format_only_reward(text) == 1.0 or ANSWER_RE.search(text or ""):
            hits += 1
    return hits / len(completions)


def train_ppo(
    prompts: list[dict],
    reward_fn,
    *,
    kl_coef: float = DEFAULT_KL_COEF["ppo"],
    kl_estimator: str = DEFAULT_KL_ESTIMATOR,
    kl_in_reward: bool = True,
    clip: bool = True,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    value_clip: float | None = 0.2,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    entropy_coef: float = 0.0,
    sft_coef: float = 0.0,
    sft_examples: list | None = None,
    refresh_reference_every: int = 0,
    adaptive_kl_target: float | None = None,
    reward_mode: str = "ground_truth",
    group_size: int = 2,
    max_new_tokens: int = 4,
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
    require_kl("ppo", kl_coef)
    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    if reference is None:
        reference = clone_frozen(model)
    if sft_anchor is None:
        sft_anchor = clone_frozen(reference)
    ref_before = snapshot_state(reference)
    anchor_before = snapshot_state(sft_anchor)
    value_head = nn.Linear(model.config.n_embd, 1)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(value_head.parameters()),
        lr=lr,
    )
    controller = None
    if adaptive_kl_target is not None and kl_coef > 0:
        controller = AdaptiveKLController(kl_coef, adaptive_kl_target)
    logger = logger or StepLogger()
    losses = []
    texts = [p["prompt"] for p in prompts]
    metas = [p for p in prompts]
    for step in range(steps):
        if refresh_reference_every and step > 0 and step % refresh_reference_every == 0:
            reference.load_state_dict(model.state_dict())
            for p in reference.parameters():
                p.requires_grad_(False)
        rows = generate_rollouts(
            model, tokenizer, texts, metas, group_size, max_new_tokens, temperature, seed + step,
        )
        score_rollouts(rows, reward_fn, reward_mode, seed)
        if reward_mode == "prior_only":
            _log_prior(logger, step, rows, "ppo", kl_coef)
            print(f"ppo step {step} prior_only no parameter update", flush=True)
            continue
        batch = collate_rollouts(tokenizer, rows)
        truncated = torch.tensor([r.truncated for r in rows])
        outcome = torch.tensor([r.reward for r in rows], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            old_out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            old_lp = token_logprobs(old_out.logits, batch["input_ids"]).detach()
            old_values = value_head(old_out.hidden_states[-1]).squeeze(-1).detach()
            ref_lp, _ = _ref_lp(reference, batch)
            anchor_lp, _ = _ref_lp(sft_anchor, batch)
        shift_mask = completion_mask_shifted(batch["completion_mask"])
        token_kl = kl_term(old_lp, ref_lp, kl_estimator)
        token_rewards = torch.zeros(batch["input_ids"].shape, dtype=torch.float32)
        if kl_in_reward and kl_coef:
            # Detached: the KL changes the advantage, it is not a second path
            # through log pi. The direct path is kl_in_reward=False.
            token_rewards[:, 1:] = token_rewards[:, 1:] - kl_coef * token_kl.detach() * shift_mask
        for i, row_mask in enumerate(batch["completion_mask"]):
            idx = (row_mask > 0).nonzero(as_tuple=False)
            if len(idx):
                last = int(idx[-1].item())
                token_rewards[i, last] = token_rewards[i, last] + outcome[i]
        if normalize_advantages:
            # Normalise the outcome part only after GAE, below.
            pass
        advantages = gae(token_rewards, old_values, batch["completion_mask"], gamma=gamma, lam=gae_lambda)
        returns = (advantages + old_values).detach()
        adv_for_policy = advantages.detach()
        if normalize_advantages:
            adv_for_policy = _normalize(adv_for_policy, batch["completion_mask"])
        adv_shift = adv_for_policy[:, 1:]
        ran = False
        last_parts = {"loss": 0.0}
        approx = 0.0
        ent_value = 0.0
        g = 0.0
        groups = {}
        bind = torch.zeros_like(shift_mask, dtype=torch.bool)
        for _epoch in range(epochs):
            model.train()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            logp = token_logprobs(out.logits, batch["input_ids"])
            values = value_head(out.hidden_states[-1]).squeeze(-1)
            ent = masked_entropy(out.logits, batch["completion_mask"])
            loss, last_parts, approx = policy_loss(
                logp, old_lp, ref_lp, adv_shift, shift_mask,
                kl_coef=kl_coef, kl_estimator=kl_estimator,
                kl_in_loss=not kl_in_reward,
                eps_low=eps_low, eps_high=eps_high, clip=clip, cispo=False,
                token_level=True, entropy=ent, entropy_coef=entropy_coef,
            )
            v_loss = value_loss(values, returns, old_values, clip_eps=value_clip, mask=batch["completion_mask"])
            loss = loss + v_loss
            last_parts["value"] = float(v_loss.detach())
            if sft_coef and sft_examples:
                loss = loss + sft_coef * _sft_nll(model, tokenizer, sft_examples)
            if not torch.isfinite(loss):
                raise RuntimeError(f"PPO loss is not finite at step {step}")
            if float(shift_mask.sum()) == 0:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = grad_norm(list(model.parameters()) + list(value_head.parameters()))
            groups = grad_norms_by_group(model)
            opt.step()
            ran = True
            ent_value = float(ent.detach())
            bind = bind_mask(logp, old_lp, adv_shift, eps_low, eps_high, clip)
            losses.append(float(loss.detach()))
        if not ran:
            losses.append(0.0)
        with torch.no_grad():
            anchor_kl = float(reduce_token_values(kl_term(old_lp, anchor_lp, "k3"), shift_mask, token_level=True))
        if controller is not None and approx is not None:
            kl_coef = controller.update(approx)
        completions = [r.completion for r in rows]
        proxy = sum(r.reward for r in rows) / len(rows)
        gold = sum(r.gold for r in rows) / len(rows)
        flags = _maybe_monitor(step, monitor_every, rows, proxy, gold, stop_on_hack)
        buckets = clip_fraction_by_bucket(old_lp.exp(), bind, shift_mask)
        logger.add(StepRecord(
            step=step, method="ppo", loss=losses[-1], loss_parts=last_parts,
            grad_norm=g, grad_norm_by_group=groups,
            clip_fraction_by_prob_bucket=buckets,
            approx_kl=approx, kl_to_sft=anchor_kl, entropy=ent_value,
            reward_mean=proxy, reward_std=_std([r.reward for r in rows]),
            response_length_mean=sum(sum(r.completion_mask) for r in rows) / len(rows),
            advantage_mean=float(adv_shift.mean()), advantage_std=float(adv_shift.std(unbiased=False)),
            format_compliance=_format_rate(completions), gold_metric=gold,
            style_distance_to_sft=_style_distance(completions, style_reference),
            hacking_flags=flags, kl_coef=kl_coef,
        ))
        print(
            f"ppo step {step} loss {losses[-1]:.4f} reward {proxy:.3f} gold {gold:.3f} "
            f"kl_to_sft {anchor_kl:.4f} grad_norm {g:.4f} finite True",
            flush=True,
        )
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, steps, extra={"method": "ppo", "kl_coef": kl_coef})
    return _done("ppo", losses, model, tokenizer, reference, sft_anchor, ref_before, anchor_before, logger, kl_coef)


def _ref_lp(model, batch):
    from posttrain.batching import frozen_logprobs
    return frozen_logprobs(model, batch)


def _normalize(adv, mask):
    flat = adv[mask.bool()]
    if flat.numel() == 0 or float(flat.std(unbiased=False)) == 0.0:
        return adv * mask
    centered = (adv - flat.mean()) / (flat.std(unbiased=False) + 1e-6)
    return centered * mask


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.std(unbiased=False))


def _sft_nll(model, tokenizer, examples):
    from posttrain.batching import collate, encode_example
    pairs = [encode_example(tokenizer, r["prompt"], r["completion"], 48) for r in examples[:2]]
    batch = collate(tokenizer, pairs)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return sft_nll(out.logits, batch["input_ids"], batch["completion_mask"])


def _log_prior(logger, step, rows, method, kl_coef):
    proxy = sum(r.reward for r in rows) / max(len(rows), 1)
    gold = sum((r.gold or 0.0) for r in rows) / max(len(rows), 1)
    logger.add(StepRecord(
        step=step, method=method, loss=0.0, reward_mean=proxy, gold_metric=gold,
        kl_coef=kl_coef, status="prior_only",
    ))


def _maybe_monitor(step, every, rows, proxy, gold, stop_on_hack):
    if not every:
        return []
    from posttrain.hacking_checks import monitor_step
    payload = [_row_dict(r) for r in rows]
    report = monitor_step(step, every, payload, proxy, gold, stop_on_flag=stop_on_hack)
    if not report:
        return []
    flags = [d["name"] for d in report["detectors"] if d["triggered"]]
    if report.get("proxy_gold_divergence"):
        flags.append("proxy_gold_divergence")
    return flags


def _row_dict(row: Rollout) -> dict:
    from posttrain.reward_baselines import extract_answer_text
    return {
        "prompt": row.prompt,
        "completion": row.completion,
        "reward": row.reward,
        "gold": row.gold,
        "truncated": row.truncated,
        "answer": extract_answer_text(row.completion),
        "label": row.meta.get("key"),
        "task": row.meta.get("task", "toy"),
    }


def _done(method, losses, model, tokenizer, reference, anchor, ref_before, anchor_before, logger, kl_coef):
    final = losses[-1] if losses else 0.0
    return {
        "losses": losses,
        "final_loss": final,
        "model": model,
        "tokenizer": tokenizer,
        "reference": reference,
        "sft_anchor": anchor,
        "reference_unchanged": states_equal(ref_before, snapshot_state(reference)),
        "sft_anchor_unchanged": states_equal(anchor_before, snapshot_state(anchor)),
        "logger": logger,
        "kl_coef": kl_coef,
        "status": "ran",
        "method": method,
    }


def smoke() -> dict:
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    ref = clone_frozen(sft["model"])
    anchor = clone_frozen(sft["model"])
    out = train_ppo(
        TOY_PROMPTS, toy_protocol_reward,
        model=sft["model"], tokenizer=sft["tokenizer"],
        reference=ref, sft_anchor=anchor,
        steps=2, group_size=2, max_new_tokens=4, seed=0, monitor_every=1,
        reward_mode="random_p50",
        style_reference=[r["completion"] for r in TOY_SFT],
    )
    print(
        f"ppo smoke done steps 2 final_loss {out['final_loss']:.4f} "
        f"reference_unchanged {out['reference_unchanged']}",
        flush=True,
    )
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="PPO against a frozen SFT reference.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--kl-coef", type=float, default=None)
    args = p.parse_args(argv)
    if args.kl_coef is None:
        smoke()
        return
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    ref = clone_frozen(sft["model"])
    out = train_ppo(
        TOY_PROMPTS, toy_protocol_reward,
        model=sft["model"], tokenizer=sft["tokenizer"], reference=ref,
        sft_anchor=clone_frozen(sft["model"]),
        steps=1, kl_coef=args.kl_coef, seed=0,
    )
    print(f"ppo smoke done steps 1 final_loss {out['final_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
