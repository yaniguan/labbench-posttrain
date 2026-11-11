"""DPO. The reference is the SFT checkpoint. beta is the KL weight.

loss = -log sigmoid(beta * ((log pi(c)-log ref(c)) - (log pi(r)-log ref(r))))

beta 0 is an ablation: the loss becomes the constant -log(0.5) and the
gradient is 0. The trainer prints a warning. See docs/posttrain/kl_anchor.md.
"""
from __future__ import annotations

import argparse

import torch

from posttrain.batching import (
    clone_frozen,
    collate,
    encode_example,
    forward_logprobs,
    frozen_logprobs,
    model_device,
    sequence_logprob,
    snapshot_state,
    states_equal,
)
from posttrain.checkpoint import save_checkpoint
from posttrain.kl import DEFAULT_KL_COEF, warn_if_kl_disabled
from posttrain.losses import dpo_loss, likelihood_displacement, sft_nll
from posttrain.metrics import StepLogger, StepRecord, grad_norm, grad_norms_by_group
from posttrain.sft import TOY_SFT, train_sft
from posttrain.tiny import build_tiny_lm

TOY_PAIRS = [
    {"prompt": "Heat the sample to", "chosen": " 37 C for 10 min.", "rejected": " unknown."},
    {"prompt": "Spin the tube at", "chosen": " 3000 rpm for 5 min.", "rejected": " no."},
    {"prompt": "Wash the cells with", "chosen": " PBS twice.", "rejected": " skip."},
    {"prompt": "Incubate the plate at", "chosen": " 37 C overnight.", "rejected": " maybe."},
]


def _pair_logprobs(model, tokenizer, pairs, average: bool, max_length: int, frozen: bool):
    chosen = [encode_example(tokenizer, p["prompt"], p["chosen"], max_length) for p in pairs]
    rejected = [encode_example(tokenizer, p["prompt"], p["rejected"], max_length) for p in pairs]
    batch = collate(tokenizer, chosen + rejected, device=model_device(model))
    if frozen:
        lp, mask = frozen_logprobs(model, batch)
    else:
        _, lp, mask = forward_logprobs(model, batch)
    seq = sequence_logprob(lp, mask, average=average)
    n = len(pairs)
    return seq[:n], seq[n:], batch, lp, mask


def train_dpo(
    pairs: list[dict],
    *,
    beta: float = DEFAULT_KL_COEF["dpo"],
    steps: int = 20,
    lr: float = 1e-3,
    average_logprobs: bool = False,
    sft_coef: float = 0.0,
    seed: int = 0,
    max_length: int = 48,
    model=None,
    tokenizer=None,
    reference=None,
    checkpoint_path: str | None = None,
    logger: StepLogger | None = None,
) -> dict:
    """Train DPO.

    `reference` must be the SFT model when you have one. If it is omitted we
    freeze a copy of `model` at the start of this call. That copy is the
    anchor. Passing the raw pretrained model here is the wrong default for a
    lab: the scientist's format lives in the SFT checkpoint, and DPO's beta
    pulls toward whatever reference you hand it.
    """
    warn_if_kl_disabled(beta, "dpo")
    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    if reference is None:
        reference = clone_frozen(model)
    ref_before = snapshot_state(reference)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    logger = logger or StepLogger()
    losses = []
    probe_pairs = pairs[: min(4, len(pairs))]
    with torch.no_grad():
        probe_c_start_t, probe_r_start_t, _, _, _ = _pair_logprobs(
            model, tokenizer, probe_pairs, average_logprobs, max_length, frozen=True,
        )
        first_chosen = float(probe_c_start_t.mean())
        first_rejected = float(probe_r_start_t.mean())
    last_chosen = first_chosen
    last_rejected = first_rejected
    for step in range(steps):
        batch_pairs = [pairs[(step + i) % len(pairs)] for i in range(min(2, len(pairs)))]
        n_bp = len(batch_pairs)
        logp_c, logp_r, chosen_batch, _, _ = _pair_logprobs(
            model, tokenizer, batch_pairs, average_logprobs, max_length, frozen=False,
        )
        with torch.no_grad():
            ref_c, ref_r, _, _, _ = _pair_logprobs(
                reference, tokenizer, batch_pairs, average_logprobs, max_length, frozen=True,
            )
        loss, stats = dpo_loss(logp_c, logp_r, ref_c, ref_r, beta=beta)
        if sft_coef:
            # A second anchor: NLL on the chosen completion, the text the
            # scientist actually wrote. beta already pulls toward the SFT
            # model; this term pulls toward the SFT data.
            out = model(input_ids=chosen_batch["input_ids"][:n_bp],
                        attention_mask=chosen_batch["attention_mask"][:n_bp])
            nll = sft_nll(
                out.logits,
                chosen_batch["input_ids"][:n_bp],
                chosen_batch["completion_mask"][:n_bp],
            )
            loss = loss + sft_coef * nll
            stats = {**stats, "sft_nll": float(nll.detach())}
        if not torch.isfinite(loss):
            raise RuntimeError(f"DPO loss is not finite at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = grad_norm(model.parameters())
        groups = grad_norms_by_group(model)
        opt.step()
        losses.append(float(loss.detach()))
        displaced = None
        if step == steps - 1:
            with torch.no_grad():
                probe_c_end_t, probe_r_end_t, _, _, _ = _pair_logprobs(
                    model, tokenizer, probe_pairs, average_logprobs, max_length, frozen=True,
                )
                last_chosen = float(probe_c_end_t.mean())
                last_rejected = float(probe_r_end_t.mean())
            displaced = likelihood_displacement(first_chosen, last_chosen, first_rejected, last_rejected)
        # KL to the SFT reference on the CHOSEN completions only (excluding rejected half).
        with torch.no_grad():
            from posttrain.kl import kl_k3
            from posttrain.losses import reduce_token_values
            _, _, batch, lp, mask = _pair_logprobs(
                model, tokenizer, batch_pairs, average=False, max_length=max_length, frozen=True,
            )
            ref_lp, _ = frozen_logprobs(reference, batch)
            token_kl = reduce_token_values(kl_k3(lp[:n_bp], ref_lp[:n_bp]), mask[:n_bp], token_level=True)
            kl_value = float(token_kl)
        logger.add(StepRecord(
            step=step, method="dpo", loss=losses[-1],
            loss_parts={"dpo": losses[-1], **{k: v for k, v in stats.items() if k == "sft_nll"}},
            grad_norm=g, grad_norm_by_group=groups,
            chosen_logprob=stats["chosen_logprob"],
            rejected_logprob=stats["rejected_logprob"],
            reward_margin=stats["reward_margin"],
            preference_accuracy=stats["accuracy"],
            likelihood_displacement=displaced,
            kl_to_sft=kl_value, approx_kl=kl_value, kl_coef=beta,
        ))
        print(
            f"dpo step {step} loss {losses[-1]:.4f} margin {stats['margin']:.4f} "
            f"acc {stats['accuracy']:.2f} chosen_lp {stats['chosen_logprob']:.3f} "
            f"rejected_lp {stats['rejected_logprob']:.3f} kl_to_sft {kl_value:.4f} finite True",
            flush=True,
        )
    displaced = likelihood_displacement(first_chosen, last_chosen, first_rejected, last_rejected)
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, steps, extra={"method": "dpo", "beta": beta})
    return {
        "losses": losses,
        "final_loss": losses[-1],
        "model": model,
        "tokenizer": tokenizer,
        "reference": reference,
        "reference_unchanged": states_equal(ref_before, snapshot_state(reference)),
        "likelihood_displacement": displaced,
        "chosen_logprob_start": first_chosen,
        "chosen_logprob_end": last_chosen,
        "rejected_logprob_start": first_rejected,
        "rejected_logprob_end": last_rejected,
        "logger": logger,
        "beta": beta,
        "status": "ran",
    }


def smoke() -> dict:
    """SFT for one step, then DPO against that frozen checkpoint."""
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    ref = clone_frozen(sft["model"])
    out = train_dpo(TOY_PAIRS, model=sft["model"], tokenizer=sft["tokenizer"], reference=ref, steps=2, seed=0)
    print(
        f"dpo smoke done steps 2 final_loss {out['final_loss']:.4f} "
        f"reference_unchanged {out['reference_unchanged']} "
        f"likelihood_displacement {out['likelihood_displacement']}",
        flush=True,
    )
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="DPO against a frozen SFT reference.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--beta", type=float, default=None)
    args = p.parse_args(argv)
    if args.beta is None:
        out = smoke()
    else:
        # Still the tiny CPU path. A non-default beta is how the ablation is invoked.
        sft = train_sft(TOY_SFT, steps=1, seed=0)
        ref = clone_frozen(sft["model"])
        out = train_dpo(
            TOY_PAIRS, model=sft["model"], tokenizer=sft["tokenizer"],
            reference=ref, steps=2, beta=args.beta, seed=0,
        )
    print(f"dpo smoke done steps 2 final_loss {out['final_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
