"""Mixed Preference Optimization (MPO): DPO + BCO (EMA reward_shift) + SFT anchor.

Follows `MPOLossFn` in `yaniguan/RL` (`nemo_rl/algorithms/loss/loss_functions.py`),
extended to seamlessly ingest BOTH:
  1. Paired preference rows: `{"prompt": ..., "chosen": ..., "rejected": ...}`
  2. Unpaired experimental outcome rows: `{"prompt": ..., "completion": ..., "label": 1 | 0}`

Why MPO matters for `labbench-posttrain`:
`data/v0` has only 274 paired `success_vs_failure` trajectories on the identical prompt,
but thousands of unpaired reaction outcomes (`yield >= 20%` vs failed) and hardware fault logs.
BCO with a persistent EMA `reward_shift` unlocks all unpaired experimental outcomes alongside
the paired DPO signal, while the SFT NLL anchor prevents DPO likelihood displacement.
"""
from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

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
from posttrain.kl import DEFAULT_KL_COEF, kl_term, warn_if_kl_disabled
from posttrain.losses import reduce_token_values, sft_nll
from posttrain.metrics import StepLogger, StepRecord, grad_norm, grad_norms_by_group
from posttrain.sft import TOY_SFT, train_sft
from posttrain.tiny import build_tiny_lm

TOY_MPO_BATCH: list[dict[str, Any]] = [
    {"prompt": "Heat the sample to", "chosen": " 37 C for 10 min.", "rejected": " 900 C.", "record_type": "paired"},
    {"prompt": "Spin the tube at", "chosen": " 3000 rpm for 5 min.", "rejected": " skip.", "record_type": "paired"},
    {"prompt": "Wash the cells with", "completion": " cold PBS twice.", "label": 1, "record_type": "unpaired"},
    {"prompt": "Transfer reagent with", "completion": " clogged tip error.", "label": 0, "record_type": "unpaired"},
]


def _prompt_to_str(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt)


def update_reward_shift(current_shift: float, reward_sum: float, reward_count: float, momentum: float = 0.95) -> float:
    """Driver-owned EMA update matching `MPOLossFn.update_reward_shift` in NeMo-RL."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("reward_shift_momentum must be in [0, 1)")
    if reward_count <= 0:
        return float(current_shift)
    batch_mean = float(reward_sum) / float(reward_count)
    return float(momentum * current_shift + (1.0 - momentum) * batch_mean)


def mpo_loss(
    *,
    logp_chosen: torch.Tensor | None = None,
    logp_rejected: torch.Tensor | None = None,
    ref_chosen: torch.Tensor | None = None,
    ref_rejected: torch.Tensor | None = None,
    logp_unpaired_pos: torch.Tensor | None = None,
    ref_unpaired_pos: torch.Tensor | None = None,
    logp_unpaired_neg: torch.Tensor | None = None,
    ref_unpaired_neg: torch.Tensor | None = None,
    sft_nll_loss: torch.Tensor | None = None,
    beta: float = 0.1,
    reward_shift: float = 0.0,
    preference_weight: float = 1.0,
    bco_weight: float = 1.0,
    sft_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute hybrid MPO objective: `w_p * L_DPO + w_q * L_BCO(shift) + w_g * L_SFT`."""
    device = None
    dtype = torch.float32
    for t in (logp_chosen, logp_unpaired_pos, logp_unpaired_neg, sft_nll_loss):
        if t is not None:
            device = t.device
            dtype = t.dtype
            break
    if device is None:
        raise ValueError("mpo_loss received an empty batch with no paired or unpaired tensors")

    zero = torch.zeros((), device=device, dtype=dtype)
    pref_loss = zero
    pref_acc = 0.0
    pos_rewards: list[torch.Tensor] = []
    neg_rewards: list[torch.Tensor] = []

    if logp_chosen is not None and logp_rejected is not None and logp_chosen.numel() > 0:
        assert ref_chosen is not None and ref_rejected is not None
        r_c = beta * (logp_chosen - ref_chosen)
        r_r = beta * (logp_rejected - ref_rejected)
        pref_loss = -F.logsigmoid(r_c - r_r).mean()
        pref_acc = float((r_c.detach() > r_r.detach()).to(torch.float32).mean())
        pos_rewards.append(r_c)
        neg_rewards.append(r_r)

    if logp_unpaired_pos is not None and logp_unpaired_pos.numel() > 0:
        assert ref_unpaired_pos is not None
        pos_rewards.append(beta * (logp_unpaired_pos - ref_unpaired_pos))

    if logp_unpaired_neg is not None and logp_unpaired_neg.numel() > 0:
        assert ref_unpaired_neg is not None
        neg_rewards.append(beta * (logp_unpaired_neg - ref_unpaired_neg))

    shift_t = torch.tensor(float(reward_shift), device=device, dtype=dtype)
    bco_loss = zero
    all_bco_detached: list[torch.Tensor] = []
    bco_correct: list[torch.Tensor] = []
    if pos_rewards:
        r_pos = torch.cat(pos_rewards, dim=0)
        bco_loss = bco_loss - F.logsigmoid(r_pos - shift_t).mean()
        all_bco_detached.append(r_pos.detach())
        bco_correct.append((r_pos.detach() > shift_t).to(torch.float32))
    if neg_rewards:
        r_neg = torch.cat(neg_rewards, dim=0)
        bco_loss = bco_loss - F.logsigmoid(-(r_neg - shift_t)).mean()
        all_bco_detached.append(r_neg.detach())
        bco_correct.append((r_neg.detach() < shift_t).to(torch.float32))

    sft_term = sft_nll_loss if sft_nll_loss is not None else zero
    total = preference_weight * pref_loss + bco_weight * bco_loss + sft_weight * sft_term

    if all_bco_detached:
        cat_bco = torch.cat(all_bco_detached, dim=0)
        bco_sum = float(cat_bco.sum())
        bco_cnt = float(cat_bco.numel())
        bco_acc = float(torch.cat(bco_correct, dim=0).mean())
    else:
        bco_sum, bco_cnt, bco_acc = 0.0, 0.0, 0.0

    stats = {
        "loss": float(total.detach()),
        "preference_loss": float(pref_loss.detach()),
        "bco_loss": float(bco_loss.detach()),
        "sft_loss": float(sft_term.detach()),
        "dpo_accuracy": pref_acc,
        "bco_accuracy": bco_acc,
        "reward_shift": float(reward_shift),
        "bco_reward_sum": bco_sum,
        "bco_reward_count": bco_cnt,
    }
    return total, stats


def _seq_scores(model, tokenizer, items: list[tuple[str, str]], average: bool, max_length: int, frozen: bool):
    if not items:
        return None, None, None, None
    encoded = [encode_example(tokenizer, p, c, max_length) for p, c in items]
    batch = collate(tokenizer, encoded, device=model_device(model))
    if frozen:
        lp, mask = frozen_logprobs(model, batch)
        logits = None
    else:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits
        from posttrain.losses import completion_mask_shifted, token_logprobs

        lp = token_logprobs(logits, batch["input_ids"])
        mask = completion_mask_shifted(batch["completion_mask"])
    seq = sequence_logprob(lp, mask, average=average)
    return seq, batch, logits, (lp, mask)


def train_mpo(
    records: list[dict[str, Any]],
    *,
    beta: float = DEFAULT_KL_COEF["dpo"],
    preference_weight: float = 1.0,
    bco_weight: float = 0.8,
    sft_weight: float = 0.2,
    reward_shift: float = 0.0,
    reward_shift_momentum: float = 0.95,
    average_logprobs: bool = False,
    steps: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_length: int = 64,
    seed: int = 0,
    model=None,
    tokenizer=None,
    reference=None,
    checkpoint_path: str | None = None,
    logger: StepLogger | None = None,
) -> dict[str, Any]:
    """Train MPO on a mixed list of paired `(chosen, rejected)` and unpaired `(completion, label)` records."""
    warn_if_kl_disabled(beta, "mpo")
    if not records:
        raise ValueError("train_mpo requires at least one record")
    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    if reference is None:
        reference = clone_frozen(model)
    ref_before = snapshot_state(reference)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    logger = logger or StepLogger()
    losses: list[float] = []
    shift_history: list[float] = [float(reward_shift)]
    cur_shift = float(reward_shift)

    for step in range(steps):
        model.train()
        sub = [records[(step * batch_size + i) % len(records)] for i in range(min(batch_size, len(records)))]
        paired = [r for r in sub if "chosen" in r and "rejected" in r]
        unpaired_pos = [r for r in sub if "completion" in r and int(r.get("label", 0)) == 1]
        unpaired_neg = [r for r in sub if "completion" in r and int(r.get("label", 0)) == 0]

        chosen_items = [(_prompt_to_str(r["prompt"]), str(r["chosen"])) for r in paired]
        rejected_items = [(_prompt_to_str(r["prompt"]), str(r["rejected"])) for r in paired]
        pos_items = [(_prompt_to_str(r["prompt"]), str(r["completion"])) for r in unpaired_pos]
        neg_items = [(_prompt_to_str(r["prompt"]), str(r["completion"])) for r in unpaired_neg]

        logp_c, c_batch, c_logits, c_tok = _seq_scores(model, tokenizer, chosen_items, average_logprobs, max_length, False)
        logp_r, _, _, _ = _seq_scores(model, tokenizer, rejected_items, average_logprobs, max_length, False)
        logp_up, up_batch, up_logits, _ = _seq_scores(model, tokenizer, pos_items, average_logprobs, max_length, False)
        logp_un, _, _, _ = _seq_scores(model, tokenizer, neg_items, average_logprobs, max_length, False)

        with torch.no_grad():
            ref_c, _, _, ref_c_tok = _seq_scores(reference, tokenizer, chosen_items, average_logprobs, max_length, True)
            ref_r, _, _, _ = _seq_scores(reference, tokenizer, rejected_items, average_logprobs, max_length, True)
            ref_up, _, _, _ = _seq_scores(reference, tokenizer, pos_items, average_logprobs, max_length, True)
            ref_un, _, _, _ = _seq_scores(reference, tokenizer, neg_items, average_logprobs, max_length, True)

        sft_terms: list[torch.Tensor] = []
        if sft_weight > 0.0:
            if c_logits is not None and c_batch is not None:
                sft_terms.append(sft_nll(c_logits, c_batch["input_ids"], c_batch["completion_mask"]))
            if up_logits is not None and up_batch is not None:
                sft_terms.append(sft_nll(up_logits, up_batch["input_ids"], up_batch["completion_mask"]))
        sft_loss_t = torch.stack(sft_terms).mean() if sft_terms else None

        loss, stats = mpo_loss(
            logp_chosen=logp_c,
            logp_rejected=logp_r,
            ref_chosen=ref_c,
            ref_rejected=ref_r,
            logp_unpaired_pos=logp_up,
            ref_unpaired_pos=ref_up,
            logp_unpaired_neg=logp_un,
            ref_unpaired_neg=ref_un,
            sft_nll_loss=sft_loss_t,
            beta=beta,
            reward_shift=cur_shift,
            preference_weight=preference_weight,
            bco_weight=bco_weight,
            sft_weight=sft_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"MPO loss is not finite at step {step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = grad_norm(model.parameters())
        groups_g = grad_norms_by_group(model)
        opt.step()

        cur_shift = update_reward_shift(
            cur_shift,
            stats["bco_reward_sum"],
            stats["bco_reward_count"],
            momentum=reward_shift_momentum,
        )
        shift_history.append(cur_shift)
        losses.append(float(loss.detach()))

        kl_sft = 0.0
        if c_tok is not None and ref_c_tok is not None:
            with torch.no_grad():
                lp_now, m_now = c_tok
                lp_ref_tok, _ = ref_c_tok
                if float(m_now.sum()) > 0:
                    kl_sft = float(reduce_token_values(kl_term(lp_now.detach(), lp_ref_tok, "k3"), m_now, token_level=True))

        logger.add(StepRecord(
            step=step,
            method="mpo",
            loss=losses[-1],
            loss_parts=stats,
            grad_norm=g,
            grad_norm_by_group=groups_g,
            kl_to_sft=kl_sft,
            kl_coef=beta,
        ))

    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, steps, extra={"method": "mpo", "beta": beta, "reward_shift": cur_shift})

    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "reward_shift": cur_shift,
        "reward_shift_history": shift_history,
        "model": model,
        "tokenizer": tokenizer,
        "reference": reference,
        "reference_unchanged": states_equal(ref_before, snapshot_state(reference)),
        "logger": logger,
        "beta": beta,
        "status": "ran",
    }


def smoke() -> dict[str, Any]:
    sft = train_sft(TOY_SFT, steps=1, seed=0)
    return train_mpo(
        TOY_MPO_BATCH,
        model=sft["model"],
        tokenizer=sft["tokenizer"],
        reference=clone_frozen(sft["model"]),
        steps=3,
        seed=0,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="MPO (DPO + BCO + SFT) offline preference trainer.")
    p.add_argument("--smoke", action="store_true")
    p.parse_args(argv)
    out = smoke()
    print(f"mpo smoke done final_loss {out['final_loss']:.4f} shift {out['reward_shift']:.5f}")


if __name__ == "__main__":
    main()
