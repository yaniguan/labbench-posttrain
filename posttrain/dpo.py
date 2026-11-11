"""Direct Preference Optimization.

The loss on one pair, with sequence log-probabilities (sums of token log-probs):

    loss = -log sigmoid(beta * ((logp_pi(chosen) - logp_ref(chosen))
                                - (logp_pi(rejected) - logp_ref(rejected))))

beta
    Scales the margin. A larger beta keeps the policy closer to the reference,
    because the implicit reward is beta * log(pi / pi_ref). A smaller beta lets the
    margin grow further before the sigmoid saturates. 0.1 is the usual start.

Reference model
    A frozen copy of the policy before this phase, usually the SFT model.
    Without it, the difference of log-probs is not a reward relative to a known policy.
    Both sides can drift and the numbers stop meaning "how much did we move".

Likelihood displacement
    The loss only looks at the gap. It can fall because the rejected log-prob fell,
    while the chosen log-prob also fell. The pair is ranked correctly and both answers
    got less likely. likelihood_displacement flags that pattern. An optional SFT term
    on the chosen completion (sft_coef) pulls the chosen log-prob back up. It is off
    by default.

Length
    A sum of token log-probs is more negative for a longer answer, so a long chosen
    answer is harder to prefer than a short one. length_normalize divides by the
    number of completion tokens. That removes the pure length penalty and can hide a
    real quality difference that lives in extra tokens. Default is the sum, which is
    the published DPO loss.
"""
from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn.functional as F

from posttrain.batching import completion_logprob_mask, encode_pair, pad_right
from posttrain.logging_utils import StepLogger, grad_norms
from posttrain.lora import apply_lora, trainable_parameters
from posttrain.tiny import build_tiny
from posttrain.toy import DPO_PAIRS


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1, sft_coef: float = 0.0,
             chosen_nll=None):
    """Mean DPO loss. Log-probs are tensors of shape [batch]."""
    policy_gap = pi_chosen - pi_rejected
    ref_gap = ref_chosen - ref_rejected
    margin = policy_gap - ref_gap
    # -log sigmoid(beta * margin). The sigmoid weight on the gradient is sigma(-beta * margin).
    loss = -F.logsigmoid(beta * margin)
    if sft_coef and chosen_nll is not None:
        loss = loss + sft_coef * chosen_nll
    stats = {
        "margin": margin.detach(),
        "accuracy": (margin > 0).to(margin.dtype).mean().detach(),
        "pi_chosen": pi_chosen.detach(),
        "pi_rejected": pi_rejected.detach(),
    }
    return loss.mean(), stats


def dpo_logit_grad_weight(margin, beta: float):
    """d(loss)/d(margin) = -beta * sigmoid(-beta * margin), for one pair before the batch mean.

    Large positive margin (policy already prefers chosen): the weight is near 0, so the pair
    stops pushing. Negative margin: the weight is near beta, so the pair pushes hard.
    """
    return -beta * torch.sigmoid(-beta * margin)


def likelihood_displacement(pi_chosen_before, pi_rejected_before, pi_chosen_after, pi_rejected_after, tol: float = 0.0):
    """True when both log-probs fell and the chosen-minus-rejected gap still grew.

    Numbers are plain floats (means over the batch, or one pair).
    """
    both_fell = (pi_chosen_after < pi_chosen_before - tol) and (pi_rejected_after < pi_rejected_before - tol)
    gap_before = pi_chosen_before - pi_rejected_before
    gap_after = pi_chosen_after - pi_rejected_after
    return bool(both_fell and gap_after > gap_before + tol)


def _sequence_logprobs(model, input_ids, attention_mask, prompt_lens, length_normalize: bool):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_logprob_mask(prompt_lens, attention_mask)
    total = (token_logp * mask).sum(dim=-1)
    if length_normalize:
        total = total / mask.sum(dim=-1).clamp(min=1.0)
    # NLL of the completion, for the optional SFT term. Always a mean over tokens, so beta and
    # the SFT term are not on wildly different scales when length_normalize is off.
    nll = -(token_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return total, nll


def _side_batch(pairs, tok, side: str, max_len: int):
    encoded = [encode_pair(tok, row["prompt"], row[side], max_len) for row in pairs]
    ids, mask = pad_right([e[0] for e in encoded], tok.pad_token_id)
    width = ids.shape[1]
    return ids, mask, [e[2] for e in encoded]


def train_steps(model, ref, tok, pairs, steps: int, lr: float, beta: float, sft_coef: float, length_normalize: bool,
                logger: StepLogger, use_lora: bool = False, lora_targets=None, max_len: int = 64, optimizer=None):
    if use_lora:
        model = apply_lora(model, target_modules=lora_targets)
    model.train()
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    opt = optimizer or torch.optim.AdamW(trainable_parameters(model), lr=lr)
    before = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        chosen_ids, chosen_mask, chosen_prompt = _side_batch(pairs, tok, "chosen", max_len)
        rejected_ids, rejected_mask, rejected_prompt = _side_batch(pairs, tok, "rejected", max_len)
        pi_c, nll_c = _sequence_logprobs(model, chosen_ids, chosen_mask, chosen_prompt, length_normalize)
        pi_r, _ = _sequence_logprobs(model, rejected_ids, rejected_mask, rejected_prompt, length_normalize)
        with torch.no_grad():
            ref_c, _ = _sequence_logprobs(ref, chosen_ids, chosen_mask, chosen_prompt, length_normalize)
            ref_r, _ = _sequence_logprobs(ref, rejected_ids, rejected_mask, rejected_prompt, length_normalize)
        loss, stats = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta, sft_coef=sft_coef, chosen_nll=nll_c)
        loss.backward()
        gnorm, groups = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        mean_c = float(stats["pi_chosen"].mean().item())
        mean_r = float(stats["pi_rejected"].mean().item())
        opt.step()
        displaced = False
        if before is not None:
            displaced = likelihood_displacement(before[0], before[1], mean_c, mean_r)
        before = (mean_c, mean_r)
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"DPO loss is not finite: {value}")
        logger.log({
            "step": step,
            "loss": value,
            "grad_norm": gnorm,
            "grad_norm_groups": groups,
            "margin": float(stats["margin"].mean().item()),
            "accuracy": float(stats["accuracy"].item()),
            "pi_chosen": mean_c,
            "pi_rejected": mean_r,
            "likelihood_displacement": float(displaced),
            "beta": beta,
        })
    return model, opt


def run_smoke(steps: int = 2, seed: int = 0, use_lora: bool = False) -> list[dict]:
    torch.manual_seed(seed)
    model, tok = build_tiny(seed=seed)
    ref = copy.deepcopy(model)
    logger = StepLogger("dpo")
    train_steps(
        model, ref, tok, DPO_PAIRS, steps=steps, lr=1e-3, beta=0.1, sft_coef=0.0,
        length_normalize=False, logger=logger, use_lora=use_lora,
    )
    print(f"dpo smoke done steps={steps}", flush=True)
    return logger.rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DPO on chosen/rejected completions.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Full DPO is for Colab. Pass --smoke, or use scripts/colab/run_posttrain.py. Not run.", flush=True)
        return 0
    rows = run_smoke(steps=args.steps, use_lora=args.lora)
    if not rows or not all(r["finite"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
