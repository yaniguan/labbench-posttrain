"""Supervised fine-tuning: token cross-entropy on the completion only.

Why the prompt is masked
    The prompt is given. It is not something the model should be trained to copy.
    A long protocol would otherwise dominate the loss, and the gradient would spend
    its budget on reproducing the question.

Token-level vs sequence-level
    token: every completion token has the same weight, so a long answer moves the
    weights more than a short one.
    sequence: each example has the same weight. Inside a long answer, each token's
    gradient is smaller. That can favour short answers when the reduction is a mean
    of per-token losses and the target is a whole-answer reward. SFT does not have
    a reward, but the same reduction choice shows up again in GRPO.

Packing
    Not implemented. Concatenating examples into one sequence saves padding, but only
    if attention cannot cross the boundary. A packer without a block mask lets the
    answer of example 1 train on the prompt of example 2. We pad instead.

Loss vs length
    Summing the token loss (no division) makes long examples dominate even more than
    a token mean. We always divide, either by the number of completion tokens or by
    the number of sequences. See sft_nll.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from posttrain.batching import collate_labeled
from posttrain.logging_utils import StepLogger, grad_norms
from posttrain.lora import apply_lora, trainable_parameters
from posttrain.tiny import build_tiny
from posttrain.toy import SFT_PAIRS


def sft_nll(logits, labels, normalize: str = "token"):
    """Cross-entropy. labels use -100 on prompt tokens and on padding.

    logits and labels are aligned with the input ids. The shift is done here:
    logit t predicts token t+1.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100).to(per_token.dtype)
    token_sum = (per_token * mask).sum(dim=-1)
    token_count = mask.sum(dim=-1).clamp(min=1.0)
    per_sequence = token_sum / token_count
    if normalize == "token":
        # Long completions contribute more tokens, so they pull harder.
        loss = token_sum.sum() / mask.sum().clamp(min=1.0)
    elif normalize == "sequence":
        # Each example pulls equally, whatever its length.
        loss = per_sequence.mean()
    else:
        raise ValueError(f"normalize must be 'token' or 'sequence', got {normalize}")
    stats = {
        "n_completion_tokens": mask.sum().detach(),
        "mean_completion_tokens": token_count.mean().detach(),
        "per_sequence_mean": per_sequence.mean().detach(),
    }
    return loss, stats


def masked_prompt_logit_grad(model, input_ids, labels, attention_mask):
    """Gradient of the SFT loss w.r.t. logits. Prompt-target positions are exactly zero.

    Logit index t predicts token t+1. If labels[t+1] is -100, logit t is unused by the loss,
    so its gradient is 0. The last prompt logit predicts the first completion token, so that
    one prompt position is *not* zero. Prompt embeddings can still have gradient, because
    later tokens attend to them. That is context, not a copied target.
    """
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits.retain_grad()
    loss, _ = sft_nll(logits, labels, normalize="token")
    loss.backward()
    grad = logits.grad
    predict_index = torch.arange(labels.shape[1] - 1)
    target = labels[:, 1:]
    prompt_targets = target.eq(-100)
    prompt_grad = grad[:, predict_index, :][prompt_targets]
    completion_targets = target.ne(-100)
    completion_grad = grad[:, predict_index, :][completion_targets]
    return {
        "prompt_target_grad_abs_max": float(prompt_grad.abs().max().item()) if prompt_grad.numel() else 0.0,
        "completion_grad_abs_max": float(completion_grad.abs().max().item()) if completion_grad.numel() else 0.0,
        "loss": float(loss.detach().item()),
    }


def train_steps(model, tok, pairs, steps: int, lr: float, normalize: str, logger: StepLogger, use_lora: bool = False,
                lora_targets=None, max_len: int = 64, optimizer=None):
    if use_lora:
        model = apply_lora(model, target_modules=lora_targets)
    model.train()
    # A caller that is resuming passes the optimizer whose moments were loaded from the checkpoint.
    opt = optimizer or torch.optim.AdamW(trainable_parameters(model), lr=lr)
    batch = collate_labeled(pairs, tok, max_len=max_len)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss, stats = sft_nll(logits, batch["labels"], normalize=normalize)
        loss.backward()
        gnorm, groups = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        opt.step()
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"SFT loss is not finite: {value}")
        logger.log({
            "step": step,
            "loss": value,
            "grad_norm": gnorm,
            "grad_norm_groups": groups,
            "n_completion_tokens": stats["n_completion_tokens"],
            "mean_response_tokens": stats["mean_completion_tokens"],
            "normalize": normalize,
        })
    return model, opt


def run_smoke(steps: int = 2, seed: int = 0, use_lora: bool = True) -> list[dict]:
    torch.manual_seed(seed)
    model, tok = build_tiny(seed=seed)
    logger = StepLogger("sft")
    train_steps(model, tok, SFT_PAIRS, steps=steps, lr=1e-3, normalize="token", logger=logger, use_lora=use_lora)
    print(f"sft smoke done steps={steps}", flush=True)
    return logger.rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SFT on completion tokens only.")
    parser.add_argument("--smoke", action="store_true", help="CPU, tiny random model, toy strings, a few steps.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Full SFT is for Colab. Pass --smoke, or use scripts/colab/run_posttrain.py. Not run.", flush=True)
        return 0
    rows = run_smoke(steps=args.steps, use_lora=not args.no_lora)
    if not rows or not all(r["finite"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
