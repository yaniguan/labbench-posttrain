"""SFT: masked cross-entropy on the completion. Optional LoRA.

Why the prompt is masked, and token-level vs sequence-level loss, are in
the docstring of posttrain.losses.sft_nll and in
docs/posttrain/losses_and_gradients.md.
"""
from __future__ import annotations

import argparse

import torch

from posttrain.batching import collate, encode_example, maybe_lora, trainable_count
from posttrain.checkpoint import save_checkpoint
from posttrain.losses import sft_nll
from posttrain.metrics import StepLogger, StepRecord, grad_norm, grad_norms_by_group
from posttrain.tiny import build_tiny_lm

TOY_SFT = [
    {"prompt": "Heat the sample to", "completion": " 37 C for 10 min."},
    {"prompt": "Spin the tube at", "completion": " 3000 rpm for 5 min."},
    {"prompt": "Wash the cells with", "completion": " PBS twice."},
    {"prompt": "Incubate the plate at", "completion": " 37 C overnight."},
]


def train_sft(
    examples: list[dict],
    *,
    steps: int = 20,
    lr: float = 1e-3,
    token_level: bool = True,
    use_lora: bool = False,
    seed: int = 0,
    max_length: int = 48,
    model=None,
    tokenizer=None,
    checkpoint_path: str | None = None,
    logger: StepLogger | None = None,
) -> dict:
    if model is None or tokenizer is None:
        model, tokenizer = build_tiny_lm(seed=seed)
    model = maybe_lora(model, use_lora)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    logger = logger or StepLogger()
    losses = []
    for step in range(steps):
        batch_rows = [examples[(step + i) % len(examples)] for i in range(min(2, len(examples)))]
        pairs = [encode_example(tokenizer, r["prompt"], r["completion"], max_length) for r in batch_rows]
        batch = collate(tokenizer, pairs)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = sft_nll(out.logits, batch["input_ids"], batch["completion_mask"], token_level=token_level)
        if not torch.isfinite(loss):
            raise RuntimeError(f"SFT loss is not finite at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = grad_norm(model.parameters())
        groups = grad_norms_by_group(model)
        opt.step()
        losses.append(float(loss.detach()))
        logger.add(StepRecord(
            step=step, method="sft", loss=losses[-1],
            loss_parts={"nll": losses[-1]},
            grad_norm=g, grad_norm_by_group=groups,
            kl_coef=None, status="ran",
        ))
        print(f"sft step {step} loss {losses[-1]:.4f} grad_norm {g:.4f} finite True", flush=True)
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, opt, steps, extra={"method": "sft", "token_level": token_level})
    n_train, n_all = trainable_count(model)
    return {
        "losses": losses,
        "final_loss": losses[-1],
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": opt,
        "logger": logger,
        "trainable": n_train,
        "n_parameters": n_all,
        "status": "ran",
    }


def smoke() -> dict:
    """A few Adam steps on the tiny model. This is a code check, not a result."""
    out = train_sft(TOY_SFT, steps=2, lr=1e-3, seed=0, checkpoint_path=None)
    print(f"sft smoke done steps 2 final_loss {out['final_loss']:.4f}", flush=True)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SFT on completion tokens only.")
    p.add_argument("--smoke", action="store_true", help="CPU tiny model, toy data, 2 steps.")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args(argv)
    # This entry point only runs the tiny CPU model. A real SFT job is the Colab script.
    out = train_sft(TOY_SFT, steps=args.steps, use_lora=args.lora, seed=0)
    print(
        f"sft smoke done steps {args.steps} final_loss {out['final_loss']:.4f} "
        "(tiny CPU model, not a lab result)",
        flush=True,
    )


if __name__ == "__main__":
    main()
