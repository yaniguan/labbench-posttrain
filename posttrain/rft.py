"""Rejection-sampling fine-tuning and best-of-n, built on the SFT loss.

Rejection sampling: score N completions, keep those at or above a threshold, SFT on them.
Best-of-n: keep only the highest-scoring completion of each prompt, then SFT on that.

Both are offline. They need completions you already have (or a generator). The smoke
path uses toy strings and the toy reward. It does not call a lab model.

Not the same as online GRPO. There is no importance ratio and no clip.
"""
from __future__ import annotations

import argparse

from posttrain.logging_utils import StepLogger
from posttrain.ppo import toy_reward
from posttrain.sft import train_steps
from posttrain.tiny import build_tiny
from posttrain.toy import RL_PROMPTS


def select_rejection(rows: list[dict], threshold: float) -> list[tuple[str, str]]:
    """rows need prompt, completion, reward. Keep reward >= threshold."""
    return [(r["prompt"], r["completion"]) for r in rows if float(r["reward"]) >= threshold]


def select_best_of_n(rows: list[dict]) -> list[tuple[str, str]]:
    """One completion per prompt: the one with the highest reward. Ties keep the first."""
    best: dict[str, dict] = {}
    for row in rows:
        key = row["prompt"]
        if key not in best or float(row["reward"]) > float(best[key]["reward"]):
            best[key] = row
    return [(r["prompt"], r["completion"]) for r in best.values()]


def run_smoke(steps: int = 2, seed: int = 0) -> list[dict]:
    # Fixed toy rollouts so the filter is obvious and does not need generation.
    rows = []
    for prompt in RL_PROMPTS:
        for completion in (" 37 C.", " no.", " 37 C. please"):
            score, _ = toy_reward(prompt, completion, None)
            rows.append({"prompt": prompt, "completion": completion, "reward": score})
    kept = select_rejection(rows, threshold=0.5)
    if not kept:
        kept = select_best_of_n(rows)
    best = select_best_of_n(rows)
    model, tok = build_tiny(seed=seed)
    logger = StepLogger("rft")
    train_steps(model, tok, kept, steps=steps, lr=1e-3, normalize="token", logger=logger, use_lora=False)
    print(f"rft smoke kept={len(kept)} best_of_n={len(best)} steps={steps}", flush=True)
    return logger.rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rejection sampling, then SFT.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.smoke:
        print("Full rejection-sampling training is for Colab. Pass --smoke. Not run.", flush=True)
        return 0
    rows = run_smoke(steps=args.steps)
    if not rows or not all(r["finite"] for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
