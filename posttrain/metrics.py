"""Per-step numbers. One row is one optimiser step.

The fields exist so a later run can tell a real gain from a format trick,
a KL drift, or a clip that only hits rare tokens. Smoke fills what it can
and leaves the rest None. None means that quantity was not computed, not
that it was zero.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class StepRecord:
    step: int
    method: str
    loss: float
    loss_parts: dict = field(default_factory=dict)
    grad_norm: float | None = None
    grad_norm_by_group: dict = field(default_factory=dict)
    clip_fraction_by_prob_bucket: dict = field(default_factory=dict)
    approx_kl: float | None = None
    kl_to_sft: float | None = None
    entropy: float | None = None
    reward_mean: float | None = None
    reward_std: float | None = None
    zero_variance_group_fraction: float | None = None
    response_length_mean: float | None = None
    advantage_mean: float | None = None
    advantage_std: float | None = None
    format_compliance: float | None = None
    gold_metric: float | None = None
    style_distance_to_sft: float | None = None
    chosen_logprob: float | None = None
    rejected_logprob: float | None = None
    reward_margin: float | None = None
    preference_accuracy: float | None = None
    likelihood_displacement: bool | None = None
    hacking_flags: list = field(default_factory=list)
    kl_coef: float | None = None
    status: str = "ran"


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def grad_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    """L2 of the gradient, grouped by the first two name parts.

    A value head and a LoRA adapter should not be averaged into the stack.
    """
    buckets: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        if "lora_" in name:
            key = "lora"
        elif name.startswith("value") or ".value" in name or name.startswith("proj"):
            key = "value_head"
        elif len(parts) >= 3 and parts[0] in {"transformer", "base_model"}:
            key = ".".join(parts[:3])
        else:
            key = parts[0]
        buckets[key] = buckets.get(key, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: v ** 0.5 for k, v in buckets.items()}


def clip_fraction_by_bucket(
    token_prob: torch.Tensor,
    bind: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Fraction of masked tokens in each probability bucket whose clip binds.

    This is the measurement behind Shao et al. Section 4: high-probability
    tokens rarely bind on the upper clip, low-probability tokens do.
    """
    from posttrain.losses import prob_buckets

    mask = mask.bool()
    out = {}
    for name, bucket in prob_buckets(token_prob).items():
        sel = mask & bucket
        n = int(sel.sum())
        out[name] = float((bind & sel).sum() / n) if n else 0.0
        out[name + "_count"] = n
    return out


class StepLogger:
    def __init__(self, path: str | Path | None = None):
        self.rows: list[StepRecord] = []
        self.path = Path(path) if path else None

    def add(self, row: StepRecord) -> None:
        self.rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row)) + "\n")

    def as_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.rows]
