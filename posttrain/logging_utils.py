"""Per-step training log.

The fields are the ones a reader needs in order to tell a real learning signal from a
clipping artifact, a length change, or a dead group. See docs/posttrain/losses_and_gradients.md.
"""
from __future__ import annotations

import json
import math

import torch

# Old-policy token probability. High bins are where Shao et al. §4 say the upper clip cannot fire,
# because (1 + epsilon) * p can sit above 1. Low bins can cross the upper clip after a small move.
PROB_BUCKETS = (
    (0.0, 0.01),
    (0.01, 0.1),
    (0.1, 0.5),
    (0.5, 0.9),
    (0.9, 1.01),
)


def _num(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().float().cpu().tolist()
        value = value.detach().float().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return value
    if isinstance(value, (float, int)):
        return float(value)
    return value


def layer_group(name: str) -> str:
    parts = [p for p in name.split(".") if p not in {"weight", "bias", "default"}]
    if "h" in parts:
        i = parts.index("h")
        return ".".join(parts[: i + 2])
    return ".".join(parts[:3]) or name


def grad_norms(model) -> tuple[float, dict]:
    total = 0.0
    groups: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        square = float(param.grad.detach().float().pow(2).sum().item())
        total += square
        key = layer_group(name)
        groups[key] = groups.get(key, 0.0) + square
    return math.sqrt(total), {k: math.sqrt(v) for k, v in sorted(groups.items())}


def clip_fraction_by_bucket(old_prob, was_clipped, token_mask) -> dict:
    """Fraction of masked tokens whose surrogate gradient is zero, split by old-policy probability.

    Empty buckets are null. A missing bucket is not a zero rate.
    """
    old_prob = old_prob.detach().float().reshape(-1)
    was_clipped = was_clipped.detach().bool().reshape(-1)
    token_mask = token_mask.detach().bool().reshape(-1)
    out = {}
    for lo, hi in PROB_BUCKETS:
        in_bin = token_mask & (old_prob >= lo) & (old_prob < hi)
        key = f"[{lo:.2f},{hi:.2f})"
        count = int(in_bin.sum().item())
        if count == 0:
            out[key] = None
        else:
            out[key] = float((was_clipped & in_bin).sum().item() / count)
    clipped = token_mask & was_clipped
    denom = int(token_mask.sum().item())
    out["all"] = None if denom == 0 else float(clipped.sum().item() / denom)
    return out


def clipped_token_mask(ratio, advantage, epsilon: float):
    """True where the clipped surrogate does not depend on the ratio.

    This follows the cases in Shao et al., Appendix B: the gradient is zero when the
    advantage is non-negative and the ratio is above 1+epsilon, and when the advantage
    is negative and the ratio is below 1-epsilon.
    """
    adv = advantage
    while adv.ndim < ratio.ndim:
        adv = adv.unsqueeze(-1)
    positive = adv >= 0
    return (positive & (ratio > 1.0 + epsilon)) | ((~positive) & (ratio < 1.0 - epsilon))


class StepLogger:
    def __init__(self, method: str):
        self.method = method
        self.rows: list[dict] = []

    def log(self, row: dict) -> dict:
        clean = {"method": self.method}
        for key, value in row.items():
            if isinstance(value, dict):
                clean[key] = {k: _num(v) if not isinstance(v, dict) else v for k, v in value.items()}
            else:
                clean[key] = _num(value)
        loss = clean.get("loss")
        clean["finite"] = bool(isinstance(loss, float) and math.isfinite(loss))
        self.rows.append(clean)
        short = (
            f"{self.method} step={clean.get('step')} loss={clean.get('loss')} "
            f"finite={clean['finite']}"
        )
        extra = []
        for key in ("grad_norm", "approx_kl", "entropy", "reward_mean", "clip_frac", "zero_var_frac", "margin"):
            if key in clean and clean[key] is not None:
                extra.append(f"{key}={clean[key]}")
        if extra:
            short += " " + " ".join(extra)
        print(short, flush=True)
        return clean

    def dumps(self) -> str:
        return json.dumps(self.rows, indent=2, sort_keys=True)
