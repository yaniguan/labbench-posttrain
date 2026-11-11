"""One reward door: reward(prompt, completion, meta) -> (float, diagnostics).

The float is what the trainer optimises. The dict is for logs and for the hacking checks.
Ground truth always calls the environment's own score(), never a learned reward model.

BioProBench is CC BY-NC 4.0. This module only reads the existing bioenvs wrappers.
"""
from __future__ import annotations

from bioenvs import make_env
from bioenvs.base import Sample

_ENVS: dict[tuple, object] = {}


def get_env(name: str, split: str = "train", seed: int = 0):
    key = (name, split, int(seed))
    if key not in _ENVS:
        _ENVS[key] = make_env(name, split=split, seed=seed)
    return _ENVS[key]


def ground_truth(prompt: str, completion: str, meta: dict | None) -> tuple[float, dict]:
    """The environment's checked reward. meta needs env and info (the sample's hidden fields)."""
    meta = meta or {}
    name = meta.get("env")
    info = meta.get("info")
    if not name or info is None:
        raise KeyError("ground_truth reward needs meta['env'] and meta['info']")
    env = get_env(name, meta.get("split", "train"), meta.get("seed", 0))
    sample = Sample(id=str(info.get("id", "")), prompt=prompt, info=info)
    scored = env.score(sample, completion)
    diag = {
        "reward_name": "ground_truth",
        "parsed": bool(scored.parsed),
        "env": name,
        **{k: float(v) for k, v in scored.components.items() if isinstance(v, (int, float))},
    }
    return float(scored.value), diag


def reward(prompt: str, completion: str, meta: dict | None = None) -> tuple[float, dict]:
    """Swap the reward with meta['reward_name']. Default is the environment's true score.

    prior_only still reports the true score. The trainer is what skips the update.
    A prior-only run is the base model, not a constant reward.
    """
    meta = dict(meta or {})
    name = meta.get("reward_name", "ground_truth")
    if name in ("ground_truth", "prior_only"):
        value, diag = ground_truth(prompt, completion, meta)
        diag["reward_name"] = name
        diag["prior_only"] = name == "prior_only"
        return value, diag
    from posttrain.reward_baselines import compute_baseline

    return compute_baseline(name, prompt, completion, meta)


def ensemble(prompt: str, completion: str, meta: dict, fns: list) -> tuple[float, dict]:
    """Mean of several checkers. Disagreement is part of the diagnostic.

    This is a mean, not a learned ensemble. If two checkers disagree, do not treat the mean as truth.
    """
    parts = []
    diags = []
    for fn in fns:
        value, diag = fn(prompt, completion, meta)
        parts.append(float(value))
        diags.append(diag)
    spread = max(parts) - min(parts) if parts else 0.0
    return sum(parts) / len(parts), {"reward_name": "ensemble", "parts": parts, "disagreement": spread, "n": len(parts)}
