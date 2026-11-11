"""Reward interface and the BioProBench adapters.

reward(prompt, completion, meta) -> RewardOutput

`value` is the scalar the trainer optimises. `components` are named terms
with the `reward/` prefix from NeMo `docs/guides/environments.md`, so GDPO
can normalise them separately. `diagnostics` is for logs, not for the loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardOutput:
    value: float
    components: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


class RewardFunction:
    """Callable reward. Subclass or pass a function to `from_callable`."""

    def __call__(self, prompt: str, completion: str, meta: dict) -> RewardOutput:
        raise NotImplementedError


def from_callable(fn) -> RewardFunction:
    class _Wrap(RewardFunction):
        def __call__(self, prompt, completion, meta):
            out = fn(prompt, completion, meta)
            if isinstance(out, RewardOutput):
                return out
            value, diagnostics = out if isinstance(out, tuple) else (out, {})
            return RewardOutput(float(value), {"reward/score": float(value)}, diagnostics)

    return _Wrap()


class BioEnvReward(RewardFunction):
    """Wrap one bioenvs environment. The scalar is the code-checked score.

    `mode` is passed to the env (`true`, `random`, `format`, `incorrect`).
    The ladder in reward_baselines.py is the wider set. This class is the
    ground-truth checker, so the default mode is `true`.
    """

    def __init__(self, name: str, split: str = "train", seed: int = 0, max_items: int | None = None, mode: str = "true"):
        from bioenvs import make_env
        from bioenvs.base import Sample

        self._Sample = Sample
        self.env = make_env(name, split=split, seed=seed, max_items=max_items, reward_mode=mode)
        self.name = name

    def __call__(self, prompt: str, completion: str, meta: dict) -> RewardOutput:
        info = dict(meta or {})
        sample = self._Sample(id=str(info.get("id", "")), prompt=prompt, info=info)
        scored = self.env.score(sample, completion)
        components = {f"reward/{k}": float(v) for k, v in scored.components.items()}
        return RewardOutput(
            float(scored.value),
            components,
            {"parsed": bool(scored.parsed), "env": self.name},
        )


def toy_protocol_reward(prompt: str, completion: str, meta: dict) -> RewardOutput:
    """A checker a tiny CPU model can sometimes hit. Not a lab reward.

    meta['key'] is a short string that must appear. Used by --smoke only.
    """
    key = str((meta or {}).get("key", "37"))
    hit = key in (completion or "")
    return RewardOutput(1.0 if hit else 0.0, {"reward/hit": 1.0 if hit else 0.0}, {"parsed": hit, "key": key})
