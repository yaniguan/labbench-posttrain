"""Core data contracts, reward packets, and cross-stage lineage artifacts for `labrl`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from bioenvs.base import Reward


class SplitLeakageError(RuntimeError):
    """Raised when pipeline stages attempt to mix incompatible or overlapping dataset splits."""


@dataclass(frozen=True)
class RewardPacket:
    """Unified verifiable reward contract across scalar PPO, multi-channel GDPO, and NeMo-RL."""

    scalar: float
    parsed: bool = True
    channels: dict[str, float] = field(default_factory=dict)
    telemetry: dict[str, float] = field(default_factory=dict)

    def normalized_channels(self) -> dict[str, float]:
        if self.channels:
            return {
                (k if k.startswith("reward/") else f"reward/{k}"): float(v)
                for k, v in self.channels.items()
            }
        return {
            "reward/score": float(self.scalar),
            "reward/parsed": float(self.parsed),
        }

    def to_bioenv_reward(self) -> Reward:
        return Reward(
            value=float(self.scalar),
            parsed=bool(self.parsed),
            components={**self.normalized_channels(), **self.telemetry},
        )

    @classmethod
    def from_any(cls, obj: Any) -> "RewardPacket":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Reward):
            ch = obj.gdpo_channels()
            tel = {k: float(v) for k, v in obj.components.items() if not k.startswith("reward/")}
            return cls(scalar=float(obj.value), parsed=bool(obj.parsed), channels=ch, telemetry=tel)
        val = float(getattr(obj, "value", obj if isinstance(obj, (int, float)) else 0.0))
        comps = dict(getattr(obj, "components", {}) or {})
        parsed = bool(getattr(obj, "parsed", True))
        ch = {k: float(v) for k, v in comps.items() if k.startswith("reward/")}
        tel = {k: float(v) for k, v in comps.items() if not k.startswith("reward/")}
        return cls(scalar=val, parsed=parsed, channels=ch, telemetry=tel)

    @staticmethod
    def batch_to_nemo_tensors(packets: list["RewardPacket"], *, multi_reward: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        if not multi_reward:
            return torch.tensor([float(p.scalar) for p in packets], dtype=torch.float32)
        rows = [p.normalized_channels() for p in packets]
        keys = sorted({k for r in rows for k in r})
        return {
            k: torch.tensor([float(r.get(k, 0.0)) for r in rows], dtype=torch.float32)
            for k in keys
        }


@dataclass
class StageArtifact:
    """Immutable-by-convention stage handoff token carrying model handles and frozen SFT anchors."""

    stage_name: str
    method: str
    base_model_name: str
    corpus_name: str
    split_signature: str
    status: str = "ran"
    checkpoint_path: str | None = None
    sft_anchor_path: str | None = None
    model: Any | None = field(default=None, repr=False)
    tokenizer: Any | None = field(default=None, repr=False)
    sft_reference: Any | None = field(default=None, repr=False)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "method": self.method,
            "base_model_name": self.base_model_name,
            "corpus_name": self.corpus_name,
            "split_signature": self.split_signature,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
            "sft_anchor_path": self.sft_anchor_path,
            "metrics": dict(self.metrics),
        }


@dataclass
class PipelineManifest:
    """Full-lifecycle execution lineage produced by `PostTrainPipeline.run()`."""

    corpus_name: str
    env_name: str
    base_model_name: str
    engine_name: str
    split_signature: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    evaluations: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: dict[str, StageArtifact] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_name": self.corpus_name,
            "env_name": self.env_name,
            "base_model_name": self.base_model_name,
            "engine_name": self.engine_name,
            "split_signature": self.split_signature,
            "stages": list(self.stages),
            "evaluations": dict(self.evaluations),
        }
