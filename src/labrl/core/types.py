"""Core data contracts, reward packets, and cross-stage lineage artifacts for `labrl`."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import torch

from bioenvs.base import Reward
from posttrain.advantages import extract_reasoning_span


class SplitLeakageError(RuntimeError):
    """Raised when pipeline stages attempt to mix incompatible or overlapping dataset splits."""


@dataclass(frozen=True)
class EpistemicPrediction:
    """Typed conformal/abstention envelope exported from `labrl` (`L3`) to `lab-decide` and `lab-ask-why` (`L6`)."""

    value: float | None
    sigma: float | None = None
    interval_lo: float | None = None
    interval_hi: float | None = None
    abstain: bool = False
    rationale_spans: tuple[str, ...] = ()
    confidence_level: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "sigma": self.sigma,
            "interval_lo": self.interval_lo,
            "interval_hi": self.interval_hi,
            "abstain": self.abstain,
            "rationale_spans": list(self.rationale_spans),
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpistemicPrediction":
        return cls(
            value=float(data["value"]) if data.get("value") is not None else None,
            sigma=float(data["sigma"]) if data.get("sigma") is not None else None,
            interval_lo=float(data["interval_lo"]) if data.get("interval_lo") is not None else None,
            interval_hi=float(data["interval_hi"]) if data.get("interval_hi") is not None else None,
            abstain=bool(data.get("abstain", False)),
            rationale_spans=tuple(str(s) for s in (data.get("rationale_spans") or ())),
            confidence_level=float(data.get("confidence_level", 0.90)),
        )


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

    def to_epistemic_prediction(
        self,
        *,
        completion: str | None = None,
        value: float | None = None,
        sigma: float | None = None,
        z_score: float = 1.6448536269514722,
        confidence_level: float = 0.90,
        abstain_sigma_threshold: float | None = None,
        rationale_spans: tuple[str, ...] | list[str] | None = None,
    ) -> EpistemicPrediction:
        """Export a typed conformal/abstention envelope for `lab-decide` and `lab-ask-why` (#42)."""
        tel = self.telemetry or {}
        resolved_val = value
        if resolved_val is None:
            for k in ("normalized_value", "estimated_k_mS_cm", "estimated_observable", "pred_k", "value"):
                if k in tel and math.isfinite(float(tel[k])):
                    resolved_val = float(tel[k])
                    break
        resolved_sig = sigma
        if resolved_sig is None:
            for k in ("sigma", "normalized_sigma", "sigma_mS_cm"):
                if k in tel and math.isfinite(float(tel[k])):
                    resolved_sig = float(tel[k])
                    break
        lo = hi = None
        if resolved_val is not None and resolved_sig is not None and resolved_sig >= 0.0:
            half_width = float(z_score) * resolved_sig
            lo = resolved_val - half_width
            hi = resolved_val + half_width
        abstained = (
            (not self.parsed)
            or float(tel.get("abstained", 0.0)) > 0.5
            or (
                abstain_sigma_threshold is not None
                and resolved_sig is not None
                and resolved_sig > float(abstain_sigma_threshold)
            )
        )
        spans: tuple[str, ...] = tuple(rationale_spans) if rationale_spans is not None else ()
        if not spans and completion:
            think_txt = extract_reasoning_span(completion)
            if think_txt:
                parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", think_txt) if s.strip()]
                spans = tuple(parts)
        return EpistemicPrediction(
            value=resolved_val,
            sigma=resolved_sig,
            interval_lo=lo,
            interval_hi=hi,
            abstain=bool(abstained),
            rationale_spans=spans,
            confidence_level=float(confidence_level),
        )

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
