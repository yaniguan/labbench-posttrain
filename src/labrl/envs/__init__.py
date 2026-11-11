"""Verifiable scientific environment base class & dual-backend adapter registry (`labrl.envs`)."""
from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any

import torch

from bioenvs import make_env as make_bioenv
from bioenvs.base import Sample
from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv
from labpt_nemo_rl.interfaces import EnvironmentInterface, EnvironmentReturn
from labrl.core.registry import ENV_REGISTRY, register_env
from labrl.core.types import RewardPacket


class VerifiableEnv(EnvironmentInterface[dict[str, Any]]):
    """Subclass this in ~20 lines to turn any lab measurement/observable into an RLVR environment.

    Automatically exposes:
      - `posttrain_reward_fn()` for `TorchLocalEngine` (`PPOStage` / `GRPOStage` / `TROPDStage`)
      - `step(message_log_batch, metadata) -> EnvironmentReturn` for `NeMoRLEngine` (`vLLM` + Ray)
    """

    name: str = "custom_verifiable_env"
    multi_reward: bool = True

    @abstractmethod
    def score_observable(self, info: dict[str, Any], completion: str) -> RewardPacket:
        """Compute scalar + multi-channel verifiable reward from ground-truth `info` and model `completion`."""
        raise NotImplementedError

    def posttrain_reward_fn(self):
        def _fn(prompt: str, completion: str, meta: dict[str, Any]):
            info = meta.get("info", meta)
            if isinstance(info, str):
                info = json.loads(info)
            pkt = self.score_observable(info, completion)
            return pkt.to_bioenv_reward()

        return _fn

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn[dict[str, Any]]:
        b = len(message_log_batch)
        packets: list[RewardPacket] = []
        observations: list[dict[str, str]] = []
        updated_meta: list[dict[str, Any]] = []
        answers: list[str | None] = []

        for idx in range(b):
            msgs = message_log_batch[idx]
            comp = ""
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    comp = str(m.get("content", ""))
                    break
            meta = dict(metadata[idx] or {})
            info = meta.get("info", meta)
            if isinstance(info, str):
                info = json.loads(info)
            pkt = self.score_observable(info, comp)
            packets.append(pkt)
            observations.append({"role": "environment", "content": ""})
            updated_meta.append({
                **meta,
                "parsed": pkt.parsed,
                "scalar_reward": pkt.scalar,
                "channels": pkt.normalized_channels(),
                **pkt.telemetry,
            })
            answers.append(comp)

        rewards_out = RewardPacket.batch_to_nemo_tensors(packets, multi_reward=self.multi_reward)
        return EnvironmentReturn(
            observations=observations,
            metadata=updated_meta,
            next_stop_strings=[None] * b,
            rewards=rewards_out,
            terminateds=torch.ones(b, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(self, batch: Any) -> tuple[Any, dict[str, Any]]:
        return batch, {}


class _SingleBioEnvWrapper(VerifiableEnv):
    """Adapts any single `bioenvs.BioEnv` (`pqa`, `ord`, `err`, `gen`, `calisol_design`, `calisol_predict`) into `VerifiableEnv`."""

    def __init__(self, task_name: str, *, split: str = "train", seed: int = 0, **kw: Any) -> None:
        self.name = task_name
        self._env = make_bioenv(task_name, split=split, seed=seed, **kw)

    def score_observable(self, info: dict[str, Any], completion: str) -> RewardPacket:
        r = self._env.reward(Sample(id=str(info.get("id", "")), prompt="", info=info), completion)
        return RewardPacket.from_any(r)


# Register built-in batched and single-task environments
register_env("calisol_conductivity")(CALiSolEnv)
register_env("bioprobench")(BioProBenchEnv)
for _tname in ("calisol_design", "calisol_predict", "pqa", "ord", "err", "gen"):
    register_env(_tname)(lambda split="train", seed=0, _t=_tname, **kw: _SingleBioEnvWrapper(_t, split=split, seed=seed, **kw))


def make_verifiable_env(name_or_obj: str | Any, **kwargs: Any) -> Any:
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in ENV_REGISTRY:
        raise KeyError(f"Unknown verifiable environment {name_or_obj!r}; registered: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[name_or_obj](**kwargs)


__all__ = [
    "BioProBenchEnv",
    "CALiSolEnv",
    "VerifiableEnv",
    "make_verifiable_env",
]
