"""NeMo-RL & `posttrain` unified CALiSol-23 Conductivity RLVR environment.

Supports both:
  - `calisol_design`: direct ionic conductivity (`k` in mS/cm) maximization with experimental support gating.
  - `calisol_predict`: forward conductivity (`k` in mS/cm) + calibrated uncertainty (`sigma`) prediction.
"""
from __future__ import annotations

import json
from typing import Any, Literal

import torch

from bioenvs import make_env
from bioenvs.base import BioEnv, Sample
from labpt.rewards import counterfactual_consistency_reward
from labpt_nemo_rl.interfaces import EnvironmentInterface, EnvironmentReturn
from posttrain.advantages import extract_reasoning_span

CALISOL_UNIFIED_CHANNELS = ("reward/conductivity", "reward/support", "reward/validity")


def _assistant_text(message_log: list[dict[str, Any]]) -> str:
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _map_calisol_channels(task: str, comps: dict[str, float]) -> dict[str, float]:
    """Align `calisol_design` and `calisol_predict` onto 3 orthogonal GDPO channels."""
    if task == "calisol_design":
        return {
            "reward/conductivity": float(comps.get("reward/conductivity", 0.0)),
            "reward/support": float(comps.get("reward/support", 0.0)),
            "reward/validity": float(comps.get("reward/validity", 0.0)),
        }
    return {
        "reward/conductivity": float(comps.get("reward/accuracy", 0.0)),
        "reward/support": float(comps.get("reward/calibration", 0.0)),
        "reward/validity": float(comps.get("reward/validity", 0.0)),
    }


class CALiSolEnv(EnvironmentInterface[dict[str, Any]]):
    """Batched verifiable electrolyte conductivity environment for NeMo-RL and `posttrain`."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        tasks: list[str] | None = None,
        split: str = "train",
        seed: int = 0,
        reward_version: str = "v2",
        reward_mode: str = "true",
        multi_reward: bool = True,
        channel_schema: Literal["unified", "task_native"] = "unified",
        max_items_per_task: int | None = None,
    ) -> None:
        cfg = dict(cfg or {})
        self.tasks = list(cfg.get("tasks", tasks or ["calisol_design", "calisol_predict"]))
        self.split = str(cfg.get("split", split))
        self.seed = int(cfg.get("seed", seed))
        self.reward_version = str(cfg.get("reward_version", reward_version))
        self.reward_mode = str(cfg.get("reward_mode", reward_mode))
        self.multi_reward = bool(cfg.get("multi_reward", multi_reward))
        self.channel_schema = cfg.get("channel_schema", channel_schema)
        max_items = cfg.get("max_items_per_task", max_items_per_task)

        self.envs: dict[str, BioEnv] = {
            t: make_env(
                t,
                split=self.split,
                seed=self.seed,
                max_items=max_items,
                reward_mode=self.reward_mode,
                reward_version=self.reward_version,
            )
            for t in self.tasks
        }

    def build_dataset(
        self,
        n_per_task: int = 32,
        *,
        include_counterfactuals: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task_name, env in self.envs.items():
            limit = min(n_per_task, len(env))
            for i in range(limit):
                if include_counterfactuals:
                    orig, cf = env.counterfactual_pair(i)
                    pair_id = f"{task_name}:{orig.id}"
                    for variant, samp in (("orig", orig), ("cf", cf)):
                        meta = {
                            "task": task_name,
                            "id": samp.id,
                            "pair_id": pair_id,
                            "variant": variant,
                            "info": samp.info,
                        }
                        rows.append({
                            "prompt": samp.prompt,
                            "messages": [{"role": "user", "content": samp.prompt}],
                            "task_name": "calisol",
                            "metadata": meta,
                            **meta,
                        })
                else:
                    samp = env.sample(i)
                    meta = {
                        "task": task_name,
                        "id": samp.id,
                        "pair_id": None,
                        "variant": "orig",
                        "info": samp.info,
                    }
                    rows.append({
                        "prompt": samp.prompt,
                        "messages": [{"role": "user", "content": samp.prompt}],
                        "task_name": "calisol",
                        "metadata": meta,
                        **meta,
                    })
        return rows

    def posttrain_reward_fn(self):
        """Adapter `f(prompt, completion, meta) -> bioenvs.Reward` for `posttrain` PPO / GRPO / GDPO."""
        def fn(prompt: str, completion: str, meta: dict[str, Any]):
            task = str(meta.get("task", self.tasks[0]))
            info = meta.get("info", meta)
            if isinstance(info, str):
                info = json.loads(info)
            env = self.envs[task]
            r = env.reward(Sample(id=str(meta.get("id", "")), prompt=prompt, info=info), completion)
            if self.channel_schema == "unified":
                aligned = _map_calisol_channels(task, r.components)
                extra = {k: v for k, v in r.components.items() if not k.startswith("reward/")}
                return type(r)(r.value, r.parsed, {**aligned, **extra})
            return r
        return fn

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn[dict[str, Any]]:
        b = len(message_log_batch)
        observations: list[dict[str, str]] = []
        updated_meta: list[dict[str, Any]] = []
        answers: list[str | None] = []
        scalar_rewards: list[float] = []
        channel_rows: list[dict[str, float]] = []

        for idx in range(b):
            completion = _assistant_text(message_log_batch[idx])
            meta = dict(metadata[idx] or {})
            task = str(meta.get("task", self.tasks[0]))
            info = meta.get("info", meta)
            if isinstance(info, str):
                info = json.loads(info)
            env = self.envs[task]
            r = env.reward(Sample(id=str(meta.get("id", idx)), prompt="", info=info), completion)
            scalar_rewards.append(float(r.value))
            ch = _map_calisol_channels(task, r.components) if self.channel_schema == "unified" else r.gdpo_channels()
            channel_rows.append(ch)
            think_span = extract_reasoning_span(completion)
            updated_meta.append({
                **meta,
                "parsed": bool(r.parsed),
                "scalar_reward": float(r.value),
                "estimated_k_mS_cm": float(r.components.get("estimated_k_mS_cm", 0.0)),
                "think_words": len(think_span.split()),
                "channels": ch,
            })
            observations.append({"role": "environment", "content": ""})
            answers.append(completion)

        terminateds = torch.ones(b, dtype=torch.bool)
        next_stops: list[None] = [None] * b
        if self.multi_reward:
            all_keys = (
                list(CALISOL_UNIFIED_CHANNELS)
                if self.channel_schema == "unified"
                else sorted({k for r in channel_rows for k in r})
            )
            rewards_out: torch.Tensor | dict[str, torch.Tensor] = {
                k: torch.tensor([float(row.get(k, 0.0)) for row in channel_rows], dtype=torch.float32)
                for k in all_keys
            }
        else:
            rewards_out = torch.tensor(scalar_rewards, dtype=torch.float32)

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_meta,
            next_stop_strings=next_stops,
            rewards=rewards_out,
            terminateds=terminateds,
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: Any
    ) -> tuple[Any, dict[str, Any]]:
        metas = batch.get("metadata", []) if hasattr(batch, "get") else []
        valid_metas = [m for m in metas if isinstance(m, dict)]
        if not valid_metas:
            return batch, {}

        metrics: dict[str, Any] = {
            "calisol/parse_rate": sum(bool(m.get("parsed", False)) for m in valid_metas) / len(valid_metas),
            "calisol/scalar_reward_mean": sum(float(m.get("scalar_reward", 0.0)) for m in valid_metas) / len(valid_metas),
            "calisol/think_words_mean": sum(float(m.get("think_words", 0.0)) for m in valid_metas) / len(valid_metas),
        }
        design_ks = [
            float(m.get("estimated_k_mS_cm", 0.0))
            for m in valid_metas
            if m.get("task") == "calisol_design" and m.get("parsed")
        ]
        if design_ks:
            metrics["calisol/design_estimated_k_mS_cm_mean"] = sum(design_ks) / len(design_ks)

        pairs: dict[str, dict[str, float]] = {}
        for m in valid_metas:
            pid = m.get("pair_id")
            var = m.get("variant")
            if pid and var in ("orig", "cf"):
                pairs.setdefault(str(pid), {})[str(var)] = float(m.get("scalar_reward", 0.0))

        cf_scores = [
            counterfactual_consistency_reward(v["orig"], v["cf"])
            for v in pairs.values()
            if "orig" in v and "cf" in v
        ]
        if cf_scores:
            metrics["calisol/counterfactual_consistency"] = sum(cf_scores) / len(cf_scores)
        return batch, metrics
