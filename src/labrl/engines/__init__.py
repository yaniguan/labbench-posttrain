"""Pluggable execution backends (`TorchLocalEngine` for 0-1 GPU & CI; `NeMoRLEngine` for 4-8x A100/H100 clusters)."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import yaml

from labpt_nemo_rl.config_validation import compose_recipe, validate_nemo_rl_yaml
from labpt_nemo_rl.interfaces import NEMO_RL_AVAILABLE
from labrl.core.types import RewardPacket, StageArtifact
from posttrain.batching import clone_frozen
from posttrain.distill_opd import train_tropd
from posttrain.dpo import train_dpo
from posttrain.grpo import train_grpo
from posttrain.mpo import train_mpo
from posttrain.ppo import train_ppo
from posttrain.rl_common import generate_rollouts
from posttrain.sft import train_sft
from posttrain.tiny import build_tiny_lm


class ExecutionEngine(ABC):
    """Hardware-agnostic execution backend consumed by `PostTrainPipeline`."""

    name: str = "base"

    @abstractmethod
    def run_stage(
        self,
        stage: Any,
        *,
        corpus: Any,
        env: Any,
        base_model_name: str,
        upstream: StageArtifact | None,
        output_dir: Path | None,
        seed: int = 0,
    ) -> StageArtifact:
        raise NotImplementedError


class TorchLocalEngine(ExecutionEngine):
    """Zero-dependency PyTorch backend for CPU smoke tests (`smoke=True`) and single-GPU HF+LoRA jobs."""

    name = "torch_local"

    def __init__(self, *, smoke: bool = False) -> None:
        self.smoke = bool(smoke)

    def _init_model_and_tokenizer(self, base_model_name: str, seed: int):
        if self.smoke or base_model_name in ("tiny", ""):
            return build_tiny_lm(seed=seed, n_positions=1536)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(base_model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model, tok

    def run_stage(
        self,
        stage: Any,
        *,
        corpus: Any,
        env: Any,
        base_model_name: str,
        upstream: StageArtifact | None,
        output_dir: Path | None,
        seed: int = 0,
    ) -> StageArtifact:
        method = stage.method
        split_sig = corpus.split_guard.signature
        max_len = 1024 if self.smoke else int(getattr(stage, "max_length", 1536))
        ckpt = str(output_dir / f"{stage.name}.pt") if output_dir else None

        if upstream is not None and upstream.model is not None and upstream.tokenizer is not None:
            model = copy.deepcopy(upstream.model)
            tokenizer = upstream.tokenizer
            sft_ref = upstream.sft_reference if upstream.sft_reference is not None else clone_frozen(upstream.model)
            sft_anchor_path = upstream.sft_anchor_path or upstream.checkpoint_path
        else:
            model, tokenizer = self._init_model_and_tokenizer(base_model_name, seed)
            sft_ref = clone_frozen(model)
            sft_anchor_path = None

        if method == "sft":
            rows = corpus.to_sft(split="train", limit=6 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_sft(
                rows,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                use_lora=bool(stage.use_lora),
                seed=seed,
                max_length=max_len,
                model=model,
                tokenizer=tokenizer,
                checkpoint_path=ckpt,
            )
            frozen_sft = clone_frozen(out["model"])
            return StageArtifact(
                stage_name=stage.name,
                method="sft",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=ckpt or "in_memory_sft_anchor",
                model=out["model"],
                tokenizer=out["tokenizer"],
                sft_reference=frozen_sft,
                metrics={"final_loss": out["final_loss"], "num_examples": len(rows)},
            )

        if method == "dpo":
            pairs = corpus.to_preference(split="train", limit=6 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_dpo(
                pairs,
                beta=stage.beta,
                sft_coef=stage.sft_coef,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                max_length=max_len,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                reference=sft_ref,
                checkpoint_path=ckpt,
            )
            return StageArtifact(
                stage_name=stage.name,
                method="dpo",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=sft_anchor_path,
                model=out["model"],
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics={
                    "final_loss": out["final_loss"],
                    "reference_unchanged": out["reference_unchanged"],
                    "likelihood_displacement": out["likelihood_displacement"],
                    "num_pairs": len(pairs),
                },
            )

        if method == "mpo":
            records = corpus.to_mpo(split="train", limit=6 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_mpo(
                records,
                beta=stage.beta,
                preference_weight=stage.preference_weight,
                bco_weight=stage.bco_weight,
                sft_weight=stage.sft_weight,
                reward_shift_momentum=stage.reward_shift_momentum,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                max_length=max_len,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                reference=sft_ref,
                checkpoint_path=ckpt,
            )
            return StageArtifact(
                stage_name=stage.name,
                method="mpo",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=sft_anchor_path,
                model=out["model"],
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics={
                    "final_loss": out["final_loss"],
                    "reference_unchanged": out["reference_unchanged"],
                    "reward_shift": out["reward_shift"],
                    "num_records": len(records),
                },
            )

        if method == "tropd":
            prompts = corpus.to_rl_prompts(split="train", limit=2 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_tropd(
                prompts,
                env.posttrain_reward_fn(),
                teachers=[sft_ref],
                proximal_alpha=stage.proximal_alpha,
                distill_weight=stage.distill_weight,
                task_reward_weight=stage.task_reward_weight,
                group_size=2 if self.smoke else stage.group_size,
                max_new_tokens=8 if self.smoke else stage.max_new_tokens,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
            )
            return StageArtifact(
                stage_name=stage.name,
                method="tropd",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=sft_anchor_path,
                model=out["model"],
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics={"final_loss": out["final_loss"], "reference_unchanged": out["reference_unchanged"]},
            )

        if method == "ppo":
            prompts = corpus.to_rl_prompts(split="train", limit=2 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_ppo(
                prompts,
                env.posttrain_reward_fn(),
                kl_coef=stage.kl_coef,
                kl_estimator="k3",
                kl_in_reward=True,
                eps_low=stage.clip_range[0],
                eps_high=stage.clip_range[1],
                vapo_length_adaptive_lam=stage.vapo_length_adaptive_lam,
                positive_nll_coef=stage.positive_nll_coef,
                group_size=2 if self.smoke else stage.group_size,
                max_new_tokens=8 if self.smoke else stage.max_new_tokens,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                reference=sft_ref,
                sft_anchor=clone_frozen(sft_ref),
                checkpoint_path=ckpt,
            )
            return StageArtifact(
                stage_name=stage.name,
                method="ppo",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=sft_anchor_path,
                model=out["model"],
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics={
                    "final_loss": out["final_loss"],
                    "reference_unchanged": out["reference_unchanged"],
                    "sft_anchor_unchanged": out["sft_anchor_unchanged"],
                },
            )

        if method == "grpo":
            prompts = corpus.to_rl_prompts(split="train", limit=2 if self.smoke else stage.dataset_limit, seed=seed)
            out = train_grpo(
                prompts,
                env.posttrain_reward_fn(),
                kl_coef=stage.kl_coef,
                kl_estimator="k3",
                eps_low=stage.clip_range[0],
                eps_high=stage.clip_range[1],
                dual_clip_c=3.0,
                sequence_level_ratios=stage.sequence_level_ratios,
                token_level=not stage.sequence_level_ratios,
                advantage_mode=stage.advantage,
                gdpo_weights=stage.gdpo_weights,
                dynamic_sampling=False if self.smoke else stage.dynamic_sampling,
                dynamic_sampling_max_rounds=2 if self.smoke else stage.dynamic_sampling_max_rounds,
                overlong_filtering=not self.smoke,
                stop_properly_coef=0.2,
                reasoning_length_coef=stage.reasoning_length_coef,
                overlong_buffer_length=6 if self.smoke else 192,
                overlong_buffer_penalty=0.3,
                use_is_correction=stage.use_is_correction,
                is_mode="icepop",
                group_size=4 if self.smoke else stage.group_size,
                max_new_tokens=8 if self.smoke else stage.max_new_tokens,
                steps=2 if self.smoke else stage.steps,
                lr=1e-3 if self.smoke else stage.lr,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                reference=sft_ref,
                sft_anchor=clone_frozen(sft_ref),
                checkpoint_path=ckpt,
            )
            return StageArtifact(
                stage_name=stage.name,
                method="grpo",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                checkpoint_path=ckpt,
                sft_anchor_path=sft_anchor_path,
                model=out["model"],
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics={
                    "final_loss": out["final_loss"],
                    "reference_unchanged": out["reference_unchanged"],
                    "sft_anchor_unchanged": out["sft_anchor_unchanged"],
                },
            )

        if method == "eval":
            from labrl.core.registry import ENV_REGISTRY
            from labrl.envs import make_verifiable_env

            eval_metrics: dict[str, float] = {}
            default_env = corpus.default_env_name()
            for sp in stage.splits:
                sp_prompts = corpus.to_rl_prompts(split=sp, limit=2 if self.smoke else stage.limit_per_split, seed=seed)
                sp_env = env
                if default_env in ENV_REGISTRY and (env is None or getattr(env, "name", default_env) == default_env):
                    try:
                        sp_env = make_verifiable_env(default_env, split=sp, seed=seed)
                    except TypeError:
                        sp_env = env
                rf = sp_env.posttrain_reward_fn()
                rollouts = generate_rollouts(
                    model,
                    tokenizer,
                    [p["prompt"] for p in sp_prompts],
                    sp_prompts,
                    group_size=1,
                    max_new=8 if self.smoke else stage.max_new_tokens,
                    temperature=0.0,
                    seed=seed,
                )
                pkts = [RewardPacket.from_any(rf(r.prompt, r.completion, r.meta)) for r in rollouts]
                eval_metrics[f"{sp}/scalar_reward_mean"] = sum(p.scalar for p in pkts) / max(len(pkts), 1)
                eval_metrics[f"{sp}/parse_rate"] = sum(float(p.parsed) for p in pkts) / max(len(pkts), 1)
            return StageArtifact(
                stage_name=stage.name,
                method="eval",
                base_model_name=base_model_name,
                corpus_name=corpus.name,
                split_signature=split_sig,
                sft_anchor_path=sft_anchor_path,
                model=model,
                tokenizer=tokenizer,
                sft_reference=sft_ref,
                metrics=eval_metrics,
            )

        raise ValueError(f"Unsupported stage method {method!r} on TorchLocalEngine")


class NeMoRLEngine(ExecutionEngine):
    """Multi-GPU / multi-node NeMo-RL (`yaniguan/RL`) backend compiler & launcher."""

    name = "nemo"

    def __init__(self, *, hardware_axis: str = "node_8xa100_colocated", model_axis: str = "qwen3_8b") -> None:
        self.hardware_axis = hardware_axis
        self.model_axis = model_axis

    def run_stage(
        self,
        stage: Any,
        *,
        corpus: Any,
        env: Any,
        base_model_name: str,
        upstream: StageArtifact | None,
        output_dir: Path | None,
        seed: int = 0,
    ) -> StageArtifact:
        domain_axis = "calisol_conductivity" if corpus.name == "calisol23" else "bioenvs_pqa_ord_err"
        algo_map = {
            "grpo": "gdpo_multireward" if getattr(stage, "advantage", "gdpo") == "gdpo" else "dapo",
            "mpo": "mpo_offline",
            "tropd": "tropd_distill",
        }
        algo_axis = algo_map.get(stage.method, "gdpo_multireward")
        cfg = compose_recipe(
            algo=algo_axis,
            domain=domain_axis,
            model=self.model_axis,
            hardware=self.hardware_axis,
            overrides=[f"policy.model_name={base_model_name}"],
        )
        export_cfg = {k: v for k, v in cfg.items() if k != "recipe_meta"}
        validate_nemo_rl_yaml(export_cfg, source=f"nemo_{stage.name}")

        yaml_path = None
        if output_dir:
            nemo_dir = output_dir / "nemo_configs"
            nemo_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = nemo_dir / f"{stage.name}.yaml"
            yaml_path.write_text(yaml.safe_dump(export_cfg, sort_keys=False), encoding="utf-8")

        return StageArtifact(
            stage_name=stage.name,
            method=stage.method,
            base_model_name=base_model_name,
            corpus_name=corpus.name,
            split_signature=corpus.split_guard.signature,
            status="validated_ready",
            checkpoint_path=str(yaml_path) if yaml_path else None,
            sft_anchor_path=upstream.sft_anchor_path if upstream else None,
            metrics={
                "nemo_rl_available": NEMO_RL_AVAILABLE,
                "compiled_yaml": str(yaml_path) if yaml_path else None,
                "hardware_axis": self.hardware_axis,
            },
        )


def resolve_engine(engine: str | ExecutionEngine = "auto", *, smoke: bool = False) -> ExecutionEngine:
    if isinstance(engine, ExecutionEngine):
        return engine
    if engine == "auto":
        if smoke or not NEMO_RL_AVAILABLE:
            return TorchLocalEngine(smoke=smoke)
        return NeMoRLEngine()
    if engine in ("torch", "torch_local", "local"):
        return TorchLocalEngine(smoke=smoke)
    if engine in ("nemo", "nemo_rl"):
        return NeMoRLEngine()
    raise ValueError(f"Unknown engine {engine!r}; choose 'auto', 'torch_local', or 'nemo'")


__all__ = [
    "ExecutionEngine",
    "NeMoRLEngine",
    "TorchLocalEngine",
    "resolve_engine",
]
