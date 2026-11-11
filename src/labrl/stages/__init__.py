"""Declarative post-training stage specifications (`SFTStage` -> `DPO/MPO/TROPD` -> `PPO/GRPO` -> `EvalStage`)."""
from __future__ import annotations

from dataclasses import dataclass, field

from labrl.core.registry import register_stage


@dataclass
class Stage:
    name: str = "stage"
    method: str = "base"


@register_stage("sft")
@dataclass
class SFTStage(Stage):
    name: str = "sft"
    method: str = "sft"
    steps: int = 200
    lr: float = 2e-5
    use_lora: bool = False
    max_length: int = 1536
    dataset_limit: int = 256


@register_stage("dpo")
@dataclass
class DPOStage(Stage):
    name: str = "dpo"
    method: str = "dpo"
    steps: int = 150
    lr: float = 5e-6
    beta: float = 0.10
    sft_coef: float = 0.10
    max_length: int = 1536
    dataset_limit: int = 256


@register_stage("mpo")
@dataclass
class MPOStage(Stage):
    name: str = "mpo"
    method: str = "mpo"
    steps: int = 150
    lr: float = 5e-6
    beta: float = 0.10
    preference_weight: float = 1.0
    bco_weight: float = 0.8
    sft_weight: float = 0.2
    reward_shift_momentum: float = 0.95
    max_length: int = 1536
    dataset_limit: int = 256


@register_stage("tropd")
@dataclass
class TROPDStage(Stage):
    name: str = "tropd"
    method: str = "tropd"
    steps: int = 100
    lr: float = 1e-6
    proximal_alpha: float = 0.50
    distill_weight: float = 0.60
    task_reward_weight: float = 0.40
    group_size: int = 4
    max_new_tokens: int = 256
    dataset_limit: int = 32


@register_stage("ppo")
@dataclass
class PPOStage(Stage):
    name: str = "ppo"
    method: str = "ppo"
    steps: int = 100
    lr: float = 1e-6
    kl_coef: float = 0.02
    clip_range: tuple[float, float] = (0.20, 0.20)
    vapo_length_adaptive_lam: bool = True
    positive_nll_coef: float = 0.05
    group_size: int = 4
    max_new_tokens: int = 256
    dataset_limit: int = 32


@register_stage("grpo")
@dataclass
class GRPOStage(Stage):
    name: str = "grpo"
    method: str = "grpo"
    steps: int = 120
    lr: float = 1e-6
    kl_coef: float = 0.01
    clip_range: tuple[float, float] = (0.20, 0.28)
    advantage: str = "gdpo"
    gdpo_weights: dict[str, float] | None = field(
        default_factory=lambda: {
            "reward/conductivity": 1.2,
            "reward/support": 0.8,
            "reward/validity": 0.5,
            "reward/accuracy": 1.0,
            "reward/structure": 0.6,
            "reward/precision": 0.8,
        }
    )
    dynamic_sampling: bool = True
    dynamic_sampling_max_rounds: int = 4
    reasoning_length_coef: float = 0.05
    sequence_level_ratios: bool = False
    use_is_correction: bool = False
    group_size: int = 8
    max_new_tokens: int = 256
    dataset_limit: int = 32


@register_stage("eval")
@dataclass
class EvalStage(Stage):
    name: str = "eval"
    method: str = "eval"
    splits: tuple[str, ...] = ("test",)
    limit_per_split: int = 16
    max_new_tokens: int = 256


__all__ = [
    "DPOStage",
    "EvalStage",
    "GRPOStage",
    "MPOStage",
    "PPOStage",
    "SFTStage",
    "Stage",
    "TROPDStage",
]
